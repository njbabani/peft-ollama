"""Markdown summary of a run, built only from its saved JSON files."""

import json
from pathlib import Path

VARIANT_LABELS = {
    "base": "Base",
    "base_rag": "Base + RAG",
    "adapter": "Adapter",
    "adapter_rag": "Adapter + RAG",
}
MODE_LABELS = {
    "dense": "Dense (MiniLM, exact cosine)",
    "bm25": "BM25",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + cross-encoder rerank",
}


def _f(value, spec=".3f"):
    return "n/a" if value is None else format(value, spec)


def _read(run_dir: Path, name: str):
    path = run_dir / name
    return json.loads(path.read_text()) if path.exists() else None


def build_summary(run_dir: Path) -> str:
    manifest = _read(run_dir, "manifest.json") or {}
    config = manifest.get("config", {})
    lines = [f"# Run summary: {run_dir.name}", ""]
    lines += [
        f"- Model: `{config.get('model', {}).get('model_id')}` @ `{manifest.get('revisions', {}).get('model', '')[:12]}`",
        f"- Device: `{manifest.get('device')}` on `{manifest.get('platform')}`",
        f"- Retrieval for QA: `{config.get('retrieval', {}).get('mode')}`, top-{config.get('retrieval', {}).get('top_k')}",
        f"- Protocol: {manifest.get('protocol', '')}",
        "",
    ]
    if data := _read(run_dir, "data/summary.json"):
        lines += [
            "## Data",
            "",
            "| Split | Titles | Passages | Chunks | Questions used |",
            "|---|---|---|---|---|",
        ]
        for split in ("train", "validation", "test"):
            s = data[split]
            used = ", ".join(f"{k} {v}" for k, v in s["questions_selected"].items())
            lines.append(f"| {split} | {s['titles']} | {s['passages']:,} | {s['chunks']:,} | {used} |")
        lines.append("")
    if bench := _read(run_dir, "retrieval-benchmark.json"):
        lines += [
            "## Retrieval benchmark",
            "",
            f"{bench['n_questions']:,} answerable test questions over {bench['corpus_chunks']:,} chunks "
            "(every paragraph of the held-out articles).",
            "",
            "| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | ms / query |",
            "|---|---|---|---|---|---|---|",
        ]
        for mode, m in bench["modes"].items():
            lines.append(
                f"| {MODE_LABELS.get(mode, mode)} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | "
                f"{m['recall@5']:.3f} | {m['recall@10']:.3f} | {m['mrr']:.3f} | {m['seconds_per_query'] * 1000:.2g} |"
            )
        lines.append("")
    if training := _read(run_dir, "training.json"):
        initial, final = training["initial_monitor"], training["final_monitor"]
        lines += [
            "## Training",
            "",
            f"- {training['optimizer_steps']} optimizer steps over {training['examples_seen']:,} examples "
            f"in {training['wall_seconds'] / 60:.1f} min on `{training['device']}`",
            f"- LoRA parameters: {training['trainable_parameters']:,} ({training['trainable_fraction']:.2%})",
            f"- Held-out decision AUROC: {_f(initial['eval_decision_auroc'])} → {_f(final['eval_decision_auroc'])}; "
            f"held-out loss {_f(initial['eval_loss'])} → {_f(final['eval_loss'])}",
            f"- Peak device memory: {_f(training.get('peak_memory_gib'), '.1f')} GiB",
            "",
        ]
    if metrics := _read(run_dir, "metrics.json"):
        lines += [
            "## Question answering (test, thresholds calibrated on validation)",
            "",
            "| Variant | Threshold | Token F1 | EM | Answerable F1 | Answer rate | Abstain (unans.) | AUROC | F1 at 0.5 | Citation hit |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for variant, values in metrics["variants"].items():
            t, u = values["test"], values["test_at_0.5"]
            lines.append(
                f"| {VARIANT_LABELS.get(variant, variant)} | {values['threshold']:.3f} | {_f(t['token_f1'])} | "
                f"{_f(t['exact_match'])} | {_f(t['answerable_f1'])} | {_f(t['answerable_answer_rate'])} | "
                f"{_f(t['unanswerable_abstain_rate'])} | {_f(t['abstention_auroc'])} | {_f(u['token_f1'])} | "
                f"{_f(t['citation_hit_rate'])} |"
            )
        lines += [
            "",
            "### Paired contrasts (95% bootstrap intervals)",
            "",
            "| Metric | Contrast | Δ | 95% CI |",
            "|---|---|---|---|",
        ]
        for c in metrics["contrasts"]:
            if c["mean"] is None:
                continue
            lines.append(
                f"| {c['metric']} | {c['contrast']} | {c['mean']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] |"
            )
        lines.append("")
    quantised = _read(run_dir, "ollama-q4_k_m/metrics.json")
    if metrics and quantised:
        lines += [
            "## fp16 (PyTorch) vs 4-bit (Ollama Q4_K_M)",
            "",
            "Same prompts, retrieval and validation-calibrated thresholds; test split.",
            "",
            "| Variant | Backend | Token F1 | Answerable F1 | Abstain (unans.) | AUROC | p50 s / question |",
            "|---|---|---|---|---|---|---|",
        ]
        for variant in ("base_rag", "adapter_rag"):
            for backend, source in (("PyTorch fp16", metrics), ("Ollama Q4_K_M", quantised)):
                t = source["variants"].get(variant, {}).get("test")
                if t:
                    lines.append(
                        f"| {VARIANT_LABELS[variant]} | {backend} | {_f(t['token_f1'])} | {_f(t['answerable_f1'])} | "
                        f"{_f(t['unanswerable_abstain_rate'])} | {_f(t['abstention_auroc'])} | {_f(t['latency_p50_s'], '.2f')} |"
                    )
        lines.append("")
    figures = sorted((run_dir / "figures").glob("*.png")) if (run_dir / "figures").exists() else []
    if figures:
        lines += ["## Figures", ""] + [f"![{p.stem}](figures/{p.name})" for p in figures] + [""]
    return "\n".join(lines)
