"""Summaries, figures and the Markdown report from synthetic run files."""

import json
import random

from groundedqa.config import load_config
from groundedqa.figures import render_all
from groundedqa.pipeline import summarise
from groundedqa.report import build_summary

PROFILE = {  # (answer rate on answerable, P(correct | answered), abstain rate on unanswerable)
    "base": (0.6, 0.3, 0.5),
    "base_rag": (0.8, 0.6, 0.3),
    "adapter": (0.3, 0.3, 0.8),
    "adapter_rag": (0.9, 0.8, 0.8),
}


def synthetic_records(split, n, seed):
    rng = random.Random(seed)
    rows = []
    for variant, (answer_rate, correct, abstain_rate) in PROFILE.items():
        for i in range(n):
            unanswerable = i % 2 == 1
            abstains = rng.random() < (abstain_rate if unanswerable else 1 - answer_rate)
            na_prob = min(max(rng.gauss(0.7 if abstains else 0.3, 0.15), 0.0), 1.0)
            good = rng.random() < correct
            rows.append(
                {
                    "id": f"{split}-{i}",
                    "split": split,
                    "variant": variant,
                    "unanswerable": unanswerable,
                    "references": [] if unanswerable else ["Paris"],
                    "na_prob": na_prob,
                    "answer": "Paris" if good else "Lyon",
                    "answer_valid": True,
                    "strict_format": True,
                    "citation_validity": 1.0,
                    "citation_hit": good,
                    "retrieval_hit": (not unanswerable) if variant.endswith("rag") else None,
                    "latency_s": rng.uniform(0.5, 2.0),
                    "output_tokens": 12,
                    "generation_s": 0.4,
                }
            )
    return rows


def _write(path, value):
    path.write_text(json.dumps(value))


def test_summary_figures_and_report(tmp_path):
    config = load_config("configs/smoke.toml")
    records = {"validation": synthetic_records("validation", 60, 1), "test": synthetic_records("test", 60, 2)}
    for split, rows in records.items():
        _write(tmp_path / f"predictions-{split}.json", rows)
    summary = summarise(config, tmp_path, log=lambda _m: None)
    assert set(summary["variants"]) == set(PROFILE)
    for values in summary["variants"].values():
        assert 0.0 <= values["threshold"] <= 1.0 + 1e-9
        assert 0.0 <= values["test"]["token_f1"] <= 1.0
        assert len(values["intervals"]["token_f1"]) == 3
    assert len(summary["contrasts"]) == 12
    header = (tmp_path / "metrics.csv").read_text().splitlines()[0].split(",")
    assert header.count("threshold") == 1

    _write(
        tmp_path / "training-log.json",
        [{"step": 0, "eval_loss": 1.0, "eval_decision_auroc": 0.55}]
        + [{"step": s, "loss": 1 / s, "learning_rate": 1e-4, "grad_norm": 1.0} for s in range(1, 20)]
        + [{"step": 20, "loss": 0.05, "eval_loss": 0.3, "eval_decision_auroc": 0.8}],
    )
    _write(
        tmp_path / "training.json",
        {
            "device": "mps",
            "wall_seconds": 600,
            "optimizer_steps": 20,
            "examples_seen": 160,
            "trainable_parameters": 18_000_000,
            "trainable_fraction": 0.012,
            "peak_memory_gib": 8.5,
            "initial_monitor": {"eval_loss": 1.0, "eval_decision_auroc": 0.55},
            "final_monitor": {"eval_loss": 0.3, "eval_decision_auroc": 0.8},
        },
    )
    balanced = {"0": 10, "1": 10, "2": 10}
    by_role = {
        "answerable": {"n": 30, "evidence_count": {"3": 30}, "anchor_position": balanced},
        "unanswerable": {"n": 30, "evidence_count": {"3": 30}, "anchor_position": balanced},
        "retrieval_miss": {"n": 6, "evidence_count": {"3": 6}, "anchor_position": {}},
    }
    _write(
        tmp_path / "training-data-audit.json",
        {
            "top_k": 3,
            "retrieval_mode": "hybrid_rerank",
            "splits": {"train": {"selected": 70, "retained": 66, "by_role": by_role, "dropped": {}}},
            "shortcut_failures": {"train": [], "validation": []},
        },
    )
    modes = {
        mode: {
            "recall@1": r,
            "recall@3": r + 0.1,
            "recall@5": r + 0.15,
            "recall@10": r + 0.2,
            "mrr": r + 0.05,
            "seconds_per_query": 0.002,
        }
        for mode, r in [("dense", 0.5), ("bm25", 0.45), ("hybrid", 0.55), ("hybrid_rerank", 0.65)]
    }
    _write(
        tmp_path / "retrieval-benchmark.json",
        {"n_questions": 200, "corpus_chunks": 2500, "candidates": 30, "modes": modes},
    )
    written = render_all(tmp_path, "smoke")
    assert {p.name for p in written} == {
        "training-data-audit.png",
        "retrieval-benchmark.png",
        "training-dashboard.png",
        "calibration.png",
        "results-overview.png",
        "paired-contrasts.png",
    }
    assert all(p.stat().st_size > 10_000 for p in written)
    markdown = build_summary(tmp_path)
    assert "## Retrieval benchmark" in markdown and "## Question answering" in markdown
    assert (
        "Hybrid + cross-encoder rerank" in markdown and "![calibration](figures/calibration.png)" in markdown
    )
