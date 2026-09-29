"""Pipeline stages. Each stage reads and writes files inside one run directory."""

import collections
import csv
import json
import platform
import random
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from groundedqa.chunking import gold_chunk_ids
from groundedqa.config import Config
from groundedqa.data import (
    SPLITS,
    build_corpus,
    load_squad_rows,
    question_record,
    remove_cross_split_duplicates,
    select_questions,
    split_by_title,
)
from groundedqa.metrics import (
    aggregate_at_threshold,
    best_threshold,
    bootstrap_interval,
    paired_contrasts,
    question_outcomes,
    roc_auc,
)
from groundedqa.retrieval import MODES, CrossEncoderReranker, Retriever, ranking_metrics
from groundedqa.scoring import citation_stats, parse_prediction
from groundedqa.text import context_id
from groundedqa.training_data import (
    build_training_example,
    evidence_shape_audit,
    evidence_shortcut_failures,
)

PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "datasets",
    "sentence-transformers",
    "bm25s",
    "numpy",
]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `groundedqa prepare` first")
    return read_json(path)


def _embedder(config: Config, revisions: dict, device) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        config.retrieval.embedding_id, revision=revisions["embedding"], device=str(device)
    )


def _reranker(config: Config, revisions: dict, device) -> CrossEncoderReranker:
    return CrossEncoderReranker(config.retrieval.reranker_id, revisions["reranker"], str(device))


def _load_split(run_dir: Path, split: str) -> tuple[list[dict], np.ndarray, list[dict]]:
    data_dir = run_dir / "data"
    return (
        read_json(data_dir / f"corpus-{split}.json"),
        np.load(data_dir / f"embeddings-{split}.npy"),
        read_json(data_dir / f"questions-{split}.json"),
    )


def _by_context(corpus: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for chunk in corpus:
        grouped[chunk["context_id"]].append(chunk)
    return grouped


def prepare(config: Config, run_dir: Path, device, log=print) -> dict:
    """Pin inputs, split SQuAD v2 by title, sample questions, chunk and embed corpora."""
    from groundedqa.hub import resolve_all

    run_dir.mkdir(parents=True, exist_ok=True)
    revisions = resolve_all(config)
    manifest = {
        "config": config.to_dict(),
        "revisions": revisions,
        "versions": {name: metadata.version(name) for name in PACKAGES},
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "custom title-disjoint SQuAD v2; no official benchmark claim",
    }
    write_json(run_dir / "manifest.json", manifest)

    rows = load_squad_rows(config.data.dataset_id, revisions["dataset"])
    parts = remove_cross_split_duplicates(split_by_title(rows, config.data.seed))
    selected = select_questions(parts, config.data)
    embedder = _embedder(config, revisions, device)
    token_limit = embedder.max_seq_length
    summary = {}
    for split in SPLITS:
        corpus = build_corpus(
            parts[split], embedder.tokenizer, token_limit, config.retrieval.chunk_overlap_tokens
        )
        log(f"{split}: embedding {len(corpus):,} chunks from the full articles")
        embeddings = embedder.encode(
            [c["text"] for c in corpus],
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")
        data_dir = run_dir / "data"
        write_json(data_dir / f"corpus-{split}.json", corpus)
        np.save(data_dir / f"embeddings-{split}.npy", embeddings)
        write_json(data_dir / f"questions-{split}.json", [question_record(r) for r in selected[split]])
        summary[split] = {
            "titles": len({r["title"] for r in parts[split]}),
            "passages": len({context_id(r["context"]) for r in parts[split]}),
            "chunks": len(corpus),
            "questions_available": len(parts[split]),
            "questions_selected": dict(collections.Counter(r["role"] for r in selected[split])),
        }

    # Retrieval-only benchmark: answerable test questions with a locatable gold chunk.
    test_corpus = read_json(run_dir / "data" / "corpus-test.json")
    grouped = _by_context(test_corpus)
    pool = [r for r in parts["test"] if r["answers"]["text"]]
    random.Random(config.data.seed).shuffle(pool)
    bench = []
    for row in pool:
        if len(bench) >= config.data.retrieval_benchmark_questions:
            break
        if gold_chunk_ids(row, grouped.get(context_id(row["context"]), [])):
            bench.append(question_record(row))
    write_json(run_dir / "data" / "retrieval-benchmark-questions.json", bench)
    summary["retrieval_benchmark_questions"] = len(bench)
    write_json(run_dir / "data" / "summary.json", summary)
    log(json.dumps(summary, indent=2))
    return summary


def retrieval_benchmark(config: Config, run_dir: Path, device, log=print) -> dict:
    """Recall@k, MRR and per-query time of all four retrieval modes on the test corpus."""
    revisions = load_manifest(run_dir)["revisions"]
    corpus, embeddings, _ = _load_split(run_dir, "test")
    questions = read_json(run_dir / "data" / "retrieval-benchmark-questions.json")
    grouped = _by_context(corpus)
    gold = [gold_chunk_ids(q, grouped.get(context_id(q["context"]), [])) for q in questions]
    retriever = Retriever(
        corpus,
        _embedder(config, revisions, device),
        embeddings,
        _reranker(config, revisions, device),
        config.retrieval.candidates,
        config.retrieval.rrf_k,
    )
    texts = [q["question"] for q in questions]
    retriever.rank(texts[:8], "hybrid_rerank", 10)  # warm-up, excluded
    results = {}
    for mode in MODES:
        rankings, seconds = retriever.rank(texts, mode, 10)
        ids = [[corpus[i]["id"] for i in ranking] for ranking in rankings]
        results[mode] = {**ranking_metrics(ids, gold), "seconds_per_query": seconds}
        log(f"{mode:14s} " + " ".join(f"{k}={v:.3f}" for k, v in results[mode].items()))
    report = {
        "split": "test",
        "n_questions": len(questions),
        "corpus_chunks": len(corpus),
        "candidates": config.retrieval.candidates,
        "relevance": "chunk fully contains an annotated answer span",
        "modes": results,
    }
    write_json(run_dir / "retrieval-benchmark.json", report)
    return report


def build_training_data(config: Config, run_dir: Path, device, log=print) -> dict:
    """Training and monitor examples with retrieval-matched, label-blind evidence."""
    revisions = load_manifest(run_dir)["revisions"]
    embedder = _embedder(config, revisions, device)
    reranker = _reranker(config, revisions, device) if config.retrieval.mode == "hybrid_rerank" else None
    k = config.retrieval.top_k
    audit: dict[str, Any] = {"top_k": k, "retrieval_mode": config.retrieval.mode, "splits": {}}
    failures: dict[str, list[str]] = {}
    for split in ("train", "validation"):
        corpus, embeddings, questions = _load_split(run_dir, split)
        if split == "validation":
            questions = questions[: config.data.monitor_questions]
        retriever = Retriever(
            corpus, embedder, embeddings, reranker, config.retrieval.candidates, config.retrieval.rrf_k
        )
        rankings, _ = retriever.rank(
            [q["question"] for q in questions], config.retrieval.mode, config.retrieval.candidates
        )
        grouped = _by_context(corpus)
        slots = collections.Counter()
        examples, dropped = [], collections.Counter()
        for row, ranking in zip(questions, rankings):
            ranked = [corpus[i] for i in ranking]
            built = build_training_example(row, ranked, grouped, k, slots[row["role"]] % k)
            if isinstance(built, str):
                dropped[built] += 1
                continue
            slots[row["role"]] += 1
            examples.append({"id": row["id"], "role": row["role"], "question": row["question"], **built})
        write_json(run_dir / "data" / f"training-examples-{split}.json", examples)
        shape = evidence_shape_audit(examples)
        audit["splits"][split] = {
            "selected": len(questions),
            "retained": len(examples),
            "by_role": shape,
            "dropped": dict(dropped),
        }
        failures[split] = evidence_shortcut_failures(shape, k)
        log(f"{split}: {len(examples)} examples; dropped {dict(dropped)}")
    audit["shortcut_failures"] = failures
    write_json(run_dir / "training-data-audit.json", audit)
    return audit


VARIANT_FLAGS = {
    "base": (False, False),
    "base_rag": (False, True),
    "adapter": (True, False),
    "adapter_rag": (True, True),
}


def _evaluate_split(
    config, run_dir, split, decoders, retriever_parts, variants, log, resume=True
) -> list[dict]:
    corpus, embeddings, questions, embedder, reranker = retriever_parts
    retriever = Retriever(
        corpus, embedder, embeddings, reranker, config.retrieval.candidates, config.retrieval.rrf_k
    )
    k = config.retrieval.top_k
    rankings, retrieval_s = retriever.rank([q["question"] for q in questions], config.retrieval.mode, k)
    grouped = _by_context(corpus)
    records = []
    question_ids = [q["id"] for q in questions]
    for variant in variants:
        # Each finished split x variant is cached, so an interrupted evaluation resumes.
        cache = run_dir / "eval-cache" / f"{split}-{variant}.json"
        if resume and cache.exists():
            cached = read_json(cache)
            if cached["ids"] == question_ids:
                records.extend(cached["records"])
                log(f"{split} {variant}: reused {len(cached['records'])} cached predictions")
                continue
        first = len(records)
        adapted, rag = VARIANT_FLAGS[variant]
        decoder = decoders[adapted]
        items = []
        for row, ranking in zip(questions, rankings):
            evidence = [corpus[i] for i in ranking] if rag else []
            evidence, ids = decoder.prompt_ids(row["question"], evidence)
            items.append((row, evidence, ids))
        order = sorted(range(len(items)), key=lambda i: -len(items[i][2]))
        outputs: dict[int, dict] = {}
        started = time.perf_counter()
        for start in range(0, len(order), config.eval.batch_size):
            chunk = order[start : start + config.eval.batch_size]
            for index, result in zip(chunk, decoder.run([items[i][2] for i in chunk], adapted)):
                outputs[index] = result
        log(f"{split} {variant}: {len(items)} questions in {time.perf_counter() - started:.0f}s")
        for index, (row, evidence, ids) in enumerate(items):
            out = outputs[index]
            parsed = parse_prediction(out["raw"])
            gold = gold_chunk_ids(row, grouped.get(context_id(row["context"]), []))
            records.append(
                {
                    "id": row["id"],
                    "split": split,
                    "variant": variant,
                    "question": row["question"],
                    "references": row["answers"]["text"],
                    "unanswerable": not row["answers"]["text"],
                    "evidence": [{"id": c["id"], "title": c["title"], "text": c["text"]} for c in evidence],
                    "gold_chunk_ids": sorted(gold),
                    "retrieval_hit": (bool(gold & {c["id"] for c in evidence}) if rag and gold else None),
                    "prompt_tokens": len(ids),
                    "na_prob": out["na_prob"],
                    "raw": out["raw"],
                    "answer": parsed["answer"],
                    "answer_valid": bool(parsed["valid"] and not parsed["abstain"]),
                    "strict_format": bool(parsed["strict_format"]),
                    "citations": parsed["citations"],
                    **citation_stats(parsed["citations"], evidence, gold),
                    "output_tokens": out["output_tokens"],
                    "prefill_s": out["prefill_s"],
                    "generation_s": out["decode_s"],
                    "retrieval_s": retrieval_s if rag else 0.0,
                    "latency_s": out["prefill_s"] + out["decode_s"] + (retrieval_s if rag else 0.0),
                }
            )
        write_json(cache, {"ids": question_ids, "records": records[first:]})
    return records


INTERVAL_KEYS = [
    ("token_f1", None),
    ("exact_match", None),
    ("token_f1", "answerable"),
    ("answered", "answerable"),
    ("abstained", "unanswerable"),
]


def variant_intervals(outcomes: list[dict], seed: int, n_boot: int) -> dict[str, list]:
    rng = np.random.default_rng(seed)
    result = {}
    for key, subset in INTERVAL_KEYS:
        chosen = [o for o in outcomes if subset is None or o["unanswerable"] == (subset == "unanswerable")]
        mean, low, high = bootstrap_interval([float(o[key]) for o in chosen], rng, n_boot)
        result[f"{key}{'' if subset is None else '@' + subset}"] = [mean, low, high]
    return result


def evaluate(config: Config, run_dir: Path, device, log=print, resume: bool = True) -> dict:
    """Run every variant on validation and test, calibrate thresholds on validation only."""
    from groundedqa.device import dtype_from_name
    from groundedqa.inference import AnswerDecoder, load_model

    revisions = load_manifest(run_dir)["revisions"]
    variants = list(config.eval.variants)
    adapter_dir = run_dir / "adapter"
    if any(VARIANT_FLAGS[v][0] for v in variants) and not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"{adapter_dir} has no adapter; run `groundedqa train` first")
    dtype = dtype_from_name(config.model.eval_dtype)
    # Adapter variants run on a copy with LoRA merged into the weights: the same
    # function as the unmerged adapter, ~40% faster, and how it would be served.
    decoders = {}
    for adapted in sorted({VARIANT_FLAGS[v][0] for v in variants}):
        tokenizer, model = load_model(
            config.model.model_id, revisions["model"], dtype, device, adapter_dir if adapted else None
        )
        if adapted:
            model = model.merge_and_unload().eval()
        decoders[adapted] = AnswerDecoder(
            tokenizer, model, device, config.model.max_new_tokens, config.model.max_seq_length
        )
        decoders[adapted].run([decoders[adapted].prompt_ids("Warm-up?", [])[1]], adapted)  # excluded
    embedder = _embedder(config, revisions, device)
    reranker = _reranker(config, revisions, device) if config.retrieval.mode == "hybrid_rerank" else None

    records = {}
    for split in ("validation", "test"):
        corpus, embeddings, questions = _load_split(run_dir, split)
        records[split] = _evaluate_split(
            config,
            run_dir,
            split,
            decoders,
            (corpus, embeddings, questions, embedder, reranker),
            variants,
            log,
            resume,
        )
        write_json(run_dir / f"predictions-{split}.json", records[split])
    return summarise(config, run_dir, records, log)


def summarise(config: Config, run_dir: Path, records: dict[str, list[dict]] | None = None, log=print) -> dict:
    """Calibrate on validation, score test, bootstrap and write metrics files."""
    if records is None:
        records = {s: read_json(run_dir / f"predictions-{s}.json") for s in ("validation", "test")}
    variants = [v for v in VARIANT_FLAGS if any(r["variant"] == v for r in records["test"])]
    by = {
        split: {v: [r for r in rows if r["variant"] == v] for v in variants}
        for split, rows in records.items()
    }
    summary: dict[str, Any] = {"retrieval_mode": config.retrieval.mode, "variants": {}}
    outcomes = {}
    for variant in variants:
        threshold = best_threshold(by["validation"][variant])
        test_rows = by["test"][variant]
        outcomes[variant] = question_outcomes(test_rows, threshold)
        summary["variants"][variant] = {
            "threshold": threshold,
            "validation_auroc": roc_auc(
                [r["unanswerable"] for r in by["validation"][variant]],
                [r["na_prob"] for r in by["validation"][variant]],
            ),
            "test": aggregate_at_threshold(test_rows, threshold),
            "test_at_0.5": aggregate_at_threshold(test_rows, 0.5),
            "intervals": variant_intervals(
                outcomes[variant], config.data.seed, config.eval.bootstrap_samples
            ),
        }
        test = summary["variants"][variant]["test"]
        log(
            f"{variant:12s} tau={threshold:.3f} F1={test['token_f1']:.3f} "
            f"ansF1={test['answerable_f1']:.3f} abstain@unans={test['unanswerable_abstain_rate']:.3f} "
            f"AUROC={test['abstention_auroc']:.3f}"
        )
    summary["contrasts"] = paired_contrasts(outcomes, config.data.seed, config.eval.bootstrap_samples)
    write_json(run_dir / "metrics.json", summary)
    fields = ["variant", "threshold", "validation_auroc"] + [
        k for k in next(iter(summary["variants"].values()))["test"] if k != "threshold"
    ]
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, values in summary["variants"].items():
            writer.writerow(
                {
                    "variant": variant,
                    "threshold": values["threshold"],
                    "validation_auroc": values["validation_auroc"],
                    **{k: v for k, v in values["test"].items() if k != "threshold"},
                }
            )
    return summary


def ask(
    config: Config, run_dir: Path, device, question: str, split: str = "test", adapter: bool = True
) -> dict:
    """Answer one question from a split's corpus with the calibrated threshold of a finished run."""
    from groundedqa.device import dtype_from_name
    from groundedqa.inference import AnswerDecoder, load_model

    revisions = load_manifest(run_dir)["revisions"]
    variant = "adapter_rag" if adapter else "base_rag"
    metrics_path = run_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {"variants": {}}
    threshold = metrics["variants"].get(variant, {}).get("threshold", 0.5)
    tokenizer, model = load_model(
        config.model.model_id,
        revisions["model"],
        dtype_from_name(config.model.eval_dtype),
        device,
        run_dir / "adapter" if adapter else None,
    )
    if adapter:
        model = model.merge_and_unload().eval()
    decoder = AnswerDecoder(
        tokenizer, model, device, config.model.max_new_tokens, config.model.max_seq_length
    )
    corpus, embeddings, _ = _load_split(run_dir, split)
    reranker = _reranker(config, revisions, device) if config.retrieval.mode == "hybrid_rerank" else None
    retriever = Retriever(
        corpus,
        _embedder(config, revisions, device),
        embeddings,
        reranker,
        config.retrieval.candidates,
        config.retrieval.rrf_k,
    )
    rankings, retrieval_s = retriever.rank([question], config.retrieval.mode, config.retrieval.top_k)
    evidence, ids = decoder.prompt_ids(question, [corpus[i] for i in rankings[0]])
    out = decoder.run([ids], adapted=adapter)[0]
    parsed = parse_prediction(out["raw"])
    labels = {f"E{i}": chunk for i, chunk in enumerate(evidence, start=1)}
    return {
        "question": question,
        "variant": variant,
        "abstain": out["na_prob"] >= threshold,
        "no_answer_probability": out["na_prob"],
        "threshold": threshold,
        "answer": parsed["answer"] if parsed["valid"] else None,
        "citations": [label for label in parsed["citations"] if label in labels],
        "evidence": [{"label": label, "title": c["title"], "text": c["text"]} for label, c in labels.items()],
        "seconds": retrieval_s + out["prefill_s"] + out["decode_s"],
    }


def evaluate_ollama(config: Config, run_dir: Path, device, log=print, resume: bool = True) -> dict:
    """Score the 4-bit Ollama export with the same prompts, calibration and metrics."""
    from transformers import AutoTokenizer

    from groundedqa.inference import AnswerDecoder
    from groundedqa.serving import QUANTIZATION, OllamaDecoder, model_names, ollama_ready

    if not ollama_ready():
        raise RuntimeError("Start the Ollama server first: `ollama serve`")
    revisions = load_manifest(run_dir)["revisions"]
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, revision=revisions["model"])
    builder = AnswerDecoder(tokenizer, None, device, config.model.max_new_tokens, config.model.max_seq_length)
    names = model_names(run_dir)
    decoders = {adapted: OllamaDecoder(builder, names[adapted]) for adapted in (False, True)}
    for adapted, decoder in decoders.items():
        decoder.run([builder.prompt_ids("Warm-up?", [])[1]], adapted)  # loads the model; excluded
    out_dir = run_dir / f"ollama-{QUANTIZATION.lower()}"
    embedder = _embedder(config, revisions, device)
    reranker = _reranker(config, revisions, device) if config.retrieval.mode == "hybrid_rerank" else None
    records = {}
    for split in ("validation", "test"):
        corpus, embeddings, questions = _load_split(run_dir, split)
        records[split] = _evaluate_split(
            config,
            out_dir,
            split,
            decoders,
            (corpus, embeddings, questions, embedder, reranker),
            list(config.eval.variants),
            log,
            resume,
        )
        write_json(out_dir / f"predictions-{split}.json", records[split])
    return summarise(config, out_dir, records, log)
