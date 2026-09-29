"""Threshold calibration, abstention AUROC, aggregate metrics and bootstrap intervals.

Evaluation stores, for every question, the model's no-answer probability
(``na_prob``) and the answer it gives when forced to answer. Any abstention
threshold can then be applied afterwards, as in the official SQuAD 2.0 script.
"""

import math
import statistics
from typing import Any

import numpy as np

from groundedqa.scoring import score_decision


def roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    """Probability a random positive outranks a random negative (ties count half)."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if not n_pos or not n_neg:
        return None
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(labels: list[bool], scores: list[float]) -> tuple[list[float], list[float]]:
    """False- and true-positive rates at every distinct threshold (positives = label)."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    n_pos, n_neg = max(labels.sum(), 1), max((~labels).sum(), 1)
    fpr = [float(((scores >= t) & ~labels).sum() / n_neg) for t in thresholds]
    tpr = [float(((scores >= t) & labels).sum() / n_pos) for t in thresholds]
    return fpr, tpr


def _decision_scores(record: dict) -> tuple[float, float, float, float]:
    """(EM, F1) if the model abstains and (EM, F1) if it answers."""
    em_abstain, f1_abstain = score_decision(True, "", True, record["references"])
    em_answer, f1_answer = score_decision(
        False, record["answer"], record["answer_valid"], record["references"]
    )
    return em_abstain, f1_abstain, em_answer, f1_answer


def threshold_curve(records: list[dict]) -> tuple[list[float], list[float]]:
    """Mean F1 when abstaining at na_prob >= t, for every candidate threshold t."""
    na = np.array([r["na_prob"] for r in records], dtype=float)
    scores = np.array([_decision_scores(r) for r in records], dtype=float)
    candidates = np.unique(np.r_[0.0, na, np.nextafter(1.0, 2.0)])
    f1 = [float(np.where(na >= t, scores[:, 1], scores[:, 3]).mean()) for t in candidates]
    return candidates.tolist(), f1


def best_threshold(records: list[dict]) -> float:
    """Threshold maximising mean F1; ties go to the one nearest 0.5."""
    candidates, f1 = threshold_curve(records)
    best = max(f1)
    tied = [t for t, v in zip(candidates, f1) if math.isclose(v, best, abs_tol=1e-12)]
    return float(min(tied, key=lambda t: abs(t - 0.5)))


def question_outcomes(records: list[dict], threshold: float) -> list[dict]:
    """Per-question decision and scores at one abstention threshold."""
    outcomes = []
    for record in records:
        abstain = record["na_prob"] >= threshold
        em, f1 = score_decision(abstain, record["answer"], record["answer_valid"], record["references"])
        outcomes.append(
            {
                "id": record["id"],
                "unanswerable": record["unanswerable"],
                "abstained": abstain,
                "answered": (not abstain) and record["answer_valid"],
                "exact_match": em,
                "token_f1": f1,
            }
        )
    return outcomes


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def aggregate_at_threshold(records: list[dict], threshold: float) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty evaluation")
    outcomes = question_outcomes(records, threshold)
    answerable = [o for o, r in zip(outcomes, records) if not r["unanswerable"]]
    unanswerable = [o for o, r in zip(outcomes, records) if r["unanswerable"]]
    answered_answerable = [r for o, r in zip(outcomes, records) if o["answered"] and not r["unanswerable"]]
    latency = sorted(r["latency_s"] for r in records)
    return {
        "n": len(records),
        "threshold": threshold,
        "exact_match": _mean([o["exact_match"] for o in outcomes]),
        "token_f1": _mean([o["token_f1"] for o in outcomes]),
        "answerable_f1": _mean([o["token_f1"] for o in answerable]),
        "answerable_answer_rate": _mean([float(o["answered"]) for o in answerable]),
        "unanswerable_abstain_rate": _mean([float(o["abstained"]) for o in unanswerable]),
        "abstention_auroc": roc_auc([r["unanswerable"] for r in records], [r["na_prob"] for r in records]),
        "format_valid_rate": _mean([float(r["answer_valid"]) for r in records]),
        "strict_format_rate": _mean([float(r["strict_format"]) for r in records]),
        "citation_validity": _mean([r["citation_validity"] for r in answered_answerable]),
        "citation_hit_rate": _mean(
            [float(r["citation_hit"]) for r in answered_answerable if r["citation_hit"] is not None]
        ),
        "gold_chunk_recall": _mean(
            [float(r["retrieval_hit"]) for r in records if r.get("retrieval_hit") is not None]
        ),
        "latency_p50_s": statistics.median(latency),
        "latency_p95_s": latency[math.ceil(0.95 * len(latency)) - 1],
        "tokens_per_second": sum(r["output_tokens"] for r in records)
        / max(sum(r["generation_s"] for r in records), 1e-9),
    }


def bootstrap_interval(
    values: list[float], rng: np.random.Generator, n_boot: int = 2000
) -> tuple[float | None, float | None, float | None]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None, None, None
    means = values[rng.integers(0, values.size, (n_boot, values.size))].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


CONTRASTS = [
    ("adapter_rag", "base_rag", "Adapter effect, with RAG"),
    ("adapter", "base", "Adapter effect, closed-book"),
    ("base_rag", "base", "RAG effect, base model"),
    ("adapter_rag", "adapter", "RAG effect, adapter"),
]
CONTRAST_METRICS = [
    ("token_f1", "Token F1 · all questions", None),
    ("token_f1", "Token F1 · answerable", "answerable"),
    ("abstained", "Abstention · unanswerable", "unanswerable"),
]


def paired_contrasts(outcomes: dict[str, list[dict]], seed: int = 42, n_boot: int = 2000) -> list[dict]:
    """Paired per-question differences between variants, each at its own threshold."""
    rng = np.random.default_rng(seed)
    by_variant = {v: {o["id"]: o for o in rows} for v, rows in outcomes.items()}
    results = []
    for key, metric_label, subset in CONTRAST_METRICS:
        for treated, control, label in CONTRASTS:
            if treated not in by_variant or control not in by_variant:
                continue
            ids = [
                i
                for i, o in by_variant[control].items()
                if i in by_variant[treated]
                and (subset is None or o["unanswerable"] == (subset == "unanswerable"))
            ]
            deltas = [float(by_variant[treated][i][key]) - float(by_variant[control][i][key]) for i in ids]
            mean, low, high = bootstrap_interval(deltas, rng, n_boot)
            results.append(
                {
                    "metric": metric_label,
                    "contrast": label,
                    "treated": treated,
                    "control": control,
                    "n": len(deltas),
                    "mean": mean,
                    "ci95": [low, high],
                }
            )
    return results
