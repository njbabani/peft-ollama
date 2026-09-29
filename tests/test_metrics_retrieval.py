"""Calibration metrics, bootstrap contrasts and the four retrieval modes."""

import numpy as np
import pytest

from groundedqa.metrics import (
    aggregate_at_threshold,
    best_threshold,
    bootstrap_interval,
    paired_contrasts,
    question_outcomes,
    roc_auc,
    roc_curve,
)
from groundedqa.retrieval import BM25Index, Retriever, ranking_metrics, rrf_fuse


def test_roc_auc_known_values_and_ties():
    assert roc_auc([True, True, False, False], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert roc_auc([True, True, False, False], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert roc_auc([True, False], [0.5, 0.5]) == 0.5
    assert roc_auc([True, True], [0.1, 0.2]) is None
    fpr, tpr = roc_curve([True, False], [0.9, 0.1])
    assert fpr[0] == 0 and tpr[-1] == 1.0


def record(i, unanswerable, na_prob, answer="Paris", valid=True):
    return {
        "id": str(i),
        "unanswerable": unanswerable,
        "references": [] if unanswerable else ["Paris"],
        "na_prob": na_prob,
        "answer": answer,
        "answer_valid": valid,
        "strict_format": valid,
        "citation_validity": 1.0,
        "citation_hit": True,
        "retrieval_hit": None if unanswerable else True,
        "latency_s": 1.0,
        "output_tokens": 10,
        "generation_s": 0.5,
    }


def test_threshold_calibration_separates_classes():
    records = [record(i, False, 0.1 + 0.01 * i) for i in range(5)] + [
        record(10 + i, True, 0.7 + 0.01 * i) for i in range(5)
    ]
    threshold = best_threshold(records)
    assert 0.14 < threshold <= 0.7
    metrics = aggregate_at_threshold(records, threshold)
    assert metrics["token_f1"] == 1.0 and metrics["abstention_auroc"] == 1.0
    assert metrics["answerable_answer_rate"] == 1.0 and metrics["unanswerable_abstain_rate"] == 1.0
    never = aggregate_at_threshold(records, 1.01)
    assert never["token_f1"] == 0.5 and never["unanswerable_abstain_rate"] == 0.0


def test_invalid_answers_count_as_wrong_not_abstentions():
    outcomes = question_outcomes([record(0, False, 0.1, answer="", valid=False)], 0.5)
    assert outcomes[0]["token_f1"] == 0.0 and not outcomes[0]["abstained"] and not outcomes[0]["answered"]


def test_bootstrap_and_paired_contrasts():
    rng = np.random.default_rng(0)
    mean, low, high = bootstrap_interval([1, 1, 0, 0], rng, 500)
    assert mean == 0.5 and 0 <= low <= 0.5 <= high <= 1
    assert bootstrap_interval([], rng) == (None, None, None)
    base = [
        {"id": str(i), "unanswerable": i % 2 == 1, "token_f1": 0.0, "abstained": False} for i in range(10)
    ]
    better = [dict(o, token_f1=1.0, abstained=o["unanswerable"]) for o in base]
    contrasts = paired_contrasts({"base_rag": base, "adapter_rag": better}, n_boot=200)
    f1_all = next(c for c in contrasts if c["metric"].startswith("Token F1 · all"))
    assert f1_all["mean"] == 1.0 and f1_all["n"] == 10
    assert {c["contrast"] for c in contrasts} == {"Adapter effect, with RAG"}


def test_rrf_and_ranking_metrics():
    # 1 and 3 tie on 1/61 + 1/63 and beat 2 (2/62); ties break by id.
    assert rrf_fuse([[1, 2, 3], [3, 2, 1]]) == [1, 3, 2]
    assert rrf_fuse([[5, 6], [6, 7]])[0] == 6
    assert rrf_fuse([[1, 2, 3]], limit=2) == [1, 2]
    metrics = ranking_metrics([["a", "b", "c"], ["x", "y", "z"]], [{"b"}, {"q"}], ks=(1, 3))
    assert metrics == {"recall@1": 0.0, "recall@3": 0.5, "mrr": 0.25}
    with pytest.raises(ValueError):
        ranking_metrics([["a"]], [])


CORPUS = [
    {"id": "0", "text": "The Eiffel Tower is a wrought-iron tower in Paris, France."},
    {"id": "1", "text": "Mount Fuji is the highest mountain in Japan."},
    {"id": "2", "text": "The Great Wall of China stretches thousands of kilometres."},
    {"id": "3", "text": "Paris is the capital city of France on the Seine."},
]


class KeywordEmbedder:
    """Bag-of-keywords embeddings so dense search is deterministic and meaningful."""

    words = ["paris", "tower", "japan", "mountain", "china", "wall", "france", "capital"]

    def _vector(self, text):
        text = text.lower()
        vector = np.array([float(w in text) for w in self.words]) + 1e-3
        return vector / np.linalg.norm(vector)

    def encode(self, texts, **kwargs):
        return np.stack([self._vector(t) for t in texts]).astype("float32")


class CapitalReranker:
    def scores(self, pairs):
        return np.array([("capital" in p) * 2.0 + ("tower" in p) for _q, p in pairs], dtype="float32")


def test_bm25_and_every_retrieval_mode():
    assert BM25Index([c["text"] for c in CORPUS]).search(["highest mountain in Japan"], 1).tolist() == [[1]]
    embedder = KeywordEmbedder()
    embeddings = embedder.encode([c["text"] for c in CORPUS])
    retriever = Retriever(CORPUS, embedder, embeddings, CapitalReranker(), candidates=4)
    for mode in ("dense", "bm25", "hybrid"):
        rankings, seconds = retriever.rank(["capital of France Paris"], mode, 2)
        assert set(rankings[0]) == {0, 3} and seconds >= 0
    reranked, _ = retriever.rank(["Paris tower"], "hybrid_rerank", 2)
    assert reranked[0][0] == 3  # the reranker prefers the "capital" passage
    with pytest.raises(ValueError):
        retriever.rank(["q"], "sparse", 1)
    with pytest.raises(ValueError):
        Retriever(CORPUS, embedder, embeddings).rank(["q"], "hybrid_rerank", 1)
