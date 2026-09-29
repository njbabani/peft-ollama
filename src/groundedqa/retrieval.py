"""Dense, BM25, hybrid (reciprocal rank fusion) and cross-encoder reranked retrieval."""

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")


def rrf_fuse(rankings: Sequence[Sequence[int]], k: int = 60, limit: int | None = None) -> list[int]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank(d))."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    fused = sorted(scores, key=lambda d: (-scores[d], d))
    return fused[:limit] if limit else fused


def ranking_metrics(
    rankings: Sequence[Sequence[str]], gold: Sequence[set[str]], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    """Hit rate at k (any gold chunk in the top k) and MRR over the full ranking."""
    if len(rankings) != len(gold):
        raise ValueError("One gold set per ranking is required")
    if not rankings:
        raise ValueError("No rankings to score")
    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean([bool(set(r[:k]) & g) for r, g in zip(rankings, gold)]))
    reciprocal = []
    for ranking, g in zip(rankings, gold):
        hit = next((i for i, doc in enumerate(ranking, start=1) if doc in g), None)
        reciprocal.append(1.0 / hit if hit else 0.0)
    metrics["mrr"] = float(np.mean(reciprocal))
    return metrics


class DenseIndex:
    """Exact cosine search: inner products of L2-normalised embeddings, in NumPy.

    For tens of thousands of vectors this matches a flat FAISS index in speed and
    results, and avoids loading a second OpenMP runtime beside PyTorch on macOS
    (faiss-cpu + torch in one process aborts with "OMP: Error #15").
    """

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = np.ascontiguousarray(embeddings, dtype="float32")

    def search(self, query_embeddings: np.ndarray, k: int, batch_size: int = 512) -> np.ndarray:
        k = min(k, len(self.embeddings))
        queries = np.asarray(query_embeddings, dtype="float32")
        results = []
        for start in range(0, len(queries), batch_size):
            scores = queries[start : start + batch_size] @ self.embeddings.T
            top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1, kind="stable")
            results.append(np.take_along_axis(top, order, axis=1))
        return np.vstack(results) if results else np.empty((0, k), dtype=int)


class BM25Index:
    """Okapi BM25 with English stop-word removal and Snowball stemming."""

    def __init__(self, texts: Sequence[str]):
        import bm25s
        import Stemmer

        self._bm25s = bm25s
        self.stemmer = Stemmer.Stemmer("english")
        self.size = len(texts)
        self.model = bm25s.BM25()
        self.model.index(self._tokenize(texts), show_progress=False)

    def _tokenize(self, texts: Sequence[str]):
        return self._bm25s.tokenize(list(texts), stopwords="en", stemmer=self.stemmer, show_progress=False)

    def search(self, queries: Sequence[str], k: int) -> np.ndarray:
        ids, _scores = self.model.retrieve(self._tokenize(queries), k=min(k, self.size), show_progress=False)
        return np.asarray(ids)


class CrossEncoderReranker:
    def __init__(self, model_id: str, revision: str, device: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_id, revision=revision, device=device)

    def scores(self, pairs: list[tuple[str, str]], batch_size: int = 64) -> np.ndarray:
        return np.asarray(
            self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False),
            dtype="float32",
        )


class Retriever:
    """Rank a split's corpus for questions with any of the four retrieval modes."""

    def __init__(
        self,
        corpus: list[dict],
        embedder: Any,
        embeddings: np.ndarray,
        reranker: CrossEncoderReranker | None = None,
        candidates: int = 30,
        rrf_k: int = 60,
    ):
        self.corpus = corpus
        self.embedder = embedder
        self.dense = DenseIndex(embeddings)
        self.bm25 = BM25Index([c["text"] for c in corpus])
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k

    def _encode(self, questions: Sequence[str]) -> np.ndarray:
        return self.embedder.encode(
            list(questions),
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def rank(self, questions: Sequence[str], mode: str, depth: int) -> tuple[list[list[int]], float]:
        """Corpus indices, best first, and mean seconds per question for the batch."""
        if mode not in MODES:
            raise ValueError(f"Unknown retrieval mode {mode!r}; choose from {MODES}")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("hybrid_rerank needs a reranker")
        started = time.perf_counter()
        width = max(depth, self.candidates)
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            dense = self.dense.search(self._encode(questions), width).tolist()
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            sparse = self.bm25.search(questions, width).tolist()
        if mode == "dense":
            rankings = [r[:depth] for r in dense]
        elif mode == "bm25":
            rankings = [r[:depth] for r in sparse]
        else:
            fused = [rrf_fuse([d, s], k=self.rrf_k, limit=self.candidates) for d, s in zip(dense, sparse)]
            if mode == "hybrid":
                rankings = [r[:depth] for r in fused]
            else:
                rankings = self._rerank(questions, fused, depth)
        elapsed = time.perf_counter() - started
        return rankings, elapsed / max(len(questions), 1)

    def _rerank(self, questions: Sequence[str], candidates: list[list[int]], depth: int) -> list[list[int]]:
        pairs = [
            (question, self.corpus[i]["text"]) for question, ids in zip(questions, candidates) for i in ids
        ]
        scores = self.reranker.scores(pairs)
        rankings, offset = [], 0
        for ids in candidates:
            chunk_scores = scores[offset : offset + len(ids)]
            offset += len(ids)
            order = np.argsort(-chunk_scores, kind="stable")
            rankings.append([ids[i] for i in order[:depth]])
        return rankings
