"""SQuAD v2 loading, title-disjoint splits, question sampling and full-article corpora."""

import random
from typing import Any

from groundedqa.chunking import chunk_passages
from groundedqa.text import context_id, normalize_answer

SPLITS = ("train", "validation", "test")


def load_squad_rows(dataset_id: str, revision: str) -> list[dict]:
    """All rows from the dataset's train and validation partitions."""
    from datasets import load_dataset

    raw = load_dataset(dataset_id, revision=revision)
    return [dict(row) for split in raw.values() for row in split]


def split_by_title(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    """Deterministic 80/10/10 split by article title, so no article crosses splits."""
    titles = sorted({r["title"] for r in rows})
    if len(titles) < 3:
        raise ValueError("At least three distinct titles are required")
    random.Random(seed).shuffle(titles)
    n_holdout = max(1, len(titles) // 10)
    groups = {
        "test": set(titles[:n_holdout]),
        "validation": set(titles[n_holdout : 2 * n_holdout]),
        "train": set(titles[2 * n_holdout :]),
    }
    return {name: [r for r in rows if r["title"] in group] for name, group in groups.items()}


def remove_cross_split_duplicates(parts: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Drop rows whose passage or normalised question already appears in an earlier split.

    Priority is train, then validation, then test.
    """
    seen_contexts: set[str] = set()
    seen_questions: set[str] = set()
    cleaned = {}
    for name in SPLITS:
        keep = [
            r
            for r in parts[name]
            if context_id(r["context"]) not in seen_contexts
            and normalize_answer(r["question"]) not in seen_questions
        ]
        cleaned[name] = keep
        seen_contexts.update(context_id(r["context"]) for r in keep)
        seen_questions.update(normalize_answer(r["question"]) for r in keep)
    return cleaned


def _take(pool: list[dict], count: int, what: str) -> list[dict]:
    if len(pool) < count:
        raise ValueError(f"Need {count} {what} questions, only {len(pool)} available")
    return pool[:count]


def select_questions(parts: dict[str, list[dict]], data_cfg: Any) -> dict[str, list[dict]]:
    """Sample each split's questions and tag them with a training role.

    Roles: ``answerable``, ``unanswerable`` (SQuAD v2 no-answer), and
    ``retrieval_miss`` (answerable question whose source will be withheld).
    """
    rng = random.Random(data_cfg.seed)
    selected = {}
    for name in SPLITS:
        rows = list(parts[name])
        rng.shuffle(rows)
        yes = [r for r in rows if r["answers"]["text"]]
        no = [r for r in rows if not r["answers"]["text"]]
        if name == "train":
            n_yes = data_cfg.train_answerable + data_cfg.train_retrieval_miss
            answerable = _take(yes, n_yes, "answerable train")
            chosen = (
                [dict(r, role="retrieval_miss") for r in answerable[: data_cfg.train_retrieval_miss]]
                + [dict(r, role="answerable") for r in answerable[data_cfg.train_retrieval_miss :]]
                + [
                    dict(r, role="unanswerable")
                    for r in _take(no, data_cfg.train_unanswerable, "unanswerable train")
                ]
            )
        else:
            total = data_cfg.validation_questions if name == "validation" else data_cfg.test_questions
            half = total // 2
            chosen = [dict(r, role="answerable") for r in _take(yes, half, f"answerable {name}")] + [
                dict(r, role="unanswerable") for r in _take(no, total - half, f"unanswerable {name}")
            ]
        rng.shuffle(chosen)
        selected[name] = chosen
    return selected


def build_corpus(rows: list[dict], tokenizer: Any, token_limit: int, overlap_tokens: int) -> list[dict]:
    """Chunk every distinct passage in ``rows`` (the whole split, not just sampled questions)."""
    passages: dict[str, tuple[str, str]] = {}
    for r in rows:
        passages.setdefault(context_id(r["context"]), (r["title"], r["context"]))
    corpus = []
    for cid, (title, text) in passages.items():
        for i, chunk in enumerate(chunk_passages(text, tokenizer, token_limit, overlap_tokens)):
            corpus.append({"id": f"{cid}-{i}", "context_id": cid, "title": title, **chunk})
    return corpus


def question_record(row: dict) -> dict:
    """The fields of a SQuAD row that later stages need (drops nothing they use)."""
    return {
        "id": row["id"],
        "title": row["title"],
        "context": row["context"],
        "question": row["question"],
        "answers": {
            "text": list(row["answers"]["text"]),
            "answer_start": list(row["answers"]["answer_start"]),
        },
        "role": row.get("role", "answerable" if row["answers"]["text"] else "unanswerable"),
    }
