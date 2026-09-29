"""Token-bounded passage chunks that keep their character offsets in the source."""

from typing import Any

from groundedqa.text import context_id


def chunk_passages(text: str, tokenizer: Any, token_limit: int, overlap_tokens: int = 32) -> list[dict]:
    """Split text so each chunk, with special tokens, fits the embedding model's limit.

    Chunks are slices of the original text, so answer offsets can be checked
    against ``start``/``end`` later.
    """
    budget = token_limit - tokenizer.num_special_tokens_to_add(pair=False)
    if budget <= 0 or not 0 <= overlap_tokens < budget:
        raise ValueError("Token budget must exceed overlap and special tokens")
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)[
        "offset_mapping"
    ]
    chunks = []
    start = 0
    while start < len(offsets):
        end = min(start + budget, len(offsets))
        while end > start:
            char_start, char_end = offsets[start][0], offsets[end - 1][1]
            piece = text[char_start:char_end]
            if len(tokenizer.encode(piece, add_special_tokens=True)) <= token_limit:
                break
            end -= 1
        if end == start:
            raise ValueError("A source token cannot fit the embedding budget")
        chunks.append({"text": piece, "start": char_start, "end": char_end})
        if end == len(offsets):
            break
        start = max(start + 1, end - overlap_tokens)
    return chunks


def gold_chunk_ids(row: dict, corpus: list[dict]) -> set[str]:
    """IDs of chunks that fully contain an annotated answer span from the row's source."""
    cid = context_id(row["context"])
    spans = [
        (start, start + len(answer))
        for start, answer in zip(row["answers"]["answer_start"], row["answers"]["text"])
        if answer
    ]
    return {
        c["id"]
        for c in corpus
        if c["context_id"] == cid and any(c["start"] <= start and end <= c["end"] for start, end in spans)
    }
