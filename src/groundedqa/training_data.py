"""Supervised examples with label-blind evidence layouts and exact completion masks."""

import collections
from typing import Any

import torch

from groundedqa.chunking import gold_chunk_ids
from groundedqa.prompting import DECISION_PREFIX, abstain_target, answer_target, build_messages
from groundedqa.text import contains_answer, context_id

ANCHORED = ("answerable", "unanswerable")


class SupervisedExampleTooLong(ValueError):
    """A complete answer cannot fit without truncating supervised tokens."""


def place_anchor_evidence(
    anchor: dict, ranked: list[dict], blocked_context_ids: set, k: int, slot: int
) -> list[dict] | None:
    """Put one source chunk at ``slot`` among k - 1 ranked chunks from other passages."""
    if k < 1 or not 0 <= slot < k:
        raise ValueError("Require k >= 1 and 0 <= slot < k")
    distractors, seen = [], {anchor["id"]}
    for chunk in ranked:
        if len(distractors) == k - 1:
            break
        if chunk["id"] in seen or chunk["context_id"] in blocked_context_ids:
            continue
        seen.add(chunk["id"])
        distractors.append(chunk)
    if len(distractors) < k - 1:
        return None
    return distractors[:slot] + [anchor] + distractors[slot:]


def build_training_example(
    row: dict, ranked: list[dict], corpus_by_context: dict[str, list[dict]], k: int, slot: int
) -> dict | str:
    """Evidence and target for one row, or a drop reason.

    ``ranked`` is the retriever's ranking of the row's split corpus for its
    question, so distractors match what the model sees at inference.
    """
    source_id = context_id(row["context"])
    source_chunks = corpus_by_context.get(source_id, [])
    if not source_chunks:
        return "missing_source_chunk"
    blocked = {source_id}
    answers = row["answers"]["text"]
    role = row["role"]
    if role == "retrieval_miss":
        # Label is safe only when no supplied passage contains any gold answer.
        evidence = [
            c for c in ranked if c["context_id"] not in blocked and not contains_answer(c["text"], answers)
        ][:k]
        if len(evidence) < k:
            return "insufficient_answer_free_distractors"
        return {"evidence": evidence, "target": abstain_target(), "anchor_position": None}
    if role == "answerable":
        gold = gold_chunk_ids(row, source_chunks)
        ranked_ids = {c["id"]: i for i, c in enumerate(ranked)}
        candidates = sorted(
            (c for c in source_chunks if c["id"] in gold),
            key=lambda c: ranked_ids.get(c["id"], len(ranked)),
        )
        if not candidates:
            return "answer_crosses_chunk_boundary"  # never invent the target
        anchor = candidates[0]
        answer = next(
            a
            for start, a in zip(row["answers"]["answer_start"], answers)
            if anchor["start"] <= start and start + len(a) <= anchor["end"]
        )
        target = answer_target(answer, [f"E{slot + 1}"])
    elif role == "unanswerable":
        # The passage the question was written against: the hardest negative.
        in_ranking = [c for c in ranked if c["context_id"] == source_id]
        anchor = in_ranking[0] if in_ranking else source_chunks[0]
        target = abstain_target()
    else:
        raise ValueError(f"Unknown role {role!r}")
    evidence = place_anchor_evidence(anchor, ranked, blocked, k, slot)
    if evidence is None:
        return "insufficient_distractors"
    return {"evidence": evidence, "target": target, "anchor_position": slot}


def evidence_shape_audit(examples: list[dict]) -> dict[str, dict]:
    """Per-role passage counts and source-anchor positions."""
    audit: dict[str, dict] = {}
    for example in examples:
        stats = audit.setdefault(
            example["role"],
            {"n": 0, "evidence_count": collections.Counter(), "anchor_position": collections.Counter()},
        )
        stats["n"] += 1
        stats["evidence_count"][len(example["evidence"])] += 1
        if example["anchor_position"] is not None:
            stats["anchor_position"][example["anchor_position"]] += 1
    return {
        role: {
            "n": stats["n"],
            "evidence_count": dict(sorted(stats["evidence_count"].items())),
            "anchor_position": dict(sorted(stats["anchor_position"].items())),
        }
        for role, stats in sorted(audit.items())
    }


def evidence_shortcut_failures(audit: dict, top_k: int, max_position_gap: float = 0.10) -> list[str]:
    """Reject training sets whose evidence layout alone predicts the target."""
    missing = [role for role in ANCHORED if not audit.get(role, {}).get("n")]
    if missing:
        return [f"no {role} training examples" for role in missing]
    failures = [
        f"{role} evidence counts {stats['evidence_count']} differ from exactly {top_k}"
        for role, stats in audit.items()
        if set(stats["evidence_count"]) != {top_k}
    ]
    shares = {
        role: {slot: audit[role]["anchor_position"].get(slot, 0) / audit[role]["n"] for slot in range(top_k)}
        for role in ANCHORED
    }
    gap = max(abs(shares[ANCHORED[0]][s] - shares[ANCHORED[1]][s]) for s in range(top_k))
    # Rotation leaves at most one example of imbalance per role; allow for small n.
    tolerance = max_position_gap + 1 / min(audit[role]["n"] for role in ANCHORED)
    if gap > tolerance:
        failures.append(f"source-anchor position shares differ by {gap:.2f} (> {tolerance:.2f})")
    return failures


def tokenize_supervised_chat(
    tokenizer: Any, prompt: list[dict], completion: list[dict], max_length: int
) -> dict[str, list[int]]:
    """Use the chat template once for each exact side of the loss boundary."""
    prompt_ids = tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True, return_dict=False, truncation=False
    )
    full_ids = tokenizer.apply_chat_template(
        prompt + completion,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        truncation=False,
    )
    for ids in (prompt_ids, full_ids):
        if type(ids) is not list or not all(type(token) is int for token in ids):
            raise TypeError("Chat template must return a flat list of token IDs")
    if not prompt_ids or full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt is not an exact prefix of the full reply")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("Chat template produced no supervised completion tokens")
    if len(full_ids) > max_length:
        raise SupervisedExampleTooLong("Supervised sequence exceeds maximum length")
    return {
        "input_ids": full_ids,
        "completion_mask": [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids)),
    }


def decision_token_ids(tokenizer: Any) -> tuple[int, int, int]:
    """(offset into the reply, " true" id, " false" id) for the abstain decision token."""
    true_ids = tokenizer.encode(DECISION_PREFIX + " true", add_special_tokens=False)
    false_ids = tokenizer.encode(DECISION_PREFIX + " false", add_special_tokens=False)
    offset = next(i for i, (a, b) in enumerate(zip(true_ids, false_ids)) if a != b)
    if true_ids[:offset] != tokenizer.encode(DECISION_PREFIX, add_special_tokens=False):
        raise ValueError("Decision prefix does not tokenise independently of the decision")
    return offset, true_ids[offset], false_ids[offset]


def encode_example(
    tokenizer: Any,
    example: dict,
    max_length: int,
    decision: tuple[int, int, int],
    decision_weight: float,
) -> dict[str, Any]:
    """Token IDs, labels and per-token loss weights for one training example."""
    encoded = tokenize_supervised_chat(
        tokenizer,
        build_messages(example["question"], example["evidence"]),
        [{"role": "assistant", "content": example["target"]}],
        max_length,
    )
    ids, mask = encoded["input_ids"], encoded["completion_mask"]
    offset, true_id, false_id = decision
    position = mask.index(1) + offset
    if ids[position] not in (true_id, false_id):
        raise ValueError("Reply does not place the decision token where expected")
    weights = [float(m) for m in mask]
    weights[position] = decision_weight
    return {
        "input_ids": ids,
        "labels": [t if m else -100 for t, m in zip(ids, mask)],
        "weights": weights,
        "decision_position": position,
        "abstain": ids[position] == true_id,
    }


def collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch; padding has label -100 and weight 0."""
    width = max(len(item["input_ids"]) for item in batch)

    def pad(key, value):
        return torch.tensor([item[key] + [value] * (width - len(item[key])) for item in batch])

    return {
        "input_ids": pad("input_ids", pad_id),
        "attention_mask": torch.tensor(
            [[1] * len(item["input_ids"]) + [0] * (width - len(item["input_ids"])) for item in batch]
        ),
        "labels": pad("labels", -100),
        "weights": pad("weights", 0.0).float(),
        "decision_position": torch.tensor([item["decision_position"] for item in batch]),
        "abstain": torch.tensor([item["abstain"] for item in batch]),
    }


def validate_supervised_batch(examples: list[dict], batch: dict) -> None:
    """Every real token keeps its ID, prompt and padding carry no loss."""
    for index, example in enumerate(examples):
        ids = batch["input_ids"][index].tolist()
        mask = batch["attention_mask"][index].tolist()
        labels = batch["labels"][index].tolist()
        weights = batch["weights"][index].tolist()
        active = [i for i, m in enumerate(mask) if m]
        if [ids[i] for i in active] != example["input_ids"]:
            raise ValueError("Collator changed training tokens")
        if [labels[i] for i in active] != example["labels"]:
            raise ValueError("Collator changed completion loss boundary")
        if any(labels[i] != -100 or weights[i] != 0 for i, m in enumerate(mask) if not m):
            raise ValueError("Padding tokens must not contribute to loss")
