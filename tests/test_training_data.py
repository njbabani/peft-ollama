"""Training examples: label-blind layouts, safe retrieval-miss labels, exact loss masks."""

import json
import re

import pytest
import torch

from groundedqa.text import context_id
from groundedqa.training_data import (
    SupervisedExampleTooLong,
    build_training_example,
    collate,
    decision_token_ids,
    encode_example,
    evidence_shape_audit,
    evidence_shortcut_failures,
    place_anchor_evidence,
    tokenize_supervised_chat,
    validate_supervised_batch,
)


def chunk(cid, i=0, text="filler text", start=0, end=None):
    return {"id": f"{cid}-{i}", "context_id": cid, "text": text, "start": start, "end": end or len(text)}


def test_anchor_evidence_has_fixed_size_and_skips_blocked_passages():
    anchor = chunk("source")
    ranked = [chunk("source"), chunk("sibling"), chunk("a"), chunk("a"), chunk("b"), chunk("c")]
    for slot in range(3):
        evidence = place_anchor_evidence(anchor, ranked, {"source", "sibling"}, 3, slot)
        assert len(evidence) == 3 and evidence[slot] is anchor
        assert [c["id"] for c in evidence if c is not anchor] == ["a-0", "b-0"]
    assert place_anchor_evidence(anchor, ranked[:3], {"source", "sibling"}, 3, 0) is None
    with pytest.raises(ValueError):
        place_anchor_evidence(anchor, ranked, set(), 3, 3)


def _row(role, context, answers=(), starts=()):
    return {
        "id": role,
        "role": role,
        "question": "Where is it?",
        "context": context,
        "answers": {"text": list(answers), "answer_start": list(starts)},
    }


def test_build_example_for_each_role():
    context = "The tower is in Paris near the river."
    cid = context_id(context)
    source = chunk(cid, 0, context, 0, len(context))
    grouped = {cid: [source]}
    ranked = [chunk("d1", text="Paris is large."), chunk("d2"), source, chunk("d3"), chunk("d4")]

    answerable = build_training_example(_row("answerable", context, ["Paris"], [16]), ranked, grouped, 3, 1)
    assert answerable["evidence"][1] is source and len(answerable["evidence"]) == 3
    assert json.loads(answerable["target"]) == {"abstain": False, "answer": "Paris", "citations": ["E2"]}

    unanswerable = build_training_example(_row("unanswerable", context), ranked, grouped, 3, 2)
    assert unanswerable["evidence"][2] is source
    assert json.loads(unanswerable["target"])["abstain"] is True

    miss = build_training_example(_row("retrieval_miss", context, ["Paris"], [16]), ranked, grouped, 3, 0)
    ids = [c["id"] for c in miss["evidence"]]
    assert cid + "-0" not in ids and "d1-0" not in ids  # d1 mentions the answer, so it cannot be used
    assert miss["anchor_position"] is None and json.loads(miss["target"])["abstain"] is True

    crossing = _row("answerable", context, ["Paris near"], [16])
    short_source = chunk(cid, 0, context[:18], 0, 18)
    assert (
        build_training_example(crossing, ranked, {cid: [short_source]}, 3, 0)
        == "answer_crosses_chunk_boundary"
    )
    assert (
        build_training_example(_row("unanswerable", "other text"), ranked, grouped, 3, 0)
        == "missing_source_chunk"
    )


def test_shortcut_audit_rejects_the_passage_count_cue():
    # The failed Colab run: answerable had three passages, unanswerable mostly one.
    old = [{"role": "answerable", "evidence": [1, 2, 3], "anchor_position": i % 3} for i in range(30)] + [
        {"role": "unanswerable", "evidence": [1], "anchor_position": 0} for _ in range(10)
    ]
    failures = evidence_shortcut_failures(evidence_shape_audit(old), 3)
    assert any("unanswerable evidence counts" in f for f in failures)
    assert any("position" in f for f in failures)
    fixed = [
        {"role": "answerable" if i < 30 else "unanswerable", "evidence": [1, 2, 3], "anchor_position": i % 3}
        for i in range(40)
    ] + [{"role": "retrieval_miss", "evidence": [1, 2, 3], "anchor_position": None}]
    audit = evidence_shape_audit(fixed)
    assert audit["retrieval_miss"]["anchor_position"] == {}
    assert evidence_shortcut_failures(audit, 3) == []
    assert evidence_shortcut_failures(evidence_shape_audit(fixed[:30]), 3) == [
        "no unanswerable training examples"
    ]


class ToyTokenizer:
    """Regex tokens; the chat template adds role markers; 'true'/'false' are single tokens."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, token):
        return self.vocab.setdefault(token, len(self.vocab) + 10)

    def encode(self, text, add_special_tokens=False):
        return [self._id(t) for t in re.findall(r'\{"abstain":|true|false|,|[^\s,]+', text)]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        ids = []
        for message in messages:
            ids += [self._id(f"<{message['role']}>")] + self.encode(message["content"])
        if add_generation_prompt:
            ids.append(self._id("<assistant>"))
        return ids


def test_decision_token_and_weights():
    tok = ToyTokenizer()
    offset, true_id, false_id = decision_token_ids(tok)
    assert offset == 1 and true_id != false_id
    example = {
        "question": "Q?",
        "evidence": [{"id": "x", "text": "t"}],
        "target": '{"abstain": true, "answer": "", "citations": []}',
    }
    encoded = encode_example(tok, example, 500, (offset, true_id, false_id), 4.0)
    position = encoded["decision_position"]
    assert encoded["input_ids"][position] == true_id and encoded["abstain"]
    assert encoded["weights"][position] == 4.0
    assert all(w == 0 for w, label in zip(encoded["weights"], encoded["labels"]) if label == -100)
    with pytest.raises(SupervisedExampleTooLong):
        encode_example(tok, example, 5, (offset, true_id, false_id), 4.0)


def test_supervised_chat_rejects_bad_templates():
    class Bad:
        def __init__(self, full):
            self.full = full

        def apply_chat_template(self, messages, **kwargs):
            return [11, 12, 13] if kwargs["add_generation_prompt"] else self.full

    prompt, completion = [{"role": "user", "content": "Q"}], [{"role": "assistant", "content": "A"}]
    assert tokenize_supervised_chat(Bad([11, 12, 13, 21]), prompt, completion, 6)["completion_mask"] == [
        0,
        0,
        0,
        1,
    ]
    with pytest.raises(TypeError):
        tokenize_supervised_chat(Bad({"input_ids": [11, 12, 13, 21]}), prompt, completion, 6)
    with pytest.raises(ValueError, match="prefix"):
        tokenize_supervised_chat(Bad([11, 99, 13, 21]), prompt, completion, 6)
    with pytest.raises(ValueError, match="completion"):
        tokenize_supervised_chat(Bad([11, 12, 13]), prompt, completion, 6)


def test_collate_pads_without_loss_and_validator_catches_changes():
    items = [
        {
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, -100, 3, 4],
            "weights": [0, 0, 4.0, 1],
            "decision_position": 2,
            "abstain": True,
        },
        {
            "input_ids": [5, 6],
            "labels": [-100, 6],
            "weights": [0, 4.0],
            "decision_position": 1,
            "abstain": False,
        },
    ]
    batch = collate(items, pad_id=0)
    assert batch["input_ids"].tolist()[1] == [5, 6, 0, 0]
    assert batch["labels"].tolist()[1] == [-100, 6, -100, -100]
    assert batch["weights"].tolist()[1] == [0, 4.0, 0, 0]
    validate_supervised_batch(items, batch)
    broken = dict(batch, labels=batch["labels"].clone())
    broken["labels"][0, 2] = -100
    with pytest.raises(ValueError, match="boundary"):
        validate_supervised_batch(items, broken)
    leaky = dict(batch, weights=batch["weights"].clone())
    leaky["weights"][1, 3] = 1.0
    with pytest.raises(ValueError, match="Padding"):
        validate_supervised_batch(items, leaky)
    assert batch["abstain"].dtype == torch.bool
