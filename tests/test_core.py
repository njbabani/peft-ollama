"""Pure helpers: text scoring, chunking, prompts, parsing, data splits, config, Hub pinning."""

import json
from types import SimpleNamespace

import pytest

from groundedqa.chunking import chunk_passages, gold_chunk_ids
from groundedqa.config import config_from_dict, load_config
from groundedqa.data import build_corpus, remove_cross_split_duplicates, select_questions, split_by_title
from groundedqa.hub import resolve_hub_revision
from groundedqa.prompting import build_messages, evidence_label_map
from groundedqa.scoring import citation_stats, parse_prediction, score_decision
from groundedqa.text import answer_scores, contains_answer, context_id, normalize_answer


class CharacterTokenizer:
    """One token per character; two special tokens."""

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def encode(self, text, add_special_tokens=True):
        return [0] * (len(text) + (2 if add_special_tokens else 0))

    def num_special_tokens_to_add(self, pair=False):
        return 2


def test_normalization_and_scores():
    assert normalize_answer("The, QUICK fox!") == "quick fox"
    assert answer_scores("Paris", ["Paris", "City of Paris"]) == (1.0, 1.0)
    assert answer_scores("", []) == (1.0, 1.0)
    assert answer_scores("wrong", ["Paris"]) == (0.0, 0.0)
    assert answer_scores("red blue", ["red green"])[1] == pytest.approx(0.5)


def test_context_id_and_contains_answer():
    assert context_id("a  b\n") == context_id("a b")
    assert context_id("x") != context_id("y")
    assert contains_answer("The Eiffel Tower is in Paris.", ["paris"])
    assert not contains_answer("Lyon is in France.", ["Paris", ""])


def test_chunks_keep_offsets_and_respect_budget():
    tok = CharacterTokenizer()
    assert chunk_passages("AbCdEf", tok, 6, 1) == [
        {"text": "AbCd", "start": 0, "end": 4},
        {"text": "dEf", "start": 3, "end": 6},
    ]
    assert chunk_passages("", tok, 6, 1) == []
    with pytest.raises(ValueError):
        chunk_passages("x", tok, 2, 0)

    class Expanding(CharacterTokenizer):
        def encode(self, text, add_special_tokens=True):
            return [0] * (2 * len(text) + 2)

    assert all(len(c["text"]) <= 2 for c in chunk_passages("abcdef", Expanding(), 6, 1))


def test_gold_chunks_use_annotated_occurrence():
    row = {"context": "Paris then Paris", "answers": {"text": ["Paris"], "answer_start": [11]}}
    cid = context_id(row["context"])
    chunks = [
        {"id": "wrong-occurrence", "context_id": cid, "start": 0, "end": 5},
        {"id": "gold", "context_id": cid, "start": 11, "end": 16},
        {"id": "partial", "context_id": cid, "start": 12, "end": 16},
        {"id": "other-source", "context_id": "other", "start": 0, "end": 20},
    ]
    assert gold_chunk_ids(row, chunks) == {"gold"}
    assert gold_chunk_ids({"context": "x", "answers": {"text": [], "answer_start": []}}, chunks) == set()


def test_prompt_uses_short_labels_and_hides_ids():
    evidence = [
        {"id": "hash-1", "text": "Paris", "context_id": "private"},
        {"id": "hash-2", "text": "France"},
    ]
    messages = build_messages("Question?", evidence)
    payload = json.loads(messages[1]["content"])
    assert payload["evidence"] == [{"id": "E1", "text": "Paris"}, {"id": "E2", "text": "France"}]
    assert "hash-1" not in messages[1]["content"] and "private" not in messages[1]["content"]
    assert evidence_label_map(evidence) == {"E1": "hash-1", "E2": "hash-2"}
    assert "No evidence" in build_messages("Q", [])[1]["content"]
    assert '{"abstain": true, "answer": "", "citations": []}' in messages[0]["content"]


def test_parser_contract_and_fences():
    ok = parse_prediction('{"abstain": false, "answer": "Paris", "citations": ["E1"]}')
    assert ok["valid"] and ok["strict_format"]
    fenced = parse_prediction('```json\n{"abstain": true, "answer": "", "citations": []}\n```')
    assert fenced["valid"] and not fenced["strict_format"]
    for bad in [
        "bad",
        "{}",
        "[]",
        '{"answer":"x","citations":[],"abstain":"false"}',
        '{"answer":"x","citations":[3],"abstain":false}',
        '{"answer":"x","citations":[],"abstain":true}',
        '{"answer":"","citations":[],"abstain":false}',
        'Sure: {"abstain":false,"answer":"x","citations":[]}',
    ]:
        assert not parse_prediction(bad)["valid"], bad


def test_citations_and_decision_scoring():
    evidence = [{"id": "gold-chunk"}, {"id": "other"}]
    stats = citation_stats(["E1", "E9"], evidence, {"gold-chunk"})
    assert stats["resolved_citation_ids"] == ["gold-chunk", None]
    assert stats["citation_validity"] == 0.5 and stats["citation_hit"] is True
    assert citation_stats([], evidence, set())["citation_hit"] is None
    assert score_decision(True, "", True, []) == (1.0, 1.0)
    assert score_decision(True, "", True, ["Paris"]) == (0.0, 0.0)
    assert score_decision(False, "Paris", True, ["Paris"]) == (1.0, 1.0)
    assert score_decision(False, "Paris", False, ["Paris"]) == (0.0, 0.0)
    assert score_decision(False, "Paris", True, []) == (0.0, 0.0)


def test_title_split_is_disjoint_and_deterministic():
    rows = [{"title": str(i % 30), "id": str(i), "context": f"c{i}", "question": f"q{i}"} for i in range(90)]
    parts = split_by_title(rows, 4)
    assert parts == split_by_title(rows, 4)
    assert sum(map(len, parts.values())) == 90
    titles = [{r["title"] for r in p} for p in parts.values()]
    assert not (titles[0] & titles[1] or titles[0] & titles[2] or titles[1] & titles[2])
    with pytest.raises(ValueError):
        split_by_title(rows[:2], 4)


def test_duplicates_removed_with_train_priority():
    parts = {
        "train": [{"context": "shared", "question": "Q one?"}],
        "validation": [{"context": "shared", "question": "other"}, {"context": "v", "question": "q one"}],
        "test": [{"context": "t", "question": "fresh"}],
    }
    cleaned = remove_cross_split_duplicates(parts)
    assert cleaned["validation"] == [] and len(cleaned["test"]) == 1


def _rows(prefix, n_yes, n_no):
    make = lambda i, text: {  # noqa: E731
        "id": f"{prefix}{i}",
        "title": prefix,
        "context": f"{prefix} context {i}",
        "question": f"{prefix} q{i}",
        "answers": {"text": text, "answer_start": [0] if text else []},
    }
    return [make(i, ["a"]) for i in range(n_yes)] + [make(n_yes + i, []) for i in range(n_no)]


def test_question_selection_roles_and_counts():
    cfg = SimpleNamespace(
        seed=1,
        train_answerable=5,
        train_unanswerable=4,
        train_retrieval_miss=2,
        validation_questions=6,
        test_questions=4,
    )
    parts = {"train": _rows("tr", 10, 10), "validation": _rows("va", 5, 5), "test": _rows("te", 5, 5)}
    selected = select_questions(parts, cfg)
    roles = [r["role"] for r in selected["train"]]
    assert (
        roles.count("answerable") == 5
        and roles.count("retrieval_miss") == 2
        and roles.count("unanswerable") == 4
    )
    assert sum(bool(r["answers"]["text"]) for r in selected["validation"]) == 3
    assert len(selected["test"]) == 4
    with pytest.raises(ValueError, match="unanswerable train"):
        select_questions({**parts, "train": _rows("tr", 10, 1)}, cfg)


def test_corpus_covers_every_passage_once():
    rows = _rows("x", 3, 0) + _rows("x", 1, 0)  # duplicate passage "x context 0"
    corpus = build_corpus(rows, CharacterTokenizer(), 400, 8)
    assert len({c["context_id"] for c in corpus}) == 3
    assert all(c["title"] == "x" and c["id"].startswith(c["context_id"]) for c in corpus)


def test_configs_load_and_reject_typos():
    full, smoke = load_config("configs/full.toml"), load_config("configs/smoke.toml")
    assert full.retrieval.mode == "hybrid_rerank" and smoke.train.max_steps > 0
    with pytest.raises(ValueError, match="Unknown keys"):
        config_from_dict({"train": {"learning_rat": 1e-4}})
    with pytest.raises(ValueError, match="sections"):
        config_from_dict({"trian": {}})


class HubError(Exception):
    def __init__(self, status):
        super().__init__(status)
        self.response = SimpleNamespace(status_code=status)


def test_hub_retry_recovers_transient_errors():
    calls, waits = [], []

    def lookup(repo, **kwargs):
        calls.append(kwargs)
        if len(calls) < 3:
            raise HubError(504)
        return SimpleNamespace(sha="a" * 40)

    assert resolve_hub_revision(lookup, "repo", "main", (HubError,), sleep=waits.append) == "a" * 40
    assert waits == [2, 4] and calls[0] == {"revision": "main", "timeout": 30}


def test_hub_retry_fails_bounded_and_keeps_permanent_errors():
    with pytest.raises(RuntimeError) as caught:
        resolve_hub_revision(
            lambda *a, **k: (_ for _ in ()).throw(TimeoutError("t")),
            "repo",
            "main",
            (TimeoutError,),
            sleep=lambda _: None,
            attempts=3,
        )
    assert isinstance(caught.value.__cause__, TimeoutError)
    with pytest.raises(HubError):
        resolve_hub_revision(
            lambda *a, **k: (_ for _ in ()).throw(HubError(401)),
            "repo",
            "main",
            (HubError,),
            sleep=lambda _: None,
        )
    assert resolve_hub_revision(lambda *a, **k: pytest.fail("no lookup"), "r", "B" * 40, ()) == "b" * 40
    with pytest.raises(ValueError):
        resolve_hub_revision(lambda *a, **k: SimpleNamespace(sha=None), "repo", "main", (HubError,))
