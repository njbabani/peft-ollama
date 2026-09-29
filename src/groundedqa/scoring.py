"""Parse model replies and score one evaluated question."""

import json
import re
from typing import Any

from groundedqa.prompting import evidence_label_map
from groundedqa.text import answer_scores

FENCED_JSON = re.compile(r"\A```(?:json)?[ \t]*\n(?P<body>.*?)\n?```\Z", re.DOTALL | re.IGNORECASE)


def parse_prediction(raw: str) -> dict[str, Any]:
    """Validate the reply contract; one wrapping Markdown fence is valid but not strict."""
    invalid = {
        "answer": "",
        "citations": [],
        "abstain": False,
        "valid": False,
        "strict_format": False,
    }
    text, strict = raw, True
    if isinstance(raw, str) and (fenced := FENCED_JSON.match(raw.strip())):
        text, strict = fenced["body"], False
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return invalid
    if not isinstance(value, dict) or set(value) != {"answer", "citations", "abstain"}:
        return invalid
    if not isinstance(value["answer"], str) or type(value["abstain"]) is not bool:
        return invalid
    if not isinstance(value["citations"], list) or not all(isinstance(x, str) for x in value["citations"]):
        return invalid
    if value["abstain"] and (value["answer"].strip() or value["citations"]):
        return invalid
    if not value["abstain"] and not value["answer"].strip():
        return invalid
    return {**value, "valid": True, "strict_format": strict}


def citation_stats(citations: list[str], evidence: list[dict], gold_ids: set[str]) -> dict[str, Any]:
    """Resolve E labels to chunk IDs; a hit means a cited chunk holds the gold span."""
    label_map = evidence_label_map(evidence)
    resolved = [label_map.get(label) for label in citations]
    return {
        "cited": bool(citations),
        "citation_validity": (sum(r is not None for r in resolved) / len(citations) if citations else None),
        "citation_hit": (any(r in gold_ids for r in resolved if r is not None) if gold_ids else None),
        "resolved_citation_ids": resolved,
    }


def score_decision(
    abstain: bool, answer: str, answer_valid: bool, references: list[str]
) -> tuple[float, float]:
    """SQuAD v2 scoring: abstaining is correct only for no-answer questions."""
    if abstain:
        return (0.0, 0.0) if references else (1.0, 1.0)
    if not references or not answer_valid:
        return 0.0, 0.0
    return answer_scores(answer, references)
