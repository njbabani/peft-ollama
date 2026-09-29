"""SQuAD-style answer normalisation, scoring and stable text identifiers."""

import collections
import hashlib
import re
import string


def normalize_answer(text: str) -> str:
    """Lower-case, strip punctuation and articles, collapse whitespace (SQuAD rules)."""
    text = str(text).lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction: str, references: list[str]) -> tuple[float, float]:
    """Best exact-match and token-F1 against any reference; no references means no-answer."""
    pred = normalize_answer(prediction).split()
    scores = []
    for reference in references or [""]:
        gold = normalize_answer(reference).split()
        exact = float(pred == gold)
        common = sum((collections.Counter(pred) & collections.Counter(gold)).values())
        f1 = (2 * common / (len(pred) + len(gold))) if pred and gold else exact
        scores.append((exact, f1))
    return max(s[0] for s in scores), max(s[1] for s in scores)


def context_id(text: str) -> str:
    """Whitespace-insensitive 16-hex-digit identifier for a passage."""
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:16]


def contains_answer(text: str, answers: list[str]) -> bool:
    """True when any normalised answer string occurs in the normalised text."""
    haystack = normalize_answer(text)
    return any((needle := normalize_answer(answer)) and needle in haystack for answer in answers)
