# Abstention-collapse experiment

The corrected-mask run still mostly abstained. The proposed explanation involving
JSON field order, repetitive negative targets, and difficult citation copying is
plausible, not established causally. Conditional token probabilities cannot be
inferred from the global frequency of answer strings alone.

This experiment changes several factors together: abstain-first target order,
short per-prompt evidence labels, 75/25 training sampling, and positive examples
with a gold chunk and retrieved distractors. Keep the model, rank, seed, and
training budget unchanged. Recompute all four baselines with the new prompt;
comparisons with old prompts cannot isolate any single factor's contribution.

Validation/test question sampling remains balanced and unchanged. Training
positives contain oracle-inserted evidence, so this reduces but does not eliminate
the training/inference mismatch. Negative labels are valid only for their source
passage; source-only negative training avoids asserting that unrelated retrieved
chunks lack an answer, but leaves an evidence-count/distribution difference.
Validation negatives with retrieved evidence likewise need manual review.

After saving the adapter, the Generation sanity check examines up to 20 retained
answerable training examples using their exact training evidence, and 20+20
validation cases using retrieval. It saves generations and diagnostic metrics,
then stops collapsed runs before the full held-out benchmark and GGUF export.
The numeric thresholds are engineering stop conditions, not statistically
validated quality requirements. A pass does not establish generalization.

Preserve prior ZIPs. Start a fresh run from Configuration and provenance, inspect
training-data-audit.json and mask-audit.json, then generation-sanity.json before
using new results in CV claims. No new GPU result has been established by local
unit tests.

## Follow-up: failed negative probe

The `20260928-001727-full` generation probe scored 0.8071 token F1 on 20
answerable training questions and 0.6161 on 20 answerable validation questions,
but abstained on none of 20 source-unanswerable validation questions. Several
answers visibly substituted a related fact (boiling for freezing, a different
festival, or reversed negation). This supports an over-answering diagnosis,
without proving a particular training cause.

Before retraining, use **Abstention diagnostic — no retraining** with the existing
adapter. Compare exact retained training-negative evidence and paired
validation-negative questions under source-only versus retrieved evidence.
If exact training negatives fail, learning abstention itself remains a problem.
If source-only validation succeeds but retrieval fails, evidence-distribution
sensitivity is implicated; the comparison cannot isolate chunk count from
content, and retrieved no-answer labels remain passage-relative. Invalid JSON
is a separate outcome, never counted as a valid answer or abstention. The
original quality gate remains in force regardless of diagnostic results.

## Diagnosis: evidence-count shortcut (fixed, awaiting a GPU run)

The diagnostic for `20260928-001727-full` isolated the cause. Each group had
20 examples:

| Group | Passages supplied | Abstained | Answered |
|---|---|---|---|
| Train negatives, exact training evidence | 1 (18), 2 (2) | 90% | 10% |
| Validation negatives, source passage only | 1 | 100% | 0% |
| Validation negatives, retrieved top-k | 3 | 0% | 100% |

The same validation questions went from 100% abstention to 0% when their
evidence changed from one passage to three. The training set had exactly that
split: positives had 3 passages, and 466 of 500 negatives had 1. So the
adapter learned "one passage → abstain, three → answer" instead of reading the
evidence.

**Fix (in the notebook):** both labels now get one layout. Each example has
exactly `top_k` passages: its source passage at a rotating position balanced
per label, plus `top_k - 1` retrieved distractors from other articles. Only
the content of the source passage separates answerable from unanswerable.
Excluding same-article distractors keeps negative labels valid, because SQuAD
v2 labels only the source passage. The training-data cell writes
`training-data-audit.json`/`.png` and refuses to train if passage counts or
source positions differ by label. A regression test rejects the old layout.

Remaining mismatch: at inference, retrieval often returns sibling paragraphs
from the same article, which training never shows as distractors. This applies
equally to both labels, so it is not a label cue, but it is a residual
distribution difference.

**Scoring change:** `format_valid` now accepts a single Markdown code fence
around otherwise valid JSON. `strict_format_rate` keeps the bare-JSON
measurement. Re-parsing the archived `20260927-184112-full` predictions this
way changes Base + RAG from 2% to 94% valid JSON, and its answerable F1 from
0.00 to 0.45. The old strict parser had turned the RAG baseline into a straw
man. All four baselines must be recomputed with the new notebook.

**Run flow:** the sanity probe no longer raises. The diagnostic always runs,
and the Generation gate cell draws `generation-gate.png` and records the verdict
in `manifest.json`. A failure is a warning by default (`enforce_gate = False`), so
the run still produces results; each results figure is then labelled as a
diagnostic.

## Result: `20260928-011948-full` (layout fix applied)

The audit passed: all 2,000 training examples had 3 passages, and source
positions were balanced within ±1 per label. The passage-count shortcut is
gone. The adapter no longer abstains on one-passage evidence either, but it
now answers almost everything:

| Probe (n = 20 each) | Answered | Abstained |
|---|---|---|
| Train negatives, exact training evidence | 85% | 15% |
| Validation negatives, source passage only | 95% | 5% |
| Validation negatives, retrieved top-k | 95% | 0% (5% invalid) |
| Validation answerable (token F1 0.65) | 95% | 0% |

The adapter fails to abstain even on negatives it was trained on, with the
exact training evidence. So this is under-learning of the abstain decision,
not a train/inference distribution gap. Plausible contributors, none yet
tested in isolation:

- the decision is a single `true`/`false` token among roughly 15 target tokens,
  so the loss barely weights it;
- 75/25 answerable sampling;
- only 0.5 epoch, so about 250 negatives were seen;
- SQuAD v2 negatives are adversarially close to answerable questions.

Teacher-forced validation loss (0.077) is low because it averages over the
fully determined JSON tokens; it cannot show this failure.

Next, without retraining: score P(`abstain` = true) at the decision token on the
validation set. A good AUROC with a badly placed threshold would mean
calibration fixes it. An AUROC near 0.5 would mean the adapter cannot tell the
classes apart, so the training signal needs changing (balance, more steps, or
extra weight on the decision token).
