# Completion-mask repair

## Evidence and scope

The archived `20260927-184112-full` run used TRL 0.24.0 and Transformers
5.5.0. Its adapted models abstained on all 200 evaluation questions, on both
backends. The original mask check accepted any batch with at least one ignored
token and at least one supervised token. It could not detect a two-token prompt
mask. The original length check also relied on the default return type of
`apply_chat_template`.

The compatibility problem is reported in
https://github.com/huggingface/trl/issues/6105. The archived evidence does not
contain the actual training labels, so it does not prove this was the sole cause
of collapse. Treat the old adapter's 50% score as an always-abstain baseline,
not a successful fine-tuning result.

## Acceptance criteria

- Explicitly tokenize the prompt and complete conversation without relying on
  a default return type; require an exact prompt-token prefix.
- Construct completion masks before handing examples to TRL.
- Check actual trainer labels for every training and validation example:
  prompt and padding ignored, complete assistant response supervised.
- Reject malformed boundaries and avoid truncating assistant targets.
- Regression checks must reject the old two-token-only mask.
- Preserve prior results and run the corrected experiment in a new directory.

Keep model, seed, data selection, class balance, and training budget unchanged
for the first corrected comparison. Oracle single-chunk training versus
retrieved multi-chunk inference is a separate follow-up experiment, not evidence
that this repair has failed or succeeded.

## Runtime verification

The corrected Colab run must pass its pre-training mask audit. Then inspect
answerable-question performance, abstention frequency, raw generations, and
format validity alongside aggregate F1. A low teacher-forced validation loss
alone does not establish useful question answering.
