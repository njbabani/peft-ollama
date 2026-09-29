# Project history: from a collapsed adapter to calibrated abstention

This project started as a Colab notebook (Unsloth QLoRA on an L4, FAISS retrieval,
Ollama export). Four full runs failed in instructive ways. Each diagnosis shaped
the current package design, so the path is recorded here. The notebooks and
their run artefacts are kept locally, outside the repository.

## Run 1 — the adapter always abstained

`20260927-184112-full`. On both the NF4 and GGUF backends, the adapted model
abstained on all 200 test questions. Its 0.50 token F1 was exactly the
always-abstain score on a 50/50 test set.

Suspected cause: the completion mask. The mask check accepted any batch with at
least one ignored and one supervised token, so it could not catch a mask covering
only two tokens (see [TRL #6105](https://github.com/huggingface/trl/issues/6105)).

Fix: tokenize the prompt and the full reply separately, and require the prompt
tokens to be an exact prefix of the reply. Then audit the labels of every batch
before training.

## Run 2 — masks fixed, still abstaining

With the loss boundary verified, the adapter still mostly abstained. The next run
changed several things at once:

- abstain-first JSON (`{"abstain": ..., "answer": ..., "citations": ...}`);
- short `E1..E3` citation labels;
- 75/25 answerable/unanswerable training data;
- positives shown with retrieved distractors.

## Run 3 — the evidence-count shortcut

`20260928-001727-full`. Now the adapter never abstained on retrieved evidence. A
no-retraining diagnostic tested the same 20 validation negatives two ways:

| Evidence supplied | Abstained |
|---|---|
| The one source passage | 100% |
| Three retrieved passages | 0% |

The training data had exactly this split: positives always got 3 passages, while
466 of 500 negatives got 1. The adapter had learned *passage count → decision*.

Fix: give both labels the same layout — `top_k` passages, with the source passage
at a balanced rotating position. A pre-training audit now refuses to train if the
layout differs by label. This check is `evidence_shortcut_failures` in the package.

## Run 4 — extraction learned, abstention not

`20260928-040756-full`. The layout audit passed, and answer extraction clearly
worked: adapter + RAG reached 0.77 answerable F1, +0.76 over base + RAG (95% CI
[0.68, 0.83]). But the adapter still answered 86% of unanswerable questions, and
even 85% of its **own training negatives**. It had learned "evidence → answer,
no evidence → abstain" rather than whether the evidence supports an answer.

The run also showed an unfair baseline. The abstain-first prompt made the base
model write schema-inconsistent replies such as
`{"abstain": true, "answer": "None of the above"}`. The strict parser rejected
94% of them, so the baselines measured formatting rather than question answering.

## What the package changes

| Problem | Change |
|---|---|
| Abstention decided by one token, diluted in a ~20-token loss | The `" true"`/`" false"` decision token gets extra loss weight (`decision_weight`). |
| Greedy decoding hides whether the model can separate the classes | Score P(abstain) at the decision token; choose each variant's threshold on validation (the SQuAD 2.0 `best_f1_thresh` rule); report threshold-free AUROC. |
| Baselines lost on formatting | Force the prefix `{"abstain": false, "answer": "` so every variant is scored on content; strict formatting is still reported. |
| Few negatives, and a 75/25 imbalance | 1100 answerable / 1000 unanswerable, plus 300 **retrieval-miss** negatives: answerable questions with the source passage withheld and no passage containing the answer. |
| Tiny retrieval corpus (about 200 chunks; recall@3 0.98 was trivial) | Index every paragraph of every held-out article (thousands of chunks). Benchmark dense, BM25, hybrid RRF and hybrid + cross-encoder rerank. |
| Distractors never looked like inference-time retrieval | Training evidence comes from the same retriever used at evaluation. |
| CUDA-only stack (Unsloth, bitsandbytes) | Plain PyTorch + Hugging Face PEFT LoRA; runs on Apple-silicon MPS, CUDA or CPU. |
| bf16 inference moved P(abstain) by up to about 0.1 on MPS | Evaluate in fp16, which matches fp32 to within about 0.002; train in bf16. |
| faiss-cpu and torch load duplicate OpenMP runtimes on macOS | Exact cosine search in NumPy, with the same results as a flat FAISS index at this scale. |
