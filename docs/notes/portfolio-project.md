# GroundedQA: Colab fine-tuning and retrieval experiments

Status: notebook implemented and locally checked; Colab GPU smoke run and real benchmark pending. Target: AI/MLE roles. Runtime: Colab GPU through VS Code. All executable project code lives in `grounded_qa_colab.ipynb`.

## Product and acceptance criteria

Answer questions against a versioned SQuAD v2 passage corpus, including unanswerable questions. This benchmark foundation can later be adapted to technical-support documents.
Show supporting passages and source citations; abstain when evidence is insufficient.
Use Unsloth for efficient training, PEFT LoRA/QLoRA adapters for behavior tuning,
and Ollama for deployment. Sentence Transformers and FAISS provide retrieval. RAG supplies factual context.
Do not assume fine-tuning improves factual accuracy: measure its contribution.

Preserve `train.ipynb` and `data/people_data.json`. The existing 300 prompt/response
records are excluded from training until their provenance and relevance are checked.
No GPU purchases, cloud jobs, publication, or model downloads are part of planning.

## Experiment design

Compare base, base + RAG, fine-tuned, and fine-tuned + RAG with the same held-out
questions, base checkpoint, decoding parameters, runtime quantization, and hardware.
Use document/topic-disjoint training, validation, and test splits; avoid near-duplicate
questions across splits. Include answerable and unanswerable cases. Conflicting-evidence
and prompt-injection challenge sets are future extensions, not claimed benchmark coverage.
Keep evaluation answers out of training. Test documents may be indexed for retrieval
at inference; their reference answers must never enter the index.

Report sample counts, raw predictions, configuration, corpus/model identifiers, and:

- Answer correctness with a documented rubric and human-reviewed subset.
- Citation precision and evidence coverage; citation existence alone is insufficient.
- Abstention precision/recall on labeled unanswerable cases.
- Retrieval Recall@k and MRR with annotated relevant passages.
- End-to-end p50/p95 latency and generation tokens/sec, with warm/cold runs separated.
- Training wall time, peak allocated/reserved GPU memory, trainable parameter fraction,
  adapter size, and final deployment artifact size.

Only claim improvements after measuring comparable runs. A demo fixture is not a
benchmark. Preserve negative results and report uncertainty and evaluation-set size.

Implementation uses annotated answer-span containment for chunk relevance; this does
not enumerate every semantically relevant passage. Abstention CSV columns are explicitly
source-passage-relative. The no-RAG adapter arm is an extrapolation stress test because
training examples provide evidence.

## Architecture

SQuAD passages -> validated records -> provenance-bearing chunks -> embeddings/index
-> retrieved evidence -> shared prompt template -> Ollama -> answer and citations.

Training examples -> split/validate -> Unsloth + PEFT -> adapter/checkpoint -> GGUF
export with correct chat template -> Ollama model -> the same evaluation pipeline.

All executable code stays in a single Colab notebook, including unit tests, configuration,
training, evaluation, plots, a demo, and artifact export. No API server or external
application package is required. Use Qwen2.5-1.5B-Instruct as a modest T4-targeted
starting point, and detect available NVIDIA GPU memory before training. Colab GPU
allocation remains controlled by the user's runtime/account.

## Delivery steps

1. **Data and evaluation contract.** Notebook setup, provenance, title-disjoint splits,
   structured answer schema, scoring helpers and embedded offline tests. SQuAD v2
   labels are passage-specific; acknowledge limitations for broader retrieval.
2. **RAG and training.** Sentence Transformers + FAISS, bounded cited context,
   QLoRA with Unsloth/PEFT and completion-only SFT. Capture resolved revisions,
   environment, dataset IDs, trainable parameters and measured GPU resources.
3. **Controlled experiments and deployment.** Same-checkpoint adapter-off/on ×
   retrieval-off/on evaluation; export GGUF and serve through Ollama within Colab.
   Keep native and GGUF runtime results separate because quantization differs.
4. **Portfolio evidence.** Raw predictions, summary tables, plots, human review sheet,
   artifact archive and CV templates populated only after real measurements.

Verification: failing tests first for pure helpers, then embedded tests, notebook
schema/syntax checks, independent code/RAG review. GPU training, conversion and
real Ollama inference must be validated in Colab and cannot be claimed from local
static/offline validation. Target >=80% measured line coverage of pure helper code;
this does not represent coverage of GPU or deployment execution.

Local result: schema, syntax and Ruff E4/E7/E9/F checks pass; 16 helper tests and
2 stubbed pipeline tests pass. Pure-helper line coverage: 98.1% (154 statements).
Independent review informed span-based retrieval labels, explicit metric scope,
checkpoint revision pinning and PEFT-aware export. No model quality results yet.

Preserve `train.ipynb` and existing data. No Git repository currently exists. Do not
publish or claim benchmark improvements before actually executing the experiments.

## Primary references

- https://unsloth.ai/docs/get-started/fine-tuning-llms-guide
- https://huggingface.co/docs/peft/main/index
- https://ollama.com/blog/embedding-models
- https://huggingface.co/datasets/rajpurkar/squad_v2
- https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
- https://huggingface.co/docs/trl/sft_trainer
