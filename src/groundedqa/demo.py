"""Gradio demo: base vs fine-tuned, side by side, on retrieved evidence with citations.

uv sync --extra demo
uv run groundedqa demo --run-dir runs/full            # PyTorch backend
uv run groundedqa demo --run-dir runs/full --ollama   # 4-bit Ollama backend
"""

import html
import random
import time
from pathlib import Path

VARIANTS = {False: "Base Qwen2.5-1.5B", True: "Fine-tuned (LoRA)"}
ACCENT = {False: "#2a78d6", True: "#1baf7a"}

CSS = """
.gqa-card {border: 1px solid #e1e0d9; border-radius: 12px; padding: 16px 18px; background: #fcfcfb;}
.gqa-head {display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;}
.gqa-name {font-weight: 700; font-size: 15px; color: #0b0b0b;}
.gqa-badge {font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 999px; letter-spacing: .04em;}
.gqa-answer {font-size: 22px; font-weight: 700; color: #0b0b0b; margin: 6px 0 4px; min-height: 30px;}
.gqa-muted {color: #52514e; font-size: 13px;}
.gqa-meter {position: relative; height: 10px; background: #f0efec; border-radius: 999px; margin: 14px 0 6px;}
.gqa-fill {position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px;}
.gqa-tick {position: absolute; top: -5px; width: 2px; height: 20px; background: #0b0b0b;}
.gqa-evidence {border: 1px solid #e1e0d9; border-radius: 10px; padding: 12px 14px; margin-bottom: 10px;
  background: #fcfcfb;}
.gqa-evidence.cited {border: 2px solid #1baf7a; background: #f3fbf7;}
.gqa-label {font-weight: 700; font-size: 13px; color: #0b0b0b;}
.gqa-chip {display: inline-block; font-size: 11px; font-weight: 700; border-radius: 6px; padding: 1px 7px;
  margin-left: 6px;}
.gqa-text {font-size: 13.5px; color: #2b2a28; line-height: 1.5; margin-top: 6px;}
"""


class DemoEngine:
    """Loads the retriever and both models once; answers questions for the UI."""

    def __init__(self, run_dir: Path, device, use_ollama: bool = False, split: str = "test"):
        from groundedqa.config import config_from_dict
        from groundedqa.device import dtype_from_name
        from groundedqa.inference import AnswerDecoder, load_model
        from groundedqa.pipeline import _embedder, _load_split, _reranker, load_manifest, read_json
        from groundedqa.retrieval import Retriever

        manifest = load_manifest(run_dir)
        self.config = config = config_from_dict(manifest["config"])
        revisions = manifest["revisions"]
        metrics_dir = run_dir / "ollama-q4_k_m" if use_ollama else run_dir
        metrics_path = metrics_dir / "metrics.json"
        metrics = read_json(metrics_path) if metrics_path.exists() else {"variants": {}}
        self.thresholds = {
            adapted: metrics["variants"]
            .get("adapter_rag" if adapted else "base_rag", {})
            .get("threshold", 0.5)
            for adapted in (False, True)
        }
        self.corpus, embeddings, self.questions = _load_split(run_dir, split)
        reranker = _reranker(config, revisions, device) if config.retrieval.mode == "hybrid_rerank" else None
        self.retriever = Retriever(
            self.corpus,
            _embedder(config, revisions, device),
            embeddings,
            reranker,
            config.retrieval.candidates,
            config.retrieval.rrf_k,
        )
        self.decoders = {}
        if use_ollama:
            from transformers import AutoTokenizer

            from groundedqa.serving import OllamaDecoder, model_names, ollama_ready

            if not ollama_ready():
                raise RuntimeError("Start the Ollama server first: `ollama serve`")
            tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, revision=revisions["model"])
            builder = AnswerDecoder(
                tokenizer, None, device, config.model.max_new_tokens, config.model.max_seq_length
            )
            names = model_names(run_dir)
            self.decoders = {adapted: OllamaDecoder(builder, names[adapted]) for adapted in (False, True)}
            self.backend = "Ollama · 4-bit Q4_K_M"
        else:
            dtype = dtype_from_name(config.model.eval_dtype)
            for adapted in (False, True):
                tokenizer, model = load_model(
                    config.model.model_id,
                    revisions["model"],
                    dtype,
                    device,
                    run_dir / "adapter" if adapted else None,
                )
                if adapted:
                    model = model.merge_and_unload().eval()
                self.decoders[adapted] = AnswerDecoder(
                    tokenizer, model, device, config.model.max_new_tokens, config.model.max_seq_length
                )
            self.backend = f"PyTorch · {config.model.eval_dtype} · {device}"

    def examples(self, n: int = 6, seed: int = 7) -> list[str]:
        rng = random.Random(seed)
        yes = [q["question"] for q in self.questions if q["answers"]["text"]]
        no = [q["question"] for q in self.questions if not q["answers"]["text"]]
        return rng.sample(yes, min(n // 2, len(yes))) + rng.sample(no, min(n - n // 2, len(no)))

    def answer(self, question: str) -> dict:
        from groundedqa.scoring import parse_prediction

        question = question.strip()
        started = time.perf_counter()
        rankings, _ = self.retriever.rank([question], self.config.retrieval.mode, self.config.retrieval.top_k)
        retrieval_s = time.perf_counter() - started
        evidence, ids = self.decoders[True].prompt_ids(question, [self.corpus[i] for i in rankings[0]])
        results = {}
        for adapted, decoder in self.decoders.items():
            t0 = time.perf_counter()
            out = decoder.run([ids], adapted)[0]
            parsed = parse_prediction(out["raw"])
            results[adapted] = {
                "p_no_answer": out["na_prob"],
                "threshold": self.thresholds[adapted],
                "abstain": out["na_prob"] >= self.thresholds[adapted] or not parsed["valid"],
                "answer": parsed["answer"] if parsed["valid"] else "",
                "citations": parsed["citations"] if parsed["valid"] else [],
                "seconds": time.perf_counter() - t0,
            }
        return {"question": question, "evidence": evidence, "results": results, "retrieval_s": retrieval_s}


def render_card(adapted: bool, result: dict) -> str:
    color = ACCENT[adapted]
    p, t = result["p_no_answer"], result["threshold"]
    badge = (
        '<span class="gqa-badge" style="background:#eceae4;color:#52514e">ABSTAINS</span>'
        if result["abstain"]
        else '<span class="gqa-badge" style="background:#e6f6ec;color:#006300">ANSWERS</span>'
    )
    answer = (
        '<span class="gqa-muted" style="font-size:18px">“The evidence does not support an answer.”</span>'
        if result["abstain"]
        else html.escape(result["answer"])
    )
    cites = ", ".join(result["citations"]) if result["citations"] and not result["abstain"] else "none"
    return f"""
<div class="gqa-card">
  <div class="gqa-head"><span class="gqa-name" style="border-left:4px solid {color};padding-left:8px">
  {VARIANTS[adapted]}</span>{badge}</div>
  <div class="gqa-answer">{answer}</div>
  <div class="gqa-muted">cites: {cites}</div>
  <div class="gqa-meter"><div class="gqa-fill" style="width:{p * 100:.1f}%;background:{color};opacity:.85"></div>
  <div class="gqa-tick" style="left:calc({t * 100:.1f}% - 1px)"></div></div>
  <div class="gqa-muted">P(no answer) <b>{p:.2f}</b> · abstains at ≥ {t:.2f} (calibrated on validation)
  · {result["seconds"]:.2f}s</div>
</div>"""


def render_evidence(evidence: list[dict], results: dict) -> str:
    cited = {adapted: set(r["citations"]) if not r["abstain"] else set() for adapted, r in results.items()}
    cards = []
    for index, chunk in enumerate(evidence, start=1):
        label = f"E{index}"
        chips = "".join(
            f'<span class="gqa-chip" style="background:{ACCENT[a]}22;color:{ACCENT[a]}">cited by '
            f"{'fine-tuned' if a else 'base'}</span>"
            for a in (True, False)
            if label in cited[a]
        )
        text = html.escape(chunk["text"][:520] + ("…" if len(chunk["text"]) > 520 else ""))
        cls = "gqa-evidence cited" if label in cited[True] else "gqa-evidence"
        cards.append(
            f'<div class="{cls}"><span class="gqa-label">[{label}] {html.escape(chunk["title"])}</span>'
            f'{chips}<div class="gqa-text">{text}</div></div>'
        )
    return "".join(cards) or '<div class="gqa-muted">No evidence retrieved.</div>'


def build_app(engine: DemoEngine):
    import gradio as gr

    def respond(question: str):
        if not question or not question.strip():
            return "", "", ""
        result = engine.answer(question)
        return (
            render_card(False, result["results"][False]),
            render_card(True, result["results"][True]),
            render_evidence(result["evidence"], result["results"]),
        )

    with gr.Blocks(title="GroundedQA demo") as app:
        gr.Markdown(
            "## GroundedQA · answer from evidence, or abstain\n"
            "Questions are answered from Wikipedia articles **held out from training** "
            f"(hybrid BM25 + dense retrieval, cross-encoder rerank, top-3). Backend: {engine.backend}."
        )
        with gr.Row():
            question = gr.Textbox(
                label="Question", placeholder="Ask about the held-out articles…", scale=5, autofocus=True
            )
            ask = gr.Button("Ask", variant="primary", scale=1)
        gr.Examples(engine.examples(), inputs=question, label="Try a test-set question")
        with gr.Row(equal_height=True):
            base_card = gr.HTML()
            tuned_card = gr.HTML()
        gr.Markdown("#### Retrieved evidence")
        evidence = gr.HTML()
        outputs = [base_card, tuned_card, evidence]
        ask.click(respond, question, outputs)
        question.submit(respond, question, outputs)
    return app


def launch(run_dir: Path, device, use_ollama: bool, port: int) -> None:
    import gradio as gr

    app = build_app(DemoEngine(run_dir, device, use_ollama))
    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="stone"),
    )
