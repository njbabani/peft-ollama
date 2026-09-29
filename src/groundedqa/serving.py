"""Local deployment: merge the LoRA adapter, quantise with Ollama, and query it.

``OllamaDecoder`` mirrors ``inference.AnswerDecoder``: it takes the same prompt
token IDs and returns the same fields, so the evaluation pipeline can score a
4-bit Ollama model exactly like the fp16 PyTorch one. The no-answer probability
comes from Ollama's ``top_logprobs`` at the decision token.
"""

import json
import math
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from groundedqa.prompting import ANSWER_PREFIX, DECISION_PREFIX
from groundedqa.training_data import decision_token_ids

OLLAMA_URL = "http://127.0.0.1:11434"
QUANTIZATION = "q4_K_M"


def ollama_ready(url: str = OLLAMA_URL) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def model_names(run_dir: Path) -> dict[bool, str]:
    """Ollama model names for the base (False) and adapted (True) variants of a run."""
    tag = QUANTIZATION.lower()
    return {
        False: f"groundedqa-{run_dir.name}-base:{tag}",
        True: f"groundedqa-{run_dir.name}-adapter:{tag}",
    }


def merge_adapter(model_id: str, revision: str, adapter_dir: Path, out_dir: Path) -> Path:
    """Fold the LoRA weights into the base model and save fp16 safetensors."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if out_dir.exists():  # only ever created by the rename below, so always complete
        return out_dir
    staging = out_dir.with_name(out_dir.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    base = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(staging), safe_serialization=True)
    AutoTokenizer.from_pretrained(model_id, revision=revision).save_pretrained(str(staging))
    staging.rename(out_dir)
    return out_dir


def base_snapshot(model_id: str, revision: str) -> Path:
    """The cached Hugging Face snapshot of the base model (safetensors + tokenizer)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, revision=revision))


def llama_cpp_converter(llama_cpp_dir: Path | None = None) -> Path:
    """convert_hf_to_gguf.py from a llama.cpp checkout matching the installed llama-quantize.

    Homebrew ships the binaries but not the Python converter. Clone the same tag:
    ``git clone --depth 1 --branch <tag> https://github.com/ggml-org/llama.cpp runs/tools/llama.cpp``
    (``brew info --json=v2 llama.cpp`` shows the tag), or set ``LLAMA_CPP_DIR``.
    """
    import os

    root = Path(llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR", "runs/tools/llama.cpp"))
    script = root / "convert_hf_to_gguf.py"
    if not script.exists():
        raise RuntimeError(f"{script} not found; see groundedqa.serving.llama_cpp_converter for setup")
    if shutil.which("llama-quantize") is None:
        raise RuntimeError("`llama-quantize` not found (macOS: `brew install llama.cpp`)")
    return script


def quantize_to_gguf(source_dir: Path, out_path: Path, converter: Path) -> Path:
    """HF safetensors -> f16 GGUF (llama.cpp converter) -> Q4_K_M GGUF (llama-quantize)."""
    import sys

    if out_path.exists():
        return out_path
    f16 = out_path.with_suffix(".f16.gguf")
    partial = out_path.with_suffix(".partial")
    subprocess.run(
        [sys.executable, str(converter), str(source_dir), "--outfile", str(f16), "--outtype", "f16"],
        check=True,
        timeout=3600,
    )
    subprocess.run(["llama-quantize", str(f16), str(partial), QUANTIZATION.upper()], check=True, timeout=3600)
    partial.rename(out_path)  # an interrupted quantisation never leaves a file that looks finished
    f16.unlink()
    return out_path


def create_ollama_model(name: str, gguf_path: Path, work_dir: Path, num_ctx: int) -> None:
    """``ollama create`` from a quantised GGUF; prompts are sent raw, so no template is needed."""
    if shutil.which("ollama") is None:
        raise RuntimeError("The `ollama` CLI is not installed (macOS: `brew install ollama`).")
    modelfile = work_dir / f"Modelfile-{name.split(':')[0]}"
    modelfile.write_text(
        f"FROM {gguf_path.resolve()}\n"
        f"PARAMETER num_ctx {num_ctx}\n"
        'PARAMETER stop "<|im_end|>"\n'
        'PARAMETER stop "<|endoftext|>"\n'
        "PARAMETER temperature 0\n"
    )
    subprocess.run(["ollama", "create", name, "-f", str(modelfile)], check=True, timeout=3600)


def export_to_ollama(config, run_dir: Path, revision: str, log=print) -> dict[bool, str]:
    if not ollama_ready():
        raise RuntimeError("Start the Ollama server first: `ollama serve`")
    converter = llama_cpp_converter()
    names = model_names(run_dir)
    export_dir = run_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    log("merging the LoRA adapter into the base weights (fp16)")
    merged = merge_adapter(config.model.model_id, revision, run_dir / "adapter", export_dir / "merged")
    sources = {False: base_snapshot(config.model.model_id, revision), True: merged}
    for adapted, name in names.items():
        label = "adapter" if adapted else "base"
        log(f"converting {label} to GGUF and quantising to {QUANTIZATION}")
        gguf_path = quantize_to_gguf(sources[adapted], export_dir / f"{label}.{QUANTIZATION}.gguf", converter)
        log(f"ollama create {name} ({gguf_path.stat().st_size / 2**30:.2f} GiB)")
        create_ollama_model(name, gguf_path, export_dir, config.model.max_seq_length)
    (export_dir / "ollama-models.json").write_text(
        json.dumps({"base": names[False], "adapter": names[True]}, indent=2)
    )
    return names


def _generate(url: str, payload: dict, timeout: float = 300) -> dict:
    request = urllib.request.Request(
        f"{url}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def decision_probability(top_logprobs: list[dict], true_token: str, false_token: str) -> float:
    """P(abstain) renormalised over the two decision tokens.

    A token missing from the top list gets the smallest listed log-probability,
    an upper bound on its true value.
    """
    if not top_logprobs:
        raise ValueError("Ollama returned no top_logprobs for the decision token")
    by_token = {entry["token"]: entry["logprob"] for entry in top_logprobs}
    floor = min(by_token.values())
    lp_true, lp_false = by_token.get(true_token, floor), by_token.get(false_token, floor)
    top = max(lp_true, lp_false)
    return math.exp(lp_true - top) / (math.exp(lp_true - top) + math.exp(lp_false - top))


class OllamaDecoder:
    """Same interface as ``AnswerDecoder``, served by a quantised Ollama model.

    ``prompt_builder`` is an ``AnswerDecoder`` used only for its tokenizer-exact
    prompt fitting, so both backends see identical prompts.
    """

    def __init__(self, prompt_builder: Any, model: str, url: str = OLLAMA_URL):
        self.builder = prompt_builder
        self.tokenizer = prompt_builder.tokenizer
        self.model = model
        self.url = url
        _, true_id, false_id = decision_token_ids(self.tokenizer)
        self.true_token = self.tokenizer.decode([true_id])
        self.false_token = self.tokenizer.decode([false_id])

    def prompt_ids(self, question: str, evidence: list[dict]):
        return self.builder.prompt_ids(question, evidence)

    def _chat_text(self, ids: list[int]) -> str:
        chat = ids[: len(ids) - len(self.builder.prefix_ids)]
        return self.tokenizer.decode(chat, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def run(self, batch_ids: list[list[int]], adapted: bool) -> list[dict]:
        options = {"temperature": 0, "seed": 0, "num_ctx": self.builder.max_seq_length}
        results = []
        for ids in batch_ids:
            chat = self._chat_text(ids)
            started = time.perf_counter()
            decision = _generate(
                self.url,
                {
                    "model": self.model,
                    "prompt": chat + DECISION_PREFIX,
                    "raw": True,
                    "stream": False,
                    "keep_alive": "30m",
                    "logprobs": True,
                    "top_logprobs": 20,
                    "options": {**options, "num_predict": 1},
                },
            )
            answer = _generate(
                self.url,
                {
                    "model": self.model,
                    "prompt": chat + ANSWER_PREFIX,
                    "raw": True,
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {**options, "num_predict": self.builder.max_new_tokens},
                },
            )
            wall = time.perf_counter() - started
            decode_s = answer.get("eval_duration", 0) / 1e9
            results.append(
                {
                    "na_prob": decision_probability(
                        decision["logprobs"][0]["top_logprobs"], self.true_token, self.false_token
                    ),
                    "completion": answer["response"],
                    "raw": ANSWER_PREFIX + answer["response"],
                    "output_tokens": answer.get("eval_count", 0),
                    "prefill_s": max(wall - decode_s, 0.0),
                    "decode_s": decode_s,
                }
            )
        return results
