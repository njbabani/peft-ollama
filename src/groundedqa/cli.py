"""Command-line entry point: `groundedqa <stage> --config ... --run-dir ...`."""

import argparse
import json
import sys
import time
from pathlib import Path

STAGES = ["prepare", "retrieval-bench", "build-training-data", "train", "evaluate", "summarise", "report"]


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _resolve(args):
    from groundedqa.config import config_from_dict, load_config

    if args.config:
        config = load_config(args.config)
    elif args.run_dir and (Path(args.run_dir) / "manifest.json").exists():
        config = config_from_dict(json.loads((Path(args.run_dir) / "manifest.json").read_text())["config"])
    else:
        raise SystemExit("Pass --config, or --run-dir of a prepared run")
    run_dir = Path(args.run_dir or Path("runs") / config.name)
    return config, run_dir


def run_stage(stage: str, config, run_dir: Path, device, resume: bool = True) -> None:
    from groundedqa import pipeline

    if stage == "prepare":
        pipeline.prepare(config, run_dir, device, _log)
    elif stage == "retrieval-bench":
        pipeline.retrieval_benchmark(config, run_dir, device, _log)
    elif stage == "build-training-data":
        audit = pipeline.build_training_data(config, run_dir, device, _log)
        failures = [f"{s}: {f}" for s, fs in audit["shortcut_failures"].items() for f in fs]
        if failures:
            raise SystemExit("Evidence layout predicts the label; refusing to train: " + "; ".join(failures))
    elif stage == "train":
        from groundedqa.train import train_adapter

        revision = pipeline.load_manifest(run_dir)["revisions"]["model"]
        summary = train_adapter(config, run_dir, device, revision, _log, resume=resume)
        _log(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))
    elif stage == "evaluate":
        pipeline.evaluate(config, run_dir, device, _log, resume=resume)
    elif stage == "export":
        from groundedqa.serving import export_to_ollama

        revision = pipeline.load_manifest(run_dir)["revisions"]["model"]
        names = export_to_ollama(config, run_dir, revision, _log)
        _log(f"Ollama models: {names[False]} (base), {names[True]} (adapter)")
    elif stage == "evaluate-ollama":
        pipeline.evaluate_ollama(config, run_dir, device, _log, resume=resume)
    elif stage == "summarise":
        pipeline.summarise(config, run_dir, log=_log)
    elif stage == "report":
        from groundedqa.figures import render_all
        from groundedqa.report import build_summary

        for path in render_all(run_dir, config.name):
            _log(f"wrote {path}")
        (run_dir / "summary.md").write_text(build_summary(run_dir))
        _log(f"wrote {run_dir / 'summary.md'}")
    else:
        raise SystemExit(f"Unknown stage {stage}")


def ask_question(config, run_dir: Path, device, args) -> dict:
    from groundedqa.pipeline import ask

    return ask(config, run_dir, device, args.question, split=args.split, adapter=not args.base)


def _print_answer(result: dict) -> None:
    decision = "ABSTAIN" if result["abstain"] else "ANSWER"
    print(f"\nQ: {result['question']}")
    print(
        f"{decision} ({result['variant']}): P(no answer) = {result['no_answer_probability']:.3f} "
        f"vs threshold {result['threshold']:.3f}"
    )
    if not result["abstain"]:
        print(f"A: {result['answer']}  cites {', '.join(result['citations']) or 'nothing'}")
    for item in result["evidence"]:
        marker = "*" if item["label"] in result["citations"] and not result["abstain"] else " "
        print(f" {marker}[{item['label']}] {item['title']}: {item['text'][:160]}...")
    print(f"({result['seconds']:.2f} s)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="groundedqa",
        description="Grounded QA with LoRA fine-tuning, hybrid retrieval and calibrated abstention.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    helps = {
        "run": "all training/evaluation stages in order",
        "export": "merge the adapter and create 4-bit Ollama models (needs `ollama serve`)",
        "evaluate-ollama": "score the 4-bit Ollama models with the same protocol",
    }
    for name in STAGES + ["run", "export", "evaluate-ollama"]:
        stage = sub.add_parser(name, help=helps.get(name, f"{name} stage"))
        stage.add_argument("--config", help="TOML config (required for prepare/run)")
        stage.add_argument("--run-dir", help="run directory (default: runs/<config name>)")
        stage.add_argument("--device", help="cuda, mps or cpu (default: best available)")
        stage.add_argument(
            "--no-resume", action="store_true", help="ignore saved training checkpoints and evaluation caches"
        )
        if name == "run":
            stage.add_argument(
                "--from", dest="start", choices=STAGES, default="prepare", help="resume from this stage"
            )
    demo = sub.add_parser("demo", help="Gradio app: base vs fine-tuned on retrieved evidence")
    demo.add_argument("--run-dir", required=True)
    demo.add_argument("--config", help=argparse.SUPPRESS)
    demo.add_argument(
        "--ollama", action="store_true", help="serve the 4-bit Ollama models instead of PyTorch"
    )
    demo.add_argument("--port", type=int, default=7860)
    demo.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    ask = sub.add_parser("ask", help="answer one question with a finished run's model and threshold")
    ask.add_argument("question")
    ask.add_argument("--run-dir", required=True)
    ask.add_argument("--config", help=argparse.SUPPRESS)
    ask.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ask.add_argument("--base", action="store_true", help="use the base model instead of the adapter")
    ask.add_argument("--device", help="cuda, mps or cpu (default: best available)")
    args = parser.parse_args(argv)

    from groundedqa.device import pick_device

    config, run_dir = _resolve(args)
    device = pick_device(args.device)
    if args.command == "demo":
        from groundedqa.demo import launch

        launch(run_dir, device, args.ollama, args.port)
        return
    if args.command == "ask":
        _print_answer(ask_question(config, run_dir, device, args))
        return
    _log(f"{args.command}: config={config.name} run_dir={run_dir} device={device}")
    stages = STAGES[STAGES.index(args.start) :] if args.command == "run" else [args.command]
    for stage in stages:
        started = time.perf_counter()
        _log(f"── {stage} ──")
        run_stage(stage, config, run_dir, device, resume=not args.no_resume)
        _release_memory(device)
        _log(f"{stage} finished in {(time.perf_counter() - started) / 60:.1f} min")


def _release_memory(device) -> None:
    """Return cached accelerator memory between stages.

    Without this, the allocator keeps ~10 GB of training state cached, and on a
    24 GB Mac the evaluation models then push the machine into heavy swapping.
    """
    import gc

    import torch

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
