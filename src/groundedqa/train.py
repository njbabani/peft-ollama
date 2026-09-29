"""LoRA fine-tuning with a plain PyTorch loop (CUDA, Apple MPS or CPU).

The loss is a weighted token cross-entropy over the reply only: prompt and
padding carry no loss, and the " true"/" false" abstain decision token is
up-weighted so the one token that decides answer-vs-abstain is not diluted by
the ~15-25 formatting tokens around it.
"""

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from groundedqa.device import dtype_from_name, peak_memory_gib, seed_everything, synchronize
from groundedqa.metrics import roc_auc
from groundedqa.training_data import (
    SupervisedExampleTooLong,
    collate,
    decision_token_ids,
    encode_example,
    validate_supervised_batch,
)


def weighted_reply_loss(model: Any, batch: dict, true_id: int, false_id: int):
    """Weighted CE over reply tokens, plus P(abstain) at each decision position.

    Only the last ``keep`` positions go through the LM head, which is enough to
    cover every reply in a right-padded batch.
    """
    input_ids, labels, weights = batch["input_ids"], batch["labels"], batch["weights"]
    width = input_ids.shape[1]
    first_reply = (labels != -100).int().argmax(-1)
    keep = width - int(first_reply.min()) + 1
    out = model(
        input_ids=input_ids,
        attention_mask=batch["attention_mask"],
        logits_to_keep=keep,
        use_cache=False,
    )
    offset = width - keep
    logits = out.logits[:, :-1].float()
    targets = labels[:, offset + 1 :]
    token_weights = weights[:, offset + 1 :]
    ce = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100, reduction="none")
    loss = (ce * token_weights).sum() / token_weights.sum()
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    decision_logits = out.logits[rows, batch["decision_position"] - offset - 1].float()
    p_abstain = decision_logits[:, [true_id, false_id]].softmax(-1)[:, 0]
    return loss, p_abstain.detach()


def _rng_state(device: torch.device) -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict, device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])
    elif device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(run_dir: Path, model, trainer_state: dict, keep: int) -> Path:
    """Write LoRA weights and trainer state to checkpoints/step-N atomically; prune old ones."""
    root = run_dir / "checkpoints"
    final = root / f"step-{trainer_state['step']:05d}"
    staging = root / f".staging-{trainer_state['step']:05d}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    model.save_pretrained(str(staging))
    torch.save(trainer_state, staging / "trainer_state.pt")
    shutil.rmtree(final, ignore_errors=True)
    staging.rename(final)
    for old in sorted(root.glob("step-*"))[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    return final


def latest_checkpoint(run_dir: Path) -> Path | None:
    complete = sorted(
        p for p in (run_dir / "checkpoints").glob("step-*") if (p / "trainer_state.pt").exists()
    )
    return complete[-1] if complete else None


def _encode_all(tokenizer, examples, config, decision) -> tuple[list[dict], int]:
    encoded, dropped = [], 0
    for example in examples:
        try:
            encoded.append(
                encode_example(
                    tokenizer,
                    example,
                    config.model.max_seq_length,
                    decision,
                    config.train.decision_weight,
                )
            )
        except SupervisedExampleTooLong:
            dropped += 1
    return encoded, dropped


@torch.inference_mode()
def evaluate_monitor(model, encoded, pad_id, device, true_id, false_id) -> dict[str, float]:
    model.eval()
    losses, probs, labels = [], [], []
    for item in encoded:
        batch = {k: v.to(device) for k, v in collate([item], pad_id).items()}
        loss, p_abstain = weighted_reply_loss(model, batch, true_id, false_id)
        losses.append(loss.item())
        probs.extend(p_abstain.tolist())
        labels.extend(batch["abstain"].tolist())
    model.train()
    return {
        "eval_loss": sum(losses) / len(losses),
        "eval_decision_auroc": roc_auc(labels, probs),
        "eval_decision_acc": sum((p >= 0.5) == y for p, y in zip(probs, labels)) / len(labels),
    }


def train_adapter(
    config, run_dir: Path, device: torch.device, revision: str, log=print, resume: bool = True
) -> dict:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    cfg = config.train
    seed_everything(config.data.seed)
    data_dir = run_dir / "data"
    train_examples = json.loads((data_dir / "training-examples-train.json").read_text())
    monitor_examples = json.loads((data_dir / "training-examples-validation.json").read_text())

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, revision=revision)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    decision = decision_token_ids(tokenizer)
    _, true_id, false_id = decision
    train_encoded, train_dropped = _encode_all(tokenizer, train_examples, config, decision)
    monitor_encoded, _ = _encode_all(tokenizer, monitor_examples, config, decision)
    if not train_encoded:
        raise ValueError("No training examples fit max_seq_length")

    # Audit the exact tensors the loop will see before any optimisation.
    for start in range(0, min(len(train_encoded), 64), 4):
        chunk = train_encoded[start : start + 4]
        validate_supervised_batch(chunk, collate(chunk, pad_id))

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_id, revision=revision, dtype=dtype_from_name(config.model.train_dtype)
    ).to(device)
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    params = [p for p in model.parameters() if p.requires_grad]

    rng = random.Random(config.data.seed)
    order: list[int] = []
    for _ in range(math.ceil(cfg.epochs)):
        epoch = list(range(len(train_encoded)))
        rng.shuffle(epoch)
        order.extend(epoch)
    order = order[: int(len(train_encoded) * cfg.epochs)]
    micro_batches = [order[i : i + cfg.batch_size] for i in range(0, len(order), cfg.batch_size)]
    total_steps = math.ceil(len(micro_batches) / cfg.grad_accum)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)

    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, max(1, int(cfg.warmup_ratio * total_steps)), total_steps
    )
    log(
        f"Training {trainable:,} LoRA parameters ({trainable / total:.2%}) on {device}: "
        f"{len(train_encoded)} examples, {total_steps} optimizer steps"
    )

    # Identifies the schedule a checkpoint belongs to; resuming a different one is refused.
    plan = {
        "examples": len(train_encoded),
        "total_steps": total_steps,
        "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
        "seed": config.data.seed,
    }
    checkpoint = latest_checkpoint(run_dir) if resume else None
    if checkpoint is not None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        state = torch.load(checkpoint / "trainer_state.pt", weights_only=False)
        if state["plan"] != plan:
            raise ValueError(
                f"{checkpoint} was saved for {state['plan']}, not {plan}. "
                "Delete the checkpoints folder or pass --no-resume to start over."
            )
        set_peft_model_state_dict(
            model, load_file(str(checkpoint / "adapter_model.safetensors"), device=str(device))
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        _set_rng_state(state["rng"], device)
        history, cursor, elapsed_before = state["history"], state["cursor"], state["elapsed_s"]
        first_step = state["step"] + 1
        initial = {k: history[0][k] for k in ("eval_loss", "eval_decision_auroc", "eval_decision_acc")}
        log(f"Resumed from {checkpoint.name}: continuing at step {first_step}/{total_steps}")
    else:
        history = []
        initial = evaluate_monitor(model, monitor_encoded, pad_id, device, true_id, false_id)
        history.append({"step": 0, **initial})
        log(f"step 0 | {json.dumps({k: round(v, 4) for k, v in initial.items()})}")
        cursor, elapsed_before, first_step = 0, 0.0, 1
    model.train()
    synchronize(device)
    started = time.perf_counter()
    for step in range(first_step, total_steps + 1):
        step_loss, correct, seen = 0.0, 0, 0
        for _ in range(cfg.grad_accum):
            if cursor >= len(micro_batches):
                break
            items = [train_encoded[i] for i in micro_batches[cursor]]
            cursor += 1
            batch = {k: v.to(device) for k, v in collate(items, pad_id).items()}
            loss, p_abstain = weighted_reply_loss(model, batch, true_id, false_id)
            (loss / cfg.grad_accum).backward()
            step_loss += loss.item() / cfg.grad_accum
            correct += int(((p_abstain >= 0.5) == batch["abstain"]).sum())
            seen += len(items)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        row = {
            "step": step,
            "loss": step_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
            "decision_acc": correct / max(seen, 1),
            "elapsed_s": elapsed_before + time.perf_counter() - started,
        }
        if step % cfg.eval_every == 0 or step == total_steps:
            row.update(evaluate_monitor(model, monitor_encoded, pad_id, device, true_id, false_id))
        history.append(row)
        if step % 10 == 0 or "eval_loss" in row or step == 1:
            eta = row["elapsed_s"] / step * (total_steps - step)
            log(
                f"step {step}/{total_steps} | loss {step_loss:.4f} | lr {row['learning_rate']:.2e}"
                + (
                    f" | eval_loss {row['eval_loss']:.4f} | decision AUROC {row['eval_decision_auroc']:.3f}"
                    if "eval_loss" in row
                    else ""
                )
                + f" | eta {eta / 60:.1f} min"
            )
        (run_dir / "training-log.json").write_text(json.dumps(history, indent=2))
        if step % cfg.save_every == 0 and step < total_steps:
            saved = save_checkpoint(
                run_dir,
                model,
                {
                    "plan": plan,
                    "step": step,
                    "cursor": cursor,
                    "history": history,
                    "elapsed_s": row["elapsed_s"],
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": _rng_state(device),
                },
                cfg.keep_checkpoints,
            )
            log(f"checkpoint saved: {saved.relative_to(run_dir)}")
    synchronize(device)
    wall = elapsed_before + time.perf_counter() - started

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    summary = {
        "device": str(device),
        "wall_seconds": wall,
        "resumed_from": checkpoint.name if checkpoint is not None else None,
        "optimizer_steps": total_steps,
        "examples_seen": min(cursor * cfg.batch_size, len(order)),
        "train_examples": len(train_encoded),
        "train_dropped_too_long": train_dropped,
        "monitor_examples": len(monitor_encoded),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "peak_memory_gib": peak_memory_gib(device),
        "adapter_bytes": sum(p.stat().st_size for p in adapter_dir.rglob("*") if p.is_file()),
        "initial_monitor": initial,
        "final_monitor": {k: history[-1][k] for k in initial},
    }
    (run_dir / "training.json").write_text(json.dumps(summary, indent=2))
    return summary
