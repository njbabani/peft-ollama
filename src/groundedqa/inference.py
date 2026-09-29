"""Batched abstention scoring and forced-prefix answer decoding.

One forward pass over ``prompt + '{"abstain": false, "answer": "'`` gives both
the probability of the " true" decision token (the no-answer probability) and
the start of the answer. Greedy decoding then continues from the KV cache, so
every question gets a no-answer probability *and* the answer the model would
give. Any abstention threshold can be applied afterwards.
"""

import time
from contextlib import nullcontext
from typing import Any

import torch

from groundedqa.device import synchronize
from groundedqa.prompting import ANSWER_PREFIX, build_messages
from groundedqa.training_data import decision_token_ids


def load_model(model_id: str, revision: str, dtype: torch.dtype, device: torch.device, adapter_dir=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
    model.to(device)
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return tokenizer, model


class AnswerDecoder:
    def __init__(
        self, tokenizer: Any, model: Any, device: torch.device, max_new_tokens: int, max_seq_length: int
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.max_seq_length = max_seq_length
        offset, self.true_id, self.false_id = decision_token_ids(tokenizer)
        self.prefix_ids = tokenizer.encode(ANSWER_PREFIX, add_special_tokens=False)
        if self.prefix_ids[offset] != self.false_id:
            raise ValueError("Answer prefix does not contain the decision token")
        # Logit rows kept: the first predicts the decision, the last the first answer token.
        self.keep = len(self.prefix_ids) - offset + 1
        self.stop_ids = {
            token
            for token in (
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|im_end|>"),
                tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            )
            if isinstance(token, int) and token >= 0
        }
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def prompt_ids(self, question: str, evidence: list[dict]) -> tuple[list[dict], list[int]]:
        """Drop whole trailing passages (never truncate one) until the prompt fits."""
        evidence = list(evidence)
        while True:
            text = self.tokenizer.apply_chat_template(
                build_messages(question, evidence), tokenize=False, add_generation_prompt=True
            )
            ids = self.tokenizer.encode(text, add_special_tokens=False) + self.prefix_ids
            if len(ids) + self.max_new_tokens <= self.max_seq_length:
                return evidence, ids
            if not evidence:
                raise ValueError("Question alone exceeds the model context budget")
            evidence.pop()

    @torch.inference_mode()
    def run(self, batch_ids: list[list[int]], adapted: bool) -> list[dict]:
        """Score and decode a batch of prompt ID lists (left-padded internally)."""
        model = self.model
        context = (
            nullcontext() if adapted or not hasattr(model, "disable_adapter") else model.disable_adapter()
        )
        size, width = len(batch_ids), max(len(ids) for ids in batch_ids)
        input_ids = torch.full((size, width), self.pad_id, dtype=torch.long)
        attention = torch.zeros((size, width), dtype=torch.long)
        for row, ids in enumerate(batch_ids):
            input_ids[row, width - len(ids) :] = torch.tensor(ids)
            attention[row, width - len(ids) :] = 1
        positions = (attention.cumsum(-1) - 1).clamp(min=0)
        input_ids, attention, positions = (t.to(self.device) for t in (input_ids, attention, positions))
        with context:
            synchronize(self.device)
            started = time.perf_counter()
            out = model(
                input_ids=input_ids,
                attention_mask=attention,
                position_ids=positions,
                use_cache=True,
                logits_to_keep=self.keep,
            )
            logits = out.logits.float()
            decision = logits[:, 0, [self.true_id, self.false_id]].softmax(-1)[:, 0]
            next_token = logits[:, -1].argmax(-1)
            na_prob = decision.tolist()
            synchronize(self.device)
            prefill_s = time.perf_counter() - started
            started = time.perf_counter()
            past = out.past_key_values
            generated: list[list[int]] = [[] for _ in range(size)]
            finished = [False] * size
            position = positions[:, -1:]
            for _ in range(self.max_new_tokens):
                for row, token in enumerate(next_token.tolist()):
                    if not finished[row]:
                        generated[row].append(token)
                        finished[row] = token in self.stop_ids
                if all(finished):
                    break
                attention = torch.cat([attention, attention.new_ones((size, 1))], dim=1)
                position = position + 1
                out = model(
                    input_ids=next_token[:, None],
                    attention_mask=attention,
                    position_ids=position,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
                next_token = out.logits[:, -1].argmax(-1)
            synchronize(self.device)
            decode_s = time.perf_counter() - started
        results = []
        for row in range(size):
            tokens = [t for t in generated[row] if t not in self.stop_ids]
            completion = self.tokenizer.decode(tokens, skip_special_tokens=True)
            results.append(
                {
                    "na_prob": na_prob[row],
                    "completion": completion,
                    "raw": ANSWER_PREFIX + completion,
                    "output_tokens": len(generated[row]),
                    "prefill_s": prefill_s / size,
                    "decode_s": decode_s / size,
                }
            )
        return results
