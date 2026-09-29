"""Decoder and loss on a tiny random Qwen2 with the real (cached) tokenizer; CPU, float32."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from groundedqa.inference import AnswerDecoder  # noqa: E402
from groundedqa.prompting import ANSWER_PREFIX  # noqa: E402
from groundedqa.train import weighted_reply_loss  # noqa: E402
from groundedqa.training_data import collate, decision_token_ids, encode_example  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except OSError:
        pytest.skip("Qwen tokenizer not in the local Hugging Face cache")


@pytest.fixture(scope="module")
def tiny_model(tokenizer):
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
    )
    return transformers.Qwen2ForCausalLM(config).eval()


EVIDENCE = [{"id": "c1", "text": "Hamlet was written by William Shakespeare."}]


def test_batched_decoding_matches_single(tokenizer, tiny_model):
    decoder = AnswerDecoder(tokenizer, tiny_model, torch.device("cpu"), max_new_tokens=6, max_seq_length=1024)
    prompts = [
        decoder.prompt_ids("Who wrote Hamlet?", EVIDENCE)[1],
        decoder.prompt_ids("A much longer question about the author of the play Hamlet?", EVIDENCE * 2)[1],
        decoder.prompt_ids("No evidence here?", [])[1],
    ]
    batched = decoder.run(prompts, adapted=True)
    single = [decoder.run([p], adapted=True)[0] for p in prompts]
    for b, s in zip(batched, single):
        assert b["completion"] == s["completion"]
        assert b["na_prob"] == pytest.approx(s["na_prob"], abs=1e-5)
        assert 0.0 < b["na_prob"] < 1.0
        assert b["raw"].startswith(ANSWER_PREFIX) and b["output_tokens"] <= 6


def test_prompt_fitting_drops_whole_passages(tokenizer, tiny_model):
    decoder = AnswerDecoder(tokenizer, tiny_model, torch.device("cpu"), max_new_tokens=8, max_seq_length=260)
    long = [{"id": f"p{i}", "text": "word " * 60} for i in range(3)]
    kept, ids = decoder.prompt_ids("Q?", long)
    assert len(kept) < 3 and kept == long[: len(kept)]
    assert len(ids) + 8 <= 260
    with pytest.raises(ValueError):
        AnswerDecoder(tokenizer, tiny_model, torch.device("cpu"), 8, 20).prompt_ids("Q?", [])


def test_weighted_loss_matches_manual_computation(tokenizer, tiny_model):
    decision = decision_token_ids(tokenizer)
    examples = [
        {
            "question": "Who wrote Hamlet?",
            "evidence": EVIDENCE,
            "target": ANSWER_PREFIX + 'Shakespeare", "citations": ["E1"]}',
        },
        {
            "question": "When?",
            "evidence": EVIDENCE,
            "target": '{"abstain": true, "answer": "", "citations": []}',
        },
    ]
    items = [encode_example(tokenizer, e, 1024, decision, 4.0) for e in examples]
    batch = collate(items, tokenizer.pad_token_id)
    loss, p_abstain = weighted_reply_loss(tiny_model, batch, decision[1], decision[2])

    logits = tiny_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits.float()
    ce = torch.nn.functional.cross_entropy(
        logits[:, :-1].transpose(1, 2), batch["labels"][:, 1:], ignore_index=-100, reduction="none"
    )
    weights = batch["weights"][:, 1:]
    assert loss.item() == pytest.approx(((ce * weights).sum() / weights.sum()).item(), rel=1e-4)
    for row, item in enumerate(items):
        pair = logits[row, item["decision_position"] - 1, [decision[1], decision[2]]].softmax(-1)
        assert p_abstain[row].item() == pytest.approx(pair[0].item(), rel=1e-4)
    assert batch["abstain"].tolist() == [False, True]
