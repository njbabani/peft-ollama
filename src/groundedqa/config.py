"""Experiment configuration loaded from TOML (see configs/)."""

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    dataset_id: str = "rajpurkar/squad_v2"
    dataset_revision: str = "main"
    seed: int = 42
    # Training mix. Retrieval-miss examples are answerable questions whose source
    # passage is withheld (and no supplied passage contains the answer): target abstain.
    train_answerable: int = 1100
    train_unanswerable: int = 1000
    train_retrieval_miss: int = 300
    # Held-out questions, half answerable and half source-unanswerable.
    validation_questions: int = 400
    test_questions: int = 400
    # Subset of validation used for loss / decision-AUROC tracking during training.
    monitor_questions: int = 100
    # Answerable test questions used for the retrieval-only benchmark.
    retrieval_benchmark_questions: int = 2000


@dataclass
class RetrievalConfig:
    embedding_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision: str = "main"
    reranker_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_revision: str = "main"
    chunk_overlap_tokens: int = 32
    top_k: int = 3
    candidates: int = 30
    rrf_k: int = 60
    mode: str = "hybrid_rerank"  # dense | bm25 | hybrid | hybrid_rerank


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = "main"
    # bf16 is the stable choice for gradients. For inference, fp16 keeps the
    # no-answer probability within ~0.002 of fp32 on MPS, while bf16 can drift ~0.1.
    train_dtype: str = "bfloat16"
    eval_dtype: str = "float16"
    max_seq_length: int = 1536
    max_new_tokens: int = 64


@dataclass
class TrainConfig:
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    epochs: float = 1.0
    max_steps: int = 0  # >0 caps optimizer steps (smoke runs)
    batch_size: int = 1
    grad_accum: int = 8
    max_grad_norm: float = 1.0
    # Loss weight on the " true"/" false" decision token; other reply tokens weigh 1.
    decision_weight: float = 4.0
    eval_every: int = 50
    # Checkpoint (LoRA weights + optimizer/scheduler/RNG state) every N optimizer steps.
    save_every: int = 25
    keep_checkpoints: int = 2
    gradient_checkpointing: bool = False


@dataclass
class EvalConfig:
    batch_size: int = 8
    variants: list[str] = field(default_factory=lambda: ["base", "base_rag", "adapter", "adapter_rag"])
    bootstrap_samples: int = 2000


@dataclass
class Config:
    name: str = "experiment"
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "data": DataConfig,
    "retrieval": RetrievalConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build a Config, rejecting unknown sections or keys so typos fail loudly."""
    unknown = set(raw) - set(_SECTIONS) - {"name"}
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    sections = {}
    for name, cls in _SECTIONS.items():
        values = raw.get(name, {})
        allowed = {f.name for f in fields(cls)}
        bad = set(values) - allowed
        if bad:
            raise ValueError(f"Unknown keys in [{name}]: {sorted(bad)}")
        sections[name] = cls(**values)
    return Config(name=raw.get("name", "experiment"), **sections)


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as handle:
        return config_from_dict(tomllib.load(handle))
