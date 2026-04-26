from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    d_model: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 512
    weight_tying: bool = True
    encoder_activation: str = "relu"   # FFN activation in encoder: "relu" | "gelu" | "silu"
    decoder_activation: str = "relu"   # FFN activation in decoder: "relu" | "gelu" | "silu"
    architecture: str = "encoder_decoder"  # "encoder_decoder" | "encoder_only" | "decoder_only"
    attn_impl: str = "manual"          # attention backend: "manual" | "sdpa"


@dataclass
class TokenizerConfig:
    vocab_size: int = 32000
    model_type: str = "bpe"
    character_coverage: float = 0.9995
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    model_path: str = "tokenizer/bpe32k.model"


@dataclass
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1.0
    warmup_steps: int = 4000
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    amp: bool = True
    save_top_k: int = 3
    checkpoint_dir: str = "checkpoints"
    log_every: int = 100
    eval_every: int = 1
    seed: int = 42
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.001
    optimizer: str = "adam"    # "adam" | "adamw"
    weight_decay: float = 0.0
    run_name: str = ""         # auto-generated from config if empty
    mlm_probability: float = 0.15   # fraction of tokens masked (encoder_only)
    mask_id: int = 1                # token id used as [MASK]; defaults to unk_id
    torch_compile: str = "eager"    # "eager" | "graph"


@dataclass
class DataConfig:
    data_path: str = "data/dataset.csv"
    data_format: str = "csv"       # "csv" | "jsonl" | "text"
    input_col: str = "input"       # column/field for input text
    output_col: str = "output"     # column/field for output text; "" = single-column mode
    preprocessing: str = "none"    # "none" | "basic"
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    max_input_len: int = 512       # max input sequence length (tokens)
    max_output_len: int = 128      # max output sequence length (tokens); encoder_decoder only
    num_workers: int = 4


@dataclass
class InferenceConfig:
    beam_size: int = 4
    length_penalty: float = 0.6
    max_decode_len: int = 64
    min_decode_len: int = 3
    checkpoint_path: str = ""
    torch_compile: str = "eager"    # "eager" | "graph"


@dataclass
class CheckpointConfig:
    formats: list = field(default_factory=lambda: ["ckpt", "safetensors"])
    # Formats saved at each training checkpoint:
    #   "ckpt"        - full training state (model + optimizer + scheduler + metadata), resumable
    #   "safetensors" - model weights only (no pickle, memory-mapped, for inference/sharing)
    #   "pt"          - legacy PyTorch format (same content as ckpt)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def _merge(dataclass_instance, d: dict):
    """Recursively merge dict d into a dataclass instance."""
    for key, value in d.items():
        if not hasattr(dataclass_instance, key):
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(dataclass_instance, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dataclass_instance, key, value)


def load_config(yaml_path: str | Path) -> Config:
    cfg = Config()
    with open(yaml_path) as f:
        d = yaml.safe_load(f) or {}
    _merge(cfg, d)
    return cfg
