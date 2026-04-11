"""Encoder-only Transformer for masked language modeling (BERT-style)."""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

from .embedding import PositionalEncoding, TokenEmbedding
from .encoder import Encoder


class EncoderOnlyTransformer(nn.Module):
    """Encoder stack with an output projection — suitable for masked LM pretraining.

    The model reads a (possibly masked) token sequence and produces logits over
    the vocabulary at every position. Loss is then computed only at masked positions.

    Weight tying: output_proj.weight = embedding.weight (optional, saves parameters).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_id: int = 0,
        weight_tying: bool = True,
        encoder_activation: str = "relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = TokenEmbedding(vocab_size, d_model, pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout, encoder_activation)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        if weight_tying:
            self.output_proj.weight = self.embedding.weight

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            src:      (B, S) token ids (some replaced by mask_id for MLM)
            src_mask: (B, S) bool — True = padding position

        Returns:
            logits: (B, S, vocab_size)
        """
        x = self.pos_enc(self.embedding(src))
        x = self.encoder(x, src_mask)
        return self.output_proj(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_encoder_only_from_config(cfg) -> EncoderOnlyTransformer:
    """Construct an EncoderOnlyTransformer from a Config object."""
    from ..config.config import Config
    assert isinstance(cfg, Config)
    return EncoderOnlyTransformer(
        vocab_size=cfg.tokenizer.vocab_size,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_encoder_layers=cfg.model.num_encoder_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
        pad_id=cfg.tokenizer.pad_id,
        weight_tying=cfg.model.weight_tying,
        encoder_activation=cfg.model.encoder_activation,
    )
