"""Decoder-only Transformer for autoregressive causal language modeling (GPT-style)."""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

from .causal_decoder import CausalDecoder
from .embedding import PositionalEncoding, TokenEmbedding


class DecoderOnlyTransformer(nn.Module):
    """Causal decoder stack with an output projection — suitable for autoregressive LM.

    The model attends only to past tokens (causal masking is handled internally by
    CausalDecoderLayer via MultiHeadAttention(is_causal=True)). There is no encoder
    and no cross-attention sublayer.

    For headline generation training: the article and title are concatenated into one
    sequence; loss is computed only on the title portion.

    Weight tying: output_proj.weight = embedding.weight (optional, saves parameters).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_id: int = 0,
        weight_tying: bool = True,
        decoder_activation: str = "relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.embedding = TokenEmbedding(vocab_size, d_model, pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)
        self.decoder = CausalDecoder(num_decoder_layers, d_model, num_heads, d_ff, dropout, decoder_activation)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        if weight_tying:
            self.output_proj.weight = self.embedding.weight

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            src:      (B, T) token ids
            src_mask: (B, T) bool — True = padding position

        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.pos_enc(self.embedding(src))
        x = self.decoder(x, src_mask)
        return self.output_proj(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_decoder_only_from_config(cfg) -> DecoderOnlyTransformer:
    """Construct a DecoderOnlyTransformer from a Config object."""
    from ..config.config import Config
    assert isinstance(cfg, Config)
    return DecoderOnlyTransformer(
        vocab_size=cfg.tokenizer.vocab_size,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_decoder_layers=cfg.model.num_decoder_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
        pad_id=cfg.tokenizer.pad_id,
        weight_tying=cfg.model.weight_tying,
        decoder_activation=cfg.model.decoder_activation,
    )
