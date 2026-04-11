"""Causal Decoder: self-attention only (no cross-attention), used for decoder-only LM."""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention
from .ffn import PositionwiseFeedForward


class CausalDecoderLayer(nn.Module):
    """Single causal decoder layer: masked self-attn → add & norm → FFN → add & norm.

    Unlike the full DecoderLayer, there is no cross-attention sublayer — this is
    the architecture used by decoder-only models (GPT-style).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, is_causal=True)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        # Causal self-attention + residual
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        # FFN + residual
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class CausalDecoder(nn.Module):
    """Stack of N CausalDecoderLayers."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [CausalDecoderLayer(d_model, num_heads, d_ff, dropout, activation) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
