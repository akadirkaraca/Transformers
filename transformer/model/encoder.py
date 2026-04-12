"""Transformer Encoder (single layer + N-layer stack)."""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention
from .ffn import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Single encoder layer: self-attn → add & norm → FFN → add & norm."""

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,
        activation: str = "relu", attn_impl: str = "manual",
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, is_causal=False, attn_impl=attn_impl)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        # Self-attention + residual
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        # FFN + residual
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    """Stack of N EncoderLayers."""

    def __init__(
        self, num_layers: int, d_model: int, num_heads: int, d_ff: int,
        dropout: float = 0.1, activation: str = "relu", attn_impl: str = "manual",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout, activation, attn_impl) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
