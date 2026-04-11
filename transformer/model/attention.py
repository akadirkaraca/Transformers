"""Multi-Head Attention with built-in causal masking."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention.

    Args:
        d_model:    Model dimensionality.
        num_heads:  Number of attention heads.
        dropout:    Attention weight dropout probability.
        is_causal:  If True, applies an auto-regressive (upper-triangular) mask.
                    Used for decoder self-attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        is_causal: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_causal = is_causal
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for layer in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(layer.weight)

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, d_model) → (B, h, T, d_k)"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, h, T, d_k) → (B, T, d_model)"""
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            query:            (B, Tq, d_model)
            key:              (B, Tk, d_model)
            value:            (B, Tk, d_model)
            key_padding_mask: (B, Tk) bool — True positions will be masked (-inf)

        Returns:
            (B, Tq, d_model)
        """
        Q = self._split_heads(self.W_q(query))  # (B, h, Tq, d_k)
        K = self._split_heads(self.W_k(key))    # (B, h, Tk, d_k)
        V = self._split_heads(self.W_v(value))  # (B, h, Tk, d_k)

        # Scaled dot-product scores: (B, h, Tq, Tk)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Causal mask (decoder self-attn): prevent attending to future tokens
        if self.is_causal:
            Tq, Tk = query.size(1), key.size(1)
            causal = torch.ones(Tq, Tk, dtype=torch.bool, device=query.device)
            causal = torch.triu(causal, diagonal=1)          # True above diagonal
            scores = scores.masked_fill(causal, float("-inf"))

        # Key padding mask: (B, Tk) → (B, 1, 1, Tk)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # (B, h, Tq, d_k)
        out = self._merge_heads(out)  # (B, Tq, d_model)
        return self.W_o(out)
