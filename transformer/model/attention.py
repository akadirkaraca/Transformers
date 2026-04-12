"""Multi-Head Attention with built-in causal masking."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_VALID_ATTN_IMPL = {"manual", "sdpa"}


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention.

    Args:
        d_model:    Model dimensionality.
        num_heads:  Number of attention heads.
        dropout:    Attention weight dropout probability.
        is_causal:  If True, applies an auto-regressive (upper-triangular) mask.
                    Used for decoder self-attention.
        attn_impl:  Attention backend. "manual" uses explicit matmul + softmax.
                    "sdpa" delegates to torch.nn.functional.scaled_dot_product_attention,
                    which selects the best available kernel (Flash Attention on Ampere+,
                    memory-efficient attention on older GPUs, math fallback elsewhere).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        is_causal: bool = False,
        attn_impl: str = "manual",
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        assert attn_impl in _VALID_ATTN_IMPL, (
            f"attn_impl must be one of {_VALID_ATTN_IMPL}, got {attn_impl!r}"
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_causal = is_causal
        self.attn_impl = attn_impl
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

        if self.attn_impl == "sdpa":
            out = self._sdpa(Q, K, V, key_padding_mask)
        else:
            out = self._manual_attn(Q, K, V, key_padding_mask)

        out = self._merge_heads(out)  # (B, Tq, d_model)
        return self.W_o(out)

    def _manual_attn(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        """Explicit matmul + softmax attention. (B, h, Tq, d_k)"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, h, Tq, Tk)

        if self.is_causal:
            Tq, Tk = Q.size(2), K.size(2)
            causal = torch.triu(
                torch.ones(Tq, Tk, dtype=torch.bool, device=Q.device), diagonal=1
            )
            scores = scores.masked_fill(causal, float("-inf"))

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)  # (B, h, Tq, d_k)

    def _sdpa(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        """torch.nn.functional.scaled_dot_product_attention backend.

        Fast path (Flash Attention eligible on Ampere+):
          causal self-attention with no padding mask — passes is_causal=True directly
          so PyTorch can select the optimal kernel without an explicit attn_mask.

        Fallback (Memory-Efficient Attention on Volta+, math elsewhere):
          causal mask and/or padding mask are merged into a single additive float
          tensor so both constraints coexist without conflicting SDPA flags.
        """
        if self.is_causal and key_padding_mask is None:
            # Flash Attention eligible: let PyTorch pick the best kernel.
            return F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True,
            )

        # Fallback: build combined additive mask, pass is_causal=False to avoid conflicts.
        attn_mask = None

        if self.is_causal:
            Tq, Tk = Q.size(2), K.size(2)
            attn_mask = torch.triu(
                torch.full((Tq, Tk), float("-inf"), device=Q.device, dtype=Q.dtype),
                diagonal=1,
            )  # (Tq, Tk) — broadcasts over batch/heads

        if key_padding_mask is not None:
            pad = torch.zeros(
                Q.size(0), 1, 1, K.size(2), device=Q.device, dtype=Q.dtype
            ).masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            # (Tq, Tk) + (B, 1, 1, Tk) → (B, 1, Tq, Tk) via broadcast
            attn_mask = attn_mask + pad if attn_mask is not None else pad

        return F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )  # (B, h, Tq, d_k)
