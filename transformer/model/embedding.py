"""Token embedding + sinusoidal positional encoding."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Embedding):
    """Standard embedding scaled by sqrt(d_model) per the paper."""

    def __init__(self, vocab_size: int, d_model: int, pad_id: int = 0):
        super().__init__(vocab_size, d_model, padding_idx=pad_id)
        self.d_model = d_model
        nn.init.normal_(self.weight, mean=0.0, std=d_model ** -0.5)
        # Zero-out the padding embedding
        with torch.no_grad():
            self.weight[pad_id].zero_()

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding registered as a buffer."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
