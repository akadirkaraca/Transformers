"""Position-wise Feed-Forward Network."""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_ACTIVATIONS = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
}


class PositionwiseFeedForward(nn.Module):
    """Two-layer FFN with configurable activation and dropout.

    Supported activations: relu (original paper), gelu (BERT/GPT-2), silu (LLaMA).
    """

    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Choose from: {list(_ACTIVATIONS)}"
            )
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self._act = _ACTIVATIONS[activation]
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self._act(self.fc1(x))))
