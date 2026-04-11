"""Label-smoothing cross-entropy loss with built-in padding mask."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with optional label smoothing and padding masking.

    Args:
        vocab_size:    Size of the output vocabulary.
        pad_id:        Token id to ignore (padding).
        smoothing:     Label smoothing epsilon (0.0 = standard CE).
        ignore_index:  Additional token id to ignore (used for MLM/causal LM -100 positions).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            logits:  (B, T, vocab_size) — raw model output
            targets: (B, T)             — ground-truth token ids; pad_id and ignore_index
                                          positions are excluded from the loss

        Returns:
            Scalar loss (mean over active positions).
        """
        B, T, V = logits.shape
        logits_flat = logits.contiguous().view(-1, V)      # (B*T, V)
        targets_flat = targets.contiguous().view(-1)        # (B*T,)

        log_probs = F.log_softmax(logits_flat, dim=-1)

        if self.smoothing == 0.0:
            # Map pad positions to ignore_index so nll_loss skips both in one pass
            targets_for_loss = targets_flat.clone()
            targets_for_loss[targets_for_loss == self.pad_id] = self.ignore_index
            loss = F.nll_loss(log_probs, targets_for_loss, ignore_index=self.ignore_index)
            return loss

        # Label smoothing: distribute (smoothing / V-1) to non-target classes
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (V - 1))
            # Clamp to valid range before scatter (ignore_index may be negative)
            safe_targets = targets_flat.clamp(min=0)
            smooth_dist.scatter_(1, safe_targets.unsqueeze(1), 1.0 - self.smoothing)
            # Zero-out ignored positions (padding and ignore_index)
            ignore_mask = targets_flat.eq(self.pad_id) | targets_flat.eq(self.ignore_index)
            smooth_dist[ignore_mask] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1)
        # Average only over active tokens
        n_tokens = (~ignore_mask).sum().clamp(min=1)
        return loss.sum() / n_tokens
