"""Causal LM collator for decoder-only autoregressive training.

Two operating modes selected automatically per sample:

Paired mode (output_col is non-empty, dec.numel() > 0):
    Concatenates input and output into one sequence:
        full = input_tokens + [BOS, out_1, ..., out_m, EOS]
    Loss is computed only on the output portion (article positions → -100).
    Use case: conditional generation (e.g. article → headline).

Single-column mode (output_col="", dec.numel() == 0):
    Treats the full input text as the causal sequence:
        full = enc_ids  (already has BOS..EOS)
    Loss is computed on all tokens (standard GPT pretraining).
    Use case: language model pretraining on raw text.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


class CausalCollator:
    """Collates (enc_ids, dec_ids) batches for causal language model training.

    Returns:
        input_ids: (B, T) — sequence to feed the model (shifted right)
        labels:    (B, T) — next-token targets (shifted left); -100 at ignored positions
        attn_mask: (B, T) bool — True where padding
    """

    IGNORE_INDEX = -100

    def __init__(self, pad_id: int, max_len: int):
        self.pad_id = pad_id
        self.max_len = max_len

    def __call__(
        self, batch: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        enc_seqs, dec_seqs = zip(*batch)

        input_list: List[Tensor] = []
        label_list: List[Tensor] = []

        for enc, dec in zip(enc_seqs, dec_seqs):
            if dec.numel() == 0:
                # Single-column mode: GPT-style causal LM on the full text.
                # enc = [BOS, t_1, ..., t_n, EOS]; loss on all tokens.
                full = enc
                n_masked = 0
            else:
                # Paired mode: conditional generation.
                # full = [inp_1..inp_n, BOS, out_1..out_m, EOS]
                # BOS from dec[0] serves as the separator token.
                input_tokens = enc[1:-1]   # strip BOS/EOS from input
                n_masked = input_tokens.size(0)
                full = torch.cat([input_tokens, dec], dim=0)

            # Truncate so input_ids and labels each fit within max_len tokens
            if full.size(0) > self.max_len + 1:
                full = full[: self.max_len + 1]
                n_masked = min(n_masked, full.size(0) - 1)

            input_ids_i = full[:-1]
            labels_i = full[1:].clone()

            # Mask input portion in labels (paired mode only) — loss on output tokens only
            mask_len = min(n_masked, labels_i.size(0))
            if mask_len > 0:
                labels_i[:mask_len] = self.IGNORE_INDEX

            input_list.append(input_ids_i)
            label_list.append(labels_i)

        # Pad to batch max length
        batch_max = max(t.size(0) for t in input_list)

        input_padded = torch.full((len(input_list), batch_max), self.pad_id, dtype=torch.long)
        labels_padded = torch.full((len(label_list), batch_max), self.IGNORE_INDEX, dtype=torch.long)

        for i, (inp, lab) in enumerate(zip(input_list, label_list)):
            L = inp.size(0)
            input_padded[i, :L] = inp
            labels_padded[i, :L] = lab

        attn_mask = input_padded.eq(self.pad_id)   # (B, T) True at padding

        return input_padded, labels_padded, attn_mask
