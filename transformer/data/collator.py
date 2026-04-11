"""Dynamic padding collator for (encoder_ids, decoder_ids) batches."""
from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


class PaddingCollator:
    """Collates variable-length (enc_ids, dec_ids) tuples into padded batches.

    Returns:
        src       : (B, S) — padded encoder input
        tgt_in    : (B, T) — decoder input (all tokens except last)
        tgt_out   : (B, T) — decoder target (all tokens except first = BOS)
        src_mask  : (B, S) bool — True where src is padding
        tgt_mask  : (B, T) bool — True where tgt_in is padding
    """

    def __init__(self, pad_id: int = 0):
        self.pad_id = pad_id

    def __call__(
        self, batch: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        enc_seqs, dec_seqs = zip(*batch)

        src = self._pad(list(enc_seqs))
        dec_padded = self._pad(list(dec_seqs))

        tgt_in = dec_padded[:, :-1]
        tgt_out = dec_padded[:, 1:]

        src_mask = src == self.pad_id        # (B, S)
        tgt_mask = tgt_in == self.pad_id     # (B, T)

        return src, tgt_in, tgt_out, src_mask, tgt_mask

    def _pad(self, seqs: List[Tensor]) -> Tensor:
        max_len = max(s.size(0) for s in seqs)
        out = torch.full(
            (len(seqs), max_len), self.pad_id, dtype=torch.long
        )
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out
