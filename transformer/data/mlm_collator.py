"""MLM collator for encoder-only (BERT-style) masked language model training."""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor


class MLMCollator:
    """Collates (enc_ids, dec_ids) batches for masked language model training.

    Uses only enc_ids (articles); dec_ids are ignored. Applies BERT-style masking:
        - 80% of selected tokens → replaced with mask_id
        - 10% of selected tokens → replaced with a random vocabulary token
        - 10% of selected tokens → kept as original

    Returns:
        masked_input: (B, S) — token ids after masking
        labels:       (B, S) — original token ids at masked positions, -100 elsewhere
        attn_mask:    (B, S) bool — True where padding
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        pad_id: int,
        mask_id: int,
        vocab_size: int,
        mlm_probability: float = 0.15,
        seed: Optional[int] = None,
    ):
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def __call__(
        self, batch: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        enc_seqs, _ = zip(*batch)

        # Pad encoder sequences to batch max length
        input_ids = self._pad(list(enc_seqs))            # (B, S)

        # Attention mask: True at padding positions (computed before masking)
        attn_mask = input_ids.eq(self.pad_id)            # (B, S)

        # Labels: start as full copy; non-masked positions will be set to -100
        labels = input_ids.clone()

        # Candidate mask: non-padding positions selected by mlm_probability
        prob_matrix = torch.rand(input_ids.shape, generator=self.rng)
        candidate_mask = (prob_matrix < self.mlm_probability) & (~attn_mask)

        # Non-candidates get ignore_index in labels
        labels[~candidate_mask] = self.IGNORE_INDEX

        # 80/10/10 split for candidate positions
        rand2 = torch.rand(input_ids.shape, generator=self.rng)

        # 80%: replace with mask_id
        replace_with_mask = candidate_mask & (rand2 < 0.8)
        input_ids[replace_with_mask] = self.mask_id

        # 10%: replace with a random token
        replace_with_random = candidate_mask & (rand2 >= 0.8) & (rand2 < 0.9)
        random_tokens = torch.randint(
            0, self.vocab_size, input_ids.shape, generator=self.rng
        )
        input_ids[replace_with_random] = random_tokens[replace_with_random]

        # 10%: keep original — no action needed

        return input_ids, labels, attn_mask

    def _pad(self, seqs: List[Tensor]) -> Tensor:
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), self.pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out
