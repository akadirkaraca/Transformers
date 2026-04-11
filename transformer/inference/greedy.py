"""Greedy decoding (batch-capable)."""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor

from ..model.transformer import Transformer


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: Tensor,
    src_mask: Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int = 64,
    device: torch.device = None,
) -> List[List[int]]:
    """Greedy decoding for a batch of source sequences.

    Args:
        model:    The Transformer model (in eval mode).
        src:      (B, S) source token ids.
        src_mask: (B, S) bool padding mask.
        bos_id:   Begin-of-sequence token id.
        eos_id:   End-of-sequence token id.
        max_len:  Maximum number of tokens to generate.
        device:   Target device.

    Returns:
        List of generated token id lists (one per example, without BOS/EOS).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    B = src.size(0)

    memory = model.encode(src, src_mask)              # (B, S, d_model)
    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_mask = ys.eq(model.pad_id)                 # (B, T)
        dec_out = model.decode(ys, memory, tgt_mask, src_mask)
        logits = model.output_proj(dec_out[:, -1, :])  # (B, vocab_size)
        next_token = logits.argmax(dim=-1)             # (B,)

        # Replace finished sequences with pad
        next_token = next_token.masked_fill(finished, model.pad_id)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

        finished = finished | next_token.eq(eos_id)
        if finished.all():
            break

    # Collect outputs, stripping BOS and everything after EOS
    results = []
    for i in range(B):
        ids = ys[i, 1:].tolist()  # strip BOS
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        results.append(ids)

    return results
