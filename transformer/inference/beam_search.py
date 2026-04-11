"""Beam search decoding with length penalty."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ..model.transformer import Transformer


@torch.no_grad()
def beam_search_decode(
    model: Transformer,
    src: Tensor,
    src_mask: Tensor,
    bos_id: int,
    eos_id: int,
    beam_size: int = 4,
    max_len: int = 64,
    min_len: int = 3,
    length_penalty: float = 0.6,
    device: torch.device = None,
) -> List[List[int]]:
    """Beam search for a *single* source example.

    For batched generation, call this in a loop.

    Args:
        model:          The Transformer model (in eval mode).
        src:            (1, S) or (S,) source token ids.
        src_mask:       (1, S) or (S,) bool padding mask.
        bos_id:         Begin-of-sequence token id.
        eos_id:         End-of-sequence token id.
        beam_size:      Number of beams.
        max_len:        Maximum generation length.
        min_len:        Minimum generation length (suppress EOS before this).
        length_penalty: LP exponent α; score / len^α.
        device:         Target device.

    Returns:
        List of token ids for the best hypothesis (without BOS/EOS).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    if src.dim() == 1:
        src = src.unsqueeze(0)
        src_mask = src_mask.unsqueeze(0)

    # Encode once, then expand for all beams
    memory = model.encode(src, src_mask)               # (1, S, d_model)
    memory = memory.expand(beam_size, -1, -1)          # (B, S, d_model)
    src_mask = src_mask.expand(beam_size, -1)          # (B, S)

    # Each beam: (token_ids, cumulative_log_prob, finished)
    beams: List[Tuple[List[int], float, bool]] = [
        ([bos_id], 0.0, False) for _ in range(beam_size)
    ]
    completed: List[Tuple[float, List[int]]] = []

    for step in range(max_len):
        if all(b[2] for b in beams):
            break

        # Build decoder input from all beams
        max_t = max(len(b[0]) for b in beams)
        ys = torch.full((beam_size, max_t), model.pad_id, dtype=torch.long, device=device)
        for i, (ids, _, _) in enumerate(beams):
            ys[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

        tgt_mask = ys.eq(model.pad_id)
        dec_out = model.decode(ys, memory, tgt_mask, src_mask)

        # Get log-probs for the last valid token of each beam
        all_candidates: List[Tuple[float, List[int], bool]] = []

        for i, (ids, score, finished) in enumerate(beams):
            if finished:
                all_candidates.append((score, ids, True))
                continue

            pos = len(ids) - 1
            log_probs = F.log_softmax(model.output_proj(dec_out[i, pos]), dim=-1)

            # Suppress EOS before min_len
            if step < min_len:
                log_probs[eos_id] = float("-inf")

            top_lp, top_ids = log_probs.topk(beam_size)
            for lp, tok in zip(top_lp.tolist(), top_ids.tolist()):
                new_ids = ids + [tok]
                new_score = score + lp
                is_done = tok == eos_id
                all_candidates.append((new_score, new_ids, is_done))

        # Length-penalized ranking
        def lp_score(candidate):
            s, ids, done = candidate
            length = len(ids) - 1  # exclude BOS
            return s / ((5 + max(length, 1)) / 6) ** length_penalty

        all_candidates.sort(key=lp_score, reverse=True)

        beams = []
        for s, ids, done in all_candidates:
            if done:
                completed.append((lp_score((s, ids, done)), ids[1:]))  # strip BOS
            else:
                beams.append((ids, s, False))
            if len(beams) == beam_size:
                break

        # Pad beams list if we ran out of active hypotheses
        while len(beams) < beam_size:
            beams.append(([bos_id], float("-inf"), True))

    # Add remaining active beams to completed
    for ids, score, _ in beams:
        lp = score / ((5 + max(len(ids) - 1, 1)) / 6) ** length_penalty
        completed.append((lp, ids[1:]))

    if not completed:
        return []

    completed.sort(key=lambda x: x[0], reverse=True)
    best = completed[0][1]

    # Strip trailing EOS if present
    if best and best[-1] == eos_id:
        best = best[:-1]

    return best
