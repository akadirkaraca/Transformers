"""Full Seq2Seq Transformer with shared embeddings and weight tying."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .decoder import Decoder
from .embedding import PositionalEncoding, TokenEmbedding
from .encoder import Encoder


class Transformer(nn.Module):
    """Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Weight tying: the decoder output projection shares weights with the shared
    token embedding matrix, which halves the parameter count.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        pad_id: int = 0,
        weight_tying: bool = True,
        encoder_activation: str = "relu",
        decoder_activation: str = "relu",
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        # Shared embedding (encoder and decoder use the same weights)
        self.embedding = TokenEmbedding(vocab_size, d_model, pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout, encoder_activation)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, dropout, decoder_activation)

        # Output projection: (B, T, d_model) → (B, T, vocab_size)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        if weight_tying:
            # Tie output projection weights to the embedding matrix
            self.output_proj.weight = self.embedding.weight

    # ------------------------------------------------------------------ #
    #  Core Methods                                                        #
    # ------------------------------------------------------------------ #

    def encode(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        """Encode source tokens.

        Args:
            src:      (B, S) token ids
            src_mask: (B, S) bool — True = padding

        Returns:
            memory: (B, S, d_model)
        """
        x = self.pos_enc(self.embedding(src))
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """One decode step (or full sequence during training).

        Args:
            tgt:      (B, T) decoder input token ids
            memory:   (B, S, d_model) encoder output
            tgt_mask: (B, T) bool — True = padding
            src_mask: (B, S) bool — True = padding

        Returns:
            (B, T, d_model)
        """
        x = self.pos_enc(self.embedding(tgt))
        return self.decoder(x, memory, tgt_mask, src_mask)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Full forward pass for teacher-forced training.

        Returns:
            logits: (B, T, vocab_size)
        """
        memory = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, tgt_mask, src_mask)
        return self.output_proj(dec_out)

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_transformer_from_config(cfg) -> Transformer:
    """Construct a Transformer from a Config object."""
    from ..config.config import Config
    assert isinstance(cfg, Config)
    return Transformer(
        vocab_size=cfg.tokenizer.vocab_size,
        d_model=cfg.model.d_model,
        num_heads=cfg.model.num_heads,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_decoder_layers=cfg.model.num_decoder_layers,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        max_seq_len=cfg.model.max_seq_len,
        pad_id=cfg.tokenizer.pad_id,
        weight_tying=cfg.model.weight_tying,
        encoder_activation=cfg.model.encoder_activation,
        decoder_activation=cfg.model.decoder_activation,
    )
