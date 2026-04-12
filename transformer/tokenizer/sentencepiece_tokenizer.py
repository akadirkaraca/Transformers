"""SentencePiece BPE tokenizer (shared 32k vocab for Turkish news)."""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm

from .base import TokenizerBase


class SentencePieceTokenizer(TokenizerBase):
    """BPE tokenizer trained with SentencePiece.

    Special tokens (fixed ids):
        pad=0, unk=1, bos=2, eos=3
    """

    def __init__(self, model_path: str | Path):
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(model_path))
        self.pad_id = self._sp.pad_id() if self._sp.pad_id() >= 0 else 0
        self.unk_id = self._sp.unk_id()
        self.bos_id = self._sp.bos_id()
        self.eos_id = self._sp.eos_id()
        self.vocab_size = self._sp.get_piece_size()

    # ---- Encoding ----

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids: List[int] = self._sp.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    # ---- Decoding ----

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            special = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special]
        return self._sp.decode(ids)

    # ---- Vocabulary introspection ----

    def id_to_piece(self, token_id: int) -> str:
        """Return the string piece for a given token ID."""
        return self._sp.IdToPiece(token_id)

    def get_score(self, token_id: int) -> float:
        """Return the log-probability score assigned to a piece during BPE training."""
        return self._sp.GetScore(token_id)

    # ---- Persistence ----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # SentencePiece model is a single .model file; copy it to path
        model_proto = self._sp.serialized_model_proto()
        with open(path, "wb") as f:
            f.write(model_proto)

    @classmethod
    def load(cls, path: str | Path) -> "SentencePieceTokenizer":
        return cls(model_path=path)

    # ---- Training ----

    @classmethod
    def train(
        cls,
        texts: List[str],
        vocab_size: int = 32000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
        output_model_path: str = "bpe32k.model",
        num_threads: int = 4,
        input_sentence_size: int = 2_000_000,
        **kwargs,
    ) -> "SentencePieceTokenizer":
        """Train BPE model from a list of text strings."""
        # Write texts to a temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
            for line in texts:
                line = line.strip()
                if line:
                    tmp.write(line + "\n")

        output_prefix = str(output_model_path).replace(".model", "")
        try:
            spm.SentencePieceTrainer.train(
                input=tmp_path,
                model_prefix=output_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                pad_id=pad_id,
                unk_id=unk_id,
                bos_id=bos_id,
                eos_id=eos_id,
                num_threads=num_threads,
                input_sentence_size=input_sentence_size,
                shuffle_input_sentence=True,
                **kwargs,
            )
        finally:
            os.unlink(tmp_path)

        return cls(model_path=output_prefix + ".model")
