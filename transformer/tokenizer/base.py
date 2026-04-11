"""Abstract base class for all tokenizers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional


class TokenizerBase(ABC):
    """Common interface for SentencePiece and HuggingFace tokenizers."""

    # Special token ids — subclasses must populate these in __init__ / load
    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int
    vocab_size: int

    # ---- Encoding ----

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode a single string to a list of token ids."""

    def encode_batch(
        self, texts: List[str], add_special_tokens: bool = True
    ) -> List[List[int]]:
        return [self.encode(t, add_special_tokens) for t in texts]

    # ---- Decoding ----

    @abstractmethod
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token ids back to a string."""

    def decode_batch(
        self, batch: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        return [self.decode(ids, skip_special_tokens) for ids in batch]

    # ---- Persistence ----

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save tokenizer artefacts to *path* (directory or file)."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "TokenizerBase":
        """Load tokenizer artefacts from *path*."""

    # ---- Training ----

    @classmethod
    @abstractmethod
    def train(cls, texts: List[str], **kwargs) -> "TokenizerBase":
        """Train a tokenizer from a list of strings."""

    # ---- Utilities ----

    def __len__(self) -> int:
        return self.vocab_size
