"""HuggingFace tokenizer wrapper (alternative to SentencePiece)."""
from __future__ import annotations

from pathlib import Path
from typing import List

from .base import TokenizerBase


class HuggingFaceTokenizer(TokenizerBase):
    """Thin wrapper around any HuggingFace PreTrainedTokenizerFast.

    Usage::

        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        tok = HuggingFaceTokenizer(hf_tok)
    """

    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer
        self.pad_id = hf_tokenizer.pad_token_id or 0
        self.unk_id = hf_tokenizer.unk_token_id or 1
        self.bos_id = hf_tokenizer.bos_token_id or hf_tokenizer.cls_token_id or 2
        self.eos_id = hf_tokenizer.eos_token_id or hf_tokenizer.sep_token_id or 3
        self.vocab_size = hf_tokenizer.vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self._tok.save_pretrained(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "HuggingFaceTokenizer":
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(str(path))
        return cls(hf_tok)

    @classmethod
    def train(cls, texts: List[str], **kwargs) -> "HuggingFaceTokenizer":
        raise NotImplementedError(
            "Use SentencePieceTokenizer.train() or train a HF tokenizer externally."
        )
