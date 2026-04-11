"""Generic text dataset and DataLoader builder for LLM training.

Supports three data formats:
    "csv"   — CSV file with configurable input_col / output_col columns
    "jsonl" — JSON Lines file with configurable field names
    "text"  — Plain text file; each non-empty line is one training sample

Single-column vs paired mode:
    output_col=""  → single-column mode; dec_ids will be an empty tensor.
                     Use with encoder_only (MLM) or decoder_only (causal LM).
    output_col!="" → paired mode; both enc_ids and dec_ids are populated.
                     Use with encoder_decoder (seq2seq).
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from ..tokenizer.base import TokenizerBase
from .preprocessing import preprocess_text


class TextDataset(Dataset):
    """Maps text samples to tokenized (enc_ids, dec_ids) tensor pairs.

    When texts_b is None (single-column mode), dec_ids is an empty tensor.
    Collators detect this and switch to the appropriate single-sequence mode.
    """

    def __init__(
        self,
        texts_a: List[str],
        texts_b: Optional[List[str]],
        tokenizer: TokenizerBase,
        max_len_a: int = 512,
        max_len_b: int = 128,
    ):
        assert texts_b is None or len(texts_a) == len(texts_b)
        self.texts_a = texts_a
        self.texts_b = texts_b
        self.tokenizer = tokenizer
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b

    def __len__(self) -> int:
        return len(self.texts_a)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_ids = self.tokenizer.encode(self.texts_a[idx], add_special_tokens=True)
        enc_ids = enc_ids[: self.max_len_a]

        if self.texts_b is not None:
            dec_ids = self.tokenizer.encode(self.texts_b[idx], add_special_tokens=True)
            dec_ids = dec_ids[: self.max_len_b]
            return torch.tensor(enc_ids, dtype=torch.long), torch.tensor(dec_ids, dtype=torch.long)
        else:
            return torch.tensor(enc_ids, dtype=torch.long), torch.tensor([], dtype=torch.long)


# Backward-compatibility alias
NewsHeadlineDataset = TextDataset


def _load_raw_texts(
    data_path: str,
    data_format: str,
    input_col: str,
    output_col: str,
    preprocessing: str,
) -> Tuple[List[str], Optional[List[str]]]:
    """Load raw text lists from the data file.

    Returns:
        (texts_a, texts_b) where texts_b is None when output_col is empty.
    """
    paired = output_col != ""

    if data_format == "csv":
        df = pd.read_csv(data_path)
        texts_a = df[input_col].fillna("").map(lambda t: preprocess_text(t, preprocessing)).tolist()
        texts_b = (
            df[output_col].fillna("").map(lambda t: preprocess_text(t, preprocessing)).tolist()
            if paired else None
        )

    elif data_format == "jsonl":
        texts_a, texts_b_list = [], []
        with open(data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                texts_a.append(preprocess_text(str(obj.get(input_col, "")), preprocessing))
                if paired:
                    texts_b_list.append(preprocess_text(str(obj.get(output_col, "")), preprocessing))
        texts_b = texts_b_list if paired else None

    elif data_format == "text":
        with open(data_path, encoding="utf-8") as f:
            texts_a = [preprocess_text(line.rstrip("\n"), preprocessing) for line in f if line.strip()]
        texts_b = None

    else:
        raise ValueError(
            f"Unknown data_format: {data_format!r}. Choose from: 'csv', 'jsonl', 'text'."
        )

    return texts_a, texts_b


def build_dataloaders(
    data_path: str,
    tokenizer: TokenizerBase,
    data_format: str = "csv",
    input_col: str = "input",
    output_col: str = "output",
    preprocessing: str = "none",
    max_input_len: int = 512,
    max_output_len: int = 128,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    collate_fn=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load data, split into train/val/test, and return DataLoaders.

    Args:
        data_path:     Path to the data file.
        tokenizer:     Tokenizer instance.
        data_format:   "csv", "jsonl", or "text".
        input_col:     Column/field name for input text.
        output_col:    Column/field name for output text; "" = single-column mode.
        preprocessing: Text preprocessing level: "none" or "basic".
        max_input_len: Max tokens for input sequences.
        max_output_len: Max tokens for output sequences (encoder_decoder only).
        train_ratio:   Fraction of data for training.
        val_ratio:     Fraction of data for validation.
        batch_size:    Batch size for all DataLoaders.
        num_workers:   DataLoader worker processes.
        seed:          Random seed for the train/val/test split.
        collate_fn:    Custom collate function; defaults to PaddingCollator.

    Returns:
        (train_dl, val_dl, test_dl)
    """
    texts_a, texts_b = _load_raw_texts(data_path, data_format, input_col, output_col, preprocessing)

    full_ds = TextDataset(texts_a, texts_b, tokenizer, max_input_len, max_output_len)

    n = len(full_ds)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=generator
    )

    if collate_fn is None:
        from .collator import PaddingCollator
        collate_fn = PaddingCollator(pad_id=tokenizer.pad_id)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    return train_dl, val_dl, test_dl
