"""Train a SentencePiece BPE tokenizer on any text dataset.

Supports the same data formats as the training pipeline:
    csv   — reads input_col and optionally output_col from a CSV file
    jsonl — reads input_col (and optionally output_col) fields from a JSONL file
    text  — reads every non-empty line from a plain text file

Usage::

    uv run python scripts/build_tokenizer.py --config configs/base.yaml
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from transformer.config.config import load_config
from transformer.data.preprocessing import preprocess_text
from transformer.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


def _load_texts(cfg) -> list:
    """Load all training texts from the configured data source."""
    d = cfg.data
    paired = d.output_col != ""

    if d.data_format == "csv":
        df = pd.read_csv(d.data_path)
        texts = df[d.input_col].fillna("").map(lambda t: preprocess_text(t, d.preprocessing)).tolist()
        if paired:
            texts += df[d.output_col].fillna("").map(lambda t: preprocess_text(t, d.preprocessing)).tolist()

    elif d.data_format == "jsonl":
        texts = []
        with open(d.data_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                texts.append(preprocess_text(str(obj.get(d.input_col, "")), d.preprocessing))
                if paired:
                    texts.append(preprocess_text(str(obj.get(d.output_col, "")), d.preprocessing))

    elif d.data_format == "text":
        with open(d.data_path, encoding="utf-8") as f:
            texts = [preprocess_text(line.rstrip("\n"), d.preprocessing) for line in f if line.strip()]

    else:
        raise ValueError(
            f"Unknown data_format: {d.data_format!r}. Choose from: 'csv', 'jsonl', 'text'."
        )

    return [t for t in texts if t]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Loading data from: {cfg.data.data_path}  (format={cfg.data.data_format})")
    texts = _load_texts(cfg)
    print(f"Training BPE on {len(texts):,} texts  (vocab_size={cfg.tokenizer.vocab_size})")

    output_path = Path(cfg.tokenizer.model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    SentencePieceTokenizer.train(
        texts=texts,
        vocab_size=cfg.tokenizer.vocab_size,
        model_type=cfg.tokenizer.model_type,
        character_coverage=cfg.tokenizer.character_coverage,
        pad_id=cfg.tokenizer.pad_id,
        unk_id=cfg.tokenizer.unk_id,
        bos_id=cfg.tokenizer.bos_id,
        eos_id=cfg.tokenizer.eos_id,
        output_model_path=str(output_path),
    )
    print(f"Tokenizer saved to: {output_path}")


if __name__ == "__main__":
    main()
