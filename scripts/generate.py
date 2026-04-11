"""Interactive and batch headline generation.

Interactive mode::

    uv run python scripts/generate.py --config configs/base.yaml --checkpoint checkpoints/checkpoint_epoch010.pt

Batch mode (reads articles from a text file, one per line)::

    uv run python scripts/generate.py --config configs/base.yaml \\
        --checkpoint checkpoints/checkpoint_epoch010.pt \\
        --input articles.txt --output headlines.txt
"""
import argparse
import sys

import torch

from transformer.config.config import load_config
from transformer.data.collator import PaddingCollator
from transformer.data.preprocessing import preprocess_article
from transformer.inference.beam_search import beam_search_decode
from transformer.model.transformer import build_transformer_from_config
from transformer.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


def generate_headline(
    model,
    tokenizer,
    article: str,
    cfg,
    device: torch.device,
) -> str:
    article = preprocess_article(article)
    ids = tokenizer.encode(article, add_special_tokens=True)
    ids = ids[: cfg.data.max_input_len]

    src = torch.tensor([ids], dtype=torch.long)
    src_mask = src.eq(tokenizer.pad_id)

    hyp_ids = beam_search_decode(
        model=model,
        src=src,
        src_mask=src_mask,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        beam_size=cfg.inference.beam_size,
        max_len=cfg.inference.max_decode_len,
        min_len=cfg.inference.min_decode_len,
        length_penalty=cfg.inference.length_penalty,
        device=device,
    )
    return tokenizer.decode(hyp_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default=None, help="Path to input file (one article per line)")
    parser.add_argument("--output", default=None, help="Path to output file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = SentencePieceTokenizer.load(cfg.tokenizer.model_path)

    model = build_transformer_from_config(cfg)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.input:
        # Batch mode
        with open(args.input, encoding="utf-8") as f:
            articles = [line.strip() for line in f if line.strip()]

        headlines = [generate_headline(model, tokenizer, a, cfg, device) for a in articles]

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for h in headlines:
                    f.write(h + "\n")
            print(f"Saved {len(headlines)} headlines to {args.output}")
        else:
            for h in headlines:
                print(h)
    else:
        # Interactive mode
        print("Enter article text (empty line to quit):")
        while True:
            try:
                article = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not article:
                break
            headline = generate_headline(model, tokenizer, article, cfg, device)
            print(f"Headline: {headline}\n")


if __name__ == "__main__":
    main()
