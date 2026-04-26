"""Headline generation for decoder-only (autoregressive) models.

The article is fed as a prefix, followed by a BOS separator token that signals
the model to begin generating the title. Generation stops at the first EOS token
or when max_decode_len is reached.

Interactive mode::

    uv run python scripts/causal_generate.py --config configs/base.yaml \\
        --checkpoint checkpoints/checkpoint_epoch010.pt

Batch mode (one article per line)::

    uv run python scripts/causal_generate.py --config configs/base.yaml \\
        --checkpoint checkpoints/checkpoint_epoch010.pt \\
        --input articles.txt --output headlines.txt
"""
import argparse

import torch

from transformer.config.config import load_config
from transformer.data.preprocessing import preprocess_article
from transformer.model.decoder_only import build_decoder_only_from_config
from transformer.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from transformer.training.trainer import load_model_weights


def generate_headline(
    model,
    tokenizer,
    article: str,
    cfg,
    device: torch.device,
) -> str:
    """Generate a headline for a single article using greedy decoding.

    The prompt is:  article_tokens + [BOS]
    Generation continues until EOS or max_decode_len tokens are produced.
    """
    article = preprocess_article(article)
    ids = tokenizer.encode(article, add_special_tokens=True)  # [BOS, art_1..art_n, EOS]
    ids = ids[: cfg.data.max_input_len]                        # cap same as training
    article_tokens = ids[1:-1]                                 # strip BOS and EOS

    # Prompt: article content + BOS as the separator before the title
    prompt_ids = article_tokens + [tokenizer.bos_id]
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)
    prompt_len = generated.size(1)

    with torch.no_grad():
        for _ in range(cfg.inference.max_decode_len):
            src_mask = generated.eq(tokenizer.pad_id)          # (1, T) — no padding in prompt
            logits = model(generated, src_mask)                 # (1, T, vocab_size)
            next_token = logits[0, -1, :].argmax().item()
            if next_token == tokenizer.eos_id:
                break
            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
            generated = torch.cat([generated, next_tensor], dim=1)

    title_ids = generated[0, prompt_len:].tolist()
    return tokenizer.decode(title_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default=None, help="Path to input file (one article per line)")
    parser.add_argument("--output", default=None, help="Path to output file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.model.architecture != "decoder_only":
        raise ValueError(
            f"This script requires architecture=decoder_only, "
            f"but base.yaml has {cfg.model.architecture!r}. "
            "Use scripts/generate.py for encoder_decoder models."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = SentencePieceTokenizer.load(cfg.tokenizer.model_path)

    model = build_decoder_only_from_config(cfg)
    state = load_model_weights(args.checkpoint, model, device)
    # For .ckpt/.pt checkpoints, validate the saved architecture matches.
    # .safetensors files carry weights only; trust the config check above.
    if state is not None:
        saved_arch = state.get("architecture", "decoder_only")
        if saved_arch != "decoder_only":
            raise ValueError(
                f"Checkpoint was saved with architecture={saved_arch!r}, not decoder_only."
            )
    model.to(device).eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.input:
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
