"""BLEU/ROUGE evaluation on the test split (encoder_decoder only).

Usage::

    uv run python scripts/evaluate.py --config configs/base.yaml --checkpoint checkpoints/checkpoint_epoch010.pt
"""
import argparse

import torch
from tqdm import tqdm

from transformer.config.config import load_config
from transformer.data.dataset import build_dataloaders
from transformer.evaluation.metrics import compute_bleu, compute_rouge
from transformer.inference.beam_search import beam_search_decode
from transformer.model import build_model
from transformer.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from transformer.training.trainer import load_model_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.model.architecture != "encoder_decoder":
        raise ValueError(
            "evaluate.py uses beam search and is designed for encoder_decoder models. "
            f"Got architecture={cfg.model.architecture!r}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = SentencePieceTokenizer.load(cfg.tokenizer.model_path)
    _, val_dl, test_dl = build_dataloaders(
        data_path=cfg.data.data_path,
        tokenizer=tokenizer,
        data_format=cfg.data.data_format,
        input_col=cfg.data.input_col,
        output_col=cfg.data.output_col,
        preprocessing=cfg.data.preprocessing,
        max_input_len=cfg.data.max_input_len,
        max_output_len=cfg.data.max_output_len,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        batch_size=1,   # beam search is per-example
        num_workers=0,
        seed=cfg.training.seed,
    )
    dl = test_dl if args.split == "test" else val_dl

    model = build_model(cfg.model.architecture, cfg)
    load_model_weights(args.checkpoint, model, device)
    model.to(device).eval()
    if cfg.inference.torch_compile == "graph":
        model = torch.compile(model)

    hypotheses, references = [], []

    for src, tgt_in, tgt_out, src_mask, _ in tqdm(dl, desc="Evaluating"):
        ref_ids = tgt_out[0].tolist()
        if tokenizer.eos_id in ref_ids:
            ref_ids = ref_ids[: ref_ids.index(tokenizer.eos_id)]
        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

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
        hyp_text = tokenizer.decode(hyp_ids, skip_special_tokens=True)

        hypotheses.append(hyp_text)
        references.append(ref_text)

    bleu = compute_bleu(hypotheses, references)
    rouge = compute_rouge(hypotheses, references)

    print("\n=== Evaluation Results ===")
    for k, v in bleu.items():
        print(f"  {k.upper()}: {v:.4f}")
    for k, v in rouge.items():
        print(f"  {k.upper()}: {v:.4f}")


if __name__ == "__main__":
    main()
