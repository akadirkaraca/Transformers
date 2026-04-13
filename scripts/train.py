"""Training entry point.

Usage::

    uv run python scripts/train.py --config configs/base.yaml
    uv run python scripts/train.py --config configs/base.yaml --resume checkpoints/checkpoint_epoch005.pt

Custom architecture::

    # my_model.py registers itself via register_architecture() on import
    uv run python scripts/train.py --config configs/base.yaml --arch-module my_model.py

Telegram notifications — set credentials in .env or environment:
    TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
    TELEGRAM_CHAT_ID=123456789
"""
import argparse
import importlib.util
import random
import os

import numpy as np
import torch
from dotenv import load_dotenv
from rich.console import Console

from transformer.config.config import load_config
from transformer.data.causal_collator import CausalCollator
from transformer.data.dataset import build_dataloaders
from transformer.data.mlm_collator import MLMCollator
from transformer.model import build_model  # uses the architecture registry
from transformer.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from transformer.training.notifier import TelegramNotifier
from transformer.training.trainer import Trainer

_console = Console()


def _make_run_name(cfg) -> str:
    """Auto-generate a human-readable run identifier from key config values."""
    m, t = cfg.model, cfg.training
    arch = m.architecture
    arch_prefix = {"encoder_decoder": "encdec", "encoder_only": "enconly", "decoder_only": "deconly"}.get(arch, arch)
    n_layers = m.num_decoder_layers if arch == "decoder_only" else m.num_encoder_layers
    name = f"{arch_prefix}_d{m.d_model}_L{n_layers}_{t.optimizer}_do{int(m.dropout * 100):02d}"
    if t.weight_decay > 0:
        name += f"_wd{str(t.weight_decay).lstrip('0').replace('.', '')}"
    enc, dec = m.encoder_activation, m.decoder_activation
    if enc == dec and enc != "relu":
        name += f"_{enc}"
    elif enc != dec:
        name += f"_enc{enc}_dec{dec}"
    return name


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_arch_module(path: str) -> None:
    """Import a Python file to trigger register_architecture() side effects."""
    spec = importlib.util.spec_from_file_location("_custom_arch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--arch-module", default=None, metavar="PATH",
        help="Python file to import before training (use to register custom architectures)"
    )
    args = parser.parse_args()

    if args.arch_module:
        _load_arch_module(args.arch_module)

    cfg = load_config(args.config)
    set_seed(cfg.training.seed)
    arch = cfg.model.architecture

    # Validate positional encoding capacity — must hold for all architectures.
    # The same PE buffer is shared by encoder and decoder in encoder_decoder models.
    if arch == "decoder_only":
        combined_len = cfg.data.max_input_len + cfg.data.max_output_len
        if combined_len > cfg.model.max_seq_len:
            raise ValueError(
                f"decoder_only: max_input_len ({cfg.data.max_input_len}) + "
                f"max_output_len ({cfg.data.max_output_len}) = {combined_len} "
                f"exceeds model.max_seq_len ({cfg.model.max_seq_len}). "
                "Increase model.max_seq_len in base.yaml."
            )
    else:
        if cfg.data.max_input_len > cfg.model.max_seq_len:
            raise ValueError(
                f"{arch}: data.max_input_len ({cfg.data.max_input_len}) exceeds "
                f"model.max_seq_len ({cfg.model.max_seq_len}). "
                "Set model.max_seq_len >= data.max_input_len in base.yaml."
            )
        if cfg.data.max_output_len > cfg.model.max_seq_len:
            raise ValueError(
                f"{arch}: data.max_output_len ({cfg.data.max_output_len}) exceeds "
                f"model.max_seq_len ({cfg.model.max_seq_len}). "
                "Set model.max_seq_len >= data.max_output_len in base.yaml."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _console.print(f"[dim]device      :[/] [cyan]{device}[/]")

    with _console.status("[dim]tokenizer   :[/] loading..."):
        tokenizer = SentencePieceTokenizer.load(cfg.tokenizer.model_path)
    _console.print(f"[dim]tokenizer   :[/] vocab_size={tokenizer.vocab_size}")

    # Select collator based on architecture
    if arch == "encoder_only":
        collate_fn = MLMCollator(
            pad_id=cfg.tokenizer.pad_id,
            mask_id=cfg.training.mask_id,
            vocab_size=cfg.tokenizer.vocab_size,
            mlm_probability=cfg.training.mlm_probability,
            seed=cfg.training.seed,
        )
    elif arch == "decoder_only":
        collate_fn = CausalCollator(
            pad_id=cfg.tokenizer.pad_id,
            max_len=cfg.data.max_input_len + cfg.data.max_output_len,
        )
    else:
        collate_fn = None   # build_dataloaders defaults to PaddingCollator

    with _console.status("[dim]dataloaders :[/] building..."):
        train_dl, val_dl, _ = build_dataloaders(
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
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
            seed=cfg.training.seed,
            collate_fn=collate_fn,
        )
    _console.print(
        f"[dim]dataloaders :[/] train={len(train_dl.dataset):,}  "
        f"val={len(val_dl.dataset):,}  "
        f"batch={cfg.training.batch_size}  "
        f"accum={cfg.training.gradient_accumulation_steps}  "
        f"→ effective={cfg.training.batch_size * cfg.training.gradient_accumulation_steps}"
    )

    with _console.status("[dim]model       :[/] building..."):
        model = build_model(arch, cfg)

    if arch == "decoder_only":
        layer_str = f"layers={cfg.model.num_decoder_layers}"
    elif arch == "encoder_only":
        layer_str = f"layers={cfg.model.num_encoder_layers}"
    else:
        layer_str = f"layers={cfg.model.num_encoder_layers}/{cfg.model.num_decoder_layers}"
    _console.print(
        f"[dim]model       :[/] arch={arch}  d_model={cfg.model.d_model}  "
        f"heads={cfg.model.num_heads}  {layer_str}  "
        f"params={model.count_parameters():,}"
    )

    run_name = cfg.training.run_name or _make_run_name(cfg)
    checkpoint_dir = f"{cfg.training.checkpoint_dir}/{run_name}"
    log_dir = f"runs/{run_name}"
    _console.print(f"[dim]run         :[/] [cyan]{run_name}[/]")
    _console.print("[dim]tensorboard :[/] [dim]tensorboard --logdir runs[/]")

    notifier = TelegramNotifier(bot_token=os.environ.get("TELEGRAM_BOT_TOKEN"), chat_id=os.environ.get("TELEGRAM_CHAT_ID"))
    if notifier.enabled:
        _console.print("[dim]telegram    :[/] [green]enabled[/] — /stop /status /help")
    else:
        _console.print("[dim]telegram    :[/] [dim]disabled (TELEGRAM_BOT_TOKEN not set)[/]")

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        device=device,
        d_model=cfg.model.d_model,
        warmup_steps=cfg.training.warmup_steps,
        label_smoothing=cfg.training.label_smoothing,
        gradient_clip=cfg.training.gradient_clip,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        amp=cfg.training.amp,
        save_top_k=cfg.training.save_top_k,
        checkpoint_dir=checkpoint_dir,
        log_every=cfg.training.log_every,
        log_dir=log_dir,
        early_stopping_patience=cfg.training.early_stopping_patience,
        early_stopping_min_delta=cfg.training.early_stopping_min_delta,
        optimizer=cfg.training.optimizer,
        weight_decay=cfg.training.weight_decay,
        notifier=notifier,
        architecture=arch,
    )

    if cfg.training.torch_compile == "graph":
        trainer.model = torch.compile(trainer.model)
        _console.print("[dim]compile     :[/] [cyan]graph (torch.compile)[/]")
    else:
        _console.print("[dim]compile     :[/] [dim]eager[/]")

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    trainer.fit(cfg.training.epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
