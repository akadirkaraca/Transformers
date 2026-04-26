"""Multi-format model export script.

Converts a trained checkpoint into one or more deployment formats.
Each format is written into its own sub-directory under --output-dir.

Usage::

    uv run python scripts/export.py \\
        --config configs/base.yaml \\
        --checkpoint checkpoints/run_name/checkpoint_epoch010.ckpt \\
        --format safetensors \\
        --output-dir exports

    # Multiple formats at once
    uv run python scripts/export.py ... --format onnx,torchscript,safetensors

    # Everything available (skips formats with missing deps)
    uv run python scripts/export.py ... --format all

Output layout::

    exports/
    ├── safetensors/          # HuggingFace-compatible directory
    │   ├── model.safetensors
    │   ├── config.json
    │   ├── generation_config.json
    │   ├── tokenizer_config.json
    │   ├── special_tokens_map.json
    │   └── <tokenizer model file>
    ├── onnx/
    │   └── model.onnx
    ├── torchscript/
    │   └── model.pt
    ├── tensorflow/
    │   └── saved_model/      # TF SavedModel directory
    ├── tflite/
    │   └── model.tflite
    ├── tf_ckpt/              # TF Variables checkpoint
    │   ├── checkpoint
    │   ├── model.index
    │   └── model.data-*
    ├── jax/
    │   ├── weights.npz       # NumPy archive (key = state_dict key)
    │   └── metadata.json
    └── coreml/
        └── model.mlpackage   # macOS / iOS

Optional dependencies (install only what you need)::

    uv pip install safetensors                        # safetensors
    uv pip install onnx onnxruntime                   # onnx
    uv pip install onnx onnx-tf tensorflow            # tensorflow / tflite / tf_ckpt
    uv pip install coremltools                        # coreml  (macOS only)
    # jax format uses numpy only — no extra install needed

    # Or install all at once:
    uv pip install "transformer[export]"
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from transformer.config.config import load_config
from transformer.model import build_model
from transformer.training.trainer import load_model_weights

SUPPORTED_FORMATS = [
    "safetensors",
    "onnx",
    "torchscript",
    "tensorflow",
    "tflite",
    "tf_ckpt",
    "jax",
    "coreml",
]

# --------------------------------------------------------------------------- #
#  Dummy input helpers                                                          #
# --------------------------------------------------------------------------- #

def _dummy_inputs(cfg, device: torch.device) -> tuple:
    """Minimal concrete inputs for each architecture."""
    arch = cfg.model.architecture
    B = 1
    if arch == "encoder_decoder":
        src_len = min(cfg.data.max_input_len, 64)
        tgt_len = min(cfg.data.max_output_len, 16)
        src = torch.randint(4, cfg.tokenizer.vocab_size, (B, src_len), device=device)
        tgt = torch.randint(4, cfg.tokenizer.vocab_size, (B, tgt_len), device=device)
        src_mask = torch.zeros(B, src_len, dtype=torch.bool, device=device)
        tgt_mask = torch.zeros(B, tgt_len, dtype=torch.bool, device=device)
        return (src, tgt, src_mask, tgt_mask)
    else:
        seq_len = min(cfg.data.max_input_len, 64)
        input_ids = torch.randint(4, cfg.tokenizer.vocab_size, (B, seq_len), device=device)
        attn_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
        return (input_ids, attn_mask)


def _input_names(cfg) -> list[str]:
    if cfg.model.architecture == "encoder_decoder":
        return ["src", "tgt_in", "src_mask", "tgt_mask"]
    return ["input_ids", "attn_mask"]


def _dynamic_axes(cfg) -> dict:
    if cfg.model.architecture == "encoder_decoder":
        return {
            "src":      {0: "batch", 1: "src_len"},
            "tgt_in":   {0: "batch", 1: "tgt_len"},
            "src_mask": {0: "batch", 1: "src_len"},
            "tgt_mask": {0: "batch", 1: "tgt_len"},
            "logits":   {0: "batch", 1: "tgt_len"},
        }
    return {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attn_mask": {0: "batch", 1: "seq_len"},
        "logits":    {0: "batch", 1: "seq_len"},
    }


# --------------------------------------------------------------------------- #
#  Format exporters                                                             #
# --------------------------------------------------------------------------- #

def export_safetensors(model: nn.Module, cfg, out_dir: Path, checkpoint_path: str) -> None:
    """HuggingFace-compatible safetensors directory.

    Produces the same layout as a HF Hub model repository:
      model.safetensors, config.json, generation_config.json,
      tokenizer_config.json, special_tokens_map.json, <tokenizer model>.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError(
            "safetensors not installed.\n"
            "  uv pip install safetensors"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    m, t, inf = cfg.model, cfg.training, cfg.inference

    # model.safetensors
    save_file(model.state_dict(), str(out_dir / "model.safetensors"))

    # config.json
    config_dict = {
        "architectures": [f"Transformer_{m.architecture}"],
        "model_type": "transformer",
        "architecture": m.architecture,
        "d_model": m.d_model,
        "num_heads": m.num_heads,
        "num_encoder_layers": m.num_encoder_layers,
        "num_decoder_layers": m.num_decoder_layers,
        "d_ff": m.d_ff,
        "dropout": m.dropout,
        "max_seq_len": m.max_seq_len,
        "weight_tying": m.weight_tying,
        "encoder_activation": m.encoder_activation,
        "decoder_activation": m.decoder_activation,
        "attn_impl": m.attn_impl,
        "vocab_size": cfg.tokenizer.vocab_size,
        "pad_token_id": cfg.tokenizer.pad_id,
        "bos_token_id": cfg.tokenizer.bos_id,
        "eos_token_id": cfg.tokenizer.eos_id,
        "unk_token_id": cfg.tokenizer.unk_id,
        "torch_dtype": "float32",
    }
    _write_json(out_dir / "config.json", config_dict)

    # generation_config.json
    gen_config = {
        "bos_token_id": cfg.tokenizer.bos_id,
        "eos_token_id": cfg.tokenizer.eos_id,
        "pad_token_id": cfg.tokenizer.pad_id,
        "max_new_tokens": inf.max_decode_len,
        "min_new_tokens": inf.min_decode_len,
        "num_beams": inf.beam_size,
        "length_penalty": inf.length_penalty,
        "_from_model_config": True,
    }
    _write_json(out_dir / "generation_config.json", gen_config)

    # tokenizer_config.json
    tok_config = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "model_max_length": m.max_seq_len,
        "tokenizer_class": "PreTrainedTokenizer",
        "sp_model_kwargs": {},
    }
    _write_json(out_dir / "tokenizer_config.json", tok_config)

    # special_tokens_map.json
    _write_json(out_dir / "special_tokens_map.json", {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
    })

    # Copy tokenizer model file
    tok_src = Path(cfg.tokenizer.model_path)
    if tok_src.exists():
        shutil.copy2(tok_src, out_dir / tok_src.name)
    else:
        print(f"    warning: tokenizer file not found at {tok_src}, skipping.")

    _print_tree(out_dir)


def export_onnx(model: nn.Module, cfg, out_dir: Path, device: torch.device) -> None:
    """ONNX graph exported via torch.onnx.

    The model's full forward() pass is exported with dynamic batch and
    sequence-length axes. For encoder_decoder, this is the teacher-forced
    training forward (src + tgt_in → logits).
    """
    try:
        import onnx
    except ImportError:
        raise ImportError(
            "onnx not installed.\n"
            "  uv pip install onnx onnxruntime"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.onnx"
    model.eval()

    torch.onnx.export(
        model,
        _dummy_inputs(cfg, device),
        str(out_path),
        opset_version=17,
        input_names=_input_names(cfg),
        output_names=["logits"],
        dynamic_axes=_dynamic_axes(cfg),
        do_constant_folding=True,
    )

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    _print_tree(out_dir)


def export_torchscript(model: nn.Module, cfg, out_dir: Path, device: torch.device) -> None:
    """TorchScript traced model.

    Note: torch.jit.trace records concrete shapes at trace time. The exported
    model works for any batch size but sequence lengths must match those used
    during tracing for operations that depend on shape (e.g. causal masks).
    For shape-agnostic export consider using the ONNX exporter instead.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.pt"
    model.eval()

    with torch.no_grad():
        traced = torch.jit.trace(model, _dummy_inputs(cfg, device))

    torch.jit.save(traced, str(out_path))
    _print_tree(out_dir)


def export_tensorflow(model: nn.Module, cfg, out_dir: Path, device: torch.device) -> None:
    """TF SavedModel via ONNX → onnx-tf."""
    _require_onnx_tf()

    import onnx
    import onnx_tf.backend as tf_backend

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_model_dir = out_dir / "saved_model"

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "model.onnx"
        _export_onnx_to_path(model, cfg, device, onnx_path)
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = tf_backend.prepare(onnx_model)
        tf_rep.export_graph(str(saved_model_dir))

    _print_tree(out_dir)


def export_tflite(model: nn.Module, cfg, out_dir: Path, device: torch.device) -> None:
    """TFLite flatbuffer via ONNX → SavedModel → TFLite."""
    _require_onnx_tf()

    import onnx
    import onnx_tf.backend as tf_backend
    import tensorflow as tf

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.tflite"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        onnx_path = tmp_p / "model.onnx"
        saved_model_dir = tmp_p / "saved_model"

        _export_onnx_to_path(model, cfg, device, onnx_path)
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = tf_backend.prepare(onnx_model)
        tf_rep.export_graph(str(saved_model_dir))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        tflite_model = converter.convert()

    out_path.write_bytes(tflite_model)
    _print_tree(out_dir)


def export_tf_ckpt(model: nn.Module, cfg, out_dir: Path, device: torch.device) -> None:
    """TF Variables checkpoint via ONNX → SavedModel → tf.train.Checkpoint."""
    _require_onnx_tf()

    import onnx
    import onnx_tf.backend as tf_backend
    import tensorflow as tf

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        onnx_path = tmp_p / "model.onnx"
        saved_model_dir = tmp_p / "saved_model"

        _export_onnx_to_path(model, cfg, device, onnx_path)
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = tf_backend.prepare(onnx_model)
        tf_rep.export_graph(str(saved_model_dir))

        loaded = tf.saved_model.load(str(saved_model_dir))
        ckpt = tf.train.Checkpoint(model=loaded)
        ckpt.write(str(out_dir / "model"))

    _print_tree(out_dir)


def export_jax(model: nn.Module, cfg, out_dir: Path) -> None:
    """NumPy weights archive for JAX / Flax / Haiku loading.

    Saves every entry in the PyTorch state_dict as a named array in a
    single .npz file. Load with::

        import numpy as np
        weights = dict(np.load("exports/jax/weights.npz", allow_pickle=False))
    """
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    np_weights = {k: v.cpu().float().numpy() for k, v in state_dict.items()}
    np.savez(str(out_dir / "weights.npz"), **np_weights)

    _write_json(out_dir / "metadata.json", {
        "framework": "jax",
        "num_tensors": len(np_weights),
        "keys": list(np_weights.keys()),
        "note": (
            "Load with np.load('weights.npz', allow_pickle=False). "
            "Keys match PyTorch state_dict — map to your Flax/Haiku module manually."
        ),
    })

    _print_tree(out_dir)


def export_coreml(model: nn.Module, cfg, out_dir: Path, device: torch.device) -> None:
    """Core ML package via coremltools (macOS / iOS deployment).

    Requires macOS and coremltools>=7.0.
    """
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError(
            "coremltools not installed.\n"
            "  uv pip install coremltools"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.mlpackage"
    model.eval()

    dummy = _dummy_inputs(cfg, device)
    input_names = _input_names(cfg)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)

    inputs = []
    for name, tensor in zip(input_names, dummy):
        dtype = bool if tensor.dtype == torch.bool else int
        inputs.append(ct.TensorType(name=name, dtype=dtype))

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.save(str(out_path))
    _print_tree(out_dir)


# --------------------------------------------------------------------------- #
#  Internal helpers                                                             #
# --------------------------------------------------------------------------- #

def _export_onnx_to_path(model: nn.Module, cfg, device: torch.device, path: Path) -> None:
    model.eval()
    torch.onnx.export(
        model,
        _dummy_inputs(cfg, device),
        str(path),
        opset_version=17,
        input_names=_input_names(cfg),
        output_names=["logits"],
        dynamic_axes=_dynamic_axes(cfg),
        do_constant_folding=True,
    )


def _require_onnx_tf() -> None:
    try:
        import onnx          # noqa: F401
        import onnx_tf       # noqa: F401
        import tensorflow    # noqa: F401
    except ImportError:
        raise ImportError(
            "tensorflow export requires onnx, onnx-tf, and tensorflow.\n"
            "  uv pip install onnx onnx-tf tensorflow"
        )


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _print_tree(directory: Path) -> None:
    print(f"    {directory}/")
    for p in sorted(directory.rglob("*")):
        indent = "    " + "  " * (len(p.relative_to(directory).parts) - 1)
        size = f"  ({p.stat().st_size / 1e6:.1f} MB)" if p.is_file() else ""
        print(f"    {indent}{p.name}{size}")


# --------------------------------------------------------------------------- #
#  Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained checkpoint to one or more deployment formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}, all",
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .ckpt, .pt, or .safetensors file.",
    )
    parser.add_argument(
        "--format", required=True,
        help="Comma-separated list of formats or 'all'.",
    )
    parser.add_argument(
        "--output-dir", default="exports",
        help="Root output directory. Each format gets its own sub-directory.",
    )
    args = parser.parse_args()

    if args.format.strip().lower() == "all":
        formats = SUPPORTED_FORMATS
    else:
        formats = [f.strip() for f in args.format.split(",")]
        unknown = [f for f in formats if f not in SUPPORTED_FORMATS]
        if unknown:
            parser.error(
                f"Unknown format(s): {unknown}\n"
                f"Supported: {', '.join(SUPPORTED_FORMATS)}"
            )

    cfg = load_config(args.config)
    # Always export on CPU for maximum portability.
    device = torch.device("cpu")

    print(f"Architecture : {cfg.model.architecture}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Output dir   : {args.output_dir}/")
    print(f"Formats      : {formats}\n")

    model = build_model(cfg.model.architecture, cfg)
    load_model_weights(args.checkpoint, model, device)
    model.eval()

    out_root = Path(args.output_dir)
    results: list[tuple[str, str]] = []  # (format, status)

    for fmt in formats:
        print(f"── {fmt}")
        try:
            out_dir = out_root / fmt
            if fmt == "safetensors":
                export_safetensors(model, cfg, out_dir, args.checkpoint)
            elif fmt == "onnx":
                export_onnx(model, cfg, out_dir, device)
            elif fmt == "torchscript":
                export_torchscript(model, cfg, out_dir, device)
            elif fmt == "tensorflow":
                export_tensorflow(model, cfg, out_dir, device)
            elif fmt == "tflite":
                export_tflite(model, cfg, out_dir, device)
            elif fmt == "tf_ckpt":
                export_tf_ckpt(model, cfg, out_dir, device)
            elif fmt == "jax":
                export_jax(model, cfg, out_dir)
            elif fmt == "coreml":
                export_coreml(model, cfg, out_dir, device)
            results.append((fmt, "ok"))
            print()
        except ImportError as exc:
            print(f"    SKIPPED — missing dependency: {exc}\n")
            results.append((fmt, "skipped"))
        except Exception as exc:
            print(f"    FAILED  — {exc}\n")
            results.append((fmt, "failed"))

    print("── Summary")
    for fmt, status in results:
        icon = {"ok": "✓", "skipped": "○", "failed": "✗"}[status]
        print(f"  {icon}  {fmt:<16} {status}")


if __name__ == "__main__":
    main()
