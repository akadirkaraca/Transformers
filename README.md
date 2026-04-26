# Transformer

![Python](https://img.shields.io/badge/python-%E2%89%A53.10-3776ab?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-EE4C2C?logo=pytorch&logoColor=white)
![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

A general-purpose LLM training framework built on PyTorch. Supports encoder-decoder (seq2seq), encoder-only (MLM / BERT-style), and decoder-only (causal LM / GPT-style) architectures from a single codebase. Comes with a pluggable architecture registry so you can drop in any `nn.Module` and train it with the same pipeline. Works with CSV, JSONL, and plain-text datasets in any language.

---

## Features

- Three built-in architectures: `encoder_decoder`, `encoder_only`, `decoder_only`
- Pluggable architecture registry — register any `nn.Module` factory and train it without touching the framework
- Format-agnostic data pipeline: CSV, JSONL, plain text; single-column or paired input/output
- SentencePiece BPE tokenizer with shared vocab and weight tying
- Noam warm-up schedule, AdamW/Adam, gradient accumulation, gradient clipping
- Automatic Mixed Precision (AMP) via `torch.amp`
- Beam search decoding with length penalty; greedy decoding fallback
- BLEU-1..4 and ROUGE-1/2/L evaluation
- TensorBoard logging, top-K checkpoint saving, early stopping
- Telegram training notifications (optional)
- Multi-format checkpointing: `.ckpt` (full training state) + `.safetensors` (weights-only, no pickle)
- Multi-framework export: safetensors (HF-compatible), ONNX, TorchScript, TensorFlow, TFLite, TF checkpoint, JAX/NumPy, Core ML

---

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) — package and environment manager
- CUDA GPU recommended for training (CPU works for small-scale experiments)

---

## Setup

```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install the package in editable mode
uv pip install -e .
```

---

## Quick Start

### 1. Prepare data

Place your dataset at the path set in `configs/base.yaml` (`data.data_path`). Supported formats:

**CSV** — two-column file with a header row:
```
input,output
"source text","target text"
```

**JSONL** — one JSON object per line:
```json
{"input": "source text", "output": "target text"}
```

**Plain text** — one sample per line (single-column / decoder-only only):
```
This is a training sentence.
Another sentence follows here.
```

For single-column mode (causal LM pretraining), set `output_col: ""` in `base.yaml`.

### 2. Configure

Edit `configs/base.yaml` to match your dataset and task:

```yaml
model:
  architecture: encoder_decoder   # "encoder_decoder" | "encoder_only" | "decoder_only"

data:
  data_path: data/dataset.csv
  data_format: csv                # "csv" | "jsonl" | "text"
  input_col: input                # column/field name for the input
  output_col: output              # column/field name for the output; "" = single-column mode
  preprocessing: none             # "none" | "basic" (NFC + URL/email removal + whitespace normalization)
  max_input_len: 512
  max_output_len: 128
```

### 3. Build tokenizer

Trains a SentencePiece BPE model on your dataset and saves it to `tokenizer/bpe32k.model`:

```bash
uv run python scripts/build_tokenizer.py --config configs/base.yaml
```

### 4. Train

```bash
uv run python scripts/train.py --config configs/base.yaml
```

Resume from a checkpoint (requires a `.ckpt` or `.pt` file — full training state):

```bash
uv run python scripts/train.py --config configs/base.yaml \
    --resume checkpoints/run_name/checkpoint_epoch005.ckpt
```

Load a custom architecture module before training:

```bash
uv run python scripts/train.py --config configs/base.yaml \
    --arch-module path/to/my_model.py
```

### 5. Evaluate

BLEU and ROUGE evaluation on the test split (encoder_decoder only).
Accepts `.ckpt`, `.pt`, or `.safetensors` checkpoints:

```bash
uv run python scripts/evaluate.py --config configs/base.yaml \
    --checkpoint checkpoints/run_name/checkpoint_epoch010.ckpt
```

### 6. Generate

**Encoder-decoder** — beam search decoding:

```bash
# Interactive
uv run python scripts/generate.py --config configs/base.yaml \
    --checkpoint checkpoints/run_name/checkpoint_epoch010.ckpt

# Batch (one input per line)
uv run python scripts/generate.py --config configs/base.yaml \
    --checkpoint checkpoints/run_name/checkpoint_epoch010.ckpt \
    --input inputs.txt --output outputs.txt
```

**Decoder-only** — autoregressive greedy decoding:

```bash
# Interactive
uv run python scripts/causal_generate.py --config configs/base.yaml \
    --checkpoint checkpoints/run_name/checkpoint_epoch010.ckpt

# Batch
uv run python scripts/causal_generate.py --config configs/base.yaml \
    --checkpoint checkpoints/run_name/checkpoint_epoch010.ckpt \
    --input prompts.txt --output completions.txt
```

### 7. Export

Convert a checkpoint to a deployment format:

```bash
# HuggingFace-compatible safetensors directory
uv run python scripts/export.py --config configs/base.yaml \
    --checkpoint checkpoints/run_name/checkpoint_epoch010.ckpt \
    --format safetensors

# ONNX graph
uv run python scripts/export.py ... --format onnx

# Multiple formats in one run
uv run python scripts/export.py ... --format onnx,torchscript,safetensors

# Try all formats (skips any with missing optional dependencies)
uv run python scripts/export.py ... --format all --output-dir exports
```

Each format is written to its own sub-directory under `--output-dir` (default: `exports/`).
See the [Checkpoint & Export](#checkpoint--export) section for available formats and install instructions.

---

## Architecture Modes

### encoder_decoder

Full encoder-decoder Transformer (Vaswani et al., 2017). The encoder reads the input with bidirectional
attention; the decoder generates the output token by token using cross-attention over the encoder output
and causal self-attention. Training uses teacher forcing; inference uses beam search.

Suitable for: translation, summarization, any seq2seq task.

```yaml
model:
  architecture: encoder_decoder
data:
  output_col: output    # required — target sequences
```

### encoder_only

Bidirectional encoder with a Masked Language Modeling (MLM) objective. At each training step,
15% of input tokens are masked (80% replaced with `mask_id`, 10% replaced with a random token,
10% kept as-is) and the model predicts the original tokens. Mirrors the BERT pre-training setup.

Suitable for: text representation learning, classification pre-training.

```yaml
model:
  architecture: encoder_only
training:
  mlm_probability: 0.15
  mask_id: 1            # token used as [MASK]; defaults to unk_id
```

### decoder_only

Causal decoder stack with no cross-attention. Each position can only attend to preceding positions.
Training minimizes next-token prediction loss (causal LM).

Two sub-modes controlled by `output_col`:
- **Paired** (`output_col` non-empty): input tokens are prepended as a read-only prefix (masked from loss);
  loss is computed only on output tokens. For conditional generation (e.g., prompt → completion).
- **Single-column** (`output_col: ""`): loss computed over the entire sequence. For GPT-style pretraining.

```yaml
model:
  architecture: decoder_only
data:
  output_col: ""        # single-column pretraining
  # output_col: output  # paired / conditional generation
```

---

## Custom Architectures

Implement any `nn.Module`, register it, and train it with the same pipeline:

```python
# my_model.py
import torch.nn as nn
from transformer.model.registry import register_architecture

class MyLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # cfg.model.d_model, cfg.model.num_heads, etc.
        ...

    def forward(self, src, src_mask=None):
        # Return logits of shape (batch, seq_len, vocab_size)
        ...

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

register_architecture("my_llm", lambda cfg: MyLLM(cfg))
```

Set `model.architecture: my_llm` in `base.yaml`, then:

```bash
uv run python scripts/train.py --config configs/base.yaml --arch-module my_model.py
```

The `--arch-module` flag imports the file before training starts, which triggers registration.

---

## Data Formats

| Format | `data_format` | Input structure | When to use |
|---|---|---|---|
| CSV | `csv` | Header row + delimiter-separated columns | Tabular data with named columns |
| JSONL | `jsonl` | One JSON object per line | Streaming or nested data |
| Plain text | `text` | One sample per line | Pretraining corpora, single-column only |

For `text` format, `input_col` is ignored — every non-empty line is one sample. Set `output_col: ""`.

---

## Configuration Reference

All hyperparameters are set in `configs/base.yaml`. Values shown are the current defaults in that file.

### model

| Key | Default | Description |
|---|---|---|
| `architecture` | `encoder_decoder` | Training mode: `encoder_decoder`, `encoder_only`, `decoder_only` |
| `d_model` | `512` | Embedding and hidden dimension |
| `num_heads` | `8` | Number of attention heads |
| `num_encoder_layers` | `4` | Encoder stack depth |
| `num_decoder_layers` | `4` | Decoder stack depth |
| `d_ff` | `2048` | Feed-forward inner dimension |
| `dropout` | `0.2` | Dropout probability |
| `max_seq_len` | `512` | Maximum positional encoding capacity |
| `weight_tying` | `true` | Tie embedding and output projection weights |
| `encoder_activation` | `gelu` | FFN activation in encoder: `relu`, `gelu`, `silu` |
| `decoder_activation` | `silu` | FFN activation in decoder: `relu`, `gelu`, `silu` |

### tokenizer

| Key | Default | Description |
|---|---|---|
| `vocab_size` | `32000` | BPE vocabulary size |
| `model_type` | `bpe` | SentencePiece model type |
| `character_coverage` | `0.9995` | Fraction of characters covered |
| `model_path` | `tokenizer/bpe32k.model` | Path to trained tokenizer |
| `pad_id` | `0` | Padding token ID |
| `unk_id` | `1` | Unknown token ID |
| `bos_id` | `2` | Beginning-of-sequence token ID |
| `eos_id` | `3` | End-of-sequence token ID |

### training

| Key | Default | Description |
|---|---|---|
| `epochs` | `40` | Total training epochs |
| `batch_size` | `32` | Samples per batch |
| `learning_rate` | `1.5e-4` | Peak learning rate |
| `warmup_steps` | `4000` | Linear warm-up steps (Noam schedule) |
| `optimizer` | `adamw` | Optimizer: `adam`, `adamw` |
| `weight_decay` | `0.05` | AdamW weight decay |
| `label_smoothing` | `0.15` | Label smoothing factor |
| `gradient_clip` | `1.0` | Max gradient norm |
| `gradient_accumulation_steps` | `16` | Steps before each optimizer update |
| `amp` | `true` | Automatic Mixed Precision (GPU only) |
| `save_top_k` | `3` | Number of best checkpoints to keep |
| `checkpoint_dir` | `checkpoints` | Directory for saved checkpoints |
| `log_every` | `100` | Log loss every N steps |
| `eval_every` | `1` | Evaluate on validation set every N epochs |
| `seed` | `42` | Random seed |
| `early_stopping_patience` | `3` | Stop after N epochs without improvement (0 = disabled) |
| `early_stopping_min_delta` | `0.001` | Minimum improvement to reset patience counter |
| `mlm_probability` | `0.15` | Fraction of tokens masked (encoder_only only) |
| `mask_id` | `1` | Token ID used as `[MASK]` |

### data

| Key | Default | Description |
|---|---|---|
| `data_path` | `data/dataset.csv` | Path to the dataset file |
| `data_format` | `csv` | File format: `csv`, `jsonl`, `text` |
| `input_col` | `input` | Column/field name for input sequences |
| `output_col` | `output` | Column/field name for output sequences; `""` = single-column mode |
| `preprocessing` | `none` | Text preprocessing: `none`, `basic` |
| `train_ratio` | `0.85` | Fraction of data used for training |
| `val_ratio` | `0.10` | Fraction used for validation |
| `test_ratio` | `0.05` | Fraction used for testing |
| `max_input_len` | `512` | Maximum input sequence length in tokens |
| `max_output_len` | `128` | Maximum output sequence length in tokens (encoder_decoder only) |
| `num_workers` | `2` | DataLoader worker processes |

### inference

| Key | Default | Description |
|---|---|---|
| `beam_size` | `4` | Beam search width |
| `length_penalty` | `0.6` | Length penalty alpha: `score / ((5 + len) / 6)^alpha` |
| `max_decode_len` | `64` | Maximum tokens to generate |
| `min_decode_len` | `3` | EOS is suppressed below this length |

### checkpoint

Controls which file formats are written to disk after each training epoch.

| Key | Default | Description |
|---|---|---|
| `formats` | `[ckpt, safetensors]` | List of formats to save at each checkpoint |

Available format values:

| Value | Extension | Contents | Use case |
|---|---|---|---|
| `ckpt` | `.ckpt` | Model weights + optimizer + scheduler + metadata | Resume training, full recovery |
| `safetensors` | `.safetensors` | Model weights only (no pickle) | Inference, sharing, HF Hub |
| `pt` | `.pt` | Same as `ckpt` (legacy name) | Backwards compatibility |

**Examples:**

Save only the full-state checkpoint (smallest disk footprint):
```yaml
checkpoint:
  formats: [ckpt]
```

Save both formats (default — recommended):
```yaml
checkpoint:
  formats: [ckpt, safetensors]
```

Save only model weights for pure inference deployments:
```yaml
checkpoint:
  formats: [safetensors]
  # Note: you cannot resume training from a .safetensors file.
```

The `save_top_k` setting under `training` applies across all formats together —
when a checkpoint is evicted, all of its format variants are deleted.

---

## Checkpoint & Export

### Checkpoint file types

After each epoch the trainer writes one file per configured format into
`checkpoints/<run_name>/`. At most `save_top_k` epoch checkpoints are kept
(worst validation-loss checkpoint is removed first):

```
checkpoints/
└── encdec_d512_L4_adamw_do20/
    ├── checkpoint_epoch008.ckpt         # full training state — resumable
    ├── checkpoint_epoch008.safetensors  # model weights only
    ├── checkpoint_epoch012.ckpt
    ├── checkpoint_epoch012.safetensors
    ├── checkpoint_epoch015.ckpt
    └── checkpoint_epoch015.safetensors
```

**Resuming training** always requires a `.ckpt` or `.pt` file:

```bash
uv run python scripts/train.py --config configs/base.yaml \
    --resume checkpoints/run_name/checkpoint_epoch012.ckpt
```

**Loading for inference** (`evaluate.py`, `generate.py`, `causal_generate.py`)
accepts any format:

```bash
# From a full-state checkpoint
--checkpoint checkpoints/run_name/checkpoint_epoch015.ckpt

# From a weights-only safetensors file
--checkpoint checkpoints/run_name/checkpoint_epoch015.safetensors
```

### Export to deployment formats

`scripts/export.py` converts any checkpoint into one or more framework-specific
formats. All exports run on CPU for portability.

```bash
uv run python scripts/export.py \
    --config  configs/base.yaml \
    --checkpoint checkpoints/run_name/checkpoint_epoch015.ckpt \
    --format  <format> \
    --output-dir exports          # default
```

| `--format` | Output path | Contents | Extra install |
|---|---|---|---|
| `safetensors` | `exports/safetensors/` | HF-compatible dir: `model.safetensors` + config/tokenizer JSONs | *(core dep)* |
| `onnx` | `exports/onnx/model.onnx` | ONNX graph, opset 17, dynamic axes | `onnx onnxruntime` |
| `torchscript` | `exports/torchscript/model.pt` | TorchScript traced model | *(built-in)* |
| `tensorflow` | `exports/tensorflow/saved_model/` | TF SavedModel via ONNX→onnx-tf | `onnx onnx-tf tensorflow` |
| `tflite` | `exports/tflite/model.tflite` | TFLite flatbuffer | `onnx onnx-tf tensorflow` |
| `tf_ckpt` | `exports/tf_ckpt/` | TF Variables checkpoint | `onnx onnx-tf tensorflow` |
| `jax` | `exports/jax/weights.npz` + `metadata.json` | NumPy arrays for Flax/Haiku | *(numpy, built-in)* |
| `coreml` | `exports/coreml/model.mlpackage` | Core ML package | `coremltools` (macOS) |

Install optional dependencies for the formats you need:

```bash
uv pip install onnx onnxruntime                   # onnx
uv pip install onnx onnx-tf tensorflow            # tensorflow / tflite / tf_ckpt
uv pip install coremltools                        # coreml
# jax format requires no extra packages

# Or install everything at once:
uv pip install "transformer[export]"
```

**safetensors directory layout** (HuggingFace Hub-compatible):

```
exports/safetensors/
├── model.safetensors       # model weights, no pickle
├── config.json             # architecture + vocab config
├── generation_config.json  # beam size, length penalty, token IDs
├── tokenizer_config.json   # special token names, max length
├── special_tokens_map.json # BOS / EOS / PAD / UNK token strings
└── bpe32k.model            # SentencePiece tokenizer model file
```

**JAX weights layout:**

```
exports/jax/
├── weights.npz     # np.load(..., allow_pickle=False) → dict keyed by state_dict names
└── metadata.json   # tensor list + loading note
```

Load in JAX/Flax:
```python
import numpy as np
weights = dict(np.load("exports/jax/weights.npz", allow_pickle=False))
# weights["encoder.layers.0.self_attn.q_proj.weight"] → np.ndarray
```

**Running multiple formats at once:**

```bash
# Specific formats
uv run python scripts/export.py \
    --config configs/base.yaml \
    --checkpoint checkpoints/run_name/checkpoint_epoch015.ckpt \
    --format onnx,torchscript,safetensors

# All formats — formats with missing dependencies are skipped, not fatal
uv run python scripts/export.py \
    --config configs/base.yaml \
    --checkpoint checkpoints/run_name/checkpoint_epoch015.ckpt \
    --format all
```

Example summary output:

```
── Summary
  ✓  safetensors      ok
  ✓  onnx             ok
  ✓  torchscript      ok
  ○  tensorflow       skipped
  ○  tflite           skipped
  ○  tf_ckpt          skipped
  ✓  jax              ok
  ○  coreml           skipped
```

---

## Project Structure

```
transformer/
├── config/          # Dataclass configs + YAML loader (load_config)
├── data/            # TextDataset, PaddingCollator, MLMCollator, CausalCollator, preprocessing
├── tokenizer/       # TokenizerBase ABC, SentencePiece BPE, HuggingFace wrapper
├── model/           # TokenEmbedding, PositionalEncoding, MultiHeadAttention, FFN,
│                    # Encoder, Decoder, Transformer (enc-dec), EncoderOnlyTransformer,
│                    # CausalDecoder, DecoderOnlyTransformer, architecture registry
├── training/        # LabelSmoothingLoss, WarmupScheduler (Noam), Trainer, TelegramNotifier
├── inference/       # greedy_decode, beam_search_decode
└── evaluation/      # compute_bleu (1..4), compute_rouge (R-1, R-2, R-L)

configs/
└── base.yaml        # All hyperparameters

scripts/
├── build_tokenizer.py    # Train SentencePiece BPE tokenizer
├── train.py              # Main training entry point
├── evaluate.py           # BLEU/ROUGE evaluation on test split (encoder_decoder)
├── generate.py           # Interactive/batch generation — encoder_decoder (beam search)
├── causal_generate.py    # Interactive/batch generation — decoder_only (greedy)
├── export.py             # Export checkpoint to safetensors/ONNX/TF/TFLite/JAX/CoreML/…
├── clean_data.py         # Optional data cleaning pipeline
└── bot_launcher.py       # Telegram bot for remote training monitoring

data/
├── data.csv         # Empty CSV template
└── data.jsonl       # Empty JSONL template

tokenizer/
├── bpe32k.model     # Trained SentencePiece BPE model
└── bpe32k.vocab     # Vocabulary file
```

---

## Development

```bash
# Lint
uv run ruff check transformer/ scripts/

# Format
uv run ruff format transformer/ scripts/

# Tests (test suite not yet implemented)
uv run pytest tests/
```

---

## Telegram Notifications

The trainer can send epoch summaries and training progress to a Telegram chat. Create a `.env` file
in the project root (it is excluded from version control by `.gitignore`):

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

Create a bot via [@BotFather](https://t.me/BotFather) on Telegram to obtain the token.
The notifier activates automatically when both values are present in the environment.
