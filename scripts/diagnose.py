"""Dataset and tokenizer diagnostic report.

Analyzes token-length distributions across the configured dataset and tokenizer,
then recommends optimal max_input_len, max_output_len, and max_seq_len values.

Usage::

    uv run python scripts/diagnose.py --config configs/base.yaml
    uv run python scripts/diagnose.py --config configs/base.yaml --sample 50000
"""
from __future__ import annotations

import argparse
import math
import random

import numpy as np
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.rule import Rule
from rich.table import Table

from transformer.config.config import load_config
from transformer.data.dataset import _load_raw_texts
from transformer.tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer

_console = Console()

_PERCENTILES = [50, 60, 75, 90, 95, 99]
_COVERAGE_THRESHOLDS = [32, 64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 2048]


# ── Statistics ────────────────────────────────────────────────────────────────

def _stats(lengths: np.ndarray) -> dict:
    """Compute descriptive statistics including key percentiles."""
    pct = np.percentile(lengths, _PERCENTILES)
    return {
        "min":  int(lengths.min()),
        "max":  int(lengths.max()),
        "mean": float(lengths.mean()),
        "std":  float(lengths.std()),
        **{f"p{p}": int(v) for p, v in zip(_PERCENTILES, pct)},
    }


# ── Tokenization with progress bar ────────────────────────────────────────────

def _tokenize_batch(texts: list[str], tokenizer: SentencePieceTokenizer, label: str) -> list[list[int]]:
    """Tokenize all texts (without truncation) with a progress bar."""
    results: list[list[int]] = []
    with Progress(
        TextColumn(f"  [dim]{label:<8}[/]"),
        BarColumn(bar_width=30, complete_style="cyan"),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=_console,
        transient=True,
    ) as progress:
        task = progress.add_task("", total=len(texts))
        for text in texts:
            results.append(tokenizer.encode(text, add_special_tokens=True))
            progress.advance(task)
    return results


# ── Rich tables ───────────────────────────────────────────────────────────────

def _stats_table(label: str, word_stats: dict, tok_stats: dict) -> Table:
    """Build a statistics table with word-level and token-level columns."""
    table = Table(
        title=f"[bold]{label}[/]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        min_width=38,
    )
    table.add_column("Stat", style="dim", min_width=8)
    table.add_column("Words", justify="right", min_width=8)
    table.add_column("Tokens", justify="right", min_width=8)
    table.add_column("BPE×", justify="right", min_width=6)

    rows = [
        ("min",  "min"),
        ("mean", "mean"),
        ("std",  "std"),
        ("p50",  "p50"),
        ("p60",  "p60"),
        ("p75",  "p75"),
        ("p90",  "p90"),
        ("p95",  "p95"),
        ("p99",  "p99"),
        ("max",  "max"),
    ]

    for row_label, key in rows:
        w = word_stats[key]
        t = tok_stats[key]
        ratio = f"{t / w:.2f}" if isinstance(w, (int, float)) and w > 0 else "—"
        w_str = f"{w:.1f}" if isinstance(w, float) else str(w)
        t_str = f"{t:.1f}" if isinstance(t, float) else str(t)
        # Highlight p90/p95/p99 rows
        style = "bold" if key in ("p90", "p95", "p99") else ""
        table.add_row(row_label, w_str, t_str, ratio, style=style)

    return table


def _coverage_table(
    input_lens: np.ndarray,
    output_lens: np.ndarray | None,
    current_input_max: int,
    current_output_max: int,
) -> Table:
    """Percentage of samples that fit within each token threshold."""
    n_in = len(input_lens)
    n_out = len(output_lens) if output_lens is not None else 0

    table = Table(
        title="[bold]Coverage by Token Threshold[/]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Threshold", justify="right", style="dim", min_width=10)
    table.add_column("Input fit %", justify="right", min_width=11)
    table.add_column("Input trunc %", justify="right", min_width=13)
    if output_lens is not None:
        table.add_column("Output fit %", justify="right", min_width=12)
        table.add_column("Output trunc %", justify="right", min_width=14)

    # Determine which thresholds to show: those in range [p10 of input .. max input]
    p10_in = int(np.percentile(input_lens, 10))
    visible = [t for t in _COVERAGE_THRESHOLDS if t >= p10_in]
    if not visible:
        visible = _COVERAGE_THRESHOLDS

    for thresh in visible:
        in_fit = float((input_lens <= thresh).sum()) / n_in * 100
        in_trunc = 100.0 - in_fit

        # Style: bold + green if this is the current configured max
        is_current_in = (thresh == current_input_max)
        is_current_out = (thresh == current_output_max) if output_lens is not None else False
        row_style = "bold green" if (is_current_in or is_current_out) else ""

        in_fit_str   = f"{in_fit:5.1f}%"
        in_trunc_str = f"{in_trunc:5.1f}%"

        if in_trunc < 1.0:
            in_trunc_str = f"[green]{in_trunc_str}[/]"
        elif in_trunc < 10.0:
            in_trunc_str = f"[yellow]{in_trunc_str}[/]"
        else:
            in_trunc_str = f"[red]{in_trunc_str}[/]"

        thresh_label = f"{'→ ' if is_current_in else '  '}{thresh}"

        if output_lens is not None:
            out_fit = float((output_lens <= thresh).sum()) / n_out * 100
            out_trunc = 100.0 - out_fit
            out_trunc_str = f"{out_trunc:5.1f}%"
            if out_trunc < 1.0:
                out_trunc_str = f"[green]{out_trunc_str}[/]"
            elif out_trunc < 10.0:
                out_trunc_str = f"[yellow]{out_trunc_str}[/]"
            else:
                out_trunc_str = f"[red]{out_trunc_str}[/]"
            out_thresh_label = f"{'→ ' if is_current_out else '  '}{thresh}"
            table.add_row(
                thresh_label,
                f"{in_fit:5.1f}%", in_trunc_str,
                f"{out_fit:5.1f}%", out_trunc_str,
                style=row_style,
            )
        else:
            table.add_row(thresh_label, f"{in_fit:5.1f}%", in_trunc_str, style=row_style)

    return table


def _unk_panel(
    input_seqs: list[list[int]],
    output_seqs: list[list[int]] | None,
    unk_id: int,
) -> Panel:
    """Unknown token analysis."""
    def _unk_rate(seqs: list[list[int]]) -> tuple[int, int, float]:
        total = sum(len(s) for s in seqs)
        unks  = sum(s.count(unk_id) for s in seqs)
        return unks, total, (unks / total * 100) if total > 0 else 0.0

    in_unks, in_total, in_rate = _unk_rate(input_seqs)

    lines = [f"  [dim]unk_id[/]  = {unk_id}"]
    lines.append("")

    color = "green" if in_rate < 0.1 else ("yellow" if in_rate < 1.0 else "red")
    lines.append(
        f"  [dim]input  :[/]  {in_unks:,} unk / {in_total:,} tokens  "
        f"→ [{color}]{in_rate:.4f}%[/]"
    )

    if output_seqs is not None:
        out_unks, out_total, out_rate = _unk_rate(output_seqs)
        color = "green" if out_rate < 0.1 else ("yellow" if out_rate < 1.0 else "red")
        lines.append(
            f"  [dim]output :[/]  {out_unks:,} unk / {out_total:,} tokens  "
            f"→ [{color}]{out_rate:.4f}%[/]"
        )

    lines.append("")
    lines.append("  [dim]< 0.1% is excellent · 0.1–1% is acceptable · > 1% suggests tokenizer mismatch[/]")

    return Panel("\n".join(lines), title="[bold]Unknown Token Rate[/]", border_style="dim")


def _recommend_panel(
    input_stats: dict,
    output_stats: dict | None,
    input_lens: np.ndarray,
    output_lens: np.ndarray | None,
    cfg,
) -> Panel:
    """Compute and display config recommendations."""
    arch = cfg.model.architecture

    def _ceil_to(value: int, multiple: int) -> int:
        return math.ceil(value / multiple) * multiple

    # Input: p90 (conservative) and p95 (balanced)
    inp_p90 = _ceil_to(input_stats["p90"], 64)
    inp_p95 = _ceil_to(input_stats["p95"], 64)

    # Output: p99
    if output_stats is not None:
        out_p99 = _ceil_to(output_stats["p99"], 16)
    else:
        out_p99 = None

    # max_seq_len per architecture
    def _max_seq(inp: int, out: int | None) -> int:
        if arch == "decoder_only":
            return inp + (out or 0)
        return max(inp, out or 0)  # encoder_decoder and encoder_only

    seq_p90 = _max_seq(inp_p90, out_p99)
    seq_p95 = _max_seq(inp_p95, out_p99)

    # Coverage at recommended values
    n_in = len(input_lens)
    cov_p90 = float((input_lens <= inp_p90).sum()) / n_in * 100
    cov_p95 = float((input_lens <= inp_p95).sum()) / n_in * 100

    current_in  = cfg.data.max_input_len
    current_out = cfg.data.max_output_len
    current_seq = cfg.model.max_seq_len
    current_cov = float((input_lens <= current_in).sum()) / n_in * 100

    lines = []

    lines.append(f"  [dim]Architecture :[/] [cyan]{arch}[/]")
    lines.append("")

    # Primary recommendation (p95)
    lines.append("  [bold]Recommended — balanced (p95 input / p99 output):[/]")
    lines.append(f"    data.max_input_len   [cyan]{inp_p95:>5}[/]   covers [green]{cov_p95:.1f}%[/] of inputs")
    if out_p99 is not None:
        n_out = len(output_lens)
        cov_out = float((output_lens <= out_p99).sum()) / n_out * 100
        lines.append(f"    data.max_output_len  [cyan]{out_p99:>5}[/]   covers [green]{cov_out:.1f}%[/] of outputs")
    lines.append(f"    model.max_seq_len    [cyan]{seq_p95:>5}[/]   ({arch}: {_seq_formula(arch)})")
    lines.append("")

    # Conservative recommendation (p90)
    lines.append("  [bold]Conservative — memory-efficient (p90 input):[/]")
    lines.append(f"    data.max_input_len   [cyan]{inp_p90:>5}[/]   covers [yellow]{cov_p90:.1f}%[/] of inputs")
    if out_p99 is not None:
        lines.append(f"    data.max_output_len  [cyan]{out_p99:>5}[/]   (same as above)")
    lines.append(f"    model.max_seq_len    [cyan]{seq_p90:>5}[/]")
    lines.append("")

    # Current values + status
    lines.append("  [bold]Current config:[/]")
    cov_color = "green" if current_cov >= 95 else ("yellow" if current_cov >= 85 else "red")
    lines.append(
        f"    data.max_input_len   [dim]{current_in:>5}[/]   "
        f"covers [{cov_color}]{current_cov:.1f}%[/] of inputs"
    )
    if output_lens is not None:
        n_out = len(output_lens)
        cov_out_cur = float((output_lens <= current_out).sum()) / n_out * 100
        out_color = "green" if cov_out_cur >= 99 else ("yellow" if cov_out_cur >= 95 else "red")
        lines.append(
            f"    data.max_output_len  [dim]{current_out:>5}[/]   "
            f"covers [{out_color}]{cov_out_cur:.1f}%[/] of outputs"
        )
    lines.append(f"    model.max_seq_len    [dim]{current_seq:>5}[/]")

    # Consistency check
    lines.append("")
    ok = True
    if arch == "decoder_only":
        if current_in + current_out > current_seq:
            lines.append(
                f"  [bold red]WARNING:[/] max_input_len + max_output_len = "
                f"{current_in + current_out} > max_seq_len = {current_seq}"
            )
            ok = False
    else:
        if current_in > current_seq:
            lines.append(
                f"  [bold red]WARNING:[/] max_input_len {current_in} > max_seq_len {current_seq}"
            )
            ok = False
        if current_out > current_seq:
            lines.append(
                f"  [bold red]WARNING:[/] max_output_len {current_out} > max_seq_len {current_seq}"
            )
            ok = False
    if ok:
        lines.append("  [green]Config is self-consistent (no PE overflow risk).[/]")

    return Panel("\n".join(lines), title="[bold]Recommendations[/]", border_style="cyan")


def _seq_formula(arch: str) -> str:
    if arch == "decoder_only":
        return "max_input_len + max_output_len"
    return "max(max_input_len, max_output_len)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze dataset token-length distributions and recommend config values."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Tokenize a random subset of N samples for a quick check.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    _console.print()
    _console.print(Rule("[bold]Transformer — Dataset Diagnostic[/]", style="cyan"))
    _console.print()

    # Load tokenizer
    with _console.status("[dim]Loading tokenizer...[/]"):
        tokenizer = SentencePieceTokenizer.load(cfg.tokenizer.model_path)

    # Load raw (preprocessed) texts
    with _console.status(f"[dim]Loading {cfg.data.data_format} data from {cfg.data.data_path} ...[/]"):
        texts_a, texts_b = _load_raw_texts(
            cfg.data.data_path,
            cfg.data.data_format,
            cfg.data.input_col,
            cfg.data.output_col,
            cfg.data.preprocessing,
        )

    n_total = len(texts_a)

    # Optional sampling
    sampled = False
    if args.sample and args.sample < n_total:
        rng = random.Random(42)
        indices = rng.sample(range(n_total), args.sample)
        texts_a = [texts_a[i] for i in indices]
        texts_b = [texts_b[i] for i in indices] if texts_b is not None else None
        sampled = True

    n_used = len(texts_a)

    # Header info
    sample_note = f"  (sampled {n_used:,} / {n_total:,})" if sampled else f"  (all {n_used:,} samples)"
    _console.print(f"  [dim]dataset    :[/] {cfg.data.data_path}  [dim]{cfg.data.data_format}[/]")
    _console.print(f"  [dim]samples    :[/] {n_used:,}{sample_note}")
    _console.print(f"  [dim]tokenizer  :[/] {cfg.tokenizer.model_path}  vocab={tokenizer.vocab_size:,}")
    _console.print(f"  [dim]preprocess :[/] {cfg.data.preprocessing}")
    _console.print(f"  [dim]architecture:[/] {cfg.model.architecture}")
    _console.print()

    # Tokenize (no truncation)
    _console.print("  [dim]Tokenizing...[/]")
    input_seqs  = _tokenize_batch(texts_a, tokenizer, "input")
    output_seqs = _tokenize_batch(texts_b, tokenizer, "output") if texts_b is not None else None
    _console.print()

    # Compute lengths
    input_word_lens  = np.array([len(t.split()) for t in texts_a])
    input_tok_lens   = np.array([len(s) for s in input_seqs])

    if texts_b is not None:
        output_word_lens = np.array([len(t.split()) for t in texts_b])
        output_tok_lens  = np.array([len(s) for s in output_seqs])
    else:
        output_word_lens = output_tok_lens = None

    # Statistics
    in_word_stats = _stats(input_word_lens)
    in_tok_stats  = _stats(input_tok_lens)

    out_word_stats = _stats(output_word_lens) if output_word_lens is not None else None
    out_tok_stats  = _stats(output_tok_lens)  if output_tok_lens  is not None else None

    # ── Section 1: Statistics tables ──────────────────────────────────────────
    _console.print(Rule("[bold]Length Statistics[/]", style="dim"))
    _console.print()

    in_table = _stats_table("Input", in_word_stats, in_tok_stats)
    if out_word_stats is not None:
        out_table = _stats_table("Output", out_word_stats, out_tok_stats)
        _console.print(Columns([in_table, out_table], equal=False, expand=False))
    else:
        _console.print(in_table)

    _console.print()

    # ── Section 2: Coverage table ──────────────────────────────────────────────
    _console.print(Rule("[bold]Coverage Analysis[/]", style="dim"))
    _console.print()
    coverage = _coverage_table(
        input_tok_lens,
        output_tok_lens,
        cfg.data.max_input_len,
        cfg.data.max_output_len,
    )
    _console.print(coverage)
    _console.print("  [dim]→  marks current max_input_len / max_output_len in base.yaml[/]")
    _console.print()

    # ── Section 3: Unknown token rate ─────────────────────────────────────────
    _console.print(Rule("[bold]Unknown Token Rate[/]", style="dim"))
    _console.print()
    _console.print(_unk_panel(input_seqs, output_seqs, tokenizer.unk_id))
    _console.print()

    # ── Section 4: Recommendations ────────────────────────────────────────────
    _console.print(Rule("[bold]Recommendations[/]", style="dim"))
    _console.print()
    _console.print(
        _recommend_panel(
            in_tok_stats, out_tok_stats,
            input_tok_lens, output_tok_lens,
            cfg,
        )
    )
    _console.print()


if __name__ == "__main__":
    main()
