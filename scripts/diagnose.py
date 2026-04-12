"""Dataset and tokenizer diagnostic report.

Analyzes token-length distributions and vocabulary statistics for the configured
dataset and tokenizer, then recommends optimal config values.

Usage::

    uv run python scripts/diagnose.py --config configs/base.yaml
    uv run python scripts/diagnose.py --config configs/base.yaml --mode data
    uv run python scripts/diagnose.py --config configs/base.yaml --mode tokenizer
    uv run python scripts/diagnose.py --config configs/base.yaml --mode all --sample 50000
"""
from __future__ import annotations

import argparse
import collections
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

_PERCENTILES        = [50, 60, 75, 90, 95, 99]
_COVERAGE_THRESHOLDS = [32, 64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 2048]
_VOCAB_RANK_THRESHOLDS = [100, 500, 1_000, 2_000, 5_000, 10_000, 20_000]


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats(lengths: np.ndarray) -> dict:
    """Descriptive statistics: min/max/mean/std and key percentiles."""
    pct = np.percentile(lengths, _PERCENTILES)
    return {
        "min":  int(lengths.min()),
        "max":  int(lengths.max()),
        "mean": float(lengths.mean()),
        "std":  float(lengths.std()),
        **{f"p{p}": int(v) for p, v in zip(_PERCENTILES, pct)},
    }


def _tokenize_batch(texts: list[str], tokenizer: SentencePieceTokenizer, label: str) -> list[list[int]]:
    """Tokenize all texts without truncation, with a rich progress bar."""
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


# ─────────────────────────────────────────────────────────────────────────────
#  Data section helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stats_table(label: str, word_stats: dict, tok_stats: dict) -> Table:
    """Rich Table comparing word-level vs token-level statistics."""
    table = Table(
        title=f"[bold]{label}[/]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        min_width=38,
    )
    table.add_column("Stat",   style="dim",   min_width=8)
    table.add_column("Words",  justify="right", min_width=8)
    table.add_column("Tokens", justify="right", min_width=8)
    table.add_column("BPE×",   justify="right", min_width=6)

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
    n_in  = len(input_lens)
    n_out = len(output_lens) if output_lens is not None else 0

    table = Table(
        title="[bold]Coverage by Token Threshold[/]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Threshold",     justify="right", style="dim", min_width=10)
    table.add_column("Input fit %",   justify="right", min_width=11)
    table.add_column("Input trunc %", justify="right", min_width=13)
    if output_lens is not None:
        table.add_column("Output fit %",   justify="right", min_width=12)
        table.add_column("Output trunc %", justify="right", min_width=14)

    p10_in  = int(np.percentile(input_lens, 10))
    visible = [t for t in _COVERAGE_THRESHOLDS if t >= p10_in] or _COVERAGE_THRESHOLDS

    for thresh in visible:
        in_fit   = float((input_lens <= thresh).sum()) / n_in * 100
        in_trunc = 100.0 - in_fit

        is_cur_in  = thresh == current_input_max
        is_cur_out = thresh == current_output_max if output_lens is not None else False
        row_style  = "bold green" if (is_cur_in or is_cur_out) else ""

        def _trunc_color(v: float) -> str:
            if v < 1.0:   return f"[green]{v:5.1f}%[/]"
            if v < 10.0:  return f"[yellow]{v:5.1f}%[/]"
            return              f"[red]{v:5.1f}%[/]"

        thresh_label = f"{'→ ' if is_cur_in else '  '}{thresh}"

        if output_lens is not None:
            out_fit   = float((output_lens <= thresh).sum()) / n_out * 100
            out_trunc = 100.0 - out_fit
            table.add_row(
                thresh_label,
                f"{in_fit:5.1f}%",  _trunc_color(in_trunc),
                f"{out_fit:5.1f}%", _trunc_color(out_trunc),
                style=row_style,
            )
        else:
            table.add_row(thresh_label, f"{in_fit:5.1f}%", _trunc_color(in_trunc), style=row_style)

    return table


def _unk_panel(
    input_seqs: list[list[int]],
    output_seqs: list[list[int]] | None,
    unk_id: int,
) -> Panel:
    """Unknown token rate for input and output."""
    def _rate(seqs: list[list[int]]) -> tuple[int, int, float]:
        total = sum(len(s) for s in seqs)
        unks  = sum(s.count(unk_id) for s in seqs)
        return unks, total, (unks / total * 100) if total > 0 else 0.0

    in_unks, in_total, in_rate = _rate(input_seqs)
    lines = [f"  [dim]unk_id[/]  = {unk_id}", ""]

    def _color(r: float) -> str:
        return "green" if r < 0.1 else ("yellow" if r < 1.0 else "red")

    lines.append(
        f"  [dim]input  :[/]  {in_unks:,} unk / {in_total:,} tokens  "
        f"→ [{_color(in_rate)}]{in_rate:.4f}%[/]"
    )
    if output_seqs is not None:
        out_unks, out_total, out_rate = _rate(output_seqs)
        lines.append(
            f"  [dim]output :[/]  {out_unks:,} unk / {out_total:,} tokens  "
            f"→ [{_color(out_rate)}]{out_rate:.4f}%[/]"
        )
    lines += [
        "",
        "  [dim]< 0.1% excellent  ·  0.1–1% acceptable  ·  > 1% tokenizer mismatch[/]",
    ]
    return Panel("\n".join(lines), title="[bold]Unknown Token Rate[/]", border_style="dim")


def _recommend_panel(
    input_stats: dict,
    output_stats: dict | None,
    input_lens: np.ndarray,
    output_lens: np.ndarray | None,
    cfg,
) -> Panel:
    """Data-driven config recommendations for max_input_len / max_output_len / max_seq_len."""
    arch = cfg.model.architecture

    def _ceil(value: int, multiple: int) -> int:
        return math.ceil(value / multiple) * multiple

    inp_p90 = _ceil(input_stats["p90"], 64)
    inp_p95 = _ceil(input_stats["p95"], 64)
    out_p99 = _ceil(output_stats["p99"], 16) if output_stats else None

    def _max_seq(inp: int, out: int | None) -> int:
        return inp + (out or 0) if arch == "decoder_only" else max(inp, out or 0)

    seq_p90 = _max_seq(inp_p90, out_p99)
    seq_p95 = _max_seq(inp_p95, out_p99)

    n_in     = len(input_lens)
    cov_p90  = float((input_lens <= inp_p90).sum()) / n_in * 100
    cov_p95  = float((input_lens <= inp_p95).sum()) / n_in * 100
    cur_cov  = float((input_lens <= cfg.data.max_input_len).sum()) / n_in * 100

    formula  = "max_input_len + max_output_len" if arch == "decoder_only" else "max(max_input_len, max_output_len)"
    lines    = []

    lines.append("  [bold]Recommended — balanced (p95 input / p99 output):[/]")
    lines.append(f"    data.max_input_len   [cyan]{inp_p95:>5}[/]   covers [green]{cov_p95:.1f}%[/] of inputs")
    if out_p99 is not None:
        n_out   = len(output_lens)
        cov_out = float((output_lens <= out_p99).sum()) / n_out * 100
        lines.append(f"    data.max_output_len  [cyan]{out_p99:>5}[/]   covers [green]{cov_out:.1f}%[/] of outputs")
    lines.append(f"    model.max_seq_len    [cyan]{seq_p95:>5}[/]   ({arch}: {formula})")
    lines.append("")

    lines.append("  [bold]Conservative — memory-efficient (p90 input):[/]")
    lines.append(f"    data.max_input_len   [cyan]{inp_p90:>5}[/]   covers [yellow]{cov_p90:.1f}%[/] of inputs")
    if out_p99 is not None:
        lines.append(f"    data.max_output_len  [cyan]{out_p99:>5}[/]   (same as above)")
    lines.append(f"    model.max_seq_len    [cyan]{seq_p90:>5}[/]")
    lines.append("")

    cur_in  = cfg.data.max_input_len
    cur_out = cfg.data.max_output_len
    cur_seq = cfg.model.max_seq_len
    lines.append("  [bold]Current config:[/]")
    cov_col = "green" if cur_cov >= 95 else ("yellow" if cur_cov >= 85 else "red")
    lines.append(f"    data.max_input_len   [dim]{cur_in:>5}[/]   covers [{cov_col}]{cur_cov:.1f}%[/] of inputs")
    if output_lens is not None:
        n_out       = len(output_lens)
        cov_out_cur = float((output_lens <= cur_out).sum()) / n_out * 100
        oc          = "green" if cov_out_cur >= 99 else ("yellow" if cov_out_cur >= 95 else "red")
        lines.append(f"    data.max_output_len  [dim]{cur_out:>5}[/]   covers [{oc}]{cov_out_cur:.1f}%[/] of outputs")
    lines.append(f"    model.max_seq_len    [dim]{cur_seq:>5}[/]")
    lines.append("")

    # Consistency check
    ok = True
    if arch == "decoder_only" and cur_in + cur_out > cur_seq:
        lines.append(
            f"  [bold red]WARNING:[/] max_input_len + max_output_len = "
            f"{cur_in + cur_out} > max_seq_len = {cur_seq}"
        )
        ok = False
    elif arch != "decoder_only":
        if cur_in > cur_seq:
            lines.append(f"  [bold red]WARNING:[/] max_input_len {cur_in} > max_seq_len {cur_seq}")
            ok = False
        if cur_out > cur_seq:
            lines.append(f"  [bold red]WARNING:[/] max_output_len {cur_out} > max_seq_len {cur_seq}")
            ok = False
    if ok:
        lines.append("  [green]Config is self-consistent (no PE overflow risk).[/]")

    return Panel("\n".join(lines), title="[bold]Data Recommendations[/]", border_style="cyan")


# ─────────────────────────────────────────────────────────────────────────────
#  Tokenizer section helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_counter(all_seqs: list[list[int]]) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for seq in all_seqs:
        counter.update(seq)
    return counter


def _vocab_overview_panel(
    all_seqs: list[list[int]],
    texts_a: list[str],
    tokenizer: SentencePieceTokenizer,
    cfg,
    counter: collections.Counter,
) -> Panel:
    """High-level vocabulary and corpus statistics."""
    vocab_cfg   = cfg.tokenizer.vocab_size
    vocab_model = tokenizer.vocab_size
    total_tokens = sum(len(s) for s in all_seqs)
    unique_used  = len(counter)
    utilization  = unique_used / vocab_model * 100

    total_chars  = sum(len(t) for t in texts_a)
    chars_per_tok = total_chars / total_tokens if total_tokens > 0 else 0.0

    total_words  = sum(len(t.split()) for t in texts_a)
    fertility    = total_tokens / total_words if total_words > 0 else 0.0

    if fertility < 1.1:
        fert_label = "[yellow]oversized[/]"
        fert_note  = "many whole-word tokens; rare subwords may be missing"
    elif fertility < 1.5:
        fert_label = "[green]adequate[/]"
        fert_note  = "vocab well-sized for this corpus"
    elif fertility < 2.5:
        fert_label = "[green]ideal[/]"
        fert_note  = "good subword granularity"
    else:
        fert_label = "[red]too small[/]"
        fert_note  = "vocab likely too small; many words split into many pieces"

    vocab_match = "[green]✓ consistent[/]" if vocab_cfg == vocab_model else f"[red]✗ mismatch (config={vocab_cfg})[/]"

    lines = [
        f"  [dim]vocab_size (config)      :[/] {vocab_cfg:,}",
        f"  [dim]vocab_size (model file)  :[/] {vocab_model:,}  {vocab_match}",
        f"  [dim]unique tokens in dataset :[/] {unique_used:,}  [dim]({utilization:.1f}% of vocab used)[/]",
        f"  [dim]total tokens in dataset  :[/] {total_tokens:,}",
        "",
        f"  [dim]avg chars  / token       :[/] {chars_per_tok:.2f}  [dim](higher = better compression)[/]",
        f"  [dim]avg tokens / word        :[/] {fertility:.2f}  [dim](BPE fertility)[/]",
        f"  [dim]fertility assessment     :[/] {fert_label}  [dim]{fert_note}[/]",
        "",
        "  [dim]Fertility guide:[/]",
        "  [dim]  < 1.1  → vocab oversized    1.1–1.5 → adequate    1.5–2.5 → ideal    > 2.5 → too small[/]",
    ]
    return Panel("\n".join(lines), title="[bold]Vocabulary Overview[/]", border_style="dim")


def _token_freq_table(
    counter: collections.Counter,
    tokenizer: SentencePieceTokenizer,
    total_tokens: int,
    top_n: int = 25,
) -> Table:
    """Top-N most frequent tokens with cumulative coverage."""
    special = {tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id, tokenizer.eos_id}

    table = Table(
        title=f"[bold]Top-{top_n} Most Frequent Tokens[/]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Rank",   justify="right",  style="dim", min_width=5)
    table.add_column("Piece",  justify="left",   min_width=16)
    table.add_column("ID",     justify="right",  style="dim", min_width=6)
    table.add_column("Count",  justify="right",  min_width=10)
    table.add_column("Freq %", justify="right",  min_width=8)
    table.add_column("Cumul %",justify="right",  min_width=8)

    cumul = 0.0
    for rank, (token_id, count) in enumerate(counter.most_common(top_n), start=1):
        freq   = count / total_tokens * 100
        cumul += freq
        try:
            piece = tokenizer.id_to_piece(token_id)
        except Exception:
            piece = f"<id:{token_id}>"

        is_special = token_id in special
        piece_str  = f"[dim]{piece}[/]" if is_special else piece

        table.add_row(
            str(rank),
            piece_str,
            str(token_id),
            f"{count:,}",
            f"{freq:.3f}%",
            f"{cumul:.2f}%",
        )
    return table


def _vocab_coverage_table(
    counter: collections.Counter,
    vocab_size: int,
    total_tokens: int,
) -> Table:
    """How much of the corpus is covered by the top-N most frequent token types."""
    table = Table(
        title="[bold]Vocabulary Coverage by Token Rank[/]",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Top-N tokens",       justify="right", style="dim", min_width=14)
    table.add_column("Token coverage",     justify="right", min_width=16)
    table.add_column("Vocab utilisation",  justify="right", min_width=18)

    thresholds = [t for t in _VOCAB_RANK_THRESHOLDS if t < vocab_size] + [vocab_size]
    most_common = counter.most_common()

    cumul_tokens = 0
    rank = 0
    threshold_idx = 0

    for token_id, count in most_common:
        rank += 1
        cumul_tokens += count
        while threshold_idx < len(thresholds) and rank >= thresholds[threshold_idx]:
            n    = thresholds[threshold_idx]
            cov  = cumul_tokens / total_tokens * 100
            util = n / vocab_size * 100
            table.add_row(
                f"{n:,}",
                f"{cov:.2f}%",
                f"{util:.1f}%",
            )
            threshold_idx += 1
        if threshold_idx >= len(thresholds):
            break

    # If vocab is larger than dataset's unique tokens, fill remaining thresholds
    while threshold_idx < len(thresholds):
        n    = thresholds[threshold_idx]
        cov  = 100.0
        util = n / vocab_size * 100
        table.add_row(f"{n:,}", f"{cov:.2f}%", f"{util:.1f}%")
        threshold_idx += 1

    return table


def _tokenizer_recommend_panel(
    fertility: float,
    total_tokens: int,
    estimated_total_tokens: int,
    unk_rate: float,
    cfg,
) -> Panel:
    """Tokenizer training config recommendations."""
    current_vocab = cfg.tokenizer.vocab_size
    current_cov   = cfg.tokenizer.character_coverage
    current_type  = cfg.tokenizer.model_type

    # Use estimated full-corpus token count for vocab heuristic so that
    # --sample N does not bias the recommendation.
    eval_tokens = estimated_total_tokens

    # Vocab size heuristic based on corpus scale
    if eval_tokens < 5_000_000:
        suggested_vocab, scale_note = 8_000,  "< 5M tokens"
    elif eval_tokens < 30_000_000:
        suggested_vocab, scale_note = 16_000, "5M–30M tokens"
    elif eval_tokens < 200_000_000:
        suggested_vocab, scale_note = 32_000, "30M–200M tokens"
    elif eval_tokens < 1_000_000_000:
        suggested_vocab, scale_note = 64_000, "200M–1B tokens"
    else:
        suggested_vocab, scale_note = 96_000, "> 1B tokens"

    # Fertility-based override
    if fertility > 2.5:
        suggested_vocab = max(suggested_vocab, current_vocab * 2)
        fert_advice = f"[red]fertility={fertility:.2f} → vocab too small, consider doubling[/]"
    elif fertility < 1.1:
        suggested_vocab = max(8_000, min(suggested_vocab, current_vocab // 2))
        fert_advice = f"[yellow]fertility={fertility:.2f} → vocab may be oversized[/]"
    else:
        fert_advice = f"[green]fertility={fertility:.2f} → vocab size is appropriate[/]"

    vocab_ok  = abs(current_vocab - suggested_vocab) / max(suggested_vocab, 1) < 0.3
    vocab_sym = "[green]✓[/]" if vocab_ok else "[yellow]→[/]"

    # character_coverage advice
    if unk_rate > 1.0:
        cov_advice = "[red]unk > 1% — increase to 0.9998 or 0.9999[/]"
        cov_sym    = "[red]✗[/]"
    elif unk_rate > 0.1:
        cov_advice = "[yellow]unk 0.1–1% — current value acceptable; try 0.9998 if retraining[/]"
        cov_sym    = "[yellow]~[/]"
    else:
        cov_advice = "[green]unk < 0.1% — excellent coverage[/]"
        cov_sym    = "[green]✓[/]"

    sampled_note = (
        f"  [dim](sampled: {total_tokens:,} → estimated full corpus: {estimated_total_tokens:,})[/]"
        if estimated_total_tokens != total_tokens else ""
    )
    lines = [
        f"  [dim]Dataset scale  :[/] {estimated_total_tokens:,} tokens  [dim]({scale_note})[/]{sampled_note}",
        f"  [dim]Fertility      :[/] {fert_advice}",
        "",
        "  [bold]tokenizer.vocab_size[/]",
        f"    current   : {current_vocab:,}",
        f"    suggested : {suggested_vocab:,}  {vocab_sym}",
        f"    {('[green]current value appropriate[/]' if vocab_ok else '[yellow]consider retraining at suggested size[/]')}",
        "",
        "  [bold]tokenizer.character_coverage[/]",
        f"    current   : {current_cov}",
        f"    advice    : {cov_advice}  {cov_sym}",
        "",
        "  [bold]tokenizer.model_type[/]",
        f"    current   : {current_type}",
        f"    advice    : [green]bpe is the standard choice; unigram is an alternative if fertility is unstable[/]",
        "",
        "  [dim]To retrain tokenizer:[/]  uv run python scripts/build_tokenizer.py --config configs/base.yaml",
    ]
    return Panel("\n".join(lines), title="[bold]Tokenizer Recommendations[/]", border_style="cyan")


# ─────────────────────────────────────────────────────────────────────────────
#  Section runners
# ─────────────────────────────────────────────────────────────────────────────

def _run_data_section(
    cfg,
    input_lens: np.ndarray,
    output_lens: np.ndarray | None,
    input_word_lens: np.ndarray,
    output_word_lens: np.ndarray | None,
    input_seqs: list[list[int]],
    output_seqs: list[list[int]] | None,
    tokenizer: SentencePieceTokenizer,
) -> None:
    in_word_stats = _stats(input_word_lens)
    in_tok_stats  = _stats(input_lens)
    out_word_stats = _stats(output_word_lens) if output_word_lens is not None else None
    out_tok_stats  = _stats(output_lens)      if output_lens      is not None else None

    _console.print(Rule("[bold]Length Statistics[/]", style="dim"))
    _console.print()
    in_table = _stats_table("Input", in_word_stats, in_tok_stats)
    if out_word_stats is not None:
        out_table = _stats_table("Output", out_word_stats, out_tok_stats)
        _console.print(Columns([in_table, out_table], equal=False, expand=False))
    else:
        _console.print(in_table)
    _console.print()

    _console.print(Rule("[bold]Coverage Analysis[/]", style="dim"))
    _console.print()
    _console.print(_coverage_table(input_lens, output_lens, cfg.data.max_input_len, cfg.data.max_output_len))
    _console.print("  [dim]→  marks current max_input_len / max_output_len in base.yaml[/]")
    _console.print()

    _console.print(Rule("[bold]Unknown Token Rate[/]", style="dim"))
    _console.print()
    _console.print(_unk_panel(input_seqs, output_seqs, tokenizer.unk_id))
    _console.print()

    _console.print(Rule("[bold]Data Recommendations[/]", style="dim"))
    _console.print()
    _console.print(_recommend_panel(in_tok_stats, out_tok_stats, input_lens, output_lens, cfg))
    _console.print()


def _run_tokenizer_section(
    cfg,
    all_seqs: list[list[int]],
    texts_a: list[str],
    tokenizer: SentencePieceTokenizer,
    n_used: int,
    n_total: int,
) -> None:
    with _console.status("[dim]Building token frequency counter...[/]"):
        counter      = _build_counter(all_seqs)
        total_tokens = sum(counter.values())

    # Scale token count to full corpus when a sample was used
    scale_factor           = n_total / n_used if n_used > 0 else 1.0
    estimated_total_tokens = int(total_tokens * scale_factor)

    # Fertility (tokens / words, BOS/EOS inclusive)
    total_words = sum(len(t.split()) for t in texts_a)
    fertility   = total_tokens / total_words if total_words > 0 else 0.0

    # UNK rate (for recommendation threshold)
    unk_count = counter.get(tokenizer.unk_id, 0)
    unk_rate  = unk_count / total_tokens * 100 if total_tokens > 0 else 0.0

    _console.print(Rule("[bold]Vocabulary Overview[/]", style="dim"))
    _console.print()
    _console.print(_vocab_overview_panel(all_seqs, texts_a, tokenizer, cfg, counter))
    _console.print()

    _console.print(Rule("[bold]Token Frequency Distribution[/]", style="dim"))
    _console.print()
    _console.print(_token_freq_table(counter, tokenizer, total_tokens, top_n=25))
    _console.print()

    _console.print(Rule("[bold]Vocabulary Coverage[/]", style="dim"))
    _console.print()
    _console.print(_vocab_coverage_table(counter, tokenizer.vocab_size, total_tokens))
    _console.print()

    _console.print(Rule("[bold]Tokenizer Recommendations[/]", style="dim"))
    _console.print()
    _console.print(_tokenizer_recommend_panel(fertility, total_tokens, estimated_total_tokens, unk_rate, cfg))
    _console.print()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze dataset and tokenizer to recommend optimal training config values."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--mode", default="all", choices=["all", "data", "tokenizer"],
        help="Which analysis to run: data, tokenizer, or all (default: all)",
    )
    parser.add_argument(
        "--sample", type=int, default=None, metavar="N",
        help="Tokenize only a random subset of N samples for a quick check.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    _console.print()
    _console.print(Rule("[bold]Transformer — Diagnostic Report[/]", style="cyan"))
    _console.print()

    with _console.status("[dim]Loading tokenizer...[/]"):
        tokenizer = SentencePieceTokenizer.load(cfg.tokenizer.model_path)

    with _console.status(f"[dim]Loading {cfg.data.data_format} data...[/]"):
        texts_a, texts_b = _load_raw_texts(
            cfg.data.data_path,
            cfg.data.data_format,
            cfg.data.input_col,
            cfg.data.output_col,
            cfg.data.preprocessing,
        )

    n_total = len(texts_a)
    sampled = False
    if args.sample and args.sample < n_total:
        rng     = random.Random(42)
        indices = rng.sample(range(n_total), args.sample)
        texts_a = [texts_a[i] for i in indices]
        texts_b = [texts_b[i] for i in indices] if texts_b is not None else None
        sampled = True

    n_used      = len(texts_a)
    sample_note = f"  (sampled {n_used:,} / {n_total:,})" if sampled else f"  ({n_used:,} samples)"

    _console.print(f"  [dim]dataset     :[/] {cfg.data.data_path}  [dim][{cfg.data.data_format}][/]")
    _console.print(f"  [dim]samples     :[/] {n_used:,}{sample_note}")
    _console.print(f"  [dim]tokenizer   :[/] {cfg.tokenizer.model_path}  vocab={tokenizer.vocab_size:,}")
    _console.print(f"  [dim]preprocessing:[/] {cfg.data.preprocessing}")
    _console.print(f"  [dim]architecture:[/] {cfg.model.architecture}")
    _console.print(f"  [dim]mode        :[/] {args.mode}")
    _console.print()

    # Tokenize once — shared by both sections
    _console.print("  [dim]Tokenizing...[/]")
    input_seqs  = _tokenize_batch(texts_a, tokenizer, "input")
    output_seqs = _tokenize_batch(texts_b, tokenizer, "output") if texts_b is not None else None
    _console.print()

    input_lens       = np.array([len(s) for s in input_seqs])
    input_word_lens  = np.array([len(t.split()) for t in texts_a])
    output_lens      = np.array([len(s) for s in output_seqs]) if output_seqs else None
    output_word_lens = np.array([len(t.split()) for t in texts_b]) if texts_b else None

    # Combine input + output sequences for tokenizer-level analysis
    all_seqs = input_seqs + (output_seqs or [])

    if args.mode in ("data", "all"):
        _run_data_section(
            cfg,
            input_lens, output_lens,
            input_word_lens, output_word_lens,
            input_seqs, output_seqs,
            tokenizer,
        )

    if args.mode in ("tokenizer", "all"):
        _run_tokenizer_section(cfg, all_seqs, texts_a, tokenizer, n_used, n_total)


if __name__ == "__main__":
    main()
