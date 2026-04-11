"""BLEU and ROUGE evaluation utilities."""
from __future__ import annotations

from typing import Dict, List

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute corpus-level BLEU-1 through BLEU-4.

    Args:
        hypotheses: List of generated sentences (space-tokenized strings).
        references: List of reference sentences (space-tokenized strings).

    Returns:
        Dict with keys "bleu1", "bleu2", "bleu3", "bleu4".
    """
    assert len(hypotheses) == len(references)
    smoother = SmoothingFunction().method1

    hyp_tok = [h.split() for h in hypotheses]
    ref_tok = [[r.split()] for r in references]

    results = {}
    for n in range(1, 5):
        weights = tuple(1.0 / n for _ in range(n)) + tuple(0.0 for _ in range(4 - n))
        results[f"bleu{n}"] = corpus_bleu(
            ref_tok, hyp_tok, weights=weights, smoothing_function=smoother
        )
    return results


def compute_rouge(
    hypotheses: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        hypotheses: List of generated sentences.
        references: List of reference sentences.

    Returns:
        Dict with keys "rouge1", "rouge2", "rougeL".
    """
    assert len(hypotheses) == len(references)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

    totals: Dict[str, float] = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        totals["rouge1"] += scores["rouge1"].fmeasure
        totals["rouge2"] += scores["rouge2"].fmeasure
        totals["rougeL"] += scores["rougeL"].fmeasure

    n = len(hypotheses)
    return {k: v / n for k, v in totals.items()}
