"""Evaluation metrics for tool retrieval (L2 tier).

All functions take:
    predicted: list[str] — ranked list of predicted tool names
    gold: list[str] — set of ground-truth tool names (order doesn't matter)
"""

from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(predicted: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of gold tools appearing in the top-k predictions."""
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    hits = sum(1 for t in predicted[:k] if t in gold_set)
    return hits / len(gold_set)


def precision_at_k(predicted: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Fraction of top-k predictions that are gold tools."""
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    hits = sum(1 for t in predicted[:k] if t in gold_set)
    return hits / k


def f1_at_k(predicted: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Harmonic mean of Precision@k and Recall@k."""
    precision = precision_at_k(predicted, gold, k)
    recall = recall_at_k(predicted, gold, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ndcg_at_k(predicted: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Binary relevance: 1 if tool is in gold set, 0 otherwise.
    """
    if not gold:
        return 1.0
    gold_set = set(gold)

    # DCG
    dcg = 0.0
    for i, t in enumerate(predicted[:k]):
        if t in gold_set:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because position is 1-indexed

    # Ideal DCG (all gold tools ranked first)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


def pass_rate_at_k(predicted: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1.0 if ALL gold tools appear in top-k, else 0.0."""
    if not gold:
        return 1.0
    gold_set = set(gold)
    pred_set = set(predicted[:k])
    return 1.0 if gold_set.issubset(pred_set) else 0.0


def mrr(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first hit."""
    gold_set = set(gold)
    for i, t in enumerate(predicted):
        if t in gold_set:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    results: list[tuple[list[str], list[str]]],
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Compute all L2 metrics over a list of (predicted, gold) pairs.

    Returns:
        Dict with keys like "Recall@5", "NDCG@10", "PassRate@5", etc.
    """
    n = len(results)
    if n == 0:
        return {}

    metrics: dict[str, float] = {}
    for k in ks:
        pr_sum = sum(precision_at_k(pred, gold, k) for pred, gold in results)
        r_sum = sum(recall_at_k(pred, gold, k) for pred, gold in results)
        f_sum = sum(f1_at_k(pred, gold, k) for pred, gold in results)
        n_sum = sum(ndcg_at_k(pred, gold, k) for pred, gold in results)
        p_sum = sum(pass_rate_at_k(pred, gold, k) for pred, gold in results)
        metrics[f"Precision@{k}"] = pr_sum / n
        metrics[f"Recall@{k}"] = r_sum / n
        metrics[f"F1@{k}"] = f_sum / n
        metrics[f"NDCG@{k}"] = n_sum / n
        metrics[f"PassRate@{k}"] = p_sum / n
        # Completeness@k (also Sufficiency@k / TGR Pass Rate): binary per-query,
        # 1 iff ALL required tools are in top-k. Same value as PassRate@k.
        metrics[f"Completeness@{k}"] = p_sum / n

    metrics["MRR"] = sum(mrr(pred, gold) for pred, gold in results) / n
    return metrics
