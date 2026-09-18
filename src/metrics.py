"""
Ranking metrics for top-N recommendation evaluation.

Precision@k / Recall@k answer "how many of the k we showed were relevant,"
but they ignore *order*. NDCG@k and MAP@k reward putting relevant items
higher in the list, which is what actually matters in a ranked feed.
All operate on a ranked list of item ids and a set of relevant (liked) ids.
"""

from __future__ import annotations

import numpy as np


def precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(ranked[:k]) & relevant) / k


def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def average_precision_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """Mean of precision@i taken at each rank i (<=k) that holds a hit."""
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def ndcg_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """Binary-relevance NDCG@k (gain 1 for a liked item, discounted by log rank)."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / np.log2(i + 1)
        for i, item in enumerate(ranked[:k], start=1)
        if item in relevant
    )
    ideal = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return float(dcg / ideal) if ideal else 0.0
