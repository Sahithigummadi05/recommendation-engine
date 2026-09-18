"""
Unit tests for ranking metrics and baseline recommenders, on small
hand-built data so they run without the MovieLens download.
"""

import numpy as np
import pandas as pd

from src.baselines import BiasBaseline, GlobalMeanBaseline, PopularityRecommender
from src.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_recall_at_k():
    ranked, relevant = [1, 2, 3, 4, 5], {2, 4}
    assert precision_at_k(ranked, relevant, 5) == 2 / 5
    assert recall_at_k(ranked, relevant, 5) == 1.0


def test_average_precision_at_k():
    # hits at ranks 2 and 4 -> (1/2 + 2/4) / 2 = 0.5
    assert abs(average_precision_at_k([1, 2, 3, 4, 5], {2, 4}, 5) - 0.5) < 1e-9


def test_ndcg_rewards_higher_ranks():
    relevant = {1, 2}
    good = ndcg_at_k([1, 2, 3, 4], relevant, 4)   # relevant items on top
    bad = ndcg_at_k([3, 4, 1, 2], relevant, 4)    # relevant items at bottom
    assert good == 1.0
    assert good > bad


def test_empty_relevant_is_zero():
    assert ndcg_at_k([1, 2, 3], set(), 3) == 0.0
    assert average_precision_at_k([1, 2, 3], set(), 3) == 0.0


def _ratings() -> pd.DataFrame:
    rows = [
        (1, 10, 5.0), (1, 11, 4.0), (1, 12, 3.0),
        (2, 10, 4.0), (2, 11, 5.0),
        (3, 10, 5.0), (3, 12, 4.0),
    ]
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


def test_bias_baseline_predicts_in_range_and_excludes_seen():
    model = BiasBaseline(reg=1.0).fit(_ratings())
    pred = model.predict(1, 10)
    assert 0.5 <= pred <= 5.0
    recs = [m for m, _ in model.recommend(1, n=5)]
    assert 10 not in recs and 11 not in recs and 12 not in recs  # user 1 saw all rated


def test_popularity_recommends_most_rated_unseen():
    # Movie 10 is the most-rated (3 ratings); user 2 hasn't seen movie 12.
    model = PopularityRecommender().fit(_ratings())
    recs = [m for m, _ in model.recommend(2, n=5)]
    assert 12 in recs
    assert 10 not in recs and 11 not in recs  # user 2 already saw these


def test_global_mean_predicts_constant():
    model = GlobalMeanBaseline().fit(_ratings())
    assert model.predict(1, 10) == model.predict(2, 11)
