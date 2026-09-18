"""
Model comparison on a per-user temporal split of MovieLens.

Every model is scored on the same held-out data with the same metrics, so the
matrix-factorization model has to *earn* its complexity against simple
baselines:

  * RMSE                     -- rating-prediction accuracy (lower is better).
  * Precision@k / Recall@k   -- of the top-k recommended, how many were liked.
  * NDCG@k / MAP@k           -- the same, but rewarding relevant items ranked
                                higher (what a ranked feed actually cares about).

Ratings are split per user by time (each user's most recent ratings go to the
test set) so we never train on a user's future to predict their past.

Run:
    python -m src.evaluate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines import BiasBaseline, GlobalMeanBaseline, PopularityRecommender
from src.collaborative import MatrixFactorizationRecommender
from src.data_loader import load_movies, load_ratings
from src.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.recommender import HybridRecommender

TEST_FRACTION = 0.2
K = 10
LIKED_RATING = 4.0


def time_split(ratings: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Per-user temporal split: the newest `test_fraction` of each user's
    ratings become the test set."""
    train_parts, test_parts = [], []
    for _, group in ratings.groupby("userId"):
        group = group.sort_values("timestamp")
        n_test = max(1, int(len(group) * test_fraction))
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])
    return pd.concat(train_parts), pd.concat(test_parts)


def rmse(model, test: pd.DataFrame) -> float:
    preds = np.array([model.predict(u, m) for u, m in zip(test["userId"], test["movieId"])])
    return float(np.sqrt(np.mean((preds - test["rating"].values) ** 2)))


def _ranked_ids(model, user_id: int, k: int) -> list[int]:
    """Top-k recommended movie ids, normalizing across model return types."""
    recs = model.recommend(user_id, n=k)
    if isinstance(recs, pd.DataFrame):
        return recs["movieId"].tolist()
    return [mid for mid, _ in recs]


def ranking_metrics(model, test: pd.DataFrame, k: int = K) -> dict:
    liked = test[test["rating"] >= LIKED_RATING]
    liked_by_user = liked.groupby("userId")["movieId"].apply(set).to_dict()

    p, r, ndcg, ap = [], [], [], []
    for user_id, relevant in liked_by_user.items():
        ranked = _ranked_ids(model, user_id, k)
        if not ranked:
            continue
        p.append(precision_at_k(ranked, relevant, k))
        r.append(recall_at_k(ranked, relevant, k))
        ndcg.append(ndcg_at_k(ranked, relevant, k))
        ap.append(average_precision_at_k(ranked, relevant, k))
    return {
        "P@k": float(np.mean(p)),
        "R@k": float(np.mean(r)),
        "NDCG@k": float(np.mean(ndcg)),
        "MAP@k": float(np.mean(ap)),
    }


def main() -> None:
    ratings = load_ratings()
    movies = load_movies()
    print(f"Loaded {len(ratings)} ratings from {ratings['userId'].nunique()} users.")

    train, test = time_split(ratings)
    print(f"Train: {len(train)} ratings   Test: {len(test)} ratings\n")

    models = {
        "Global mean": GlobalMeanBaseline().fit(train),
        "Bias baseline": BiasBaseline().fit(train),
        "Popularity": PopularityRecommender().fit(train),
        "Matrix factorization": MatrixFactorizationRecommender(n_components=50).fit(train),
        "Hybrid (CF + content)": HybridRecommender(n_components=50).fit(train, movies),
    }

    print(f"{'Model':24s} {'RMSE':>6s} {'P@'+str(K):>7s} {'R@'+str(K):>7s} "
          f"{'NDCG@'+str(K):>8s} {'MAP@'+str(K):>7s}")
    print("-" * 62)
    for name, model in models.items():
        try:
            score = f"{rmse(model, test):6.3f}"
        except Exception:
            score = f"{'--':>6s}"          # ranking-only model, no rating prediction
        rank = ranking_metrics(model, test, k=K)
        print(f"{name:24s} {score} {rank['P@k']:7.3f} {rank['R@k']:7.3f} "
              f"{rank['NDCG@k']:8.3f} {rank['MAP@k']:7.3f}")


if __name__ == "__main__":
    main()
