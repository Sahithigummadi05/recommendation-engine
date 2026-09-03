"""
Evaluates the collaborative-filtering model with two complementary metrics:

  * RMSE  -- how close predicted ratings are to held-out actual ratings
             (rating-prediction accuracy).
  * Precision@k / Recall@k -- of the top-k movies we recommend to each user,
             how many did they actually rate highly in the held-out set
             (top-N recommendation quality, which is what a real product cares
             about more than raw RMSE).

Ratings are split per user by time (each user's most recent ratings go to the
test set) so we never train on a user's future to predict their past.

Run:
    python -m src.evaluate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import load_ratings, load_movies
from src.collaborative import MatrixFactorizationRecommender

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


def rmse(model: MatrixFactorizationRecommender, test: pd.DataFrame) -> float:
    preds = np.array([model.predict(u, m) for u, m in zip(test["userId"], test["movieId"])])
    errors = preds - test["rating"].values
    return float(np.sqrt(np.mean(errors ** 2)))


def precision_recall_at_k(model, test, k: int = K):
    """Averaged over users who have at least one liked movie in the test set."""
    liked = test[test["rating"] >= LIKED_RATING]
    liked_by_user = liked.groupby("userId")["movieId"].apply(set).to_dict()

    precisions, recalls = [], []
    for user_id, relevant in liked_by_user.items():
        recs = [mid for mid, _ in model.recommend(user_id, n=k)]
        if not recs:
            continue
        hits = len(set(recs) & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))

    return float(np.mean(precisions)), float(np.mean(recalls))


def main() -> None:
    ratings = load_ratings()
    load_movies()  # ensures the movies file is present/cached too
    print(f"Loaded {len(ratings)} ratings from {ratings['userId'].nunique()} users.\n")

    train, test = time_split(ratings)
    print(f"Train: {len(train)} ratings   Test: {len(test)} ratings\n")

    model = MatrixFactorizationRecommender(n_components=50).fit(train)

    print(f"RMSE on held-out ratings: {rmse(model, test):.3f}")
    p, r = precision_recall_at_k(model, test, k=K)
    print(f"Precision@{K}: {p:.3f}")
    print(f"Recall@{K}:    {r:.3f}")


if __name__ == "__main__":
    main()
