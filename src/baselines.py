"""
Baseline recommenders — the models a real evaluation must beat before a
fancier method earns its complexity.

- GlobalMeanBaseline: predict everyone's rating as the global average. The
  RMSE floor; any real model must beat it.
- BiasBaseline: global mean + regularized user bias + item bias. A classic,
  surprisingly strong rating-prediction baseline (Koren, 2009).
- PopularityRecommender: recommend the most-rated movies to everyone. The
  ranking baseline — a personalized model has to beat "just show popular
  movies" to be worth anything.

All expose the same interface as the matrix-factorization model
(`predict(user_id, movie_id)` and `recommend(user_id, n)`), so the same
evaluation harness scores every model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class GlobalMeanBaseline:
    def fit(self, ratings_df: pd.DataFrame) -> "GlobalMeanBaseline":
        self.global_mean = float(ratings_df["rating"].mean())
        self._pop = ratings_df["movieId"].value_counts()
        self._seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        return self.global_mean

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        seen = self._seen.get(user_id, set())
        recs = [m for m in self._pop.index if m not in seen][:n]
        return [(int(m), self.global_mean) for m in recs]


class BiasBaseline:
    """rating ≈ global_mean + user_bias[u] + item_bias[i] (regularized)."""

    def __init__(self, reg: float = 10.0):
        self.reg = reg

    def fit(self, ratings_df: pd.DataFrame) -> "BiasBaseline":
        self.global_mean = float(ratings_df["rating"].mean())
        df = ratings_df.copy()
        df["dev"] = df["rating"] - self.global_mean

        # Regularized item bias, then user bias on the item-adjusted residual.
        item = df.groupby("movieId")["dev"].agg(["sum", "count"])
        self.item_bias = (item["sum"] / (item["count"] + self.reg)).to_dict()

        df["item_b"] = df["movieId"].map(self.item_bias).fillna(0.0)
        df["resid"] = df["dev"] - df["item_b"]
        user = df.groupby("userId")["resid"].agg(["sum", "count"])
        self.user_bias = (user["sum"] / (user["count"] + self.reg)).to_dict()

        self._seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
        # Rank items for recommendation by item bias (user bias is constant per user).
        self._ranked_items = sorted(
            self.item_bias, key=self.item_bias.get, reverse=True
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        pred = (
            self.global_mean
            + self.user_bias.get(user_id, 0.0)
            + self.item_bias.get(movie_id, 0.0)
        )
        return float(np.clip(pred, 0.5, 5.0))

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        seen = self._seen.get(user_id, set())
        recs = [m for m in self._ranked_items if m not in seen][:n]
        return [(int(m), self.predict(user_id, m)) for m in recs]


class PopularityRecommender:
    """Recommends the most-rated movies (non-personalized ranking baseline)."""

    def fit(self, ratings_df: pd.DataFrame) -> "PopularityRecommender":
        self.global_mean = float(ratings_df["rating"].mean())
        self._pop = ratings_df["movieId"].value_counts()
        self._item_mean = ratings_df.groupby("movieId")["rating"].mean().to_dict()
        self._seen = ratings_df.groupby("userId")["movieId"].apply(set).to_dict()
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        return float(self._item_mean.get(movie_id, self.global_mean))

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        seen = self._seen.get(user_id, set())
        recs = [m for m in self._pop.index if m not in seen][:n]
        return [(int(m), float(self._item_mean.get(m, self.global_mean))) for m in recs]
