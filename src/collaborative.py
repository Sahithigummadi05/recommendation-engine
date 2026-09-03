"""
Collaborative filtering via matrix factorization.

Builds a user-by-movie rating matrix, mean-centers each user's ratings (so
we model how much a user likes a movie *relative to their own average*, which
removes the "some users just rate everything high" bias), and factorizes it
with truncated SVD to learn latent user/movie factors. Predicted ratings come
from the low-rank reconstruction plus the user's mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


class MatrixFactorizationRecommender:
    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, ratings_df: pd.DataFrame) -> "MatrixFactorizationRecommender":
        self.user_item = ratings_df.pivot_table(
            index="userId", columns="movieId", values="rating"
        )
        self.user_ids = self.user_item.index
        self.movie_ids = self.user_item.columns
        self.user_means = self.user_item.mean(axis=1)
        self.global_mean = float(ratings_df["rating"].mean())

        centered = self.user_item.sub(self.user_means, axis=0).fillna(0.0)

        # n_components must stay below the smaller matrix dimension.
        n_comp = min(self.n_components, min(centered.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        user_factors = self.svd.fit_transform(centered.values)
        self.reconstruction = user_factors @ self.svd.components_

        self._rated_mask = self.user_item.notna().values
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predicts a user's rating for a movie, clipped to the 0.5-5 scale."""
        if user_id not in self.user_ids:
            return self.global_mean
        u = self.user_ids.get_loc(user_id)
        base = self.user_means.iloc[u]
        if movie_id not in self.movie_ids:
            return float(base)
        m = self.movie_ids.get_loc(movie_id)
        return float(np.clip(self.reconstruction[u, m] + base, 0.5, 5.0))

    def recommend(self, user_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Returns the top-n unrated movies as (movie_id, predicted_rating)."""
        if user_id not in self.user_ids:
            return []
        u = self.user_ids.get_loc(user_id)
        preds = self.reconstruction[u] + self.user_means.iloc[u]
        preds = np.where(self._rated_mask[u], -np.inf, preds)  # hide seen movies
        order = np.argsort(preds)[::-1]
        # Drop the sentinel (already-rated) entries so we never recommend a
        # movie the user has seen, even when n exceeds the unrated count.
        top = [i for i in order if np.isfinite(preds[i])][:n]
        return [(int(self.movie_ids[i]), float(np.clip(preds[i], 0.5, 5.0))) for i in top]
