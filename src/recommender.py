"""
Hybrid recommender.

Combines the collaborative-filtering signal (what similar users liked) with
the content-based signal (genre similarity to what this user already likes).
Collaborative filtering is the stronger signal when a user has rating history,
so it gets the higher weight; the content term mainly helps break ties and
nudges recommendations toward genres the user actually enjoys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.collaborative import MatrixFactorizationRecommender
from src.content_based import ContentBasedRecommender

CF_WEIGHT = 0.7
CONTENT_WEIGHT = 0.3
LIKED_RATING = 4.0  # a rating at/above this counts as "liked"


class HybridRecommender:
    def __init__(self, n_components: int = 50):
        self.cf = MatrixFactorizationRecommender(n_components=n_components)
        self.content = ContentBasedRecommender()

    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> "HybridRecommender":
        self.ratings_df = ratings_df
        self.movies_df = movies_df.set_index("movieId")
        self.cf.fit(ratings_df)
        self.content.fit(movies_df)
        return self

    def _liked_movies(self, user_id: int) -> list[int]:
        user_rows = self.ratings_df[self.ratings_df["userId"] == user_id]
        return user_rows[user_rows["rating"] >= LIKED_RATING]["movieId"].tolist()

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        """Top-n hybrid recommendations for a user, with movie titles."""
        # Pull a wider CF candidate list, then re-rank with the content signal.
        cf_candidates = self.cf.recommend(user_id, n=max(n * 5, 50))
        if not cf_candidates:
            return pd.DataFrame(columns=["movieId", "title", "score"])

        liked = self._liked_movies(user_id)
        cf_scores = np.array([s for _, s in cf_candidates])
        cf_norm = (cf_scores - cf_scores.min()) / (np.ptp(cf_scores) + 1e-9)

        rows = []
        for (movie_id, _), cf_s in zip(cf_candidates, cf_norm):
            content_s = self.content.similarity_to_movies(movie_id, liked)
            score = CF_WEIGHT * cf_s + CONTENT_WEIGHT * content_s
            title = self.movies_df.loc[movie_id, "title"] if movie_id in self.movies_df.index else "?"
            rows.append({"movieId": movie_id, "title": title, "score": round(float(score), 4)})

        return (
            pd.DataFrame(rows)
            .sort_values("score", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

    def similar_movies(self, movie_id: int, n: int = 10) -> pd.DataFrame:
        """Movies with the most similar genres, with titles."""
        sims = self.content.similar_movies(movie_id, n=n)
        rows = [
            {
                "movieId": mid,
                "title": self.movies_df.loc[mid, "title"] if mid in self.movies_df.index else "?",
                "similarity": round(sim, 4),
            }
            for mid, sim in sims
        ]
        return pd.DataFrame(rows)
