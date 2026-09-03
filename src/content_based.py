"""
Content-based similarity from movie genres.

Each movie's genre list ("Action|Adventure|Sci-Fi") is turned into a TF-IDF
vector, so genres that are rare across the catalog carry more weight than
common ones. Cosine similarity between these vectors gives "movies like this
one" without needing any rating data -- which is what lets the system say
something useful about a brand-new movie that nobody has rated yet
(the cold-start case collaborative filtering can't handle).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def fit(self, movies_df: pd.DataFrame) -> "ContentBasedRecommender":
        self.movies = movies_df.reset_index(drop=True)
        genres = (
            self.movies["genres"]
            .str.replace("(no genres listed)", "", regex=False)
            .str.replace("|", " ", regex=False)
        )
        # token_pattern keeps multi-word genres like "Film-Noir" intact.
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.tfidf = self.vectorizer.fit_transform(genres)
        self.index_of = {mid: i for i, mid in enumerate(self.movies["movieId"])}
        return self

    def similar_movies(self, movie_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Returns the n most genre-similar movies as (movie_id, similarity)."""
        if movie_id not in self.index_of:
            return []
        idx = self.index_of[movie_id]
        sims = cosine_similarity(self.tfidf[idx], self.tfidf).ravel()
        order = [i for i in np.argsort(sims)[::-1] if i != idx][:n]
        return [(int(self.movies.iloc[i]["movieId"]), float(sims[i])) for i in order]

    def similarity_to_movies(self, movie_id: int, movie_ids: list[int]) -> float:
        """Max genre similarity between one movie and a set of movies (used to
        judge how well a candidate fits a user's already-liked movies)."""
        if movie_id not in self.index_of:
            return 0.0
        idxs = [self.index_of[m] for m in movie_ids if m in self.index_of]
        if not idxs:
            return 0.0
        sims = cosine_similarity(self.tfidf[self.index_of[movie_id]], self.tfidf[idxs])
        return float(sims.max())
