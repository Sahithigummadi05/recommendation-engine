"""
Unit tests for the recommender components, using small hand-built rating
data so they run fast and don't need the MovieLens download.
"""

import pandas as pd

from src.collaborative import MatrixFactorizationRecommender
from src.content_based import ContentBasedRecommender
from src.recommender import HybridRecommender


def _toy_ratings() -> pd.DataFrame:
    # Users 1 & 2 love sci-fi (10, 11), user 3 loves comedy (20, 21).
    rows = [
        (1, 10, 5.0), (1, 11, 4.5), (1, 20, 2.0),
        (2, 10, 4.5), (2, 11, 5.0), (2, 21, 2.5),
        (3, 20, 5.0), (3, 21, 4.5), (3, 10, 2.0),
    ]
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


def _toy_movies() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (10, "Space Wars", "Sci-Fi|Action"),
            (11, "Robot Future", "Sci-Fi"),
            (20, "Laugh Out Loud", "Comedy"),
            (21, "Office Jokes", "Comedy|Romance"),
        ],
        columns=["movieId", "title", "genres"],
    )


def test_mf_predictions_are_in_valid_range():
    model = MatrixFactorizationRecommender(n_components=2).fit(_toy_ratings())
    pred = model.predict(1, 11)
    assert 0.5 <= pred <= 5.0


def test_mf_unknown_user_falls_back_to_global_mean():
    ratings = _toy_ratings()
    model = MatrixFactorizationRecommender(n_components=2).fit(ratings)
    assert model.predict(999, 10) == float(ratings["rating"].mean())


def test_mf_recommend_excludes_already_rated():
    model = MatrixFactorizationRecommender(n_components=2).fit(_toy_ratings())
    recs = model.recommend(1, n=10)
    rec_ids = {mid for mid, _ in recs}
    # User 1 already rated 10, 11 and 20 -> only 21 can be recommended.
    assert 10 not in rec_ids and 11 not in rec_ids and 20 not in rec_ids
    assert 21 in rec_ids


def test_content_similar_movies_prefers_same_genre():
    model = ContentBasedRecommender().fit(_toy_movies())
    sims = model.similar_movies(10, n=3)  # Space Wars (Sci-Fi|Action)
    top_id = sims[0][0]
    assert top_id == 11  # Robot Future (Sci-Fi) is the closest by genre


def test_content_unknown_movie_returns_empty():
    model = ContentBasedRecommender().fit(_toy_movies())
    assert model.similar_movies(12345) == []


def test_hybrid_recommends_unseen_movie_with_title():
    model = HybridRecommender(n_components=2).fit(_toy_ratings(), _toy_movies())
    recs = model.recommend(1, n=5)
    assert not recs.empty
    assert {"movieId", "title", "score"}.issubset(recs.columns)
    # Recommendations must be movies the user has not already rated.
    assert 10 not in set(recs["movieId"])
