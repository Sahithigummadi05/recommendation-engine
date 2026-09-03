"""
FastAPI service exposing the recommender.

    uvicorn src.api:app --reload

Endpoints:
    GET /recommend/{user_id}?n=10   -> top-n movies for a user
    GET /similar/{movie_id}?n=10    -> movies with similar genres
    GET /health
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from src.data_loader import load_ratings, load_movies
from src.recommender import HybridRecommender

app = FastAPI(title="Movie Recommendation Engine")

_model: HybridRecommender | None = None


@app.on_event("startup")
def _load_model() -> None:
    global _model
    ratings = load_ratings()
    movies = load_movies()
    _model = HybridRecommender().fit(ratings, movies)


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    recs = _model.recommend(user_id, n=n)
    if recs.empty:
        raise HTTPException(status_code=404, detail="unknown user or no recommendations")
    return recs.to_dict(orient="records")


@app.get("/similar/{movie_id}")
def similar(movie_id: int, n: int = 10):
    sims = _model.similar_movies(movie_id, n=n)
    if sims.empty:
        raise HTTPException(status_code=404, detail="unknown movie")
    return sims.to_dict(orient="records")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}
