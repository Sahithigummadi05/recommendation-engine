"""
Downloads and loads the MovieLens ml-latest-small dataset (~100k ratings,
~9,700 movies, 610 users).

The dataset is fetched from the official GroupLens mirror on first use and
cached under data/. Nothing here is checked into git -- the raw data is
downloaded locally instead (see .gitignore).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import urllib.request

DATA_DIR = Path("data")
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
EXTRACTED_DIR = DATA_DIR / "ml-latest-small"


def download_movielens(dest: Path = DATA_DIR) -> Path:
    """Downloads and extracts the dataset if it is not already present."""
    dest.mkdir(parents=True, exist_ok=True)
    if (EXTRACTED_DIR / "ratings.csv").exists():
        return EXTRACTED_DIR

    print(f"Downloading MovieLens dataset from {MOVIELENS_URL} ...")
    with urllib.request.urlopen(MOVIELENS_URL) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(dest)
    print(f"Extracted to {EXTRACTED_DIR}")
    return EXTRACTED_DIR


def load_ratings() -> pd.DataFrame:
    """Returns the ratings table: userId, movieId, rating, timestamp."""
    download_movielens()
    return pd.read_csv(EXTRACTED_DIR / "ratings.csv")


def load_movies() -> pd.DataFrame:
    """Returns the movies table: movieId, title, genres."""
    download_movielens()
    return pd.read_csv(EXTRACTED_DIR / "movies.csv")


if __name__ == "__main__":
    ratings = load_ratings()
    movies = load_movies()
    print(f"Loaded {len(ratings)} ratings, {len(movies)} movies, "
          f"{ratings['userId'].nunique()} users.")
