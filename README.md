# Movie Recommendation Engine

A movie recommender built on the real [MovieLens](https://grouplens.org/datasets/movielens/)
dataset (ml-latest-small: ~100k ratings, 9,742 movies, 610 users). It combines
two recommendation approaches into one hybrid model:

1. **Collaborative filtering** (matrix factorization) — learns what a user will
   like from the rating patterns of similar users.
2. **Content-based filtering** (genre similarity) — recommends movies whose
   genres match what the user already enjoys, and can describe a brand-new movie
   that nobody has rated yet.

The two signals are blended so the system leans on collaborative filtering when
a user has rating history, while the content signal helps break ties and handles
the cold-start case.

## Why a hybrid instead of one method

- **Collaborative filtering alone** is accurate for active users but knows
  nothing about a movie with no ratings yet (the *cold-start* problem).
- **Content-based alone** can always say "this movie is like that one," but it
  only knows genres, so its recommendations are shallow.

Combining them keeps the accuracy of collaborative filtering while staying useful
for new movies and new users — the same "don't rely on a single signal" idea
applied to recommendations.

## How it works

```
                 ┌──────────────────────────┐
ratings.csv ───► │ Matrix Factorization (SVD)│──► predicted rating per user/movie
                 │  on mean-centered ratings │
                 └──────────────────────────┘
                                                     ┐
                 ┌──────────────────────────┐        ├─► weighted blend ─► ranked recommendations
movies.csv  ───► │ TF-IDF over genres +      │──► genre similarity        (0.7 CF + 0.3 content)
                 │ cosine similarity         │        ┘
                 └──────────────────────────┘
```

- **Matrix factorization** — ratings are arranged in a user × movie matrix and
  mean-centered per user (so we model preference *relative to each user's own
  average*, removing the "rates everything high" bias). Truncated SVD learns
  50 latent factors; predicted ratings come from the low-rank reconstruction.
- **Genre similarity** — each movie's genre list becomes a TF-IDF vector, so
  rare genres count more than common ones, and cosine similarity gives
  "movies like this."
- **Hybrid ranking** — collaborative filtering proposes candidates, which are
  then re-ranked by how well their genres match the user's already-liked movies.

## Project layout

```
movie-recommender/
├── src/
│   ├── data_loader.py     # downloads + loads the MovieLens dataset
│   ├── collaborative.py   # matrix-factorization (SVD) recommender
│   ├── content_based.py   # TF-IDF genre-similarity recommender
│   ├── recommender.py     # hybrid model combining both
│   ├── evaluate.py        # RMSE + precision@k / recall@k
│   └── api.py             # FastAPI service
├── tests/
│   └── test_recommender.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python -m src.data_loader   # downloads MovieLens into data/ (first run only)
python -m src.evaluate      # trains + prints evaluation metrics
uvicorn src.api:app --reload   # recommendation API at http://localhost:8000
```

The dataset (~1 MB) is downloaded automatically on first use and is not checked
into git.

## API

| Endpoint | What it returns |
|----------|-----------------|
| `GET /recommend/{user_id}?n=10` | Top-n recommended movies for a user |
| `GET /similar/{movie_id}?n=10`  | Movies with the most similar genres |
| `GET /health`                   | Service status |

## Evaluation

Ratings are split **per user by time** — each user's most recent 20% of ratings
form the test set — so the model is never trained on a user's future to predict
their past. Two metrics are reported:

- **RMSE** — how close predicted ratings are to actual held-out ratings.
- **Precision@10 / Recall@10** — of the 10 movies recommended to each user, how
  many they actually rated highly (≥4) in the held-out set. This matters more
  than RMSE for a real product, since users see a *ranked list*, not raw
  predicted scores.

### Results (ml-latest-small, 50 latent factors)

| Metric | Value |
|--------|-------|
| RMSE          | 0.953 |
| Precision@10  | 0.065 |
| Recall@10     | 0.065 |

**Reading these honestly:** an RMSE around 0.95 is in the normal range for
matrix factorization on MovieLens. Precision@10 looks small, but it is measured
under a deliberately hard setup — recommending 10 movies out of ~9,700 and
checking whether they land on the handful of movies a user rated highly in a
*future* time window. Recommending from the full catalog against a strict
temporal split is much harder than the random-split numbers often quoted, so
these are realistic rather than inflated. The natural next steps to push
precision up would be adding an implicit-feedback signal (what users *watched*,
not just rated) and tuning the number of latent factors.
