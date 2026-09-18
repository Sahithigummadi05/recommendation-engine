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
│   ├── baselines.py       # global-mean / bias / popularity baselines
│   ├── metrics.py         # precision, recall, NDCG, MAP @k
│   ├── evaluate.py        # compares every model on the same split
│   └── api.py             # FastAPI service
├── tests/
│   ├── test_recommender.py
│   └── test_metrics_baselines.py
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

Every model is scored on the **same** per-user temporal split — each user's most
recent 20% of ratings form the test set, so nothing trains on a user's future —
with the same metrics, so the fancier models have to earn their complexity
against simple baselines. `python -m src.evaluate`:

| Model | RMSE | P@10 | R@10 | NDCG@10 | MAP@10 |
|-------|------|------|------|---------|--------|
| Global mean | 1.069 | 0.057 | 0.052 | 0.076 | 0.038 |
| Bias baseline | **0.903** | 0.043 | 0.033 | 0.055 | 0.025 |
| Popularity | 1.023 | 0.057 | 0.052 | 0.076 | 0.038 |
| Matrix factorization | 0.953 | 0.065 | 0.065 | 0.086 | 0.041 |
| **Hybrid (CF + content)** | — | **0.069** | **0.070** | **0.088** | **0.042** |

**The interesting result — RMSE and ranking disagree.** A simple
global-mean + user-bias + item-bias baseline actually gets the *best RMSE*
(0.903, beating matrix factorization's 0.953). That's a well-known effect:
plain SVD on raw ratings is not optimized for rating-prediction error. But RMSE
is not what a recommender is for — users see a *ranked list*. On the ranking
metrics that matter (Precision/Recall/NDCG/MAP@10), the **hybrid model wins**,
and both it and matrix factorization clearly beat the non-personalized
popularity baseline. So the project is tuned for **top-N ranking, not RMSE** —
and the baselines are what make that trade-off visible rather than assumed.

The absolute ranking numbers are modest by design: recommending 10 of ~9,700
movies and checking whether they hit the handful a user liked in a *future*
window is far harder than the random-split numbers usually quoted. Natural next
steps: an implicit-feedback signal (what users *watched*, not just rated) and a
learning-to-rank objective instead of SVD reconstruction.
