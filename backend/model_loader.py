"""
model_loader.py
===============
Responsible for:
  1. Loading the pre-trained ItemCF model from model.pkl
  2. Loading the movie catalogue from movies.dat
  3. Implementing recommend_for_new_user() — cold-start scoring
     via item-item similarity (no retraining required)
  4. Implementing filter_movies() for genre/vibe filtering

The ItemCFModel stores:
  _similarity  : (n_movies × n_movies) cosine similarity matrix
  _movie_index : { movie_id → col_idx }
  _index_movie : { col_idx  → movie_id }
  _global_mean : float
  _item_biases : pd.Series  (movie_id → bias)
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Vibe → genre mapping
# ---------------------------------------------------------------------------

VIBE_MAP = {
    "happy":    ["Comedy", "Animation", "Children's"],
    "dark":     ["Thriller", "Crime", "Horror", "Film-Noir"],
    "fun":      ["Comedy", "Adventure", "Children's", "Animation"],
    "romantic": ["Romance"],
    "epic":     ["Action", "Adventure", "Sci-Fi", "War"],
    "classic":  ["Drama", "Film-Noir", "Musical"],
}


# ---------------------------------------------------------------------------
# Loader functions
# ---------------------------------------------------------------------------

def load_model(pkl_path: str, movies_dat_path: str):
    """
    Load the pickled ItemCFModel and the movies catalogue.

    Parameters
    ----------
    pkl_path       : path to model.pkl
    movies_dat_path: path to data/movies.dat

    Returns
    -------
    model      : the loaded ItemCFModel object
    movies_df  : DataFrame with columns [movie_id, title, genres, avg_rating]
    """
    # --- Load the model ---
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    # The pickle may contain the high-level Recommender wrapper (rec = Recommender(...); rec.fit())
    # rather than the bare ItemCFModel. Unwrap it so we always work with ItemCFModel directly.
    # Case 1: Recommender wrapper (your actual situation)
    if hasattr(model, "_models"):
        print("Detected Recommender wrapper — extracting ItemCFModel.")
        model = model._models.get("item_cf")

        if model is None:
            raise ValueError(
                "ItemCFModel not found inside Recommender._models. "
                "Make sure it was trained and saved correctly."
            )

    # Case 2: Already a proper ItemCFModel (no change needed)
    elif hasattr(model, "_similarity"):
        pass

    # Case 3: Wrong export (dict or something else)
    elif isinstance(model, dict):
        raise ValueError(
            "Loaded pickle is a dict, not ItemCFModel. "
            "Please export only the trained ItemCFModel."
        )

    print(f"Model loaded from {pkl_path}  |  type: {type(model).__name__}")

    # --- Load & enrich the movie catalogue ---
    movies_df = _load_movies(movies_dat_path, model)

    return model, movies_df


def _load_movies(movies_dat_path: str, model) -> pd.DataFrame:
    """Read movies.dat and compute per-item average rating from model biases."""
    df = pd.read_csv(
        movies_dat_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )

    # Compute an estimated average rating = global_mean + item_bias
    # (Falls back to global_mean if the movie wasn't in the training set)
    global_mean  = model._global_mean
    item_biases  = model._item_biases  # pd.Series indexed by movie_id

    def avg_rating(movie_id):
        bias = item_biases.get(movie_id, 0.0) if item_biases is not None else 0.0
        return round(float(np.clip(global_mean + bias, 1.0, 5.0)), 2)

    df["avg_rating"] = df["movie_id"].apply(avg_rating)

    # Only keep movies that exist in the trained model
    known_ids    = set(model._movie_index.keys())
    df = df[df["movie_id"].isin(known_ids)].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Cold-start recommendation for a brand-new user
# ---------------------------------------------------------------------------

def recommend_for_new_user(
    model,
    movies_df: pd.DataFrame,
    user_ratings: dict,          # { movie_id (int): rating (float) }
    n: int = 10,
    k_neighbours: int = 20,
) -> list[dict]:
    """
    Recommend movies for a user we have never seen before.

    Strategy (pure item-CF):
    For every candidate movie i that the user has NOT rated:
        score(i) = Σ_j [ sim(i,j) * r(u,j) ] / Σ_j |sim(i,j)|
    where j ranges over the user's rated movies,
    limited to the top-k_neighbours most similar ones.

    Parameters
    ----------
    model        : the loaded ItemCFModel
    movies_df    : DataFrame with movie metadata
    user_ratings : dict of { movie_id: rating }
    n            : number of recommendations to return
    k_neighbours : neighbourhood size

    Returns
    -------
    List of dicts: [{ movie_id, title, genres, score, avg_rating }, ...]
    """
    # Resolve rated movie IDs to their column indices in the similarity matrix
    rated_col_map = {}   # { col_idx: rating }
    for movie_id, rating in user_ratings.items():
        col = model._movie_index.get(int(movie_id))
        if col is not None:
            rated_col_map[col] = float(rating)

    if not rated_col_map:
        # None of the rated movies are in the model — return top-rated by avg
        return _fallback_recs(movies_df, user_ratings, n)

    rated_cols    = np.array(list(rated_col_map.keys()),   dtype=np.int32)
    rated_ratings = np.array(list(rated_col_map.values()), dtype=np.float32)

    # Extract similarity sub-matrix: all movies × rated movies
    # shape: (n_movies, len(rated_cols))
    sim_sub = model._similarity[:, rated_cols]   # numpy fancy indexing

    # For each candidate movie, take only top-k neighbours
    n_movies = len(model._index_movie)
    scores   = []

    for col_idx in range(n_movies):
        movie_id = model._index_movie[col_idx]

        # Skip movies the user already rated
        if col_idx in rated_col_map:
            continue

        sims = sim_sub[col_idx]          # similarities to each rated movie

        # Pick top-k by similarity
        if len(sims) <= k_neighbours:
            top_idx = np.arange(len(sims))
        else:
            top_idx = np.argpartition(sims, -k_neighbours)[-k_neighbours:]

        k_sims    = sims[top_idx]
        k_ratings = rated_ratings[top_idx]

        denom = np.abs(k_sims).sum()
        if denom == 0.0:
            score = model._global_mean
        else:
            score = float(np.dot(k_sims, k_ratings) / denom)

        score = float(np.clip(score, 1.0, 5.0))
        scores.append((movie_id, score))

    # Sort descending by predicted score
    scores.sort(key=lambda x: x[1], reverse=True)
    top_n = scores[:n]

    # Enrich with metadata
    meta = movies_df.set_index("movie_id")[["title", "genres", "avg_rating"]]
    result = []
    for movie_id, score in top_n:
        row = meta.loc[movie_id] if movie_id in meta.index else None
        result.append({
            "movie_id":   movie_id,
            "title":      row["title"]      if row is not None else f"Movie {movie_id}",
            "genres":     row["genres"]     if row is not None else "",
            "avg_rating": float(row["avg_rating"]) if row is not None else 0.0,
            "score":      round(score, 3),
        })

    return result


def _fallback_recs(movies_df: pd.DataFrame, user_ratings: dict, n: int) -> list[dict]:
    """Fallback: return highest avg-rated unseen movies."""
    seen = set(user_ratings.keys())
    df   = movies_df[~movies_df["movie_id"].isin(seen)]
    top  = df.nlargest(n, "avg_rating")
    return top[["movie_id", "title", "genres", "avg_rating"]].to_dict(orient="records")


# ---------------------------------------------------------------------------
# Genre / vibe filtering
# ---------------------------------------------------------------------------

def filter_movies(
    movies_df: pd.DataFrame,
    genre: str = "",
    vibe: str  = "",
) -> pd.DataFrame:
    """
    Filter the movie catalogue by genre and/or vibe.

    - genre: partial, case-insensitive match against the genres string
    - vibe:  mapped to a list of genres via VIBE_MAP; any match qualifies
    """
    df = movies_df.copy()

    # Apply genre filter
    if genre:
        df = df[df["genres"].str.contains(genre, case=False, na=False)]

    # Apply vibe filter (union of mapped genres)
    if vibe:
        vibe_genres = VIBE_MAP.get(vibe.lower(), [])
        if vibe_genres:
            pattern = "|".join(vibe_genres)
            df = df[df["genres"].str.contains(pattern, case=False, na=False)]

    return df.sort_values("avg_rating", ascending=False).reset_index(drop=True)