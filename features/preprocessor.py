"""
features/preprocessor.py
========================
All feature-engineering and train/test splitting lives here.
Returns objects that models can consume directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


@dataclass
class SplitData:
    """Holds train / test rating DataFrames plus handy derived artefacts."""
    train:        pd.DataFrame
    test:         pd.DataFrame
    # Sparse user-item matrix (train only)
    user_item_matrix: csr_matrix
    # Contiguous index mappings  (original IDs → row/col indices)
    user_index:   dict[int, int] = field(default_factory=dict)
    movie_index:  dict[int, int] = field(default_factory=dict)
    # Reverse mappings
    index_user:   dict[int, int] = field(default_factory=dict)
    index_movie:  dict[int, int] = field(default_factory=dict)
    # Global rating statistics (computed on train)
    global_mean:  float = 0.0
    user_biases:  pd.Series = field(default_factory=pd.Series)
    item_biases:  pd.Series = field(default_factory=pd.Series)


@dataclass
class MovieFeatures:
    """Genre one-hot matrix and associated metadata."""
    genre_matrix: np.ndarray          # shape: (n_movies, n_genres)
    genre_names:  list[str]
    movie_ids:    np.ndarray          # movie_id in the same row order


class Preprocessor:
    """
    Orchestrates all preprocessing steps.

    Parameters
    ----------
    test_size : float
        Fraction of ratings held out for evaluation.
    random_state : int
        Reproducibility seed.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size    = test_size
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_ratings(self, ratings: pd.DataFrame) -> SplitData:
        """Split ratings, build the sparse matrix, and compute biases."""
        logger.info("Splitting ratings (test_size=%.0f%%)", self.test_size * 100)
        train, test = train_test_split(
            ratings,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=None,
        )
        train = train.reset_index(drop=True)
        test  = test.reset_index(drop=True)

        user_index, index_user   = self._make_index(train["user_id"])
        movie_index, index_movie = self._make_index(train["movie_id"])

        logger.info(
            "Train: %d | Test: %d | Users: %d | Movies: %d",
            len(train), len(test), len(user_index), len(movie_index),
        )

        matrix = self._build_matrix(train, user_index, movie_index)
        global_mean, user_biases, item_biases = self._compute_biases(train)

        return SplitData(
            train=train,
            test=test,
            user_item_matrix=matrix,
            user_index=user_index,
            movie_index=movie_index,
            index_user=index_user,
            index_movie=index_movie,
            global_mean=global_mean,
            user_biases=user_biases,
            item_biases=item_biases,
        )

    def build_movie_features(self, movies: pd.DataFrame) -> MovieFeatures:
        """One-hot-encode genres for content-based filtering."""
        logger.info("Building genre feature matrix")
        mlb = MultiLabelBinarizer()
        genre_matrix = mlb.fit_transform(movies["genre_list"])
        return MovieFeatures(
            genre_matrix=genre_matrix.astype("float32"),
            genre_names=list(mlb.classes_),
            movie_ids=movies["movie_id"].to_numpy(),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_index(series: pd.Series) -> tuple[dict, dict]:
        unique = series.unique()
        fwd    = {v: i for i, v in enumerate(unique)}
        rev    = {i: v for v, i in fwd.items()}
        return fwd, rev

    @staticmethod
    def _build_matrix(
        train: pd.DataFrame,
        user_index: dict,
        movie_index: dict,
    ) -> csr_matrix:
        rows = train["user_id"].map(user_index)
        cols = train["movie_id"].map(movie_index)
        data = train["rating"].values
        shape = (len(user_index), len(movie_index))
        return csr_matrix((data, (rows, cols)), shape=shape, dtype="float32")

    @staticmethod
    def _compute_biases(
        train: pd.DataFrame,
    ) -> tuple[float, pd.Series, pd.Series]:
        """Compute global mean, per-user mean deviation, per-item mean deviation."""
        global_mean  = train["rating"].mean()
        user_biases  = train.groupby("user_id")["rating"].mean() - global_mean
        item_biases  = train.groupby("movie_id")["rating"].mean() - global_mean
        return float(global_mean), user_biases, item_biases
