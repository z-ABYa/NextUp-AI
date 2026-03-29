"""
models/content_based.py
=======================
Content-Based Filter using genre similarity.

For a given user, we build a "taste profile" as the weighted average of
genre vectors of movies the user rated (weights = normalised ratings).
We then rank unseen movies by cosine similarity to that profile.

This model does NOT predict star ratings — recommend() returns similarity
scores in [0, 1].  predict() falls back to biases for rating estimation.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.base import BaseRecommender
from features.preprocessor import MovieFeatures

logger = logging.getLogger(__name__)


class ContentBasedModel(BaseRecommender):
    """
    Genre-based content filter.

    Parameters
    ----------
    movie_features : MovieFeatures
        Pre-computed genre matrix from Preprocessor.build_movie_features().
    """

    def __init__(self, movie_features: MovieFeatures):
        self._mf            = movie_features
        self._movie_id_to_row: dict[int, int] = {}

        self._global_mean = 0.0
        self._user_biases = None
        self._item_biases = None
        self._seen        = {}
        self._user_profiles: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------

    def fit(self, split_data) -> "ContentBasedModel":
        logger.info("%s fitting …", self.name)

        self._global_mean = split_data.global_mean
        self._user_biases = split_data.user_biases
        self._item_biases = split_data.item_biases

        self._movie_id_to_row = {
            mid: idx for idx, mid in enumerate(self._mf.movie_ids)
        }

        self._seen = (
            split_data.train
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        # Pre-compute user taste profiles
        self._user_profiles = self._build_profiles(split_data.train)
        logger.info(
            "%s fitted | %d user profiles built",
            self.name, len(self._user_profiles),
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        """Falls back to bias model (content model gives similarity, not stars)."""
        u_bias = self._user_biases.get(user_id, 0.0)
        i_bias = self._item_biases.get(movie_id, 0.0)
        return float(np.clip(self._global_mean + u_bias + i_bias, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        profile = self._user_profiles.get(user_id)
        if profile is None:
            return []

        seen    = self._seen.get(user_id, set()) if exclude_seen else set()
        results = []

        for col_idx, movie_id in enumerate(self._mf.movie_ids):
            if movie_id in seen:
                continue
            genre_vec = self._mf.genre_matrix[col_idx]
            sim       = self._cosine(profile, genre_vec)
            results.append((int(movie_id), float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    # ------------------------------------------------------------------

    def get_user_profile(self, user_id: int) -> dict[str, float] | None:
        """Return a readable genre → weight mapping for a user's taste profile."""
        profile = self._user_profiles.get(user_id)
        if profile is None:
            return None
        return dict(zip(self._mf.genre_names, profile.tolist()))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_profiles(
        self, train: "pd.DataFrame"
    ) -> dict[int, np.ndarray]:
        """
        For each user, compute a weighted-average genre vector
        (weight = normalised rating so highly-rated movies count more).
        """
        profiles: dict[int, np.ndarray] = {}
        n_genres = self._mf.genre_matrix.shape[1]

        for user_id, group in train.groupby("user_id"):
            weighted_sum = np.zeros(n_genres, dtype="float32")
            weight_total = 0.0

            for _, row in group.iterrows():
                row_idx = self._movie_id_to_row.get(row["movie_id"])
                if row_idx is None:
                    continue
                w             = float(row["rating"])
                weighted_sum += w * self._mf.genre_matrix[row_idx]
                weight_total += w

            if weight_total > 0:
                profile = weighted_sum / weight_total
                # L2-normalise so cosine_similarity is equivalent to dot-product
                norm = np.linalg.norm(profile)
                profiles[int(user_id)] = profile / norm if norm > 0 else profile

        return profiles

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0
