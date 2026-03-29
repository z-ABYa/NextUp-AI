"""
models/baseline.py
==================
Bias-only baseline: prediction = global_mean + user_bias + item_bias.
Fast to train, interpretable, and a solid lower-bound benchmark.
"""

from __future__ import annotations

import logging

import numpy as np

from models.base import BaseRecommender

logger = logging.getLogger(__name__)


class BaselineModel(BaseRecommender):
    """
    Predicts ratings using additive biases:

        r̂(u, i) = μ + b_u + b_i

    where μ is the global mean, b_u is the user bias, and b_i is the
    item bias (all estimated from the training set).
    """

    def __init__(self):
        self._global_mean  = None
        self._user_biases  = None
        self._item_biases  = None
        self._seen         = {}        # user_id → set of movie_ids
        self._all_movies   = []

    # ------------------------------------------------------------------

    def fit(self, split_data) -> "BaselineModel":
        self._global_mean  = split_data.global_mean
        self._user_biases  = split_data.user_biases
        self._item_biases  = split_data.item_biases
        self._all_movies   = split_data.train["movie_id"].unique().tolist()

        # Build seen-set per user
        self._seen = (
            split_data.train
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        logger.info(
            "%s fitted | global_mean=%.3f",
            self.name, self._global_mean,
        )
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        u_bias = self._user_biases.get(user_id, 0.0)
        i_bias = self._item_biases.get(movie_id, 0.0)
        score  = self._global_mean + u_bias + i_bias
        return float(np.clip(score, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        seen = self._seen.get(user_id, set()) if exclude_seen else set()
        candidates = [m for m in self._all_movies if m not in seen]
        scored = [(m, self.predict(user_id, m)) for m in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]
