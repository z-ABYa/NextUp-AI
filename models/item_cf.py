"""
Item-based Collaborative Filtering using cosine similarity.

Prediction for (u, i):
    r̂(u,i) = Σ_j [ sim(i,j) * r(u,j) ] / Σ_j |sim(i,j)|
    summed over the k nearest neighbours of i that user u has rated.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.base import BaseRecommender

logger = logging.getLogger(__name__)


class ItemCFModel(BaseRecommender):
    """
    Item-based Collaborative Filter.

    Parameters
    ----------
    k_neighbours : Number of similar items to use when predicting a rating.
    """

    def __init__(self, k_neighbours: int = 20):
        self.k_neighbours = k_neighbours
        self._similarity  = None
        self._user_item   = None
        self._user_index  = {}
        self._movie_index = {}
        self._index_movie = {}
        self._global_mean = 0.0
        self._user_biases = None
        self._item_biases = None
        self._seen        = {}

    def fit(self, split_data) -> "ItemCFModel":
        self._user_index  = split_data.user_index
        self._movie_index = split_data.movie_index
        self._index_movie = split_data.index_movie
        self._global_mean = split_data.global_mean
        self._user_biases = split_data.user_biases
        self._item_biases = split_data.item_biases
        self._seen = (
            split_data.train.groupby("user_id")["movie_id"].apply(set).to_dict()
        )

        self._user_item  = split_data.user_item_matrix.toarray().astype("float32")
        self._similarity = cosine_similarity(self._user_item.T, dense_output=True)
        logger.info("%s fitted | sim_matrix=%s", self.name, self._similarity.shape)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        u = self._user_index.get(user_id)
        i = self._movie_index.get(movie_id)

        def _bias_fallback():
            return float(np.clip(
                self._global_mean
                + self._user_biases.get(user_id, 0.0)
                + self._item_biases.get(movie_id, 0.0),
                1.0, 5.0,
            ))

        if u is None or i is None:
            return _bias_fallback()

        rated_mask    = self._user_item[u] > 0
        rated_mask[i] = False

        if not rated_mask.any():
            return _bias_fallback()

        sims    = self._similarity[i]
        top_k   = np.argsort(sims[rated_mask])[::-1][: self.k_neighbours]
        k_sims  = sims[rated_mask][top_k]
        k_rats  = self._user_item[u][rated_mask][top_k]

        denom = np.abs(k_sims).sum()
        if denom == 0:
            return float(self._global_mean)
        return float(np.clip(np.dot(k_sims, k_rats) / denom, 1.0, 5.0))

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
        u = self._user_index.get(user_id)
        if u is None:
            return []

        seen    = self._seen.get(user_id, set()) if exclude_seen else set()
        results = [
            (self._index_movie[col], self.predict(user_id, self._index_movie[col]))
            for col in range(len(self._index_movie))
            if self._index_movie[col] not in seen
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]
