"""
models/item_cf.py
=================
Item-based Collaborative Filtering using cosine similarity on the
user-item matrix (items as rows in the item-user space).

Prediction for (u, i):
    r̂(u, i) = Σ_j [ sim(i,j) * r(u,j) ] / Σ_j |sim(i,j)|
               summed over the k nearest neighbours of i that user u rated.
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
    k_neighbours : int
        Number of similar items to use when predicting a rating.
    """

    def __init__(self, k_neighbours: int = 20):
        self.k_neighbours = k_neighbours

        self._similarity  = None   # (n_movies, n_movies)
        self._user_item   = None   # dense (n_users, n_movies)
        self._user_index  = {}
        self._movie_index = {}
        self._index_movie = {}
        self._global_mean = 0.0
        self._user_biases = None
        self._item_biases = None
        self._seen        = {}

    # ------------------------------------------------------------------

    def fit(self, split_data) -> "ItemCFModel":
        logger.info("%s fitting (k=%d) …", self.name, self.k_neighbours)

        self._user_index  = split_data.user_index
        self._movie_index = split_data.movie_index
        self._index_movie = split_data.index_movie
        self._global_mean = split_data.global_mean
        self._user_biases = split_data.user_biases
        self._item_biases = split_data.item_biases

        self._seen = (
            split_data.train
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        # Dense copy of the user-item matrix (users × movies)
        self._user_item = split_data.user_item_matrix.toarray().astype("float32")

        # Item-user matrix (movies × users) for cosine similarity
        item_user = self._user_item.T           # (n_movies, n_users)
        logger.info("Computing item-item cosine similarity …")
        self._similarity = cosine_similarity(item_user, dense_output=True)
        logger.info("%s fitted | sim_matrix=%s", self.name, self._similarity.shape)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        u = self._user_index.get(user_id)
        i = self._movie_index.get(movie_id)

        # Cold-start
        if u is None or i is None:
            u_bias = self._user_biases.get(user_id, 0.0)
            i_bias = self._item_biases.get(movie_id, 0.0)
            return float(np.clip(self._global_mean + u_bias + i_bias, 1.0, 5.0))

        # Similarities between target item and all other items
        sims      = self._similarity[i]           # (n_movies,)
        user_row  = self._user_item[u]             # (n_movies,) — 0 if not rated

        # Mask: only items the user has rated (non-zero) and not the item itself
        rated_mask = (user_row > 0)
        rated_mask[i] = False

        if rated_mask.sum() == 0:
            # No neighbourhood → bias fallback
            u_bias = self._user_biases.get(user_id, 0.0)
            i_bias = self._item_biases.get(movie_id, 0.0)
            return float(np.clip(self._global_mean + u_bias + i_bias, 1.0, 5.0))

        # Pick top-k neighbours
        neighbour_sims   = sims[rated_mask]
        neighbour_ratings = user_row[rated_mask]

        top_k_idx = np.argsort(neighbour_sims)[::-1][: self.k_neighbours]
        k_sims    = neighbour_sims[top_k_idx]
        k_ratings = neighbour_ratings[top_k_idx]

        denom = np.abs(k_sims).sum()
        if denom == 0:
            return float(self._global_mean)

        score = np.dot(k_sims, k_ratings) / denom
        return float(np.clip(score, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        u = self._user_index.get(user_id)
        if u is None:
            return []

        seen      = self._seen.get(user_id, set()) if exclude_seen else set()
        user_row  = self._user_item[u]
        n_movies  = len(self._index_movie)

        results = []
        for col_idx in range(n_movies):
            movie_id = self._index_movie[col_idx]
            if movie_id in seen:
                continue
            score = self.predict(user_id, movie_id)
            results.append((movie_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]
