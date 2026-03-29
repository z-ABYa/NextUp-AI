"""
models/svd_model.py
===================
Truncated-SVD (matrix factorisation) collaborative filter.

After decomposing the bias-adjusted user-item matrix  R̃ ≈ U Σ Vᵀ,
predictions are:

    r̂(u, i) = μ + b_u + b_i + (U_u · V_i)

Attributes tunable via constructor parameters so they can be swept in
a hyper-parameter search without touching model logic.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse.linalg import svds

from models.base import BaseRecommender

logger = logging.getLogger(__name__)


class SVDModel(BaseRecommender):
    """
    Collaborative filter based on truncated Singular Value Decomposition.

    Parameters
    ----------
    n_factors : int
        Number of latent factors to keep.
    """

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors

        self._U           = None   # (n_users, k)
        self._sigma       = None   # (k,)
        self._Vt          = None   # (k, n_movies)
        self._pred_matrix = None   # (n_users, n_movies) — dense reconstruction

        self._user_index  = {}
        self._movie_index = {}
        self._index_movie = {}
        self._global_mean = 0.0
        self._user_biases = None
        self._item_biases = None
        self._seen        = {}

    # ------------------------------------------------------------------

    def fit(self, split_data) -> "SVDModel":
        logger.info("%s fitting with %d factors …", self.name, self.n_factors)

        self._user_index  = split_data.user_index
        self._movie_index = split_data.movie_index
        self._index_movie = split_data.index_movie
        self._global_mean = split_data.global_mean
        self._user_biases = split_data.user_biases
        self._item_biases = split_data.item_biases

        # Build seen-set
        self._seen = (
            split_data.train
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        # ---- bias-adjust the sparse matrix ----
        matrix = split_data.user_item_matrix.astype("float64")
        matrix = self._subtract_biases(matrix, split_data)

        # ---- truncated SVD ----
        k = min(self.n_factors, min(matrix.shape) - 1)
        self._U, self._sigma, self._Vt = svds(matrix, k=k)

        # ---- reconstruct full dense matrix (add biases back) ----
        self._pred_matrix = self._reconstruct(split_data)

        logger.info("%s fitted | shape=%s", self.name, self._pred_matrix.shape)
        return self

    def predict(self, user_id: int, movie_id: int) -> float:
        u = self._user_index.get(user_id)
        i = self._movie_index.get(movie_id)
        if u is None or i is None:
            # Cold-start fallback
            u_bias = self._user_biases.get(user_id, 0.0) if u is None else 0.0
            i_bias = self._item_biases.get(movie_id, 0.0) if i is None else 0.0
            return float(np.clip(self._global_mean + u_bias + i_bias, 1.0, 5.0))
        return float(np.clip(self._pred_matrix[u, i], 1.0, 5.0))

    def predict_batch(self, pairs) -> np.ndarray:
        """Vectorised batch prediction via index look-ups."""
        rows, cols, mask = [], [], []
        for row in pairs.itertuples(index=False):
            u = self._user_index.get(row.user_id)
            i = self._movie_index.get(row.movie_id)
            rows.append(u if u is not None else 0)
            cols.append(i if i is not None else 0)
            mask.append(u is not None and i is not None)

        rows  = np.array(rows)
        cols  = np.array(cols)
        preds = self._pred_matrix[rows, cols]
        # Cold-start rows: fall back to scalar predict
        for idx, (ok, row) in enumerate(zip(mask, pairs.itertuples(index=False))):
            if not ok:
                preds[idx] = self.predict(row.user_id, row.movie_id)
        return np.clip(preds, 1.0, 5.0).astype("float32")

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        u = self._user_index.get(user_id)
        if u is None:
            return []

        scores = self._pred_matrix[u]          # (n_movies,)
        seen   = self._seen.get(user_id, set()) if exclude_seen else set()

        results = []
        for col_idx, score in enumerate(scores):
            movie_id = self._index_movie[col_idx]
            if movie_id not in seen:
                results.append((movie_id, float(np.clip(score, 1.0, 5.0))))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _subtract_biases(self, matrix, split_data):
        """Remove global mean + per-user + per-item biases from sparse matrix."""
        cx = matrix.tocoo()
        for k, (r, c) in enumerate(zip(cx.row, cx.col)):
            user_id  = split_data.index_user[r]
            movie_id = split_data.index_movie[c]
            u_bias   = self._user_biases.get(user_id, 0.0)
            i_bias   = self._item_biases.get(movie_id, 0.0)
            cx.data[k] -= self._global_mean + u_bias + i_bias
        return cx.tocsr()

    def _reconstruct(self, split_data) -> np.ndarray:
        """Reconstruct the dense matrix and add biases back."""
        low_rank = self._U @ np.diag(self._sigma) @ self._Vt  # (n_users, n_movies)

        n_users  = low_rank.shape[0]
        n_movies = low_rank.shape[1]

        # Add global mean
        low_rank += self._global_mean

        # Add user biases (row-wise)
        for row_idx in range(n_users):
            user_id  = split_data.index_user[row_idx]
            low_rank[row_idx] += self._user_biases.get(user_id, 0.0)

        # Add item biases (col-wise)
        for col_idx in range(n_movies):
            movie_id  = split_data.index_movie[col_idx]
            low_rank[:, col_idx] += self._item_biases.get(movie_id, 0.0)

        return low_rank.astype("float32")
