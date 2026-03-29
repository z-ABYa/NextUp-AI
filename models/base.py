"""
models/base.py
==============
Abstract base class enforcing a consistent fit / predict / recommend interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseRecommender(ABC):

    @abstractmethod
    def fit(self, split_data) -> "BaseRecommender":
        """Train the model on split_data.train."""

    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """Return a predicted rating for (user_id, movie_id)."""

    def predict_batch(self, pairs: pd.DataFrame) -> np.ndarray:
        """Row-wise predict() over a DataFrame with columns ['user_id', 'movie_id']."""
        return np.array([
            self.predict(row.user_id, row.movie_id)
            for row in pairs.itertuples(index=False)
        ], dtype="float32")

    @abstractmethod
    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> list[tuple[int, float]]:
        """Return the top-n (movie_id, score) tuples for user_id."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}()"
