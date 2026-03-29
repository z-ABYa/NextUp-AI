from __future__ import annotations
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from data.loader import MovieLensLoader, RawDataset
from features.preprocessor import Preprocessor, SplitData
from models.item_cf import ItemCFModel
from models.base import BaseRecommender
from evaluation.metrics import Evaluator, EvalReport

logger = logging.getLogger(__name__)
ModelName = Literal["item_cf"]


class Recommender:

    def __init__(
        self,
        data_dir: str | Path,
        test_size: float = 0.2,
        random_state: int = 42,
        model_configs: dict | None = None,
        eval_k: int = 10,
    ):
        self.data_dir      = Path(data_dir)
        self.test_size     = test_size
        self.random_state  = random_state
        self.model_configs = model_configs or {}
        self.eval_k        = eval_k

        self.dataset:    RawDataset | None = None
        self.split_data: SplitData  | None = None
        self._model:     BaseRecommender | None = None
        self._report:    EvalReport | None = None

    def fit(self) -> "Recommender":
        self.dataset    = MovieLensLoader(self.data_dir).load()
        self.split_data = Preprocessor(self.test_size, self.random_state).split_ratings(
            self.dataset.ratings
        )

        cfg          = self.model_configs.get("item_cf", {})
        self._model  = ItemCFModel(k_neighbours=cfg.get("k_neighbours", 20))
        self._model.fit(self.split_data)

        all_movie_ids = self.dataset.movies["movie_id"].tolist()
        self._report  = Evaluator(k=self.eval_k).evaluate(
            self._model, self.split_data, all_movie_ids
        )
        return self

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        recs = self._model.recommend(user_id, n=n, exclude_seen=exclude_seen)
        df   = pd.DataFrame(recs, columns=["movie_id", "score"])
        if self.dataset is not None:
            meta = self.dataset.movies[["movie_id", "title", "genres"]]
            df   = df.merge(meta, on="movie_id", how="left")
        return df

    def predict(self, user_id: int, movie_id: int) -> float:
        return self._model.predict(user_id, movie_id)

    def compare_models(self) -> pd.DataFrame:
        return Evaluator.compare([self._report])

    def user_history(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        df = (
            self.split_data.train[self.split_data.train["user_id"] == user_id]
            .sort_values("rating", ascending=False)
            .head(top_n)
            .merge(self.dataset.movies[["movie_id", "title", "genres"]], on="movie_id")
        )
        return df[["movie_id", "title", "genres", "rating"]]
    