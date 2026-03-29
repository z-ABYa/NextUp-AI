"""
recommender.py
==============
The public-facing Recommender class.  Users of the library only need to
interact with this class — it orchestrates the loader, preprocessor,
models, and evaluator behind a clean API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from data.loader import MovieLensLoader, RawDataset
from features.preprocessor import Preprocessor, SplitData
from models.baseline import BaselineModel
from models.svd_model import SVDModel
from models.item_cf import ItemCFModel
from models.content_based import ContentBasedModel
from models.base import BaseRecommender
from evaluation.metrics import Evaluator, EvalReport

logger = logging.getLogger(__name__)

ModelName = Literal["baseline", "svd", "item_cf", "content"]


class Recommender:
    """
    High-level façade for the MovieLens recommendation pipeline.

    Usage
    -----
    >>> rec = Recommender("/path/to/data", models=["svd", "baseline"])
    >>> rec.fit()
    >>> print(rec.recommend(user_id=42, n=10))
    >>> print(rec.compare_models())
    """

    _MODEL_REGISTRY = {
        "baseline": lambda _: BaselineModel(),
        "svd":      lambda cfg: SVDModel(n_factors=cfg.get("n_factors", 50)),
        "item_cf":  lambda cfg: ItemCFModel(k_neighbours=cfg.get("k_neighbours", 20)),
        "content":  None,   # requires movie_features — handled separately
    }

    def __init__(
        self,
        data_dir: str | Path,
        models: list[ModelName] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        model_configs: dict | None = None,
        eval_k: int = 10,
    ):
        self.data_dir     = Path(data_dir)
        self.model_names  = models or ["baseline", "svd"]
        self.test_size    = test_size
        self.random_state = random_state
        self.model_configs = model_configs or {}
        self.eval_k       = eval_k

        # Set after fit()
        self.dataset:       RawDataset | None  = None
        self.split_data:    SplitData  | None  = None
        self._models:       dict[str, BaseRecommender] = {}
        self._active_model: str = self.model_names[0]
        self._reports:      list[EvalReport] = []

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def fit(self) -> "Recommender":
        """Run the full pipeline: load → preprocess → train → evaluate."""
        # 1. Load
        loader       = MovieLensLoader(self.data_dir)
        self.dataset = loader.load()

        # 2. Preprocess
        prep             = Preprocessor(self.test_size, self.random_state)
        self.split_data  = prep.split_ratings(self.dataset.ratings)
        movie_features   = prep.build_movie_features(self.dataset.movies)

        # 3. Train each requested model
        evaluator = Evaluator(k=self.eval_k)
        all_movie_ids = self.dataset.movies["movie_id"].tolist()

        for name in self.model_names:
            cfg = self.model_configs.get(name, {})

            if name == "content":
                model = ContentBasedModel(movie_features)
            else:
                factory = self._MODEL_REGISTRY.get(name)
                if factory is None:
                    logger.warning("Unknown model '%s' — skipping.", name)
                    continue
                model = factory(cfg)

            model.fit(self.split_data)
            self._models[name] = model

            # 4. Evaluate
            report = evaluator.evaluate(model, self.split_data, all_movie_ids)
            self._reports.append(report)

        self._active_model = self.model_names[0]
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def use(self, model_name: ModelName) -> "Recommender":
        """Switch the active model used by recommend() / predict()."""
        if model_name not in self._models:
            raise ValueError(
                f"Model '{model_name}' not fitted. "
                f"Available: {list(self._models.keys())}"
            )
        self._active_model = model_name
        return self

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        enrich: bool = True,
    ) -> pd.DataFrame:
        """
        Return top-n recommendations for a user.

        Parameters
        ----------
        user_id : int
        n : int
        exclude_seen : bool
        enrich : bool
            If True, join movie title & genres onto the output.
        """
        model = self._models[self._active_model]
        recs  = model.recommend(user_id, n=n, exclude_seen=exclude_seen)

        df = pd.DataFrame(recs, columns=["movie_id", "score"])

        if enrich and self.dataset is not None:
            meta = self.dataset.movies[["movie_id", "title", "genres"]]
            df   = df.merge(meta, on="movie_id", how="left")

        return df

    def predict(self, user_id: int, movie_id: int) -> float:
        """Return predicted rating for a (user, movie) pair."""
        return self._models[self._active_model].predict(user_id, movie_id)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def compare_models(self) -> pd.DataFrame:
        """Side-by-side metric comparison of all fitted models."""
        return Evaluator.compare(self._reports)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def movie_info(self, movie_id: int) -> dict | None:
        """Look up movie metadata by movie_id."""
        row = self.dataset.movies[self.dataset.movies["movie_id"] == movie_id]
        return row.iloc[0].to_dict() if not row.empty else None

    def user_history(self, user_id: int, top_n: int = 10) -> pd.DataFrame:
        """Return a user's highest-rated movies from the training set."""
        df = (
            self.split_data.train[self.split_data.train["user_id"] == user_id]
            .sort_values("rating", ascending=False)
            .head(top_n)
            .merge(self.dataset.movies[["movie_id", "title", "genres"]], on="movie_id")
        )
        return df[["movie_id", "title", "genres", "rating"]]
