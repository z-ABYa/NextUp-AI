"""
Rating-accuracy metrics (require predict()):  RMSE, MAE
Ranking metrics     (require recommend()):    Precision@K, Coverage@K
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from models.base import BaseRecommender

logger = logging.getLogger(__name__)


@dataclass
class RatingMetrics:
    rmse: float
    mae:  float


@dataclass
class RankingMetrics:
    precision_at_k: float
    coverage:       float


@dataclass
class EvalReport:
    model_name:      str
    rating_metrics:  RatingMetrics
    ranking_metrics: RankingMetrics

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            **asdict(self.rating_metrics),
            **asdict(self.ranking_metrics),
        }


class Evaluator:
    def __init__(
        self,
        k: int = 10,
        relevance_threshold: float = 4.0,
        n_users_sample: int | None = 500,
        random_state: int = 42,
    ):
        self.k                   = k
        self.relevance_threshold = relevance_threshold
        self.n_users_sample      = n_users_sample
        self.random_state        = random_state

    def evaluate(
        self,
        model: BaseRecommender,
        split_data,
        all_movie_ids: list[int] | None = None,
    ) -> EvalReport:
        rating_m  = self._rating_metrics(model, split_data.test)
        ranking_m = self._ranking_metrics(
            model, split_data.train, split_data.test, all_movie_ids
        )
        report = EvalReport(
            model_name=model.name,
            rating_metrics=rating_m,
            ranking_metrics=ranking_m,
        )
        rm, rk = report.rating_metrics, report.ranking_metrics
        logger.info(
            "%s | RMSE=%.4f  MAE=%.4f | P@K=%.4f  Cov=%.2f%%",
            report.model_name, rm.rmse, rm.mae,
            rk.precision_at_k, rk.coverage * 100,
        )
        return report

    def _rating_metrics(self, model: BaseRecommender, test: pd.DataFrame) -> RatingMetrics:
        preds  = model.predict_batch(test[["user_id", "movie_id"]])
        truth  = test["rating"].to_numpy(dtype="float32")
        errors = preds - truth
        return RatingMetrics(
            rmse=float(np.sqrt(np.mean(errors ** 2))),
            mae=float(np.mean(np.abs(errors))),
        )

    def _ranking_metrics(
        self,
        model: BaseRecommender,
        train: pd.DataFrame,
        test: pd.DataFrame,
        all_movie_ids: list[int] | None = None,
    ) -> RankingMetrics:
        rng = np.random.default_rng(self.random_state)

        test_users  = test["user_id"].unique()
        train_users = set(train["user_id"].unique())
        common      = [u for u in test_users if u in train_users]

        if self.n_users_sample and len(common) > self.n_users_sample:
            common = rng.choice(common, size=self.n_users_sample, replace=False).tolist()

        gt = (
            test[test["rating"] >= self.relevance_threshold]
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        precisions: list[float] = []
        recommended_set: set[int] = set()

        for user_id in common:
            relevant = gt.get(user_id, set())
            if not relevant:
                continue
            recs = [mid for mid, _ in model.recommend(user_id, n=self.k)]
            recommended_set.update(recs)
            precisions.append(sum(1 for m in recs if m in relevant) / self.k)

        catalogue = set(all_movie_ids) if all_movie_ids else set()
        coverage  = len(recommended_set & catalogue) / max(len(catalogue), 1)

        return RankingMetrics(
            precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
            coverage=coverage,
        )

    @staticmethod
    def compare(reports: list[EvalReport]) -> pd.DataFrame:
        return pd.DataFrame([r.as_dict() for r in reports]).set_index("model")
