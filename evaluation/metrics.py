"""
evaluation/metrics.py
=====================
All evaluation logic in one place.  Models are never aware of metrics —
metrics only need a model that implements BaseRecommender.

Rating-accuracy metrics (require predict()):
    - RMSE
    - MAE

Ranking metrics (require recommend()):
    - Precision@K
    - Recall@K
    - NDCG@K
    - Coverage@K   (catalogue coverage)
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
    recall_at_k:    float
    ndcg_at_k:      float
    coverage:       float   # fraction of catalogue recommended ≥ once


@dataclass
class EvalReport:
    model_name:     str
    rating_metrics: RatingMetrics
    ranking_metrics: RankingMetrics

    def as_dict(self) -> dict:
        return {
            "model": self.model_name,
            **asdict(self.rating_metrics),
            **asdict(self.ranking_metrics),
        }


class Evaluator:
    """
    Evaluates any BaseRecommender against a held-out test set.

    Parameters
    ----------
    k : int
        Cut-off for ranking metrics.
    relevance_threshold : float
        Minimum rating to count as "relevant" for ranking metrics.
    n_users_sample : int | None
        If set, evaluates ranking metrics on a random sample of users
        (ranking metrics are expensive to compute for all users).
    """

    def __init__(
        self,
        k: int = 10,
        relevance_threshold: float = 4.0,
        n_users_sample: int | None = 500,
        random_state: int = 42,
    ):
        self.k = k
        self.relevance_threshold = relevance_threshold
        self.n_users_sample      = n_users_sample
        self.random_state        = random_state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: BaseRecommender,
        split_data,
        all_movie_ids: list[int] | None = None,
    ) -> EvalReport:
        logger.info("Evaluating %s …", model.name)

        rating_m  = self.rating_metrics(model, split_data.test)
        ranking_m = self.ranking_metrics(
            model, split_data.train, split_data.test, all_movie_ids
        )

        report = EvalReport(
            model_name=model.name,
            rating_metrics=rating_m,
            ranking_metrics=ranking_m,
        )
        self._log_report(report)
        return report

    # ------------------------------------------------------------------
    # Rating-accuracy metrics
    # ------------------------------------------------------------------

    def rating_metrics(
        self,
        model: BaseRecommender,
        test: pd.DataFrame,
    ) -> RatingMetrics:
        preds = model.predict_batch(test[["user_id", "movie_id"]])
        truth = test["rating"].to_numpy(dtype="float32")
        errors = preds - truth
        return RatingMetrics(
            rmse=float(np.sqrt(np.mean(errors ** 2))),
            mae=float(np.mean(np.abs(errors))),
        )

    # ------------------------------------------------------------------
    # Ranking metrics
    # ------------------------------------------------------------------

    def ranking_metrics(
        self,
        model: BaseRecommender,
        train: pd.DataFrame,
        test: pd.DataFrame,
        all_movie_ids: list[int] | None = None,
    ) -> RankingMetrics:
        rng = np.random.default_rng(self.random_state)

        # Users that appear in both train and test
        test_users  = test["user_id"].unique()
        train_users = set(train["user_id"].unique())
        common      = [u for u in test_users if u in train_users]

        if self.n_users_sample and len(common) > self.n_users_sample:
            common = rng.choice(common, size=self.n_users_sample, replace=False).tolist()

        # Ground-truth relevant items per user (from test set)
        gt = (
            test[test["rating"] >= self.relevance_threshold]
            .groupby("user_id")["movie_id"]
            .apply(set)
            .to_dict()
        )

        precisions, recalls, ndcgs = [], [], []
        recommended_set: set[int] = set()

        for user_id in common:
            relevant = gt.get(user_id, set())
            if not relevant:
                continue

            recs        = [mid for mid, _ in model.recommend(user_id, n=self.k)]
            recommended_set.update(recs)

            hits         = [1 if m in relevant else 0 for m in recs]
            precision    = sum(hits) / self.k
            recall       = sum(hits) / len(relevant)
            ndcg         = self._ndcg(hits)

            precisions.append(precision)
            recalls.append(recall)
            ndcgs.append(ndcg)

        catalogue   = set(all_movie_ids) if all_movie_ids else set()
        coverage    = len(recommended_set & catalogue) / max(len(catalogue), 1)

        return RankingMetrics(
            precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
            recall_at_k=float(np.mean(recalls))    if recalls    else 0.0,
            ndcg_at_k=float(np.mean(ndcgs))        if ndcgs      else 0.0,
            coverage=coverage,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ndcg(self, hits: list[int]) -> float:
        if not any(hits):
            return 0.0
        dcg  = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(sum(hits), self.k)))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def _log_report(report: EvalReport):
        rm = report.rating_metrics
        rk = report.ranking_metrics
        logger.info(
            "%-20s | RMSE=%.4f  MAE=%.4f | P@K=%.4f  R@K=%.4f  NDCG@K=%.4f  Cov=%.2f%%",
            report.model_name,
            rm.rmse, rm.mae,
            rk.precision_at_k, rk.recall_at_k, rk.ndcg_at_k,
            rk.coverage * 100,
        )

    # ------------------------------------------------------------------
    # Comparison helper
    # ------------------------------------------------------------------

    @staticmethod
    def compare(reports: list[EvalReport]) -> pd.DataFrame:
        """Return a tidy DataFrame for side-by-side model comparison."""
        return pd.DataFrame([r.as_dict() for r in reports]).set_index("model")
