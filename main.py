"""
main.py
=======
End-to-end pipeline runner.  Run with:

    python main.py

Trains all four models, evaluates them, and prints recommendations
for a sample user with each model.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

from recommender import Recommender

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 120)

# ── Config ─────────────────────────────────────────────────────────────
DATA_DIR    = Path('data/')
SAMPLE_USER = 100          # user_id to demo recommendations for
TOP_N       = 5
MODELS      = ["baseline", "svd", "item_cf", "content"]

MODEL_CONFIGS = {
    "svd":     {"n_factors": 50},
    "item_cf": {"k_neighbours": 20},
}


def main():
    logger.info("=" * 60)
    logger.info("MovieLens Modular Recommendation System")
    logger.info("=" * 60)

    # ── 1. Build & fit ─────────────────────────────────────────────────
    rec = Recommender(
        data_dir=DATA_DIR,
        models=MODELS,
        model_configs=MODEL_CONFIGS,
        test_size=0.2,
        random_state=42,
        eval_k=10,
    )
    rec.fit()

    # ── 2. Compare models ──────────────────────────────────────────────
    logger.info("\n%s\nModel Comparison\n%s", "=" * 60, "=" * 60)
    comparison = rec.compare_models()
    print(comparison.to_string())
    print()

    # ── 3. Sample user history ─────────────────────────────────────────
    logger.info("User %d — top-rated movies (training set):", SAMPLE_USER)
    history = rec.user_history(SAMPLE_USER, top_n=5)
    print(history.to_string(index=False))
    print()

    # ── 4. Recommendations from each model ─────────────────────────────
    for model_name in MODELS:
        rec.use(model_name)
        recs = rec.recommend(SAMPLE_USER, n=TOP_N)
        logger.info("Top-%d recommendations for user %d  [%s]:",
                    TOP_N, SAMPLE_USER, model_name.upper())
        print(recs[["title", "genres", "score"]].to_string(index=False))
        print()

    logger.info("Done.")


if __name__ == "__main__":
    main()
