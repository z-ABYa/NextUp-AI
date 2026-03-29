import logging
import sys
from pathlib import Path

import pandas as pd
from recommender import Recommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

pd.set_option("display.max_colwidth", 50)
pd.set_option("display.width", 120)

DATA_DIR    = Path("data/")
SAMPLE_USER = 100
TOP_N       = 5

MODEL_CONFIGS = {
    "item_cf": {"k_neighbours": 20},
}


def main():
    rec = Recommender(
        data_dir=DATA_DIR,
        model_configs=MODEL_CONFIGS,
        test_size=0.2,
        random_state=42,
        eval_k=10,
    )
    rec.fit()

    print(rec.compare_models().to_string())
    print()

    history = rec.user_history(SAMPLE_USER, top_n=5)
    print(f"User {SAMPLE_USER} — top-rated movies (training set):")
    print(history.to_string(index=False))
    print()

    recs = rec.recommend(SAMPLE_USER, n=TOP_N)
    print(f"Top-{TOP_N} recommendations for user {SAMPLE_USER}:")
    print(recs[["title", "genres", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
