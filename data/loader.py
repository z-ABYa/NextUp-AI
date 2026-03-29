"""
data/loader.py
==============
Responsible solely for reading the raw .dat files from disk and returning
tidy DataFrames. No transformation logic lives here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RawDataset:
    """Container for the three raw MovieLens tables."""
    ratings: pd.DataFrame
    movies:  pd.DataFrame
    users:   pd.DataFrame


class MovieLensLoader:
    """
    Loads the MovieLens 1M dataset from a directory that contains
    ratings.dat, movies.dat, and users.dat.

    Parameters
    ----------
    data_dir : str | Path
        Directory that holds the three .dat files.
    """

    RATINGS_COLS = ["user_id", "movie_id", "rating", "timestamp"]
    MOVIES_COLS  = ["movie_id", "title", "genres"]
    USERS_COLS   = ["user_id", "gender", "age", "occupation", "zip_code"]

    # Occupation labels from the README
    OCCUPATION_MAP = {
        0: "other/not specified", 1: "academic/educator", 2: "artist",
        3: "clerical/admin", 4: "college/grad student", 5: "customer service",
        6: "doctor/health care", 7: "executive/managerial", 8: "farmer",
        9: "homemaker", 10: "K-12 student", 11: "lawyer", 12: "programmer",
        13: "retired", 14: "sales/marketing", 15: "scientist",
        16: "self-employed", 17: "technician/engineer", 18: "tradesman/craftsman",
        19: "unemployed", 20: "writer",
    }

    AGE_MAP = {
        1: "Under 18", 18: "18-24", 25: "25-34",
        35: "35-44", 45: "45-49", 50: "50-55", 56: "56+",
    }

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._validate_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> RawDataset:
        """Load all three files and return a RawDataset."""
        logger.info("Loading MovieLens data from %s", self.data_dir)
        ratings = self._load_ratings()
        movies  = self._load_movies()
        users   = self._load_users()
        logger.info(
            "Loaded  %d ratings | %d movies | %d users",
            len(ratings), len(movies), len(users),
        )
        return RawDataset(ratings=ratings, movies=movies, users=users)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_dir(self):
        required = ["ratings.dat", "movies.dat", "users.dat"]
        missing  = [f for f in required if not (self.data_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing files in {self.data_dir}: {missing}"
            )

    def _read_dat(self, filename: str, columns: list[str]) -> pd.DataFrame:
        path = self.data_dir / filename
        return pd.read_csv(
            path, sep="::", engine="python",
            names=columns, encoding="latin-1",
        )

    def _load_ratings(self) -> pd.DataFrame:
        df = self._read_dat("ratings.dat", self.RATINGS_COLS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df["rating"]    = df["rating"].astype("float32")
        return df

    def _load_movies(self) -> pd.DataFrame:
        df = self._read_dat("movies.dat", self.MOVIES_COLS)
        # Split pipe-separated genres into a list
        df["genre_list"] = df["genres"].str.split("|")
        # Extract release year from title
        df["year"] = df["title"].str.extract(r"\((\d{4})\)$").astype("Int32")
        return df

    def _load_users(self) -> pd.DataFrame:
        df = self._read_dat("users.dat", self.USERS_COLS)
        df["age_label"]        = df["age"].map(self.AGE_MAP)
        df["occupation_label"] = df["occupation"].map(self.OCCUPATION_MAP)
        return df
