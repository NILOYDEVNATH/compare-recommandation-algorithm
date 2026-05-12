"""Shared data preparation and evaluation helpers for MovieLens experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


SPARSITY_LEVELS = [0.50, 0.70, 0.90, 0.95, 0.99]
SPARSITY_LABELS = ["50%", "70%", "90%", "95%", "99%"]


@dataclass(frozen=True)
class MatrixData:
    matrix: csr_matrix
    user_map: dict[int, int]
    item_map: dict[int, int]


@dataclass(frozen=True)
class EvaluationResult:
    rmse: float
    inference_time: float
    evaluated_count: int
    skipped_count: int
    diagnostics: dict[str, float]


class GlobalMeanBaseline:
    """Predict every known test pair with the training-set global mean."""

    def fit(self, matrix: csr_matrix) -> "GlobalMeanBaseline":
        self.global_mean = float(matrix.data.mean())
        return self

    def predict_batch(self, pairs: list[tuple[int, int]]) -> list[float]:
        return [self.global_mean] * len(pairs)


def load_movielens(path: str | Path) -> pd.DataFrame:
    """Load MovieLens 100K, 1M, or similarly formatted ratings files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {path}\n"
            "Download/extract MovieLens first, then pass the real ratings file path.\n"
            "Common paths are: data/ml-1m/ratings.dat, data/ml-100k/u.data, "
            "or data/ml-latest-small/ratings.csv."
        )

    if path.name == "ratings.csv":
        ratings = pd.read_csv(path)
        required = ["userId", "movieId", "rating", "timestamp"]
        missing = [column for column in required if column not in ratings.columns]
        if missing:
            raise ValueError(
                f"{path} is missing required MovieLens columns: {missing}"
            )
        return ratings[required]

    sep = "::" if path.name == "ratings.dat" else "\t"
    names = ["userId", "movieId", "rating", "timestamp"]
    return pd.read_csv(path, sep=sep, engine="python", names=names)


def natural_missing_fraction(ratings: pd.DataFrame) -> float:
    """Return missingness of the observed MovieLens utility matrix."""
    total_cells = ratings.userId.nunique() * ratings.movieId.nunique()
    return 1.0 - (len(ratings) / total_cells)


def simulate_observed_rating_mask(
    ratings: pd.DataFrame,
    missing_fraction: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Remove a fraction of observed ratings.

    This controls sparsity relative to the available MovieLens ratings, not the
    full user x item grid, which is already naturally sparse.
    """
    rng = np.random.default_rng(seed)
    keep_count = int(len(ratings) * (1.0 - missing_fraction))
    kept_positions = rng.choice(len(ratings), size=keep_count, replace=False)
    return ratings.iloc[kept_positions].reset_index(drop=True)


def split_train_test(
    ratings: pd.DataFrame,
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(ratings, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_matrix(train: pd.DataFrame) -> MatrixData:
    user_map = {user_id: idx for idx, user_id in enumerate(train.userId.unique())}
    item_map = {movie_id: idx for idx, movie_id in enumerate(train.movieId.unique())}
    rows = train.userId.map(user_map).to_numpy()
    cols = train.movieId.map(item_map).to_numpy()
    matrix = csr_matrix(
        (train.rating.to_numpy(dtype=float), (rows, cols)),
        shape=(len(user_map), len(item_map)),
    )
    return MatrixData(matrix=matrix, user_map=user_map, item_map=item_map)


def test_pairs(
    test: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
) -> tuple[list[tuple[int, int]], np.ndarray, int]:
    pairs: list[tuple[int, int]] = []
    truth: list[float] = []
    skipped = 0

    for row in test.itertuples(index=False):
        user_idx = user_map.get(row.userId)
        item_idx = item_map.get(row.movieId)
        if user_idx is None or item_idx is None:
            skipped += 1
            continue
        pairs.append((user_idx, item_idx))
        truth.append(float(row.rating))

    return pairs, np.asarray(truth, dtype=float), skipped


def evaluate_model(
    model,
    test: pd.DataFrame,
    user_map: dict[int, int],
    item_map: dict[int, int],
) -> EvaluationResult:
    pairs, truth, skipped = test_pairs(test, user_map, item_map)
    if not pairs:
        return EvaluationResult(
            rmse=float("nan"),
            inference_time=0.0,
            evaluated_count=0,
            skipped_count=skipped,
            diagnostics={},
        )

    if hasattr(model, "reset_diagnostics"):
        model.reset_diagnostics()

    started_at = perf_counter()
    predictions = np.asarray(model.predict_batch(pairs), dtype=float)
    inference_time = perf_counter() - started_at
    rmse = float(np.sqrt(np.mean((truth - predictions) ** 2)))
    diagnostics: dict[str, float] = {}
    if hasattr(model, "get_diagnostics"):
        diagnostics = dict(model.get_diagnostics())

    return EvaluationResult(
        rmse=rmse,
        inference_time=inference_time,
        evaluated_count=len(pairs),
        skipped_count=skipped,
        diagnostics=diagnostics,
    )


def run_method_over_sparsity(
    ratings: pd.DataFrame,
    model_factory,
    method_name: str,
    levels: Iterable[float] = SPARSITY_LEVELS,
    labels: Iterable[str] = SPARSITY_LABELS,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []

    for level, label in zip(levels, labels):
        masked = simulate_observed_rating_mask(ratings, level, seed=seed)
        train, test = split_train_test(masked, seed=seed)
        matrix_data = build_matrix(train)

        baseline = GlobalMeanBaseline().fit(matrix_data.matrix)
        baseline_evaluation = evaluate_model(
            baseline,
            test,
            matrix_data.user_map,
            matrix_data.item_map,
        )

        model = model_factory()
        started_at = perf_counter()
        model.fit(matrix_data.matrix)
        fit_time = perf_counter() - started_at

        evaluation = evaluate_model(
            model,
            test,
            matrix_data.user_map,
            matrix_data.item_map,
        )

        density = matrix_data.matrix.nnz / (
            matrix_data.matrix.shape[0] * matrix_data.matrix.shape[1]
        )
        total_test = evaluation.evaluated_count + evaluation.skipped_count
        skip_rate = evaluation.skipped_count / total_test if total_test else 0.0

        row: dict[str, Any] = {
            "method": method_name,
            "label": label,
            "observed_ratings_removed": level,
            "matrix_density": density,
            "actual_matrix_missing": 1.0 - density,
            "n_train": len(train),
            "n_test": len(test),
            "n_evaluated": evaluation.evaluated_count,
            "n_skipped": evaluation.skipped_count,
            "skip_rate": skip_rate,
            "rmse": evaluation.rmse,
            "baseline_rmse": baseline_evaluation.rmse,
            "rmse_vs_baseline": evaluation.rmse - baseline_evaluation.rmse,
            "beats_baseline": evaluation.rmse < baseline_evaluation.rmse,
            "fit_time": fit_time,
            "inference_time": evaluation.inference_time,
            "total_time": fit_time + evaluation.inference_time,
        }
        row.update(evaluation.diagnostics)

        rows.append(row)

    return pd.DataFrame(rows)
