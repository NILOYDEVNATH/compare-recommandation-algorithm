"""Method 2: item-item collaborative filtering with cosine similarity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from data_pipeline import load_movielens, run_method_over_sparsity


class ItemItemCF:
    """Item-item CF using cosine similarity between item rating vectors."""

    def __init__(self, k: int = 20):
        self.k = k
        self.reset_diagnostics()

    def fit(self, matrix: csr_matrix) -> "ItemItemCF":
        self.matrix = matrix.tocsr().astype(np.float32)
        self.dense = self.matrix.toarray()
        self.global_mean = float(self.matrix.data.mean())

        rating_sums = np.asarray(self.matrix.sum(axis=1)).ravel()
        rating_counts = np.diff(self.matrix.indptr)
        self.user_means = np.divide(
            rating_sums,
            rating_counts,
            out=np.full_like(rating_sums, self.global_mean, dtype=np.float32),
            where=rating_counts > 0,
        )

        item_matrix = self.matrix.T.tocsr()
        norms = np.sqrt(item_matrix.multiply(item_matrix).sum(axis=1)).A1
        safe_norms = np.where(norms > 0, norms, 1.0)
        normalized_items = item_matrix.multiply(1.0 / safe_norms[:, None])

        self.similarity = (normalized_items @ normalized_items.T).toarray()
        np.fill_diagonal(self.similarity, 0.0)
        return self

    def reset_diagnostics(self) -> None:
        self.prediction_count = 0
        self.fallback_count = 0
        self.underfilled_count = 0
        self.neighbour_count_sum = 0

    def get_diagnostics(self) -> dict[str, float]:
        if self.prediction_count == 0:
            return {
                "fallback_rate": 0.0,
                "underfilled_neighbour_rate": 0.0,
                "avg_neighbours_used": 0.0,
            }

        return {
            "fallback_rate": self.fallback_count / self.prediction_count,
            "underfilled_neighbour_rate": self.underfilled_count
            / self.prediction_count,
            "avg_neighbours_used": self.neighbour_count_sum / self.prediction_count,
        }

    def _predict_one(self, user_idx: int, item_idx: int) -> float:
        self.prediction_count += 1

        user_ratings = self.dense[user_idx]
        rated_items = np.flatnonzero(user_ratings > 0)
        if len(rated_items) == 0:
            self.fallback_count += 1
            self.underfilled_count += 1
            return self.global_mean

        similarities = self.similarity[item_idx, rated_items]
        positive_mask = similarities > 0
        if not positive_mask.any():
            self.fallback_count += 1
            self.underfilled_count += 1
            return float(self.user_means[user_idx])

        rated_items = rated_items[positive_mask]
        similarities = similarities[positive_mask]

        if len(similarities) > self.k:
            top_positions = np.argpartition(similarities, -self.k)[-self.k :]
            rated_items = rated_items[top_positions]
            similarities = similarities[top_positions]

        neighbour_count = len(similarities)
        self.neighbour_count_sum += neighbour_count
        if neighbour_count < self.k:
            self.underfilled_count += 1

        weight_sum = similarities.sum()
        if weight_sum == 0:
            self.fallback_count += 1
            return float(self.user_means[user_idx])

        prediction = np.dot(similarities, user_ratings[rated_items]) / weight_sum
        return float(np.clip(prediction, 1.0, 5.0))

    def predict_batch(self, pairs: list[tuple[int, int]]) -> list[float]:
        return [self._predict_one(user_idx, item_idx) for user_idx, item_idx in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Method 2 item-item CF.")
    parser.add_argument("ratings_path", help="Path to MovieLens ratings file.")
    parser.add_argument("--k", type=int, default=20, help="Neighbourhood size.")
    parser.add_argument(
        "--output",
        default="results/method2_item_item_cf.csv",
        help="CSV path for experiment results.",
    )
    args = parser.parse_args()

    ratings = load_movielens(args.ratings_path)
    results = run_method_over_sparsity(
        ratings,
        model_factory=lambda: ItemItemCF(k=args.k),
        method_name=f"Item-Item CF (cosine, k={args.k})",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
