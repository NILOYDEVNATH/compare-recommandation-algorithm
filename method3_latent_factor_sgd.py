from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from data_pipeline import load_movielens, run_method_over_sparsity


class LatentFactorSGD:
    # Biased matrix factorization: r_ui ~= mu + b_u + b_i + p_u dot q_i

    def __init__(
        self,
        n_factors: int = 20,
        n_epochs: int = 20,
        learning_rate: float = 0.01,
        regularization: float = 0.05,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.seed = seed

    def fit(self, matrix: csr_matrix) -> "LatentFactorSGD":
        coo = matrix.tocoo()
        self.n_users, self.n_items = matrix.shape
        self.global_mean = float(coo.data.mean())

        rng = np.random.default_rng(self.seed)
        self.user_factors = 0.1 * rng.normal(
            size=(self.n_users, self.n_factors)
        ).astype(np.float32)
        self.item_factors = 0.1 * rng.normal(
            size=(self.n_items, self.n_factors)
        ).astype(np.float32)
        self.user_biases = np.zeros(self.n_users, dtype=np.float32)
        self.item_biases = np.zeros(self.n_items, dtype=np.float32)

        users = coo.row.astype(np.int64)
        items = coo.col.astype(np.int64)
        ratings = coo.data.astype(np.float32)

        for _ in range(self.n_epochs):
            order = rng.permutation(len(ratings))
            for idx in order:
                user_idx = users[idx]
                item_idx = items[idx]
                rating = ratings[idx]

                user_vector = self.user_factors[user_idx].copy()
                item_vector = self.item_factors[item_idx].copy()
                prediction = (
                    self.global_mean
                    + self.user_biases[user_idx]
                    + self.item_biases[item_idx]
                    + np.dot(user_vector, item_vector)
                )
                error = rating - prediction

                self.user_biases[user_idx] += self.learning_rate * (
                    error - self.regularization * self.user_biases[user_idx]
                )
                self.item_biases[item_idx] += self.learning_rate * (
                    error - self.regularization * self.item_biases[item_idx]
                )
                self.user_factors[user_idx] += self.learning_rate * (
                    error * item_vector - self.regularization * user_vector
                )
                self.item_factors[item_idx] += self.learning_rate * (
                    error * user_vector - self.regularization * item_vector
                )

        return self

    def _predict_one(self, user_idx: int, item_idx: int) -> float:
        prediction = (
            self.global_mean
            + self.user_biases[user_idx]
            + self.item_biases[item_idx]
            + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )
        return float(np.clip(prediction, 1.0, 5.0))

    def predict_batch(self, pairs: list[tuple[int, int]]) -> list[float]:
        return [self._predict_one(user_idx, item_idx) for user_idx, item_idx in pairs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Method 3 latent factor SGD.")
    parser.add_argument("ratings_path", help="Path to MovieLens ratings file.")
    parser.add_argument("--factors", type=int, default=20, help="Latent factors.")
    parser.add_argument("--epochs", type=int, default=20, help="SGD epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--reg", type=float, default=0.05, help="Regularization.")
    parser.add_argument(
        "--output",
        default="results/method3_latent_factor_sgd.csv",
        help="CSV path for experiment results.",
    )
    args = parser.parse_args()

    ratings = load_movielens(args.ratings_path)
    results = run_method_over_sparsity(
        ratings,
        model_factory=lambda: LatentFactorSGD(
            n_factors=args.factors,
            n_epochs=args.epochs,
            learning_rate=args.lr,
            regularization=args.reg,
        ),
        method_name=f"Latent Factor SGD (factors={args.factors})",
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
