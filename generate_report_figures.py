from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")

import matplotlib.pyplot as plt


METHOD_LABELS = {
    "User-User CF (Pearson, k=20)": "User-User CF",
    "Item-Item CF (cosine, k=20)": "Item-Item CF",
    "Latent Factor SGD (factors=20)": "Latent Factor",
}

METHOD_ORDER = [
    "User-User CF (Pearson, k=20)",
    "Item-Item CF (cosine, k=20)",
    "Latent Factor SGD (factors=20)",
]

SPARSITY_ORDER = ["50%", "70%", "90%", "95%", "99%"]
COLORS = {
    "user": "#4F88D9",
    "item": "#4C9A7A",
    "latent": "#C66D48",
    "baseline": "#F4E9E0",
    "header": "#173B91",
    "green": "#62C777",
    "red": "#D85A49",
}


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    results = pd.read_csv(path)
    results["method_short"] = results["method"].map(METHOD_LABELS)
    return results


def ordered_method(results: pd.DataFrame, method: str) -> pd.DataFrame:
    return (
        results[results["method"] == method]
        .set_index("label")
        .loc[SPARSITY_ORDER]
        .reset_index()
    )


def save(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_user_fallback_fill(results: pd.DataFrame, output_dir: Path) -> None:
    user = ordered_method(results, METHOD_ORDER[0])
    x = np.arange(len(SPARSITY_ORDER))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(
        x - width / 2,
        user["fallback_rate"] * 100,
        width,
        label="Fallback rate (%)",
        color=COLORS["user"],
    )
    bars2 = ax2.bar(
        x + width / 2,
        user["avg_neighbours_used"],
        width,
        label="Avg neighbours used",
        color="#AFCBE5",
    )

    ax1.set_title("User-User CF: fallback rate and neighbourhood fill")
    ax1.set_xlabel("Sparsity level")
    ax1.set_ylabel("Fallback rate (%)", color=COLORS["user"])
    ax2.set_ylabel("Avg neighbours used (k=20)", color="#3567B0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SPARSITY_ORDER)
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 20.5)
    ax1.grid(axis="y", alpha=0.25)

    handles = [bars1, bars2]
    labels = [handle.get_label() for handle in handles]
    ax1.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
    )

    save(fig, output_dir, "figure_01_user_user_fallback_fill.png")


def plot_baseline_heatmap(results: pd.DataFrame, output_dir: Path) -> None:
    matrix = []
    for method in METHOD_ORDER:
        group = ordered_method(results, method)
        matrix.append(group["beats_baseline"].astype(int).to_numpy())
    matrix_array = np.asarray(matrix)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    cmap = ListedColormap([COLORS["red"], COLORS["green"]])
    ax.imshow(matrix_array, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_title("Does each method beat the global mean baseline?")
    ax.set_xticks(np.arange(len(SPARSITY_ORDER)))
    ax.set_xticklabels(SPARSITY_ORDER)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABELS[method] for method in METHOD_ORDER])
    ax.set_xticks(np.arange(-0.5, len(SPARSITY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(METHOD_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for y in range(matrix_array.shape[0]):
        for x in range(matrix_array.shape[1]):
            text = r"$\checkmark$" if matrix_array[y, x] else "x"
            ax.text(x, y, text, ha="center", va="center", color="white", fontsize=26)

    for spine in ax.spines.values():
        spine.set_visible(False)

    save(fig, output_dir, "figure_02_baseline_heatmap.png")


def plot_rmse_table(results: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    row_labels = ["User-User CF", "Item-Item CF", "Latent Factor", "Baseline"]

    for method in METHOD_ORDER:
        group = ordered_method(results, method)
        cells = []
        for row in group.itertuples(index=False):
            mark = r"$\checkmark$" if row.beats_baseline else "x"
            if row.method == METHOD_ORDER[0] and row.label == "99%":
                mark = r"$\checkmark$*"
            cells.append(f"{row.rmse:.3f} {mark}")
        rows.append(cells)

    baseline = (
        results.sort_values("observed_ratings_removed")
        .drop_duplicates("label")
        .set_index("label")
        .loc[SPARSITY_ORDER]
    )
    rows.append([f"{value:.3f}" for value in baseline["baseline_rmse"]])

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axis("off")
    table = ax.table(
        cellText=[[label, *cells] for label, cells in zip(row_labels, rows)],
        colLabels=["Method", *SPARSITY_ORDER],
        loc="center",
        cellLoc="center",
        colWidths=[0.27, 0.145, 0.145, 0.145, 0.145, 0.145],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#C8C8C8")
        if row == 0:
            cell.set_facecolor(COLORS["header"])
            cell.set_text_props(color="white", weight="bold")
        elif row == 2:
            cell.set_facecolor("#EEF2FA")
        elif row == 4:
            cell.set_facecolor(COLORS["baseline"])
        else:
            cell.set_facecolor("white")
        if col == 0:
            cell.set_text_props(ha="left", weight="bold")

    save(fig, output_dir, "figure_03_rmse_table.png")


def plot_underfilled_neighbourhood(results: pd.DataFrame, output_dir: Path) -> None:
    user = ordered_method(results, METHOD_ORDER[0])
    item = ordered_method(results, METHOD_ORDER[1])
    x = np.asarray([50, 70, 90, 95, 99], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    user_values = user["underfilled_neighbour_rate"] * 100
    item_values = item["underfilled_neighbour_rate"] * 100
    ax.plot(x, user_values, marker="o", linewidth=2, color=COLORS["user"], label="User-User CF")
    ax.plot(x, item_values, marker="s", linewidth=2, color=COLORS["item"], label="Item-Item CF")
    ax.fill_between(x, 0, np.maximum(user_values, item_values), color="#BFDDE1", alpha=0.6)
    ax.axhline(100, linestyle=":", linewidth=1.2, color="#E86B6B", label="100%")

    for xs, ys in [(x, user_values), (x, item_values)]:
        for x_pos, y_pos in zip(xs, ys):
            ax.annotate(
                f"{y_pos:.0f}%",
                (x_pos, y_pos),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

    ax.set_title("Fraction of predictions with fewer than k=20 neighbours")
    ax.set_xlabel("Sparsity (%)")
    ax.set_ylabel("Underfilled neighbourhood rate (%)")
    ax.set_xlim(48, 101)
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    save(fig, output_dir, "figure_04_underfilled_neighbourhood.png")


def plot_runtime(results: pd.DataFrame, output_dir: Path) -> None:
    x = np.arange(len(SPARSITY_ORDER))
    width = 0.12
    offsets = np.linspace(-2.5 * width, 2.5 * width, 6)

    series = [
        (METHOD_ORDER[0], "fit_time", "User-User CF (train)", COLORS["user"], 0.85),
        (METHOD_ORDER[0], "inference_time", "User-User CF (infer)", "#AFCBE5", 0.85),
        (METHOD_ORDER[1], "fit_time", "Item-Item CF (train)", COLORS["item"], 0.85),
        (METHOD_ORDER[1], "inference_time", "Item-Item CF (infer)", "#A8D4C2", 0.85),
        (METHOD_ORDER[2], "fit_time", "Latent Factor (train)", COLORS["latent"], 0.9),
        (METHOD_ORDER[2], "inference_time", "Latent Factor (infer)", "#E2A284", 0.9),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for offset, (method, column, label, color, alpha) in zip(offsets, series):
        group = ordered_method(results, method)
        ax.bar(x + offset, group[column], width, label=label, color=color, alpha=alpha)

    ax.set_yscale("log")
    ax.set_title("Training vs inference time per method")
    ax.set_xlabel("Sparsity level")
    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(SPARSITY_ORDER)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=8, loc="upper right")

    save(fig, output_dir, "figure_05_runtime_train_inference.png")


def plot_skip_rate(results: pd.DataFrame, output_dir: Path) -> None:
    skip = (
        results.sort_values("observed_ratings_removed")
        .drop_duplicates("label")
        .set_index("label")
        .loc[SPARSITY_ORDER]
    )
    values = skip["skip_rate"] * 100
    x = np.arange(len(SPARSITY_ORDER))

    fig, ax = plt.subplots(figsize=(8, 5.2))
    bars = ax.bar(x, values, color=COLORS["latent"], width=0.5)
    ax.set_title("Fraction of test ratings skipped due to unseen users/items")
    ax.set_xlabel("Sparsity level")
    ax.set_ylabel("Skip rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(SPARSITY_ORDER)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}%",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
        )

    ax.annotate(
        "At 99%: 27% of\ntest ratings skipped",
        xy=(x[-1], values.iloc[-1]),
        xytext=(x[-1] - 1.0, values.iloc[-1] * 0.72),
        arrowprops={"arrowstyle": "->", "color": "#E94434"},
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#E94434"},
        color="#E94434",
        fontsize=9,
    )

    save(fig, output_dir, "figure_06_skip_rate.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures.")
    parser.add_argument(
        "--results",
        default="results/combined_results.csv",
        help="Path to combined result CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Directory where figure PNGs will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(Path(args.results))

    plot_user_fallback_fill(results, output_dir)
    plot_baseline_heatmap(results, output_dir)
    plot_rmse_table(results, output_dir)
    plot_underfilled_neighbourhood(results, output_dir)
    plot_runtime(results, output_dir)
    plot_skip_rate(results, output_dir)

    print(f"Saved report figures to {output_dir}")


if __name__ == "__main__":
    main()
