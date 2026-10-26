from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
SEEDS = [11, 22, 44]
METHODS = [
    ("proposed", "Carbon CDA + Ensemble-Uncertainty"),
    ("fixed_weight_mappo", "Fixed-weight MAPPO"),
]


def run_dir(method_key, seed):
    if method_key == "proposed":
        base = "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02"
    else:
        base = "baseline_fixed_weight_mappo"
    return RUN_ROOT / (base if seed == 44 else f"{base}_seed_{seed}")


def load_objectives(path):
    values = np.loadtxt(path, delimiter=",")
    return np.atleast_2d(values)


def unique_front(objectives, decimals=2):
    rounded = np.round(objectives, decimals=decimals)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    points = objectives[np.sort(indices)]
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    keep = []
    best_carbon = np.inf
    for index, point in enumerate(points):
        if point[1] < best_carbon:
            keep.append(index)
            best_carbon = point[1]
    return points[keep]


def igd(front, reference):
    distances = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def main():
    records = []
    all_objectives = []
    for method_key, method in METHODS:
        for seed in SEEDS:
            path = run_dir(method_key, seed)
            for generation in range(1, 13):
                objectives_path = path / str(generation) / "ep" / "objs.txt"
                if not objectives_path.exists():
                    raise FileNotFoundError(objectives_path)
                objectives = load_objectives(objectives_path)
                records.append((method_key, method, seed, generation, objectives))
                all_objectives.append(objectives)

    combined = np.vstack(all_objectives)
    minimum = combined.min(axis=0)
    ranges = combined.max(axis=0) - minimum
    ranges = np.where(ranges <= 1e-12, 1.0, ranges)
    reference = (unique_front(combined) - minimum) / ranges
    hypervolume = Hypervolume(ref_point=np.array([1.1, 1.1]))

    rows = []
    for method_key, method, seed, generation, objectives in records:
        front = unique_front(objectives)
        normalized = (front - minimum) / ranges
        rows.append(
            {
                "method_key": method_key,
                "method": method,
                "seed": seed,
                "generation": generation,
                "normalized_hypervolume": float(hypervolume.do(normalized)),
                "igd": igd(normalized, reference),
                "front_points": len(front),
            }
        )

    detail = pd.DataFrame(rows)
    detail_path = OUTPUT_DIR / "main_vs_fixed_convergence_3seed_detail.csv"
    detail.to_csv(detail_path, index=False)
    summary = (
        detail.groupby(["method_key", "method", "generation"], as_index=False)
        .agg(
            hv_mean=("normalized_hypervolume", "mean"),
            hv_std=("normalized_hypervolume", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            front_points_mean=("front_points", "mean"),
            front_points_std=("front_points", "std"),
        )
    )
    summary_path = OUTPUT_DIR / "main_vs_fixed_convergence_3seed_summary.csv"
    summary.to_csv(summary_path, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), constrained_layout=True)
    specifications = [
        ("hv_mean", "hv_std", "Normalized HV"),
        ("igd_mean", "igd_std", "IGD"),
        ("front_points_mean", "front_points_std", "Pareto front points"),
    ]
    colors = {"proposed": "#C44E52", "fixed_weight_mappo": "#4C78A8"}
    for method_key, method in METHODS:
        selected = summary[summary["method_key"] == method_key].sort_values("generation")
        x = selected["generation"].to_numpy(dtype=float)
        for ax, (mean_column, std_column, ylabel) in zip(axes, specifications):
            mean = selected[mean_column].to_numpy(dtype=float)
            std = selected[std_column].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, mean, marker="o", markersize=3.5, linewidth=1.7, color=colors[method_key], label=method)
            ax.fill_between(x, mean - std, mean + std, color=colors[method_key], alpha=0.14, linewidth=0)
            ax.set_xlabel("Generation")
            ax.set_ylabel(ylabel)
            ax.set_xticks(np.arange(1, 13))
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[0].legend(frameon=False)
    figure_path = OUTPUT_DIR / "main_vs_fixed_convergence_3seed.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    final_rows = summary[summary["generation"] == 12]
    print(final_rows.to_string(index=False))
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
