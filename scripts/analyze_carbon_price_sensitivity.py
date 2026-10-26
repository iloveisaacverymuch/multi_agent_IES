from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume

from pareto_style import PARETO_COLORS, apply_pareto_rcparams, scatter_pareto, style_pareto_axis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

PRICE_RUNS = [
    (
        0.00,
        PROJECT_ROOT
        / "outputs"
        / "runs"
        / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_00",
    ),
    (
        0.02,
        PROJECT_ROOT
        / "outputs"
        / "runs"
        / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02",
    ),
    (
        0.05,
        PROJECT_ROOT
        / "outputs"
        / "runs"
        / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_05",
    ),
    (
        0.08,
        PROJECT_ROOT
        / "outputs"
        / "runs"
        / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_08",
    ),
    (
        0.10,
        PROJECT_ROOT
        / "outputs"
        / "runs"
        / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_10",
    ),
]


def load_objectives(run_dir):
    objectives = np.loadtxt(run_dir / "final" / "objs.txt", delimiter=",")
    return np.atleast_2d(objectives)


def pareto_indices(objectives):
    dominated = np.zeros(len(objectives), dtype=bool)
    for i in range(len(objectives)):
        for j in range(len(objectives)):
            if i == j:
                continue
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


def unique_front(objectives, decimals=2):
    front = objectives[pareto_indices(objectives)]
    rounded = np.round(front, decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    front = front[np.sort(idx)]
    return front[np.argsort(front[:, 0])]


def sparsity(front):
    if len(front) < 2:
        return 0.0
    sorted_front = front[np.argsort(front[:, 0])]
    diffs = np.diff(sorted_front, axis=0)
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


def read_elapsed(run_dir):
    elapsed_path = run_dir / "elapsed_seconds.txt"
    if elapsed_path.exists():
        return float(elapsed_path.read_text(encoding="utf-8").strip())
    log_path = run_dir / "train.log"
    if log_path.exists():
        return max(log_path.stat().st_mtime - log_path.stat().st_ctime, 0.0)
    return np.nan


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = [(price, run_dir, load_objectives(run_dir)) for price, run_dir in PRICE_RUNS]
    all_objectives = np.vstack([item[2] for item in loaded])
    mins = all_objectives.min(axis=0)
    maxs = all_objectives.max(axis=0)
    ranges = np.where(maxs - mins <= 1e-12, 1.0, maxs - mins)
    ref_point = maxs + 0.1 * ranges

    rows = []
    fronts = []
    for price, run_dir, objectives in loaded:
        front = unique_front(objectives)
        front_norm = (front - mins) / ranges
        fronts.append((price, front))
        rows.append(
            {
                "carbon_price": price,
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "num_points": len(objectives),
                "num_unique_front_points": len(front),
                "min_cost": objectives[:, 0].min(),
                "min_carbon": objectives[:, 1].min(),
                "cost_range": objectives[:, 0].max() - objectives[:, 0].min(),
                "carbon_range": objectives[:, 1].max() - objectives[:, 1].min(),
                "hypervolume": float(Hypervolume(ref_point=ref_point).do(front)),
                "normalized_hypervolume": float(Hypervolume(ref_point=np.array([1.1, 1.1])).do(front_norm)),
                "sparsity": sparsity(front),
                "normalized_sparsity": sparsity(front_norm),
                "elapsed_seconds": read_elapsed(run_dir),
            }
        )
    summary = pd.DataFrame(rows).sort_values("carbon_price")
    csv_path = OUTPUT_DIR / "carbon_price_sensitivity_summary.csv"
    summary.to_csv(csv_path, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.2), constrained_layout=True)
    axes[0].plot(summary["carbon_price"], summary["normalized_hypervolume"], marker="o", color="#4C78A8")
    axes[0].set_ylabel("Normalized HV")
    axes[0].set_xlabel("CDA carbon price")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    axes[1].plot(summary["carbon_price"], summary["min_cost"] / 1e6, marker="s", label="Minimum cost", color="#F28E2B")
    axes[1].plot(summary["carbon_price"], summary["min_carbon"] / 1e6, marker="^", label="Minimum carbon", color="#59A14F")
    axes[1].set_xlabel("CDA carbon price")
    axes[1].set_ylabel("Objective value (million)")
    axes[1].legend(frameon=False)
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    axes[2].plot(summary["carbon_price"], summary["num_unique_front_points"], marker="D", color="#E15759")
    axes[2].set_xlabel("CDA carbon price")
    axes[2].set_ylabel("Unique front points")
    axes[2].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(OUTPUT_DIR / "carbon_price_sensitivity_metrics.png", bbox_inches="tight")
    plt.close(fig)

    apply_pareto_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    handles = []
    for idx, (price, front) in enumerate(fronts):
        handles.append(scatter_pareto(ax, front[:, 0], front[:, 1], f"{price:.2f}", PARETO_COLORS[idx]))
    all_fronts = np.vstack([front for _, front in fronts])
    style_pareto_axis(ax, all_fronts[:, 0], all_fronts[:, 1], handles, legend_title="Carbon price")
    fig.savefig(OUTPUT_DIR / "carbon_price_sensitivity_pareto.png", bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
