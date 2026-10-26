from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"

METHODS = [
    (
        "standard_prediction",
        "Standard CDA + Prediction",
        {44: RUN_ROOT / "mappo_raw_reward_full"},
    ),
    (
        "carbon_cda_only",
        "Carbon CDA Only",
        {44: RUN_ROOT / "mappo_carbon_aware_cda_full"},
    ),
    (
        "standard_contribution",
        "Standard CDA + Contribution",
        {44: RUN_ROOT / "mappo_standard_cda_contribution_full"},
    ),
    (
        "carbon_contribution",
        "Carbon CDA + Contribution",
        {44: RUN_ROOT / "mappo_carbon_cda_contribution_full"},
    ),
]

SEEDS = [11, 22, 33, 44, 55]


def seed_dir(method_key, seed, existing):
    return existing.get(seed, RUN_ROOT / f"seed_stability_{method_key}_seed_{seed}")


def load_objectives(run_dir):
    path = run_dir / "final" / "objs.txt"
    if not path.exists():
        return None
    return np.atleast_2d(np.loadtxt(path, delimiter=","))


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
    return np.nan


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = []
    for method_key, method, existing in METHODS:
        for seed in SEEDS:
            run_dir = seed_dir(method_key, seed, existing)
            objectives = load_objectives(run_dir)
            if objectives is not None:
                loaded.append((method_key, method, seed, run_dir, objectives))

    if not loaded:
        raise RuntimeError("No seed stability runs found.")

    all_objectives = np.vstack([item[4] for item in loaded])
    mins = all_objectives.min(axis=0)
    maxs = all_objectives.max(axis=0)
    ranges = np.where(maxs - mins <= 1e-12, 1.0, maxs - mins)
    ref_point = maxs + 0.1 * ranges

    rows = []
    for method_key, method, seed, run_dir, objectives in loaded:
        front = unique_front(objectives)
        front_norm = (front - mins) / ranges
        rows.append(
            {
                "method_key": method_key,
                "method": method,
                "seed": seed,
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

    detail = pd.DataFrame(rows)
    detail_path = OUTPUT_DIR / "seed_stability_detail.csv"
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby(["method_key", "method"], as_index=False)
        .agg(
            completed_seeds=("seed", "count"),
            hv_mean=("normalized_hypervolume", "mean"),
            hv_std=("normalized_hypervolume", "std"),
            sparsity_mean=("normalized_sparsity", "mean"),
            sparsity_std=("normalized_sparsity", "std"),
            min_cost_mean=("min_cost", "mean"),
            min_cost_std=("min_cost", "std"),
            min_carbon_mean=("min_carbon", "mean"),
            min_carbon_std=("min_carbon", "std"),
            front_points_mean=("num_unique_front_points", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
        )
        .sort_values("hv_mean", ascending=False)
    )
    summary_path = OUTPUT_DIR / "seed_stability_summary.csv"
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
    order = [item[0] for item in METHODS]
    labels = {item[0]: item[1] for item in METHODS}
    plot_data = summary.set_index("method_key").reindex(order).reset_index()
    x = np.arange(len(plot_data))
    yerr = plot_data["hv_std"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(6.6, 3.6), constrained_layout=True)
    ax.bar(x, plot_data["hv_mean"], yerr=yerr, color="#4C78A8", edgecolor="black", linewidth=0.6, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[key] for key in plot_data["method_key"]], rotation=30, ha="right")
    ax.set_ylabel("Normalized HV")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(OUTPUT_DIR / "seed_stability_hv.png", bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
