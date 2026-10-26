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

RUNS = [
    (
        "standard_contribution",
        "Standard CDA + Contribution",
        PROJECT_ROOT / "outputs" / "runs" / "mappo_standard_cda_contribution_full",
    ),
    (
        "carbon_contribution",
        "Carbon CDA + Contribution",
        PROJECT_ROOT / "outputs" / "runs" / "carbon_cda_contribution_price_0_02",
    ),
    (
        "standard_ensemble",
        "Standard CDA + Ensemble-Uncertainty",
        PROJECT_ROOT / "outputs" / "runs" / "standard_cda_ensemble_uncertainty_capped32_kappa_0_20",
    ),
    (
        "carbon_ensemble",
        "Carbon CDA + Ensemble-Uncertainty (kappa=0.20)",
        PROJECT_ROOT
        / "outputs"
        / "runs"
        / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02",
    ),
]


def load_objectives(run_dir):
    path = run_dir / "final" / "objs.txt"
    objectives = np.loadtxt(path, delimiter=",")
    objectives = np.atleast_2d(objectives)
    if objectives.shape[1] != 2:
        raise ValueError(f"Expected two objectives in {path}, got {objectives.shape}.")
    if not np.isfinite(objectives).all():
        raise ValueError(f"Non-finite objective value found in {path}.")
    return objectives


def pareto_indices(objectives):
    n = len(objectives)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


def unique_sorted_front(objectives, decimals=2):
    front = objectives[pareto_indices(objectives)]
    rounded = np.round(front, decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    front = front[np.sort(idx)]
    return front[np.argsort(front[:, 0])]


def normalize(values, mins, ranges):
    return (values - mins) / ranges


def igd(front_norm, reference_norm):
    distances = np.linalg.norm(reference_norm[:, None, :] - front_norm[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def spacing(front_norm):
    if len(front_norm) < 2:
        return 0.0
    dist = np.linalg.norm(front_norm[:, None, :] - front_norm[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    return float(np.sqrt(np.mean((nearest - nearest.mean()) ** 2)))


def sparsity(front):
    if len(front) < 2:
        return 0.0
    sorted_front = front[np.argsort(front[:, 0])]
    diffs = np.diff(sorted_front, axis=0)
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


def spread_delta(front_norm, reference_norm):
    if len(front_norm) < 2:
        return 1.0
    front = front_norm[np.argsort(front_norm[:, 0])]
    gaps = np.linalg.norm(np.diff(front, axis=0), axis=1)
    gap_mean = gaps.mean()
    if gap_mean <= 1e-12:
        return 1.0
    ref_sorted = reference_norm[np.argsort(reference_norm[:, 0])]
    df = np.linalg.norm(front[0] - ref_sorted[0])
    dl = np.linalg.norm(front[-1] - ref_sorted[-1])
    numerator = df + dl + np.sum(np.abs(gaps - gap_mean))
    denominator = df + dl + len(gaps) * gap_mean
    return float(numerator / max(denominator, 1e-12))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = [(key, label, path, load_objectives(path)) for key, label, path in RUNS]
    all_objectives = np.vstack([item[3] for item in loaded])
    mins = all_objectives.min(axis=0)
    maxs = all_objectives.max(axis=0)
    ranges = np.where(maxs - mins <= 1e-12, 1.0, maxs - mins)
    ref_point = maxs + 0.1 * ranges

    reference_front = unique_sorted_front(all_objectives)
    reference_norm = normalize(reference_front, mins, ranges)

    rows = []
    fronts = []
    for key, label, run_dir, objectives in loaded:
        front = unique_sorted_front(objectives)
        front_norm = normalize(front, mins, ranges)
        hv = float(Hypervolume(ref_point=ref_point).do(front))
        hv_norm = float(Hypervolume(ref_point=np.array([1.1, 1.1])).do(front_norm))
        fronts.append((key, label, front))
        rows.append(
            {
                "method_key": key,
                "method": label,
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "num_points": len(objectives),
                "num_unique_front_points": len(front),
                "min_cost": objectives[:, 0].min(),
                "min_carbon": objectives[:, 1].min(),
                "cost_range": objectives[:, 0].max() - objectives[:, 0].min(),
                "carbon_range": objectives[:, 1].max() - objectives[:, 1].min(),
                "hypervolume": hv,
                "normalized_hypervolume": hv_norm,
                "igd": igd(front_norm, reference_norm),
                "spacing": spacing(front_norm),
                "sparsity": sparsity(front),
                "normalized_sparsity": sparsity(front_norm),
                "spread_delta": spread_delta(front_norm, reference_norm),
            }
        )

    summary = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "additional_ablation_metrics.csv"
    summary.to_csv(csv_path, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.2), constrained_layout=True)
    plot_specs = [
        ("normalized_hypervolume", "Normalized HV", "higher"),
        ("igd", "IGD", "lower"),
        ("normalized_sparsity", "Sparsity", "lower"),
        ("spread_delta", "Spread Delta", "lower"),
    ]
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]
    for ax, (metric, ylabel, direction) in zip(axes, plot_specs):
        ax.bar(summary["method_key"], summary[metric], color=colors, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} ({direction} is better)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(OUTPUT_DIR / "additional_ablation_metrics.png", bbox_inches="tight")
    plt.close(fig)

    apply_pareto_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    handles = []
    for idx, (_, label, front) in enumerate(fronts):
        handles.append(scatter_pareto(ax, front[:, 0], front[:, 1], label, PARETO_COLORS[idx]))
    all_fronts = np.vstack([front for _, _, front in fronts])
    style_pareto_axis(ax, all_fronts[:, 0], all_fronts[:, 1], handles)
    fig.savefig(OUTPUT_DIR / "pareto_compare_ablation_new_main.png", bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
