from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume

from pareto_style import PARETO_COLORS, apply_pareto_rcparams, scatter_pareto, style_pareto_axis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
PROPOSED_RUN_DIR = RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02"

RUNS = [
    ("proposed", "Carbon CDA + Ensemble-Uncertainty (kappa=0.20)", PROPOSED_RUN_DIR),
    ("fixed_weight_mappo", "Fixed-weight MAPPO", RUN_ROOT / "baseline_fixed_weight_mappo"),
    ("masac", "MASAC", RUN_ROOT / "baseline_masac"),
    ("maddpg", "MADDPG", RUN_ROOT / "baseline_maddpg"),
    ("mopso", "MOPSO", RUN_ROOT / "baseline_mopso"),
]


def load_objectives(run_dir):
    obj_path = run_dir / "final" / "objs.txt"
    if not obj_path.exists():
        return None
    objectives = np.loadtxt(obj_path, delimiter=",")
    objectives = np.atleast_2d(objectives)
    if objectives.shape[1] != 2:
        raise ValueError(f"Expected two objectives in {obj_path}, got {objectives.shape}")
    return objectives


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


def igd(front_norm, reference_norm):
    dist = np.linalg.norm(reference_norm[:, None, :] - front_norm[None, :, :], axis=2)
    return float(np.mean(np.min(dist, axis=1)))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = []
    for key, label, run_dir in RUNS:
        objectives = load_objectives(run_dir)
        if objectives is not None:
            loaded.append((key, label, run_dir, objectives))
        else:
            print(f"Skipping missing run: {label} ({run_dir})")
    if len(loaded) < 2:
        raise RuntimeError("Need at least two completed runs for comparison.")

    all_objectives = np.vstack([item[3] for item in loaded])
    mins = all_objectives.min(axis=0)
    maxs = all_objectives.max(axis=0)
    ranges = np.where(maxs - mins <= 1e-12, 1.0, maxs - mins)
    ref_point = maxs + 0.1 * ranges
    normalized_ref = np.array([1.1, 1.1])
    reference_front = unique_front(all_objectives)
    reference_norm = (reference_front - mins) / ranges

    rows = []
    fronts = []
    for key, label, run_dir, objectives in loaded:
        front = unique_front(objectives)
        front_norm = (front - mins) / ranges
        fronts.append((key, label, front))
        rows.append(
            {
                "method_key": key,
                "method": label,
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "num_points": len(objectives),
                "num_front_points": len(front),
                "min_cost": float(objectives[:, 0].min()),
                "min_carbon": float(objectives[:, 1].min()),
                "hypervolume": float(Hypervolume(ref_point=ref_point).do(front)),
                "normalized_hypervolume": float(Hypervolume(ref_point=normalized_ref).do(front_norm)),
                "igd": igd(front_norm, reference_norm),
                "spacing": spacing(front_norm),
                "sparsity": sparsity(front),
                "normalized_sparsity": sparsity(front_norm),
            }
        )
    summary = pd.DataFrame(rows).sort_values("normalized_hypervolume", ascending=False)
    summary_path = OUTPUT_DIR / "algorithm_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    apply_pareto_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {
        "proposed": PARETO_COLORS[0],
        "fixed_weight_mappo": PARETO_COLORS[1],
        "masac": PARETO_COLORS[2],
        "maddpg": PARETO_COLORS[3],
        "mopso": PARETO_COLORS[4],
    }
    handles = []
    for key, label, front in fronts:
        handles.append(scatter_pareto(ax, front[:, 0], front[:, 1], label, colors.get(key)))
    all_fronts = np.vstack([front for _, _, front in fronts])
    style_pareto_axis(ax, all_fronts[:, 0], all_fronts[:, 1], handles)
    fig.savefig(OUTPUT_DIR / "algorithm_comparison_pareto.png", bbox_inches="tight")
    plt.close(fig)

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
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ordered = summary.sort_values("normalized_hypervolume", ascending=False)
    ax.bar(
        ordered["method"],
        ordered["normalized_hypervolume"],
        color=[colors.get(key, "#777777") for key in ordered["method_key"]],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_ylabel("Normalized HV")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.65)
    fig.savefig(OUTPUT_DIR / "algorithm_comparison_hv.png", bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
