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

RUNS = [
    ("proposed", "Carbon CDA + Contribution", RUN_ROOT / "carbon_cda_contribution_price_0_02"),
    (
        "uncertainty",
        "Carbon CDA + Uncertainty-Contribution",
        RUN_ROOT / "carbon_cda_uncertainty_contribution_price_0_02",
    ),
    (
        "ensemble_uncertainty",
        "Carbon CDA + Ensemble-Uncertainty",
        RUN_ROOT / "carbon_cda_ensemble_uncertainty_price_0_02",
    ),
    ("fixed_weight_mappo", "Fixed-weight MAPPO", RUN_ROOT / "baseline_fixed_weight_mappo"),
    ("masac", "MASAC", RUN_ROOT / "baseline_masac"),
    ("mopso", "MOPSO", RUN_ROOT / "baseline_mopso"),
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


def spacing(front_norm):
    if len(front_norm) < 2:
        return 0.0
    distances = np.linalg.norm(front_norm[:, None, :] - front_norm[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(np.sqrt(np.mean((nearest - nearest.mean()) ** 2)))


def sparsity(front):
    if len(front) < 2:
        return 0.0
    sorted_front = front[np.argsort(front[:, 0])]
    diffs = np.diff(sorted_front, axis=0)
    return float(np.mean(np.sum(diffs ** 2, axis=1)))


def igd(front_norm, reference_norm):
    distances = np.linalg.norm(reference_norm[:, None, :] - front_norm[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def read_elapsed(run_dir):
    elapsed = run_dir / "elapsed_seconds.txt"
    if elapsed.exists():
        return float(elapsed.read_text(encoding="utf-8").strip())
    return np.nan


def weight_summary(run_dir):
    rows = []
    for weights_path in sorted(run_dir.glob("[0-9]*/elites/weights.txt"), key=lambda p: int(p.parts[-3])):
        generation = int(weights_path.parts[-3])
        weights = np.loadtxt(weights_path, delimiter=",")
        weights = np.atleast_2d(weights)
        for weight in weights:
            rows.append(
                {
                    "generation": generation,
                    "weight_cost": float(weight[0]),
                    "weight_carbon": float(weight[1]),
                }
            )
    if not rows:
        return pd.DataFrame(), {}
    detail = pd.DataFrame(rows)
    rounded = detail[["weight_cost", "weight_carbon"]].round(4).drop_duplicates()
    summary = {
        "selected_weight_records": len(detail),
        "unique_selected_weights": len(rounded),
        "mean_weight_cost": detail["weight_cost"].mean(),
        "std_weight_cost": detail["weight_cost"].std(ddof=0),
        "min_weight_cost": detail["weight_cost"].min(),
        "max_weight_cost": detail["weight_cost"].max(),
    }
    return detail, summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = [(key, label, run_dir, load_objectives(run_dir)) for key, label, run_dir in RUNS]
    all_objectives = np.vstack([item[3] for item in loaded])
    mins = all_objectives.min(axis=0)
    ranges = np.max(all_objectives, axis=0) - mins
    ranges = np.where(ranges <= 1e-12, 1.0, ranges)
    ref_point = np.max(all_objectives, axis=0) + 0.1 * ranges
    reference_front = unique_front(all_objectives)
    reference_norm = (reference_front - mins) / ranges

    rows = []
    fronts = []
    weight_rows = []
    for key, label, run_dir, objectives in loaded:
        front = unique_front(objectives)
        front_norm = (front - mins) / ranges
        fronts.append((key, label, front))
        weight_detail, weight_stats = weight_summary(run_dir)
        if not weight_detail.empty:
            weight_detail.insert(0, "method_key", key)
            weight_detail.insert(1, "method", label)
            weight_rows.append(weight_detail)
        rows.append(
            {
                "method_key": key,
                "method": label,
                "num_points": len(objectives),
                "num_front_points": len(front),
                "min_cost": float(objectives[:, 0].min()),
                "min_carbon": float(objectives[:, 1].min()),
                "normalized_hypervolume": float(
                    Hypervolume(ref_point=np.array([1.1, 1.1])).do(front_norm)
                ),
                "hypervolume": float(Hypervolume(ref_point=ref_point).do(front)),
                "igd": igd(front_norm, reference_norm),
                "spacing": spacing(front_norm),
                "sparsity": sparsity(front),
                "normalized_sparsity": sparsity(front_norm),
                "elapsed_hours": read_elapsed(run_dir) / 3600.0,
                **weight_stats,
            }
        )

    summary = pd.DataFrame(rows).sort_values("normalized_hypervolume", ascending=False)
    summary_path = OUTPUT_DIR / "uncertainty_contribution_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)
    if weight_rows:
        pd.concat(weight_rows, ignore_index=True).to_csv(
            OUTPUT_DIR / "uncertainty_contribution_weight_selection.csv",
            index=False,
        )

    apply_pareto_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    handles = []
    for idx, (_, label, front) in enumerate(fronts):
        handles.append(scatter_pareto(ax, front[:, 0], front[:, 1], label, PARETO_COLORS[idx]))
    all_fronts = np.vstack([front for _, _, front in fronts])
    style_pareto_axis(ax, all_fronts[:, 0], all_fronts[:, 1], handles)
    fig.savefig(OUTPUT_DIR / "uncertainty_contribution_comparison_pareto.png", bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
