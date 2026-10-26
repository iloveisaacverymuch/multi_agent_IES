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
V2_RUN = (
    RUN_ROOT
    / "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02"
)
RUNS = (
    ("v2", "Carbon CDA V2 + HV-Ensemble", V2_RUN),
    (
        "current_main",
        "Carbon CDA V1 + Ensemble-Uncertainty",
        RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02",
    ),
    (
        "standard_ensemble",
        "Standard CDA + Ensemble-Uncertainty",
        RUN_ROOT / "standard_cda_ensemble_uncertainty_capped32_kappa_0_20",
    ),
    ("fixed_mappo", "Fixed-weight MAPPO", RUN_ROOT / "baseline_fixed_weight_mappo"),
)


def load_objectives(path):
    objectives = np.atleast_2d(np.loadtxt(path / "final" / "objs.txt", delimiter=","))
    if objectives.shape[1] != 2:
        raise ValueError(f"Expected two objectives in {path}, got {objectives.shape}")
    return objectives


def unique_front(objectives, decimals=2):
    order = np.lexsort((objectives[:, 1], objectives[:, 0]))
    sorted_points = objectives[order]
    keep = []
    best_carbon = np.inf
    for index, point in enumerate(sorted_points):
        if point[1] < best_carbon:
            keep.append(index)
            best_carbon = point[1]
    front = sorted_points[keep]
    rounded = np.round(front, decimals=decimals)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return front[np.sort(indices)]


def igd(front, reference):
    distances = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def spacing(front):
    if len(front) < 2:
        return 0.0
    distances = np.linalg.norm(front[:, None, :] - front[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(np.sqrt(np.mean((nearest - nearest.mean()) ** 2)))


def sparsity(front):
    if len(front) < 2:
        return 0.0
    differences = np.diff(front[np.argsort(front[:, 0])], axis=0)
    return float(np.mean(np.sum(differences**2, axis=1)))


def main():
    loaded = [(key, label, path, load_objectives(path)) for key, label, path in RUNS]
    all_objectives = np.vstack([item[3] for item in loaded])
    minimum = all_objectives.min(axis=0)
    ranges = np.maximum(all_objectives.max(axis=0) - minimum, 1e-12)
    reference = unique_front(all_objectives)
    reference_normalized = (reference - minimum) / ranges
    hv = Hypervolume(ref_point=np.array([1.1, 1.1]))

    rows = []
    fronts = []
    for key, label, path, objectives in loaded:
        front = unique_front(objectives)
        normalized = (front - minimum) / ranges
        fronts.append((key, label, front))
        rows.append(
            {
                "method_key": key,
                "method": label,
                "run_dir": str(path.relative_to(PROJECT_ROOT)),
                "front_points": len(front),
                "normalized_hv": float(hv.do(normalized)),
                "igd": igd(normalized, reference_normalized),
                "spacing": spacing(normalized),
                "normalized_sparsity": sparsity(normalized),
                "min_cost": float(front[:, 0].min()),
                "min_carbon": float(front[:, 1].min()),
                "cost_span": float(front[:, 0].max() - front[:, 0].min()),
                "carbon_span": float(front[:, 1].max() - front[:, 1].min()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("normalized_hv", ascending=False)
    summary_path = OUTPUT_DIR / "v2_pilot_comparison_summary.csv"
    summary.to_csv(summary_path, index=False)

    apply_pareto_rcparams()
    fig, axis = plt.subplots(figsize=(7.4, 5.3))
    handles = []
    for index, (_, label, front) in enumerate(fronts):
        handles.append(
            scatter_pareto(
                axis,
                front[:, 0],
                front[:, 1],
                label,
                PARETO_COLORS[index],
            )
        )
    combined = np.vstack([front for _, _, front in fronts])
    style_pareto_axis(axis, combined[:, 0], combined[:, 1], handles)
    figure_path = OUTPUT_DIR / "v2_pilot_pareto_comparison.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
