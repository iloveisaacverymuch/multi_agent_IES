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

KAPPA_RUNS = [
    (0.00, RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_00_price_0_02"),
    (0.05, RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_05_price_0_02"),
    (0.10, RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_10_price_0_02"),
    (0.20, RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02"),
    (0.30, RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_30_price_0_02"),
    (0.50, RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_50_price_0_02"),
]

BASELINE_RUNS = [
    ("carbon_contribution", "Carbon CDA + Contribution", RUN_ROOT / "carbon_cda_contribution_price_0_02"),
    ("fixed_weight_mappo", "Fixed-weight MAPPO", RUN_ROOT / "baseline_fixed_weight_mappo"),
]


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


def read_elapsed_hours(run_dir):
    path = run_dir / "elapsed_seconds.txt"
    if not path.exists():
        return np.nan
    return float(path.read_text(encoding="utf-8").strip()) / 3600.0


def weight_stats(run_dir):
    rows = []
    for path in sorted(run_dir.glob("[0-9]*/elites/weights.txt"), key=lambda p: int(p.parts[-3])):
        if path.stat().st_size == 0:
            continue
        weights = np.atleast_2d(np.loadtxt(path, delimiter=","))
        if weights.size == 0:
            continue
        for weight in weights:
            rows.append(weight)
    if not rows:
        return {
            "selected_weight_records": np.nan,
            "unique_selected_weights": np.nan,
            "mean_weight_cost": np.nan,
            "std_weight_cost": np.nan,
            "min_weight_cost": np.nan,
            "max_weight_cost": np.nan,
        }
    weights = np.asarray(rows, dtype=float)
    rounded = np.round(weights, decimals=4)
    return {
        "selected_weight_records": len(weights),
        "unique_selected_weights": len(np.unique(rounded, axis=0)),
        "mean_weight_cost": float(weights[:, 0].mean()),
        "std_weight_cost": float(weights[:, 0].std(ddof=0)),
        "min_weight_cost": float(weights[:, 0].min()),
        "max_weight_cost": float(weights[:, 0].max()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded = []
    for kappa, run_dir in KAPPA_RUNS:
        objectives = load_objectives(run_dir)
        if objectives is not None:
            loaded.append((f"kappa_{kappa:.2f}", f"Ensemble kappa={kappa:.2f}", run_dir, objectives, kappa))
    for key, label, run_dir in BASELINE_RUNS:
        objectives = load_objectives(run_dir)
        if objectives is not None:
            loaded.append((key, label, run_dir, objectives, np.nan))

    if not loaded:
        raise RuntimeError("No completed kappa sweep runs found.")

    all_objectives = np.vstack([item[3] for item in loaded])
    mins = all_objectives.min(axis=0)
    ranges = np.max(all_objectives, axis=0) - mins
    ranges = np.where(ranges <= 1e-12, 1.0, ranges)
    ref_point = np.max(all_objectives, axis=0) + 0.1 * ranges
    reference_front = unique_front(all_objectives)
    reference_norm = (reference_front - mins) / ranges

    rows = []
    fronts = []
    for key, label, run_dir, objectives, kappa in loaded:
        front = unique_front(objectives)
        front_norm = (front - mins) / ranges
        fronts.append((key, label, front))
        rows.append(
            {
                "method_key": key,
                "method": label,
                "kappa": kappa,
                "num_points": len(objectives),
                "num_front_points": len(front),
                "min_cost": float(objectives[:, 0].min()),
                "min_carbon": float(objectives[:, 1].min()),
                "normalized_hypervolume": float(Hypervolume(ref_point=np.array([1.1, 1.1])).do(front_norm)),
                "hypervolume": float(Hypervolume(ref_point=ref_point).do(front)),
                "igd": igd(front_norm, reference_norm),
                "spacing": spacing(front_norm),
                "sparsity": sparsity(front),
                "normalized_sparsity": sparsity(front_norm),
                "elapsed_hours": read_elapsed_hours(run_dir),
                **weight_stats(run_dir),
            }
        )

    summary = pd.DataFrame(rows).sort_values("normalized_hypervolume", ascending=False)
    summary_path = OUTPUT_DIR / "ensemble_kappa_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    apply_pareto_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    handles = []
    kappa_fronts = [(key, label, front) for key, label, front in fronts if key.startswith("kappa_")]
    for idx, (_, label, front) in enumerate(kappa_fronts):
        handles.append(scatter_pareto(ax, front[:, 0], front[:, 1], label, PARETO_COLORS[idx % len(PARETO_COLORS)]))
    all_fronts = np.vstack([front for _, _, front in kappa_fronts])
    style_pareto_axis(ax, all_fronts[:, 0], all_fronts[:, 1], handles)
    fig.savefig(OUTPUT_DIR / "ensemble_kappa_sweep_pareto.png", bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
