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
PROPOSED_RUN_NAME = "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02"

SEEDS = [11, 22, 44]
METHODS = [
    ("proposed", "Carbon CDA + Ensemble-Uncertainty (kappa=0.20)"),
    ("fixed_weight_mappo", "Fixed-weight MAPPO"),
    ("masac", "MASAC"),
    ("maddpg", "MADDPG"),
    ("mopso", "MOPSO"),
]


def run_dir(method_key, seed):
    if seed == 44:
        legacy = {
            "proposed": PROPOSED_RUN_NAME,
            "fixed_weight_mappo": "baseline_fixed_weight_mappo",
            "masac": "baseline_masac",
            "maddpg": "baseline_maddpg",
            "mopso": "baseline_mopso",
        }
        return RUN_ROOT / legacy[method_key]
    if method_key == "proposed":
        return RUN_ROOT / f"{PROPOSED_RUN_NAME}_seed_{seed}"
    return RUN_ROOT / f"baseline_{method_key}_seed_{seed}"


def load_objectives(path):
    obj_path = path / "final" / "objs.txt"
    if not obj_path.exists():
        return None
    return np.atleast_2d(np.loadtxt(obj_path, delimiter=","))


def read_elapsed(path):
    elapsed_path = path / "elapsed_seconds.txt"
    if elapsed_path.exists():
        return float(elapsed_path.read_text(encoding="utf-8").strip())
    return np.nan


def pareto_indices(objectives):
    dominated = np.zeros(len(objectives), dtype=bool)
    for i in range(len(objectives)):
        for j in range(len(objectives)):
            if i != j and np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


def unique_front(objectives, decimals=2):
    front = objectives[pareto_indices(objectives)]
    rounded = np.round(front, decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    front = front[np.sort(idx)]
    return front[np.argsort(front[:, 0])]


def igd(front_norm, ref_norm):
    dist = np.linalg.norm(ref_norm[:, None, :] - front_norm[None, :, :], axis=2)
    return float(np.mean(np.min(dist, axis=1)))


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = []
    for seed in SEEDS:
        for method_key, method in METHODS:
            path = run_dir(method_key, seed)
            objectives = load_objectives(path)
            if objectives is None:
                print(f"Missing: seed={seed}, method={method}, path={path}")
                continue
            loaded.append((seed, method_key, method, path, objectives))

    if len(loaded) < 2:
        raise RuntimeError("Not enough completed runs.")

    all_objectives = np.vstack([item[4] for item in loaded])
    mins = all_objectives.min(axis=0)
    maxs = all_objectives.max(axis=0)
    ranges = np.where(maxs - mins <= 1e-12, 1.0, maxs - mins)
    ref_point = maxs + 0.1 * ranges
    ref_norm_point = np.array([1.1, 1.1])
    reference_front = unique_front(all_objectives)
    reference_norm = (reference_front - mins) / ranges

    rows = []
    for seed, method_key, method, path, objectives in loaded:
        front = unique_front(objectives)
        front_norm = (front - mins) / ranges
        rows.append(
            {
                "seed": seed,
                "method_key": method_key,
                "method": method,
                "run_dir": str(path.relative_to(PROJECT_ROOT)),
                "num_points": len(objectives),
                "num_front_points": len(front),
                "min_cost": float(objectives[:, 0].min()),
                "min_carbon": float(objectives[:, 1].min()),
                "normalized_hypervolume": float(Hypervolume(ref_point=ref_norm_point).do(front_norm)),
                "hypervolume": float(Hypervolume(ref_point=ref_point).do(front)),
                "igd": igd(front_norm, reference_norm),
                "spacing": spacing(front_norm),
                "sparsity": sparsity(front),
                "normalized_sparsity": sparsity(front_norm),
                "elapsed_seconds": read_elapsed(path),
            }
        )

    detail = pd.DataFrame(rows)
    detail_path = OUTPUT_DIR / "algorithm_comparison_3seed_detail.csv"
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby(["method_key", "method"], as_index=False)
        .agg(
            completed_seeds=("seed", "count"),
            hv_mean=("normalized_hypervolume", "mean"),
            hv_std=("normalized_hypervolume", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            sparsity_mean=("normalized_sparsity", "mean"),
            sparsity_std=("normalized_sparsity", "std"),
            min_cost_mean=("min_cost", "mean"),
            min_cost_std=("min_cost", "std"),
            min_carbon_mean=("min_carbon", "mean"),
            min_carbon_std=("min_carbon", "std"),
            front_points_mean=("num_front_points", "mean"),
            elapsed_hours_mean=("elapsed_seconds", lambda x: np.nanmean(x) / 3600.0),
        )
        .sort_values("hv_mean", ascending=False)
    )
    summary_path = OUTPUT_DIR / "algorithm_comparison_3seed_summary.csv"
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
    order = [key for key, _ in METHODS]
    labels = {key: label for key, label in METHODS}
    plot_data = summary.set_index("method_key").reindex(order).reset_index()
    x = np.arange(len(plot_data))
    yerr = plot_data["hv_std"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    ax.bar(x, plot_data["hv_mean"], yerr=yerr, color="#4C78A8", edgecolor="black", linewidth=0.6, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[key] for key in plot_data["method_key"]], rotation=25, ha="right")
    ax.set_ylabel("Normalized HV")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.65)
    fig.savefig(OUTPUT_DIR / "algorithm_comparison_3seed_hv.png", bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
