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
    ("standard_contribution", "Standard CDA + Contribution"),
    ("carbon_contribution", "Carbon CDA + Contribution"),
    ("standard_ensemble", "Standard CDA + Ensemble-Uncertainty"),
    ("carbon_ensemble", "Carbon CDA + Ensemble-Uncertainty"),
]


def run_dir(method_key, seed):
    existing = {
        "standard_contribution": {
            11: RUN_ROOT / "seed_stability_standard_contribution_seed_11",
            44: RUN_ROOT / "mappo_standard_cda_contribution_full",
        },
        "carbon_contribution": {
            11: RUN_ROOT / "carbon_cda_contribution_price_0_02_seed_11",
            22: RUN_ROOT / "carbon_cda_contribution_price_0_02_seed_22",
            44: RUN_ROOT / "carbon_cda_contribution_price_0_02",
        },
        "standard_ensemble": {
            44: RUN_ROOT / "standard_cda_ensemble_uncertainty_capped32_kappa_0_20",
        },
        "carbon_ensemble": {
            11: RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02_seed_11",
            22: RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02_seed_22",
            44: RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02",
        },
    }
    return existing.get(method_key, {}).get(
        seed,
        RUN_ROOT / f"{method_key}_capped32_kappa_0_20_price_0_02_seed_{seed}",
    )


def load_objectives(path):
    objectives_path = path / "final" / "objs.txt"
    if not objectives_path.exists():
        return None
    return np.atleast_2d(np.loadtxt(objectives_path, delimiter=","))


def unique_front(objectives, decimals=2):
    dominated = np.zeros(len(objectives), dtype=bool)
    for i in range(len(objectives)):
        for j in range(len(objectives)):
            if i != j and np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                dominated[i] = True
                break
    front = objectives[~dominated]
    rounded = np.round(front, decimals=decimals)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    front = front[np.sort(indices)]
    return front[np.argsort(front[:, 0])]


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


def elapsed_seconds(path):
    elapsed_path = path / "elapsed_seconds.txt"
    return float(elapsed_path.read_text(encoding="utf-8").strip()) if elapsed_path.exists() else np.nan


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

    expected = len(SEEDS) * len(METHODS)
    if len(loaded) != expected:
        raise RuntimeError(f"Expected {expected} completed runs, found {len(loaded)}.")

    all_objectives = np.vstack([item[4] for item in loaded])
    minimum = all_objectives.min(axis=0)
    ranges = all_objectives.max(axis=0) - minimum
    ranges = np.where(ranges <= 1e-12, 1.0, ranges)
    reference_front = unique_front(all_objectives)
    reference_front_norm = (reference_front - minimum) / ranges
    hv = Hypervolume(ref_point=np.array([1.1, 1.1]))

    rows = []
    for seed, method_key, method, path, objectives in loaded:
        front = unique_front(objectives)
        front_norm = (front - minimum) / ranges
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
                "normalized_hypervolume": float(hv.do(front_norm)),
                "igd": igd(front_norm, reference_front_norm),
                "spacing": spacing(front_norm),
                "normalized_sparsity": sparsity(front_norm),
                "elapsed_seconds": elapsed_seconds(path),
            }
        )

    detail = pd.DataFrame(rows)
    detail_path = OUTPUT_DIR / "ablation_3seed_detail.csv"
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
            elapsed_hours_mean=("elapsed_seconds", lambda values: np.nanmean(values) / 3600.0),
        )
        .sort_values("hv_mean", ascending=False)
    )
    summary_path = OUTPUT_DIR / "ablation_3seed_summary.csv"
    summary.to_csv(summary_path, index=False)

    order = [key for key, _ in METHODS]
    labels = {key: label for key, label in METHODS}
    plot_data = summary.set_index("method_key").reindex(order).reset_index()
    x = np.arange(len(plot_data))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)
    for ax, column, error, ylabel, color in [
        (axes[0], "hv_mean", "hv_std", "Normalized HV", "#4C78A8"),
        (axes[1], "igd_mean", "igd_std", "IGD", "#E15759"),
    ]:
        ax.bar(
            x,
            plot_data[column],
            yerr=plot_data[error].fillna(0.0),
            color=color,
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([labels[key] for key in plot_data["method_key"]], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.65)
    figure_path = OUTPUT_DIR / "ablation_3seed_metrics.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
