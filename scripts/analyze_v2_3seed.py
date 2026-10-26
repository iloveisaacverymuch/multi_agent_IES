from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from scipy.stats import ttest_rel, wilcoxon

from pareto_style import PARETO_COLORS, apply_pareto_rcparams, scatter_pareto, style_pareto_axis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
SEEDS = (11, 22, 44)
METHODS = (
    ("v2", "Carbon CDA V2 + HV-Ensemble"),
    ("current_main", "Carbon CDA V1 + Ensemble-Uncertainty"),
    ("fixed_mappo", "Fixed-weight MAPPO"),
)


def run_dir(method_key, seed):
    if method_key == "v2":
        base = "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02"
    elif method_key == "current_main":
        base = "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02"
    elif method_key == "fixed_mappo":
        base = "baseline_fixed_weight_mappo"
    else:
        raise KeyError(method_key)
    return RUN_ROOT / (base if seed == 44 else f"{base}_seed_{seed}")


def load_objectives(path):
    objectives_path = path / "final" / "objs.txt"
    if not objectives_path.exists():
        raise FileNotFoundError(objectives_path)
    objectives = np.atleast_2d(np.loadtxt(objectives_path, delimiter=","))
    if objectives.shape[1] != 2:
        raise ValueError(f"Expected two objectives in {objectives_path}, got {objectives.shape}")
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


def elapsed_seconds(path):
    elapsed_path = path / "elapsed_seconds.txt"
    if not elapsed_path.exists():
        return np.nan
    return float(elapsed_path.read_text(encoding="utf-8").strip())


def validate_final(path, objective_count):
    final_dir = path / "final"
    policy_count = len(list(final_dir.glob("EP_policy_*.pt")))
    metadata_count = len(pd.read_csv(final_dir / "metadata.csv"))
    if policy_count != objective_count or metadata_count != objective_count:
        raise RuntimeError(
            f"Incomplete final output in {path}: objectives={objective_count}, "
            f"policies={policy_count}, metadata={metadata_count}"
        )
    return policy_count, metadata_count


def make_summary(detail):
    return (
        detail.groupby(["method_key", "method"], as_index=False)
        .agg(
            completed_seeds=("seed", "count"),
            hv_mean=("normalized_hv", "mean"),
            hv_std=("normalized_hv", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            sparsity_mean=("normalized_sparsity", "mean"),
            sparsity_std=("normalized_sparsity", "std"),
            min_cost_mean=("min_cost", "mean"),
            min_cost_std=("min_cost", "std"),
            min_carbon_mean=("min_carbon", "mean"),
            min_carbon_std=("min_carbon", "std"),
            front_points_mean=("front_points", "mean"),
            front_points_std=("front_points", "std"),
            elapsed_hours_mean=("elapsed_seconds", lambda values: np.nanmean(values) / 3600.0),
        )
        .sort_values("hv_mean", ascending=False)
    )


def make_paired_deltas(detail):
    rows = []
    proposed = detail[detail["method_key"] == "v2"].set_index("seed")
    for baseline_key in ("current_main", "fixed_mappo"):
        baseline = detail[detail["method_key"] == baseline_key].set_index("seed")
        for seed in SEEDS:
            new = proposed.loc[seed]
            old = baseline.loc[seed]
            rows.append(
                {
                    "baseline_key": baseline_key,
                    "baseline": old["method"],
                    "seed": seed,
                    "hv_change_percent": 100.0 * (new["normalized_hv"] / old["normalized_hv"] - 1.0),
                    "igd_reduction_percent": 100.0 * (1.0 - new["igd"] / old["igd"]),
                    "sparsity_reduction_percent": 100.0
                    * (1.0 - new["normalized_sparsity"] / old["normalized_sparsity"]),
                    "min_cost_change_percent": 100.0 * (new["min_cost"] / old["min_cost"] - 1.0),
                    "min_carbon_change_percent": 100.0 * (new["min_carbon"] / old["min_carbon"] - 1.0),
                }
            )
    return pd.DataFrame(rows)


def make_significance(detail):
    rows = []
    proposed = detail[detail["method_key"] == "v2"].sort_values("seed")
    for baseline_key in ("current_main", "fixed_mappo"):
        baseline = detail[detail["method_key"] == baseline_key].sort_values("seed")
        for metric, alternative in (
            ("normalized_hv", "greater"),
            ("igd", "less"),
            ("normalized_sparsity", "less"),
        ):
            proposed_values = proposed[metric].to_numpy()
            baseline_values = baseline[metric].to_numpy()
            t_result = ttest_rel(proposed_values, baseline_values, alternative=alternative)
            w_result = wilcoxon(proposed_values, baseline_values, alternative=alternative)
            rows.append(
                {
                    "baseline_key": baseline_key,
                    "metric": metric,
                    "alternative": alternative,
                    "paired_t_statistic": float(t_result.statistic),
                    "paired_t_pvalue": float(t_result.pvalue),
                    "wilcoxon_statistic": float(w_result.statistic),
                    "wilcoxon_pvalue": float(w_result.pvalue),
                    "num_pairs": len(proposed_values),
                }
            )
    return pd.DataFrame(rows)


def plot_metrics(summary):
    labels = {key: label for key, label in METHODS}
    order = [key for key, _ in METHODS]
    data = summary.set_index("method_key").reindex(order).reset_index()
    x = np.arange(len(data))
    colors = [PARETO_COLORS[index] for index in range(len(data))]
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 600,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), constrained_layout=True)
    specs = (
        ("hv_mean", "hv_std", "Normalized HV", "Higher is better"),
        ("igd_mean", "igd_std", "IGD", "Lower is better"),
        ("sparsity_mean", "sparsity_std", "Normalized sparsity", "Lower is better"),
    )
    for axis, (column, error, ylabel, title) in zip(axes, specs):
        axis.bar(
            x,
            data[column],
            yerr=data[error].fillna(0.0),
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
        )
        axis.set_xticks(x)
        axis.set_xticklabels([labels[key] for key in order], rotation=22, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    path = OUTPUT_DIR / "v2_3seed_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_v2_fronts(fronts):
    apply_pareto_rcparams()
    fig, axis = plt.subplots(figsize=(7.4, 5.3))
    handles = []
    for index, seed in enumerate(SEEDS):
        front = fronts[("v2", seed)]
        handles.append(
            scatter_pareto(
                axis,
                front[:, 0],
                front[:, 1],
                f"Seed {seed}",
                PARETO_COLORS[index],
            )
        )
    combined = np.vstack([fronts[("v2", seed)] for seed in SEEDS])
    style_pareto_axis(axis, combined[:, 0], combined[:, 1], handles)
    path = OUTPUT_DIR / "v2_3seed_pareto.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = []
    for method_key, method in METHODS:
        for seed in SEEDS:
            path = run_dir(method_key, seed)
            objectives = load_objectives(path)
            validate_final(path, len(objectives))
            loaded.append((method_key, method, seed, path, objectives))

    all_objectives = np.vstack([item[4] for item in loaded])
    minimum = all_objectives.min(axis=0)
    ranges = np.maximum(all_objectives.max(axis=0) - minimum, 1e-12)
    reference_front = unique_front(all_objectives)
    reference_normalized = (reference_front - minimum) / ranges
    hv = Hypervolume(ref_point=np.array([1.1, 1.1]))

    rows = []
    fronts = {}
    for method_key, method, seed, path, objectives in loaded:
        front = unique_front(objectives)
        normalized = (front - minimum) / ranges
        fronts[(method_key, seed)] = front
        rows.append(
            {
                "method_key": method_key,
                "method": method,
                "seed": seed,
                "run_dir": str(path.relative_to(PROJECT_ROOT)),
                "objective_points": len(objectives),
                "front_points": len(front),
                "normalized_hv": float(hv.do(normalized)),
                "igd": igd(normalized, reference_normalized),
                "spacing": spacing(normalized),
                "normalized_sparsity": sparsity(normalized),
                "min_cost": float(front[:, 0].min()),
                "min_carbon": float(front[:, 1].min()),
                "elapsed_seconds": elapsed_seconds(path),
            }
        )

    detail = pd.DataFrame(rows)
    summary = make_summary(detail)
    paired = make_paired_deltas(detail)
    paired_summary = (
        paired.groupby(["baseline_key", "baseline"], as_index=False)
        .agg(
            hv_change_mean_percent=("hv_change_percent", "mean"),
            hv_change_std_percent=("hv_change_percent", "std"),
            igd_reduction_mean_percent=("igd_reduction_percent", "mean"),
            igd_reduction_std_percent=("igd_reduction_percent", "std"),
            sparsity_reduction_mean_percent=("sparsity_reduction_percent", "mean"),
            min_cost_change_mean_percent=("min_cost_change_percent", "mean"),
            min_carbon_change_mean_percent=("min_carbon_change_percent", "mean"),
        )
        .sort_values("hv_change_mean_percent", ascending=False)
    )
    significance = make_significance(detail)

    paths = {
        "detail": OUTPUT_DIR / "v2_3seed_detail.csv",
        "summary": OUTPUT_DIR / "v2_3seed_summary.csv",
        "paired": OUTPUT_DIR / "v2_3seed_paired_deltas.csv",
        "paired_summary": OUTPUT_DIR / "v2_3seed_paired_summary.csv",
        "significance": OUTPUT_DIR / "v2_3seed_significance.csv",
    }
    detail.to_csv(paths["detail"], index=False)
    summary.to_csv(paths["summary"], index=False)
    paired.to_csv(paths["paired"], index=False)
    paired_summary.to_csv(paths["paired_summary"], index=False)
    significance.to_csv(paths["significance"], index=False)
    paths["metrics"] = plot_metrics(summary)
    paths["pareto"] = plot_v2_fronts(fronts)

    print(summary.to_string(index=False))
    print("\nPaired V2 changes:")
    print(paired_summary.to_string(index=False))
    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
