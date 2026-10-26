from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from scipy.stats import ttest_rel, wilcoxon

from analyze_v2_3seed import (
    elapsed_seconds,
    igd,
    spacing,
    sparsity,
    unique_front,
    validate_final,
)
from pareto_style import PARETO_COLORS, apply_pareto_rcparams, scatter_pareto, style_pareto_axis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
SEEDS = (11, 22, 44)
METHODS = (
    (
        "current_main",
        "Current main",
        "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02",
    ),
    (
        "endpoint",
        "+ Endpoint protection",
        "carbon_cda_v1_ensemble_extreme_capped32_kappa_0_20_price_0_02",
    ),
    (
        "hv_ensemble",
        "+ HV-Ensemble",
        "carbon_cda_v1_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02",
    ),
    (
        "full_v2",
        "+ Carbon CDA V2",
        "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02",
    ),
)
STEPS = (
    ("endpoint", "current_main", "Endpoint protection"),
    ("hv_ensemble", "endpoint", "HV-Ensemble"),
    ("full_v2", "hv_ensemble", "Carbon CDA V2"),
)


def run_dir(base_name, seed):
    return RUN_ROOT / (base_name if seed == 44 else f"{base_name}_seed_{seed}")


def load_objectives(path):
    objectives = np.atleast_2d(np.loadtxt(path / "final" / "objs.txt", delimiter=","))
    if objectives.shape[1] != 2:
        raise ValueError(f"Expected two objectives in {path}, got {objectives.shape}")
    return objectives


def summarize(detail):
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
            elapsed_hours_mean=("elapsed_seconds", lambda values: np.mean(values) / 3600.0),
        )
        .assign(
            order=lambda frame: frame["method_key"].map(
                {key: index for index, (key, _, _) in enumerate(METHODS)}
            )
        )
        .sort_values("order")
        .drop(columns="order")
    )


def incremental_results(detail):
    indexed = detail.set_index(["method_key", "seed"])
    rows = []
    for new_key, old_key, component in STEPS:
        for seed in SEEDS:
            new = indexed.loc[(new_key, seed)]
            old = indexed.loc[(old_key, seed)]
            rows.append(
                {
                    "component": component,
                    "from_method": old_key,
                    "to_method": new_key,
                    "seed": seed,
                    "hv_change_percent": 100.0 * (new["normalized_hv"] / old["normalized_hv"] - 1.0),
                    "igd_reduction_percent": 100.0 * (1.0 - new["igd"] / old["igd"]),
                    "sparsity_reduction_percent": 100.0
                    * (1.0 - new["normalized_sparsity"] / old["normalized_sparsity"]),
                    "min_cost_change_percent": 100.0 * (new["min_cost"] / old["min_cost"] - 1.0),
                    "min_carbon_change_percent": 100.0 * (new["min_carbon"] / old["min_carbon"] - 1.0),
                    "front_points_change": int(new["front_points"] - old["front_points"]),
                }
            )
    increments = pd.DataFrame(rows)
    summary = (
        increments.groupby(["component", "from_method", "to_method"], as_index=False)
        .agg(
            hv_change_mean_percent=("hv_change_percent", "mean"),
            hv_change_std_percent=("hv_change_percent", "std"),
            igd_reduction_mean_percent=("igd_reduction_percent", "mean"),
            igd_reduction_std_percent=("igd_reduction_percent", "std"),
            sparsity_reduction_mean_percent=("sparsity_reduction_percent", "mean"),
            min_cost_change_mean_percent=("min_cost_change_percent", "mean"),
            min_carbon_change_mean_percent=("min_carbon_change_percent", "mean"),
            front_points_change_mean=("front_points_change", "mean"),
        )
        .assign(order=lambda frame: frame["component"].map({step[2]: i for i, step in enumerate(STEPS)}))
        .sort_values("order")
        .drop(columns="order")
    )
    return increments, summary


def overall_results(detail):
    indexed = detail.set_index(["method_key", "seed"])
    rows = []
    for seed in SEEDS:
        new = indexed.loc[("full_v2", seed)]
        old = indexed.loc[("current_main", seed)]
        rows.append(
            {
                "seed": seed,
                "hv_change_percent": 100.0 * (new["normalized_hv"] / old["normalized_hv"] - 1.0),
                "igd_reduction_percent": 100.0 * (1.0 - new["igd"] / old["igd"]),
                "sparsity_reduction_percent": 100.0
                * (1.0 - new["normalized_sparsity"] / old["normalized_sparsity"]),
                "min_cost_change_percent": 100.0 * (new["min_cost"] / old["min_cost"] - 1.0),
                "min_carbon_change_percent": 100.0 * (new["min_carbon"] / old["min_carbon"] - 1.0),
                "front_points_change": int(new["front_points"] - old["front_points"]),
            }
        )
    overall = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "comparison": "Full V2 vs current main",
                "hv_change_mean_percent": overall["hv_change_percent"].mean(),
                "hv_change_std_percent": overall["hv_change_percent"].std(),
                "igd_reduction_mean_percent": overall["igd_reduction_percent"].mean(),
                "igd_reduction_std_percent": overall["igd_reduction_percent"].std(),
                "sparsity_reduction_mean_percent": overall["sparsity_reduction_percent"].mean(),
                "min_cost_change_mean_percent": overall["min_cost_change_percent"].mean(),
                "min_carbon_change_mean_percent": overall["min_carbon_change_percent"].mean(),
                "front_points_change_mean": overall["front_points_change"].mean(),
            }
        ]
    )
    return overall, summary


def significance(detail):
    indexed = detail.set_index(["method_key", "seed"])
    rows = []
    comparisons = STEPS + (("full_v2", "current_main", "Full V2 vs current main"),)
    for new_key, old_key, component in comparisons:
        for metric, alternative in (
            ("normalized_hv", "greater"),
            ("igd", "less"),
            ("normalized_sparsity", "less"),
        ):
            new = np.array([indexed.loc[(new_key, seed), metric] for seed in SEEDS])
            old = np.array([indexed.loc[(old_key, seed), metric] for seed in SEEDS])
            t_result = ttest_rel(new, old, alternative=alternative)
            w_result = wilcoxon(new, old, alternative=alternative)
            rows.append(
                {
                    "component": component,
                    "metric": metric,
                    "alternative": alternative,
                    "paired_t_statistic": float(t_result.statistic),
                    "paired_t_pvalue": float(t_result.pvalue),
                    "wilcoxon_statistic": float(w_result.statistic),
                    "wilcoxon_pvalue": float(w_result.pvalue),
                    "num_pairs": len(SEEDS),
                }
            )
    return pd.DataFrame(rows)


def plot_metrics(summary):
    x = np.arange(len(summary))
    colors = PARETO_COLORS[: len(summary)]
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
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), constrained_layout=True)
    for axis, (column, error, ylabel, title) in zip(
        axes,
        (
            ("hv_mean", "hv_std", "Normalized HV", "Higher is better"),
            ("igd_mean", "igd_std", "IGD", "Lower is better"),
            ("sparsity_mean", "sparsity_std", "Normalized sparsity", "Lower is better"),
        ),
    ):
        axis.bar(
            x,
            summary[column],
            yerr=summary[error].fillna(0.0),
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(summary["method"], rotation=22, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    path = OUTPUT_DIR / "v2_ablation_3seed_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seedwise(detail):
    labels = [label for _, label, _ in METHODS]
    order = [key for key, _, _ in METHODS]
    x = np.arange(len(order))
    plt.rcParams.update({"font.family": "Times New Roman", "savefig.dpi": 600})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    for seed, color in zip(SEEDS, PARETO_COLORS):
        seed_data = detail[detail["seed"] == seed].set_index("method_key").reindex(order)
        axes[0].plot(x, seed_data["normalized_hv"], marker="o", linewidth=1.5, color=color, label=f"Seed {seed}")
        axes[1].plot(x, seed_data["igd"], marker="o", linewidth=1.5, color=color, label=f"Seed {seed}")
    for axis, ylabel in zip(axes, ("Normalized HV", "IGD")):
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    axes[0].legend(frameon=True, edgecolor="black")
    path = OUTPUT_DIR / "v2_ablation_3seed_seedwise.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seed44_pareto(fronts):
    apply_pareto_rcparams()
    fig, axis = plt.subplots(figsize=(7.4, 5.3))
    handles = []
    for index, (method_key, label, _) in enumerate(METHODS):
        front = fronts[(method_key, 44)]
        handles.append(scatter_pareto(axis, front[:, 0], front[:, 1], label, PARETO_COLORS[index]))
    combined = np.vstack([fronts[(key, 44)] for key, _, _ in METHODS])
    style_pareto_axis(axis, combined[:, 0], combined[:, 1], handles)
    path = OUTPUT_DIR / "v2_ablation_seed44_pareto.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = []
    for method_key, method, base_name in METHODS:
        for seed in SEEDS:
            path = run_dir(base_name, seed)
            objectives = load_objectives(path)
            validate_final(path, len(objectives))
            loaded.append((method_key, method, seed, path, objectives))

    all_objectives = np.vstack([item[4] for item in loaded])
    minimum = all_objectives.min(axis=0)
    ranges = np.maximum(all_objectives.max(axis=0) - minimum, 1e-12)
    reference = (unique_front(all_objectives) - minimum) / ranges
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
                "igd": igd(normalized, reference),
                "spacing": spacing(normalized),
                "normalized_sparsity": sparsity(normalized),
                "min_cost": float(front[:, 0].min()),
                "min_carbon": float(front[:, 1].min()),
                "elapsed_seconds": elapsed_seconds(path),
            }
        )

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    increments, increment_summary = incremental_results(detail)
    overall, overall_summary = overall_results(detail)
    tests = significance(detail)
    paths = {
        "detail": OUTPUT_DIR / "v2_ablation_3seed_detail.csv",
        "summary": OUTPUT_DIR / "v2_ablation_3seed_summary.csv",
        "increments": OUTPUT_DIR / "v2_ablation_3seed_incremental.csv",
        "increment_summary": OUTPUT_DIR / "v2_ablation_3seed_incremental_summary.csv",
        "overall": OUTPUT_DIR / "v2_ablation_3seed_overall.csv",
        "overall_summary": OUTPUT_DIR / "v2_ablation_3seed_overall_summary.csv",
        "significance": OUTPUT_DIR / "v2_ablation_3seed_significance.csv",
    }
    detail.to_csv(paths["detail"], index=False)
    summary.to_csv(paths["summary"], index=False)
    increments.to_csv(paths["increments"], index=False)
    increment_summary.to_csv(paths["increment_summary"], index=False)
    overall.to_csv(paths["overall"], index=False)
    overall_summary.to_csv(paths["overall_summary"], index=False)
    tests.to_csv(paths["significance"], index=False)
    paths["metrics"] = plot_metrics(summary)
    paths["seedwise"] = plot_seedwise(detail)
    paths["pareto"] = plot_seed44_pareto(fronts)

    print(summary.to_string(index=False))
    print("\nIncremental component effects:")
    print(increment_summary.to_string(index=False))
    print("\nOverall V2 effect:")
    print(overall_summary.to_string(index=False))
    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
