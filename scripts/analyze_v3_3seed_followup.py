from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from scipy.stats import ttest_rel, wilcoxon

from analyze_v2_3seed import elapsed_seconds, igd, sparsity, unique_front, validate_final
from pareto_style import PARETO_COLORS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
SEEDS = (11, 22, 44)
METHODS = (
    ("standard", "Standard CDA", "standard_cda_hv_ensemble_extreme_capped32_kappa_0_10"),
    (
        "v2",
        "Carbon CDA V2",
        "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02",
    ),
    (
        "v3_0_01",
        "Carbon-Dividend V3 (0.01)",
        "carbon_dividend_v3_hv_ensemble_extreme_capped32_kappa_0_10_price_0_01",
    ),
    (
        "v3_0_05",
        "Carbon-Dividend V3 (0.05)",
        "carbon_dividend_v3_hv_ensemble_extreme_capped32_kappa_0_10_price_0_05",
    ),
)
COMPARISONS = (
    ("v3_0_01", "standard", "V3-0.01 vs Standard"),
    ("v3_0_05", "standard", "V3-0.05 vs Standard"),
    ("v3_0_01", "v2", "V3-0.01 vs V2"),
    ("v3_0_05", "v2", "V3-0.05 vs V2"),
    ("v3_0_05", "v3_0_01", "V3-0.05 vs V3-0.01"),
)


def run_dir(base_name, seed):
    return RUN_ROOT / (base_name if seed == 44 else f"{base_name}_seed_{seed}")


def load_objectives(path):
    objectives = np.atleast_2d(np.loadtxt(path / "final" / "objs.txt", delimiter=","))
    if objectives.shape[1] != 2 or not np.isfinite(objectives).all():
        raise ValueError(f"Invalid objectives in {path}: shape={objectives.shape}")
    return objectives


def paired_results(detail):
    indexed = detail.set_index(["method_key", "seed"])
    effect_rows = []
    test_rows = []
    for new_key, old_key, comparison in COMPARISONS:
        for seed in SEEDS:
            new = indexed.loc[(new_key, seed)]
            old = indexed.loc[(old_key, seed)]
            effect_rows.append(
                {
                    "comparison": comparison,
                    "seed": seed,
                    "hv_change_percent": 100.0 * (new["normalized_hv"] / old["normalized_hv"] - 1.0),
                    "igd_reduction_percent": 100.0 * (1.0 - new["igd"] / old["igd"]),
                    "sparsity_reduction_percent": 100.0
                    * (1.0 - new["normalized_sparsity"] / old["normalized_sparsity"]),
                    "min_cost_change_percent": 100.0 * (new["min_cost"] / old["min_cost"] - 1.0),
                    "min_carbon_change_percent": 100.0
                    * (new["min_carbon"] / old["min_carbon"] - 1.0),
                    "front_points_change": int(new["front_points"] - old["front_points"]),
                }
            )
        for metric, alternative in (
            ("normalized_hv", "greater"),
            ("igd", "less"),
            ("normalized_sparsity", "less"),
        ):
            new_values = np.array([indexed.loc[(new_key, seed), metric] for seed in SEEDS])
            old_values = np.array([indexed.loc[(old_key, seed), metric] for seed in SEEDS])
            t_result = ttest_rel(new_values, old_values, alternative=alternative)
            try:
                w_result = wilcoxon(new_values, old_values, alternative=alternative)
                wilcoxon_pvalue = float(w_result.pvalue)
            except ValueError:
                wilcoxon_pvalue = np.nan
            test_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "alternative": alternative,
                    "paired_t_pvalue": float(t_result.pvalue),
                    "wilcoxon_pvalue": wilcoxon_pvalue,
                    "num_pairs": len(SEEDS),
                }
            )
    effects = pd.DataFrame(effect_rows)
    effect_summary = effects.groupby("comparison", as_index=False).agg(
        hv_change_mean_percent=("hv_change_percent", "mean"),
        hv_change_std_percent=("hv_change_percent", "std"),
        igd_reduction_mean_percent=("igd_reduction_percent", "mean"),
        igd_reduction_std_percent=("igd_reduction_percent", "std"),
        sparsity_reduction_mean_percent=("sparsity_reduction_percent", "mean"),
        min_cost_change_mean_percent=("min_cost_change_percent", "mean"),
        min_carbon_change_mean_percent=("min_carbon_change_percent", "mean"),
        front_points_change_mean=("front_points_change", "mean"),
    )
    return effects, effect_summary, pd.DataFrame(test_rows)


def plot_metrics(summary):
    order = [key for key, _, _ in METHODS]
    labels = {key: label for key, label, _ in METHODS}
    data = summary.set_index("method_key").reindex(order)
    x = np.arange(len(order))
    plt.rcParams.update({"font.family": "Times New Roman", "savefig.dpi": 600})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    specifications = (
        ("hv_mean", "hv_std", "Normalized HV", "Higher is better"),
        ("igd_mean", "igd_std", "IGD", "Lower is better"),
        ("sparsity_mean", "sparsity_std", "Normalized sparsity", "Lower is better"),
    )
    for axis, (column, error, ylabel, title) in zip(axes, specifications):
        axis.bar(
            x,
            data[column],
            yerr=data[error].fillna(0.0),
            color=PARETO_COLORS[: len(order)],
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
        )
        axis.set_xticks(x)
        axis.set_xticklabels([labels[key] for key in order], rotation=22, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    path = OUTPUT_DIR / "v3_3seed_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seedwise(detail):
    order = [key for key, _, _ in METHODS]
    labels = {key: label for key, label, _ in METHODS}
    x = np.arange(len(order))
    plt.rcParams.update({"font.family": "Times New Roman", "savefig.dpi": 600})
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), constrained_layout=True)
    for seed, color in zip(SEEDS, PARETO_COLORS):
        values = detail[detail["seed"] == seed].set_index("method_key").reindex(order)
        axes[0].plot(x, values["normalized_hv"], marker="o", color=color, label=f"Seed {seed}")
        axes[1].plot(x, values["igd"], marker="o", color=color, label=f"Seed {seed}")
    for axis, ylabel in zip(axes, ("Normalized HV", "IGD")):
        axis.set_xticks(x)
        axis.set_xticklabels([labels[key] for key in order], rotation=22, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    axes[0].legend(frameon=True, edgecolor="black")
    path = OUTPUT_DIR / "v3_3seed_seedwise.png"
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
    for method_key, method, seed, path, objectives in loaded:
        front = unique_front(objectives)
        normalized = (front - minimum) / ranges
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
                "normalized_sparsity": sparsity(normalized),
                "min_cost": float(front[:, 0].min()),
                "min_carbon": float(front[:, 1].min()),
                "elapsed_seconds": elapsed_seconds(path),
            }
        )
    detail = pd.DataFrame(rows)
    order = {key: index for index, (key, _, _) in enumerate(METHODS)}
    summary = (
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
            min_carbon_mean=("min_carbon", "mean"),
            front_points_mean=("front_points", "mean"),
            elapsed_hours_mean=("elapsed_seconds", lambda values: np.mean(values) / 3600.0),
        )
        .sort_values("method_key", key=lambda column: column.map(order))
    )
    effects, effect_summary, significance = paired_results(detail)
    paths = {
        "detail": OUTPUT_DIR / "v3_3seed_detail.csv",
        "summary": OUTPUT_DIR / "v3_3seed_summary.csv",
        "effects": OUTPUT_DIR / "v3_3seed_paired_effects.csv",
        "effect_summary": OUTPUT_DIR / "v3_3seed_paired_summary.csv",
        "significance": OUTPUT_DIR / "v3_3seed_significance.csv",
    }
    detail.to_csv(paths["detail"], index=False)
    summary.to_csv(paths["summary"], index=False)
    effects.to_csv(paths["effects"], index=False)
    effect_summary.to_csv(paths["effect_summary"], index=False)
    significance.to_csv(paths["significance"], index=False)
    paths["metrics"] = plot_metrics(summary)
    paths["seedwise"] = plot_seedwise(detail)
    print(summary.to_string(index=False))
    print("\nPaired effects:")
    print(effect_summary.to_string(index=False))
    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
