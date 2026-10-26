from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from scipy.stats import ttest_1samp, ttest_rel, wilcoxon

from analyze_v2_3seed import elapsed_seconds, igd, sparsity, unique_front, validate_final
from analyze_v3_dividend_ablation_3seed import (
    add_grouped_pvalue_corrections,
    bootstrap_mean_ci,
    paired_rank_biserial,
)
from pareto_style import PARETO_COLORS, apply_pareto_rcparams, scatter_pareto, style_pareto_axis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
SEEDS = (11, 22, 33, 44, 55)
METHODS = (
    (
        "standard_contribution",
        "Standard CDA + Contribution",
        "standard",
        "contribution",
        "standard_cda_contribution_current_matched",
    ),
    (
        "standard_ensemble",
        "Standard CDA + HV-Ensemble",
        "standard",
        "ensemble",
        "standard_cda_hv_ensemble_extreme_capped32_kappa_0_10",
    ),
    (
        "v3_contribution",
        "V3 targeted dividend + Contribution",
        "v3",
        "contribution",
        "carbon_dividend_v3_contribution_current_matched_price_0_01",
    ),
    (
        "v3_ensemble",
        "V3 targeted dividend + HV-Ensemble",
        "v3",
        "ensemble",
        "carbon_dividend_v3_hv_ensemble_extreme_capped32_kappa_0_10_price_0_01",
    ),
)
COMPARISONS = (
    ("standard_ensemble", "standard_contribution", "Ensemble effect under Standard CDA"),
    ("v3_ensemble", "v3_contribution", "Ensemble effect under V3"),
    ("v3_contribution", "standard_contribution", "V3 effect under Contribution"),
    ("v3_ensemble", "standard_ensemble", "V3 effect under Ensemble"),
    ("v3_ensemble", "standard_contribution", "Full method vs base configuration"),
)
METRICS = (
    ("normalized_hv", "greater"),
    ("igd", "less"),
    ("normalized_sparsity", "less"),
)


def run_dir(method_key, base_name, seed):
    if method_key in ("standard_ensemble", "v3_ensemble") and seed == 44:
        return RUN_ROOT / base_name
    return RUN_ROOT / f"{base_name}_seed_{seed}"


def load_objectives(path):
    objectives = np.atleast_2d(np.loadtxt(path / "final" / "objs.txt", delimiter=","))
    if objectives.shape[1] != 2 or not np.isfinite(objectives).all():
        raise ValueError(f"Invalid objectives in {path}: shape={objectives.shape}")
    return objectives


def improvement_percent(new_values, old_values, alternative):
    if alternative == "greater":
        return 100.0 * (new_values / old_values - 1.0), new_values - old_values
    return 100.0 * (1.0 - new_values / old_values), old_values - new_values


def paired_analysis(detail):
    indexed = detail.set_index(["method_key", "seed"])
    effect_rows = []
    test_rows = []
    for comparison_index, (new_key, old_key, comparison) in enumerate(COMPARISONS):
        for seed in SEEDS:
            new = indexed.loc[(new_key, seed)]
            old = indexed.loc[(old_key, seed)]
            effect_rows.append(
                {
                    "comparison": comparison,
                    "seed": seed,
                    "hv_improvement_percent": 100.0 * (new["normalized_hv"] / old["normalized_hv"] - 1.0),
                    "igd_improvement_percent": 100.0 * (1.0 - new["igd"] / old["igd"]),
                    "sparsity_improvement_percent": 100.0
                    * (1.0 - new["normalized_sparsity"] / old["normalized_sparsity"]),
                    "front_points_change": int(new["front_points"] - old["front_points"]),
                }
            )
        for metric_index, (metric, alternative) in enumerate(METRICS):
            new_values = np.array([indexed.loc[(new_key, seed), metric] for seed in SEEDS])
            old_values = np.array([indexed.loc[(old_key, seed), metric] for seed in SEEDS])
            improvements, oriented_differences = improvement_percent(new_values, old_values, alternative)
            ci_low, ci_high = bootstrap_mean_ci(
                improvements,
                seed=3000 + 10 * comparison_index + metric_index,
            )
            t_result = ttest_rel(new_values, old_values, alternative=alternative)
            try:
                w_pvalue = float(wilcoxon(new_values, old_values, alternative=alternative).pvalue)
            except ValueError:
                w_pvalue = np.nan
            test_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "alternative": alternative,
                    "paired_t_pvalue": float(t_result.pvalue),
                    "wilcoxon_pvalue": w_pvalue,
                    "mean_improvement_percent": float(improvements.mean()),
                    "bootstrap_ci95_low": float(ci_low),
                    "bootstrap_ci95_high": float(ci_high),
                    "rank_biserial": paired_rank_biserial(oriented_differences),
                    "improved_seeds": int((improvements > 0.0).sum()),
                    "num_pairs": len(SEEDS),
                }
            )
    effects = pd.DataFrame(effect_rows)
    effect_summary = effects.groupby("comparison", as_index=False).agg(
        hv_improvement_mean_percent=("hv_improvement_percent", "mean"),
        igd_improvement_mean_percent=("igd_improvement_percent", "mean"),
        sparsity_improvement_mean_percent=("sparsity_improvement_percent", "mean"),
        front_points_change_mean=("front_points_change", "mean"),
    )
    tests = add_grouped_pvalue_corrections(
        pd.DataFrame(test_rows),
        ("paired_t_pvalue", "wilcoxon_pvalue"),
    )
    return effects, effect_summary, tests


def interaction_analysis(detail):
    indexed = detail.set_index(["method_key", "seed"])
    rows = []
    summary_rows = []
    for metric_index, (metric, alternative) in enumerate(METRICS):
        interactions = []
        for seed in SEEDS:
            standard_contribution = float(indexed.loc[("standard_contribution", seed), metric])
            standard_ensemble = float(indexed.loc[("standard_ensemble", seed), metric])
            v3_contribution = float(indexed.loc[("v3_contribution", seed), metric])
            v3_ensemble = float(indexed.loc[("v3_ensemble", seed), metric])
            standard_gain = improvement_percent(
                np.array([standard_ensemble]),
                np.array([standard_contribution]),
                alternative,
            )[0][0]
            v3_gain = improvement_percent(
                np.array([v3_ensemble]),
                np.array([v3_contribution]),
                alternative,
            )[0][0]
            interaction = v3_gain - standard_gain
            interactions.append(interaction)
            rows.append(
                {
                    "metric": metric,
                    "seed": seed,
                    "standard_selection_gain_percent": standard_gain,
                    "v3_selection_gain_percent": v3_gain,
                    "interaction_percentage_points": interaction,
                }
            )
        interactions = np.asarray(interactions)
        ci_low, ci_high = bootstrap_mean_ci(interactions, seed=4000 + metric_index)
        t_result = ttest_1samp(interactions, 0.0, alternative="greater")
        try:
            w_pvalue = float(wilcoxon(interactions, alternative="greater").pvalue)
        except ValueError:
            w_pvalue = np.nan
        summary_rows.append(
            {
                "comparison": "Selection x market interaction",
                "metric": metric,
                "alternative": "greater",
                "mean_interaction_percentage_points": float(interactions.mean()),
                "bootstrap_ci95_low": float(ci_low),
                "bootstrap_ci95_high": float(ci_high),
                "paired_t_pvalue": float(t_result.pvalue),
                "wilcoxon_pvalue": w_pvalue,
                "rank_biserial": paired_rank_biserial(interactions),
                "positive_seeds": int((interactions > 0.0).sum()),
                "num_pairs": len(SEEDS),
            }
        )
    summary = add_grouped_pvalue_corrections(
        pd.DataFrame(summary_rows),
        ("paired_t_pvalue", "wilcoxon_pvalue"),
    )
    return pd.DataFrame(rows), summary


def plot_metrics(summary):
    indexed = summary.set_index("method_key")
    markets = ("standard", "v3")
    market_labels = ("Standard CDA", "V3 targeted dividend")
    selections = ("contribution", "ensemble")
    selection_labels = ("Contribution", "HV-Ensemble")
    colors = (PARETO_COLORS[0], PARETO_COLORS[2])
    x = np.arange(len(markets))
    width = 0.34
    specs = (
        ("hv_mean", "hv_std", "Normalized HV", "Higher is better"),
        ("igd_mean", "igd_std", "IGD", "Lower is better"),
        ("sparsity_mean", "sparsity_std", "Normalized sparsity", "Lower is better"),
    )
    plt.rcParams.update({"font.family": "Times New Roman", "savefig.dpi": 600})
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), constrained_layout=True)
    method_lookup = {(market, selection): key for key, _, market, selection, _ in METHODS}
    for axis, (column, error, ylabel, title) in zip(axes, specs):
        for index, (selection, label, color) in enumerate(zip(selections, selection_labels, colors)):
            keys = [method_lookup[(market, selection)] for market in markets]
            positions = x + (index - 0.5) * width
            axis.bar(
                positions,
                indexed.loc[keys, column],
                width,
                yerr=indexed.loc[keys, error].fillna(0.0),
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                capsize=4,
            )
        axis.set_xticks(x)
        axis.set_xticklabels(market_labels, rotation=12, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    axes[0].legend(frameon=True, edgecolor="black")
    path = OUTPUT_DIR / "v3_dual_module_ablation_5seed_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_seedwise(detail):
    order = [key for key, _, _, _, _ in METHODS]
    labels = {key: label for key, label, _, _, _ in METHODS}
    x = np.arange(len(order))
    plt.rcParams.update({"font.family": "Times New Roman", "savefig.dpi": 600})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for seed, color in zip(SEEDS, PARETO_COLORS):
        values = detail[detail["seed"] == seed].set_index("method_key").reindex(order)
        axes[0].plot(x, values["normalized_hv"], marker="o", color=color, label=f"Seed {seed}")
        axes[1].plot(x, values["igd"], marker="o", color=color, label=f"Seed {seed}")
    for axis, ylabel in zip(axes, ("Normalized HV", "IGD")):
        axis.set_xticks(x)
        axis.set_xticklabels([labels[key] for key in order], rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    axes[0].legend(frameon=True, edgecolor="black")
    path = OUTPUT_DIR / "v3_dual_module_ablation_5seed_seedwise.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pareto(fronts, seed=44):
    apply_pareto_rcparams()
    fig, axis = plt.subplots(figsize=(7.6, 5.4))
    handles = []
    for index, (key, label, _, _, _) in enumerate(METHODS):
        front = fronts[(key, seed)]
        handles.append(scatter_pareto(axis, front[:, 0], front[:, 1], label, PARETO_COLORS[index]))
    combined = np.vstack([fronts[(key, seed)] for key, _, _, _, _ in METHODS])
    style_pareto_axis(axis, combined[:, 0], combined[:, 1], handles)
    path = OUTPUT_DIR / f"v3_dual_module_ablation_seed{seed}_pareto.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = []
    for method_key, method, market, selection, base_name in METHODS:
        for seed in SEEDS:
            path = run_dir(method_key, base_name, seed)
            objectives = load_objectives(path)
            validate_final(path, len(objectives))
            loaded.append((method_key, method, market, selection, seed, path, objectives))

    all_objectives = np.vstack([item[6] for item in loaded])
    minimum = all_objectives.min(axis=0)
    ranges = np.maximum(all_objectives.max(axis=0) - minimum, 1e-12)
    reference = (unique_front(all_objectives) - minimum) / ranges
    hv = Hypervolume(ref_point=np.array([1.1, 1.1]))
    rows = []
    fronts = {}
    for method_key, method, market, selection, seed, path, objectives in loaded:
        front = unique_front(objectives)
        normalized = (front - minimum) / ranges
        fronts[(method_key, seed)] = front
        rows.append(
            {
                "method_key": method_key,
                "method": method,
                "market": market,
                "selection": selection,
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
    order = {key: index for index, (key, _, _, _, _) in enumerate(METHODS)}
    summary = (
        detail.groupby(["method_key", "method", "market", "selection"], as_index=False)
        .agg(
            completed_seeds=("seed", "count"),
            hv_mean=("normalized_hv", "mean"),
            hv_std=("normalized_hv", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            sparsity_mean=("normalized_sparsity", "mean"),
            sparsity_std=("normalized_sparsity", "std"),
            front_points_mean=("front_points", "mean"),
            min_cost_mean=("min_cost", "mean"),
            min_carbon_mean=("min_carbon", "mean"),
            elapsed_hours_mean=("elapsed_seconds", lambda values: np.mean(values) / 3600.0),
        )
        .sort_values("method_key", key=lambda column: column.map(order))
    )
    effects, effect_summary, significance = paired_analysis(detail)
    interactions, interaction_summary = interaction_analysis(detail)
    paths = {
        "detail": OUTPUT_DIR / "v3_dual_module_ablation_5seed_detail.csv",
        "summary": OUTPUT_DIR / "v3_dual_module_ablation_5seed_summary.csv",
        "effects": OUTPUT_DIR / "v3_dual_module_ablation_5seed_paired_effects.csv",
        "effect_summary": OUTPUT_DIR / "v3_dual_module_ablation_5seed_paired_summary.csv",
        "significance": OUTPUT_DIR / "v3_dual_module_ablation_5seed_significance.csv",
        "interactions": OUTPUT_DIR / "v3_dual_module_ablation_5seed_interactions.csv",
        "interaction_summary": OUTPUT_DIR / "v3_dual_module_ablation_5seed_interaction_summary.csv",
    }
    detail.to_csv(paths["detail"], index=False)
    summary.to_csv(paths["summary"], index=False)
    effects.to_csv(paths["effects"], index=False)
    effect_summary.to_csv(paths["effect_summary"], index=False)
    significance.to_csv(paths["significance"], index=False)
    interactions.to_csv(paths["interactions"], index=False)
    interaction_summary.to_csv(paths["interaction_summary"], index=False)
    paths["metrics"] = plot_metrics(summary)
    paths["seedwise"] = plot_seedwise(detail)
    paths["pareto"] = plot_pareto(fronts)
    print(summary.to_string(index=False))
    print("\nPaired effects:")
    print(effect_summary.to_string(index=False))
    print("\nSelection x market interaction:")
    print(interaction_summary.to_string(index=False))
    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
