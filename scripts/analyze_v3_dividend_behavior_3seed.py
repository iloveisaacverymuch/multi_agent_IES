import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_rel, wilcoxon

from analyze_current_carbon_cda_trading_3seed import (
    PREFERENCES,
    SEASONS,
    ensure_test_arrays_have_terminal_row,
    load_actor,
    representative_policies,
    simulate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
SEEDS = (11, 22, 44)
OUTPUT_TAG = "3seed"
METHODS = (
    ("standard", "Standard CDA", "standard_cda_hv_ensemble_extreme_capped32_kappa_0_10", "standard", 0.0),
    (
        "equal",
        "Carbon levy + equal dividend",
        "carbon_dividend_equal_v3_hv_ensemble_extreme_capped32_kappa_0_10_price_0_01",
        "carbon_dividend_equal_v3",
        0.01,
    ),
    (
        "targeted",
        "Carbon levy + targeted dividend",
        "carbon_dividend_v3_hv_ensemble_extreme_capped32_kappa_0_10_price_0_01",
        "carbon_dividend_v3",
        0.01,
    ),
)
COMPARISONS = (
    ("equal", "standard", "Equal dividend vs Standard"),
    ("targeted", "equal", "Targeted vs Equal dividend"),
    ("targeted", "standard", "Targeted vs Standard"),
)


def bootstrap_mean_ci(values, seed, samples=20000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def paired_rank_biserial(oriented_differences):
    differences = np.asarray(oriented_differences, dtype=float)
    differences = differences[np.abs(differences) > 1e-12]
    if len(differences) == 0:
        return 0.0
    ranks = rankdata(np.abs(differences))
    denominator = ranks.sum()
    return float((ranks[differences > 0.0].sum() - ranks[differences < 0.0].sum()) / denominator)


def add_grouped_pvalue_corrections(frame, columns):
    frame = frame.copy()
    for column in columns:
        frame[f"{column}_holm"] = np.nan
        frame[f"{column}_bh_fdr"] = np.nan
        for indices in frame.groupby("comparison").groups.values():
            indices = np.asarray(list(indices))
            values = frame.loc[indices, column].to_numpy(dtype=float)
            valid = np.isfinite(values)
            valid_indices = indices[valid]
            values = values[valid]
            if len(values) == 0:
                continue
            order = np.argsort(values)
            ordered = values[order]
            count = len(ordered)
            holm = np.maximum.accumulate((count - np.arange(count)) * ordered)
            bh = np.minimum.accumulate((ordered * count / np.arange(1, count + 1))[::-1])[::-1]
            frame.loc[valid_indices[order], f"{column}_holm"] = np.minimum(holm, 1.0)
            frame.loc[valid_indices[order], f"{column}_bh_fdr"] = np.minimum(bh, 1.0)
    return frame


def run_dir(base_name, seed):
    return RUN_ROOT / (base_name if seed == 44 else f"{base_name}_seed_{seed}")


def evaluate():
    rows = []
    selected_rows = []
    for key, label, base_name, mechanism, price in METHODS:
        for seed in SEEDS:
            path = run_dir(base_name, seed)
            selected = representative_policies(path)
            for preference in PREFERENCES:
                policy_info = selected[preference]
                selected_rows.append(
                    {
                        "method_key": key,
                        "method": label,
                        "seed": seed,
                        "preference": preference,
                        **policy_info,
                    }
                )
                actors = load_actor(path, policy_info["policy_file"])
                for season, start in SEASONS.items():
                    episode = simulate(
                        actors,
                        mechanism,
                        seed,
                        preference,
                        policy_info,
                        season,
                        start,
                        "v3_dividend_ablation_native",
                        carbon_price=price,
                    )
                    for row in episode:
                        row["method_key"] = key
                        row["method"] = label
                    rows.extend(episode)
    return pd.DataFrame(rows), pd.DataFrame(selected_rows)


def summarize(detail):
    summary = detail.groupby(["method_key", "method", "seed"], as_index=False).agg(
        total_cost=("step_cost", "sum"),
        total_carbon=("step_carbon", "sum"),
        grid_buy=("grid_buy", "sum"),
        grid_sell=("grid_sell", "sum"),
        peer_trade=("peer_trade_amount", "sum"),
        waste_heat=("waste_heat", "sum"),
        dividend_fund=("carbon_dividend_fund", "sum"),
        absolute_transfer=("carbon_transfer_abs", "sum"),
        max_balance_error=("carbon_dividend_balance_error", lambda values: float(np.abs(values).max())),
        active_hours=("carbon_dividend_fund", lambda values: int((values > 1e-12).sum())),
        eligible_trade_hours=("avoided_carbon_credit_basis", lambda values: int((values > 1e-12).sum())),
        hours=("hour", "count"),
    )
    summary["fund_active_rate"] = summary["active_hours"] / summary["hours"]
    summary["eligible_trade_rate"] = summary["eligible_trade_hours"] / summary["hours"]
    return summary


def paired_effects(seed_summary):
    indexed = seed_summary.set_index(["method_key", "seed"])
    rows = []
    tests = []
    for comparison_index, (new_key, old_key, comparison) in enumerate(COMPARISONS):
        for seed in SEEDS:
            row = {"comparison": comparison, "seed": seed}
            for metric in ("total_cost", "total_carbon", "grid_buy", "peer_trade", "waste_heat"):
                new_value = float(indexed.loc[(new_key, seed), metric])
                old_value = float(indexed.loc[(old_key, seed), metric])
                row[f"{metric}_change_percent"] = (
                    100.0 * (new_value / old_value - 1.0) if abs(old_value) > 1e-12 else np.nan
                )
            rows.append(row)
        for metric_index, (metric, alternative) in enumerate((
            ("total_cost", "less"),
            ("total_carbon", "less"),
            ("grid_buy", "less"),
            ("peer_trade", "greater"),
        )):
            new_values = np.array([indexed.loc[(new_key, seed), metric] for seed in SEEDS])
            old_values = np.array([indexed.loc[(old_key, seed), metric] for seed in SEEDS])
            if alternative == "greater":
                improvements = 100.0 * (new_values / old_values - 1.0)
                oriented_differences = new_values - old_values
            else:
                improvements = 100.0 * (1.0 - new_values / old_values)
                oriented_differences = old_values - new_values
            ci_low, ci_high = bootstrap_mean_ci(
                improvements,
                seed=2000 + 10 * comparison_index + metric_index,
            )
            t_result = ttest_rel(new_values, old_values, alternative=alternative)
            try:
                w_pvalue = float(wilcoxon(new_values, old_values, alternative=alternative).pvalue)
            except ValueError:
                w_pvalue = np.nan
            tests.append(
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
                    "num_pairs": len(SEEDS),
                }
            )
    effects = pd.DataFrame(rows)
    summary = effects.groupby("comparison", as_index=False).agg(
        cost_change_mean_percent=("total_cost_change_percent", "mean"),
        cost_change_std_percent=("total_cost_change_percent", "std"),
        carbon_change_mean_percent=("total_carbon_change_percent", "mean"),
        carbon_change_std_percent=("total_carbon_change_percent", "std"),
        grid_buy_change_mean_percent=("grid_buy_change_percent", "mean"),
        peer_trade_change_mean_percent=("peer_trade_change_percent", "mean"),
        waste_heat_change_mean_percent=("waste_heat_change_percent", "mean"),
    )
    tests = add_grouped_pvalue_corrections(
        pd.DataFrame(tests),
        ("paired_t_pvalue", "wilcoxon_pvalue"),
    )
    return effects, summary, tests


def plot_summary(seed_summary):
    order = [key for key, _, _, _, _ in METHODS]
    labels = {key: label for key, label, _, _, _ in METHODS}
    aggregated = seed_summary.groupby("method_key").agg(
        total_cost_mean=("total_cost", "mean"),
        total_cost_std=("total_cost", "std"),
        total_carbon_mean=("total_carbon", "mean"),
        total_carbon_std=("total_carbon", "std"),
        grid_buy_mean=("grid_buy", "mean"),
        grid_buy_std=("grid_buy", "std"),
        peer_trade_mean=("peer_trade", "mean"),
        peer_trade_std=("peer_trade", "std"),
    ).reindex(order)
    x = np.arange(len(order))
    colors = ["#4C78A8", "#F28E2B", "#59A14F"]
    plt.rcParams.update({"font.family": "Times New Roman", "savefig.dpi": 600})
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.8), constrained_layout=True)
    specs = (
        ("total_cost", "Test cost (CNY)"),
        ("total_carbon", "Test emissions (kg CO$_2$)"),
        ("grid_buy", "Grid imports (kWh)"),
        ("peer_trade", "Peer trade (kWh)"),
    )
    for axis, (metric, ylabel) in zip(axes, specs):
        axis.bar(
            x,
            aggregated[f"{metric}_mean"],
            yerr=aggregated[f"{metric}_std"],
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            capsize=4,
        )
        axis.set_xticks(x)
        axis.set_xticklabels([labels[key] for key in order], rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
    path = OUTPUT_DIR / f"v3_dividend_ablation_behavior_{OUTPUT_TAG}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    global SEEDS, OUTPUT_TAG
    parser = argparse.ArgumentParser(description="Analyze V3 dividend behavior across seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()
    SEEDS = tuple(args.seeds)
    OUTPUT_TAG = args.output_tag or f"{len(SEEDS)}seed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_test_arrays_have_terminal_row()
    detail, selected = evaluate()
    seed_summary = summarize(detail)
    effects, effect_summary, significance = paired_effects(seed_summary)
    paths = {
        "detail": OUTPUT_DIR / f"v3_dividend_ablation_behavior_detail_{OUTPUT_TAG}.csv",
        "seed_summary": OUTPUT_DIR / f"v3_dividend_ablation_behavior_seed_summary_{OUTPUT_TAG}.csv",
        "effects": OUTPUT_DIR / f"v3_dividend_ablation_behavior_effects_{OUTPUT_TAG}.csv",
        "effect_summary": OUTPUT_DIR / f"v3_dividend_ablation_behavior_effect_summary_{OUTPUT_TAG}.csv",
        "significance": OUTPUT_DIR / f"v3_dividend_ablation_behavior_significance_{OUTPUT_TAG}.csv",
        "policies": OUTPUT_DIR / f"v3_dividend_ablation_behavior_policies_{OUTPUT_TAG}.csv",
    }
    detail.to_csv(paths["detail"], index=False)
    seed_summary.to_csv(paths["seed_summary"], index=False)
    effects.to_csv(paths["effects"], index=False)
    effect_summary.to_csv(paths["effect_summary"], index=False)
    significance.to_csv(paths["significance"], index=False)
    selected.to_csv(paths["policies"], index=False)
    paths["figure"] = plot_summary(seed_summary)
    print(seed_summary.to_string(index=False))
    print("\nPaired behavior effects:")
    print(effect_summary.to_string(index=False))
    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
