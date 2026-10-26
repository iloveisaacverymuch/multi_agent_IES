from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_rel, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
INPUT_PATH = OUTPUT_DIR / "v3_dividend_ablation_behavior_detail_5seed.csv"
SEEDS = (11, 22, 33, 44, 55)
PREFERENCES = ("cost_priority", "balanced", "carbon_priority")
PREFERENCE_LABELS = {
    "cost_priority": "Cost priority",
    "balanced": "Balanced",
    "carbon_priority": "Carbon priority",
}
SEASONS = ("spring", "summer", "autumn", "winter")
SEASON_LABELS = {season: season.title() for season in SEASONS}
COMPARISONS = (
    ("targeted", "equal", "Targeted vs Equal dividend"),
    ("targeted", "standard", "Targeted vs Standard CDA"),
    ("equal", "standard", "Equal dividend vs Standard CDA"),
)
METRICS = (
    ("total_cost", "Cost", "less", "%"),
    ("total_carbon", "Carbon emissions", "less", "%"),
    ("grid_buy", "Grid imports", "less", "%"),
    ("peer_trade", "Peer trade", "greater", "kWh"),
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
    return float(
        (ranks[differences > 0.0].sum() - ranks[differences < 0.0].sum())
        / ranks.sum()
    )


def corrected(values, method):
    values = np.asarray(values, dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values)
    ordered_indices = np.where(valid)[0][np.argsort(values[valid])]
    ordered = values[ordered_indices]
    count = len(ordered)
    if count == 0:
        return result
    if method == "holm":
        adjusted = np.maximum.accumulate((count - np.arange(count)) * ordered)
    elif method == "bh":
        adjusted = np.minimum.accumulate(
            (ordered * count / np.arange(1, count + 1))[::-1]
        )[::-1]
    else:
        raise ValueError(f"Unsupported correction method: {method}")
    result[ordered_indices] = np.minimum(adjusted, 1.0)
    return result


def add_family_corrections(frame, family_columns):
    frame = frame.copy()
    for column in ("paired_t_pvalue", "wilcoxon_pvalue"):
        frame[f"{column}_holm"] = np.nan
        frame[f"{column}_bh_fdr"] = np.nan
        for indices in frame.groupby(family_columns, dropna=False).groups.values():
            indices = list(indices)
            values = frame.loc[indices, column].to_numpy(dtype=float)
            frame.loc[indices, f"{column}_holm"] = corrected(values, "holm")
            frame.loc[indices, f"{column}_bh_fdr"] = corrected(values, "bh")
    return frame


def load_and_aggregate():
    detail = pd.read_csv(INPUT_PATH)
    expected_rows = len(SEEDS) * 3 * len(PREFERENCES) * len(SEASONS) * 24
    if len(detail) != expected_rows:
        raise ValueError(f"Expected {expected_rows} hourly rows, found {len(detail)}")
    counts = detail.groupby(["method_key", "seed", "preference", "season"]).size()
    if not (counts == 24).all():
        raise ValueError("Every method/seed/preference/season block must contain 24 hours")
    grouped = detail.groupby(
        ["method_key", "method", "seed", "preference", "season"],
        as_index=False,
    ).agg(
        total_cost=("step_cost", "sum"),
        total_carbon=("step_carbon", "sum"),
        grid_buy=("grid_buy", "sum"),
        grid_sell=("grid_sell", "sum"),
        peer_trade=("peer_trade_amount", "sum"),
        dividend_fund=("carbon_dividend_fund", "sum"),
        carbon_transfer=("carbon_transfer_abs", "sum"),
        max_balance_error=("carbon_dividend_balance_error", lambda values: float(np.abs(values).max())),
        hours=("hour", "count"),
    )
    return grouped


def aggregate_level(cell_detail, strata):
    group_columns = ["method_key", "method", "seed", *strata]
    return cell_detail.groupby(group_columns, as_index=False).agg(
        total_cost=("total_cost", "sum"),
        total_carbon=("total_carbon", "sum"),
        grid_buy=("grid_buy", "sum"),
        grid_sell=("grid_sell", "sum"),
        peer_trade=("peer_trade", "sum"),
        dividend_fund=("dividend_fund", "sum"),
        carbon_transfer=("carbon_transfer", "sum"),
        max_balance_error=("max_balance_error", "max"),
        hours=("hours", "sum"),
    )


def summarize_methods(data, strata):
    group_columns = ["method_key", "method", *strata]
    aggregations = {}
    for metric, _, _, _ in METRICS:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    aggregations["completed_seeds"] = ("seed", "count")
    aggregations["max_balance_error"] = ("max_balance_error", "max")
    return data.groupby(group_columns, as_index=False).agg(**aggregations)


def effect_values(new_values, old_values, alternative, unit):
    oriented_differences = (
        new_values - old_values if alternative == "greater" else old_values - new_values
    )
    if unit == "%":
        return 100.0 * oriented_differences / old_values, oriented_differences
    return oriented_differences, oriented_differences


def analyze_level(data, strata, level_name):
    indexed = data.set_index(["method_key", *strata, "seed"])
    stratum_values = data[list(strata)].drop_duplicates().to_dict("records") if strata else [{}]
    effects = []
    statistics = []
    for comparison_index, (new_key, old_key, comparison) in enumerate(COMPARISONS):
        for stratum_index, stratum in enumerate(stratum_values):
            new_rows = []
            old_rows = []
            for seed in SEEDS:
                index = (new_key, *[stratum[column] for column in strata], seed)
                old_index = (old_key, *[stratum[column] for column in strata], seed)
                new_rows.append(indexed.loc[index])
                old_rows.append(indexed.loc[old_index])
            for metric_index, (metric, _, alternative, unit) in enumerate(METRICS):
                new_values = np.array([float(row[metric]) for row in new_rows])
                old_values = np.array([float(row[metric]) for row in old_rows])
                metric_effects, oriented_differences = effect_values(
                    new_values,
                    old_values,
                    alternative,
                    unit,
                )
                for seed, new_value, old_value, effect in zip(SEEDS, new_values, old_values, metric_effects):
                    effects.append(
                        {
                            "level": level_name,
                            "comparison": comparison,
                            **stratum,
                            "seed": seed,
                            "metric": metric,
                            "effect_unit": unit,
                            "new_value": new_value,
                            "old_value": old_value,
                            "effect_value": effect,
                        }
                    )
                ci_low, ci_high = bootstrap_mean_ci(
                    metric_effects,
                    seed=5000 + 100 * comparison_index + 10 * stratum_index + metric_index,
                )
                t_result = ttest_rel(new_values, old_values, alternative=alternative)
                try:
                    w_pvalue = float(wilcoxon(new_values, old_values, alternative=alternative).pvalue)
                except ValueError:
                    w_pvalue = np.nan
                statistics.append(
                    {
                        "level": level_name,
                        "comparison": comparison,
                        **stratum,
                        "metric": metric,
                        "effect_unit": unit,
                        "alternative": alternative,
                        "mean_effect": float(metric_effects.mean()),
                        "std_effect": float(metric_effects.std(ddof=1)),
                        "bootstrap_ci95_low": float(ci_low),
                        "bootstrap_ci95_high": float(ci_high),
                        "improved_seeds": int((metric_effects > 0.0).sum()),
                        "paired_t_pvalue": float(t_result.pvalue),
                        "wilcoxon_pvalue": w_pvalue,
                        "rank_biserial": paired_rank_biserial(oriented_differences),
                        "num_pairs": len(SEEDS),
                    }
                )
    statistics = add_family_corrections(
        pd.DataFrame(statistics),
        ["level", "comparison", "metric"],
    )
    return pd.DataFrame(effects), statistics


def plot_heatmaps(cell_statistics, comparison, output_name):
    selected = cell_statistics[cell_statistics["comparison"] == comparison]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.8), constrained_layout=True)
    plt.rcParams.update({"font.family": "Times New Roman", "savefig.dpi": 600})
    for axis, (metric, label, _, _) in zip(axes.flat, METRICS):
        metric_rows = selected[selected["metric"] == metric]
        matrix = (
            metric_rows.pivot(index="preference", columns="season", values="mean_effect")
            .reindex(index=PREFERENCES, columns=SEASONS)
        )
        significance = (
            metric_rows.pivot(index="preference", columns="season", values="wilcoxon_pvalue_bh_fdr")
            .reindex(index=PREFERENCES, columns=SEASONS)
        )
        limit = max(1.0, float(np.nanmax(np.abs(matrix.to_numpy()))))
        image = axis.imshow(
            matrix,
            cmap="RdYlGn",
            norm=colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            aspect="auto",
        )
        for row in range(len(PREFERENCES)):
            for column in range(len(SEASONS)):
                value = matrix.iloc[row, column]
                marker = "*" if significance.iloc[row, column] < 0.05 else ""
                axis.text(column, row, f"{value:+.1f}{marker}", ha="center", va="center", fontsize=10)
        axis.set_xticks(range(len(SEASONS)), [SEASON_LABELS[value] for value in SEASONS])
        axis.set_yticks(range(len(PREFERENCES)), [PREFERENCE_LABELS[value] for value in PREFERENCES])
        effect_label = "improvement (%)" if metric_rows["effect_unit"].iloc[0] == "%" else "increase (kWh)"
        axis.set_title(f"{label} {effect_label}")
        fig.colorbar(image, ax=axis, shrink=0.82)
    path = OUTPUT_DIR / output_name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_level_effects(statistics, stratum, order, labels, output_name):
    comparisons = ("Targeted vs Equal dividend", "Targeted vs Standard CDA")
    fig, axes = plt.subplots(2, 4, figsize=(13.4, 6.5), constrained_layout=True)
    plt.rcParams.update({"font.family": "Times New Roman", "savefig.dpi": 600})
    for row, comparison in enumerate(comparisons):
        selected = statistics[statistics["comparison"] == comparison]
        for column, (metric, label, _, unit) in enumerate(METRICS):
            axis = axes[row, column]
            values = selected[selected["metric"] == metric].set_index(stratum).reindex(order)
            means = values["mean_effect"].to_numpy()
            lower = means - values["bootstrap_ci95_low"].to_numpy()
            upper = values["bootstrap_ci95_high"].to_numpy() - means
            bars = axis.bar(
                np.arange(len(order)),
                means,
                yerr=np.vstack([lower, upper]),
                color="#59A14F" if row == 0 else "#4C78A8",
                edgecolor="black",
                linewidth=0.6,
                capsize=4,
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xticks(np.arange(len(order)), [labels[value] for value in order], rotation=18, ha="right")
            axis.set_ylabel("Improvement (%)" if unit == "%" else "Increase (kWh)")
            axis.set_title(label)
            axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
            for bar, qvalue in zip(bars, values["wilcoxon_pvalue_bh_fdr"]):
                if qvalue < 0.05:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height(),
                        "*",
                        ha="center",
                        va="bottom" if bar.get_height() >= 0.0 else "top",
                        fontsize=12,
                    )
            if column == 0:
                axis.text(
                    0.02,
                    0.97,
                    comparison,
                    transform=axis.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                )
    path = OUTPUT_DIR / output_name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cell_detail = load_and_aggregate()
    preference_detail = aggregate_level(cell_detail, ("preference",))
    season_detail = aggregate_level(cell_detail, ("season",))

    cell_summary = summarize_methods(cell_detail, ("preference", "season"))
    preference_summary = summarize_methods(preference_detail, ("preference",))
    season_summary = summarize_methods(season_detail, ("season",))
    cell_effects, cell_statistics = analyze_level(cell_detail, ("preference", "season"), "cell")
    preference_effects, preference_statistics = analyze_level(
        preference_detail,
        ("preference",),
        "preference",
    )
    season_effects, season_statistics = analyze_level(season_detail, ("season",), "season")

    paths = {
        "cell_detail": OUTPUT_DIR / "v3_preference_season_5seed_cell_detail.csv",
        "cell_summary": OUTPUT_DIR / "v3_preference_season_5seed_cell_summary.csv",
        "cell_effects": OUTPUT_DIR / "v3_preference_season_5seed_cell_effects.csv",
        "cell_statistics": OUTPUT_DIR / "v3_preference_season_5seed_cell_statistics.csv",
        "preference_summary": OUTPUT_DIR / "v3_preference_season_5seed_preference_summary.csv",
        "preference_effects": OUTPUT_DIR / "v3_preference_season_5seed_preference_effects.csv",
        "preference_statistics": OUTPUT_DIR / "v3_preference_season_5seed_preference_statistics.csv",
        "season_summary": OUTPUT_DIR / "v3_preference_season_5seed_season_summary.csv",
        "season_effects": OUTPUT_DIR / "v3_preference_season_5seed_season_effects.csv",
        "season_statistics": OUTPUT_DIR / "v3_preference_season_5seed_season_statistics.csv",
    }
    cell_detail.to_csv(paths["cell_detail"], index=False)
    cell_summary.to_csv(paths["cell_summary"], index=False)
    cell_effects.to_csv(paths["cell_effects"], index=False)
    cell_statistics.to_csv(paths["cell_statistics"], index=False)
    preference_summary.to_csv(paths["preference_summary"], index=False)
    preference_effects.to_csv(paths["preference_effects"], index=False)
    preference_statistics.to_csv(paths["preference_statistics"], index=False)
    season_summary.to_csv(paths["season_summary"], index=False)
    season_effects.to_csv(paths["season_effects"], index=False)
    season_statistics.to_csv(paths["season_statistics"], index=False)

    paths["heatmap_equal"] = plot_heatmaps(
        cell_statistics,
        "Targeted vs Equal dividend",
        "v3_preference_season_5seed_heatmap_vs_equal.png",
    )
    paths["heatmap_standard"] = plot_heatmaps(
        cell_statistics,
        "Targeted vs Standard CDA",
        "v3_preference_season_5seed_heatmap_vs_standard.png",
    )
    paths["preference_figure"] = plot_level_effects(
        preference_statistics,
        "preference",
        PREFERENCES,
        PREFERENCE_LABELS,
        "v3_preference_season_5seed_by_preference.png",
    )
    paths["season_figure"] = plot_level_effects(
        season_statistics,
        "season",
        SEASONS,
        SEASON_LABELS,
        "v3_preference_season_5seed_by_season.png",
    )

    print("Targeted vs Equal dividend by preference:")
    print(
        preference_statistics[
            preference_statistics["comparison"] == "Targeted vs Equal dividend"
        ][
            [
                "preference",
                "metric",
                "effect_unit",
                "mean_effect",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "improved_seeds",
                "wilcoxon_pvalue",
                "wilcoxon_pvalue_bh_fdr",
                "rank_biserial",
            ]
        ].to_string(index=False)
    )
    print("\nTargeted vs Equal dividend by season:")
    print(
        season_statistics[season_statistics["comparison"] == "Targeted vs Equal dividend"][
            [
                "season",
                "metric",
                "effect_unit",
                "mean_effect",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "improved_seeds",
                "wilcoxon_pvalue",
                "wilcoxon_pvalue_bh_fdr",
                "rank_biserial",
            ]
        ].to_string(index=False)
    )
    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
