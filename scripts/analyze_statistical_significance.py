from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

ANALYSES = [
    {
        "group": "Algorithm comparison",
        "path": OUTPUT_DIR / "algorithm_comparison_3seed_detail.csv",
        "reference": "proposed",
        "comparators": ["fixed_weight_mappo", "masac", "maddpg", "mopso"],
    },
    {
        "group": "Ablation",
        "path": OUTPUT_DIR / "ablation_3seed_detail.csv",
        "reference": "carbon_ensemble",
        "comparators": [
            "carbon_contribution",
            "standard_ensemble",
            "standard_contribution",
        ],
    },
]

METRICS = [
    ("normalized_hypervolume", "Norm HV", True),
    ("igd", "IGD", False),
    ("normalized_sparsity", "Normalized sparsity", False),
]


def rank_biserial(oriented_difference):
    nonzero = oriented_difference[np.abs(oriented_difference) > 1e-15]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_result(data, group, reference, comparator, metric, metric_label, higher_is_better):
    reference_rows = data[data["method_key"] == reference][["seed", metric]].rename(
        columns={metric: "reference_value"}
    )
    comparator_rows = data[data["method_key"] == comparator][["seed", metric]].rename(
        columns={metric: "comparator_value"}
    )
    paired = reference_rows.merge(comparator_rows, on="seed", validate="one_to_one").sort_values("seed")
    if len(paired) < 2:
        raise RuntimeError(f"Not enough paired seeds for {group}: {reference} vs {comparator}")

    reference_values = paired["reference_value"].to_numpy(dtype=float)
    comparator_values = paired["comparator_value"].to_numpy(dtype=float)
    raw_difference = reference_values - comparator_values
    oriented_difference = raw_difference if higher_is_better else -raw_difference
    if np.all(np.abs(raw_difference) <= 1e-15):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(reference_values, comparator_values, alternative="two-sided", method="auto")
        statistic, p_value = float(result.statistic), float(result.pvalue)

    denominator = np.maximum(np.abs(comparator_values), 1e-15)
    relative_improvement = 100.0 * oriented_difference / denominator
    return {
        "group": group,
        "reference": reference,
        "comparator": comparator,
        "metric": metric,
        "metric_label": metric_label,
        "higher_is_better": higher_is_better,
        "paired_seeds": len(paired),
        "reference_mean": float(reference_values.mean()),
        "comparator_mean": float(comparator_values.mean()),
        "mean_oriented_difference": float(oriented_difference.mean()),
        "mean_relative_improvement_percent": float(relative_improvement.mean()),
        "wins": int(np.sum(oriented_difference > 1e-15)),
        "ties": int(np.sum(np.abs(oriented_difference) <= 1e-15)),
        "losses": int(np.sum(oriented_difference < -1e-15)),
        "wilcoxon_statistic": statistic,
        "p_value_two_sided": p_value,
        "rank_biserial_effect": rank_biserial(oriented_difference),
    }


def main():
    rows = []
    for analysis in ANALYSES:
        data = pd.read_csv(analysis["path"])
        for comparator in analysis["comparators"]:
            for metric, metric_label, higher_is_better in METRICS:
                rows.append(
                    paired_result(
                        data,
                        analysis["group"],
                        analysis["reference"],
                        comparator,
                        metric,
                        metric_label,
                        higher_is_better,
                    )
                )

    results = pd.DataFrame(rows)
    results["p_value_holm"] = np.nan
    for _, indices in results.groupby(["group", "metric"]).groups.items():
        results.loc[indices, "p_value_holm"] = holm_adjust(
            results.loc[indices, "p_value_two_sided"].to_numpy()
        )
    results["significant_0_05"] = results["p_value_holm"] < 0.05

    csv_path = OUTPUT_DIR / "statistical_significance_pairwise.csv"
    results.to_csv(csv_path, index=False)

    core = results[results["metric"].isin(["normalized_hypervolume", "igd"])].copy()
    core["comparison"] = core["group"] + ": " + core["reference"] + " vs " + core["comparator"]
    comparisons = list(dict.fromkeys(core["comparison"]))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), constrained_layout=True)
    colors = {True: "#2E8B57", False: "#C44E52"}
    for ax, (metric, metric_label, _) in zip(axes, METRICS[:2]):
        metric_rows = core[core["metric"] == metric].set_index("comparison").reindex(comparisons)
        values = metric_rows["rank_biserial_effect"].to_numpy()
        y = np.arange(len(metric_rows))
        ax.barh(y, values, color=[colors[value >= 0] for value in values], edgecolor="black", linewidth=0.5)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(metric_rows.index if ax is axes[0] else [])
        ax.set_xlim(-1.05, 1.05)
        ax.set_xlabel("Paired rank-biserial effect\n(positive favors reference method)")
        ax.set_title(metric_label)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    figure_path = OUTPUT_DIR / "statistical_significance_effect_sizes.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(
        results[
            [
                "group",
                "reference",
                "comparator",
                "metric_label",
                "wins",
                "ties",
                "losses",
                "mean_relative_improvement_percent",
                "p_value_two_sided",
                "p_value_holm",
                "rank_biserial_effect",
            ]
        ].to_string(index=False)
    )
    print(f"Saved: {csv_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
