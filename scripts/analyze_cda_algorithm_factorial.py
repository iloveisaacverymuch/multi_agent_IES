from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "cda_algorithm_factorial"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"
SEEDS = [11, 22, 33, 44, 55]
GROUPS = [
    ("standard_maddpg", "Standard CDA", "Fixed MADDPG", "strict_standard_maddpg_seed_{seed}"),
    (
        "standard_all_evolving",
        "Standard CDA",
        "All-task E-MADDPG",
        "strict_standard_all_evolving_kappa0_seed_{seed}",
    ),
    ("v3_maddpg", "Carbon-Dividend V3", "Fixed MADDPG", "strict_v3_maddpg_seed_{seed}"),
    (
        "v3_all_evolving",
        "Carbon-Dividend V3",
        "All-task E-MADDPG",
        "strict_v3_all_evolving_kappa0_seed_{seed}",
    ),
]


def unique_front(objectives):
    objectives = np.atleast_2d(np.asarray(objectives, dtype=np.float64))
    dominated = np.zeros(len(objectives), dtype=bool)
    for index, value in enumerate(objectives):
        dominated[index] = np.any(
            np.all(objectives <= value, axis=1) & np.any(objectives < value, axis=1)
        )
    front = objectives[~dominated]
    _, indices = np.unique(np.round(front, 6), axis=0, return_index=True)
    front = front[np.sort(indices)]
    return front[np.argsort(front[:, 0])]


def igd(front, reference):
    distances = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def sparsity(front):
    if len(front) < 2:
        return 0.0
    differences = np.diff(front[np.argsort(front[:, 0])], axis=0)
    return float(np.mean(np.sum(differences**2, axis=1)))


def one_sided(values, alternative="greater"):
    values = np.asarray(values, dtype=np.float64)
    if np.allclose(values, 0.0):
        return 1.0
    zeros = np.zeros_like(values)
    return float(wilcoxon(values, zeros, alternative=alternative).pvalue)


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    loaded = []
    for key, market, algorithm, pattern in GROUPS:
        for seed in SEEDS:
            path = RUN_ROOT / pattern.format(seed=seed) / "strict_test" / "objs.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            loaded.append(
                (key, market, algorithm, seed, np.atleast_2d(np.loadtxt(path, delimiter=",")))
            )
    reference = unique_front(np.vstack([item[4] for item in loaded]))
    all_fronts = np.vstack([unique_front(item[4]) for item in loaded])
    ideal = reference.min(axis=0)
    scale = np.maximum(all_fronts.max(axis=0) - ideal, 1e-12)
    reference_norm = (reference - ideal) / scale
    indicator = Hypervolume(ref_point=np.array([1.1, 1.1]))
    rows = []
    for key, market, algorithm, seed, objectives in loaded:
        front = unique_front(objectives)
        normalized = (front - ideal) / scale
        rows.append(
            {
                "group_key": key,
                "market": market,
                "algorithm": algorithm,
                "seed": seed,
                "normalized_hv": float(indicator.do(normalized)),
                "igd": igd(normalized, reference_norm),
                "normalized_sparsity": sparsity(normalized),
                "front_points": len(front),
                "min_cost": float(front[:, 0].min()),
                "min_carbon": float(front[:, 1].min()),
            }
        )
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["group_key", "market", "algorithm"], as_index=False).agg(
        hv_mean=("normalized_hv", "mean"),
        hv_std=("normalized_hv", "std"),
        igd_mean=("igd", "mean"),
        igd_std=("igd", "std"),
        sparsity_mean=("normalized_sparsity", "mean"),
        front_points_mean=("front_points", "mean"),
        min_cost_mean=("min_cost", "mean"),
        min_carbon_mean=("min_carbon", "mean"),
    )

    indexed = detail.set_index(["group_key", "seed"])
    effects = []
    comparisons = [
        ("CDA effect with Fixed MADDPG", "v3_maddpg", "standard_maddpg"),
        ("CDA effect with All-task E-MADDPG", "v3_all_evolving", "standard_all_evolving"),
        ("Algorithm effect under Standard CDA", "standard_all_evolving", "standard_maddpg"),
        ("Algorithm effect under Carbon-Dividend V3", "v3_all_evolving", "v3_maddpg"),
    ]
    for name, preferred, baseline in comparisons:
        for metric, higher_better in (("normalized_hv", True), ("igd", False)):
            first = indexed.loc[pd.IndexSlice[preferred, SEEDS], metric].to_numpy()
            second = indexed.loc[pd.IndexSlice[baseline, SEEDS], metric].to_numpy()
            oriented = first - second if higher_better else second - first
            effects.append(
                {
                    "effect": name,
                    "metric": metric,
                    "wins": int(np.sum(oriented > 0)),
                    "losses": int(np.sum(oriented < 0)),
                    "mean_improvement": float(oriented.mean()),
                    "one_sided_p": one_sided(oriented),
                }
            )
    for metric, higher_better in (("normalized_hv", True), ("igd", False)):
        v3_main = indexed.loc[pd.IndexSlice["v3_all_evolving", SEEDS], metric].to_numpy()
        standard_main = indexed.loc[
            pd.IndexSlice["standard_all_evolving", SEEDS], metric
        ].to_numpy()
        v3_fixed = indexed.loc[pd.IndexSlice["v3_maddpg", SEEDS], metric].to_numpy()
        standard_fixed = indexed.loc[pd.IndexSlice["standard_maddpg", SEEDS], metric].to_numpy()
        interaction = (v3_main - standard_main) - (v3_fixed - standard_fixed)
        if not higher_better:
            interaction = -interaction
        effects.append(
            {
                "effect": "Market x algorithm interaction",
                "metric": metric,
                "wins": int(np.sum(interaction > 0)),
                "losses": int(np.sum(interaction < 0)),
                "mean_improvement": float(interaction.mean()),
                "one_sided_p": one_sided(interaction),
            }
        )
    effects = pd.DataFrame(effects)
    detail.to_csv(RESULT_ROOT / "detail.csv", index=False)
    summary.to_csv(RESULT_ROOT / "summary.csv", index=False)
    effects.to_csv(RESULT_ROOT / "factor_effects.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), constrained_layout=True)
    markets = ["Standard CDA", "Carbon-Dividend V3"]
    algorithms = ["Fixed MADDPG", "All-task E-MADDPG"]
    colors = ["#F28E2B", "#C44E52"]
    settings = [
        ("hv_mean", "hv_std", "Normalized HV"),
        ("igd_mean", "igd_std", "IGD"),
        ("front_points_mean", None, "Pareto points"),
    ]
    x = np.arange(len(markets))
    width = 0.34
    for ax, (mean, std, ylabel) in zip(axes, settings):
        for index, algorithm in enumerate(algorithms):
            frame = summary[summary.algorithm == algorithm].set_index("market").loc[markets]
            errors = frame[std] if std else None
            ax.bar(
                x + (index - 0.5) * width,
                frame[mean],
                width,
                yerr=errors,
                color=colors[index],
                edgecolor="black",
                linewidth=0.5,
                capsize=3,
                label=algorithm,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(markets, rotation=12, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURE_ROOT / "cda_algorithm_factorial.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))
    print("\nFactor effects:\n" + effects.to_string(index=False))


if __name__ == "__main__":
    main()
