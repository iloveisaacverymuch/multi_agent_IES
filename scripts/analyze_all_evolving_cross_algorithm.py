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
RESULT_ROOT = PROJECT_ROOT / "outputs" / "all_evolving_cross_algorithm"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"
SEEDS = [11, 22, 33, 44, 55]
METHODS = [
    ("all_evolving_kappa0", "All-task Mean-HV E-MADDPG"),
    ("anchor_evolutionary_maddpg", "Anchor-guided E-MADDPG"),
    ("maddpg", "Fixed-weight MADDPG"),
    ("fixed_weight_mappo", "Fixed-weight MAPPO"),
    ("masac", "MASAC"),
    ("mopso", "MOPSO"),
]
COLORS = ["#C44E52", "#8172B3", "#F28E2B", "#4C78A8", "#59A14F", "#8C8C8C"]
MARKERS = ["o", "P", "D", "s", "^", "X"]


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


def load_runs():
    loaded = []
    for method, label in METHODS:
        for seed in SEEDS:
            path = RUN_ROOT / f"strict_v3_{method}_seed_{seed}" / "strict_test" / "objs.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            loaded.append((method, label, seed, np.atleast_2d(np.loadtxt(path, delimiter=","))))
    return loaded


def calculate(loaded):
    reference_front = unique_front(np.vstack([item[3] for item in loaded]))
    all_fronts = np.vstack([unique_front(item[3]) for item in loaded])
    ideal = reference_front.min(axis=0)
    nadir = all_fronts.max(axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
    reference_norm = (reference_front - ideal) / scale
    hv = Hypervolume(ref_point=np.array([1.1, 1.1]))
    rows = []
    for method, label, seed, objectives in loaded:
        front = unique_front(objectives)
        normalized = (front - ideal) / scale
        rows.append(
            {
                "method_key": method,
                "method": label,
                "seed": seed,
                "archive_points": len(objectives),
                "front_points": len(front),
                "normalized_hv": float(hv.do(normalized)),
                "igd": igd(normalized, reference_norm),
                "normalized_sparsity": sparsity(normalized),
                "min_cost": float(front[:, 0].min()),
                "min_carbon": float(front[:, 1].min()),
            }
        )
    return pd.DataFrame(rows), ideal, nadir, reference_front


def paired_tests(detail):
    proposed = detail[detail.method_key == "all_evolving_kappa0"].set_index("seed")
    rows = []
    for method, label in METHODS[1:]:
        baseline = detail[detail.method_key == method].set_index("seed")
        for metric, alternative in (("normalized_hv", "greater"), ("igd", "less")):
            first = proposed.loc[SEEDS, metric].to_numpy()
            second = baseline.loc[SEEDS, metric].to_numpy()
            oriented = first - second if alternative == "greater" else second - first
            p_value = 1.0 if np.allclose(first, second) else float(
                wilcoxon(first, second, alternative=alternative).pvalue
            )
            rows.append(
                {
                    "comparator": label,
                    "metric": metric,
                    "wins": int(np.sum(oriented > 0)),
                    "losses": int(np.sum(oriented < 0)),
                    "mean_improvement": float(np.mean(oriented)),
                    "one_sided_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def plot_metrics(summary):
    order = [key for key, _ in METHODS]
    frame = summary.set_index("method_key").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    settings = [
        ("hv_mean", "hv_std", "Normalized HV"),
        ("igd_mean", "igd_std", "IGD"),
        ("front_points_mean", "front_points_std", "Pareto points"),
    ]
    x = np.arange(len(frame))
    for ax, (mean, std, ylabel) in zip(axes, settings):
        ax.bar(
            x,
            frame[mean],
            yerr=frame[std].fillna(0.0),
            color=COLORS,
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(frame["method"], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(FIGURE_ROOT / "all_evolving_cross_algorithm_metrics.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_fronts(loaded):
    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    for (method, label), color, marker in zip(METHODS, COLORS, MARKERS):
        pooled = np.vstack([values for key, _, _, values in loaded if key == method])
        front = unique_front(pooled)
        ax.plot(
            front[:, 0],
            front[:, 1],
            color=color,
            marker=marker,
            markersize=4,
            linewidth=1.15,
            alpha=0.86,
            label=label,
        )
    ax.set_xlabel("96-hour operating cost")
    ax.set_ylabel("96-hour carbon emissions")
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURE_ROOT / "all_evolving_cross_algorithm_pareto.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    loaded = load_runs()
    detail, ideal, nadir, reference = calculate(loaded)
    summary = (
        detail.groupby(["method_key", "method"], as_index=False)
        .agg(
            hv_mean=("normalized_hv", "mean"),
            hv_std=("normalized_hv", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            sparsity_mean=("normalized_sparsity", "mean"),
            front_points_mean=("front_points", "mean"),
            front_points_std=("front_points", "std"),
            min_cost_mean=("min_cost", "mean"),
            min_carbon_mean=("min_carbon", "mean"),
        )
        .sort_values("hv_mean", ascending=False)
    )
    tests = paired_tests(detail)
    detail.to_csv(RESULT_ROOT / "detail.csv", index=False)
    summary.to_csv(RESULT_ROOT / "summary.csv", index=False)
    tests.to_csv(RESULT_ROOT / "paired_tests.csv", index=False)
    np.savetxt(RESULT_ROOT / "pooled_reference_front.txt", reference, delimiter=",", fmt="%.6f")
    pd.Series({"ideal": ideal.tolist(), "nadir": nadir.tolist()}).to_json(
        RESULT_ROOT / "metric_bounds.json", indent=2
    )
    plot_metrics(summary)
    plot_fronts(loaded)
    print(summary.to_string(index=False))
    print("\nPaired tests:\n" + tests.to_string(index=False))


if __name__ == "__main__":
    main()
