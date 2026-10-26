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
RESULT_ROOT = PROJECT_ROOT / "outputs" / "all_evolving_selection_ablation"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"
SEEDS = [11, 22, 33, 44, 55]
METHODS = [
    ("all_evolving_kappa0", "Mean-HV selection"),
    ("all_evolving_random", "Random preference"),
]
COLORS = ["#C44E52", "#4C78A8"]


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


def calculate():
    loaded = []
    for method, label in METHODS:
        for seed in SEEDS:
            path = RUN_ROOT / f"strict_v3_{method}_seed_{seed}" / "strict_test" / "objs.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            loaded.append((method, label, seed, np.atleast_2d(np.loadtxt(path, delimiter=","))))
    reference = unique_front(np.vstack([item[3] for item in loaded]))
    fronts = np.vstack([unique_front(item[3]) for item in loaded])
    ideal = reference.min(axis=0)
    scale = np.maximum(fronts.max(axis=0) - ideal, 1e-12)
    reference_norm = (reference - ideal) / scale
    indicator = Hypervolume(ref_point=np.array([1.1, 1.1]))
    rows = []
    for method, label, seed, objectives in loaded:
        front = unique_front(objectives)
        normalized = (front - ideal) / scale
        rows.append(
            {
                "method_key": method,
                "method": label,
                "seed": seed,
                "normalized_hv": float(indicator.do(normalized)),
                "igd": igd(normalized, reference_norm),
                "normalized_sparsity": sparsity(normalized),
                "front_points": len(front),
            }
        )
    return pd.DataFrame(rows)


def paired_tests(detail):
    full = detail[detail.method_key == METHODS[0][0]].set_index("seed")
    rows = []
    for method, label in METHODS[1:]:
        other = detail[detail.method_key == method].set_index("seed")
        for metric, alternative in (
            ("normalized_hv", "greater"),
            ("igd", "less"),
            ("normalized_sparsity", "less"),
        ):
            first = full.loc[SEEDS, metric].to_numpy()
            second = other.loc[SEEDS, metric].to_numpy()
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


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    detail = calculate()
    summary = (
        detail.groupby(["method_key", "method"], as_index=False)
        .agg(
            hv_mean=("normalized_hv", "mean"),
            hv_std=("normalized_hv", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            sparsity_mean=("normalized_sparsity", "mean"),
            sparsity_std=("normalized_sparsity", "std"),
            front_points_mean=("front_points", "mean"),
            front_points_std=("front_points", "std"),
        )
        .sort_values("hv_mean", ascending=False)
    )
    tests = paired_tests(detail)
    detail.to_csv(RESULT_ROOT / "detail.csv", index=False)
    summary.to_csv(RESULT_ROOT / "summary.csv", index=False)
    tests.to_csv(RESULT_ROOT / "paired_tests.csv", index=False)

    order = [key for key, _ in METHODS]
    frame = summary.set_index("method_key").loc[order]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), constrained_layout=True)
    settings = [
        ("hv_mean", "hv_std", "Normalized HV"),
        ("igd_mean", "igd_std", "IGD"),
        ("sparsity_mean", "sparsity_std", "Normalized sparsity"),
        ("front_points_mean", "front_points_std", "Pareto points"),
    ]
    x = np.arange(len(frame))
    for ax, (mean, std, ylabel) in zip(axes.ravel(), settings):
        errors = frame[std].fillna(0.0) if std else None
        ax.bar(x, frame[mean], yerr=errors, color=COLORS, edgecolor="black", linewidth=0.5, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(frame["method"], rotation=22, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(FIGURE_ROOT / "all_evolving_selection_ablation.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))
    print("\nPaired tests:\n" + tests.to_string(index=False))


if __name__ == "__main__":
    main()
