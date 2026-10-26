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
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULT_DIR = PROJECT_ROOT / "outputs" / "anchor_evolutionary_maddpg_experiment"
SEEDS = [11, 22, 33, 44, 55]
METHODS = [
    ("anchor_evolutionary_maddpg", "Anchor-guided E-MADDPG"),
    ("maddpg", "Fixed-weight MADDPG"),
    ("fixed_weight_mappo", "Fixed-weight MAPPO"),
    ("proposed", "Previous V3 + HV-Ensemble"),
    ("masac", "MASAC"),
    ("mopso", "MOPSO"),
]
COLORS = ["#C44E52", "#F28E2B", "#4C78A8", "#8172B3", "#59A14F", "#8C8C8C"]
MARKERS = ["o", "D", "s", "P", "^", "X"]


def unique_front(objectives):
    objectives = np.atleast_2d(np.asarray(objectives, dtype=np.float64))
    dominated = np.zeros(len(objectives), dtype=bool)
    for i in range(len(objectives)):
        dominated[i] = np.any(
            np.all(objectives <= objectives[i], axis=1)
            & np.any(objectives < objectives[i], axis=1)
        )
    front = objectives[~dominated]
    _, indices = np.unique(np.round(front, 6), axis=0, return_index=True)
    front = front[np.sort(indices)]
    return front[np.argsort(front[:, 0])]


def igd(front, reference):
    distance = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(np.mean(np.min(distance, axis=1)))


def sparsity(front):
    if len(front) < 2:
        return 0.0
    differences = np.diff(front[np.argsort(front[:, 0])], axis=0)
    return float(np.mean(np.sum(differences**2, axis=1)))


def load_data():
    rows = []
    for method, label in METHODS:
        for seed in SEEDS:
            path = RUN_ROOT / f"strict_v3_{method}_seed_{seed}" / "strict_test" / "objs.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            rows.append((method, label, seed, np.atleast_2d(np.loadtxt(path, delimiter=","))))
    return rows


def metrics(loaded):
    pooled_front = unique_front(np.vstack([values for _, _, _, values in loaded]))
    all_fronts = np.vstack([unique_front(values) for _, _, _, values in loaded])
    ideal = pooled_front.min(axis=0)
    nadir = all_fronts.max(axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
    reference = (pooled_front - ideal) / scale
    indicator = Hypervolume(ref_point=np.array([1.1, 1.1]))
    rows = []
    for method, label, seed, values in loaded:
        front = unique_front(values)
        normalized = (front - ideal) / scale
        rows.append(
            {
                "method_key": method,
                "method": label,
                "seed": seed,
                "archive_points": len(values),
                "front_points": len(front),
                "normalized_hv": float(indicator.do(normalized)),
                "igd": igd(normalized, reference),
                "normalized_sparsity": sparsity(normalized),
                "min_cost": float(front[:, 0].min()),
                "min_carbon": float(front[:, 1].min()),
            }
        )
    return pd.DataFrame(rows), ideal, scale, pooled_front


def paired_tests(detail):
    proposed = detail[detail.method_key == "anchor_evolutionary_maddpg"].set_index("seed")
    rows = []
    for method, label in METHODS[1:]:
        baseline = detail[detail.method_key == method].set_index("seed")
        for metric, alternative in (("normalized_hv", "greater"), ("igd", "less")):
            first = proposed.loc[SEEDS, metric].to_numpy()
            second = baseline.loc[SEEDS, metric].to_numpy()
            if np.allclose(first, second):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(first, second, alternative=alternative).pvalue)
            oriented = first - second if alternative == "greater" else second - first
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
    fig.savefig(OUTPUT_DIR / "anchor_evolutionary_maddpg_pareto.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(summary):
    frame = summary.set_index("method_key").loc[[item[0] for item in METHODS]]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    columns = [
        ("hv_mean", "hv_std", "Normalized HV"),
        ("igd_mean", "igd_std", "IGD"),
        ("sparsity_mean", "sparsity_std", "Normalized sparsity"),
    ]
    x = np.arange(len(frame))
    for ax, (mean, std, ylabel) in zip(axes, columns):
        ax.bar(x, frame[mean], yerr=frame[std].fillna(0), color=COLORS, edgecolor="black", linewidth=0.5, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(frame["method"], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(OUTPUT_DIR / "anchor_evolutionary_maddpg_metrics.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = load_data()
    detail, ideal, scale, reference = metrics(loaded)
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
            min_cost_mean=("min_cost", "mean"),
            min_carbon_mean=("min_carbon", "mean"),
        )
        .sort_values("hv_mean", ascending=False)
    )
    tests = paired_tests(detail)
    detail.to_csv(RESULT_DIR / "comparison_detail.csv", index=False)
    summary.to_csv(RESULT_DIR / "comparison_summary.csv", index=False)
    tests.to_csv(RESULT_DIR / "paired_tests.csv", index=False)
    np.savetxt(RESULT_DIR / "pooled_reference_front.txt", reference, delimiter=",", fmt="%.8f")
    (RESULT_DIR / "metric_bounds.json").write_text(
        pd.Series({"ideal": ideal.tolist(), "nadir": (ideal + scale).tolist()}).to_json(indent=2),
        encoding="utf-8",
    )
    plot_fronts(loaded)
    plot_metrics(summary)
    print(summary.to_string(index=False))
    print("\nPaired tests:\n" + tests.to_string(index=False))


if __name__ == "__main__":
    main()
