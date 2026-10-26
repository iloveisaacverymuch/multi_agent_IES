import json
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
RESULT_ROOT = PROJECT_ROOT / "outputs" / "main_vs_maddpg_10seed"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
METHODS = [
    ("all_evolving_kappa0", "All-task Mean-HV E-MADDPG"),
    ("maddpg", "Fixed-weight MADDPG"),
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


def bootstrap_ci(values, rng, repetitions=20000):
    values = np.asarray(values, dtype=np.float64)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    loaded = []
    for method, label in METHODS:
        for seed in SEEDS:
            path = RUN_ROOT / f"strict_v3_{method}_seed_{seed}" / "strict_test" / "objs.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            loaded.append((method, label, seed, np.atleast_2d(np.loadtxt(path, delimiter=","))))

    reference = unique_front(np.vstack([item[3] for item in loaded]))
    all_fronts = np.vstack([unique_front(item[3]) for item in loaded])
    ideal = reference.min(axis=0)
    nadir = all_fronts.max(axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
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
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["method_key", "method"], as_index=False)
        .agg(
            hv_mean=("normalized_hv", "mean"),
            hv_std=("normalized_hv", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            sparsity_mean=("normalized_sparsity", "mean"),
            front_points_mean=("front_points", "mean"),
        )
        .sort_values("hv_mean", ascending=False)
    )

    main_values = detail[detail.method_key == METHODS[0][0]].set_index("seed")
    baseline = detail[detail.method_key == METHODS[1][0]].set_index("seed")
    rng = np.random.default_rng(20261010)
    tests = []
    for metric, alternative in (("normalized_hv", "greater"), ("igd", "less")):
        first = main_values.loc[SEEDS, metric].to_numpy()
        second = baseline.loc[SEEDS, metric].to_numpy()
        oriented = first - second if alternative == "greater" else second - first
        p_value = float(wilcoxon(first, second, alternative=alternative).pvalue)
        ci_low, ci_high = bootstrap_ci(oriented, rng)
        tests.append(
            {
                "metric": metric,
                "wins": int(np.sum(oriented > 0)),
                "ties": int(np.sum(oriented == 0)),
                "losses": int(np.sum(oriented < 0)),
                "mean_improvement": float(oriented.mean()),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "one_sided_wilcoxon_p": p_value,
                "cohen_dz": float(oriented.mean() / oriented.std(ddof=1)),
            }
        )
    tests = pd.DataFrame(tests)
    detail.to_csv(RESULT_ROOT / "detail.csv", index=False)
    summary.to_csv(RESULT_ROOT / "summary.csv", index=False)
    tests.to_csv(RESULT_ROOT / "paired_tests.csv", index=False)
    (RESULT_ROOT / "metric_bounds.json").write_text(
        json.dumps({"ideal": ideal.tolist(), "nadir": nadir.tolist()}, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    colors = ["#C44E52", "#F28E2B"]
    settings = [("hv_mean", "hv_std", "Normalized HV"), ("igd_mean", "igd_std", "IGD")]
    for ax, (mean, std, label) in zip(axes, settings):
        ax.bar(
            np.arange(2),
            summary.set_index("method_key").loc[[item[0] for item in METHODS], mean],
            yerr=summary.set_index("method_key").loc[[item[0] for item in METHODS], std],
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels([item[1] for item in METHODS], rotation=15, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(FIGURE_ROOT / "main_vs_maddpg_10seed.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))
    print("\nPaired tests:\n" + tests.to_string(index=False))


if __name__ == "__main__":
    main()
