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
RESULT_ROOT = PROJECT_ROOT / "outputs" / "vector_replay_ablation"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"
SEEDS = [11, 22, 33, 44, 55]
METHODS = [
    ("vector_relabel", "Vector relabelling"),
    ("scalar_stale", "Stored scalar reward"),
    ("clear_on_change", "Clear on preference change"),
]


def run_dir(mode, seed):
    if mode == "vector_relabel":
        return RUN_ROOT / f"strict_v3_all_evolving_kappa0_seed_{seed}"
    return RUN_ROOT / f"strict_v3_replay_{mode}_seed_{seed}"


def unique_front(objectives):
    values = np.atleast_2d(np.asarray(objectives, dtype=np.float64))
    dominated = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        dominated[index] = np.any(
            np.all(values <= value, axis=1) & np.any(values < value, axis=1)
        )
    front = values[~dominated]
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


def paired_tests(detail):
    proposed = detail[detail.method_key == "vector_relabel"].set_index("seed")
    rows = []
    for method_key, label in METHODS[1:]:
        baseline = detail[detail.method_key == method_key].set_index("seed")
        for metric, alternative in (
            ("normalized_hv", "greater"),
            ("igd", "less"),
            ("normalized_sparsity", "less"),
        ):
            first = proposed.loc[SEEDS, metric].to_numpy()
            second = baseline.loc[SEEDS, metric].to_numpy()
            improvement = first - second if alternative == "greater" else second - first
            p_value = 1.0 if np.allclose(first, second) else float(
                wilcoxon(first, second, alternative=alternative).pvalue
            )
            rows.append(
                {
                    "comparator": label,
                    "metric": metric,
                    "wins": int(np.sum(improvement > 0)),
                    "losses": int(np.sum(improvement < 0)),
                    "mean_improvement": float(np.mean(improvement)),
                    "one_sided_wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    loaded = []
    for method_key, label in METHODS:
        for seed in SEEDS:
            path = run_dir(method_key, seed) / "strict_test" / "objs.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            loaded.append(
                (method_key, label, seed, np.atleast_2d(np.loadtxt(path, delimiter=",")))
            )
    pooled = np.vstack([values for _, _, _, values in loaded])
    reference_front = unique_front(pooled)
    all_fronts = np.vstack([unique_front(values) for _, _, _, values in loaded])
    ideal = reference_front.min(axis=0)
    scale = np.maximum(all_fronts.max(axis=0) - ideal, 1e-12)
    reference_norm = (reference_front - ideal) / scale
    indicator = Hypervolume(ref_point=np.array([1.1, 1.1]))
    rows = []
    for method_key, label, seed, objectives in loaded:
        front = unique_front(objectives)
        normalized = (front - ideal) / scale
        rows.append(
            {
                "method_key": method_key,
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
            sparsity_std=("normalized_sparsity", "std"),
            front_points_mean=("front_points", "mean"),
            front_points_std=("front_points", "std"),
        )
    )
    order = [key for key, _ in METHODS]
    summary["order"] = summary.method_key.map({key: i for i, key in enumerate(order)})
    summary = summary.sort_values("order").drop(columns="order")
    detail.to_csv(RESULT_ROOT / "detail.csv", index=False)
    summary.to_csv(RESULT_ROOT / "summary.csv", index=False)
    paired_tests(detail).to_csv(RESULT_ROOT / "paired_tests.csv", index=False)
    np.savetxt(RESULT_ROOT / "pooled_reference_front.txt", reference_front, delimiter=",")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )
    specs = [
        ("hv_mean", "hv_std", "Normalized HV"),
        ("igd_mean", "igd_std", "IGD"),
        ("sparsity_mean", "sparsity_std", "Normalized sparsity"),
        ("front_points_mean", "front_points_std", "Pareto points"),
    ]
    colors = ["#C44E52", "#4C78A8", "#59A14F"]
    labels = [label for _, label in METHODS]
    x = np.arange(len(METHODS))
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
    for ax, (mean_col, std_col, ylabel) in zip(axes.ravel(), specs):
        ax.bar(
            x,
            summary[mean_col],
            yerr=summary[std_col],
            capsize=3,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(FIGURE_ROOT / "vector_replay_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
