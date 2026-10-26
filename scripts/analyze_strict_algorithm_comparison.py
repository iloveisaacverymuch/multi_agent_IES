import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from scipy.stats import rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
PROTOCOL_DIR = PROJECT_ROOT / "outputs" / "strict_algorithm_comparison"
SEEDS = [11, 22, 33, 44, 55]
METHODS = [
    ("proposed", "Proposed V3 + HV-Ensemble"),
    ("fixed_weight_mappo", "Fixed-weight MAPPO"),
    ("masac", "MASAC (matched outer loop)"),
    ("maddpg", "MADDPG"),
    ("mopso", "MOPSO"),
]
COLORS = {
    "proposed": "#C44E52",
    "fixed_weight_mappo": "#4C78A8",
    "masac": "#59A14F",
    "maddpg": "#F28E2B",
    "mopso": "#8E6C8A",
}
MARKERS = {
    "proposed": "o",
    "fixed_weight_mappo": "s",
    "masac": "^",
    "maddpg": "D",
    "mopso": "P",
}


def run_dir(method, seed):
    return RUN_ROOT / f"strict_v3_{method}_seed_{seed}"


def pareto_indices(objectives):
    dominated = np.zeros(len(objectives), dtype=bool)
    for i in range(len(objectives)):
        dominated[i] = np.any(
            np.all(objectives <= objectives[i], axis=1)
            & np.any(objectives < objectives[i], axis=1)
        )
    return np.where(~dominated)[0]


def unique_front(objectives):
    front = objectives[pareto_indices(objectives)]
    rounded = np.round(front, decimals=6)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    front = front[np.sort(indices)]
    return front[np.argsort(front[:, 0])]


def igd(front, reference):
    distances = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def spacing(front):
    if len(front) < 2:
        return 0.0
    distances = np.linalg.norm(front[:, None, :] - front[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(np.sqrt(np.mean((nearest - nearest.mean()) ** 2)))


def sparsity(front):
    if len(front) < 2:
        return 0.0
    differences = np.diff(front[np.argsort(front[:, 0])], axis=0)
    return float(np.mean(np.sum(differences**2, axis=1)))


def rank_biserial(oriented_difference):
    values = oriented_difference[np.abs(oriented_difference) > 1e-15]
    if len(values) == 0:
        return 0.0
    ranks = rankdata(np.abs(values))
    positive = ranks[values > 0].sum()
    negative = ranks[values < 0].sum()
    return float((positive - negative) / (positive + negative))


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def load_runs():
    loaded = []
    for method_key, method in METHODS:
        for seed in SEEDS:
            path = run_dir(method_key, seed)
            objective_path = path / "strict_test" / "objs.txt"
            if not objective_path.exists():
                raise FileNotFoundError(f"Missing strict test result: {objective_path}")
            objectives = np.atleast_2d(np.loadtxt(objective_path, delimiter=","))
            loaded.append((method_key, method, seed, path, objectives))
    return loaded


def compute_metrics(loaded):
    all_objectives = np.vstack([item[4] for item in loaded])
    reference_front = unique_front(all_objectives)
    all_run_fronts = np.vstack([unique_front(item[4]) for item in loaded])
    ideal = reference_front.min(axis=0)
    nadir = all_run_fronts.max(axis=0)
    ranges = np.maximum(nadir - ideal, 1e-12)
    reference_norm = (reference_front - ideal) / ranges
    hv = Hypervolume(ref_point=np.array([1.1, 1.1]))
    rows = []
    for method_key, method, seed, path, objectives in loaded:
        front = unique_front(objectives)
        front_norm = (front - ideal) / ranges
        elapsed_path = path / "launcher_elapsed_seconds.txt"
        elapsed = float(elapsed_path.read_text().strip()) if elapsed_path.exists() else np.nan
        rows.append(
            {
                "method_key": method_key,
                "method": method,
                "seed": seed,
                "run_dir": str(path.relative_to(PROJECT_ROOT)),
                "archive_points": len(objectives),
                "test_front_points": len(front),
                "normalized_hv": float(hv.do(front_norm)),
                "igd": igd(front_norm, reference_norm),
                "spacing": spacing(front_norm),
                "normalized_sparsity": sparsity(front_norm),
                "min_test_cost": float(front[:, 0].min()),
                "min_test_carbon": float(front[:, 1].min()),
                "elapsed_hours": elapsed / 3600.0,
            }
        )
    bounds = {
        "normalization_source": (
            "ideal from pooled nondominated strict test front; nadir from all "
            "per-run nondominated strict test fronts"
        ),
        "ideal": ideal.tolist(),
        "nadir": nadir.tolist(),
        "normalized_hv_reference_point": [1.1, 1.1],
        "reference_front_points": len(reference_front),
    }
    (PROTOCOL_DIR / "metric_protocol.json").write_text(
        json.dumps(bounds, indent=2), encoding="utf-8"
    )
    return pd.DataFrame(rows), ideal, ranges


def summarize(detail):
    return (
        detail.groupby(["method_key", "method"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            hv_mean=("normalized_hv", "mean"),
            hv_std=("normalized_hv", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            sparsity_mean=("normalized_sparsity", "mean"),
            sparsity_std=("normalized_sparsity", "std"),
            front_points_mean=("test_front_points", "mean"),
            min_cost_mean=("min_test_cost", "mean"),
            min_carbon_mean=("min_test_carbon", "mean"),
            elapsed_hours_mean=("elapsed_hours", "mean"),
        )
        .sort_values("hv_mean", ascending=False)
    )


def statistical_tests(detail):
    rows = []
    metrics = [
        ("normalized_hv", True),
        ("igd", False),
        ("normalized_sparsity", False),
    ]
    proposed = detail[detail.method_key == "proposed"].set_index("seed")
    for comparator, label in METHODS[1:]:
        baseline = detail[detail.method_key == comparator].set_index("seed")
        for metric, higher_is_better in metrics:
            proposed_values = proposed.loc[SEEDS, metric].to_numpy(dtype=float)
            baseline_values = baseline.loc[SEEDS, metric].to_numpy(dtype=float)
            oriented = (
                proposed_values - baseline_values
                if higher_is_better
                else baseline_values - proposed_values
            )
            if np.all(np.abs(oriented) <= 1e-15):
                p_two_sided = p_one_sided = 1.0
            else:
                p_two_sided = float(
                    wilcoxon(proposed_values, baseline_values, alternative="two-sided").pvalue
                )
                if higher_is_better:
                    p_one_sided = float(
                        wilcoxon(proposed_values, baseline_values, alternative="greater").pvalue
                    )
                else:
                    p_one_sided = float(
                        wilcoxon(proposed_values, baseline_values, alternative="less").pvalue
                    )
            rows.append(
                {
                    "reference": "proposed",
                    "comparator": comparator,
                    "comparator_label": label,
                    "metric": metric,
                    "wins": int(np.sum(oriented > 1e-15)),
                    "ties": int(np.sum(np.abs(oriented) <= 1e-15)),
                    "losses": int(np.sum(oriented < -1e-15)),
                    "mean_relative_improvement_percent": float(
                        np.mean(100.0 * oriented / np.maximum(np.abs(baseline_values), 1e-15))
                    ),
                    "rank_biserial": rank_biserial(oriented),
                    "p_two_sided": p_two_sided,
                    "p_one_sided": p_one_sided,
                }
            )
    results = pd.DataFrame(rows)
    results["p_one_sided_holm"] = np.nan
    for _, indices in results.groupby("metric").groups.items():
        results.loc[indices, "p_one_sided_holm"] = holm(
            results.loc[indices, "p_one_sided"].to_numpy()
        )
    results["significant_0_05"] = results.p_one_sided_holm < 0.05
    return results


def plot_metrics(summary):
    order = [key for key, _ in METHODS]
    frame = summary.set_index("method_key").reindex(order)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    settings = [
        ("hv_mean", "hv_std", "Normalized HV", True),
        ("igd_mean", "igd_std", "IGD", False),
        ("sparsity_mean", "sparsity_std", "Normalized sparsity", False),
    ]
    x = np.arange(len(frame))
    for ax, (mean_col, std_col, label, _) in zip(axes, settings):
        ax.bar(
            x,
            frame[mean_col],
            yerr=frame[std_col].fillna(0.0),
            color=[COLORS[key] for key in order],
            edgecolor="black",
            linewidth=0.6,
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in METHODS], rotation=25, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(OUTPUT_DIR / "strict_algorithm_comparison_metrics.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_fronts(loaded):
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for method_key, label in METHODS:
        objectives = np.vstack([item[4] for item in loaded if item[0] == method_key])
        front = unique_front(objectives)
        ax.plot(
            front[:, 0],
            front[:, 1],
            marker=MARKERS[method_key],
            markersize=4.0,
            linewidth=1.2,
            alpha=0.85,
            color=COLORS[method_key],
            label=label,
        )
    ax.set_xlabel("96-hour operating cost")
    ax.set_ylabel("96-hour carbon emissions")
    ax.grid(linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUTPUT_DIR / "strict_algorithm_comparison_pareto.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
    loaded = load_runs()
    detail, _, _ = compute_metrics(loaded)
    summary = summarize(detail)
    statistics = statistical_tests(detail)
    detail.to_csv(OUTPUT_DIR / "strict_algorithm_comparison_detail.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "strict_algorithm_comparison_summary.csv", index=False)
    statistics.to_csv(OUTPUT_DIR / "strict_algorithm_comparison_statistics.csv", index=False)
    plot_metrics(summary)
    plot_fronts(loaded)
    print(summary.to_string(index=False))
    print("\nPaired tests:\n", statistics.to_string(index=False))


if __name__ == "__main__":
    main()
