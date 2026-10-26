from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from scipy.stats import wilcoxon

from pareto_style import PARETO_COLORS, apply_pareto_rcparams, scatter_pareto, style_pareto_axis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "current_carbon_price_sensitivity"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"
PRICES = [0.0, 0.005, 0.01, 0.02, 0.05]
SEEDS = [11, 22, 44]


def price_code(price):
    return f"{price:.3f}".replace(".", "p")


def run_dir(price, seed):
    if abs(price - 0.01) <= 1e-12:
        return RUN_ROOT / f"strict_v3_all_evolving_kappa0_seed_{seed}"
    return RUN_ROOT / f"strict_v3_all_evolving_kappa0_price_{price_code(price)}_seed_{seed}"


def unique_front(objectives):
    values = np.atleast_2d(np.asarray(objectives, dtype=np.float64))
    dominated = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        dominated[index] = np.any(
            np.all(values <= value, axis=1) & np.any(values < value, axis=1)
        )
    front = values[~dominated]
    _, indices = np.unique(np.round(front, 6), axis=0, return_index=True)
    return front[np.sort(indices)][np.argsort(front[np.sort(indices), 0])]


def igd(front, reference):
    distances = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def sparsity(front):
    if len(front) < 2:
        return 0.0
    differences = np.diff(front[np.argsort(front[:, 0])], axis=0)
    return float(np.mean(np.sum(differences**2, axis=1)))


def paired_tests(detail):
    base = detail[np.isclose(detail.carbon_price, 0.01)].set_index("seed")
    rows = []
    for price in PRICES:
        if abs(price - 0.01) <= 1e-12:
            continue
        other = detail[np.isclose(detail.carbon_price, price)].set_index("seed")
        for metric, preference in (
            ("normalized_hv", "greater"),
            ("igd", "less"),
            ("normalized_sparsity", "less"),
        ):
            first = base.loc[SEEDS, metric].to_numpy()
            second = other.loc[SEEDS, metric].to_numpy()
            improvement = first - second if preference == "greater" else second - first
            p_value = 1.0 if np.allclose(first, second) else float(
                wilcoxon(first, second, alternative=preference).pvalue
            )
            rows.append(
                {
                    "comparison": f"0.010 vs {price:.3f}",
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
    for price in PRICES:
        for seed in SEEDS:
            path = run_dir(price, seed) / "strict_test" / "objs.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            loaded.append((price, seed, np.atleast_2d(np.loadtxt(path, delimiter=","))))
    pooled = np.vstack([values for _, _, values in loaded])
    reference_front = unique_front(pooled)
    all_fronts = np.vstack([unique_front(values) for _, _, values in loaded])
    ideal = reference_front.min(axis=0)
    scale = np.maximum(all_fronts.max(axis=0) - ideal, 1e-12)
    reference_norm = (reference_front - ideal) / scale
    indicator = Hypervolume(ref_point=np.array([1.1, 1.1]))
    rows = []
    seed44_fronts = []
    for price, seed, objectives in loaded:
        front = unique_front(objectives)
        normalized = (front - ideal) / scale
        rows.append(
            {
                "carbon_price": price,
                "seed": seed,
                "normalized_hv": float(indicator.do(normalized)),
                "igd": igd(normalized, reference_norm),
                "normalized_sparsity": sparsity(normalized),
                "front_points": len(front),
                "min_cost": float(front[:, 0].min()),
                "min_carbon": float(front[:, 1].min()),
            }
        )
        if seed == 44:
            seed44_fronts.append((price, front))
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("carbon_price", as_index=False)
        .agg(
            hv_mean=("normalized_hv", "mean"),
            hv_std=("normalized_hv", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            sparsity_mean=("normalized_sparsity", "mean"),
            sparsity_std=("normalized_sparsity", "std"),
            front_points_mean=("front_points", "mean"),
            front_points_std=("front_points", "std"),
            min_cost_mean=("min_cost", "mean"),
            min_carbon_mean=("min_carbon", "mean"),
        )
        .sort_values("carbon_price")
    )
    detail.to_csv(RESULT_ROOT / "detail.csv", index=False)
    summary.to_csv(RESULT_ROOT / "summary.csv", index=False)
    paired_tests(detail).to_csv(RESULT_ROOT / "paired_tests.csv", index=False)
    np.savetxt(RESULT_ROOT / "pooled_reference_front.txt", reference_front, delimiter=",")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )
    specs = [
        ("hv_mean", "hv_std", "Normalized HV"),
        ("igd_mean", "igd_std", "IGD"),
        ("sparsity_mean", "sparsity_std", "Normalized sparsity"),
        ("front_points_mean", "front_points_std", "Pareto points"),
        ("min_cost_mean", None, "Minimum cost"),
        ("min_carbon_mean", None, "Minimum carbon"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2), constrained_layout=True)
    x = summary["carbon_price"].to_numpy()
    for ax, (mean_col, std_col, ylabel) in zip(axes.ravel(), specs):
        yerr = summary[std_col].to_numpy() if std_col is not None else None
        ax.errorbar(x, summary[mean_col], yerr=yerr, marker="o", capsize=3, color="#C44E52")
        ax.axvline(0.01, color="#4C78A8", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Carbon price")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(FIGURE_ROOT / "current_main_carbon_price_sensitivity_metrics.png", bbox_inches="tight")
    plt.close(fig)

    apply_pareto_rcparams()
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    handles = []
    for index, (price, front) in enumerate(seed44_fronts):
        handles.append(
            scatter_pareto(
                ax,
                front[:, 0],
                front[:, 1],
                f"{price:.3f}",
                PARETO_COLORS[index % len(PARETO_COLORS)],
            )
        )
    values = np.vstack([front for _, front in seed44_fronts])
    style_pareto_axis(ax, values[:, 0], values[:, 1], handles, legend_title="Carbon price")
    fig.savefig(FIGURE_ROOT / "current_main_carbon_price_sensitivity_pareto.png", bbox_inches="tight")
    plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
