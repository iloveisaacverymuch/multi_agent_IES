from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume

from pareto_style import PARETO_COLORS, apply_pareto_rcparams, scatter_pareto, style_pareto_axis


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
RUNS = (
    (
        "standard",
        "Standard CDA",
        RUN_ROOT / "standard_cda_hv_ensemble_extreme_capped32_kappa_0_10",
        np.nan,
    ),
    (
        "v2",
        "Carbon CDA V2",
        RUN_ROOT / "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02",
        0.02,
    ),
    *(
        (
            f"v3_{str(price).replace('.', '_')}",
            f"Carbon-Dividend V3 ({price:.2f})",
            RUN_ROOT
            / (
                "carbon_dividend_v3_hv_ensemble_extreme_capped32_kappa_0_10_"
                f"price_{price:.2f}".replace(".", "_")
            ),
            price,
        )
        for price in (0.01, 0.02, 0.05)
    ),
)


def load_objectives(run_dir):
    path = run_dir / "final" / "objs.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    objectives = np.atleast_2d(np.loadtxt(path, delimiter=","))
    if objectives.shape[1] != 2 or not np.isfinite(objectives).all():
        raise ValueError(f"Invalid objectives in {path}: shape={objectives.shape}")
    return objectives


def unique_front(objectives, decimals=2):
    order = np.lexsort((objectives[:, 1], objectives[:, 0]))
    sorted_points = objectives[order]
    keep = []
    best_carbon = np.inf
    for index, point in enumerate(sorted_points):
        if point[1] < best_carbon:
            keep.append(index)
            best_carbon = point[1]
    front = sorted_points[keep]
    rounded = np.round(front, decimals=decimals)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return front[np.sort(indices)]


def igd(front, reference):
    distances = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def sparsity(front):
    if len(front) < 2:
        return 0.0
    differences = np.diff(front[np.argsort(front[:, 0])], axis=0)
    return float(np.mean(np.sum(differences**2, axis=1)))


def elapsed_seconds(run_dir):
    path = run_dir / "elapsed_seconds.txt"
    if not path.exists():
        return np.nan
    return float(path.read_text(encoding="utf-8").strip())


def validate_final(run_dir, objective_count):
    final_dir = run_dir / "final"
    policy_count = len(list(final_dir.glob("EP_policy_*.pt")))
    metadata_path = final_dir / "metadata.csv"
    metadata_count = len(pd.read_csv(metadata_path)) if metadata_path.exists() else 0
    if policy_count != objective_count or metadata_count != objective_count:
        raise RuntimeError(
            f"Incomplete final output in {run_dir}: objectives={objective_count}, "
            f"policies={policy_count}, metadata={metadata_count}"
        )
    return policy_count, metadata_count


def plot_metrics(summary):
    v3 = summary[summary["method_key"].str.startswith("v3_")].sort_values("carbon_price")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 600,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4), constrained_layout=True)
    specs = (
        ("normalized_hv", "Normalized HV", "Higher is better"),
        ("igd", "IGD", "Lower is better"),
        ("normalized_sparsity", "Normalized sparsity", "Lower is better"),
    )
    for axis, (column, ylabel, title) in zip(axes, specs):
        axis.plot(v3["carbon_price"], v3[column], marker="o", linewidth=1.6, color="#1f77b4")
        axis.set_xlabel("Carbon price (CNY/kg CO$_2$)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    path = OUTPUT_DIR / "v3_price_pilot_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fronts(fronts):
    apply_pareto_rcparams()
    fig, axis = plt.subplots(figsize=(7.6, 5.5))
    handles = []
    for index, (key, label, _, _) in enumerate(RUNS):
        front = fronts[key]
        handles.append(
            scatter_pareto(
                axis,
                front[:, 0],
                front[:, 1],
                label,
                PARETO_COLORS[index % len(PARETO_COLORS)],
            )
        )
    combined = np.vstack(list(fronts.values()))
    style_pareto_axis(axis, combined[:, 0], combined[:, 1], handles)
    path = OUTPUT_DIR / "v3_price_pilot_pareto.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    loaded = [(key, label, run_dir, price, load_objectives(run_dir)) for key, label, run_dir, price in RUNS]
    all_objectives = np.vstack([item[4] for item in loaded])
    minimum = all_objectives.min(axis=0)
    ranges = np.maximum(all_objectives.max(axis=0) - minimum, 1e-12)
    reference = (unique_front(all_objectives) - minimum) / ranges

    rows = []
    fronts = {}
    for key, label, run_dir, price, objectives in loaded:
        policy_count, metadata_count = validate_final(run_dir, len(objectives))
        front = unique_front(objectives)
        normalized = (front - minimum) / ranges
        fronts[key] = front
        rows.append(
            {
                "method_key": key,
                "method": label,
                "carbon_price": price,
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "objective_points": len(objectives),
                "front_points": len(front),
                "policy_count": policy_count,
                "metadata_count": metadata_count,
                "normalized_hv": float(Hypervolume(ref_point=np.array([1.1, 1.1])).do(normalized)),
                "igd": igd(normalized, reference),
                "normalized_sparsity": sparsity(normalized),
                "min_cost": float(objectives[:, 0].min()),
                "min_carbon": float(objectives[:, 1].min()),
                "cost_span": float(front[:, 0].max() - front[:, 0].min()),
                "carbon_span": float(front[:, 1].max() - front[:, 1].min()),
                "elapsed_hours": elapsed_seconds(run_dir) / 3600.0,
            }
        )

    summary = pd.DataFrame(rows)
    standard = summary.loc[summary["method_key"] == "standard"].iloc[0]
    summary["hv_change_vs_standard_percent"] = 100.0 * (
        summary["normalized_hv"] / standard["normalized_hv"] - 1.0
    )
    summary["igd_reduction_vs_standard_percent"] = 100.0 * (1.0 - summary["igd"] / standard["igd"])
    summary["min_cost_change_vs_standard_percent"] = 100.0 * (
        summary["min_cost"] / standard["min_cost"] - 1.0
    )
    summary["min_carbon_change_vs_standard_percent"] = 100.0 * (
        summary["min_carbon"] / standard["min_carbon"] - 1.0
    )

    csv_path = OUTPUT_DIR / "v3_price_pilot_summary.csv"
    summary.to_csv(csv_path, index=False)
    metrics_path = plot_metrics(summary)
    pareto_path = plot_fronts(fronts)
    print(summary.to_string(index=False))
    print(f"Saved: {csv_path}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {pareto_path}")


if __name__ == "__main__":
    main()
