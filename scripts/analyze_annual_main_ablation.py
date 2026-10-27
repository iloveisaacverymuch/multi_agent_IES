import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_all_evolving_cross_algorithm import (  # noqa: E402
    igd,
    sparsity,
    unique_front,
)
from scripts.evaluate_annual_main_ablation import RUN_ROOT, SEEDS, VARIANTS, run_dir  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "annual_main_ablation"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"
BOUNDS_PATH = PROJECT_ROOT / "outputs" / "annual_cross_algorithm" / "metric_bounds.json"
LABELS = {
    "A0_full": "CDV3-MHV-VR-EMADDPG",
    "A1_standard_cda": "SCDA-MHV-VR-EMADDPG",
    "A2_fixed_maddpg": "CDV3 fixed-weight MADDPG",
    "A3_random_preference": "CDV3-RP-VR-EMADDPG",
    "A4_local_hv": "CDV3 local-HV-VR-EMADDPG",
    "A5_scalar_stale": "CDV3-MHV-SR-EMADDPG",
}
LABELS_ZH = {
    "A0_full": "CDV3-MHV-VR-EMADDPG（主方法）",
    "A1_standard_cda": "SCDA-MHV-VR-EMADDPG",
    "A2_fixed_maddpg": "A2 移除偏好进化",
    "A3_random_preference": "CDV3-RP-VR-EMADDPG",
    "A4_local_hv": "A4 局部任务Mean-HV",
    "A5_scalar_stale": "CDV3-MHV-SR-EMADDPG",
}
COLORS = {
    "A0_full": "#D62728",
    "A1_standard_cda": "#1F77B4",
    "A2_fixed_maddpg": "#2CA02C",
    "A3_random_preference": "#FF7F0E",
    "A4_local_hv": "#9467BD",
    "A5_scalar_stale": "#17BECF",
}
MARKERS = {
    "A0_full": "o",
    "A1_standard_cda": "s",
    "A2_fixed_maddpg": "^",
    "A3_random_preference": "D",
    "A4_local_hv": "P",
    "A5_scalar_stale": "X",
}
FORMAL_VARIANTS = {
    key: VARIANTS[key]
    for key in ("A0_full", "A1_standard_cda", "A3_random_preference", "A5_scalar_stale")
}


def load_data():
    loaded = []
    for variant in FORMAL_VARIANTS:
        for seed in SEEDS:
            path = run_dir(variant, seed) / "annual_test" / "objs.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            objectives = np.atleast_2d(np.loadtxt(path, delimiter=","))
            if objectives.shape[1] != 2 or not np.isfinite(objectives).all():
                raise RuntimeError(f"Invalid annual objectives: {path}")
            loaded.append((variant, seed, objectives))
    return loaded


def metric_bounds():
    values = json.loads(BOUNDS_PATH.read_text(encoding="utf-8"))
    return (
        np.asarray(values["ideal"], dtype=np.float64),
        np.asarray(values["nadir"], dtype=np.float64),
        np.asarray(values["physical_hv_reference"], dtype=np.float64),
    )


def calculate(loaded):
    ideal, nadir, reference_point = metric_bounds()
    scale = np.maximum(nadir - ideal, 1e-12)
    pooled_reference = unique_front(np.vstack([row[2] for row in loaded]))
    pooled_reference_norm = (pooled_reference - ideal) / scale
    indicator = Hypervolume(ref_point=reference_point)
    rows = []
    pooled_rows = []
    front_rows = []
    for variant, seed, objectives in loaded:
        front = unique_front(objectives)
        normalized = (front - ideal) / scale
        rows.append(
            {
                "variant": variant,
                "method": LABELS_ZH[variant],
                "seed": seed,
                "policies": len(objectives),
                "front_points": len(front),
                "raw_hypervolume": float(indicator.do(front)),
                "normalized_igd": igd(normalized, pooled_reference_norm),
                "normalized_sparsity": sparsity(normalized),
                "min_annual_cost": float(front[:, 0].min()),
                "min_annual_carbon": float(front[:, 1].min()),
            }
        )
    detail = pd.DataFrame(rows)

    for variant in FORMAL_VARIANTS:
        objectives = np.vstack([values for key, _, values in loaded if key == variant])
        front = unique_front(objectives)
        normalized = (front - ideal) / scale
        seed_frame = detail[detail.variant == variant]
        pooled_rows.append(
            {
                "variant": variant,
                "method": LABELS_ZH[variant],
                "seeds": len(seed_frame),
                "front_points": len(front),
                "raw_hypervolume": float(indicator.do(front)),
                "normalized_igd": igd(normalized, pooled_reference_norm),
                "normalized_sparsity": sparsity(normalized),
                "min_annual_cost": float(front[:, 0].min()),
                "max_annual_cost": float(front[:, 0].max()),
                "mean_annual_cost": float(front[:, 0].mean()),
                "min_annual_carbon": float(front[:, 1].min()),
                "max_annual_carbon": float(front[:, 1].max()),
                "mean_annual_carbon": float(front[:, 1].mean()),
                "hv_reference_cost": float(reference_point[0]),
                "hv_reference_carbon": float(reference_point[1]),
            }
        )
        for point_id, (cost, carbon) in enumerate(front):
            front_rows.append(
                {
                    "variant": variant,
                    "method": LABELS_ZH[variant],
                    "pareto_point": point_id,
                    "annual_cost_CNY": cost,
                    "annual_carbon_kgCO2": carbon,
                }
            )
    return detail, pd.DataFrame(pooled_rows), pd.DataFrame(front_rows), reference_point


def summarize(detail):
    return detail.groupby(["variant", "method"], as_index=False).agg(
        hv_mean=("raw_hypervolume", "mean"),
        hv_std=("raw_hypervolume", "std"),
        igd_mean=("normalized_igd", "mean"),
        igd_std=("normalized_igd", "std"),
        sparsity_mean=("normalized_sparsity", "mean"),
        sparsity_std=("normalized_sparsity", "std"),
        front_points_mean=("front_points", "mean"),
        front_points_std=("front_points", "std"),
        min_cost_mean=("min_annual_cost", "mean"),
        min_cost_std=("min_annual_cost", "std"),
        min_carbon_mean=("min_annual_carbon", "mean"),
        min_carbon_std=("min_annual_carbon", "std"),
    )


def paired_tests(detail):
    main = detail[detail.variant == "A0_full"].set_index("seed")
    rows = []
    metrics = (
        ("raw_hypervolume", "greater"),
        ("normalized_igd", "less"),
        ("normalized_sparsity", "less"),
        ("min_annual_cost", "less"),
        ("min_annual_carbon", "less"),
    )
    for variant in list(FORMAL_VARIANTS)[1:]:
        other = detail[detail.variant == variant].set_index("seed")
        for metric, alternative in metrics:
            first = main.loc[SEEDS, metric].to_numpy()
            second = other.loc[SEEDS, metric].to_numpy()
            improvement = first - second if alternative == "greater" else second - first
            p_value = 1.0 if np.allclose(first, second) else float(
                wilcoxon(first, second, alternative=alternative).pvalue
            )
            rows.append(
                {
                    "ablation": variant,
                    "ablation_method": LABELS_ZH[variant],
                    "metric": metric,
                    "main_wins": int(np.sum(improvement > 0)),
                    "main_losses": int(np.sum(improvement < 0)),
                    "mean_improvement": float(improvement.mean()),
                    "one_sided_wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def plot_fronts(fronts):
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "Times New Roman"],
            "font.size": 12,
            "axes.labelsize": 17,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 11,
        }
    )
    all_cost = fronts.annual_cost_CNY.to_numpy(dtype=float)
    all_carbon = fronts.annual_carbon_kgCO2.to_numpy(dtype=float)
    cost_pad = 0.05 * (all_cost.max() - all_cost.min())
    carbon_pad = 0.06 * (all_carbon.max() - all_carbon.min())
    x_limits = (all_cost.min() - cost_pad, all_cost.max() + cost_pad)
    y_limits = (
        max(0.0, all_carbon.min() - carbon_pad),
        all_carbon.max() + carbon_pad,
    )
    comparisons = {
        "A1_standard_cda": "annual_main_ablation_pareto_A1.png",
        "A3_random_preference": "annual_main_ablation_pareto_A3.png",
        "A5_scalar_stale": "annual_main_ablation_pareto_A5.png",
    }
    for ablation, filename in comparisons.items():
        fig, ax = plt.subplots(figsize=(8.2, 6.2))
        for variant in ("A0_full", ablation):
            frame = fronts[fronts.variant == variant]
            ax.scatter(
                frame.annual_cost_CNY,
                frame.annual_carbon_kgCO2,
                s=34 if variant == "A0_full" else 58,
                marker="o",
                facecolors=COLORS[variant],
                edgecolors="black",
                linewidths=0.5,
                alpha=0.88 if variant == "A0_full" else 0.7,
                label=LABELS[variant],
                zorder=10 if variant == "A0_full" else 3,
            )
        ax.set_xlabel("全年运行成本（CNY）", fontweight="bold")
        ax.set_ylabel("全年碳排放量（kg CO$_2$）", fontweight="bold")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.grid(True, linestyle="--", color="#c9c9c9", linewidth=1.2)
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="black")
        fig.tight_layout()
        fig.savefig(FIGURE_ROOT / filename, dpi=600, bbox_inches="tight")
        plt.close(fig)


def plot_metrics(summary):
    order = list(FORMAL_VARIANTS)
    frame = summary.set_index("variant").loc[order]
    specs = [
        ("hv_mean", "hv_std", "全年原始超体积"),
        ("sparsity_mean", "sparsity_std", "归一化稀疏度"),
        ("min_cost_mean", "min_cost_std", "最低全年运行成本（CNY）"),
        ("min_carbon_mean", "min_carbon_std", "最低全年碳排放量（kg CO$_2$）"),
    ]
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "Times New Roman"]})
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), constrained_layout=True)
    x = np.arange(len(order))
    for ax, (mean, std, ylabel) in zip(axes.ravel(), specs):
        ax.bar(
            x,
            frame[mean],
            yerr=frame[std].fillna(0.0),
            color=[COLORS[key] for key in order],
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [LABELS[key].replace("-EMADDPG", "-\nEMADDPG") for key in order],
            rotation=12,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    fig.savefig(FIGURE_ROOT / "annual_main_ablation_metrics.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_tables(detail, summary, pooled, fronts, tests, reference_point):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(OUTPUT_ROOT / "年度随机种子明细.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_ROOT / "年度五随机种子汇总.csv", index=False, encoding="utf-8-sig")
    pooled.to_csv(OUTPUT_ROOT / "年度合并前沿指标.csv", index=False, encoding="utf-8-sig")
    fronts.to_csv(OUTPUT_ROOT / "全年帕累托前沿数据.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(OUTPUT_ROOT / "年度配对显著性检验.csv", index=False, encoding="utf-8-sig")
    protocol = {
        "evaluation_hours": 8760,
        "seeds": SEEDS,
        "variants": {
            key: {"label": LABELS_ZH[key], "run_pattern": value[0]}
            for key, value in FORMAL_VARIANTS.items()
        },
        "physical_hv_reference": reference_point.tolist(),
        "reference_source": str(BOUNDS_PATH.relative_to(PROJECT_ROOT)),
        "sparsity": "Mean squared Euclidean spacing on cost/carbon values normalized by the preset annual bounds; lower is denser.",
    }
    (OUTPUT_ROOT / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    loaded = load_data()
    detail, pooled, fronts, reference_point = calculate(loaded)
    summary = summarize(detail)
    tests = paired_tests(detail)
    write_tables(detail, summary, pooled, fronts, tests, reference_point)
    plot_fronts(fronts)
    plot_metrics(summary)
    print(summary.to_string(index=False))
    print("\nPooled fronts:\n" + pooled.to_string(index=False))


if __name__ == "__main__":
    main()
