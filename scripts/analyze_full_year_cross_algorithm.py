import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume


PROJECT_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_PATH))

from scripts.analyze_all_evolving_cross_algorithm import (
    COLORS,
    METHOD_LABELS_ZH,
    METHODS,
    OPTIONAL_METHODS,
    PROJECT_ROOT,
    RUN_ROOT,
    SEEDS,
    igd,
    paired_tests,
    sparsity,
    unique_front,
)


RESULT_ROOT = PROJECT_ROOT / "outputs" / "annual_cross_algorithm"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"


def load_runs():
    methods = list(METHODS)
    for method, label in OPTIONAL_METHODS:
        paths = [
            RUN_ROOT / f"strict_v3_{method}_seed_{seed}" / "annual_test" / "objs.txt"
            for seed in SEEDS
        ]
        if all(path.exists() for path in paths):
            methods.append((method, label))
        elif any(path.exists() for path in paths):
            raise RuntimeError(f"Incomplete annual results for {method}")

    loaded = []
    for method, label in methods:
        for seed in SEEDS:
            path = RUN_ROOT / f"strict_v3_{method}_seed_{seed}" / "annual_test" / "objs.txt"
            if not path.exists():
                raise FileNotFoundError(path)
            objectives = np.atleast_2d(np.loadtxt(path, delimiter=","))
            if objectives.shape[1] != 2 or not np.isfinite(objectives).all():
                raise RuntimeError(f"Invalid annual objectives in {path}")
            loaded.append((method, label, seed, objectives))
    return loaded, methods


def calculate(loaded):
    pooled = np.vstack([item[3] for item in loaded])
    reference_front = unique_front(pooled)
    all_fronts = np.vstack([unique_front(item[3]) for item in loaded])
    ideal = reference_front.min(axis=0)
    nadir = all_fronts.max(axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
    reference_point = ideal + 1.1 * scale
    reference_norm = (reference_front - ideal) / scale
    hv = Hypervolume(ref_point=reference_point)

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
                "hypervolume": float(hv.do(front)),
                "igd": igd(normalized, reference_norm),
                "normalized_sparsity": sparsity(normalized),
                "min_cost": float(front[:, 0].min()),
                "min_carbon": float(front[:, 1].min()),
            }
        )
    return pd.DataFrame(rows), ideal, nadir, reference_point, reference_front


def summarize(detail):
    return (
        detail.groupby(["method_key", "method"], as_index=False)
        .agg(
            hv_mean=("hypervolume", "mean"),
            hv_std=("hypervolume", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            sparsity_mean=("normalized_sparsity", "mean"),
            sparsity_std=("normalized_sparsity", "std"),
            front_points_mean=("front_points", "mean"),
            front_points_std=("front_points", "std"),
            min_cost_mean=("min_cost", "mean"),
            min_carbon_mean=("min_carbon", "mean"),
        )
        .sort_values("hv_mean", ascending=False)
    )


def pooled_metrics_and_fronts(loaded, detail, ideal, nadir, reference_point, methods):
    scale = np.maximum(nadir - ideal, 1e-12)
    hv = Hypervolume(ref_point=reference_point)
    metric_rows = []
    front_rows = []
    for method, _ in methods:
        objectives = np.vstack([values for key, _, _, values in loaded if key == method])
        front = unique_front(objectives)
        normalized = (front - ideal) / scale
        seed_rows = detail[detail.method_key == method]
        metric_rows.append(
            {
                "方法标识": method,
                "方法": METHOD_LABELS_ZH[method],
                "随机种子数": len(seed_rows),
                "合并前沿帕累托点数": len(front),
                "合并前沿原始超体积_CNY乘kgCO2": float(hv.do(front)),
                "合并前沿归一化稀疏度": sparsity(normalized),
                "五种子原始超体积均值_CNY乘kgCO2": float(seed_rows.hypervolume.mean()),
                "五种子原始超体积标准差_CNY乘kgCO2": float(seed_rows.hypervolume.std()),
                "五种子归一化IGD均值": float(seed_rows.igd.mean()),
                "五种子归一化IGD标准差": float(seed_rows.igd.std()),
                "全年HV参考点_运行成本_CNY": float(reference_point[0]),
                "全年HV参考点_碳排放_kgCO2": float(reference_point[1]),
            }
        )
        for index, (cost, carbon) in enumerate(front):
            front_rows.append(
                {
                    "方法标识": method,
                    "方法": METHOD_LABELS_ZH[method],
                    "帕累托点序号": index,
                    "全年运行成本_CNY": cost,
                    "全年碳排放_kgCO2": carbon,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics["合并前沿超体积排名"] = (
        metrics["合并前沿原始超体积_CNY乘kgCO2"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    metrics["合并前沿稀疏度排名"] = (
        metrics["合并前沿归一化稀疏度"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    return metrics.sort_values("合并前沿超体积排名"), pd.DataFrame(front_rows)


def plot_fronts(loaded, methods):
    style = {
        "font.family": ["Microsoft YaHei", "Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 13,
        "figure.autolayout": True,
    }
    with plt.rc_context():
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(style)
        fig, ax = plt.subplots(figsize=(8.0, 6.0))
        all_x = []
        all_y = []
        for index, (method, label) in enumerate(methods):
            objectives = np.vstack([values for key, _, _, values in loaded if key == method])
            front = unique_front(objectives)
            all_x.append(front[:, 0])
            all_y.append(front[:, 1])
            ax.scatter(
                front[:, 0],
                front[:, 1],
                c=COLORS[method],
                marker="o",
                s=60,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.8,
                label=label,
                zorder=len(methods) - index + 3,
            )
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        ax.set_xlabel("全年运行成本（CNY）", fontweight="bold")
        ax.set_ylabel("全年碳排放量（kg CO$_2$）", fontweight="bold")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(16)
        ax.yaxis.get_offset_text().set_fontsize(16)
        ax.tick_params(axis="both", which="major", length=10, width=2)
        ax.grid(True, linestyle="--", color="#c9c9c9", linewidth=1.6)
        ax.set_xlim(float(all_x.min()) * 0.95, float(all_x.max()) * 1.05)
        ax.set_ylim(0.0, float(all_y.max()) * 1.08)
        legend = ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="black",
            borderpad=0.7,
            labelspacing=0.65,
            handletextpad=0.7,
        )
        legend.get_frame().set_linewidth(1.2)
        fig.savefig(
            FIGURE_ROOT / "annual_cross_algorithm_pareto.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)


def plot_metrics(summary, methods):
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "Times New Roman"]})
    order = [key for key, _ in methods]
    frame = summary.set_index("method_key").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4), constrained_layout=True)
    settings = [
        ("hv_mean", "hv_std", "全年超体积（CNY·kg CO$_2$）"),
        ("igd_mean", "igd_std", "反世代距离（IGD）"),
        ("front_points_mean", "front_points_std", "帕累托点数"),
    ]
    x = np.arange(len(frame))
    for ax, (mean, std, ylabel) in zip(axes, settings):
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
        labels = [
            value.replace("CDV3-MHV-VR-EMADDPG", "CDV3-MHV-VR-\nEMADDPG")
            .replace("Anchor-E-MADDPG", "Anchor-E-\nMADDPG")
            for value in frame["method"]
        ]
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(
        FIGURE_ROOT / "annual_cross_algorithm_metrics.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_chinese_tables(detail, summary, tests):
    detail_zh = detail.rename(
        columns={
            "method_key": "方法标识",
            "method": "方法",
            "seed": "随机种子",
            "archive_points": "全年测试策略数",
            "front_points": "全年帕累托点数",
            "hypervolume": "全年原始超体积_CNY乘kgCO2",
            "igd": "全年归一化IGD",
            "normalized_sparsity": "全年归一化稀疏度",
            "min_cost": "最低全年运行成本_CNY",
            "min_carbon": "最低全年碳排放_kgCO2",
        }
    )
    detail_zh["方法"] = detail_zh["方法标识"].map(METHOD_LABELS_ZH)
    detail_zh.to_csv(RESULT_ROOT / "年度随机种子明细.csv", index=False, encoding="utf-8-sig")

    summary_zh = summary.rename(
        columns={
            "method_key": "方法标识",
            "method": "方法",
            "hv_mean": "全年原始超体积均值_CNY乘kgCO2",
            "hv_std": "全年原始超体积标准差_CNY乘kgCO2",
            "igd_mean": "全年归一化IGD均值",
            "igd_std": "全年归一化IGD标准差",
            "sparsity_mean": "全年归一化稀疏度均值",
            "sparsity_std": "全年归一化稀疏度标准差",
            "front_points_mean": "全年帕累托点数均值",
            "front_points_std": "全年帕累托点数标准差",
            "min_cost_mean": "最低全年运行成本均值_CNY",
            "min_carbon_mean": "最低全年碳排放均值_kgCO2",
        }
    )
    summary_zh["方法"] = summary_zh["方法标识"].map(METHOD_LABELS_ZH)
    summary_zh.to_csv(RESULT_ROOT / "年度五随机种子汇总.csv", index=False, encoding="utf-8-sig")

    tests_zh = tests.rename(
        columns={
            "comparator": "对比方法",
            "metric": "指标",
            "wins": "主方法胜出种子数",
            "losses": "主方法落后种子数",
            "mean_improvement": "主方法平均改善量",
            "one_sided_p": "单侧Wilcoxon检验p值",
        }
    )
    tests_zh["指标"] = tests_zh["指标"].map(
        {"hypervolume": "全年原始超体积", "igd": "全年归一化IGD"}
    )
    tests_zh.to_csv(RESULT_ROOT / "年度配对显著性检验.csv", index=False, encoding="utf-8-sig")


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    loaded, methods = load_runs()
    detail, ideal, nadir, reference_point, reference_front = calculate(loaded)
    summary = summarize(detail)
    tests = paired_tests(detail, methods)
    pooled_metrics, fronts = pooled_metrics_and_fronts(
        loaded, detail, ideal, nadir, reference_point, methods
    )

    detail.to_csv(RESULT_ROOT / "detail.csv", index=False)
    summary.to_csv(RESULT_ROOT / "summary.csv", index=False)
    tests.to_csv(RESULT_ROOT / "paired_tests.csv", index=False)
    pooled_metrics.to_csv(
        RESULT_ROOT / "年度合并前沿指标.csv", index=False, encoding="utf-8-sig"
    )
    fronts.to_csv(
        RESULT_ROOT / "全年帕累托前沿数据.csv", index=False, encoding="utf-8-sig"
    )
    np.savetxt(
        RESULT_ROOT / "annual_reference_front.txt",
        reference_front,
        delimiter=",",
        fmt="%.6f",
    )
    (RESULT_ROOT / "metric_bounds.json").write_text(
        json.dumps(
            {
                "ideal": ideal.tolist(),
                "nadir": nadir.tolist(),
                "physical_hv_reference": reference_point.tolist(),
                "evaluation_hours": 8760,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_chinese_tables(detail, summary, tests)
    plot_fronts(loaded, methods)
    plot_metrics(summary, methods)
    print(summary.to_string(index=False))
    print("\nPooled annual metrics:\n" + pooled_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
