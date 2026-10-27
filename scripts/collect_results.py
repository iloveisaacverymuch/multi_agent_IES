import csv
import hashlib
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
RESULT_ROOT = PROJECT_ROOT / "results"
SKIP_NAMES = {"progress.json", "__pycache__", ".pytest_cache"}
COPIED = []


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source, relative_destination):
    source = Path(source)
    destination = RESULT_ROOT / relative_destination
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    COPIED.append((source, destination))


def copy_tree(source, relative_destination):
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        if (
            not path.is_file()
            or any(part in SKIP_NAMES for part in path.parts)
            or path.suffix == ".log"
            or path.name.endswith("_progress.json")
        ):
            continue
        copy_file(path, Path(relative_destination) / path.relative_to(source))


def copy_matching(source_dir, pattern, relative_destination):
    for path in sorted(Path(source_dir).glob(pattern)):
        if path.is_file():
            copy_file(path, Path(relative_destination) / path.name)


def write_readme():
    content = """# 硕士论文实验结果汇总

本目录汇总当前主方法 **Carbon-Dividend CDA V3 + 全任务 Mean-HV 进化 MADDPG** 的正式实验结果。原始训练策略、检查点和日志仍保存在 `outputs/runs`，这里仅保留论文使用的图表、统计结果、实验协议和可审计明细。

## 目录说明

1. `01_全年真实数据跨算法对比`：8种算法在连续8760小时全年数据上的帕累托前沿、五随机种子统计和显著性检验。这是当前最新、最完整的跨算法结果。
2. `02_四季典型日跨算法对比`：四个典型日共96小时的跨算法前沿与指标，作为典型日实验补充。
3. `03_主方法调度结果`：三种偏好、四个典型日、三个园区的电热冷调度图及逐时统计。
4. `04_园区电能交易结果`：三种偏好下各园区逐时购售电曲线、交易对象分解和交易明细。
5. `05_收敛性与多随机种子验证`：主方法与MADDPG的10随机种子比较、收敛曲线和AUC统计。
6. `06_消融实验`：主方法连续8760小时全年消融，以及CDA/进化机制、精英选择、向量经验回放和Carbon-Dividend V3机制消融。
7. `07_碳价敏感性分析`：不同碳价下的前沿、指标与配对检验。
8. `08_偏好与季节分解分析`：偏好与季节交互下的详细统计、效应量和热图。
9. `09_调度行为对比`：当前主方法相对MADDPG的调度行为差异。
10. `10_实验协议与配置`：严格跨算法协议、源代码哈希和当前主方法配置。

## 使用建议

- 论文的主要跨算法结论优先使用 `01_全年真实数据跨算法对比`。
- 典型日调度行为与园区交易讨论使用 `03` 和 `04`。
- 方法有效性论证使用 `05` 至 `08`。
- `结果文件清单.csv` 记录每个文件的来源、大小和SHA-256，可用于完整性核查。
- 图中的 `MADDPG`、`MAPPO`、`MATD3` 和 `IPPO` 是固定偏好训练的标准基线，图例按论文展示要求省略了 `Fixed-weight` 前缀。

## 关键结论

- 在连续8760小时全年数据上，当前主方法五随机种子平均原始HV和平均IGD均排名第一。
- 相对MATD3、IPPO、MAPPO、MASAC和MOPSO，主方法在5/5个种子上均取得更高HV和更低IGD，单侧Wilcoxon检验均为 `p=0.03125`。
- 相对锚点引导进化MADDPG和普通MADDPG，主方法平均指标仍为最优，但五随机种子统计尚未达到 `p<0.05`，应表述为稳定改善趋势。
"""
    path = RESULT_ROOT / "README.md"
    path.write_text(content, encoding="utf-8")


def write_manifest():
    rows = []
    for source, destination in COPIED:
        rows.append(
            {
                "结果文件": str(destination.relative_to(RESULT_ROOT)),
                "原始位置": str(source.relative_to(PROJECT_ROOT)),
                "文件大小_字节": destination.stat().st_size,
                "SHA256": sha256(destination),
            }
        )
    readme = RESULT_ROOT / "README.md"
    rows.append(
        {
            "结果文件": "README.md",
            "原始位置": "由 scripts/collect_results.py 生成",
            "文件大小_字节": readme.stat().st_size,
            "SHA256": sha256(readme),
        }
    )
    with (RESULT_ROOT / "结果文件清单.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["结果文件"]))


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    copy_tree(
        OUTPUT_ROOT / "annual_cross_algorithm",
        "01_全年真实数据跨算法对比/表格与说明",
    )
    copy_file(
        FIGURE_ROOT / "annual_cross_algorithm_pareto.png",
        "01_全年真实数据跨算法对比/全年帕累托前沿.png",
    )
    copy_file(
        FIGURE_ROOT / "annual_cross_algorithm_metrics.png",
        "01_全年真实数据跨算法对比/全年算法指标对比.png",
    )

    copy_tree(
        OUTPUT_ROOT / "all_evolving_cross_algorithm",
        "02_四季典型日跨算法对比/表格与说明",
    )
    copy_file(
        FIGURE_ROOT / "all_evolving_cross_algorithm_pareto.png",
        "02_四季典型日跨算法对比/典型日帕累托前沿.png",
    )
    copy_file(
        FIGURE_ROOT / "all_evolving_cross_algorithm_metrics.png",
        "02_四季典型日跨算法对比/典型日算法指标对比.png",
    )

    copy_tree(
        FIGURE_ROOT / "dispatch_results_main_method",
        "03_主方法调度结果/调度图",
    )
    copy_tree(
        OUTPUT_ROOT / "main_method_agent_statistics",
        "03_主方法调度结果/统计表",
    )
    copy_tree(
        FIGURE_ROOT / "peer_trade_profiles_main_method",
        "04_园区电能交易结果",
    )

    copy_tree(
        OUTPUT_ROOT / "main_convergence_10seed",
        "05_收敛性与多随机种子验证/主方法收敛10种子",
    )
    copy_tree(
        OUTPUT_ROOT / "main_vs_maddpg_10seed",
        "05_收敛性与多随机种子验证/主方法与MADDPG对比10种子",
    )
    copy_file(
        FIGURE_ROOT / "main_vs_maddpg_10seed.png",
        "05_收敛性与多随机种子验证/主方法与MADDPG对比.png",
    )
    copy_file(
        FIGURE_ROOT / "main_vs_maddpg_convergence_10seed.png",
        "05_收敛性与多随机种子验证/主方法与MADDPG收敛曲线.png",
    )

    for source_name, destination_name in (
        ("all_evolving_selection_ablation", "精英选择消融"),
        ("cda_algorithm_factorial", "CDA与进化机制析因消融"),
        ("vector_replay_ablation", "向量经验回放消融"),
    ):
        copy_tree(
            OUTPUT_ROOT / source_name,
            Path("06_消融实验") / destination_name / "表格与协议",
        )
    for filename in (
        "all_evolving_selection_ablation.png",
        "cda_algorithm_factorial.png",
        "vector_replay_ablation.png",
    ):
        copy_file(FIGURE_ROOT / filename, Path("06_消融实验") / "核心消融图" / filename)
    copy_matching(
        FIGURE_ROOT,
        "v3_dividend_ablation*",
        "06_消融实验/Carbon_Dividend_V3机制消融",
    )
    if (OUTPUT_ROOT / "annual_main_ablation").exists():
        for filename in (
            "protocol.json",
            "全年帕累托前沿数据.csv",
            "年度五随机种子汇总.csv",
            "年度合并前沿指标.csv",
            "年度配对显著性检验.csv",
            "年度随机种子明细.csv",
        ):
            copy_file(
                OUTPUT_ROOT / "annual_main_ablation" / filename,
                Path("06_消融实验") / "主方法全年消融" / "表格与协议" / filename,
            )
    for filename, destination in (
        ("annual_main_ablation_pareto_A1.png", "主方法与A1碳红利机制消融帕累托前沿.png"),
        ("annual_main_ablation_pareto_A3.png", "主方法与A3偏好选择机制消融帕累托前沿.png"),
        ("annual_main_ablation_pareto_A5.png", "主方法与A5向量经验回放消融帕累托前沿.png"),
        ("annual_main_ablation_metrics.png", "全年指标对比.png"),
    ):
        source = FIGURE_ROOT / filename
        if source.exists():
            copy_file(source, Path("06_消融实验") / "主方法全年消融" / destination)
    module_effect_root = OUTPUT_ROOT / "annual_main_ablation" / "module_effects"
    if module_effect_root.exists():
        copy_tree(
            module_effect_root,
            "06_消融实验/主方法全年消融/模块作用验证/表格与协议",
        )
    for filename, destination in (
        ("ablation_A1_mechanism_effect.png", "A1碳红利机制作用.png"),
        ("ablation_A3_mechanism_effect.png", "A3偏好选择机制作用.png"),
        ("ablation_A5_mechanism_effect.png", "A5向量经验回放作用.png"),
    ):
        source = FIGURE_ROOT / filename
        if source.exists():
            copy_file(
                source,
                Path("06_消融实验") / "主方法全年消融" / "模块作用验证" / destination,
            )

    copy_tree(
        OUTPUT_ROOT / "current_carbon_price_sensitivity",
        "07_碳价敏感性分析/表格与协议",
    )
    copy_matching(
        FIGURE_ROOT,
        "current_main_carbon_price_sensitivity*.png",
        "07_碳价敏感性分析/图",
    )

    copy_matching(
        FIGURE_ROOT,
        "v3_preference_season_5seed_*",
        "08_偏好与季节分解分析",
    )

    copy_tree(
        OUTPUT_ROOT / "dispatch_behavior_current_vs_maddpg",
        "09_调度行为对比/表格",
    )
    copy_file(
        FIGURE_ROOT / "dispatch_behavior_current_vs_maddpg.png",
        "09_调度行为对比/当前主方法与MADDPG调度行为.png",
    )

    copy_tree(
        OUTPUT_ROOT / "strict_algorithm_comparison",
        "10_实验协议与配置/严格跨算法协议",
    )
    copy_file(
        PROJECT_ROOT / "configs" / "main_method.json",
        "10_实验协议与配置/main_method.json",
    )
    copy_file(
        OUTPUT_ROOT / "README.md",
        "10_实验协议与配置/原始实验索引.md",
    )

    write_readme()
    write_manifest()
    print(f"Collected {len(COPIED)} files into {RESULT_ROOT}")


if __name__ == "__main__":
    main()
