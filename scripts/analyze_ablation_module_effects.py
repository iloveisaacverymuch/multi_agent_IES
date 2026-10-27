import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_current_carbon_cda_trading_3seed import step_metrics  # noqa: E402
from scripts.evaluate_full_year_algorithm_comparison import (  # noqa: E402
    HOURS_PER_YEAR,
    initial_states,
    install_full_year_data,
)
from scripts.evaluate_strict_algorithm_comparison import actor_action, load_actors  # noqa: E402
from src.envs.ies_env import env as IESEnv  # noqa: E402


RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "annual_main_ablation" / "module_effects"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"
SEEDS = [11, 22, 33, 44, 55]
PREFERENCES = {
    "成本优先": np.array([0.8, 0.2]),
    "均衡偏好": np.array([0.5, 0.5]),
    "碳排放优先": np.array([0.2, 0.8]),
}
A1_METHODS = {
    "A0完整方法": ("strict_v3_all_evolving_kappa0_seed_{seed}", "carbon_dividend_v3"),
    "A1移除碳红利": ("strict_standard_all_evolving_kappa0_seed_{seed}", "standard"),
}
A3_METHODS = {
    "A0完整方法": "strict_v3_all_evolving_kappa0_seed_{seed}",
    "A3随机偏好": "strict_v3_all_evolving_random_seed_{seed}",
}
A5_METHODS = {
    "A0向量重标记": "strict_v3_all_evolving_kappa0_seed_{seed}",
    "A5旧标量奖励": "strict_v3_replay_scalar_stale_seed_{seed}",
}
FIGURE_LABELS = {
    "A0完整方法": "CDV3-MHV-VR-\nEMADDPG",
    "A1移除碳红利": "SCDA-MHV-VR-\nEMADDPG",
    "A3随机偏好": "CDV3-RP-VR-\nEMADDPG",
    "A0向量重标记": "CDV3-MHV-VR-\nEMADDPG",
    "A5旧标量奖励": "CDV3-MHV-SR-\nEMADDPG",
}


def run_path(pattern, seed):
    return RUN_ROOT / pattern.format(seed=seed)


def nondominated_mask(values):
    values = np.asarray(values, dtype=np.float64)
    dominated = np.zeros(len(values), dtype=bool)
    for index, value in enumerate(values):
        dominated[index] = np.any(
            np.all(values <= value, axis=1) & np.any(values < value, axis=1)
        )
    return ~dominated


def representative_policies(path):
    trained = pd.read_csv(path / "final" / "metadata.csv")
    annual = pd.read_csv(path / "annual_test" / "metadata.csv")
    frame = annual.merge(
        trained[["policy_file", "weight_cost", "weight_carbon"]],
        on="policy_file",
        how="left",
        validate="one_to_one",
    )
    values = frame[["annual_cost", "annual_carbon"]].to_numpy(dtype=np.float64)
    frame = frame.loc[nondominated_mask(values)].copy()
    values = frame[["annual_cost", "annual_carbon"]].to_numpy(dtype=np.float64)
    ideal = values.min(axis=0)
    scale = np.maximum(values.max(axis=0) - ideal, 1e-12)
    normalized = (values - ideal) / scale
    selected = []
    for preference, target in PREFERENCES.items():
        distance = np.abs(frame["weight_cost"].to_numpy(dtype=float) - target[0])
        closest = np.flatnonzero(np.isclose(distance, distance.min()))
        scalarized = normalized[closest] @ target
        row = frame.iloc[int(closest[np.argmin(scalarized)])]
        selected.append(
            {
                "偏好": preference,
                "策略文件": row["policy_file"],
                "实际成本权重": float(row["weight_cost"]),
                "实际碳权重": float(row["weight_carbon"]),
                "全年运行成本_CNY": float(row["annual_cost"]),
                "全年碳排放_kgCO2": float(row["annual_carbon"]),
            }
        )
    return selected


def simulate_annual_policy(path, mechanism, policy):
    arrays = install_full_year_data()
    environment = IESEnv(
        market_mechanism=mechanism,
        cda_carbon_price=0.01,
        cda_dynamic_carbon_price=False,
    )
    actors = load_actors(path / "final" / policy["策略文件"])
    states = initial_states(environment, arrays)
    sums = {
        "园区交易电量_kWh": 0.0,
        "电网购电量_kWh": 0.0,
        "电网上网电量_kWh": 0.0,
        "交易卖方碳强度加权量": 0.0,
        "最低碳卖方交易电量_kWh": 0.0,
        "碳红利资金_CNY": 0.0,
        "碳资金转移绝对值_CNY": 0.0,
        "避免碳排放返还依据": 0.0,
    }
    max_balance_error = 0.0
    active_hours = 0
    for hour in range(HOURS_PER_YEAR):
        action = actor_action(actors, states)
        next_res, next_com, next_ind, _, _, _, _ = environment.step(
            action, np.concatenate(states, axis=0), hour, flag=1
        )
        step = step_metrics(environment)
        sums["园区交易电量_kWh"] += step["peer_trade_amount"]
        sums["电网购电量_kWh"] += step["grid_buy"]
        sums["电网上网电量_kWh"] += step["grid_sell"]
        sums["交易卖方碳强度加权量"] += step["seller_carbon_amount"]
        sums["最低碳卖方交易电量_kWh"] += step["cleanest_seller_amount"]
        sums["碳红利资金_CNY"] += step["carbon_dividend_fund"]
        sums["碳资金转移绝对值_CNY"] += step["carbon_transfer_abs"]
        sums["避免碳排放返还依据"] += step["avoided_carbon_credit_basis"]
        max_balance_error = max(max_balance_error, abs(step["carbon_dividend_balance_error"]))
        active_hours += int(step["carbon_dividend_fund"] > 1e-12)
        states = [next_res, next_com, next_ind]
    peer = sums["园区交易电量_kWh"]
    return {
        **policy,
        **sums,
        "交易卖方平均碳强度": sums["交易卖方碳强度加权量"] / max(peer, 1e-12),
        "最低碳卖方交易占比": sums["最低碳卖方交易电量_kWh"] / max(peer, 1e-12),
        "最大预算平衡误差_CNY": max_balance_error,
        "碳红利生效小时占比": active_hours / HOURS_PER_YEAR,
    }


def evaluate_a1_seed(seed):
    rows = []
    for method, (pattern, mechanism) in A1_METHODS.items():
        path = run_path(pattern, seed)
        for policy in representative_policies(path):
            row = simulate_annual_policy(path, mechanism, policy)
            row.update({"方法": method, "随机种子": seed})
            rows.append(row)
    return rows


def evaluate_a1(workers):
    cache = OUTPUT_ROOT / "A1碳红利机制年度明细.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(evaluate_a1_seed, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed = futures[future]
            rows.extend(future.result())
            print(f"A1 mechanism behavior complete: seed={seed}", flush=True)
    detail = pd.DataFrame(rows)
    detail.to_csv(cache, index=False, encoding="utf-8-sig")
    return detail


def summarize_a1(detail):
    seed_summary = detail.groupby(["方法", "随机种子"], as_index=False).agg(
        全年运行成本_CNY=("全年运行成本_CNY", "mean"),
        全年碳排放_kgCO2=("全年碳排放_kgCO2", "mean"),
        园区交易电量_kWh=("园区交易电量_kWh", "mean"),
        电网购电量_kWh=("电网购电量_kWh", "mean"),
        交易卖方平均碳强度=("交易卖方平均碳强度", "mean"),
        最低碳卖方交易占比=("最低碳卖方交易占比", "mean"),
        碳红利资金_CNY=("碳红利资金_CNY", "mean"),
        碳资金转移绝对值_CNY=("碳资金转移绝对值_CNY", "mean"),
        最大预算平衡误差_CNY=("最大预算平衡误差_CNY", "max"),
        碳红利生效小时占比=("碳红利生效小时占比", "mean"),
    )
    summary = seed_summary.groupby("方法", as_index=False).agg(
        全年碳排放均值_kgCO2=("全年碳排放_kgCO2", "mean"),
        园区交易电量均值_kWh=("园区交易电量_kWh", "mean"),
        电网购电量均值_kWh=("电网购电量_kWh", "mean"),
        交易卖方平均碳强度=("交易卖方平均碳强度", "mean"),
        最低碳卖方交易占比=("最低碳卖方交易占比", "mean"),
        碳红利资金均值_CNY=("碳红利资金_CNY", "mean"),
        碳资金转移绝对值均值_CNY=("碳资金转移绝对值_CNY", "mean"),
        最大预算平衡误差_CNY=("最大预算平衡误差_CNY", "max"),
        碳红利生效小时占比=("碳红利生效小时占比", "mean"),
    )
    seed_summary.to_csv(OUTPUT_ROOT / "A1碳红利机制种子汇总.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_ROOT / "A1碳红利机制总体汇总.csv", index=False, encoding="utf-8-sig")
    return seed_summary, summary


def analyze_a3():
    rows = []
    seed_rows = []
    for method, pattern in A3_METHODS.items():
        for seed in SEEDS:
            path = run_path(pattern, seed)
            selection = pd.read_csv(path / "selection_history.csv")
            episodes = pd.read_csv(path / "episode_log.csv")
            for generation, frame in selection.groupby("generation"):
                weights = np.sort(frame["new_weight_cost"].to_numpy(dtype=float))
                unique = np.unique(np.round(weights, 6))
                positive_gaps = np.diff(unique)
                rows.append(
                    {
                        "方法": method,
                        "随机种子": seed,
                        "代数": int(generation),
                        "有效偏好数量": len(unique),
                        "重复偏好率": 1.0 - len(unique) / len(weights),
                        "偏好覆盖宽度": float(weights.max() - weights.min()),
                        "平均有效偏好间距": float(positive_gaps.mean()) if len(positive_gaps) else 0.0,
                        "双极端覆盖": int(weights.min() <= 0.2 and weights.max() >= 0.8),
                        "平均偏好切换幅度": float(
                            np.abs(frame["new_weight_cost"] - frame["old_weight_cost"]).mean()
                        ),
                    }
                )
            final_archive = int(episodes["archive_size"].iloc[-1])
            acceptance = float(episodes["archive_accepted"].mean())
            seed_rows.append(
                {
                    "方法": method,
                    "随机种子": seed,
                    "最终训练档案规模": final_archive,
                    "策略档案接受率": acceptance,
                }
            )
    detail = pd.DataFrame(rows)
    seed_extra = pd.DataFrame(seed_rows)
    seed_summary = detail.groupby(["方法", "随机种子"], as_index=False).agg(
        平均有效偏好数量=("有效偏好数量", "mean"),
        平均重复偏好率=("重复偏好率", "mean"),
        平均偏好覆盖宽度=("偏好覆盖宽度", "mean"),
        双极端覆盖率=("双极端覆盖", "mean"),
        平均偏好切换幅度=("平均偏好切换幅度", "mean"),
    ).merge(seed_extra, on=["方法", "随机种子"], validate="one_to_one")
    summary = seed_summary.groupby("方法", as_index=False).mean(numeric_only=True).drop(columns="随机种子")
    detail.to_csv(OUTPUT_ROOT / "A3偏好选择逐代明细.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(OUTPUT_ROOT / "A3偏好选择种子汇总.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_ROOT / "A3偏好选择总体汇总.csv", index=False, encoding="utf-8-sig")
    return detail, seed_summary, summary


def stale_reward_mismatch(path):
    episodes = pd.read_csv(path / "episode_log.csv")
    rows = []
    for (task_id, generation), current in episodes.groupby(["task_id", "generation"]):
        if generation <= 1:
            continue
        current_weight = float(current["weight_cost"].iloc[0])
        history = episodes[(episodes.task_id == task_id) & (episodes.generation < generation)]
        old_weight = history["weight_cost"].to_numpy(dtype=float)
        cost = history["eval_cost"].to_numpy(dtype=float)
        carbon = history["eval_carbon"].to_numpy(dtype=float)
        stale = old_weight * cost + (1.0 - old_weight) * carbon
        relabelled = current_weight * cost + (1.0 - current_weight) * carbon
        relative_error = np.abs(stale - relabelled) / np.maximum(np.abs(relabelled), 1.0)
        changed = ~np.isclose(old_weight, current_weight)
        rows.append(
            {
                "代数": int(generation),
                "任务": int(task_id),
                "历史经验数量": len(history),
                "偏好不一致经验占比": float(changed.mean()),
                "潜在奖励相对错配率": float(relative_error.mean()),
                "最大潜在奖励相对错配率": float(relative_error.max()),
            }
        )
    return pd.DataFrame(rows)


def analyze_a5():
    detail_rows = []
    for method, pattern in A5_METHODS.items():
        for seed in SEEDS:
            frame = stale_reward_mismatch(run_path(pattern, seed))
            frame["方法"] = method
            frame["随机种子"] = seed
            if method == "A0向量重标记":
                frame["实际奖励相对错配率"] = 0.0
            else:
                frame["实际奖励相对错配率"] = frame["潜在奖励相对错配率"]
            detail_rows.append(frame)
    detail = pd.concat(detail_rows, ignore_index=True)
    seed_summary = detail.groupby(["方法", "随机种子"], as_index=False).agg(
        历史经验偏好不一致占比=("偏好不一致经验占比", "mean"),
        潜在奖励相对错配率=("潜在奖励相对错配率", "mean"),
        实际奖励相对错配率=("实际奖励相对错配率", "mean"),
        最大实际奖励相对错配率=("实际奖励相对错配率", "max"),
    )
    summary = seed_summary.groupby("方法", as_index=False).mean(numeric_only=True).drop(columns="随机种子")
    detail.to_csv(OUTPUT_ROOT / "A5奖励错配逐代明细.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(OUTPUT_ROOT / "A5奖励错配种子汇总.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_ROOT / "A5奖励错配总体汇总.csv", index=False, encoding="utf-8-sig")
    return detail, seed_summary, summary


def paired_tests(a1_seed, a3_seed, a5_seed):
    rows = []
    specs = [
        ("A1", a1_seed, "A0完整方法", "A1移除碳红利", ["园区交易电量_kWh", "电网购电量_kWh", "交易卖方平均碳强度"]),
        ("A3", a3_seed, "A0完整方法", "A3随机偏好", ["平均有效偏好数量", "平均重复偏好率", "平均偏好覆盖宽度", "双极端覆盖率"]),
        ("A5", a5_seed, "A0向量重标记", "A5旧标量奖励", ["实际奖励相对错配率"]),
    ]
    for module, frame, main_name, ablation_name, metrics in specs:
        indexed = frame.set_index(["方法", "随机种子"])
        for metric in metrics:
            first = indexed.loc[[(main_name, seed) for seed in SEEDS], metric].to_numpy(float)
            second = indexed.loc[[(ablation_name, seed) for seed in SEEDS], metric].to_numpy(float)
            difference = first - second
            p_value = 1.0 if np.allclose(difference, 0.0) else float(
                wilcoxon(first, second, alternative="two-sided").pvalue
            )
            rows.append(
                {
                    "消融模块": module,
                    "机制指标": metric,
                    "完整方法均值": first.mean(),
                    "消融方法均值": second.mean(),
                    "完整方法减消融方法": difference.mean(),
                    "双侧Wilcoxon检验p值": p_value,
                }
            )
    tests = pd.DataFrame(rows)
    tests.to_csv(OUTPUT_ROOT / "模块机制指标配对检验.csv", index=False, encoding="utf-8-sig")
    return tests


def write_relative_effects(a1_summary, a3_summary, a5_summary):
    a1 = a1_summary.set_index("方法")
    a3 = a3_summary.set_index("方法")
    a5 = a5_summary.set_index("方法")
    rows = []

    def add(module, metric, full, ablated, direction):
        if direction == "提高":
            change = 100.0 * (full / ablated - 1.0) if abs(ablated) > 1e-12 else np.nan
        else:
            change = 100.0 * (1.0 - full / ablated) if abs(ablated) > 1e-12 else np.nan
        rows.append(
            {
                "消融模块": module,
                "机制指标": metric,
                "完整方法": full,
                "消融方法": ablated,
                "期望方向": direction,
                "相对改善百分比": change,
            }
        )

    add("A1 Carbon-Dividend V3", "全年碳排放", a1.loc["A0完整方法", "全年碳排放均值_kgCO2"], a1.loc["A1移除碳红利", "全年碳排放均值_kgCO2"], "降低")
    add("A1 Carbon-Dividend V3", "园区交易电量", a1.loc["A0完整方法", "园区交易电量均值_kWh"], a1.loc["A1移除碳红利", "园区交易电量均值_kWh"], "提高")
    add("A1 Carbon-Dividend V3", "电网购电量", a1.loc["A0完整方法", "电网购电量均值_kWh"], a1.loc["A1移除碳红利", "电网购电量均值_kWh"], "降低")
    add("A1 Carbon-Dividend V3", "交易卖方平均碳强度", a1.loc["A0完整方法", "交易卖方平均碳强度"], a1.loc["A1移除碳红利", "交易卖方平均碳强度"], "降低")
    add("A3 Mean-HV偏好选择", "有效偏好数量", a3.loc["A0完整方法", "平均有效偏好数量"], a3.loc["A3随机偏好", "平均有效偏好数量"], "提高")
    add("A3 Mean-HV偏好选择", "偏好覆盖宽度", a3.loc["A0完整方法", "平均偏好覆盖宽度"], a3.loc["A3随机偏好", "平均偏好覆盖宽度"], "提高")
    add("A3 Mean-HV偏好选择", "双极端覆盖率", a3.loc["A0完整方法", "双极端覆盖率"], a3.loc["A3随机偏好", "双极端覆盖率"], "提高")
    add("A3 Mean-HV偏好选择", "重复偏好率", a3.loc["A0完整方法", "平均重复偏好率"], a3.loc["A3随机偏好", "平均重复偏好率"], "降低")
    add("A5 向量经验回放", "实际奖励相对错配率", a5.loc["A0向量重标记", "实际奖励相对错配率"], a5.loc["A5旧标量奖励", "实际奖励相对错配率"], "降低")
    effects = pd.DataFrame(rows)
    effects.to_csv(OUTPUT_ROOT / "模块机制相对效应汇总.csv", index=False, encoding="utf-8-sig")
    return effects


def plot_a1(summary):
    order = list(A1_METHODS)
    frame = summary.set_index("方法").loc[order]
    specs = [
        ("园区交易电量均值_kWh", "全年园区交易电量（kWh）"),
        ("电网购电量均值_kWh", "全年电网购电量（kWh）"),
        ("交易卖方平均碳强度", "交易卖方平均碳强度"),
        ("碳红利资金均值_CNY", "预算平衡碳红利资金（CNY）"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), constrained_layout=True)
    for ax, (column, ylabel) in zip(axes.ravel(), specs):
        ax.bar([0, 1], frame[column], color=["#C44E52", "#4C78A8"], edgecolor="black")
        ax.set_xticks([0, 1], [FIGURE_LABELS[key] for key in order])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.savefig(FIGURE_ROOT / "ablation_A1_mechanism_effect.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_a3(summary):
    order = list(A3_METHODS)
    frame = summary.set_index("方法").loc[order]
    specs = [
        ("平均有效偏好数量", "每代有效偏好数量"),
        ("平均重复偏好率", "重复偏好率"),
        ("平均偏好覆盖宽度", "偏好覆盖宽度"),
        ("双极端覆盖率", "极端偏好覆盖率"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), constrained_layout=True)
    for ax, (column, ylabel) in zip(axes.ravel(), specs):
        ax.bar([0, 1], frame[column], color=["#C44E52", "#F28E2B"], edgecolor="black")
        ax.set_xticks([0, 1], [FIGURE_LABELS[key] for key in order])
        ax.tick_params(axis="x", labelsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    fig.savefig(FIGURE_ROOT / "ablation_A3_mechanism_effect.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_a5(detail, summary):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    plot_labels = {
        "A0向量重标记": "CDV3-MHV-VR-EMADDPG",
        "A5旧标量奖励": "CDV3-MHV-SR-EMADDPG",
    }
    for method, color in (("A0向量重标记", "#C44E52"), ("A5旧标量奖励", "#17BECF")):
        frame = detail[detail.方法 == method].groupby("代数", as_index=False)["实际奖励相对错配率"].mean()
        axes[0].plot(frame["代数"], 100.0 * frame["实际奖励相对错配率"], marker="o", color=color, label=plot_labels[method])
    axes[0].set_xlabel("代数")
    axes[0].set_ylabel("实际奖励错配率（%）")
    axes[0].grid(linestyle="--", alpha=0.6)
    axes[0].legend(frameon=False)
    order = list(A5_METHODS)
    frame = summary.set_index("方法").loc[order]
    axes[1].bar(
        [0, 1],
        100.0 * frame["实际奖励相对错配率"],
        color=["#C44E52", "#17BECF"],
        edgecolor="black",
    )
    axes[1].set_xticks([0, 1], [FIGURE_LABELS[key] for key in order])
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].set_ylabel("平均实际奖励错配率（%）")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    fig.savefig(FIGURE_ROOT / "ablation_A5_mechanism_effect.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": ["Microsoft YaHei", "Times New Roman"]})
    parser = argparse.ArgumentParser(description="Analyze module-specific effects for A1/A3/A5.")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    a1_detail = evaluate_a1(args.workers)
    a1_seed, a1_summary = summarize_a1(a1_detail)
    _, a3_seed, a3_summary = analyze_a3()
    a5_detail, a5_seed, a5_summary = analyze_a5()
    tests = paired_tests(a1_seed, a3_seed, a5_seed)
    effects = write_relative_effects(a1_summary, a3_summary, a5_summary)
    plot_a1(a1_summary)
    plot_a3(a3_summary)
    plot_a5(a5_detail, a5_summary)
    protocol = {
        "目的": "用模块专属机制指标解释A1、A3、A5的作用路径，全年性能指标仅作为最终结果补充。",
        "年度评价小时数": HOURS_PER_YEAR,
        "随机种子": SEEDS,
        "代表性偏好": {key: value.tolist() for key, value in PREFERENCES.items()},
        "A1机制指标": ["碳红利资金", "预算平衡误差", "园区交易", "电网购电", "卖方碳强度"],
        "A3机制指标": ["有效偏好数量", "重复偏好率", "偏好覆盖宽度", "极端偏好覆盖率"],
        "A5机制指标": ["历史经验偏好不一致占比", "奖励相对错配率"],
    }
    (OUTPUT_ROOT / "机制消融实验协议.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("A1 mechanism summary:\n" + a1_summary.to_string(index=False))
    print("\nA3 mechanism summary:\n" + a3_summary.to_string(index=False))
    print("\nA5 mechanism summary:\n" + a5_summary.to_string(index=False))
    print("\nPaired tests:\n" + tests.to_string(index=False))
    print("\nRelative module effects:\n" + effects.to_string(index=False))


if __name__ == "__main__":
    main()
