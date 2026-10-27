import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_strict_algorithm_comparison import (  # noqa: E402
    actor_action,
    ensure_terminal_test_row,
    initial_states,
    load_actors,
)
from scripts.select_weight_matched_policies import (  # noqa: E402
    PREFERENCE_ORDER,
    select_weight_matched_policies,
)
from src.envs.ies_env import env as IESEnv  # noqa: E402


DEFAULT_RUN_DIR = (
    PROJECT_ROOT / "outputs" / "runs" / "strict_v3_all_evolving_kappa0_seed_44"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "figures" / "peer_trade_profiles_main_method"
)
DAILY_AGENT_SUMMARY = (
    PROJECT_ROOT / "outputs" / "main_method_agent_statistics" / "daily_agent_summary.csv"
)
SEASONS = {"spring": 24, "summer": 48, "autumn": 72, "winter": 0}
AGENTS = {
    1: ("residential", "居民区", "居民区"),
    2: ("commercial", "商业区", "商业区"),
    3: ("industry", "工业区", "工业区"),
}
PREFERENCE_NAMES_CN = {
    "cost_priority": "成本优先",
    "balanced": "相对均衡",
    "carbon_priority": "碳排放优先",
}
SEASON_NAMES_CN = {
    "spring": "春季",
    "summer": "夏季",
    "autumn": "秋季",
    "winter": "冬季",
}
COLORS = {1: "#3274A1", 2: "#E1812C", 3: "#3A923A"}
HATCHES = {1: "xx", 2: "..", 3: "\\\\"}


def configure_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "Times New Roman"],
            "font.size": 16,
            "axes.labelsize": 23,
            "xtick.labelsize": 16,
            "ytick.labelsize": 19,
            "legend.fontsize": 18,
            "axes.unicode_minus": False,
        }
    )


def signed_trade_by_counterparty(records, agent_id, counterparty_id):
    signed_quantity = 0.0
    for buyer, seller, quantity, _ in records:
        buyer = int(buyer)
        seller = int(seller)
        quantity = float(quantity)
        if buyer == agent_id and seller == counterparty_id:
            signed_quantity += quantity
        elif seller == agent_id and buyer == counterparty_id:
            signed_quantity -= quantity
    return signed_quantity


def simulate_policy(run_dir, preference, policy):
    actors = load_actors(run_dir / "final" / policy["policy_file"])
    rows = []
    max_reconstruction_error = 0.0
    for season, start in SEASONS.items():
        environment = IESEnv(
            market_mechanism="carbon_dividend_v3",
            cda_carbon_price=0.01,
            cda_dynamic_carbon_price=False,
        )
        states = initial_states(environment, start)
        for hour in range(24):
            action = actor_action(actors, states)
            next_res, next_com, next_ind, done, *_ = environment.step(
                action, np.concatenate(states, axis=0), start + hour, flag=1
            )
            records = list(environment.trade_record_e)
            for agent_id in AGENTS:
                reconstructed = 0.0
                for counterparty_id in AGENTS:
                    if counterparty_id == agent_id:
                        continue
                    signed_quantity = signed_trade_by_counterparty(
                        records, agent_id, counterparty_id
                    )
                    reconstructed += signed_quantity
                    rows.append(
                        {
                            "preference": preference,
                            "policy_file": policy["policy_file"],
                            "target_weight_cost": policy["target_weight_cost"],
                            "target_weight_carbon": policy["target_weight_carbon"],
                            "weight_cost": policy["weight_cost"],
                            "weight_carbon": policy["weight_carbon"],
                            "season": season,
                            "hour": hour,
                            "agent_id": agent_id,
                            "counterparty_id": counterparty_id,
                            "signed_trade_kwh": signed_quantity,
                        }
                    )
                expected = float(environment.trade_record_e_dict[agent_id])
                max_reconstruction_error = max(
                    max_reconstruction_error, abs(reconstructed - expected)
                )
            states = [next_res, next_com, next_ind]
            if done and hour < 23:
                raise RuntimeError(
                    f"Episode ended early for {preference}/{season} at hour {hour}."
                )
    return rows, max_reconstruction_error


def series_for(detail, preference, season, agent_id, counterparty_id):
    values = detail[
        (detail["preference"] == preference)
        & (detail["season"] == season)
        & (detail["agent_id"] == agent_id)
        & (detail["counterparty_id"] == counterparty_id)
    ].sort_values("hour")
    if len(values) != 24:
        raise RuntimeError(
            f"Expected 24 trade values for {preference}/{season}/"
            f"{agent_id}/{counterparty_id}, got {len(values)}."
        )
    return values["signed_trade_kwh"].to_numpy(dtype=np.float64)


def stacked_extents(series):
    positive = np.zeros(24, dtype=np.float64)
    negative = np.zeros(24, dtype=np.float64)
    for values in series.values():
        positive += np.maximum(values, 0.0)
        negative += np.minimum(values, 0.0)
    return float(negative.min()), float(positive.max())


def common_y_limits(detail):
    limits = {}
    for season in SEASONS:
        for agent_id in AGENTS:
            y_min = 0.0
            y_max = 0.0
            for preference in PREFERENCE_ORDER:
                series = {
                    counterparty_id: series_for(
                        detail, preference, season, agent_id, counterparty_id
                    )
                    for counterparty_id in AGENTS
                    if counterparty_id != agent_id
                }
                low, high = stacked_extents(series)
                y_min = min(y_min, low)
                y_max = max(y_max, high)
            span = y_max - y_min
            if span <= 1e-12:
                limits[(season, agent_id)] = (-1.0, 1.0)
            else:
                padding = 0.12 * span
                limits[(season, agent_id)] = (y_min - padding, y_max + padding)
    return limits


def plot_trade_profile(
    detail, output_dir, preference, policy, season, agent_id, y_limits, dpi
):
    counterparty_ids = [value for value in AGENTS if value != agent_id]
    series = {
        counterparty_id: series_for(
            detail, preference, season, agent_id, counterparty_id
        )
        for counterparty_id in counterparty_ids
    }
    x = np.arange(24)
    positive_bottom = np.zeros(24, dtype=np.float64)
    negative_bottom = np.zeros(24, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=dpi)
    for counterparty_id in counterparty_ids:
        values = series[counterparty_id]
        positive = np.maximum(values, 0.0)
        negative = np.minimum(values, 0.0)
        ax.bar(
            x,
            positive,
            width=0.74,
            bottom=positive_bottom,
            color=COLORS[counterparty_id],
            edgecolor="black",
            linewidth=0.9,
            hatch=HATCHES[counterparty_id],
        )
        ax.bar(
            x,
            negative,
            width=0.74,
            bottom=negative_bottom,
            color=COLORS[counterparty_id],
            edgecolor="black",
            linewidth=0.9,
            hatch=HATCHES[counterparty_id],
        )
        positive_bottom += positive
        negative_bottom += negative

    ax.axhline(0.0, color="black", linewidth=1.4)
    ax.set_xlabel("时间（h）", labelpad=14)
    ax.set_ylabel("交易电量（kWh）\n（正值为买入，负值为卖出）", labelpad=14)
    ax.set_xticks(x)
    ax.set_xlim(-1, 24)
    ax.set_ylim(*y_limits)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_x(-0.075)
    offset_text.set_y(1.0)
    offset_text.set_ha("center")
    offset_text.set_va("bottom")
    ax.grid(True, linestyle="--", alpha=1.0, linewidth=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("black")
    ax.tick_params(axis="both", which="major", length=10, width=2)
    handles = [
        Patch(
            facecolor=COLORS[counterparty_id],
            edgecolor="black",
            hatch=HATCHES[counterparty_id],
            label=AGENTS[counterparty_id][1],
        )
        for counterparty_id in counterparty_ids
    ]
    ax.legend(
        handles=handles,
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.58, 1.15),
        loc="upper center",
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.84)

    agent_key = AGENTS[agent_id][0]
    destination = output_dir / preference / season / agent_key
    destination.mkdir(parents=True, exist_ok=True)
    policy_stem = Path(policy["policy_file"]).stem
    path = destination / (
        f"peer_trade_{preference}_{season}_{agent_key}_{policy_stem}.png"
    )
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, dpi=dpi)
    plt.close(fig)
    return path


def localize_detail(detail):
    localized = detail.copy()
    localized["preference"] = localized["preference"].map(PREFERENCE_NAMES_CN)
    localized["season"] = localized["season"].map(SEASON_NAMES_CN)
    localized["agent_id"] = localized["agent_id"].map(
        {key: value[2] for key, value in AGENTS.items()}
    )
    localized["counterparty_id"] = localized["counterparty_id"].map(
        {key: value[2] for key, value in AGENTS.items()}
    )
    return localized.rename(
        columns={
            "preference": "偏好类型",
            "policy_file": "策略文件",
            "target_weight_cost": "目标成本权重",
            "target_weight_carbon": "目标碳排放权重",
            "weight_cost": "实际成本权重",
            "weight_carbon": "实际碳排放权重",
            "season": "典型日",
            "hour": "时刻",
            "agent_id": "本园区",
            "counterparty_id": "交易对象园区",
            "signed_trade_kwh": "交易电量_kWh_买入为正卖出为负",
        }
    )


def validate_against_agent_summary(detail, maximum_reconstruction_error):
    daily = pd.read_csv(DAILY_AGENT_SUMMARY, encoding="utf-8-sig")
    aggregate = (
        detail.groupby(["preference", "season", "agent_id"], as_index=False)[
            "signed_trade_kwh"
        ]
        .sum()
        .assign(
            偏好类型=lambda frame: frame["preference"].map(PREFERENCE_NAMES_CN),
            典型日=lambda frame: frame["season"].map(SEASON_NAMES_CN),
            园区=lambda frame: frame["agent_id"].map(
                {key: value[2] for key, value in AGENTS.items()}
            ),
        )
    )
    comparison = aggregate.merge(
        daily[
            [
                "偏好类型",
                "典型日",
                "园区",
                "园区间总购电量_kWh_买入为正卖出为负",
            ]
        ],
        on=["偏好类型", "典型日", "园区"],
        how="left",
        validate="one_to_one",
    )
    if comparison["园区间总购电量_kWh_买入为正卖出为负"].isna().any():
        raise RuntimeError("Trade plots could not be matched to daily_agent_summary.csv.")
    daily_error = np.abs(
        comparison["signed_trade_kwh"]
        - comparison["园区间总购电量_kWh_买入为正卖出为负"]
    )

    indexed = detail.set_index(
        ["preference", "season", "hour", "agent_id", "counterparty_id"]
    )["signed_trade_kwh"]
    bilateral_error = 0.0
    for preference in PREFERENCE_ORDER:
        for season in SEASONS:
            for hour in range(24):
                for first in AGENTS:
                    for second in AGENTS:
                        if first >= second:
                            continue
                        error = abs(
                            float(indexed.loc[preference, season, hour, first, second])
                            + float(indexed.loc[preference, season, hour, second, first])
                        )
                        bilateral_error = max(bilateral_error, error)

    checks = {
        "hourly_counterparty_rows": int(len(detail)),
        "maximum_agent_trade_reconstruction_error_kwh": float(
            maximum_reconstruction_error
        ),
        "maximum_bilateral_antisymmetry_error_kwh": float(bilateral_error),
        "maximum_daily_error_vs_agent_summary_kwh": float(daily_error.max()),
        "all_values_finite": bool(
            np.isfinite(detail["signed_trade_kwh"].to_numpy(dtype=np.float64)).all()
        ),
    }
    checks["validation_passed"] = bool(
        checks["all_values_finite"]
        and checks["maximum_agent_trade_reconstruction_error_kwh"] <= 1e-9
        and checks["maximum_bilateral_antisymmetry_error_kwh"] <= 1e-9
        and checks["maximum_daily_error_vs_agent_summary_kwh"] <= 1e-9
    )
    return checks


def main():
    parser = argparse.ArgumentParser(
        description="Plot signed peer electricity trades by counterparty."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    ensure_terminal_test_row()
    selected = select_weight_matched_policies(run_dir)
    rows = []
    maximum_reconstruction_error = 0.0
    for preference in PREFERENCE_ORDER:
        policy_rows, error = simulate_policy(
            run_dir, preference, selected[preference]
        )
        rows.extend(policy_rows)
        maximum_reconstruction_error = max(maximum_reconstruction_error, error)
    detail = pd.DataFrame(rows)
    checks = validate_against_agent_summary(detail, maximum_reconstruction_error)
    if not checks["validation_passed"]:
        raise RuntimeError("Peer-trade profile validation failed.")

    for path in output_dir.rglob("*.png"):
        path.unlink()
    limits = common_y_limits(detail)
    figure_paths = []
    for preference in PREFERENCE_ORDER:
        for season in SEASONS:
            for agent_id in AGENTS:
                figure_paths.append(
                    plot_trade_profile(
                        detail,
                        output_dir,
                        preference,
                        selected[preference],
                        season,
                        agent_id,
                        limits[(season, agent_id)],
                        args.dpi,
                    )
                )

    localized = localize_detail(detail)
    localized.to_csv(
        output_dir / "peer_trade_hourly_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily_counterparty = (
        localized.groupby(
            [
                "偏好类型",
                "策略文件",
                "目标成本权重",
                "目标碳排放权重",
                "实际成本权重",
                "实际碳排放权重",
                "典型日",
                "本园区",
                "交易对象园区",
            ],
            as_index=False,
            sort=False,
        )["交易电量_kWh_买入为正卖出为负"]
        .sum()
    )
    daily_counterparty.to_csv(
        output_dir / "peer_trade_daily_by_counterparty.csv",
        index=False,
        encoding="utf-8-sig",
    )
    checks["figure_count"] = len(figure_paths)
    checks["selected_policies"] = {
        preference: selected[preference]["policy_file"]
        for preference in PREFERENCE_ORDER
    }
    (output_dir / "validation.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(checks, indent=2))
    print(f"Saved peer-trade figures to: {output_dir}")


if __name__ == "__main__":
    main()
