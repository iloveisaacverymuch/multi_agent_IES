import argparse
import json
import sys
from pathlib import Path

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "main_method_agent_statistics"
SEASONS = {"spring": 24, "summer": 48, "autumn": 72, "winter": 0}
AGENTS = (
    ("residential", 1, "residential"),
    ("commercial", 2, "commercial"),
    ("industry", 3, "industry"),
)

COLUMN_NAMES_CN = {
    "preference": "偏好类型",
    "policy_file": "策略文件",
    "pareto_index": "帕累托索引",
    "season": "典型日",
    "hour": "时刻",
    "agent": "园区",
    "operating_cost": "运行成本",
    "physical_cost_before_carbon_transfer": "碳转移前物理运行成本",
    "carbon_emissions": "碳排放量_kgCO2",
    "peer_import_kwh": "园区间累计买入电量_kWh",
    "peer_export_kwh": "园区间累计卖出电量_kWh",
    "net_peer_import_kwh": "园区间总购电量_kWh_买入为正卖出为负",
    "peer_purchase_cost": "园区间购电支出",
    "peer_sales_revenue": "园区间售电收入",
    "peer_trade_count": "园区间交易次数",
    "grid_import_kwh": "电网购电量_kWh",
    "grid_export_kwh": "电网上网电量_kWh",
    "proposed_electric_surplus_kwh": "申报电能余量_kWh",
    "electricity_bid_price": "电能报价",
    "carbon_levy": "碳征费",
    "carbon_credit": "碳红利",
    "net_carbon_transfer": "净碳转移支出",
    "waste_heat": "弃热量",
    "electricity_balance_residual": "电功率平衡残差",
    "max_abs_electricity_balance_residual": "最大电功率平衡绝对残差",
    "peer_active_hours": "园区间交易活跃时数",
    "average_peer_import_price": "平均园区间购电价",
    "average_peer_export_price": "平均园区间售电价",
    "peer_trade_volume_kwh": "园区间成交电量_kWh",
    "peer_energy_balance_error_kwh": "园区间电量平衡误差_kWh",
    "peer_cash_balance_error": "交易资金平衡误差",
    "carbon_dividend_balance_error": "碳红利预算平衡误差",
    "cost_error_vs_strict_test": "严格测试运行成本误差",
    "carbon_error_vs_strict_test": "严格测试碳排放误差",
    "task_id": "任务编号",
    "role": "策略角色",
    "generation": "代数",
    "episode": "训练回合",
    "weight_cost": "成本权重",
    "weight_carbon": "碳排放权重",
    "target_weight_cost": "目标成本权重",
    "target_weight_carbon": "目标碳排放权重",
    "weight_distance": "权重距离",
    "strict_nondominated": "严格测试非支配策略",
    "cost": "系统运行成本",
    "carbon": "系统碳排放量_kgCO2",
}

CATEGORY_NAMES_CN = {
    "preference": {
        "cost_priority": "成本优先",
        "balanced": "相对均衡",
        "carbon_priority": "碳排放优先",
    },
    "season": {
        "spring": "春季",
        "summer": "夏季",
        "autumn": "秋季",
        "winter": "冬季",
    },
    "agent": {
        "residential": "居民区",
        "commercial": "商业区",
        "industry": "工业区",
    },
    "role": {
        "evolving": "进化任务",
        "anchor": "锚点任务",
        "fixed": "固定任务",
    },
    "strict_nondominated": {True: "是", False: "否"},
}


def trade_metrics(records, agent_id):
    imports = [record for record in records if int(record[0]) == agent_id]
    exports = [record for record in records if int(record[1]) == agent_id]
    import_amount = sum(float(record[2]) for record in imports)
    export_amount = sum(float(record[2]) for record in exports)
    purchase_cost = sum(float(record[2]) * float(record[3]) for record in imports)
    sales_revenue = sum(float(record[2]) * float(record[3]) for record in exports)
    return {
        "peer_import_kwh": import_amount,
        "peer_export_kwh": export_amount,
        "net_peer_import_kwh": import_amount - export_amount,
        "peer_purchase_cost": purchase_cost,
        "peer_sales_revenue": sales_revenue,
        "peer_trade_count": len(imports) + len(exports),
    }


def electricity_balance_residual(environment, suffix, net_peer_import):
    supply = (
        getattr(environment, f"P_HCHP_{suffix}")
        + getattr(environment, f"P_CHP_{suffix}")
        + getattr(environment, f"P_ESS_{suffix}")
        + getattr(environment, f"PV_{suffix}")
        + getattr(environment, f"buy_from_grid_{suffix}")
        - getattr(environment, f"sell_to_grid_{suffix}")
        + net_peer_import
    )
    demand = getattr(environment, f"electric_load_{suffix}")
    if suffix in ("residential", "commercial"):
        demand += getattr(environment, f"C_CER_{suffix}") / 4.0
        demand += getattr(environment, f"H_EB_{suffix}") / 0.95
    return float(supply - demand)


def simulate_day(run_dir, preference, policy, season, start):
    environment = IESEnv(
        market_mechanism="carbon_dividend_v3",
        cda_carbon_price=0.01,
        cda_dynamic_carbon_price=False,
    )
    actors = load_actors(run_dir / "final" / policy["policy_file"])
    states = initial_states(environment, start)
    rows = []

    for hour in range(24):
        action = actor_action(actors, states)
        next_res, next_com, next_ind, done, obj_res, obj_com, obj_ind = environment.step(
            action,
            np.concatenate(states, axis=0),
            start + hour,
            flag=1,
        )
        objectives = {
            1: np.asarray(obj_res, dtype=np.float64),
            2: np.asarray(obj_com, dtype=np.float64),
            3: np.asarray(obj_ind, dtype=np.float64),
        }
        records = list(environment.trade_record_e)

        for agent, agent_id, suffix in AGENTS:
            trade = trade_metrics(records, agent_id)
            signed_trade = float(environment.trade_record_e_dict[agent_id])
            if not np.isclose(
                trade["net_peer_import_kwh"], signed_trade, atol=1e-9, rtol=0.0
            ):
                raise RuntimeError(
                    f"Peer-trade sign mismatch for {preference}/{season}/{agent}/h{hour}."
                )

            levy = float(environment.carbon_levy[agent_id])
            credit = float(environment.carbon_credit[agent_id])
            operating_cost = float(objectives[agent_id][0])
            carbon = float(objectives[agent_id][1])
            rows.append(
                {
                    "preference": preference,
                    "policy_file": policy["policy_file"],
                    "pareto_index": policy["pareto_index"],
                    "target_weight_cost": policy["target_weight_cost"],
                    "target_weight_carbon": policy["target_weight_carbon"],
                    "weight_cost": policy["weight_cost"],
                    "weight_carbon": policy["weight_carbon"],
                    "season": season,
                    "hour": hour,
                    "agent": agent,
                    "operating_cost": operating_cost,
                    "physical_cost_before_carbon_transfer": operating_cost
                    - levy
                    + credit,
                    "carbon_emissions": carbon,
                    **trade,
                    "grid_import_kwh": float(
                        getattr(environment, f"buy_from_grid_{suffix}")
                    ),
                    "grid_export_kwh": float(
                        getattr(environment, f"sell_to_grid_{suffix}")
                    ),
                    "proposed_electric_surplus_kwh": float(
                        getattr(environment, f"trade_e_{suffix}")
                    ),
                    "electricity_bid_price": float(
                        getattr(environment, f"price_e_{suffix}")
                    ),
                    "carbon_levy": levy,
                    "carbon_credit": credit,
                    "net_carbon_transfer": levy - credit,
                    "waste_heat": float(getattr(environment, f"waste_h_{suffix}")),
                    "electricity_balance_residual": electricity_balance_residual(
                        environment, suffix, signed_trade
                    ),
                }
            )

        states = [next_res, next_com, next_ind]
        if done and hour < 23:
            raise RuntimeError(f"Episode ended early for {preference}/{season} at h{hour}.")

    return rows


def summarize_agent_days(detail):
    keys = [
        "preference",
        "policy_file",
        "pareto_index",
        "target_weight_cost",
        "target_weight_carbon",
        "weight_cost",
        "weight_carbon",
        "season",
        "agent",
    ]
    summary = (
        detail.groupby(keys, as_index=False, sort=False)
        .agg(
            operating_cost=("operating_cost", "sum"),
            physical_cost_before_carbon_transfer=(
                "physical_cost_before_carbon_transfer",
                "sum",
            ),
            carbon_emissions=("carbon_emissions", "sum"),
            net_peer_import_kwh=("net_peer_import_kwh", "sum"),
            peer_import_kwh=("peer_import_kwh", "sum"),
            peer_export_kwh=("peer_export_kwh", "sum"),
            peer_purchase_cost=("peer_purchase_cost", "sum"),
            peer_sales_revenue=("peer_sales_revenue", "sum"),
            peer_trade_count=("peer_trade_count", "sum"),
            grid_import_kwh=("grid_import_kwh", "sum"),
            grid_export_kwh=("grid_export_kwh", "sum"),
            carbon_levy=("carbon_levy", "sum"),
            carbon_credit=("carbon_credit", "sum"),
            net_carbon_transfer=("net_carbon_transfer", "sum"),
            waste_heat=("waste_heat", "sum"),
            max_abs_electricity_balance_residual=(
                "electricity_balance_residual",
                lambda values: float(np.abs(values).max()),
            ),
            peer_active_hours=("peer_trade_count", lambda values: int((values > 0).sum())),
        )
    )
    summary["average_peer_import_price"] = np.divide(
        summary["peer_purchase_cost"],
        summary["peer_import_kwh"],
        out=np.zeros(len(summary), dtype=np.float64),
        where=summary["peer_import_kwh"].to_numpy() > 1e-12,
    )
    summary["average_peer_export_price"] = np.divide(
        summary["peer_sales_revenue"],
        summary["peer_export_kwh"],
        out=np.zeros(len(summary), dtype=np.float64),
        where=summary["peer_export_kwh"].to_numpy() > 1e-12,
    )
    return summary


def summarize_system_days(agent_days):
    keys = [
        "preference",
        "policy_file",
        "pareto_index",
        "target_weight_cost",
        "target_weight_carbon",
        "weight_cost",
        "weight_carbon",
        "season",
    ]
    system = (
        agent_days.groupby(keys, as_index=False, sort=False)
        .agg(
            operating_cost=("operating_cost", "sum"),
            physical_cost_before_carbon_transfer=(
                "physical_cost_before_carbon_transfer",
                "sum",
            ),
            carbon_emissions=("carbon_emissions", "sum"),
            peer_trade_volume_kwh=("peer_import_kwh", "sum"),
            peer_import_kwh=("peer_import_kwh", "sum"),
            peer_export_kwh=("peer_export_kwh", "sum"),
            peer_purchase_cost=("peer_purchase_cost", "sum"),
            peer_sales_revenue=("peer_sales_revenue", "sum"),
            grid_import_kwh=("grid_import_kwh", "sum"),
            grid_export_kwh=("grid_export_kwh", "sum"),
            carbon_levy=("carbon_levy", "sum"),
            carbon_credit=("carbon_credit", "sum"),
            waste_heat=("waste_heat", "sum"),
            max_abs_electricity_balance_residual=(
                "max_abs_electricity_balance_residual",
                "max",
            ),
        )
    )
    system["peer_energy_balance_error_kwh"] = (
        system["peer_import_kwh"] - system["peer_export_kwh"]
    )
    system["peer_cash_balance_error"] = (
        system["peer_purchase_cost"] - system["peer_sales_revenue"]
    )
    system["carbon_dividend_balance_error"] = (
        system["carbon_levy"] - system["carbon_credit"]
    )
    return system


def validate_results(run_dir, detail, agent_days, system_days):
    strict_days = pd.read_csv(run_dir / "strict_test" / "season_detail.csv")
    validation_rows = []
    for row in system_days.itertuples(index=False):
        expected = strict_days[
            (strict_days["policy_file"] == row.policy_file)
            & (strict_days["season"] == row.season)
        ]
        if len(expected) != 1:
            raise RuntimeError(
                f"Missing strict-test row for {row.policy_file}/{row.season}."
            )
        expected_row = expected.iloc[0]
        validation_rows.append(
            {
                "preference": row.preference,
                "policy_file": row.policy_file,
                "season": row.season,
                "cost_error_vs_strict_test": row.operating_cost
                - float(expected_row["cost"]),
                "carbon_error_vs_strict_test": row.carbon_emissions
                - float(expected_row["carbon"]),
                "peer_energy_balance_error_kwh": row.peer_energy_balance_error_kwh,
                "peer_cash_balance_error": row.peer_cash_balance_error,
                "carbon_dividend_balance_error": row.carbon_dividend_balance_error,
                "waste_heat": row.waste_heat,
                "max_abs_electricity_balance_residual": (
                    row.max_abs_electricity_balance_residual
                ),
            }
        )

    validation = pd.DataFrame(validation_rows)
    hourly_peer = detail.groupby(
        ["preference", "season", "hour"], as_index=False
    ).agg(
        peer_balance=("net_peer_import_kwh", "sum"),
        levy=("carbon_levy", "sum"),
        credit=("carbon_credit", "sum"),
    )
    reconstructed_cost = (
        detail["physical_cost_before_carbon_transfer"]
        + detail["carbon_levy"]
        - detail["carbon_credit"]
    )
    preference_totals = (
        system_days.groupby("preference", as_index=True)[
            ["operating_cost", "carbon_emissions"]
        ]
        .sum()
        .reindex(PREFERENCE_ORDER)
    )
    agent_season = agent_days.pivot(
        index=["season", "agent"],
        columns="preference",
        values=["operating_cost", "carbon_emissions"],
    )
    cost_monotonic = (
        agent_season["operating_cost"]["cost_priority"]
        <= agent_season["operating_cost"]["balanced"]
    ) & (
        agent_season["operating_cost"]["balanced"]
        <= agent_season["operating_cost"]["carbon_priority"]
    )
    carbon_monotonic = (
        agent_season["carbon_emissions"]["carbon_priority"]
        <= agent_season["carbon_emissions"]["balanced"]
    ) & (
        agent_season["carbon_emissions"]["balanced"]
        <= agent_season["carbon_emissions"]["cost_priority"]
    )
    checks = {
        "rows_hourly_detail": int(len(detail)),
        "rows_agent_daily_summary": int(len(agent_days)),
        "rows_system_daily_summary": int(len(system_days)),
        "all_values_finite": bool(
            np.isfinite(
                detail.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
            ).all()
        ),
        "max_abs_cost_error_vs_strict_test": float(
            np.abs(validation["cost_error_vs_strict_test"]).max()
        ),
        "max_abs_carbon_error_vs_strict_test": float(
            np.abs(validation["carbon_error_vs_strict_test"]).max()
        ),
        "max_abs_hourly_peer_energy_balance_error_kwh": float(
            np.abs(hourly_peer["peer_balance"]).max()
        ),
        "max_abs_hourly_carbon_dividend_balance_error": float(
            np.abs(hourly_peer["levy"] - hourly_peer["credit"]).max()
        ),
        "max_abs_electricity_balance_residual": float(
            np.abs(detail["electricity_balance_residual"]).max()
        ),
        "max_abs_cost_reconstruction_error": float(
            np.abs(reconstructed_cost - detail["operating_cost"]).max()
        ),
        "total_waste_heat": float(detail["waste_heat"].sum()),
        "minimum_recorded_trade_quantity": float(
            detail[
                [
                    "peer_import_kwh",
                    "peer_export_kwh",
                    "grid_import_kwh",
                    "grid_export_kwh",
                ]
            ].min().min()
        ),
        "peer_market_total_volume_kwh": float(detail["peer_import_kwh"].sum()),
        "cost_priority_has_lowest_cost": bool(
            preference_totals.loc["cost_priority", "operating_cost"]
            <= preference_totals["operating_cost"].min() + 1e-9
        ),
        "carbon_priority_has_lowest_carbon": bool(
            preference_totals.loc["carbon_priority", "carbon_emissions"]
            <= preference_totals["carbon_emissions"].min() + 1e-9
        ),
        "balanced_cost_between_extremes": bool(
            preference_totals.loc["cost_priority", "operating_cost"]
            <= preference_totals.loc["balanced", "operating_cost"]
            <= preference_totals.loc["carbon_priority", "operating_cost"]
        ),
        "balanced_carbon_between_extremes": bool(
            preference_totals.loc["carbon_priority", "carbon_emissions"]
            <= preference_totals.loc["balanced", "carbon_emissions"]
            <= preference_totals.loc["cost_priority", "carbon_emissions"]
        ),
        "agent_season_combinations": int(len(agent_season)),
        "agent_season_cost_monotonic_count": int(cost_monotonic.sum()),
        "agent_season_carbon_monotonic_count": int(carbon_monotonic.sum()),
        "all_agent_season_preferences_monotonic": bool(
            cost_monotonic.all() and carbon_monotonic.all()
        ),
    }
    tolerances_pass = (
        checks["all_values_finite"]
        and checks["max_abs_cost_error_vs_strict_test"] <= 1e-6
        and checks["max_abs_carbon_error_vs_strict_test"] <= 1e-6
        and checks["max_abs_hourly_peer_energy_balance_error_kwh"] <= 1e-8
        and checks["max_abs_hourly_carbon_dividend_balance_error"] <= 1e-8
        and checks["max_abs_electricity_balance_residual"] <= 1e-8
        and checks["max_abs_cost_reconstruction_error"] <= 1e-8
        and abs(checks["total_waste_heat"]) <= 1e-8
        and checks["minimum_recorded_trade_quantity"] >= -1e-12
        and checks["peer_market_total_volume_kwh"] > 0.0
        and checks["cost_priority_has_lowest_cost"]
        and checks["carbon_priority_has_lowest_carbon"]
        and checks["balanced_cost_between_extremes"]
        and checks["balanced_carbon_between_extremes"]
    )
    checks["basic_function_validation_passed"] = bool(tolerances_pass)
    return validation, checks


def aggregate_preferences(frame):
    keys = [
        "preference",
        "policy_file",
        "pareto_index",
        "target_weight_cost",
        "target_weight_carbon",
        "weight_cost",
        "weight_carbon",
        "agent",
    ]
    return (
        frame.groupby(keys, as_index=False, sort=False)
        .agg(
            operating_cost=("operating_cost", "sum"),
            physical_cost_before_carbon_transfer=(
                "physical_cost_before_carbon_transfer",
                "sum",
            ),
            carbon_emissions=("carbon_emissions", "sum"),
            net_peer_import_kwh=("net_peer_import_kwh", "sum"),
            peer_import_kwh=("peer_import_kwh", "sum"),
            peer_export_kwh=("peer_export_kwh", "sum"),
            grid_import_kwh=("grid_import_kwh", "sum"),
            grid_export_kwh=("grid_export_kwh", "sum"),
            carbon_levy=("carbon_levy", "sum"),
            carbon_credit=("carbon_credit", "sum"),
            waste_heat=("waste_heat", "sum"),
        )
    )


def localize_table(frame):
    localized = frame.copy()
    for column, names in CATEGORY_NAMES_CN.items():
        if column in localized.columns:
            localized[column] = localized[column].map(
                lambda value: names.get(value, value)
            )
    return localized.rename(columns=COLUMN_NAMES_CN)


def ensure_output_files_writable(output_dir, filenames):
    locked = []
    for filename in filenames:
        path = output_dir / filename
        if not path.exists():
            continue
        try:
            with path.open("a", encoding="utf-8"):
                pass
        except PermissionError:
            locked.append(str(path))
    if locked:
        raise PermissionError(
            "Close the open result table(s) before regenerating: " + ", ".join(locked)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize current-main dispatch by preference, season, and agent."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_terminal_test_row()
    selected = select_weight_matched_policies(run_dir)
    rows = []
    for preference in PREFERENCE_ORDER:
        policy = selected[preference]
        for season, start in SEASONS.items():
            rows.extend(simulate_day(run_dir, preference, policy, season, start))

    detail = pd.DataFrame(rows)
    agent_days = summarize_agent_days(detail)
    system_days = summarize_system_days(agent_days)
    preference_agents = aggregate_preferences(agent_days)
    system_preferences = (
        system_days.groupby(
            [
                "preference",
                "policy_file",
                "pareto_index",
                "target_weight_cost",
                "target_weight_carbon",
                "weight_cost",
                "weight_carbon",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            operating_cost=("operating_cost", "sum"),
            physical_cost_before_carbon_transfer=(
                "physical_cost_before_carbon_transfer",
                "sum",
            ),
            carbon_emissions=("carbon_emissions", "sum"),
            peer_trade_volume_kwh=("peer_trade_volume_kwh", "sum"),
            grid_import_kwh=("grid_import_kwh", "sum"),
            grid_export_kwh=("grid_export_kwh", "sum"),
            carbon_levy=("carbon_levy", "sum"),
            carbon_credit=("carbon_credit", "sum"),
            waste_heat=("waste_heat", "sum"),
        )
    )
    validation_rows, validation = validate_results(
        run_dir, detail, agent_days, system_days
    )

    selected_rows = [
        {"preference": preference, **selected[preference]}
        for preference in PREFERENCE_ORDER
    ]
    outputs = {
        "selected_policies.csv": pd.DataFrame(selected_rows),
        "hourly_detail.csv": detail,
        "daily_agent_summary.csv": agent_days,
        "preference_agent_summary.csv": preference_agents,
        "daily_system_summary.csv": system_days,
        "preference_system_summary.csv": system_preferences,
        "validation_detail.csv": validation_rows,
    }
    ensure_output_files_writable(output_dir, [*outputs, "validation.json"])
    for filename, frame in outputs.items():
        localize_table(frame).to_csv(
            output_dir / filename, index=False, encoding="utf-8-sig"
        )
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print(localize_table(system_preferences).to_string(index=False))
    print(json.dumps(validation, indent=2))
    print(f"Saved agent statistics to: {output_dir}")
    if not validation["basic_function_validation_passed"]:
        raise RuntimeError("Basic-function validation failed; inspect validation.json.")


if __name__ == "__main__":
    main()
