from collections import defaultdict

import numpy as np
import pandas as pd

from analyze_cda_isolation_trading import (
    MECHANISMS,
    OUTPUT_DIR,
    PREFERENCES,
    SEEDS,
    SOURCES,
    load_actor,
    relative_change,
    representative_policies,
    run_dir,
)
from analyze_current_carbon_cda_trading_3seed import choose_action, step_metrics
from src.envs import ies_env
from src.envs.ies_env import env as IESEnv


CARBON_PRICE = 0.02


def initial_train_state(environment, start):
    states = []
    specifications = (
        (ies_env.state_train_residential, environment.residential_ESS, environment.residential_TES),
        (ies_env.state_train_commercial, environment.commercial_ESS, environment.commercial_TES),
        (ies_env.state_train_industry, environment.industry_ESS, environment.industry_TES),
    )
    for data, ess, tes in specifications:
        states.append(np.append(data[start], [ess.reset(), tes.reset()]).reshape(1, -1))
    return states


def simulate_train_day(actors, mechanism, start):
    environment = IESEnv(
        market_mechanism=mechanism,
        cda_carbon_price=CARBON_PRICE if mechanism != "standard" else 0.0,
    )
    states = initial_train_state(environment, start)
    rows = []
    for hour in range(24):
        action = choose_action(actors, states)
        state_matrix = np.concatenate(states, axis=0)
        next_res, next_com, next_ind, done, reward_res, reward_com, reward_ind = environment.step(
            action, state_matrix, start + hour, flag=0
        )
        metrics = step_metrics(environment)
        reward = np.asarray(reward_res) + np.asarray(reward_com) + np.asarray(reward_ind)
        metrics["step_cost"] = float(reward[0])
        metrics["step_carbon"] = float(reward[1])
        rows.append(metrics)
        states = [next_res, next_com, next_ind]
        if done and hour < 23:
            raise RuntimeError(f"Training-data episode ended early: start={start}, hour={hour}")
    return rows


def aggregate_rows(rows):
    peer = sum(row["peer_trade_amount"] for row in rows)
    return {
        "peer_trade_amount": peer,
        "seller_carbon_intensity": (
            sum(row["seller_carbon_amount"] for row in rows) / peer if peer > 1e-12 else 0.0
        ),
        "grid_buy": sum(row["grid_buy"] for row in rows),
        "total_cost": sum(row["step_cost"] for row in rows),
        "total_carbon": sum(row["step_carbon"] for row in rows),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_length = len(ies_env.state_train_residential)
    starts = range(0, train_length, 24)
    accumulators = defaultdict(
        lambda: {
            "changed_hours": 0,
            "total_hours": 0,
            "standard_peer_trade_amount": 0.0,
            "value_peer_trade_amount": 0.0,
            "standard_grid_buy": 0.0,
            "value_grid_buy": 0.0,
            "standard_total_cost": 0.0,
            "value_total_cost": 0.0,
            "standard_total_carbon": 0.0,
            "value_total_carbon": 0.0,
            "standard_seller_carbon_amount": 0.0,
            "value_seller_carbon_amount": 0.0,
        }
    )

    for source_key, _, base_name in SOURCES:
        for seed in SEEDS:
            path = run_dir(base_name, seed)
            selected = representative_policies(path)
            for preference in PREFERENCES:
                actors = load_actor(path, selected[preference]["policy_file"])
                for start in starts:
                    results = {
                        mechanism: simulate_train_day(actors, mechanism, start)
                        for mechanism in MECHANISMS
                    }
                    standard = results["standard"]
                    standard_aggregate = aggregate_rows(standard)
                    standard_seller_carbon_amount = sum(
                        row["seller_carbon_amount"] for row in standard
                    )
                    for mechanism in MECHANISMS[1:]:
                        selected_rows = results[mechanism]
                        selected_aggregate = aggregate_rows(selected_rows)
                        key = (source_key, seed, preference, mechanism)
                        accumulator = accumulators[key]
                        for standard_row, selected_row in zip(standard, selected_rows):
                            allocation_changed = max(
                                abs(selected_row[column] - standard_row[column])
                                for column in (
                                    "peer_trade_amount",
                                    "seller_1_amount",
                                    "seller_2_amount",
                                    "seller_3_amount",
                                )
                            ) > 1e-7
                            accumulator["changed_hours"] += int(allocation_changed)
                            accumulator["total_hours"] += 1
                        for metric in (
                            "peer_trade_amount",
                            "grid_buy",
                            "total_cost",
                            "total_carbon",
                        ):
                            accumulator[f"standard_{metric}"] += standard_aggregate[metric]
                            accumulator[f"value_{metric}"] += selected_aggregate[metric]
                        accumulator["standard_seller_carbon_amount"] += standard_seller_carbon_amount
                        accumulator["value_seller_carbon_amount"] += sum(
                            row["seller_carbon_amount"] for row in selected_rows
                        )

    rows = []
    for (source_key, seed, preference, mechanism), values in accumulators.items():
        standard_peer = values["standard_peer_trade_amount"]
        value_peer = values["value_peer_trade_amount"]
        standard_intensity = (
            values["standard_seller_carbon_amount"] / standard_peer
            if standard_peer > 1e-12
            else 0.0
        )
        value_intensity = (
            values["value_seller_carbon_amount"] / value_peer if value_peer > 1e-12 else 0.0
        )
        row = {
            "source_method": source_key,
            "seed": seed,
            "preference": preference,
            "mechanism": mechanism,
            "changed_hours": values["changed_hours"],
            "total_hours": values["total_hours"],
            "allocation_change_rate": values["changed_hours"] / values["total_hours"],
            "relative_delta_seller_carbon_intensity_pct": relative_change(
                value_intensity, standard_intensity
            ),
        }
        for metric in ("peer_trade_amount", "grid_buy", "total_cost", "total_carbon"):
            row[f"relative_delta_{metric}_pct"] = relative_change(
                values[f"value_{metric}"], values[f"standard_{metric}"]
            )
        rows.append(row)

    detail = pd.DataFrame(rows)
    summary = detail.groupby("mechanism", as_index=False).agg(
        changed_hours=("changed_hours", "sum"),
        total_hours=("total_hours", "sum"),
        allocation_change_rate=("allocation_change_rate", "mean"),
        peer_trade_change_pct=("relative_delta_peer_trade_amount_pct", "mean"),
        seller_carbon_intensity_change_pct=(
            "relative_delta_seller_carbon_intensity_pct",
            "mean",
        ),
        grid_buy_change_pct=("relative_delta_grid_buy_pct", "mean"),
        total_cost_change_pct=("relative_delta_total_cost_pct", "mean"),
        total_carbon_change_pct=("relative_delta_total_carbon_pct", "mean"),
    )
    detail_path = OUTPUT_DIR / "cda_training_distribution_activation_detail.csv"
    summary_path = OUTPUT_DIR / "cda_training_distribution_activation_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
