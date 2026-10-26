import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs import ies_env
from src.envs.ies_env import env as IESEnv


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

METHODS = [
    {
        "method_key": "standard_prediction",
        "method": "Standard CDA + Prediction",
        "run_dir": PROJECT_ROOT / "outputs" / "runs" / "mappo_raw_reward_full",
        "market_mechanism": "standard",
        "cda_carbon_price": 0.0,
    },
    {
        "method_key": "carbon_cda_only",
        "method": "Carbon CDA Only",
        "run_dir": PROJECT_ROOT / "outputs" / "runs" / "mappo_carbon_aware_cda_full",
        "market_mechanism": "carbon_aware",
        "cda_carbon_price": 0.05,
    },
    {
        "method_key": "standard_contribution",
        "method": "Standard CDA + Contribution",
        "run_dir": PROJECT_ROOT / "outputs" / "runs" / "mappo_standard_cda_contribution_full",
        "market_mechanism": "standard",
        "cda_carbon_price": 0.0,
    },
    {
        "method_key": "carbon_contribution",
        "method": "Carbon CDA + Contribution",
        "run_dir": PROJECT_ROOT / "outputs" / "runs" / "mappo_carbon_cda_contribution_full",
        "market_mechanism": "carbon_aware",
        "cda_carbon_price": 0.05,
    },
]

SEASONS = {
    "winter": 0,
    "spring": 24,
    "summer": 48,
    "autumn": 72,
}

AGENT_NAMES = {
    1: "residential",
    2: "commercial",
    3: "industry",
}

CARBON_ATTRS = {
    1: "carbon_intensity_residential",
    2: "carbon_intensity_commercial",
    3: "carbon_intensity_industry",
}


def ensure_test_arrays_have_terminal_row():
    for name in [
        "state_test_residential",
        "state_test_commercial",
        "state_test_industry",
    ]:
        arr = getattr(ies_env, name)
        if arr.shape[0] == 96:
            setattr(ies_env, name, np.vstack([arr, arr[-1:]]))


def load_objectives(run_dir):
    objectives = np.loadtxt(run_dir / "final" / "objs.txt", delimiter=",")
    return np.atleast_2d(objectives)


def representative_policies(run_dir):
    objectives = load_objectives(run_dir)
    mins = objectives.min(axis=0)
    ranges = np.where(objectives.max(axis=0) - mins <= 1e-12, 1.0, objectives.max(axis=0) - mins)
    normalized = (objectives - mins) / ranges
    balanced_target = np.array([0.5, 0.5])
    indices = {
        "cost_priority": int(np.argmin(objectives[:, 0])),
        "carbon_priority": int(np.argmin(objectives[:, 1])),
        "balanced": int(np.argmin(np.linalg.norm(normalized - balanced_target, axis=1))),
    }
    return {
        preference: {
            "policy_file": f"EP_policy_{idx}.pt",
            "pareto_index": idx,
            "cost": float(objectives[idx, 0]),
            "carbon": float(objectives[idx, 1]),
        }
        for preference, idx in indices.items()
    }


def load_actor(run_dir, policy_file):
    actor_dict = torch.load(run_dir / "final" / policy_file, map_location="cpu")
    for actor in actor_dict.values():
        actor.eval()
    return actor_dict


def make_initial_state(environment, season_start):
    state_res = np.append(
        ies_env.state_test_residential[season_start],
        np.array([environment.residential_ESS.reset(), environment.residential_TES.reset()]),
    ).reshape(1, -1)
    state_com = np.append(
        ies_env.state_test_commercial[season_start],
        np.array([environment.commercial_ESS.reset(), environment.commercial_TES.reset()]),
    ).reshape(1, -1)
    state_ind = np.append(
        ies_env.state_test_industry[season_start],
        np.array([environment.industry_ESS.reset(), environment.industry_TES.reset()]),
    ).reshape(1, -1)
    return state_res, state_com, state_ind


def choose_action(actor_dict, state_res, state_com, state_ind):
    with torch.no_grad():
        state_res_t = torch.tensor(state_res, dtype=torch.float32).squeeze()
        state_com_t = torch.tensor(state_com, dtype=torch.float32).squeeze()
        state_ind_t = torch.tensor(state_ind, dtype=torch.float32).squeeze()

        _, mean_res, _, _ = actor_dict[0](state_res_t, evaluation=True)
        _, mean_com, _, _ = actor_dict[1](state_com_t, evaluation=True)
        _, mean_ind, _, _ = actor_dict[2](state_ind_t, evaluation=True)

        action_res = torch.tanh(mean_res).reshape(1, -1)
        action_com = torch.tanh(mean_com).reshape(1, -1)
        action_ind = torch.tanh(mean_ind)
        action_ind = torch.cat([action_ind, torch.tensor([0.0])]).reshape(1, -1)
        action = torch.cat((action_res, action_com, action_ind), dim=0)
    return action.numpy()


def summarize_step(environment):
    records = list(getattr(environment, "trade_record_e", []))
    seller_intensity = {
        agent_id: float(getattr(environment, attr_name))
        for agent_id, attr_name in CARBON_ATTRS.items()
    }
    median_intensity = float(np.median(list(seller_intensity.values())))

    peer_amount = sum(float(record[2]) for record in records)
    peer_value = sum(float(record[2]) * float(record[3]) for record in records)
    low_carbon_amount = sum(
        float(record[2])
        for record in records
        if seller_intensity[int(record[1])] <= median_intensity + 1e-12
    )
    weighted_seller_carbon = (
        sum(float(record[2]) * seller_intensity[int(record[1])] for record in records) / peer_amount
        if peer_amount > 1e-12
        else 0.0
    )
    avg_trade_price = peer_value / peer_amount if peer_amount > 1e-12 else 0.0

    grid_buy = (
        environment.buy_from_grid_residential
        + environment.buy_from_grid_commercial
        + environment.buy_from_grid_industry
    )
    grid_sell = (
        environment.sell_to_grid_residential
        + environment.sell_to_grid_commercial
        + environment.sell_to_grid_industry
    )
    proposed_abs = (
        abs(environment.trade_e_residential)
        + abs(environment.trade_e_commercial)
        + abs(environment.trade_e_industry)
    )

    return {
        "peer_trade_amount": peer_amount,
        "peer_trade_value": peer_value,
        "avg_trade_price": avg_trade_price,
        "trade_count": len(records),
        "low_carbon_trade_amount": low_carbon_amount,
        "low_carbon_trade_share": low_carbon_amount / peer_amount if peer_amount > 1e-12 else 0.0,
        "weighted_seller_carbon_intensity": weighted_seller_carbon,
        "grid_buy": float(grid_buy),
        "grid_sell": float(grid_sell),
        "proposed_trade_abs": float(proposed_abs),
        "matched_trade_share": peer_amount * 2.0 / proposed_abs if proposed_abs > 1e-12 else 0.0,
        "current_cda_carbon_price": float(environment.current_cda_carbon_price),
    }


def simulate(method, preference, policy_info, season, season_start):
    environment = IESEnv(
        market_mechanism=method["market_mechanism"],
        cda_carbon_price=method["cda_carbon_price"],
    )
    actor_dict = load_actor(method["run_dir"], policy_info["policy_file"])
    state_res, state_com, state_ind = make_initial_state(environment, season_start)
    rows = []
    for hour in range(24):
        absolute_step = season_start + hour
        state_np = np.concatenate((state_res, state_com, state_ind), axis=0)
        action = choose_action(actor_dict, state_res, state_com, state_ind)
        state_res, state_com, state_ind, done, reward, *_ = environment.step(
            action, state_np, absolute_step, flag=1
        )
        row = summarize_step(environment)
        row.update(
            {
                "method_key": method["method_key"],
                "method": method["method"],
                "preference": preference,
                "policy_file": policy_info["policy_file"],
                "pareto_index": policy_info["pareto_index"],
                "policy_cost": policy_info["cost"],
                "policy_carbon": policy_info["carbon"],
                "season": season,
                "hour": hour,
                "step_cost": float(reward[0]),
                "step_carbon": float(reward[1]),
            }
        )
        rows.append(row)
        if done and hour < 23:
            raise RuntimeError(f"{method['method']} {preference} {season} ended early at hour {hour}.")
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_test_arrays_have_terminal_row()

    all_rows = []
    selected_rows = []
    for method in METHODS:
        selected = representative_policies(method["run_dir"])
        for preference, policy_info in selected.items():
            selected_rows.append({"method_key": method["method_key"], "method": method["method"], "preference": preference, **policy_info})
            for season, start in SEASONS.items():
                all_rows.extend(simulate(method, preference, policy_info, season, start))

    detail = pd.DataFrame(all_rows)
    detail_path = OUTPUT_DIR / "trading_behavior_detailed.csv"
    detail.to_csv(detail_path, index=False)

    group_cols = ["method_key", "method", "preference"]
    summary = (
        detail.groupby(group_cols, as_index=False)
        .agg(
            total_peer_trade_amount=("peer_trade_amount", "sum"),
            total_peer_trade_value=("peer_trade_value", "sum"),
            avg_trade_price=("avg_trade_price", "mean"),
            avg_trade_count=("trade_count", "mean"),
            low_carbon_trade_share=("low_carbon_trade_share", "mean"),
            weighted_seller_carbon_intensity=("weighted_seller_carbon_intensity", "mean"),
            total_grid_buy=("grid_buy", "sum"),
            total_grid_sell=("grid_sell", "sum"),
            matched_trade_share=("matched_trade_share", "mean"),
            avg_cda_carbon_price=("current_cda_carbon_price", "mean"),
            total_step_cost=("step_cost", "sum"),
            total_step_carbon=("step_carbon", "sum"),
        )
    )
    summary_path = OUTPUT_DIR / "trading_behavior_summary.csv"
    selected_path = OUTPUT_DIR / "trading_behavior_selected_policies.csv"
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(selected_rows).to_csv(selected_path, index=False)

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
    method_order = [item["method_key"] for item in METHODS]
    method_labels = {item["method_key"]: item["method"] for item in METHODS}
    plot_data = summary.groupby("method_key", as_index=False).mean(numeric_only=True)
    plot_data["method_key"] = pd.Categorical(plot_data["method_key"], categories=method_order, ordered=True)
    plot_data = plot_data.sort_values("method_key")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), constrained_layout=True)
    specs = [
        ("total_peer_trade_amount", "Peer Trade Amount"),
        ("low_carbon_trade_share", "Low-carbon Trade Share"),
        ("weighted_seller_carbon_intensity", "Seller Carbon Intensity"),
    ]
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]
    x = np.arange(len(plot_data))
    xlabels = [method_labels[key] for key in plot_data["method_key"].astype(str)]
    for ax, (metric, ylabel) in zip(axes, specs):
        ax.bar(x, plot_data[metric], color=colors, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.savefig(OUTPUT_DIR / "trading_behavior_summary.png", bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
