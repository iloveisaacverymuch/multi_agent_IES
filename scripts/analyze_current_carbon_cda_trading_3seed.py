from pathlib import Path
import sys

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


RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
SEEDS = [11, 22, 44]
SEASONS = {"winter": 0, "spring": 24, "summer": 48, "autumn": 72}
PREFERENCES = ("cost_priority", "balanced", "carbon_priority")
CARBON_PRICE = 0.02
AGENT_IDS = (1, 2, 3)
CARBON_ATTRS = {
    1: "carbon_intensity_residential",
    2: "carbon_intensity_commercial",
    3: "carbon_intensity_industry",
}


def run_dir(method, seed):
    if method == "standard":
        if seed == 44:
            name = "standard_cda_ensemble_uncertainty_capped32_kappa_0_20"
        else:
            name = f"standard_ensemble_capped32_kappa_0_20_price_0_02_seed_{seed}"
    else:
        base = "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02"
        name = base if seed == 44 else f"{base}_seed_{seed}"
    return RUN_ROOT / name


def ensure_test_arrays_have_terminal_row():
    for name in ("state_test_residential", "state_test_commercial", "state_test_industry"):
        values = getattr(ies_env, name)
        if values.shape[0] == 96:
            setattr(ies_env, name, np.vstack([values, values[-1:]]))


def load_objectives(path):
    return np.atleast_2d(np.loadtxt(path / "final" / "objs.txt", delimiter=","))


def representative_policies(path):
    objectives = load_objectives(path)
    minimum = objectives.min(axis=0)
    ranges = np.maximum(objectives.max(axis=0) - minimum, 1e-12)
    normalized = (objectives - minimum) / ranges
    indices = {
        "cost_priority": int(np.argmin(objectives[:, 0])),
        "carbon_priority": int(np.argmin(objectives[:, 1])),
        "balanced": int(np.argmin(np.linalg.norm(normalized - np.array([0.5, 0.5]), axis=1))),
    }
    return {
        preference: {
            "pareto_index": index,
            "policy_file": f"EP_policy_{index}.pt",
            "policy_cost": float(objectives[index, 0]),
            "policy_carbon": float(objectives[index, 1]),
        }
        for preference, index in indices.items()
    }


def load_actor(path, policy_file):
    policy_path = path / "final" / policy_file
    try:
        actors = torch.load(policy_path, map_location="cpu", weights_only=False)
    except TypeError:
        actors = torch.load(policy_path, map_location="cpu")
    for actor in actors.values():
        actor.eval()
    return actors


def initial_state(environment, start):
    states = []
    specifications = (
        (ies_env.state_test_residential, environment.residential_ESS, environment.residential_TES),
        (ies_env.state_test_commercial, environment.commercial_ESS, environment.commercial_TES),
        (ies_env.state_test_industry, environment.industry_ESS, environment.industry_TES),
    )
    for data, ess, tes in specifications:
        states.append(np.append(data[start], [ess.reset(), tes.reset()]).reshape(1, -1))
    return states


def choose_action(actors, states):
    actions = []
    with torch.no_grad():
        for agent_id, state in enumerate(states):
            tensor = torch.tensor(state, dtype=torch.float32).squeeze()
            _, mean, _, _ = actors[agent_id](tensor, evaluation=True)
            action = torch.tanh(mean)
            if agent_id == 2:
                action = torch.cat([action, torch.tensor([0.0])])
            actions.append(action.reshape(1, -1))
    return torch.cat(actions, dim=0).numpy()


def step_metrics(environment):
    records = list(environment.trade_record_e)
    intensities = {
        agent_id: float(getattr(environment, attribute))
        for agent_id, attribute in CARBON_ATTRS.items()
    }
    proposed = {
        1: float(environment.trade_e_residential),
        2: float(environment.trade_e_commercial),
        3: float(environment.trade_e_industry),
    }
    sellers = [agent_id for agent_id, amount in proposed.items() if amount > 1e-9]
    cleanest = min((intensities[agent_id] for agent_id in sellers), default=0.0)

    peer_amount = sum(float(record[2]) for record in records)
    peer_value = sum(float(record[2]) * float(record[3]) for record in records)
    seller_carbon_amount = sum(
        float(record[2]) * intensities[int(record[1])] for record in records
    )
    cleanest_amount = sum(
        float(record[2])
        for record in records
        if abs(intensities[int(record[1])] - cleanest) <= 1e-9
    )
    seller_amounts = {
        agent_id: sum(float(record[2]) for record in records if int(record[1]) == agent_id)
        for agent_id in AGENT_IDS
    }
    grid_buy = sum(
        float(getattr(environment, f"buy_from_grid_{name}"))
        for name in ("residential", "commercial", "industry")
    )
    grid_sell = sum(
        float(getattr(environment, f"sell_to_grid_{name}"))
        for name in ("residential", "commercial", "industry")
    )
    waste_heat = sum(
        float(getattr(environment, f"waste_h_{name}"))
        for name in ("residential", "commercial", "industry")
    )
    proposed_abs = sum(abs(amount) for amount in proposed.values())
    carbon_levy = getattr(environment, "carbon_levy", {})
    carbon_credit = getattr(environment, "carbon_credit", {})
    avoided_basis = getattr(environment, "avoided_carbon_credit_basis", {})
    return {
        "peer_trade_amount": peer_amount,
        "peer_trade_value": peer_value,
        "seller_carbon_amount": seller_carbon_amount,
        "cleanest_seller_amount": cleanest_amount,
        "seller_1_amount": seller_amounts[1],
        "seller_2_amount": seller_amounts[2],
        "seller_3_amount": seller_amounts[3],
        "trade_count": len(records),
        "grid_buy": grid_buy,
        "grid_sell": grid_sell,
        "waste_heat": waste_heat,
        "proposed_trade_abs": proposed_abs,
        "carbon_price": float(environment.current_cda_carbon_price),
        "carbon_dividend_fund": float(getattr(environment, "carbon_dividend_fund", 0.0)),
        "carbon_dividend_balance_error": float(
            getattr(environment, "carbon_dividend_balance_error", 0.0)
        ),
        "carbon_levy_1": float(getattr(environment, "carbon_levy", {}).get(1, 0.0)),
        "carbon_levy_2": float(getattr(environment, "carbon_levy", {}).get(2, 0.0)),
        "carbon_levy_3": float(getattr(environment, "carbon_levy", {}).get(3, 0.0)),
        "carbon_credit_1": float(getattr(environment, "carbon_credit", {}).get(1, 0.0)),
        "carbon_credit_2": float(getattr(environment, "carbon_credit", {}).get(2, 0.0)),
        "carbon_credit_3": float(getattr(environment, "carbon_credit", {}).get(3, 0.0)),
        "avoided_carbon_credit_basis": float(sum(avoided_basis.values())),
        "carbon_transfer_abs": float(
            sum(
                abs(carbon_levy.get(agent_id, 0.0) - carbon_credit.get(agent_id, 0.0))
                for agent_id in AGENT_IDS
            )
        ),
    }


def simulate(
    actors,
    mechanism,
    seed,
    preference,
    policy_info,
    season,
    start,
    experiment,
    carbon_price=None,
):
    configured_carbon_price = CARBON_PRICE if carbon_price is None else float(carbon_price)
    environment = IESEnv(
        market_mechanism=mechanism,
        cda_carbon_price=(
            configured_carbon_price
            if mechanism in (
                "carbon_aware",
                "carbon_aware_v2",
                "carbon_dividend_v3",
                "carbon_dividend_equal_v3",
            )
            else 0.0
        ),
    )
    states = initial_state(environment, start)
    rows = []
    for hour in range(24):
        action = choose_action(actors, states)
        state_matrix = np.concatenate(states, axis=0)
        next_res, next_com, next_ind, done, reward_res, reward_com, reward_ind = environment.step(
            action, state_matrix, start + hour, flag=1
        )
        metrics = step_metrics(environment)
        total_reward = np.asarray(reward_res) + np.asarray(reward_com) + np.asarray(reward_ind)
        metrics.update(
            {
                "experiment": experiment,
                "mechanism": mechanism,
                "seed": seed,
                "preference": preference,
                "season": season,
                "hour": hour,
                **policy_info,
                "step_cost": float(total_reward[0]),
                "step_carbon": float(total_reward[1]),
                "step_cost_1": float(np.asarray(reward_res)[0]),
                "step_cost_2": float(np.asarray(reward_com)[0]),
                "step_cost_3": float(np.asarray(reward_ind)[0]),
                "step_carbon_1": float(np.asarray(reward_res)[1]),
                "step_carbon_2": float(np.asarray(reward_com)[1]),
                "step_carbon_3": float(np.asarray(reward_ind)[1]),
            }
        )
        rows.append(metrics)
        states = [next_res, next_com, next_ind]
        if done and hour < 23:
            raise RuntimeError(f"Episode ended early: seed={seed}, {preference}, {season}, hour={hour}")
    return rows


def episode_summary(detail):
    groups = [
        "experiment",
        "mechanism",
        "seed",
        "preference",
        "season",
        "pareto_index",
        "policy_file",
        "policy_cost",
        "policy_carbon",
    ]
    summary = detail.groupby(groups, as_index=False).agg(
        peer_trade_amount=("peer_trade_amount", "sum"),
        peer_trade_value=("peer_trade_value", "sum"),
        seller_carbon_amount=("seller_carbon_amount", "sum"),
        cleanest_seller_amount=("cleanest_seller_amount", "sum"),
        seller_1_amount=("seller_1_amount", "sum"),
        seller_2_amount=("seller_2_amount", "sum"),
        seller_3_amount=("seller_3_amount", "sum"),
        trade_count=("trade_count", "sum"),
        grid_buy=("grid_buy", "sum"),
        grid_sell=("grid_sell", "sum"),
        waste_heat=("waste_heat", "sum"),
        proposed_trade_abs=("proposed_trade_abs", "sum"),
        total_cost=("step_cost", "sum"),
        total_carbon=("step_carbon", "sum"),
    )
    denominator = summary["peer_trade_amount"].replace(0.0, np.nan)
    summary["average_trade_price"] = summary["peer_trade_value"] / denominator
    summary["weighted_seller_carbon_intensity"] = summary["seller_carbon_amount"] / denominator
    summary["cleanest_seller_share"] = summary["cleanest_seller_amount"] / denominator
    summary["matched_trade_share"] = 2.0 * summary["peer_trade_amount"] / summary[
        "proposed_trade_abs"
    ].replace(0.0, np.nan)
    return summary.fillna(0.0)


def aggregate_by_seed(episodes):
    sums = episodes.groupby(["experiment", "mechanism", "seed"], as_index=False).agg(
        peer_trade_amount=("peer_trade_amount", "sum"),
        peer_trade_value=("peer_trade_value", "sum"),
        seller_carbon_amount=("seller_carbon_amount", "sum"),
        cleanest_seller_amount=("cleanest_seller_amount", "sum"),
        grid_buy=("grid_buy", "sum"),
        grid_sell=("grid_sell", "sum"),
        proposed_trade_abs=("proposed_trade_abs", "sum"),
        total_cost=("total_cost", "sum"),
        total_carbon=("total_carbon", "sum"),
        waste_heat=("waste_heat", "sum"),
    )
    denominator = sums["peer_trade_amount"].replace(0.0, np.nan)
    sums["weighted_seller_carbon_intensity"] = sums["seller_carbon_amount"] / denominator
    sums["cleanest_seller_share"] = sums["cleanest_seller_amount"] / denominator
    sums["average_trade_price"] = sums["peer_trade_value"] / denominator
    sums["matched_trade_share"] = 2.0 * sums["peer_trade_amount"] / sums[
        "proposed_trade_abs"
    ].replace(0.0, np.nan)
    return sums.fillna(0.0)


def paired_cross_evaluation(detail, seed_summary):
    controlled = detail[detail["experiment"] == "controlled_main_policy"]
    index = ["seed", "preference", "season", "hour"]
    wide = controlled.pivot(index=index, columns="mechanism")
    paired = pd.DataFrame(index=wide.index).reset_index()
    for metric in (
        "peer_trade_amount",
        "seller_carbon_amount",
        "cleanest_seller_amount",
        "seller_1_amount",
        "seller_2_amount",
        "seller_3_amount",
        "grid_buy",
        "grid_sell",
        "step_cost",
        "step_carbon",
    ):
        standard = wide[(metric, "standard")].to_numpy()
        carbon = wide[(metric, "carbon_aware")].to_numpy()
        paired[f"standard_{metric}"] = standard
        paired[f"carbon_{metric}"] = carbon
        paired[f"delta_{metric}"] = carbon - standard
    change_columns = [
        "delta_peer_trade_amount",
        "delta_seller_1_amount",
        "delta_seller_2_amount",
        "delta_seller_3_amount",
    ]
    paired["allocation_changed"] = paired[change_columns].abs().max(axis=1) > 1e-7

    controlled_seed = seed_summary[seed_summary["experiment"] == "controlled_main_policy"]
    metric_columns = [
        "peer_trade_amount",
        "weighted_seller_carbon_intensity",
        "cleanest_seller_share",
        "grid_buy",
        "grid_sell",
        "total_cost",
        "total_carbon",
        "matched_trade_share",
    ]
    standard = controlled_seed[controlled_seed["mechanism"] == "standard"].set_index("seed")
    carbon = controlled_seed[controlled_seed["mechanism"] == "carbon_aware"].set_index("seed")
    differences = []
    activation = paired.groupby("seed")["allocation_changed"].mean()
    for seed in SEEDS:
        row = {"seed": seed, "allocation_change_rate": float(activation.loc[seed])}
        for metric in metric_columns:
            baseline = float(standard.loc[seed, metric])
            value = float(carbon.loc[seed, metric])
            row[f"standard_{metric}"] = baseline
            row[f"carbon_{metric}"] = value
            row[f"delta_{metric}"] = value - baseline
            row[f"relative_delta_{metric}_pct"] = (
                100.0 * (value - baseline) / abs(baseline) if abs(baseline) > 1e-12 else 0.0
            )
        differences.append(row)
    return paired, pd.DataFrame(differences)


def plot_controlled(seed_summary):
    selected = seed_summary[seed_summary["experiment"] == "controlled_main_policy"]
    mechanisms = ["standard", "carbon_aware"]
    labels = ["Standard CDA", "Carbon-aware CDA"]
    colors = ["#4C78A8", "#59A14F"]
    specifications = [
        ("weighted_seller_carbon_intensity", "Seller carbon intensity"),
        ("cleanest_seller_share", "Cleanest-seller share"),
        ("grid_buy", "Grid imports (kWh)"),
        ("total_carbon", "Total carbon emissions (kg)"),
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), constrained_layout=True)
    x = np.arange(2)
    for axis, (metric, ylabel) in zip(axes, specifications):
        means = [selected[selected["mechanism"] == item][metric].mean() for item in mechanisms]
        stds = [selected[selected["mechanism"] == item][metric].std(ddof=1) for item in mechanisms]
        axis.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.7)
        axis.set_xticks(x, labels, rotation=18)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        axis.set_axisbelow(True)
    figure_path = OUTPUT_DIR / "current_carbon_cda_trading_controlled_3seed.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_test_arrays_have_terminal_row()
    rows = []
    selected_rows = []

    for seed in SEEDS:
        standard_path = run_dir("standard", seed)
        carbon_path = run_dir("carbon", seed)
        native_sets = (("standard", standard_path), ("carbon_aware", carbon_path))
        for mechanism, path in native_sets:
            selected = representative_policies(path)
            for preference in PREFERENCES:
                policy_info = selected[preference]
                selected_rows.append(
                    {
                        "experiment": "native_policy",
                        "mechanism": mechanism,
                        "seed": seed,
                        "preference": preference,
                        **policy_info,
                    }
                )
                actors = load_actor(path, policy_info["policy_file"])
                for season, start in SEASONS.items():
                    rows.extend(
                        simulate(
                            actors,
                            mechanism,
                            seed,
                            preference,
                            policy_info,
                            season,
                            start,
                            "native_policy",
                        )
                    )

        selected = representative_policies(carbon_path)
        for preference in PREFERENCES:
            policy_info = selected[preference]
            actors = load_actor(carbon_path, policy_info["policy_file"])
            for mechanism in ("standard", "carbon_aware"):
                selected_rows.append(
                    {
                        "experiment": "controlled_main_policy",
                        "mechanism": mechanism,
                        "seed": seed,
                        "preference": preference,
                        **policy_info,
                    }
                )
                for season, start in SEASONS.items():
                    rows.extend(
                        simulate(
                            actors,
                            mechanism,
                            seed,
                            preference,
                            policy_info,
                            season,
                            start,
                            "controlled_main_policy",
                        )
                    )

    detail = pd.DataFrame(rows)
    episodes = episode_summary(detail)
    seed_summary = aggregate_by_seed(episodes)
    paired, cross_summary = paired_cross_evaluation(detail, seed_summary)

    paths = {
        "detail": OUTPUT_DIR / "current_carbon_cda_trading_detail_3seed.csv",
        "episodes": OUTPUT_DIR / "current_carbon_cda_trading_episode_summary_3seed.csv",
        "seed_summary": OUTPUT_DIR / "current_carbon_cda_trading_seed_summary_3seed.csv",
        "paired": OUTPUT_DIR / "current_carbon_cda_trading_paired_hourly_3seed.csv",
        "cross": OUTPUT_DIR / "current_carbon_cda_trading_cross_eval_3seed.csv",
        "policies": OUTPUT_DIR / "current_carbon_cda_trading_selected_policies_3seed.csv",
    }
    detail.to_csv(paths["detail"], index=False)
    episodes.to_csv(paths["episodes"], index=False)
    seed_summary.to_csv(paths["seed_summary"], index=False)
    paired.to_csv(paths["paired"], index=False)
    cross_summary.to_csv(paths["cross"], index=False)
    pd.DataFrame(selected_rows).drop_duplicates().to_csv(paths["policies"], index=False)
    figure_path = plot_controlled(seed_summary)

    print("Controlled same-policy comparison (Carbon-aware CDA relative to Standard CDA):")
    report_columns = [
        "seed",
        "allocation_change_rate",
        "relative_delta_peer_trade_amount_pct",
        "relative_delta_weighted_seller_carbon_intensity_pct",
        "relative_delta_cleanest_seller_share_pct",
        "relative_delta_grid_buy_pct",
        "relative_delta_total_cost_pct",
        "relative_delta_total_carbon_pct",
    ]
    print(cross_summary[report_columns].to_string(index=False))
    print("\nMean across seeds:")
    print(cross_summary[report_columns[1:]].mean().to_string())
    for path in (*paths.values(), figure_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
