import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_trading_behavior import SEASONS, summarize_step  # noqa: E402
from scripts.evaluate_strict_algorithm_comparison import (  # noqa: E402
    actor_action,
    ensure_terminal_test_row,
    initial_states,
    load_actors,
)
from src.envs.ies_env import env as IESEnv  # noqa: E402


RESULT_DIR = PROJECT_ROOT / "outputs" / "dispatch_behavior_current_vs_maddpg"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
METHODS = [
    {
        "method_key": "main",
        "method": "All-task Mean-HV E-MADDPG",
        "run_dir": PROJECT_ROOT
        / "outputs"
        / "runs"
        / "strict_v3_all_evolving_kappa0_seed_44",
    },
    {
        "method_key": "fixed_maddpg",
        "method": "Fixed-weight MADDPG",
        "run_dir": PROJECT_ROOT / "outputs" / "runs" / "strict_v3_maddpg_seed_44",
    },
]
PREFERENCE_ORDER = ["cost_priority", "balanced", "carbon_priority"]


def nondominated_mask(objectives):
    dominated = np.zeros(len(objectives), dtype=bool)
    for index, value in enumerate(objectives):
        dominated[index] = np.any(
            np.all(objectives <= value, axis=1) & np.any(objectives < value, axis=1)
        )
    return ~dominated


def representative_policies(run_dir):
    metadata = pd.read_csv(run_dir / "strict_test" / "metadata.csv")
    objectives = metadata[["cost", "carbon"]].to_numpy(dtype=np.float64)
    front_indices = np.flatnonzero(nondominated_mask(objectives))
    front = objectives[front_indices]
    ideal = front.min(axis=0)
    span = np.maximum(front.max(axis=0) - ideal, 1e-12)
    normalized = (front - ideal) / span
    local_indices = {
        "cost_priority": int(np.argmin(front[:, 0])),
        "balanced": int(np.argmin(np.linalg.norm(normalized - 0.5, axis=1))),
        "carbon_priority": int(np.argmin(front[:, 1])),
    }
    selected = {}
    for preference, local_index in local_indices.items():
        row = metadata.iloc[int(front_indices[local_index])]
        selected[preference] = {
            "policy_file": row["policy_file"],
            "pareto_index": int(row["pareto_index"]),
            "cost": float(row["cost"]),
            "carbon": float(row["carbon"]),
        }
    return selected


def simulate(method, preference, policy, season, start):
    environment = IESEnv(
        market_mechanism="carbon_dividend_v3",
        cda_carbon_price=0.01,
        cda_dynamic_carbon_price=False,
    )
    actors = load_actors(method["run_dir"] / "final" / policy["policy_file"])
    states = initial_states(environment, start)
    rows = []
    for hour in range(24):
        action = actor_action(actors, states)
        next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = environment.step(
            action,
            np.concatenate(states, axis=0),
            start + hour,
            flag=1,
        )
        objective = np.asarray(obj_r) + np.asarray(obj_c) + np.asarray(obj_i)
        row = summarize_step(environment)
        row.update(
            {
                "method_key": method["method_key"],
                "method": method["method"],
                "preference": preference,
                "season": season,
                "hour": hour,
                "policy_file": policy["policy_file"],
                "pareto_index": policy["pareto_index"],
                "step_cost": float(objective[0]),
                "step_carbon": float(objective[1]),
            }
        )
        rows.append(row)
        states = [next_res, next_com, next_ind]
        if done and hour < 23:
            raise RuntimeError(f"{method['method']} ended early at hour {hour}.")
    return rows


def plot_summary(summary, path):
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
    metrics = [
        ("total_cost", "Operating cost", 1e3),
        ("total_carbon", "Carbon emissions", 1e3),
        ("total_grid_buy", "Grid purchase", 1e3),
        ("total_peer_trade", "Peer trade", 1e3),
    ]
    colors = {"main": "#C44E52", "fixed_maddpg": "#F28E2B"}
    labels = {item["method_key"]: item["method"] for item in METHODS}
    x = np.arange(len(PREFERENCE_ORDER))
    width = 0.34
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.0), constrained_layout=True)
    for ax, (column, ylabel, scale) in zip(axes.ravel(), metrics):
        for method_index, method_key in enumerate(["main", "fixed_maddpg"]):
            values = []
            for preference in PREFERENCE_ORDER:
                row = summary[
                    (summary["method_key"] == method_key)
                    & (summary["preference"] == preference)
                ]
                values.append(float(row[column].iloc[0]) / scale)
            offset = (method_index - 0.5) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                color=colors[method_key],
                edgecolor="black",
                linewidth=0.6,
                label=labels[method_key],
            )
        ax.set_ylabel(f"{ylabel} ($\\times 10^3$)")
        ax.set_xticks(x)
        ax.set_xticklabels(["Cost", "Balanced", "Carbon"])
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    axes[0, 0].legend(frameon=False, loc="best")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    ensure_terminal_test_row()
    rows = []
    selections = []
    for method in METHODS:
        selected = representative_policies(method["run_dir"])
        for preference, policy in selected.items():
            selections.append(
                {
                    "method_key": method["method_key"],
                    "method": method["method"],
                    "preference": preference,
                    **policy,
                }
            )
            for season, start in SEASONS.items():
                rows.extend(simulate(method, preference, policy, season, start))

    detail = pd.DataFrame(rows)
    season_summary = (
        detail.groupby(["method_key", "method", "preference", "season"], as_index=False)
        .agg(
            total_cost=("step_cost", "sum"),
            total_carbon=("step_carbon", "sum"),
            total_grid_buy=("grid_buy", "sum"),
            total_peer_trade=("peer_trade_amount", "sum"),
        )
    )
    summary = (
        season_summary.groupby(["method_key", "method", "preference"], as_index=False)
        .agg(
            total_cost=("total_cost", "sum"),
            total_carbon=("total_carbon", "sum"),
            total_grid_buy=("total_grid_buy", "sum"),
            total_peer_trade=("total_peer_trade", "sum"),
        )
    )
    detail.to_csv(RESULT_DIR / "detail.csv", index=False)
    season_summary.to_csv(RESULT_DIR / "season_summary.csv", index=False)
    summary.to_csv(RESULT_DIR / "summary.csv", index=False)
    pd.DataFrame(selections).to_csv(RESULT_DIR / "selected_policies.csv", index=False)
    figure_path = FIGURE_DIR / "dispatch_behavior_current_vs_maddpg.png"
    plot_summary(summary, figure_path)
    print(summary.to_string(index=False))
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
