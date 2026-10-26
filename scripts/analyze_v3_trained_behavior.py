from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_current_carbon_cda_trading_3seed import (
    PREFERENCES,
    SEASONS,
    ensure_test_arrays_have_terminal_row,
    load_actor,
    representative_policies,
    simulate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
CONFIGURATIONS = (
    (
        "standard",
        "Standard CDA",
        RUN_ROOT / "standard_cda_hv_ensemble_extreme_capped32_kappa_0_10",
        "standard",
        0.0,
    ),
    *(
        (
            f"v3_{str(price).replace('.', '_')}",
            f"Carbon-Dividend V3 ({price:.2f})",
            RUN_ROOT
            / (
                "carbon_dividend_v3_hv_ensemble_extreme_capped32_kappa_0_10_"
                f"price_{price:.2f}".replace(".", "_")
            ),
            "carbon_dividend_v3",
            price,
        )
        for price in (0.01, 0.02, 0.05)
    ),
)


def evaluate():
    rows = []
    selected_rows = []
    for key, label, run_dir, mechanism, price in CONFIGURATIONS:
        selected = representative_policies(run_dir)
        for preference in PREFERENCES:
            policy_info = selected[preference]
            selected_rows.append(
                {
                    "method_key": key,
                    "method": label,
                    "carbon_price": price,
                    "preference": preference,
                    **policy_info,
                }
            )
            actors = load_actor(run_dir, policy_info["policy_file"])
            for season, start in SEASONS.items():
                episode = simulate(
                    actors,
                    mechanism,
                    44,
                    preference,
                    policy_info,
                    season,
                    start,
                    "v3_trained_behavior",
                    carbon_price=price,
                )
                for row in episode:
                    row["method_key"] = key
                    row["method"] = label
                    row["carbon_price_setting"] = price
                rows.extend(episode)
    return pd.DataFrame(rows), pd.DataFrame(selected_rows)


def summarize(detail):
    group_columns = [
        "method_key",
        "method",
        "carbon_price_setting",
        "preference",
        "season",
        "pareto_index",
    ]
    episodes = detail.groupby(group_columns, as_index=False).agg(
        total_cost=("step_cost", "sum"),
        total_carbon=("step_carbon", "sum"),
        grid_buy=("grid_buy", "sum"),
        grid_sell=("grid_sell", "sum"),
        peer_trade=("peer_trade_amount", "sum"),
        waste_heat=("waste_heat", "sum"),
        dividend_fund=("carbon_dividend_fund", "sum"),
        absolute_transfer=("carbon_transfer_abs", "sum"),
        max_balance_error=("carbon_dividend_balance_error", lambda x: float(np.abs(x).max())),
        active_hours=("carbon_dividend_fund", lambda x: int((x > 1e-12).sum())),
        targeted_hours=("avoided_carbon_credit_basis", lambda x: int((x > 1e-12).sum())),
    )
    summary = episodes.groupby(
        ["method_key", "method", "carbon_price_setting"], as_index=False
    ).agg(
        episodes=("total_cost", "count"),
        total_cost=("total_cost", "sum"),
        total_carbon=("total_carbon", "sum"),
        grid_buy=("grid_buy", "sum"),
        grid_sell=("grid_sell", "sum"),
        peer_trade=("peer_trade", "sum"),
        waste_heat=("waste_heat", "sum"),
        dividend_fund=("dividend_fund", "sum"),
        absolute_transfer=("absolute_transfer", "sum"),
        max_balance_error=("max_balance_error", "max"),
        active_hours=("active_hours", "sum"),
        targeted_hours=("targeted_hours", "sum"),
    )
    hours = summary["episodes"] * 24
    summary["fund_active_rate"] = summary["active_hours"] / hours
    summary["targeted_dividend_rate"] = summary["targeted_hours"] / hours
    baseline = summary.loc[summary["method_key"] == "standard"].iloc[0]
    for metric in ("total_cost", "total_carbon", "grid_buy", "peer_trade", "waste_heat"):
        baseline_value = float(baseline[metric])
        summary[f"{metric}_change_vs_standard_percent"] = (
            100.0 * (summary[metric] / baseline_value - 1.0)
            if abs(baseline_value) > 1e-12
            else np.nan
        )
    return episodes, summary


def robustness_summary(episodes):
    baseline = episodes[episodes["method_key"] == "standard"].set_index(
        ["preference", "season"]
    )
    rows = []
    for key in ("v3_0_01", "v3_0_02", "v3_0_05"):
        current = episodes[episodes["method_key"] == key].set_index(
            ["preference", "season"]
        )
        row = {"method_key": key}
        changes = {}
        for metric in ("total_cost", "total_carbon", "grid_buy", "peer_trade"):
            denominator = baseline[metric].where(baseline[metric].abs() > 1e-12)
            change = 100.0 * (current[metric] / denominator - 1.0)
            changes[metric] = change
            row[f"{metric}_mean_change_percent"] = float(change.mean())
            row[f"{metric}_min_change_percent"] = float(change.min())
            row[f"{metric}_max_change_percent"] = float(change.max())
            if metric == "peer_trade":
                row[f"{metric}_improved_episodes"] = int((change > 0.0).sum())
            else:
                row[f"{metric}_improved_episodes"] = int((change < 0.0).sum())
        row["cost_and_carbon_improved_episodes"] = int(
            ((current["total_cost"] < baseline["total_cost"])
             & (current["total_carbon"] < baseline["total_carbon"])).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(summary):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "savefig.dpi": 600,
        }
    )
    labels = summary["method"].tolist()
    x = np.arange(len(summary))
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759"]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
    specs = (
        ("total_cost", "Test cost (CNY)"),
        ("total_carbon", "Test emissions (kg CO$_2$)"),
        ("grid_buy", "Grid imports (kWh)"),
    )
    for axis, (column, ylabel) in zip(axes, specs):
        axis.bar(x, summary[column], color=colors, edgecolor="black", linewidth=0.7)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=22, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)
    path = OUTPUT_DIR / "v3_trained_behavior.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_test_arrays_have_terminal_row()
    detail, selected = evaluate()
    episodes, summary = summarize(detail)
    robustness = robustness_summary(episodes)
    paths = {
        "detail": OUTPUT_DIR / "v3_trained_behavior_detail.csv",
        "episodes": OUTPUT_DIR / "v3_trained_behavior_episodes.csv",
        "summary": OUTPUT_DIR / "v3_trained_behavior_summary.csv",
        "robustness": OUTPUT_DIR / "v3_trained_behavior_robustness.csv",
        "policies": OUTPUT_DIR / "v3_trained_behavior_selected_policies.csv",
    }
    detail.to_csv(paths["detail"], index=False)
    episodes.to_csv(paths["episodes"], index=False)
    summary.to_csv(paths["summary"], index=False)
    robustness.to_csv(paths["robustness"], index=False)
    selected.to_csv(paths["policies"], index=False)
    figure_path = plot_summary(summary)
    print(summary.to_string(index=False))
    for path in (*paths.values(), figure_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
