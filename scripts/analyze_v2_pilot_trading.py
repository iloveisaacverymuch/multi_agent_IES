from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_current_carbon_cda_trading_3seed import (
    OUTPUT_DIR,
    PREFERENCES,
    SEASONS,
    aggregate_by_seed,
    ensure_test_arrays_have_terminal_row,
    episode_summary,
    load_actor,
    representative_policies,
    simulate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "runs"
    / "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02"
)
MECHANISMS = ("standard", "carbon_aware", "carbon_aware_v2")
LABELS = ("Standard CDA", "Carbon CDA V1", "Carbon CDA V2")


def main():
    ensure_test_arrays_have_terminal_row()
    selected = representative_policies(RUN_DIR)
    rows = []
    for preference in PREFERENCES:
        policy_info = selected[preference]
        actors = load_actor(RUN_DIR, policy_info["policy_file"])
        for mechanism in MECHANISMS:
            for season, start in SEASONS.items():
                rows.extend(
                    simulate(
                        actors,
                        mechanism,
                        44,
                        preference,
                        policy_info,
                        season,
                        start,
                        "v2_pilot_policy",
                    )
                )

    detail = pd.DataFrame(rows)
    episodes = episode_summary(detail)
    summary = aggregate_by_seed(episodes)
    detail_path = OUTPUT_DIR / "v2_pilot_trading_detail.csv"
    summary_path = OUTPUT_DIR / "v2_pilot_trading_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    standard = detail[detail.mechanism == "standard"].set_index(
        ["preference", "season", "hour"]
    )
    comparison_rows = []
    for mechanism in MECHANISMS[1:]:
        selected_detail = detail[detail.mechanism == mechanism].set_index(
            ["preference", "season", "hour"]
        )
        changes = np.column_stack(
            [
                np.abs(selected_detail[column] - standard[column])
                for column in (
                    "peer_trade_amount",
                    "seller_1_amount",
                    "seller_2_amount",
                    "seller_3_amount",
                )
            ]
        )
        row = {"mechanism": mechanism, "allocation_change_rate": float((changes.max(axis=1) > 1e-7).mean())}
        for metric in (
            "peer_trade_amount",
            "weighted_seller_carbon_intensity",
            "cleanest_seller_share",
            "grid_buy",
            "total_cost",
            "total_carbon",
        ):
            baseline = float(summary[summary.mechanism == "standard"][metric].iloc[0])
            value = float(summary[summary.mechanism == mechanism][metric].iloc[0])
            row[f"relative_delta_{metric}_pct"] = (
                100.0 * (value - baseline) / abs(baseline) if abs(baseline) > 1e-12 else 0.0
            )
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = OUTPUT_DIR / "v2_pilot_trading_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    specifications = (
        ("peer_trade_amount", "Peer trade amount (kWh)"),
        ("weighted_seller_carbon_intensity", "Seller carbon intensity"),
        ("grid_buy", "Grid imports (kWh)"),
        ("total_carbon", "Total carbon emissions (kg)"),
    )
    colors = ("#4C78A8", "#F28E2B", "#59A14F")
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes, specifications):
        values = [float(summary[summary.mechanism == item][metric].iloc[0]) for item in MECHANISMS]
        axis.bar(np.arange(3), values, color=colors, edgecolor="black", linewidth=0.7)
        axis.set_xticks(np.arange(3), LABELS, rotation=20)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        axis.set_axisbelow(True)
    figure_path = OUTPUT_DIR / "v2_pilot_trading.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print("\nRelative to Standard CDA:")
    print(comparison.to_string(index=False))
    for path in (detail_path, summary_path, comparison_path, figure_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
