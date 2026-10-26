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
    ensure_test_arrays_have_terminal_row,
    load_actor,
    representative_policies,
    simulate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "runs"
    / "standard_cda_hv_ensemble_extreme_capped32_kappa_0_10"
)
PRICES = (0.01, 0.02, 0.05)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_test_arrays_have_terminal_row()
    selected = representative_policies(RUN_DIR)
    rows = []
    for preference in PREFERENCES:
        policy_info = selected[preference]
        actors = load_actor(RUN_DIR, policy_info["policy_file"])
        for season, start in SEASONS.items():
            standard_rows = simulate(
                actors,
                "standard",
                44,
                preference,
                policy_info,
                season,
                start,
                "v3_signal_pilot",
                carbon_price=0.0,
            )
            for row in standard_rows:
                row["carbon_price_setting"] = 0.0
            rows.extend(standard_rows)
            for price in PRICES:
                v3_rows = simulate(
                    actors,
                    "carbon_dividend_v3",
                    44,
                    preference,
                    policy_info,
                    season,
                    start,
                    "v3_signal_pilot",
                    carbon_price=price,
                )
                for row in v3_rows:
                    row["carbon_price_setting"] = price
                rows.extend(v3_rows)

    detail = pd.DataFrame(rows)
    summaries = []
    standard = detail[detail.carbon_price_setting == 0.0]
    standard_cost = float(standard.step_cost.sum())
    standard_carbon = float(standard.step_carbon.sum())
    for price in PRICES:
        selected_rows = detail[detail.carbon_price_setting == price]
        summaries.append(
            {
                "carbon_price": price,
                "hours": len(selected_rows),
                "fund_active_rate": float((selected_rows.carbon_dividend_fund > 1e-12).mean()),
                "targeted_dividend_rate": float(
                    (selected_rows.avoided_carbon_credit_basis > 1e-12).mean()
                ),
                "total_dividend_fund": float(selected_rows.carbon_dividend_fund.sum()),
                "mean_active_fund": float(
                    selected_rows.loc[
                        selected_rows.carbon_dividend_fund > 1e-12,
                        "carbon_dividend_fund",
                    ].mean()
                ),
                "total_absolute_transfer": float(selected_rows.carbon_transfer_abs.sum()),
                "max_balance_error": float(
                    selected_rows.carbon_dividend_balance_error.abs().max()
                ),
                "system_cost_delta": float(selected_rows.step_cost.sum() - standard_cost),
                "system_carbon_delta": float(selected_rows.step_carbon.sum() - standard_carbon),
            }
        )
    summary = pd.DataFrame(summaries)
    detail_path = OUTPUT_DIR / "v3_signal_pilot_detail.csv"
    summary_path = OUTPUT_DIR / "v3_signal_pilot_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    plt.rcParams.update({"font.family": "Times New Roman", "savefig.dpi": 600})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), constrained_layout=True)
    x = np.arange(len(summary))
    labels = [f"{value:.2f}" for value in summary.carbon_price]
    for axis, column, ylabel in (
        (axes[0], "fund_active_rate", "Fund activation rate"),
        (axes[1], "targeted_dividend_rate", "Targeted-dividend rate"),
        (axes[2], "total_absolute_transfer", "Absolute transfer (CNY)"),
    ):
        axis.bar(x, summary[column], color="#59A14F", edgecolor="black", linewidth=0.7)
        axis.set_xticks(x)
        axis.set_xticklabels(labels)
        axis.set_xlabel("Carbon price")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.65)
    figure_path = OUTPUT_DIR / "v3_signal_pilot.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    for path in (detail_path, summary_path, figure_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
