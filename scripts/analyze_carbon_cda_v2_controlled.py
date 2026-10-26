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
    SEEDS,
    aggregate_by_seed,
    ensure_test_arrays_have_terminal_row,
    episode_summary,
    load_actor,
    representative_policies,
    run_dir,
    simulate,
)


MECHANISMS = ("standard", "carbon_aware", "carbon_aware_v2")
LABELS = {
    "standard": "Standard CDA",
    "carbon_aware": "Carbon CDA V1",
    "carbon_aware_v2": "Carbon CDA V2",
}


def relative_change(value, baseline):
    if abs(baseline) <= 1e-12:
        return 0.0
    return 100.0 * (value - baseline) / abs(baseline)


def paired_summary(detail, seed_summary):
    index = ["seed", "preference", "season", "hour"]
    wide = detail.pivot(index=index, columns="mechanism")
    allocation_rates = {}
    for mechanism in MECHANISMS[1:]:
        changes = []
        for metric in (
            "peer_trade_amount",
            "seller_1_amount",
            "seller_2_amount",
            "seller_3_amount",
        ):
            changes.append(
                np.abs(
                    wide[(metric, mechanism)].to_numpy()
                    - wide[(metric, "standard")].to_numpy()
                )
            )
        changed = np.max(np.vstack(changes), axis=0) > 1e-7
        allocation_rates[mechanism] = pd.Series(changed, index=wide.index).groupby("seed").mean()

    indexed = seed_summary.set_index(["mechanism", "seed"])
    metrics = (
        "peer_trade_amount",
        "weighted_seller_carbon_intensity",
        "cleanest_seller_share",
        "grid_buy",
        "total_cost",
        "total_carbon",
    )
    rows = []
    for mechanism in MECHANISMS[1:]:
        for seed in SEEDS:
            row = {
                "mechanism": mechanism,
                "seed": seed,
                "allocation_change_rate": float(allocation_rates[mechanism].loc[seed]),
            }
            for metric in metrics:
                baseline = float(indexed.loc[("standard", seed), metric])
                value = float(indexed.loc[(mechanism, seed), metric])
                row[f"standard_{metric}"] = baseline
                row[f"value_{metric}"] = value
                row[f"relative_delta_{metric}_pct"] = relative_change(value, baseline)
            rows.append(row)
    return pd.DataFrame(rows)


def plot(seed_summary):
    specifications = (
        ("peer_trade_amount", "Peer trade amount (kWh)"),
        ("weighted_seller_carbon_intensity", "Seller carbon intensity"),
        ("grid_buy", "Grid imports (kWh)"),
        ("total_carbon", "Total carbon emissions (kg)"),
    )
    colors = ("#4C78A8", "#F28E2B", "#59A14F")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), constrained_layout=True)
    x = np.arange(len(MECHANISMS))
    for axis, (metric, ylabel) in zip(axes, specifications):
        means = [seed_summary[seed_summary.mechanism == item][metric].mean() for item in MECHANISMS]
        stds = [seed_summary[seed_summary.mechanism == item][metric].std(ddof=1) for item in MECHANISMS]
        axis.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.7)
        axis.set_xticks(x, [LABELS[item] for item in MECHANISMS], rotation=20)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
        axis.set_axisbelow(True)
    path = OUTPUT_DIR / "carbon_cda_v2_controlled_3seed.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_test_arrays_have_terminal_row()
    rows = []
    for seed in SEEDS:
        path = run_dir("carbon", seed)
        selected = representative_policies(path)
        for preference in PREFERENCES:
            policy_info = selected[preference]
            actors = load_actor(path, policy_info["policy_file"])
            for mechanism in MECHANISMS:
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
                            "controlled_carbon_cda_v2",
                        )
                    )

    detail = pd.DataFrame(rows)
    episodes = episode_summary(detail)
    seed_summary = aggregate_by_seed(episodes)
    comparison = paired_summary(detail, seed_summary)

    paths = {
        "detail": OUTPUT_DIR / "carbon_cda_v2_controlled_detail_3seed.csv",
        "seed": OUTPUT_DIR / "carbon_cda_v2_controlled_seed_summary_3seed.csv",
        "comparison": OUTPUT_DIR / "carbon_cda_v2_controlled_comparison_3seed.csv",
    }
    detail.to_csv(paths["detail"], index=False)
    seed_summary.to_csv(paths["seed"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    figure = plot(seed_summary)

    columns = [
        "allocation_change_rate",
        "relative_delta_peer_trade_amount_pct",
        "relative_delta_weighted_seller_carbon_intensity_pct",
        "relative_delta_grid_buy_pct",
        "relative_delta_total_cost_pct",
        "relative_delta_total_carbon_pct",
    ]
    print(comparison.groupby("mechanism")[columns].mean().to_string())
    for path in (*paths.values(), figure):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
