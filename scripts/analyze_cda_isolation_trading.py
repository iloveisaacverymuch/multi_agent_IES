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
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
SEEDS = (11, 22, 44)
SOURCES = (
    ("standard", "Standard-trained", "standard_cda_hv_ensemble_extreme_capped32_kappa_0_10"),
    (
        "carbon_v1",
        "Carbon V1-trained",
        "carbon_cda_v1_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02",
    ),
    (
        "carbon_v2",
        "Carbon V2-trained",
        "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02",
    ),
)
MECHANISMS = ("standard", "carbon_aware", "carbon_aware_v2")
MECHANISM_LABELS = {
    "standard": "Standard CDA",
    "carbon_aware": "Carbon CDA V1",
    "carbon_aware_v2": "Carbon CDA V2",
}
NATIVE_MECHANISM = {
    "standard": "standard",
    "carbon_v1": "carbon_aware",
    "carbon_v2": "carbon_aware_v2",
}


def run_dir(base_name, seed):
    return RUN_ROOT / (base_name if seed == 44 else f"{base_name}_seed_{seed}")


def relative_change(value, baseline):
    if abs(baseline) <= 1e-12:
        return 0.0
    return 100.0 * (value - baseline) / abs(baseline)


def controlled_comparison(detail, seed_summary):
    index = ["source_method", "seed", "preference", "season", "hour"]
    wide = detail.pivot(index=index, columns="mechanism")
    summary_index = seed_summary.set_index(["source_method", "mechanism", "seed"])
    metrics = (
        "peer_trade_amount",
        "weighted_seller_carbon_intensity",
        "cleanest_seller_share",
        "grid_buy",
        "grid_sell",
        "matched_trade_share",
        "total_cost",
        "total_carbon",
        "waste_heat",
    )
    rows = []
    for source_key, _, _ in SOURCES:
        source_wide = wide.loc[source_key]
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
                        source_wide[(metric, mechanism)].to_numpy()
                        - source_wide[(metric, "standard")].to_numpy()
                    )
                )
            changed = np.max(np.vstack(changes), axis=0) > 1e-7
            activation = pd.Series(changed, index=source_wide.index).groupby("seed").mean()
            for seed in SEEDS:
                row = {
                    "source_method": source_key,
                    "mechanism": mechanism,
                    "seed": seed,
                    "allocation_change_rate": float(activation.loc[seed]),
                }
                for metric in metrics:
                    baseline = float(summary_index.loc[(source_key, "standard", seed), metric])
                    value = float(summary_index.loc[(source_key, mechanism, seed), metric])
                    row[f"standard_{metric}"] = baseline
                    row[f"value_{metric}"] = value
                    row[f"relative_delta_{metric}_pct"] = relative_change(value, baseline)
                rows.append(row)
    return pd.DataFrame(rows)


def native_summary(seed_summary):
    rows = []
    for source_key, source_label, _ in SOURCES:
        mechanism = NATIVE_MECHANISM[source_key]
        selected = seed_summary[
            (seed_summary.source_method == source_key)
            & (seed_summary.mechanism == mechanism)
        ].copy()
        selected["source_label"] = source_label
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def plot_controlled(comparison):
    specifications = (
        ("allocation_change_rate", "Allocation change rate"),
        ("relative_delta_peer_trade_amount_pct", "Peer trade change (%)"),
        ("relative_delta_grid_buy_pct", "Grid import change (%)"),
        ("relative_delta_total_carbon_pct", "System carbon change (%)"),
    )
    mechanisms = MECHANISMS[1:]
    colors = ("#F28E2B", "#59A14F")
    plt.rcParams.update({"font.family": "Times New Roman", "savefig.dpi": 600})
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.3), constrained_layout=True)
    x = np.arange(len(mechanisms))
    for axis, (metric, ylabel) in zip(axes, specifications):
        means = [comparison[comparison.mechanism == item][metric].mean() for item in mechanisms]
        stds = [comparison[comparison.mechanism == item][metric].std(ddof=1) for item in mechanisms]
        axis.bar(x, means, yerr=stds, color=colors, edgecolor="black", linewidth=0.7, capsize=3)
        axis.set_xticks(x)
        axis.set_xticklabels([MECHANISM_LABELS[item] for item in mechanisms], rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.65)
    path = OUTPUT_DIR / "cda_isolation_controlled_trading.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_test_arrays_have_terminal_row()
    rows = []
    for source_key, _, base_name in SOURCES:
        for seed in SEEDS:
            path = run_dir(base_name, seed)
            selected = representative_policies(path)
            for preference in PREFERENCES:
                policy_info = selected[preference]
                actors = load_actor(path, policy_info["policy_file"])
                for mechanism in MECHANISMS:
                    for season, start in SEASONS.items():
                        simulated = simulate(
                            actors,
                            mechanism,
                            seed,
                            preference,
                            policy_info,
                            season,
                            start,
                            f"cda_isolation_{source_key}",
                        )
                        for row in simulated:
                            row["source_method"] = source_key
                        rows.extend(simulated)

    detail = pd.DataFrame(rows)
    episodes = episode_summary(detail)
    episodes["source_method"] = episodes["experiment"].str.replace(
        "cda_isolation_", "", regex=False
    )
    seed_summary = aggregate_by_seed(episodes)
    seed_summary["source_method"] = seed_summary["experiment"].str.replace(
        "cda_isolation_", "", regex=False
    )
    comparison = controlled_comparison(detail, seed_summary)
    comparison_summary = comparison.groupby("mechanism", as_index=False).agg(
        allocation_change_rate=("allocation_change_rate", "mean"),
        peer_trade_change_pct=("relative_delta_peer_trade_amount_pct", "mean"),
        seller_carbon_intensity_change_pct=(
            "relative_delta_weighted_seller_carbon_intensity_pct",
            "mean",
        ),
        cleanest_seller_share_change_pct=(
            "relative_delta_cleanest_seller_share_pct",
            "mean",
        ),
        grid_buy_change_pct=("relative_delta_grid_buy_pct", "mean"),
        total_cost_change_pct=("relative_delta_total_cost_pct", "mean"),
        total_carbon_change_pct=("relative_delta_total_carbon_pct", "mean"),
    )
    native = native_summary(seed_summary)
    paths = {
        "detail": OUTPUT_DIR / "cda_isolation_controlled_detail.csv",
        "episodes": OUTPUT_DIR / "cda_isolation_controlled_episodes.csv",
        "seed_summary": OUTPUT_DIR / "cda_isolation_controlled_seed_summary.csv",
        "comparison": OUTPUT_DIR / "cda_isolation_controlled_comparison.csv",
        "comparison_summary": OUTPUT_DIR / "cda_isolation_controlled_comparison_summary.csv",
        "native": OUTPUT_DIR / "cda_isolation_native_policy_summary.csv",
    }
    detail.to_csv(paths["detail"], index=False)
    episodes.to_csv(paths["episodes"], index=False)
    seed_summary.to_csv(paths["seed_summary"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    comparison_summary.to_csv(paths["comparison_summary"], index=False)
    native.to_csv(paths["native"], index=False)
    paths["figure"] = plot_controlled(comparison)
    print(comparison_summary.to_string(index=False))
    for path in paths.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
