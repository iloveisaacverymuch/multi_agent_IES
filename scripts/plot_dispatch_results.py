import csv
import sys
from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs import ies_env
from src.envs.ies_env import env as IESEnv
from scripts.train_maddpg_baseline import ACT_DIMS, OBS_DIMS, Actor as MADDPGActor


RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "runs"
    / "strict_v3_all_evolving_kappa0_seed_44"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "dispatch_results_main_method"
REPRESENTATIVES_CSV = OUTPUT_DIR / "representatives.csv"
MARKET_MECHANISM = "carbon_dividend_v3"
CDA_CARBON_PRICE = 0.01

SEASONS = {
    "spring": 24,
    "summer": 48,
    "autumn": 72,
    "winter": 0,
}

RES_COM_COLUMNS = [
    "electric_load",
    "heat_load",
    "cooling_load",
    "PV",
    "P_CHP",
    "P_HCHP",
    "C_AC",
    "C_CER",
    "H_EB",
    "P_ESS",
    "H_TES",
    "buy_from_grid",
    "sell_to_grid",
    "trade_record_e",
    "price_e",
    "waste_h",
]

IND_COLUMNS = [
    "electric_load",
    "heat_load",
    "PV",
    "P_CHP",
    "P_HCHP",
    "H_GB",
    "C_AC",
    "P_ESS",
    "H_TES",
    "buy_from_grid",
    "sell_to_grid",
    "trade_record_e",
    "price_e",
    "waste_h",
]

AGENT_COLUMNS = {
    "residential": RES_COM_COLUMNS,
    "commercial": RES_COM_COLUMNS,
    "industry": IND_COLUMNS,
}

AGENT_TITLES = {
    "residential": "Residential",
    "commercial": "Commercial",
    "industry": "Industry",
}

PREFERENCE_TITLES = {
    "carbon_priority": "Carbon-priority",
    "cost_priority": "Cost-priority",
    "balanced": "Balanced",
}

PLOTS = {
    "electricity": ("Electricity balance", "Electricity power (kW)"),
    "heat": ("Heat balance", "Heat power (kW)"),
    "cooling": ("Cooling balance", "Cooling power (kW)"),
}

COLORS = {
    "Electricity load": "#800080",
    "Heat load": "#800080",
    "Cooling load": "#4169E1",
    "PV": "#1E90FF",
    "CHP": "#9370DB",
    "HCHP": "#A0522D",
    "ESS": "#FFA500",
    "Grid": "#2E8B57",
    "Peer trade": "#FF7F0E",
    "CER": "#D2691E",
    "EB": "#4169E1",
    "TES": "#9370DB",
    "GB": "#2F4F4F",
    "AC": "#FFD700",
    "Heat surplus": "#C0C0C0",
    "Balance residual": "#C0C0C0",
}

HATCHES = {
    "PV": "X",
    "CHP": "*",
    "HCHP": "//",
    "ESS": "//",
    "Grid": "\\\\",
    "Peer trade": "..",
    "CER": "..",
    "EB": "xx",
    "TES": "oo",
    "GB": "xx",
    "AC": "..",
    "Heat surplus": "++",
    "Balance residual": "++",
}


def configure_style():
    plt.rcParams["axes.unicode_minus"] = False
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 24,
            "axes.titlesize": 16,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 18,
            "figure.titlesize": 16,
            "font.family": "Times New Roman",
        }
    )


def load_final_objectives(run_dir):
    obj_path = run_dir / "final" / "objs.txt"
    objectives = np.loadtxt(obj_path, delimiter=",")
    return np.atleast_2d(objectives)


def make_selection_row(preference, idx, objectives):
    cost, carbon = objectives[idx]
    return {
        "preference": preference,
        "policy_file": f"EP_policy_{idx}.pt",
        "pareto_index": str(idx),
        "cost": str(float(cost)),
        "carbon": str(float(carbon)),
    }


def select_preferences_from_objectives(run_dir):
    objectives = load_final_objectives(run_dir)
    policy_count = len(list((run_dir / "final").glob("EP_policy_*.pt")))
    if policy_count != len(objectives):
        raise ValueError(
            f"Policy count ({policy_count}) does not match objective rows ({len(objectives)})."
        )

    mins = objectives.min(axis=0)
    ranges = objectives.max(axis=0) - mins
    ranges = np.where(ranges == 0, 1.0, ranges)
    normalized_distance = np.linalg.norm((objectives - mins) / ranges, axis=1)

    return {
        "carbon_priority": make_selection_row(
            "carbon_priority", int(np.argmin(objectives[:, 1])), objectives
        ),
        "cost_priority": make_selection_row(
            "cost_priority", int(np.argmin(objectives[:, 0])), objectives
        ),
        "balanced": make_selection_row(
            "balanced", int(np.argmin(normalized_distance)), objectives
        ),
    }


def select_preferences_from_representatives(path):
    selected = {}
    with open(path, newline="") as fp:
        for row in csv.DictReader(fp):
            preference = row["preference_type"]
            selected[preference] = {
                "preference": preference,
                "policy_file": row["policy_file"],
                "pareto_index": row.get("pareto_index", ""),
                "cost": row["cost"],
                "carbon": row["carbon"],
            }
    required = {"carbon_priority", "cost_priority", "balanced"}
    missing = required - set(selected)
    if missing:
        raise ValueError(f"Missing representative preferences: {sorted(missing)}")
    return selected


def load_actor(policy_file):
    policy_path = RUN_DIR / "final" / policy_file
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    try:
        payload = torch.load(policy_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(policy_path, map_location="cpu")
    if isinstance(payload, dict) and payload.get("format") == "maddpg_state_dict_v1":
        actor_dict = {}
        for agent_id, state_dict in enumerate(payload["actor_state_dicts"]):
            actor = MADDPGActor(
                OBS_DIMS[agent_id], ACT_DIMS[agent_id], payload["hidden_dim"]
            )
            actor.load_state_dict(state_dict)
            actor_dict[agent_id] = actor
    else:
        actor_dict = payload
    for actor in actor_dict.values():
        actor.eval()
    return actor_dict


def ensure_test_arrays_have_terminal_row():
    # env.step(flag=1) reads step + 1 before checking done. Appending one row
    # keeps the final hour of the autumn block safe to evaluate.
    for name in [
        "state_test_residential",
        "state_test_commercial",
        "state_test_industry",
    ]:
        arr = getattr(ies_env, name)
        if arr.shape[0] == 96:
            setattr(ies_env, name, np.vstack([arr, arr[-1:]]))


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
        states = [state_res, state_com, state_ind]
        actions = []
        for agent_id, state in enumerate(states):
            state_tensor = torch.tensor(state, dtype=torch.float32).squeeze()
            try:
                output = actor_dict[agent_id](state_tensor, evaluation=True)
            except TypeError:
                output = actor_dict[agent_id](state_tensor)
            if isinstance(output, tuple):
                action = torch.tanh(output[1])
            else:
                action = output
            actions.append(action.reshape(-1))

        action_res = actions[0].reshape(1, -1)
        action_com = actions[1].reshape(1, -1)
        action_ind = actions[2]
        action_ind = torch.cat([action_ind, torch.tensor([0.0])]).reshape(1, -1)
        action = torch.cat((action_res, action_com, action_ind), dim=0)
    return action.numpy()


def schedule_to_dict(values, columns):
    return {name: float(value) for name, value in zip(columns, values) if name != "waste_h"}


def simulate_day(actor_dict, season_start):
    environment = IESEnv(
        market_mechanism=MARKET_MECHANISM,
        cda_carbon_price=CDA_CARBON_PRICE,
    )
    state_res, state_com, state_ind = make_initial_state(environment, season_start)
    records = {"residential": [], "commercial": [], "industry": []}

    for hour in range(24):
        absolute_step = season_start + hour
        state_np = np.concatenate((state_res, state_com, state_ind), axis=0)
        action = choose_action(actor_dict, state_res, state_com, state_ind)
        state_res, state_com, state_ind, done, *_ = environment.step(
            action, state_np, absolute_step, flag=1
        )

        records["residential"].append(
            schedule_to_dict(
                environment.schedule_result_residential(), AGENT_COLUMNS["residential"]
            )
        )
        records["commercial"].append(
            schedule_to_dict(
                environment.schedule_result_commercial(), AGENT_COLUMNS["commercial"]
            )
        )
        records["industry"].append(
            schedule_to_dict(environment.schedule_result_industry(), AGENT_COLUMNS["industry"])
        )

        if done and hour < 23:
            raise RuntimeError(f"Episode ended early at hour {hour}.")

    return {
        agent: {key: np.array([row.get(key, 0.0) for row in rows]) for key in rows[0]}
        for agent, rows in records.items()
    }


def enrich_energy_terms(data, agent):
    enriched = {key: np.asarray(value, dtype=float) for key, value in data.items()}
    enriched["P_CER"] = enriched.get("C_CER", np.zeros(24)) / 4.0
    enriched["P_EB"] = enriched.get("H_EB", np.zeros(24)) / 0.95
    enriched["H_CHP"] = enriched.get("P_CHP", np.zeros(24)) * 0.4 / 0.45
    enriched["H_HCHP"] = enriched.get("P_HCHP", np.zeros(24)) * 0.4 / 0.45
    enriched["H_AC"] = enriched.get("C_AC", np.zeros(24)) / 0.7
    if agent == "industry":
        enriched["cooling_load"] = enriched.get("C_AC", np.zeros(24))
    return enriched


def build_signed_series(data, agent, energy_type):
    if energy_type == "electricity":
        return [
            ("PV", data["PV"]),
            ("CHP", data["P_CHP"]),
            ("HCHP", data["P_HCHP"]),
            ("ESS", data["P_ESS"]),
            ("Grid", data["buy_from_grid"] - data["sell_to_grid"]),
            ("Peer trade", data["trade_record_e"]),
            ("CER", -data["P_CER"]),
            ("EB", -data["P_EB"]),
        ]

    if energy_type == "heat":
        series = [
            ("CHP", data["H_CHP"]),
            ("HCHP", data["H_HCHP"]),
            ("TES", data["H_TES"]),
            ("AC", -data["H_AC"]),
        ]
        if agent in ("residential", "commercial"):
            series.insert(2, ("EB", data["H_EB"]))
        else:
            series.insert(2, ("GB", data["H_GB"]))
        return series

    if energy_type == "cooling":
        series = [
            ("AC", data["C_AC"]),
            ("CER", data.get("C_CER", np.zeros(24))),
        ]
        return series

    raise ValueError(f"Unknown energy type: {energy_type}")


def get_load_series(data, energy_type):
    if energy_type == "electricity":
        return "Electricity load", data["electric_load"]
    if energy_type == "heat":
        return "Heat load", data["heat_load"]
    if energy_type == "cooling":
        return "Cooling load", data["cooling_load"]
    raise ValueError(f"Unknown energy type: {energy_type}")


def max_series_residual(series, load):
    return float(np.max(np.abs(np.sum([values for _, values in series], axis=0) - load)))


def add_balance_closure(series, load, energy_type):
    residual = load - np.sum([values for _, values in series], axis=0)
    if np.max(np.abs(residual)) <= 1e-6:
        return series
    label = "Heat surplus" if energy_type == "heat" else "Balance residual"
    return series + [(label, residual)]


def stacked_signed_bars(ax, x, series, width=0.75):
    positive_bottom = np.zeros_like(x, dtype=float)
    negative_bottom = np.zeros_like(x, dtype=float)
    for label, values in series:
        values = np.asarray(values)
        pos = np.where(values > 0, values, 0.0)
        neg = np.where(values < 0, values, 0.0)
        shown_label = False
        if np.any(pos):
            ax.bar(
                x,
                pos,
                width=width,
                bottom=positive_bottom,
                color=COLORS[label],
                edgecolor="k",
                linewidth=0.8,
                hatch=HATCHES[label],
                label=label,
            )
            positive_bottom += pos
            shown_label = True
        if np.any(neg):
            ax.bar(
                x,
                neg,
                width=width,
                bottom=negative_bottom,
                color=COLORS[label],
                edgecolor="k",
                linewidth=0.8,
                hatch=HATCHES[label],
                label=None if shown_label else label,
            )
            negative_bottom += neg


def style_axis(ax, ylabel):
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Time (h)", labelpad=15)
    ax.set_ylabel(ylabel, labelpad=15)
    ax.set_xticks(np.arange(0, 24))
    ax.set_xlim(-1, 24)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_x(-0.075)
    offset_text.set_y(1.0)
    offset_text.set_ha("center")
    offset_text.set_va("bottom")
    ax.grid(True, which="both", linestyle="--", alpha=1.0, linewidth=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color("black")
    ax.tick_params(axis="both", which="major", length=10, width=2)
    ax.tick_params(axis="both", which="minor", length=6, width=1)


def pad_y_axis(ax, series, load, min_pad_ratio=0.12):
    positive_bottom = np.zeros_like(load, dtype=float)
    negative_bottom = np.zeros_like(load, dtype=float)
    for _, values in series:
        values = np.asarray(values, dtype=float)
        positive_bottom += np.where(values > 0, values, 0.0)
        negative_bottom += np.where(values < 0, values, 0.0)

    y_min = float(min(np.min(negative_bottom), np.min(load), 0.0))
    y_max = float(max(np.max(positive_bottom), np.max(load), 0.0))
    span = y_max - y_min
    if span <= 1e-9:
        span = max(abs(y_max), 1.0)
    pad = span * min_pad_ratio
    ax.set_ylim(y_min - pad, y_max + pad)


def legend_above(ax, ncol):
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label and label not in seen:
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)
    if unique_handles:
        ax.legend(
            unique_handles,
            unique_labels,
            ncol=ncol,
            framealpha=1,
            labelspacing=0.1,
            columnspacing=0.3,
            bbox_to_anchor=(0.58, 1.18),
            loc="upper center",
            frameon=False,
            facecolor="white",
        )


def plot_balance(data, preference_key, preference_row, season, agent, energy_type):
    preference_dir = OUTPUT_DIR / preference_key / season / agent
    preference_dir.mkdir(parents=True, exist_ok=True)
    policy_stem = Path(preference_row["policy_file"]).stem
    stem = preference_dir / f"{energy_type}_{preference_key}_{policy_stem}"
    png_path = stem.with_suffix(".png")

    x = np.arange(24)
    raw_series = build_signed_series(data, agent, energy_type)
    load_label, load = get_load_series(data, energy_type)
    series = add_balance_closure(raw_series, load, energy_type)
    title, ylabel = PLOTS[energy_type]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=600)
    stacked_signed_bars(ax, x, series)
    ax.plot(
        x,
        load,
        color=COLORS[load_label],
        label=load_label,
        lw=5 if energy_type == "electricity" else 4,
        marker="*",
        markersize=15,
        markerfacecolor="gold",
        markeredgecolor="k",
        markeredgewidth=1,
    )
    style_axis(ax, ylabel)
    pad_y_axis(ax, series, load)
    legend_above(ax, ncol=4 if energy_type == "electricity" else 3)

    fig.subplots_adjust(left=0.16, top=0.82)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.08, dpi=600)
    plt.close(fig)


def write_selection_summary(selected):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "selected_preferences.csv", "w", newline="") as fp:
        fieldnames = [
            "preference",
            "policy_file",
            "pareto_index",
            "cost",
            "carbon",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for preference, row in selected.items():
            record = {"preference": preference}
            record.update({key: row.get(key, "") for key in fieldnames if key != "preference"})
            writer.writerow(record)


def main():
    global RUN_DIR, OUTPUT_DIR, REPRESENTATIVES_CSV, MARKET_MECHANISM, CDA_CARBON_PRICE
    parser = argparse.ArgumentParser(description="Plot typical-day dispatch results.")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--representatives-csv", type=Path, default=REPRESENTATIVES_CSV)
    parser.add_argument(
        "--market-mechanism",
        default=MARKET_MECHANISM,
        choices=[
            "standard",
            "carbon_aware",
            "carbon_aware_v2",
            "carbon_dividend_v3",
            "carbon_dividend_equal_v3",
        ],
    )
    parser.add_argument("--cda-carbon-price", type=float, default=CDA_CARBON_PRICE)
    args = parser.parse_args()

    RUN_DIR = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    OUTPUT_DIR = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    REPRESENTATIVES_CSV = (
        args.representatives_csv
        if args.representatives_csv.is_absolute()
        else PROJECT_ROOT / args.representatives_csv
    )
    MARKET_MECHANISM = args.market_mechanism
    CDA_CARBON_PRICE = args.cda_carbon_price

    configure_style()
    ensure_test_arrays_have_terminal_row()
    if REPRESENTATIVES_CSV.exists():
        selected = select_preferences_from_representatives(REPRESENTATIVES_CSV)
    else:
        selected = select_preferences_from_objectives(RUN_DIR)
    write_selection_summary(selected)

    residuals = []
    for preference_key, row in selected.items():
        actor_dict = load_actor(row["policy_file"])
        for season, season_start in SEASONS.items():
            day_records = simulate_day(actor_dict, season_start)
            for agent, raw_data in day_records.items():
                data = enrich_energy_terms(raw_data, agent)
                for energy_type in PLOTS:
                    raw_series = build_signed_series(data, agent, energy_type)
                    _, load = get_load_series(data, energy_type)
                    plotted_series = add_balance_closure(raw_series, load, energy_type)
                    residuals.append(
                        (
                            preference_key,
                            season,
                            agent,
                            energy_type,
                            max_series_residual(raw_series, load),
                            max_series_residual(plotted_series, load),
                        )
                    )
                    plot_balance(data, preference_key, row, season, agent, energy_type)

    with open(OUTPUT_DIR / "balance_residuals.csv", "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "preference",
            "season",
            "agent",
            "energy_type",
            "raw_max_abs_residual",
            "plotted_max_abs_residual",
        ])
        writer.writerows(residuals)

    print(f"Saved dispatch figures to: {OUTPUT_DIR}")
    for preference_key, row in selected.items():
        print(
            f"{preference_key}: {row['policy_file']}, "
            f"cost={float(row['cost']):.6f}, carbon={float(row['carbon']):.6f}"
        )
    print(f"Maximum plotted balance residual: {max(residual[-1] for residual in residuals):.6e}")


if __name__ == "__main__":
    main()
