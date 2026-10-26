import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_maddpg_baseline import ACT_DIMS, OBS_DIMS, Actor as MADDPGActor
from src.envs import ies_env
from src.envs.ies_env import env as IESEnv


RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
SEEDS = [11, 22, 33, 44, 55]
METHODS = [
    "proposed",
    "fixed_weight_mappo",
    "masac",
    "maddpg",
    "mopso",
    "anchor_evolutionary_maddpg",
    "anchor_kappa0",
    "anchor_random",
    "all_evolving_maddpg",
    "all_evolving_kappa0",
    "all_evolving_random",
    "standard_maddpg",
    "standard_all_evolving_kappa0",
]
SEASONS = {"winter": 0, "spring": 24, "summer": 48, "autumn": 72}
JOINT_ACTION_DIM = 14


def run_dir(method, seed):
    if method == "standard_maddpg":
        return RUN_ROOT / f"strict_standard_maddpg_seed_{seed}"
    if method == "standard_all_evolving_kappa0":
        return RUN_ROOT / f"strict_standard_all_evolving_kappa0_seed_{seed}"
    return RUN_ROOT / f"strict_v3_{method}_seed_{seed}"


def ensure_terminal_test_row():
    for name in ("state_test_residential", "state_test_commercial", "state_test_industry"):
        values = getattr(ies_env, name)
        if len(values) == 96:
            setattr(ies_env, name, np.vstack([values, values[-1:]]))


def initial_states(environment, start):
    specs = (
        (ies_env.state_test_residential, environment.residential_ESS, environment.residential_TES),
        (ies_env.state_test_commercial, environment.commercial_ESS, environment.commercial_TES),
        (ies_env.state_test_industry, environment.industry_ESS, environment.industry_TES),
    )
    return [
        np.append(data[start], [ess.reset(), tes.reset()]).reshape(1, -1)
        for data, ess, tes in specs
    ]


def load_actors(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and payload.get("format") == "maddpg_state_dict_v1":
        actors = {}
        for agent_id, state_dict in enumerate(payload["actor_state_dicts"]):
            actor = MADDPGActor(OBS_DIMS[agent_id], ACT_DIMS[agent_id], payload["hidden_dim"])
            actor.load_state_dict(state_dict)
            actors[agent_id] = actor.eval()
        return actors
    for actor in payload.values():
        actor.eval()
    return payload


def actor_action(actors, states):
    actions = []
    with torch.no_grad():
        for agent_id, state in enumerate(states):
            state_tensor = torch.tensor(state, dtype=torch.float32).squeeze()
            try:
                output = actors[agent_id](state_tensor, evaluation=True)
            except TypeError:
                output = actors[agent_id](state_tensor)
            action = output[0] if isinstance(output, tuple) else output
            action = action.reshape(-1)
            if agent_id == 2:
                action = torch.cat([action, torch.tensor([0.0])])
            actions.append(action.reshape(1, -1))
    return torch.cat(actions, dim=0).numpy()


def mopso_action(position, hour):
    start = (hour % 24) * JOINT_ACTION_DIM
    values = position[start : start + JOINT_ACTION_DIM]
    return np.concatenate(
        [
            values[0:5].reshape(1, -1),
            values[5:10].reshape(1, -1),
            np.concatenate([values[10:14], np.array([0.0])]).reshape(1, -1),
        ],
        axis=0,
    )


def evaluate_day(action_fn, season_start, market_mechanism, cda_carbon_price):
    environment = IESEnv(
        market_mechanism=market_mechanism,
        cda_carbon_price=cda_carbon_price,
        cda_dynamic_carbon_price=False,
    )
    states = initial_states(environment, season_start)
    total = np.zeros(2, dtype=np.float64)
    for hour in range(24):
        action = action_fn(states, hour)
        next_res, next_com, next_ind, done, obj_r, obj_c, obj_i = environment.step(
            action,
            np.concatenate(states, axis=0),
            season_start + hour,
            flag=1,
        )
        total += np.asarray(obj_r) + np.asarray(obj_c) + np.asarray(obj_i)
        states = [next_res, next_com, next_ind]
        if done and hour < 23:
            raise RuntimeError(f"Test day ended early at hour {hour}")
    return total


def read_metadata(path):
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def evaluate_run(
    method,
    seed,
    force=False,
    market_mechanism="carbon_dividend_v3",
    cda_carbon_price=0.01,
    path_override=None,
):
    path = Path(path_override) if path_override is not None else run_dir(method, seed)
    display_name = path.name if path_override is not None else f"{method} seed={seed}"
    final_dir = path / "final"
    output_dir = path / "strict_test"
    aggregate_path = output_dir / "objs.txt"
    if aggregate_path.exists() and not force:
        print(f"SKIP {display_name}: strict test exists", flush=True)
        return
    if not (final_dir / "objs.txt").exists():
        raise FileNotFoundError(f"Training result is missing: {final_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(final_dir / "metadata.csv")
    positions = np.load(final_dir / "positions.npy") if method == "mopso" else None
    aggregate_rows = []
    season_rows = []

    for index, source in enumerate(metadata):
        if method == "mopso":
            position = positions[index]
            action_fn = lambda states, hour, position=position: mopso_action(position, hour)
            policy_file = ""
        else:
            policy_file = source["policy_file"]
            actors = load_actors(final_dir / policy_file)
            action_fn = lambda states, hour, actors=actors: actor_action(actors, states)
        total = np.zeros(2, dtype=np.float64)
        for season, start in SEASONS.items():
            result = evaluate_day(
                action_fn,
                start,
                market_mechanism=market_mechanism,
                cda_carbon_price=cda_carbon_price,
            )
            total += result
            season_rows.append(
                {
                    "pareto_index": index,
                    "policy_file": policy_file,
                    "season": season,
                    "cost": result[0],
                    "carbon": result[1],
                }
            )
        aggregate_rows.append(
            {
                "pareto_index": index,
                "policy_file": policy_file,
                "cost": total[0],
                "carbon": total[1],
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(metadata):
            print(f"{display_name}: tested {index + 1}/{len(metadata)}", flush=True)

    objectives = np.asarray([[row["cost"], row["carbon"]] for row in aggregate_rows])
    np.savetxt(aggregate_path, objectives, delimiter=",", fmt="%.6f")
    for filename, rows in (
        ("metadata.csv", aggregate_rows),
        ("season_detail.csv", season_rows),
    ):
        with (output_dir / filename).open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate strict comparison runs on 96 test hours.")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--market-mechanism",
        choices=["standard", "carbon_dividend_v3"],
        default="carbon_dividend_v3",
    )
    parser.add_argument("--cda-carbon-price", type=float, default=0.01)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Evaluate one explicit run directory instead of a named method/seed set.",
    )
    parser.add_argument(
        "--method-format",
        choices=["maddpg", "mopso"],
        default="maddpg",
        help="Policy format used with --run-dir.",
    )
    args = parser.parse_args()
    ensure_terminal_test_row()
    if args.run_dir is not None:
        path = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
        evaluate_run(
            args.method_format,
            0,
            force=args.force,
            market_mechanism=args.market_mechanism,
            cda_carbon_price=args.cda_carbon_price,
            path_override=path,
        )
        return
    for method in args.methods:
        for seed in args.seeds:
            evaluate_run(
                method,
                seed,
                force=args.force,
                market_mechanism=args.market_mechanism,
                cda_carbon_price=args.cda_carbon_price,
            )


if __name__ == "__main__":
    main()
