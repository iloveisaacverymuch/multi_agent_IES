import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.envs.ies_env import env as IESEnv


ACTION_DIMS = [5, 5, 4]
JOINT_ACTION_DIM = sum(ACTION_DIMS)


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def nondominated_indices(objectives):
    dominated = np.zeros(len(objectives), dtype=bool)
    for i in range(len(objectives)):
        for j in range(len(objectives)):
            if i != j and dominates(objectives[j], objectives[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


def crowding_distance(objectives):
    n, m = objectives.shape
    if n <= 2:
        return np.full(n, np.inf)
    distance = np.zeros(n)
    for obj_id in range(m):
        order = np.argsort(objectives[:, obj_id])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = objectives[order[-1], obj_id] - objectives[order[0], obj_id]
        if span <= 1e-12:
            continue
        for rank in range(1, n - 1):
            distance[order[rank]] += (
                objectives[order[rank + 1], obj_id] - objectives[order[rank - 1], obj_id]
            ) / span
    return distance


def trim_archive(positions, objectives, max_size):
    if len(objectives) <= max_size:
        return positions, objectives
    distance = crowding_distance(objectives)
    keep = np.argsort(-distance)[:max_size]
    return positions[keep], objectives[keep]


def update_archive(archive_positions, archive_objectives, positions, objectives, max_size):
    if len(archive_objectives) == 0:
        combined_positions = positions.copy()
        combined_objectives = objectives.copy()
    else:
        combined_positions = np.vstack([archive_positions, positions])
        combined_objectives = np.vstack([archive_objectives, objectives])
    keep = nondominated_indices(combined_objectives)
    combined_positions = combined_positions[keep]
    combined_objectives = combined_objectives[keep]
    rounded = np.round(combined_objectives, decimals=3)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    combined_positions = combined_positions[unique_idx]
    combined_objectives = combined_objectives[unique_idx]
    return trim_archive(combined_positions, combined_objectives, max_size)


def make_env(args):
    return IESEnv(
        market_mechanism=args.market_mechanism,
        cda_carbon_price=args.cda_carbon_price,
    )


def vector_to_action(position, hour, decision_horizon):
    template_hour = hour % decision_horizon
    start = template_hour * JOINT_ACTION_DIM
    values = position[start : start + JOINT_ACTION_DIM]
    res = values[0:5].reshape(1, -1)
    com = values[5:10].reshape(1, -1)
    ind = np.concatenate([values[10:14], np.array([0.0])]).reshape(1, -1)
    return np.concatenate([res, com, ind], axis=0)


def evaluate_position(position, args):
    environment = make_env(args)
    state_res, state_com, state_ind = environment.reset_train()
    total = np.zeros(2, dtype=np.float64)
    done = False
    hour = 0
    while not done and hour < args.eval_horizon:
        state_np = np.vstack([state_res, state_com, state_ind])
        state_res, state_com, state_ind, done, obj_r, obj_c, obj_i = environment.step(
            vector_to_action(position, hour, args.decision_horizon), state_np, hour, flag=0
        )
        total += obj_r + obj_c + obj_i
        hour += 1
    return total


def evaluate_population(positions, args):
    return np.asarray([evaluate_position(position, args) for position in positions])


def choose_leaders(archive_positions, archive_objectives, count):
    distance = crowding_distance(archive_objectives)
    finite = np.where(np.isfinite(distance), distance, np.nanmax(distance[np.isfinite(distance)]) + 1.0 if np.isfinite(distance).any() else 1.0)
    probs = finite / max(float(finite.sum()), 1e-12)
    indices = np.random.choice(len(archive_positions), size=count, replace=True, p=probs)
    return archive_positions[indices]


def update_personal_best(pbest_pos, pbest_obj, positions, objectives):
    for i in range(len(positions)):
        if dominates(objectives[i], pbest_obj[i]):
            pbest_pos[i] = positions[i]
            pbest_obj[i] = objectives[i]
        elif not dominates(pbest_obj[i], objectives[i]) and random.random() < 0.5:
            pbest_pos[i] = positions[i]
            pbest_obj[i] = objectives[i]
    return pbest_pos, pbest_obj


def main():
    parser = argparse.ArgumentParser(description="Run MOPSO offline dispatch baseline.")
    parser.add_argument("--save-dir", type=Path, default=PROJECT_ROOT / "outputs" / "runs" / "baseline_mopso")
    parser.add_argument("--particles", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--archive-size", type=int, default=120)
    parser.add_argument("--decision-horizon", type=int, default=24)
    parser.add_argument("--eval-horizon", type=int, default=1440)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--w", type=float, default=0.55)
    parser.add_argument("--c1", type=float, default=1.6)
    parser.add_argument("--c2", type=float, default=1.6)
    parser.add_argument("--vmax", type=float, default=0.2)
    parser.add_argument(
        "--market-mechanism",
        choices=[
            "standard",
            "carbon_aware",
            "carbon_aware_v2",
            "carbon_dividend_v3",
            "carbon_dividend_equal_v3",
        ],
        default="carbon_aware",
    )
    parser.add_argument("--cda-carbon-price", type=float, default=0.02)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    save_dir = args.save_dir if args.save_dir.is_absolute() else PROJECT_ROOT / args.save_dir
    final_dir = save_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    dim = args.decision_horizon * JOINT_ACTION_DIM
    positions = np.random.uniform(-1.0, 1.0, size=(args.particles, dim)).astype(np.float32)
    velocities = np.random.uniform(-args.vmax, args.vmax, size=(args.particles, dim)).astype(np.float32)
    objectives = evaluate_population(positions, args)
    pbest_pos = positions.copy()
    pbest_obj = objectives.copy()
    archive_pos, archive_obj = update_archive(
        np.empty((0, dim), dtype=np.float32),
        np.empty((0, 2), dtype=np.float64),
        positions,
        objectives,
        args.archive_size,
    )
    history_rows = []

    for iteration in range(1, args.iterations + 1):
        leaders = choose_leaders(archive_pos, archive_obj, args.particles)
        r1 = np.random.rand(args.particles, dim)
        r2 = np.random.rand(args.particles, dim)
        velocities = (
            args.w * velocities
            + args.c1 * r1 * (pbest_pos - positions)
            + args.c2 * r2 * (leaders - positions)
        )
        velocities = np.clip(velocities, -args.vmax, args.vmax)
        positions = np.clip(positions + velocities, -1.0, 1.0)
        objectives = evaluate_population(positions, args)
        pbest_pos, pbest_obj = update_personal_best(pbest_pos, pbest_obj, positions, objectives)
        archive_pos, archive_obj = update_archive(
            archive_pos, archive_obj, positions, objectives, args.archive_size
        )
        history_rows.append(
            {
                "iteration": iteration,
                "archive_size": len(archive_obj),
                "min_cost": float(np.min(archive_obj[:, 0])),
                "min_carbon": float(np.min(archive_obj[:, 1])),
            }
        )
        if iteration % 10 == 0 or iteration == 1:
            print(
                f"iter {iteration}: archive={len(archive_obj)}, "
                f"min_cost={np.min(archive_obj[:, 0]):.3f}, min_carbon={np.min(archive_obj[:, 1]):.3f}"
            )

    order = np.argsort(archive_obj[:, 0])
    archive_pos = archive_pos[order]
    archive_obj = archive_obj[order]
    np.savetxt(final_dir / "objs.txt", archive_obj, delimiter=",", fmt="%.6f")
    np.save(final_dir / "positions.npy", archive_pos)
    with (final_dir / "metadata.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["pareto_index", "policy_file", "cost", "carbon"])
        writer.writeheader()
        for idx, obj in enumerate(archive_obj):
            writer.writerow(
                {
                    "pareto_index": idx,
                    "policy_file": "",
                    "cost": obj[0],
                    "carbon": obj[1],
                }
            )
    with (save_dir / "history.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)
    print(f"Saved MOPSO baseline to {save_dir}")


if __name__ == "__main__":
    main()
