import argparse
import calendar
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_strict_algorithm_comparison import (
    METHODS,
    actor_action,
    load_actors,
    mopso_action,
    read_metadata,
    run_dir,
)
from src.envs import ies_env
from src.envs.ies_env import env as IESEnv


PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SEEDS = [11, 22, 33, 44, 55]
HOURS_PER_YEAR = 8760
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def install_full_year_data():
    arrays = {}
    for area in ("residential", "commercial", "industry"):
        path = PROCESSED_DATA_DIR / f"full_year_data_{area}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; run scripts/prepare_full_year_data.py first"
            )
        values = np.load(path)
        if values.shape != (HOURS_PER_YEAR, 5):
            raise RuntimeError(f"Unexpected shape in {path}: {values.shape}")
        arrays[area] = values
        setattr(ies_env, f"state_test_{area}", np.vstack([values, values[-1:]]))
    return arrays


def initial_states(environment, arrays):
    specs = (
        (arrays["residential"], environment.residential_ESS, environment.residential_TES),
        (arrays["commercial"], environment.commercial_ESS, environment.commercial_TES),
        (arrays["industry"], environment.industry_ESS, environment.industry_TES),
    )
    return [
        np.append(values[0], [ess.reset(), tes.reset()]).reshape(1, -1)
        for values, ess, tes in specs
    ]


def month_for_hour(hour):
    day = hour // 24
    cumulative = 0
    for month, days in enumerate(MONTH_DAYS, start=1):
        cumulative += days
        if day < cumulative:
            return month
    raise ValueError(f"Hour outside non-leap year: {hour}")


def evaluate_policy(action_fn, market_mechanism, cda_carbon_price, arrays):
    environment = IESEnv(
        market_mechanism=market_mechanism,
        cda_carbon_price=cda_carbon_price,
        cda_dynamic_carbon_price=False,
    )
    states = initial_states(environment, arrays)
    total = np.zeros(2, dtype=np.float64)
    monthly = np.zeros((12, 2), dtype=np.float64)

    for hour in range(HOURS_PER_YEAR):
        action = action_fn(states, hour)
        next_res, next_com, next_ind, _, obj_r, obj_c, obj_i = environment.step(
            action,
            np.concatenate(states, axis=0),
            hour,
            flag=1,
        )
        objectives = np.asarray(obj_r) + np.asarray(obj_c) + np.asarray(obj_i)
        total += objectives
        monthly[month_for_hour(hour) - 1] += objectives
        states = [next_res, next_com, next_ind]
    return total, monthly


def evaluate_run(method, seed, force, market_mechanism, cda_carbon_price):
    arrays = install_full_year_data()
    path = run_dir(method, seed)
    final_dir = path / "final"
    output_dir = path / "annual_test"
    aggregate_path = output_dir / "objs.txt"
    if aggregate_path.exists() and not force:
        return f"SKIP {method} seed={seed}: annual result exists"
    if not (final_dir / "objs.txt").exists():
        raise FileNotFoundError(f"Training result is missing: {final_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(final_dir / "metadata.csv")
    positions = np.load(final_dir / "positions.npy") if method == "mopso" else None
    aggregate_rows = []
    monthly_rows = []

    for index, source in enumerate(metadata):
        if method == "mopso":
            position = positions[index]
            action_fn = lambda states, hour, position=position: mopso_action(position, hour)
            policy_file = ""
        else:
            policy_file = source["policy_file"]
            actors = load_actors(final_dir / policy_file)
            action_fn = lambda states, hour, actors=actors: actor_action(actors, states)

        total, monthly = evaluate_policy(
            action_fn,
            market_mechanism=market_mechanism,
            cda_carbon_price=cda_carbon_price,
            arrays=arrays,
        )
        aggregate_rows.append(
            {
                "pareto_index": index,
                "policy_file": policy_file,
                "annual_cost": total[0],
                "annual_carbon": total[1],
            }
        )
        for month in range(1, 13):
            monthly_rows.append(
                {
                    "pareto_index": index,
                    "policy_file": policy_file,
                    "month": month,
                    "month_name": calendar.month_name[month],
                    "cost": monthly[month - 1, 0],
                    "carbon": monthly[month - 1, 1],
                }
            )
        if (index + 1) % 10 == 0 or index + 1 == len(metadata):
            print(
                f"{method} seed={seed}: tested {index + 1}/{len(metadata)} policies",
                flush=True,
            )

    objectives = np.asarray(
        [[row["annual_cost"], row["annual_carbon"]] for row in aggregate_rows]
    )
    np.savetxt(aggregate_path, objectives, delimiter=",", fmt="%.6f")
    for filename, rows in (
        ("metadata.csv", aggregate_rows),
        ("monthly_detail.csv", monthly_rows),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return f"DONE {method} seed={seed}: {len(metadata)} policies"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved policies on the complete 8760-hour year."
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--market-mechanism",
        choices=["standard", "carbon_dividend_v3"],
        default="carbon_dividend_v3",
    )
    parser.add_argument("--cda-carbon-price", type=float, default=0.01)
    args = parser.parse_args()

    jobs = [(method, seed) for method in args.methods for seed in args.seeds]
    if args.workers <= 1:
        for method, seed in jobs:
            print(
                evaluate_run(
                    method,
                    seed,
                    args.force,
                    args.market_mechanism,
                    args.cda_carbon_price,
                ),
                flush=True,
            )
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                evaluate_run,
                method,
                seed,
                args.force,
                args.market_mechanism,
                args.cda_carbon_price,
            ): (method, seed)
            for method, seed in jobs
        }
        for future in as_completed(futures):
            method, seed = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                raise RuntimeError(f"Annual evaluation failed: {method} seed={seed}") from exc


if __name__ == "__main__":
    main()
