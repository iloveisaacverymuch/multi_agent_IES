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

from scripts.evaluate_full_year_algorithm_comparison import (  # noqa: E402
    evaluate_policy,
    install_full_year_data,
)
from scripts.evaluate_strict_algorithm_comparison import (  # noqa: E402
    actor_action,
    load_actors,
    read_metadata,
)


RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
SEEDS = [11, 22, 33, 44, 55]
VARIANTS = {
    "A0_full": ("strict_v3_all_evolving_kappa0_seed_{seed}", "carbon_dividend_v3"),
    "A1_standard_cda": ("strict_standard_all_evolving_kappa0_seed_{seed}", "standard"),
    "A2_fixed_maddpg": ("strict_v3_maddpg_seed_{seed}", "carbon_dividend_v3"),
    "A3_random_preference": ("strict_v3_all_evolving_random_seed_{seed}", "carbon_dividend_v3"),
    "A4_local_hv": ("strict_v3_all_evolving_local_hv_seed_{seed}", "carbon_dividend_v3"),
    "A5_scalar_stale": ("strict_v3_replay_scalar_stale_seed_{seed}", "carbon_dividend_v3"),
}


def run_dir(variant, seed):
    return RUN_ROOT / VARIANTS[variant][0].format(seed=seed)


def evaluate_run(variant, seed, force=False):
    arrays = install_full_year_data()
    path = run_dir(variant, seed)
    final_dir = path / "final"
    output_dir = path / "annual_test"
    aggregate_path = output_dir / "objs.txt"
    if aggregate_path.exists() and not force:
        return f"SKIP {variant} seed={seed}: annual result exists"
    if not (final_dir / "metadata.csv").exists():
        raise FileNotFoundError(f"Training result is missing: {final_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(final_dir / "metadata.csv")
    mechanism = VARIANTS[variant][1]
    aggregate_rows = []
    monthly_rows = []
    for index, source in enumerate(metadata):
        policy_file = source["policy_file"]
        actors = load_actors(final_dir / policy_file)
        action_fn = lambda states, hour, actors=actors: actor_action(actors, states)
        total, monthly = evaluate_policy(
            action_fn,
            market_mechanism=mechanism,
            cda_carbon_price=0.01,
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
                f"{variant} seed={seed}: tested {index + 1}/{len(metadata)} policies",
                flush=True,
            )

    objectives = np.asarray(
        [[row["annual_cost"], row["annual_carbon"]] for row in aggregate_rows]
    )
    np.savetxt(aggregate_path, objectives, delimiter=",", fmt="%.6f")
    for filename, rows in (("metadata.csv", aggregate_rows), ("monthly_detail.csv", monthly_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return f"DONE {variant} seed={seed}: {len(metadata)} policies"


def main():
    parser = argparse.ArgumentParser(description="Evaluate A0-A5 on the true 8760-hour year.")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    jobs = [(variant, seed) for variant in args.variants for seed in args.seeds]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(evaluate_run, variant, seed, args.force): (variant, seed)
            for variant, seed in jobs
        }
        for future in as_completed(futures):
            variant, seed = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:
                raise RuntimeError(
                    f"Annual ablation evaluation failed: {variant} seed={seed}"
                ) from error


if __name__ == "__main__":
    main()
