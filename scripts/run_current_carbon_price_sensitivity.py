import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "current_carbon_price_sensitivity"
PRICES = [0.0, 0.005, 0.01, 0.02, 0.05]
SEEDS = [11, 22, 44]


def price_code(price):
    return f"{price:.3f}".replace(".", "p")


def run_dir(price, seed):
    if abs(price - 0.01) <= 1e-12:
        return RUN_ROOT / f"strict_v3_all_evolving_kappa0_seed_{seed}"
    return RUN_ROOT / f"strict_v3_all_evolving_kappa0_price_{price_code(price)}_seed_{seed}"


def train_command(python, price, seed, path):
    return [
        python,
        str(PROJECT_ROOT / "scripts" / "train_anchor_evolutionary_maddpg.py"),
        "--save-dir",
        str(path),
        "--seed",
        str(seed),
        "--device",
        "cpu",
        "--generations",
        "12",
        "--warmup-episodes",
        "30",
        "--generation-episodes",
        "12",
        "--anchor-mode",
        "evolving",
        "--kappa-start",
        "0.0",
        "--kappa-end",
        "0.0",
        "--replay-mode",
        "vector_relabel",
        "--market-mechanism",
        "carbon_dividend_v3",
        "--cda-carbon-price",
        str(price),
    ]


def evaluate_command(python, price, path):
    return [
        python,
        str(PROJECT_ROOT / "scripts" / "evaluate_strict_algorithm_comparison.py"),
        "--run-dir",
        str(path),
        "--method-format",
        "maddpg",
        "--market-mechanism",
        "carbon_dividend_v3",
        "--cda-carbon-price",
        str(price),
        "--force",
    ]


def completed(price, seed):
    path = run_dir(price, seed)
    return (
        (path / "final" / "metadata.csv").exists()
        and (path / "strict_test" / "objs.txt").exists()
    )


def subprocess_env():
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return env


def run_case(python, price, seed, force=False):
    path = run_dir(price, seed)
    reused = abs(price - 0.01) <= 1e-12
    if completed(price, seed) and not force:
        return {"price": price, "seed": seed, "state": "reused" if reused else "skipped"}
    if reused and not (path / "final" / "metadata.csv").exists():
        raise FileNotFoundError(f"Canonical price-0.01 run is missing: {path}")
    if not reused:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with (path / "train.log").open("w", encoding="utf-8") as stdout, (
            path / "train.err.log"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                train_command(python, price, seed, path),
                cwd=PROJECT_ROOT,
                stdout=stdout,
                stderr=stderr,
                env=subprocess_env(),
            )
        if result.returncode != 0:
            raise RuntimeError(f"Training failed for price={price}, seed={seed}: {path}")
        (path / "launcher_elapsed_seconds.txt").write_text(
            f"{time.time() - started:.3f}\n", encoding="utf-8"
        )
    result = subprocess.run(
        evaluate_command(python, price, path),
        cwd=PROJECT_ROOT,
        env=subprocess_env(),
    )
    if result.returncode != 0 or not completed(price, seed):
        raise RuntimeError(f"Strict evaluation failed for price={price}, seed={seed}: {path}")
    (path / "carbon_price_completion.json").write_text(
        json.dumps({"status": "complete", "price": price, "seed": seed}, indent=2),
        encoding="utf-8",
    )
    return {"price": price, "seed": seed, "state": "reused" if reused else "trained"}


def write_protocol(python, workers):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "Current main-method carbon-price sensitivity",
        "prices": PRICES,
        "seeds": SEEDS,
        "reused_price": 0.01,
        "training_budget": {
            "generations": 12,
            "warmup_episodes": 30,
            "generation_episodes": 12,
            "tasks": 6,
            "evaluations_per_run": 972,
        },
        "workers": workers,
        "commands": {
            f"price_{price_code(price)}_seed_{seed}": train_command(
                python, price, seed, run_dir(price, seed)
            )
            for price in PRICES
            for seed in SEEDS
            if abs(price - 0.01) > 1e-12
        },
    }
    (RESULT_ROOT / "protocol.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Run current-method carbon-price sensitivity.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_protocol(args.python, args.workers)
    cases = [(price, seed) for price in PRICES for seed in SEEDS]
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_case, args.python, price, seed, args.force): (price, seed)
            for price, seed in cases
        }
        for future in as_completed(futures):
            price, seed = futures[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"DONE price={price:.3f}, seed={seed}: {row['state']}", flush=True)
            except Exception as error:
                failures.append({"price": price, "seed": seed, "error": str(error)})
                print(f"FAILED price={price:.3f}, seed={seed}: {error}", flush=True)
    (RESULT_ROOT / "progress.json").write_text(
        json.dumps(
            {"completed": rows, "failures": failures, "updated_at_unix": time.time()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"{len(failures)} carbon-price runs failed")
    subprocess.run(
        [args.python, str(PROJECT_ROOT / "scripts" / "analyze_current_carbon_price_sensitivity.py")],
        cwd=PROJECT_ROOT,
        check=True,
        env=subprocess_env(),
    )


if __name__ == "__main__":
    main()
