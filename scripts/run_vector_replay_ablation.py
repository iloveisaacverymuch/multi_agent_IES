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
RESULT_ROOT = PROJECT_ROOT / "outputs" / "vector_replay_ablation"
SEEDS = [11, 22, 33, 44, 55]
TRAIN_MODES = ["scalar_stale", "clear_on_change"]


def run_dir(mode, seed):
    if mode == "vector_relabel":
        return RUN_ROOT / f"strict_v3_all_evolving_kappa0_seed_{seed}"
    return RUN_ROOT / f"strict_v3_replay_{mode}_seed_{seed}"


def train_command(python, mode, seed, path):
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
        mode,
        "--market-mechanism",
        "carbon_dividend_v3",
        "--cda-carbon-price",
        "0.01",
    ]


def evaluate_command(python, path):
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
        "0.01",
        "--force",
    ]


def completed(mode, seed):
    path = run_dir(mode, seed)
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


def run_case(python, mode, seed, force=False):
    path = run_dir(mode, seed)
    if completed(mode, seed) and not force:
        return {"mode": mode, "seed": seed, "state": "skipped"}
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (path / "train.log").open("w", encoding="utf-8") as stdout, (
        path / "train.err.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            train_command(python, mode, seed, path),
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            env=subprocess_env(),
        )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for mode={mode}, seed={seed}: {path}")
    (path / "launcher_elapsed_seconds.txt").write_text(
        f"{time.time() - started:.3f}\n", encoding="utf-8"
    )
    result = subprocess.run(
        evaluate_command(python, path),
        cwd=PROJECT_ROOT,
        env=subprocess_env(),
    )
    if result.returncode != 0 or not completed(mode, seed):
        raise RuntimeError(f"Strict evaluation failed for mode={mode}, seed={seed}: {path}")
    (path / "replay_ablation_completion.json").write_text(
        json.dumps({"status": "complete", "mode": mode, "seed": seed}, indent=2),
        encoding="utf-8",
    )
    return {"mode": mode, "seed": seed, "state": "trained"}


def write_protocol(python, workers):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "Dynamic-preference replay ablation",
        "seeds": SEEDS,
        "methods": ["vector_relabel", *TRAIN_MODES],
        "reused_method": "vector_relabel",
        "training_budget": {
            "generations": 12,
            "warmup_episodes": 30,
            "generation_episodes": 12,
            "tasks": 6,
            "evaluations_per_run": 972,
        },
        "workers": workers,
        "commands": {
            f"{mode}_seed_{seed}": train_command(python, mode, seed, run_dir(mode, seed))
            for mode in TRAIN_MODES
            for seed in SEEDS
        },
    }
    (RESULT_ROOT / "protocol.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description="Run the vector replay ablation.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_protocol(args.python, args.workers)
    cases = [(mode, seed) for mode in TRAIN_MODES for seed in SEEDS]
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_case, args.python, mode, seed, args.force): (mode, seed)
            for mode, seed in cases
        }
        for future in as_completed(futures):
            mode, seed = futures[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"DONE mode={mode}, seed={seed}: {row['state']}", flush=True)
            except Exception as error:
                failures.append({"mode": mode, "seed": seed, "error": str(error)})
                print(f"FAILED mode={mode}, seed={seed}: {error}", flush=True)
    (RESULT_ROOT / "progress.json").write_text(
        json.dumps(
            {"completed": rows, "failures": failures, "updated_at_unix": time.time()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"{len(failures)} replay-ablation runs failed")
    subprocess.run(
        [args.python, str(PROJECT_ROOT / "scripts" / "analyze_vector_replay_ablation.py")],
        cwd=PROJECT_ROOT,
        check=True,
        env=subprocess_env(),
    )


if __name__ == "__main__":
    main()
