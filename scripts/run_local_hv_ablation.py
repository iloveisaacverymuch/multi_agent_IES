import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "annual_main_ablation"
SEEDS = [11, 22, 33, 44, 55]


def run_dir(seed):
    return RUN_ROOT / f"strict_v3_all_evolving_local_hv_seed_{seed}"


def command(python, seed):
    return [
        python,
        str(PROJECT_ROOT / "scripts" / "train_anchor_evolutionary_maddpg.py"),
        "--save-dir",
        str(run_dir(seed)),
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
        "--explorer-selection",
        "local_hv",
        "--kappa-start",
        "0.0",
        "--kappa-end",
        "0.0",
        "--market-mechanism",
        "carbon_dividend_v3",
        "--cda-carbon-price",
        "0.01",
    ]


def environment():
    result = os.environ.copy()
    result.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return result


def complete(seed):
    path = run_dir(seed)
    summary_path = path / "run_summary.json"
    if not summary_path.exists() or not (path / "final" / "metadata.csv").exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return summary.get("status") == "complete" and summary.get("training_episodes") == 972


def train(python, seed, force=False):
    path = run_dir(seed)
    if complete(seed) and not force:
        return {"seed": seed, "state": "skipped"}
    path.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (path / "train.log").open("w", encoding="utf-8") as stdout, (
        path / "train.err.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            command(python, seed),
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            env=environment(),
        )
    if result.returncode != 0 or not complete(seed):
        raise RuntimeError(f"Local-HV training failed for seed={seed}; see {path}")
    return {"seed": seed, "state": "trained", "seconds": time.time() - started}


def main():
    parser = argparse.ArgumentParser(description="Train the local-task HV ablation.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "name": "Local-task versus all-task Mean-HV ablation",
        "seeds": SEEDS,
        "training_episodes_per_run": 972,
        "definition": (
            "Local-task HV computes candidate HV gain against the current task's own "
            "objective history; the main method uses the shared archive from all tasks."
        ),
        "commands": {str(seed): command(args.python, seed) for seed in SEEDS},
    }
    (RESULT_ROOT / "local_hv_training_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(train, args.python, seed, args.force): seed for seed in SEEDS
        }
        for future in as_completed(futures):
            seed = futures[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"DONE local_hv seed={seed}: {row['state']}", flush=True)
            except Exception as error:
                failures.append({"seed": seed, "error": str(error)})
                print(f"FAILED local_hv seed={seed}: {error}", flush=True)
    (RESULT_ROOT / "local_hv_training_progress.json").write_text(
        json.dumps({"completed": rows, "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"{len(failures)} Local-HV runs failed")


if __name__ == "__main__":
    main()
