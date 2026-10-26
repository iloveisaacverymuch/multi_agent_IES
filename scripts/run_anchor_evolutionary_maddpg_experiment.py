import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
PROTOCOL_ROOT = PROJECT_ROOT / "outputs" / "anchor_evolutionary_maddpg_experiment"
SEEDS = [44, 11, 22, 33, 55]
SOURCE_FILES = [
    "src/envs/ies_env.py",
    "src/envs/market.py",
    "src/emomarl/opt_graph.py",
    "src/emomarl/population.py",
    "scripts/train_maddpg_baseline.py",
    "scripts/train_anchor_evolutionary_maddpg.py",
    "scripts/evaluate_strict_algorithm_comparison.py",
    "scripts/analyze_anchor_evolutionary_maddpg.py",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir(seed):
    return RUN_ROOT / f"strict_v3_anchor_evolutionary_maddpg_seed_{seed}"


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
        "--market-mechanism",
        "carbon_dividend_v3",
        "--cda-carbon-price",
        "0.01",
    ]


def completed(path):
    summary_path = path / "run_summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "complete"
        and summary.get("training_episodes") == 972
        and (path / "final" / "objs.txt").exists()
    )


def write_protocol(python, seeds):
    PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "Carbon-Dividend CDA V3 + Anchor-guided Evolutionary MADDPG",
        "formal_seeds": seeds,
        "market_mechanism": "carbon_dividend_v3",
        "cda_carbon_price": 0.01,
        "reward": "raw negative cost/carbon scalarization from vector replay",
        "training_horizon_hours": 1440,
        "test_horizon_hours": 96,
        "fixed_anchor_cost_weights": [0.2, 0.4, 0.6, 0.8],
        "adaptive_explorer_tasks": 2,
        "episodes_per_task": 162,
        "training_episodes": 972,
        "deterministic_archive_evaluations": 972,
        "total_full_trajectory_interactions": 1944,
        "source_sha256": {
            relative: sha256(PROJECT_ROOT / relative) for relative in SOURCE_FILES
        },
        "commands": {str(seed): command(python, seed) for seed in seeds},
    }
    (PROTOCOL_ROOT / "protocol.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_progress(seed, state, completed_seeds, message=""):
    payload = {
        "current_seed": seed,
        "state": state,
        "completed_seeds": completed_seeds,
        "message": message,
        "updated_at_unix": time.time(),
    }
    (PROTOCOL_ROOT / "progress.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def train_seed(python, seed, force=False):
    path = run_dir(seed)
    if completed(path) and not force:
        print(f"SKIP seed={seed}: complete formal result exists", flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    train_log = path / "train.log"
    error_log = path / "train.err.log"
    started = time.time()
    with train_log.open("w", encoding="utf-8") as stdout, error_log.open(
        "w", encoding="utf-8"
    ) as stderr:
        cmd = command(python, seed)
        stdout.write(" ".join(cmd) + "\n")
        stdout.flush()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr)
    elapsed = time.time() - started
    (path / "launcher_elapsed_seconds.txt").write_text(f"{elapsed:.3f}\n", encoding="utf-8")
    if result.returncode != 0 or not completed(path):
        raise RuntimeError(f"seed={seed} training failed; see {error_log}")
    print(f"DONE seed={seed}: {elapsed / 3600.0:.2f} h", flush=True)


def evaluate_seed(python, seed):
    cmd = [
        python,
        str(PROJECT_ROOT / "scripts" / "evaluate_strict_algorithm_comparison.py"),
        "--methods",
        "anchor_evolutionary_maddpg",
        "--seeds",
        str(seed),
        "--force",
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run five-seed Anchor-guided E-MADDPG experiment.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_protocol(args.python, args.seeds)
    completed_seeds = []
    try:
        for seed in args.seeds:
            write_progress(seed, "training", completed_seeds)
            train_seed(args.python, seed, force=args.force)
            write_progress(seed, "evaluating", completed_seeds)
            evaluate_seed(args.python, seed)
            completed_seeds.append(seed)
            write_progress(seed, "seed_complete", completed_seeds)
        write_progress(None, "analyzing", completed_seeds)
        subprocess.run(
            [args.python, str(PROJECT_ROOT / "scripts" / "analyze_anchor_evolutionary_maddpg.py")],
            cwd=PROJECT_ROOT,
            check=True,
        )
        write_progress(None, "complete", completed_seeds)
    except Exception as error:
        write_progress(None, "failed", completed_seeds, str(error))
        raise


if __name__ == "__main__":
    main()
