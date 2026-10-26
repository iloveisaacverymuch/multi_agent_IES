import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "cda_algorithm_factorial"
SEEDS = [44, 11, 22, 33, 55]
METHODS = ["standard_all_evolving_kappa0", "standard_maddpg"]
SOURCE_FILES = [
    "src/envs/ies_env.py",
    "src/envs/market.py",
    "scripts/train_anchor_evolutionary_maddpg.py",
    "scripts/train_maddpg_baseline.py",
    "scripts/evaluate_strict_algorithm_comparison.py",
    "scripts/analyze_cda_algorithm_factorial.py",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir(method, seed):
    return RUN_ROOT / f"strict_{method}_seed_{seed}"


def command(python, method, seed):
    path = run_dir(method, seed)
    if method == "standard_all_evolving_kappa0":
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
            "--market-mechanism",
            "standard",
            "--cda-carbon-price",
            "0.01",
        ]
    return [
        python,
        str(PROJECT_ROOT / "scripts" / "train_maddpg_baseline.py"),
        "--save-dir",
        str(path),
        "--seed",
        str(seed),
        "--episodes",
        "162",
        "--num-weights",
        "6",
        "--device",
        "cpu",
        "--archive-every-episode",
        "--market-mechanism",
        "standard",
        "--cda-carbon-price",
        "0.01",
    ]


def completed(method, seed):
    path = run_dir(method, seed)
    marker = path / "factorial_completion.json"
    if not marker.exists() or not (path / "final" / "objs.txt").exists():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def progress(method, seed, state, finished, message=""):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "progress.json").write_text(
        json.dumps(
            {
                "method": method,
                "seed": seed,
                "state": state,
                "finished_runs": finished,
                "total_new_runs": len(SEEDS) * len(METHODS),
                "message": message,
                "updated_at_unix": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_protocol(python):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "2x2 factorial: CDA mechanism x MADDPG optimization",
        "seeds": sorted(SEEDS),
        "factors": {
            "market": ["standard", "carbon_dividend_v3"],
            "algorithm": ["fixed_weight_maddpg", "all_task_mean_hv_evolutionary_maddpg"],
        },
        "reused_v3_groups": ["all_evolving_kappa0", "maddpg"],
        "new_standard_groups": METHODS,
        "training_episodes_per_run": 972,
        "deterministic_evaluations_per_run": 972,
        "source_sha256": {
            relative: sha256(PROJECT_ROOT / relative) for relative in SOURCE_FILES
        },
        "commands": {
            f"{method}_seed_{seed}": command(python, method, seed)
            for seed in SEEDS
            for method in METHODS
        },
    }
    (RESULT_ROOT / "protocol.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def train(python, method, seed, force=False):
    path = run_dir(method, seed)
    if completed(method, seed) and not force:
        print(f"SKIP {method} seed={seed}: complete", flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (path / "train.log").open("w", encoding="utf-8") as stdout, (
        path / "train.err.log"
    ).open("w", encoding="utf-8") as stderr:
        cmd = command(python, method, seed)
        stdout.write(" ".join(cmd) + "\n")
        stdout.flush()
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr)
    elapsed = time.time() - started
    (path / "launcher_elapsed_seconds.txt").write_text(f"{elapsed:.3f}\n", encoding="utf-8")
    if result.returncode != 0 or not (path / "final" / "objs.txt").exists():
        raise RuntimeError(f"{method} seed={seed} failed; see {path / 'train.err.log'}")
    (path / "factorial_completion.json").write_text(
        json.dumps({"status": "complete", "elapsed_seconds": elapsed}, indent=2),
        encoding="utf-8",
    )
    print(f"DONE {method} seed={seed}: {elapsed / 3600.0:.2f} h", flush=True)


def evaluate(python, method, seed):
    subprocess.run(
        [
            python,
            str(PROJECT_ROOT / "scripts" / "evaluate_strict_algorithm_comparison.py"),
            "--methods",
            method,
            "--seeds",
            str(seed),
            "--market-mechanism",
            "standard",
            "--cda-carbon-price",
            "0.01",
            "--force",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run CDA x algorithm factorial experiment.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_protocol(args.python)
    finished = 0
    try:
        for seed in SEEDS:
            for method in METHODS:
                progress(method, seed, "training", finished)
                train(args.python, method, seed, force=args.force)
                progress(method, seed, "evaluating", finished)
                evaluate(args.python, method, seed)
                finished += 1
                progress(method, seed, "run_complete", finished)
        progress(None, None, "analyzing", finished)
        subprocess.run(
            [args.python, str(PROJECT_ROOT / "scripts" / "analyze_cda_algorithm_factorial.py")],
            cwd=PROJECT_ROOT,
            check=True,
        )
        progress(None, None, "complete", finished)
    except Exception as error:
        progress(None, None, "failed", finished, str(error))
        raise


if __name__ == "__main__":
    main()
