import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "anchor_evolutionary_maddpg_ablation"
SEEDS = [44, 11, 22, 33, 55]
VARIANTS = {
    "anchor_kappa0": ["--kappa-start", "0.0", "--kappa-end", "0.0"],
    "anchor_random": ["--explorer-selection", "random"],
    "all_evolving_maddpg": ["--anchor-mode", "evolving"],
}
SOURCE_FILES = [
    "scripts/train_anchor_evolutionary_maddpg.py",
    "scripts/evaluate_strict_algorithm_comparison.py",
    "scripts/analyze_anchor_evolutionary_maddpg_ablation.py",
    "src/envs/ies_env.py",
    "src/envs/market.py",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir(variant, seed):
    return RUN_ROOT / f"strict_v3_{variant}_seed_{seed}"


def build_command(python, variant, seed):
    command = [
        python,
        str(PROJECT_ROOT / "scripts" / "train_anchor_evolutionary_maddpg.py"),
        "--save-dir",
        str(run_dir(variant, seed)),
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
    return command + VARIANTS[variant]


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


def write_progress(variant, seed, state, finished, message=""):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": variant,
        "seed": seed,
        "state": state,
        "finished_runs": finished,
        "total_new_runs": len(VARIANTS) * len(SEEDS),
        "message": message,
        "updated_at_unix": time.time(),
    }
    (RESULT_ROOT / "progress.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def write_protocol(python):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "Five-seed ablation of Anchor-guided Evolutionary MADDPG",
        "seeds": sorted(SEEDS),
        "reused_groups": ["anchor_evolutionary_maddpg", "maddpg"],
        "new_variants": VARIANTS,
        "training_episodes_per_run": 972,
        "deterministic_evaluations_per_run": 972,
        "market_mechanism": "carbon_dividend_v3",
        "cda_carbon_price": 0.01,
        "source_sha256": {
            relative: sha256(PROJECT_ROOT / relative) for relative in SOURCE_FILES
        },
        "commands": {
            f"{variant}_seed_{seed}": build_command(python, variant, seed)
            for seed in SEEDS
            for variant in VARIANTS
        },
    }
    (RESULT_ROOT / "protocol.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def train(python, variant, seed, force=False):
    path = run_dir(variant, seed)
    if completed(path) and not force:
        print(f"SKIP {variant} seed={seed}: complete", flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (path / "train.log").open("w", encoding="utf-8") as stdout, (
        path / "train.err.log"
    ).open("w", encoding="utf-8") as stderr:
        command = build_command(python, variant, seed)
        stdout.write(" ".join(command) + "\n")
        stdout.flush()
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr)
    elapsed = time.time() - started
    (path / "launcher_elapsed_seconds.txt").write_text(f"{elapsed:.3f}\n", encoding="utf-8")
    if result.returncode != 0 or not completed(path):
        raise RuntimeError(f"{variant} seed={seed} failed; see {path / 'train.err.log'}")
    print(f"DONE {variant} seed={seed}: {elapsed / 3600.0:.2f} h", flush=True)


def evaluate(python, variant, seed):
    subprocess.run(
        [
            python,
            str(PROJECT_ROOT / "scripts" / "evaluate_strict_algorithm_comparison.py"),
            "--methods",
            variant,
            "--seeds",
            str(seed),
            "--force",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run formal Anchor E-MADDPG ablations.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    write_protocol(args.python)
    finished = 0
    try:
        for seed in SEEDS:
            for variant in VARIANTS:
                write_progress(variant, seed, "training", finished)
                train(args.python, variant, seed, force=args.force)
                write_progress(variant, seed, "evaluating", finished)
                evaluate(args.python, variant, seed)
                finished += 1
                write_progress(variant, seed, "run_complete", finished)
        write_progress(None, None, "analyzing", finished)
        subprocess.run(
            [
                args.python,
                str(PROJECT_ROOT / "scripts" / "analyze_anchor_evolutionary_maddpg_ablation.py"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        write_progress(None, None, "complete", finished)
    except Exception as error:
        write_progress(None, None, "failed", finished, str(error))
        raise


if __name__ == "__main__":
    main()
