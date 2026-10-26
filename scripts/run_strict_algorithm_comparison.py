import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
PROTOCOL_ROOT = PROJECT_ROOT / "outputs" / "strict_algorithm_comparison"
SEEDS = [11, 22, 33, 44, 55]
METHODS = ["proposed", "fixed_weight_mappo", "masac", "maddpg", "mopso"]
SOURCE_FILES = [
    "src/envs/ies_env.py",
    "src/envs/market.py",
    "src/emomarl/arguments.py",
    "src/emomarl/mappo.py",
    "src/emomarl/masac.py",
    "src/emomarl/mopg.py",
    "src/emomarl/population.py",
    "src/emomarl/runner.py",
    "src/emomarl/reward_utils.py",
    "scripts/train.py",
    "scripts/train_maddpg_baseline.py",
    "scripts/run_mopso_baseline.py",
    "scripts/run_strict_algorithm_comparison.py",
    "scripts/evaluate_strict_algorithm_comparison.py",
    "scripts/analyze_strict_algorithm_comparison.py",
    "scripts/run_strict_algorithm_pipeline.py",
]


@dataclass(frozen=True)
class Experiment:
    method: str
    seed: int
    save_dir: Path
    command: list[str]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir(method, seed):
    return RUN_ROOT / f"strict_v3_{method}_seed_{seed}"


def common_train_command(python, algorithm, seed, save_dir):
    return [
        python,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--algorithm",
        algorithm,
        "--seed",
        str(seed),
        "--device",
        "cpu",
        "--train_reward_transform",
        "none",
        "--use_reward_norm",
        "false",
        "--market_mechanism",
        "carbon_dividend_v3",
        "--cda_carbon_price",
        "0.01",
        "--cda_dynamic_carbon_price",
        "false",
        "--selection_method",
        "hv-ensemble-guided",
        "--elite_uncertainty_kappa",
        "0.10",
        "--elite_ensemble_max_samples",
        "32",
        "--elite_extreme_protection",
        "true",
        "--save_dir",
        str(save_dir),
    ]


def build_experiment(method, seed, python):
    save_dir = run_dir(method, seed)
    if method == "proposed":
        command = common_train_command(python, "mappo", seed, save_dir)
    elif method == "fixed_weight_mappo":
        command = common_train_command(python, "fixed_weight_mappo", seed, save_dir)
    elif method == "masac":
        command = common_train_command(python, "masac", seed, save_dir)
        command.extend(["--masac_diverse_task_selection", "false"])
    elif method == "maddpg":
        command = [
            python,
            str(PROJECT_ROOT / "scripts" / "train_maddpg_baseline.py"),
            "--save-dir",
            str(save_dir),
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
            "carbon_dividend_v3",
            "--cda-carbon-price",
            "0.01",
        ]
    elif method == "mopso":
        command = [
            python,
            str(PROJECT_ROOT / "scripts" / "run_mopso_baseline.py"),
            "--save-dir",
            str(save_dir),
            "--seed",
            str(seed),
            "--particles",
            "36",
            "--iterations",
            "53",
            "--archive-size",
            "120",
            "--decision-horizon",
            "24",
            "--eval-horizon",
            "1440",
            "--market-mechanism",
            "carbon_dividend_v3",
            "--cda-carbon-price",
            "0.01",
        ]
    else:
        raise ValueError(f"Unknown method: {method}")
    return Experiment(method, seed, save_dir, command)


def write_protocol(experiments):
    PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "Strict five-seed cross-algorithm comparison",
        "seeds": SEEDS,
        "methods": METHODS,
        "market_mechanism": "carbon_dividend_v3",
        "cda_carbon_price": 0.01,
        "reward": "raw negative cost/carbon scalarization",
        "training_horizon_hours": 1440,
        "test_horizon_hours": 96,
        "rl_training_episodes": 972,
        "rl_deterministic_archive_evaluations": 972,
        "total_full_trajectory_interactions": 1944,
        "mopso_particles": 36,
        "mopso_iterations": 53,
        "mopso_full_trajectory_evaluations": 1944,
        "maddpg_archive": "deterministic evaluation after every training episode",
        "masac_outer_selection": "matched HV-Ensemble; diverse-task shortcut disabled",
        "source_sha256": {
            relative: sha256(PROJECT_ROOT / relative) for relative in SOURCE_FILES
        },
        "commands": [
            {
                "method": exp.method,
                "seed": exp.seed,
                "save_dir": str(exp.save_dir.relative_to(PROJECT_ROOT)),
                "command": exp.command,
            }
            for exp in experiments
        ],
    }
    (PROTOCOL_ROOT / "protocol.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_experiment(experiment, force=False):
    final_obj = experiment.save_dir / "final" / "objs.txt"
    if final_obj.exists() and not force:
        print(f"SKIP {experiment.method} seed={experiment.seed}: result exists", flush=True)
        return
    experiment.save_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = experiment.save_dir / "train.log"
    stderr_path = experiment.save_dir / "train.err.log"
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        stdout.write(" ".join(experiment.command) + "\n")
        stdout.flush()
        result = subprocess.run(
            experiment.command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr
        )
    elapsed = time.time() - start
    (experiment.save_dir / "launcher_elapsed_seconds.txt").write_text(
        f"{elapsed:.3f}\n", encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{experiment.method} seed={experiment.seed} failed with exit code "
            f"{result.returncode}; see {stderr_path}"
        )
    print(
        f"DONE {experiment.method} seed={experiment.seed}: {elapsed / 3600:.2f} h",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Run the strict algorithm comparison.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiments = [
        build_experiment(method, seed, args.python)
        for method in args.methods
        for seed in args.seeds
    ]
    all_experiments = [
        build_experiment(method, seed, args.python)
        for method in METHODS
        for seed in SEEDS
    ]
    write_protocol(all_experiments)
    for experiment in experiments:
        print(f"RUN {experiment.method} seed={experiment.seed}", flush=True)
        run_experiment(experiment, force=args.force)


if __name__ == "__main__":
    main()
