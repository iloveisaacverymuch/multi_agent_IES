import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
PROPOSED_SELECTION_METHOD = "ensemble-uncertainty-guided"
PROPOSED_RUN_NAME = "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02"


@dataclass(frozen=True)
class Experiment:
    name: str
    save_dir: Path
    command: list[str]


def python_path(args):
    return args.python or sys.executable


def train_py_command(args, algorithm, save_dir, selection_method="contribution-guided"):
    return [
        python_path(args),
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--algorithm",
        algorithm,
        "--seed",
        str(args.seed),
        "--train_reward_transform",
        "none",
        "--use_reward_norm",
        "false",
        "--market_mechanism",
        "carbon_aware",
        "--selection_method",
        selection_method,
        "--cda_carbon_price",
        str(args.cda_carbon_price),
        "--elite_uncertainty_kappa",
        "0.20",
        "--elite_ensemble_max_samples",
        "32",
        "--save_dir",
        str(save_dir),
    ]


def seeded_dir(base_name, seed):
    if seed == 44:
        legacy = {
            "proposed": PROPOSED_RUN_NAME,
            "fixed_weight_mappo": "baseline_fixed_weight_mappo",
            "masac": "baseline_masac",
            "maddpg": "baseline_maddpg",
            "mopso": "baseline_mopso",
        }
        return RUN_ROOT / legacy[base_name]
    if base_name == "proposed":
        return RUN_ROOT / f"{PROPOSED_RUN_NAME}_seed_{seed}"
    return RUN_ROOT / f"baseline_{base_name}_seed_{seed}"


def experiments(args):
    proposed_dir = seeded_dir("proposed", args.seed)
    fixed_dir = seeded_dir("fixed_weight_mappo", args.seed)
    masac_dir = seeded_dir("masac", args.seed)
    maddpg_dir = seeded_dir("maddpg", args.seed)
    mopso_dir = seeded_dir("mopso", args.seed)
    return [
        Experiment(
            "proposed",
            proposed_dir,
            train_py_command(args, "mappo", proposed_dir, PROPOSED_SELECTION_METHOD),
        ),
        Experiment(
            "fixed_weight_mappo",
            fixed_dir,
            train_py_command(args, "fixed_weight_mappo", fixed_dir),
        ),
        Experiment(
            "masac",
            masac_dir,
            train_py_command(args, "masac", masac_dir),
        ),
        Experiment(
            "maddpg",
            maddpg_dir,
            [
                python_path(args),
                str(PROJECT_ROOT / "scripts" / "train_maddpg_baseline.py"),
                "--save-dir",
                str(maddpg_dir),
                "--seed",
                str(args.seed),
                "--episodes",
                str(args.maddpg_episodes),
                "--num-weights",
                str(args.num_weights),
                "--market-mechanism",
                "carbon_aware",
                "--cda-carbon-price",
                str(args.cda_carbon_price),
            ],
        ),
        Experiment(
            "mopso",
            mopso_dir,
            [
                python_path(args),
                str(PROJECT_ROOT / "scripts" / "run_mopso_baseline.py"),
                "--save-dir",
                str(mopso_dir),
                "--seed",
                str(args.seed),
                "--particles",
                str(args.mopso_particles),
                "--iterations",
                str(args.mopso_iterations),
                "--archive-size",
                str(args.mopso_archive_size),
                "--market-mechanism",
                "carbon_aware",
                "--cda-carbon-price",
                str(args.cda_carbon_price),
            ],
        ),
    ]


def run_experiment(experiment):
    experiment.save_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = experiment.save_dir / "train.log"
    stderr_path = experiment.save_dir / "train.err.log"
    start = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        stdout.write(" ".join(experiment.command) + "\n")
        stdout.flush()
        result = subprocess.run(experiment.command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr)
    elapsed = time.time() - start
    (experiment.save_dir / "elapsed_seconds.txt").write_text(f"{elapsed:.3f}\n", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"{experiment.name} failed with exit code {result.returncode}; "
            f"see {stdout_path} and {stderr_path}"
        )
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run cross-algorithm comparison baselines.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["proposed", "fixed_weight_mappo", "masac", "maddpg", "mopso"],
        choices=["proposed", "fixed_weight_mappo", "masac", "maddpg", "mopso"],
    )
    parser.add_argument("--python", default=None)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--cda-carbon-price", type=float, default=0.02)
    parser.add_argument("--num-weights", type=int, default=6)
    parser.add_argument("--maddpg-episodes", type=int, default=162)
    parser.add_argument("--mopso-particles", type=int, default=48)
    parser.add_argument("--mopso-iterations", type=int, default=80)
    parser.add_argument("--mopso-archive-size", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = set(args.methods)
    for experiment in experiments(args):
        if experiment.name not in selected:
            continue
        final_obj = experiment.save_dir / "final" / "objs.txt"
        if final_obj.exists() and not args.force:
            print(f"SKIP {experiment.name}: {final_obj} exists")
            continue
        print(f"RUN  {experiment.name}: {experiment.save_dir}")
        elapsed = run_experiment(experiment)
        print(f"DONE {experiment.name}: {elapsed / 3600:.2f} h")


if __name__ == "__main__":
    main()
