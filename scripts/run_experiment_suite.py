import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
MAIN_KAPPA = 0.20
MAIN_ENSEMBLE_MAX_SAMPLES = 32


@dataclass(frozen=True)
class Experiment:
    name: str
    save_dir: Path
    args: tuple[str, ...]


def price_tag(value):
    return f"{value:.2f}".replace(".", "_")


def base_args(seed=44):
    return (
        "--algorithm",
        "mappo",
        "--seed",
        str(seed),
        "--train_reward_transform",
        "none",
        "--use_reward_norm",
        "false",
    )


def carbon_price_experiments():
    experiments = []
    for price in [0.00, 0.02, 0.05, 0.08, 0.10]:
        if abs(price - 0.05) <= 1e-12:
            save_dir = RUN_ROOT / "mappo_carbon_cda_contribution_full"
        else:
            save_dir = RUN_ROOT / f"carbon_cda_contribution_price_{price_tag(price)}"
        experiments.append(
            Experiment(
                name=f"carbon_price_{price:.2f}",
                save_dir=save_dir,
                args=base_args()
                + (
                    "--market_mechanism",
                    "carbon_aware",
                    "--selection_method",
                    "contribution-guided",
                    "--cda_carbon_price",
                    str(price),
                    "--save_dir",
                    str(save_dir),
                ),
            )
        )
    return experiments


def new_main_ablation_experiments():
    configs = [
        (
            "standard_contribution",
            RUN_ROOT / "mappo_standard_cda_contribution_full",
            "standard",
            "contribution-guided",
            0.0,
        ),
        (
            "carbon_contribution_price_0.02",
            RUN_ROOT / "carbon_cda_contribution_price_0_02",
            "carbon_aware",
            "contribution-guided",
            0.02,
        ),
        (
            "standard_ensemble_uncertainty",
            RUN_ROOT / "standard_cda_ensemble_uncertainty_capped32_kappa_0_20",
            "standard",
            "ensemble-uncertainty-guided",
            0.0,
        ),
        (
            "carbon_ensemble_uncertainty_price_0.02",
            RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02",
            "carbon_aware",
            "ensemble-uncertainty-guided",
            0.02,
        ),
    ]
    experiments = []
    for name, save_dir, market, selection, carbon_price in configs:
        experiments.append(
            Experiment(
                name=name,
                save_dir=save_dir,
                args=base_args()
                + (
                    "--market_mechanism",
                    market,
                    "--selection_method",
                    selection,
                    "--cda_carbon_price",
                    str(carbon_price),
                    "--elite_uncertainty_kappa",
                    str(MAIN_KAPPA),
                    "--elite_ensemble_max_samples",
                    str(MAIN_ENSEMBLE_MAX_SAMPLES),
                    "--save_dir",
                    str(save_dir),
                ),
            )
        )
    return experiments


def new_main_ablation_3seed_experiments():
    seeds = [11, 22, 44]
    configs = [
        (
            "standard_contribution",
            "standard",
            "contribution-guided",
            0.0,
            {
                11: RUN_ROOT / "seed_stability_standard_contribution_seed_11",
                44: RUN_ROOT / "mappo_standard_cda_contribution_full",
            },
        ),
        (
            "carbon_contribution",
            "carbon_aware",
            "contribution-guided",
            0.02,
            {
                11: RUN_ROOT / "carbon_cda_contribution_price_0_02_seed_11",
                22: RUN_ROOT / "carbon_cda_contribution_price_0_02_seed_22",
                44: RUN_ROOT / "carbon_cda_contribution_price_0_02",
            },
        ),
        (
            "standard_ensemble",
            "standard",
            "ensemble-uncertainty-guided",
            0.0,
            {
                44: RUN_ROOT / "standard_cda_ensemble_uncertainty_capped32_kappa_0_20",
            },
        ),
        (
            "carbon_ensemble",
            "carbon_aware",
            "ensemble-uncertainty-guided",
            0.02,
            {
                11: RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02_seed_11",
                22: RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02_seed_22",
                44: RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02",
            },
        ),
    ]
    experiments = []
    for seed in seeds:
        for method_key, market, selection, carbon_price, existing in configs:
            save_dir = existing.get(
                seed,
                RUN_ROOT / f"{method_key}_capped32_kappa_0_20_price_0_02_seed_{seed}",
            )
            experiments.append(
                Experiment(
                    name=f"{method_key}_seed_{seed}",
                    save_dir=save_dir,
                    args=base_args(seed)
                    + (
                        "--market_mechanism",
                        market,
                        "--selection_method",
                        selection,
                        "--cda_carbon_price",
                        str(carbon_price),
                        "--elite_uncertainty_kappa",
                        str(MAIN_KAPPA),
                        "--elite_ensemble_max_samples",
                        str(MAIN_ENSEMBLE_MAX_SAMPLES),
                        "--save_dir",
                        str(save_dir),
                    ),
                )
            )
    return experiments


def ensemble_carbon_price_experiments():
    experiments = []
    for price in [0.00, 0.02, 0.05, 0.08, 0.10]:
        save_dir = (
            RUN_ROOT
            / f"carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_{price_tag(price)}"
        )
        experiments.append(
            Experiment(
                name=f"ensemble_carbon_price_{price:.2f}",
                save_dir=save_dir,
                args=base_args()
                + (
                    "--market_mechanism",
                    "carbon_aware",
                    "--selection_method",
                    "ensemble-uncertainty-guided",
                    "--cda_carbon_price",
                    str(price),
                    "--elite_uncertainty_kappa",
                    str(MAIN_KAPPA),
                    "--elite_ensemble_max_samples",
                    str(MAIN_ENSEMBLE_MAX_SAMPLES),
                    "--save_dir",
                    str(save_dir),
                ),
            )
        )
    return experiments


def v2_pilot_experiments():
    save_dir = (
        RUN_ROOT
        / "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02"
    )
    return [
        Experiment(
            name="carbon_cda_v2_hv_ensemble_seed_44",
            save_dir=save_dir,
            args=base_args()
            + (
                "--market_mechanism",
                "carbon_aware_v2",
                "--selection_method",
                "hv-ensemble-guided",
                "--cda_carbon_price",
                "0.02",
                "--elite_uncertainty_kappa",
                "0.10",
                "--elite_ensemble_max_samples",
                str(MAIN_ENSEMBLE_MAX_SAMPLES),
                "--elite_extreme_protection",
                "true",
                "--save_dir",
                str(save_dir),
            ),
        )
    ]


def v2_3seed_experiments():
    seed_44_dir = (
        RUN_ROOT
        / "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02"
    )
    experiments = []
    for seed in [11, 22, 44]:
        save_dir = (
            seed_44_dir
            if seed == 44
            else RUN_ROOT
            / (
                "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_"
                f"price_0_02_seed_{seed}"
            )
        )
        experiments.append(
            Experiment(
                name=f"carbon_cda_v2_hv_ensemble_seed_{seed}",
                save_dir=save_dir,
                args=base_args(seed)
                + (
                    "--market_mechanism",
                    "carbon_aware_v2",
                    "--selection_method",
                    "hv-ensemble-guided",
                    "--cda_carbon_price",
                    "0.02",
                    "--elite_uncertainty_kappa",
                    "0.10",
                    "--elite_ensemble_max_samples",
                    str(MAIN_ENSEMBLE_MAX_SAMPLES),
                    "--elite_extreme_protection",
                    "true",
                    "--save_dir",
                    str(save_dir),
                ),
            )
        )
    return experiments


def v2_ablation_3seed_experiments():
    configs = (
        (
            "current_main",
            "carbon_aware",
            "ensemble-uncertainty-guided",
            0.20,
            "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02",
            False,
        ),
        (
            "endpoint_protection",
            "carbon_aware",
            "ensemble-uncertainty-extreme-guided",
            0.20,
            "carbon_cda_v1_ensemble_extreme_capped32_kappa_0_20_price_0_02",
            True,
        ),
        (
            "hv_ensemble",
            "carbon_aware",
            "hv-ensemble-guided",
            0.10,
            "carbon_cda_v1_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02",
            True,
        ),
        (
            "full_v2",
            "carbon_aware_v2",
            "hv-ensemble-guided",
            0.10,
            "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02",
            True,
        ),
    )
    experiments = []
    for method_key, market, selection, kappa, base_name, protect_extremes in configs:
        for seed in [11, 22, 44]:
            save_dir = RUN_ROOT / (base_name if seed == 44 else f"{base_name}_seed_{seed}")
            experiments.append(
                Experiment(
                    name=f"v2_ablation_{method_key}_seed_{seed}",
                    save_dir=save_dir,
                    args=base_args(seed)
                    + (
                        "--market_mechanism",
                        market,
                        "--selection_method",
                        selection,
                        "--cda_carbon_price",
                        "0.02",
                        "--elite_uncertainty_kappa",
                        str(kappa),
                        "--elite_ensemble_max_samples",
                        str(MAIN_ENSEMBLE_MAX_SAMPLES),
                        "--elite_extreme_protection",
                        str(protect_extremes).lower(),
                        "--save_dir",
                        str(save_dir),
                    ),
                )
            )
    return experiments


def v2_cda_isolation_3seed_experiments():
    configs = (
        (
            "standard",
            "standard",
            "standard_cda_hv_ensemble_extreme_capped32_kappa_0_10",
        ),
        (
            "carbon_v1",
            "carbon_aware",
            "carbon_cda_v1_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02",
        ),
        (
            "carbon_v2",
            "carbon_aware_v2",
            "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02",
        ),
    )
    experiments = []
    for method_key, market, base_name in configs:
        for seed in [11, 22, 44]:
            save_dir = RUN_ROOT / (base_name if seed == 44 else f"{base_name}_seed_{seed}")
            experiments.append(
                Experiment(
                    name=f"v2_cda_isolation_{method_key}_seed_{seed}",
                    save_dir=save_dir,
                    args=base_args(seed)
                    + (
                        "--market_mechanism",
                        market,
                        "--selection_method",
                        "hv-ensemble-guided",
                        "--cda_carbon_price",
                        "0.02",
                        "--elite_uncertainty_kappa",
                        "0.10",
                        "--elite_ensemble_max_samples",
                        str(MAIN_ENSEMBLE_MAX_SAMPLES),
                        "--elite_extreme_protection",
                        "true",
                        "--save_dir",
                        str(save_dir),
                    ),
                )
            )
    return experiments


def v3_price_pilot_experiments():
    experiments = []
    for price in [0.01, 0.02, 0.05]:
        save_dir = (
            RUN_ROOT
            / (
                "carbon_dividend_v3_hv_ensemble_extreme_capped32_kappa_0_10_"
                f"price_{price_tag(price)}"
            )
        )
        experiments.append(
            Experiment(
                name=f"v3_price_pilot_{price:.2f}_seed_44",
                save_dir=save_dir,
                args=base_args(44)
                + (
                    "--market_mechanism",
                    "carbon_dividend_v3",
                    "--selection_method",
                    "hv-ensemble-guided",
                    "--cda_carbon_price",
                    str(price),
                    "--elite_uncertainty_kappa",
                    "0.10",
                    "--elite_ensemble_max_samples",
                    str(MAIN_ENSEMBLE_MAX_SAMPLES),
                    "--elite_extreme_protection",
                    "true",
                    "--save_dir",
                    str(save_dir),
                ),
            )
        )
    return experiments


def v3_3seed_followup_experiments():
    experiments = []
    for price in (0.01, 0.05):
        base_name = (
            "carbon_dividend_v3_hv_ensemble_extreme_capped32_kappa_0_10_"
            f"price_{price_tag(price)}"
        )
        for seed in (11, 22):
            save_dir = RUN_ROOT / f"{base_name}_seed_{seed}"
            experiments.append(
                Experiment(
                    name=f"v3_price_{price:.2f}_seed_{seed}",
                    save_dir=save_dir,
                    args=base_args(seed)
                    + (
                        "--market_mechanism",
                        "carbon_dividend_v3",
                        "--selection_method",
                        "hv-ensemble-guided",
                        "--cda_carbon_price",
                        str(price),
                        "--elite_uncertainty_kappa",
                        "0.10",
                        "--elite_ensemble_max_samples",
                        str(MAIN_ENSEMBLE_MAX_SAMPLES),
                        "--elite_extreme_protection",
                        "true",
                        "--save_dir",
                        str(save_dir),
                    ),
                )
            )
    return experiments


def v3_dividend_ablation_3seed_experiments():
    base_name = (
        "carbon_dividend_equal_v3_hv_ensemble_extreme_capped32_kappa_0_10_"
        "price_0_01"
    )
    experiments = []
    for seed in (11, 22, 44):
        save_dir = RUN_ROOT / (base_name if seed == 44 else f"{base_name}_seed_{seed}")
        experiments.append(
            Experiment(
                name=f"v3_equal_dividend_seed_{seed}",
                save_dir=save_dir,
                args=base_args(seed)
                + (
                    "--market_mechanism",
                    "carbon_dividend_equal_v3",
                    "--selection_method",
                    "hv-ensemble-guided",
                    "--cda_carbon_price",
                    "0.01",
                    "--elite_uncertainty_kappa",
                    "0.10",
                    "--elite_ensemble_max_samples",
                    str(MAIN_ENSEMBLE_MAX_SAMPLES),
                    "--elite_extreme_protection",
                    "true",
                    "--save_dir",
                    str(save_dir),
                ),
            )
        )
    return experiments


def v3_dividend_ablation_5seed_extension_experiments():
    configs = (
        (
            "standard",
            "standard",
            0.02,
            "standard_cda_hv_ensemble_extreme_capped32_kappa_0_10",
        ),
        (
            "equal",
            "carbon_dividend_equal_v3",
            0.01,
            "carbon_dividend_equal_v3_hv_ensemble_extreme_capped32_kappa_0_10_price_0_01",
        ),
        (
            "targeted",
            "carbon_dividend_v3",
            0.01,
            "carbon_dividend_v3_hv_ensemble_extreme_capped32_kappa_0_10_price_0_01",
        ),
    )
    experiments = []
    for seed in (33, 55):
        for method_key, market, carbon_price, base_name in configs:
            save_dir = RUN_ROOT / f"{base_name}_seed_{seed}"
            experiments.append(
                Experiment(
                    name=f"v3_dividend_5seed_extension_{method_key}_seed_{seed}",
                    save_dir=save_dir,
                    args=base_args(seed)
                    + (
                        "--market_mechanism",
                        market,
                        "--selection_method",
                        "hv-ensemble-guided",
                        "--cda_carbon_price",
                        str(carbon_price),
                        "--elite_uncertainty_kappa",
                        "0.10",
                        "--elite_ensemble_max_samples",
                        str(MAIN_ENSEMBLE_MAX_SAMPLES),
                        "--elite_extreme_protection",
                        "true",
                        "--save_dir",
                        str(save_dir),
                    ),
                )
            )
    return experiments


def v3_dual_module_ablation_5seed_experiments():
    configs = (
        (
            "standard_contribution",
            "standard",
            0.02,
            "standard_cda_contribution_current_matched",
        ),
        (
            "v3_contribution",
            "carbon_dividend_v3",
            0.01,
            "carbon_dividend_v3_contribution_current_matched_price_0_01",
        ),
    )
    experiments = []
    for seed in (11, 22, 33, 44, 55):
        for method_key, market, carbon_price, base_name in configs:
            save_dir = RUN_ROOT / f"{base_name}_seed_{seed}"
            experiments.append(
                Experiment(
                    name=f"v3_dual_module_{method_key}_seed_{seed}",
                    save_dir=save_dir,
                    args=base_args(seed)
                    + (
                        "--market_mechanism",
                        market,
                        "--selection_method",
                        "contribution-guided",
                        "--cda_carbon_price",
                        str(carbon_price),
                        "--elite_uncertainty_kappa",
                        "0.10",
                        "--elite_ensemble_max_samples",
                        str(MAIN_ENSEMBLE_MAX_SAMPLES),
                        "--elite_extreme_protection",
                        "true",
                        "--save_dir",
                        str(save_dir),
                    ),
                )
            )
    return experiments


def multiseed_experiments():
    methods = [
        (
            "standard_prediction",
            "standard",
            "prediction-guided",
            0.0,
            {44: RUN_ROOT / "mappo_raw_reward_full"},
        ),
        (
            "carbon_cda_only",
            "carbon_aware",
            "prediction-guided",
            0.05,
            {44: RUN_ROOT / "mappo_carbon_aware_cda_full"},
        ),
        (
            "standard_contribution",
            "standard",
            "contribution-guided",
            0.0,
            {44: RUN_ROOT / "mappo_standard_cda_contribution_full"},
        ),
        (
            "carbon_contribution",
            "carbon_aware",
            "contribution-guided",
            0.05,
            {44: RUN_ROOT / "mappo_carbon_cda_contribution_full"},
        ),
    ]
    experiments = []
    for seed in [11, 22, 33, 44, 55]:
        for method_key, market, selection, carbon_price, existing in methods:
            save_dir = existing.get(seed, RUN_ROOT / f"seed_stability_{method_key}_seed_{seed}")
            experiments.append(
                Experiment(
                    name=f"{method_key}_seed_{seed}",
                    save_dir=save_dir,
                    args=base_args(seed)
                    + (
                        "--market_mechanism",
                        market,
                        "--selection_method",
                        selection,
                        "--cda_carbon_price",
                        str(carbon_price),
                        "--save_dir",
                        str(save_dir),
                    ),
                )
            )
    return experiments


def get_experiments(suite):
    experiments = []
    if suite in ("carbon_price", "all"):
        experiments.extend(carbon_price_experiments())
    if suite in ("new_main_ablation", "all"):
        experiments.extend(new_main_ablation_experiments())
    if suite in ("new_main_ablation_3seed", "all"):
        experiments.extend(new_main_ablation_3seed_experiments())
    if suite in ("ensemble_carbon_price", "all"):
        experiments.extend(ensemble_carbon_price_experiments())
    if suite in ("v2_pilot", "all"):
        experiments.extend(v2_pilot_experiments())
    if suite in ("v2_3seed", "all"):
        experiments.extend(v2_3seed_experiments())
    if suite in ("v2_ablation_3seed", "all"):
        experiments.extend(v2_ablation_3seed_experiments())
    if suite in ("v2_cda_isolation_3seed", "all"):
        experiments.extend(v2_cda_isolation_3seed_experiments())
    if suite in ("v3_price_pilot", "all"):
        experiments.extend(v3_price_pilot_experiments())
    if suite in ("v3_3seed_followup", "all"):
        experiments.extend(v3_3seed_followup_experiments())
    if suite in ("v3_dividend_ablation_3seed", "all"):
        experiments.extend(v3_dividend_ablation_3seed_experiments())
    if suite in ("v3_dividend_ablation_5seed_extension", "all"):
        experiments.extend(v3_dividend_ablation_5seed_extension_experiments())
    if suite in ("v3_dual_module_ablation_5seed", "all"):
        experiments.extend(v3_dual_module_ablation_5seed_experiments())
    if suite in ("multiseed", "all"):
        experiments.extend(multiseed_experiments())
    return experiments


def final_exists(save_dir):
    return (save_dir / "final" / "objs.txt").exists()


def run_experiment(experiment, python_exe):
    experiment.save_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = experiment.save_dir / "train.log"
    stderr_path = experiment.save_dir / "train.err.log"
    started_at = time.time()
    command = [python_exe, str(TRAIN_SCRIPT), *experiment.args]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        stdout.write(" ".join(command) + "\n")
        stdout.flush()
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stdout, stderr=stderr)
    elapsed = time.time() - started_at
    elapsed_path = experiment.save_dir / "elapsed_seconds.txt"
    elapsed_path.write_text(f"{elapsed:.3f}\n", encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"{experiment.name} failed with exit code {result.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Run formal experiment suites sequentially.")
    parser.add_argument(
        "--suite",
        choices=[
            "carbon_price",
            "new_main_ablation",
            "new_main_ablation_3seed",
            "ensemble_carbon_price",
            "v2_pilot",
            "v2_3seed",
            "v2_ablation_3seed",
            "v2_cda_isolation_3seed",
            "v3_price_pilot",
            "v3_3seed_followup",
            "v3_dividend_ablation_3seed",
            "v3_dividend_ablation_5seed_extension",
            "v3_dual_module_ablation_5seed",
            "multiseed",
            "all",
        ],
        default="carbon_price",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Run at most this many non-skipped experiments.")
    parser.add_argument("--force", action="store_true", help="Run even when final/objs.txt already exists.")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    experiments = get_experiments(args.suite)
    ran = 0
    for experiment in experiments:
        if final_exists(experiment.save_dir) and not args.force:
            print(f"SKIP {experiment.name}: {experiment.save_dir / 'final' / 'objs.txt'} exists")
            continue
        if args.max_runs is not None and ran >= args.max_runs:
            print(f"STOP: reached --max-runs={args.max_runs}")
            break
        print(f"RUN  {experiment.name}: {experiment.save_dir}")
        elapsed = run_experiment(experiment, args.python)
        ran += 1
        print(f"DONE {experiment.name}: {elapsed / 60:.2f} min")
    print(f"Completed {ran} new experiments.")


if __name__ == "__main__":
    main()
