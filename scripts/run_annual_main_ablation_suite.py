import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "annual_main_ablation"
PROGRESS_PATH = OUTPUT_ROOT / "suite_progress.json"


def update_progress(phase, status, message=""):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(
            {
                "phase": phase,
                "status": status,
                "message": message,
                "updated_at_unix": time.time(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run(command, phase):
    update_progress(phase, "running")
    print(f"START {phase}: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    update_progress(phase, "complete")
    print(f"DONE {phase}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run the complete annual A0-A5 ablation suite.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--training-workers", type=int, default=5)
    parser.add_argument("--evaluation-workers", type=int, default=5)
    args = parser.parse_args()
    try:
        run(
            [
                args.python,
                str(PROJECT_ROOT / "scripts" / "run_local_hv_ablation.py"),
                "--python",
                args.python,
                "--workers",
                str(args.training_workers),
            ],
            "train_A4_local_hv",
        )
        run(
            [
                args.python,
                str(PROJECT_ROOT / "scripts" / "evaluate_annual_main_ablation.py"),
                "--workers",
                str(args.evaluation_workers),
            ],
            "evaluate_A0_A5_8760_hours",
        )
        run(
            [args.python, str(PROJECT_ROOT / "scripts" / "analyze_annual_main_ablation.py")],
            "analyze_and_plot",
        )
        run(
            [args.python, str(PROJECT_ROOT / "scripts" / "collect_results.py")],
            "collect_results",
        )
        update_progress("all", "complete")
    except Exception as error:
        update_progress("all", "failed", str(error))
        raise


if __name__ == "__main__":
    main()
