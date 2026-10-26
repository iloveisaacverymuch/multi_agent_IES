import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "strict_algorithm_comparison"
STATUS_PATH = OUTPUT_DIR / "pipeline_status.json"


def write_status(phase, message=""):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(
        json.dumps(
            {
                "phase": phase,
                "message": message,
                "pid": os.getpid(),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run(command):
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main():
    python = sys.executable
    try:
        write_status("training", "Running 25 strict training jobs")
        run(
            [
                python,
                str(PROJECT_ROOT / "scripts" / "run_strict_algorithm_comparison.py"),
                "--python",
                python,
                "--methods",
                "mopso",
                "fixed_weight_mappo",
                "proposed",
                "masac",
                "maddpg",
            ]
        )
        write_status("testing", "Evaluating every archive on the common 96-hour test set")
        run([python, str(PROJECT_ROOT / "scripts" / "evaluate_strict_algorithm_comparison.py")])
        write_status("analysis", "Computing common-reference metrics and paired statistics")
        run([python, str(PROJECT_ROOT / "scripts" / "analyze_strict_algorithm_comparison.py")])
        write_status("complete", "Strict comparison completed successfully")
    except Exception as exc:
        write_status("failed", repr(exc))
        raise


if __name__ == "__main__":
    main()
