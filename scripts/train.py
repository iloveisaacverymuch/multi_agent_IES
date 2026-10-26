import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.emomarl.arguments import get_parser
from src.emomarl.runner import run


def main():
    dispatcher = argparse.ArgumentParser(add_help=False)
    dispatcher.add_argument(
        "--algorithm",
        default="all_task_evolutionary_maddpg",
    )
    selected, remaining = dispatcher.parse_known_args()
    if selected.algorithm in {
        "all_task_evolutionary_maddpg",
        "anchor_evolutionary_maddpg",
        "main",
    }:
        from scripts.train_anchor_evolutionary_maddpg import main as run_main_method

        run_main_method(remaining)
        return

    parser = get_parser()
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
