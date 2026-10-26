import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"


def main():
    parser = argparse.ArgumentParser(description="Plot strict episode reward curves from training logs.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or run_dir.name

    log_path = run_dir / "episode_rewards.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Episode reward log not found: {log_path}")
    detail = pd.read_csv(log_path)
    summary = (
        detail.groupby("generation", as_index=False)
        .agg(
            mean_episode_reward=("episode_reward", "mean"),
            best_episode_reward=("episode_reward", "max"),
            worst_episode_reward=("episode_reward", "min"),
            mean_episode_cost=("episode_cost", "mean"),
            mean_episode_carbon=("episode_carbon", "mean"),
            mean_eval_cost=("eval_cost", "mean"),
            mean_eval_carbon=("eval_carbon", "mean"),
        )
    )

    summary_path = output_dir / f"strict_episode_reward_{name}_summary.csv"
    fig_path = output_dir / f"strict_episode_reward_{name}.png"
    summary.to_csv(summary_path, index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    x = summary["generation"]
    ax.fill_between(
        x,
        summary["worst_episode_reward"],
        summary["best_episode_reward"],
        color="#4C78A8",
        alpha=0.16,
        linewidth=0,
        label="Episode reward range",
    )
    ax.plot(
        x,
        summary["mean_episode_reward"],
        marker="o",
        linewidth=1.8,
        color="#1F4E79",
        label="Mean episode reward",
    )
    ax.plot(
        x,
        summary["best_episode_reward"],
        marker="^",
        linewidth=1.2,
        color="#59A14F",
        label="Best episode reward",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Episode scalarized reward")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend(frameon=False)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {fig_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
