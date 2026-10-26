import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "runs"
    / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_array(path):
    values = np.loadtxt(path, delimiter=",")
    return np.atleast_2d(values)


def generation_dirs(run_dir):
    dirs = []
    for child in run_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            dirs.append((int(child.name), child))
    return sorted(dirs, key=lambda item: item[0])


def collect_elite_rewards(run_dir, transform):
    rows = []
    for generation, gen_dir in generation_dirs(run_dir):
        weights_path = gen_dir / "elites" / "weights.txt"
        objs_path = gen_dir / "elites" / "elites.txt"
        if not weights_path.exists() or not objs_path.exists():
            continue
        weights = load_array(weights_path)
        objectives = load_array(objs_path)
        if len(weights) != len(objectives):
            raise ValueError(
                f"weights/elites row mismatch at generation {generation}: "
                f"{len(weights)} vs {len(objectives)}"
            )
        reward_objectives = objectives
        if transform == "log1p":
            reward_objectives = np.log1p(np.maximum(objectives, 0.0))
        elif transform != "none":
            raise ValueError(f"Unknown reward transform: {transform}")

        scalarized_reward = -np.sum(weights * reward_objectives, axis=1)
        for task_id, (weight, objective, reward) in enumerate(
            zip(weights, objectives, scalarized_reward)
        ):
            rows.append(
                {
                    "generation": generation,
                    "task_id": task_id,
                    "w_cost": weight[0],
                    "w_carbon": weight[1],
                    "cost": objective[0],
                    "carbon": objective[1],
                    "scalarized_reward": reward,
                }
            )
    if not rows:
        raise RuntimeError(f"No elite reward data found in {run_dir}")
    return pd.DataFrame(rows)


def add_normalized_reward(detail):
    ranges = detail[["cost", "carbon"]].max() - detail[["cost", "carbon"]].min()
    ranges = ranges.mask(ranges <= 1e-12, 1.0)
    normalized = (detail[["cost", "carbon"]] - detail[["cost", "carbon"]].min()) / ranges
    detail = detail.copy()
    detail["normalized_scalarized_reward"] = -(
        detail["w_cost"] * normalized["cost"]
        + detail["w_carbon"] * normalized["carbon"]
    )
    return detail


def summarize(detail):
    return (
        detail.groupby("generation", as_index=False)
        .agg(
            mean_scalarized_reward=("scalarized_reward", "mean"),
            best_scalarized_reward=("scalarized_reward", "max"),
            worst_scalarized_reward=("scalarized_reward", "min"),
            mean_normalized_reward=("normalized_scalarized_reward", "mean"),
            best_normalized_reward=("normalized_scalarized_reward", "max"),
            worst_normalized_reward=("normalized_scalarized_reward", "min"),
            mean_cost=("cost", "mean"),
            min_cost=("cost", "min"),
            mean_carbon=("carbon", "mean"),
            min_carbon=("carbon", "min"),
        )
    )


def plot_reward_curve(summary, output_path, title):
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
    x = summary["generation"].to_numpy()
    mean_reward = summary["mean_normalized_reward"].to_numpy()
    best_reward = summary["best_normalized_reward"].to_numpy()
    worst_reward = summary["worst_normalized_reward"].to_numpy()

    ax.fill_between(
        x,
        worst_reward,
        best_reward,
        color="#4C78A8",
        alpha=0.16,
        linewidth=0,
        label="Elite range",
    )
    ax.plot(
        x,
        mean_reward,
        marker="o",
        markersize=4,
        linewidth=1.8,
        color="#1F4E79",
        label="Mean elite reward",
    )
    ax.plot(
        x,
        best_reward,
        marker="^",
        markersize=3.5,
        linewidth=1.2,
        color="#59A14F",
        label="Best elite reward",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized scalarized reward")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.65)
    ax.legend(frameon=False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot generation-level reward curves.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default=None)
    parser.add_argument("--reward-transform", choices=["none", "log1p"], default="none")
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or run_dir.name

    detail = collect_elite_rewards(run_dir, args.reward_transform)
    detail = add_normalized_reward(detail)
    summary = summarize(detail)

    detail_path = output_dir / f"reward_curve_{name}_detail.csv"
    summary_path = output_dir / f"reward_curve_{name}_summary.csv"
    figure_path = output_dir / f"reward_curve_{name}.png"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_reward_curve(summary, figure_path, f"Reward Curve ({name})")

    print(summary.to_string(index=False))
    print(f"Saved: {figure_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")


if __name__ == "__main__":
    main()
