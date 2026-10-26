from pathlib import Path
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from pareto_style import PARETO_COLORS, apply_pareto_rcparams, scatter_pareto, style_pareto_axis

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "runs"
    / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"


def load_objectives(run_dir):
    obj_path = run_dir / "final" / "objs.txt"
    if not obj_path.exists():
        raise FileNotFoundError(f"Objective file not found: {obj_path}")

    objectives = np.loadtxt(obj_path, delimiter=",")
    objectives = np.atleast_2d(objectives)
    if objectives.shape[1] != 2:
        raise ValueError(f"Expected two objectives, got shape {objectives.shape}")
    if not np.isfinite(objectives).all():
        raise ValueError("Objective file contains NaN or infinite values.")
    return objectives


def unique_front(objectives, decimals=2):
    rounded = np.round(objectives, decimals=decimals)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    front = objectives[np.sort(unique_indices)]
    return front[np.argsort(front[:, 0])]


def representative_indices(objectives):
    mins = objectives.min(axis=0)
    ranges = objectives.max(axis=0) - mins
    ranges = np.where(ranges == 0, 1.0, ranges)
    normalized = (objectives - mins) / ranges
    distances = np.linalg.norm(normalized, axis=1)
    return {
        "Minimum cost": int(np.argmin(objectives[:, 0])),
        "Minimum carbon": int(np.argmin(objectives[:, 1])),
        "Balanced": int(np.argmin(distances)),
    }


def plot_pareto(objectives, output_dir, run_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    front = unique_front(objectives)
    reps = representative_indices(objectives)

    apply_pareto_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 5.2))

    handles = []
    handles.append(ax.scatter(
        objectives[:, 0],
        objectives[:, 1],
        c="#7f7f7f",
        marker="o",
        s=18,
        edgecolors="k",
        linewidths=0.3,
        alpha=0.18,
        label=f"Final EP solutions",
        rasterized=True,
    ))
    handles.append(scatter_pareto(
        ax,
        front[:, 0],
        front[:, 1],
        f"Pareto front",
        PARETO_COLORS[0],
    ))

    rep_styles = {
        "Minimum cost": PARETO_COLORS[1],
        "Minimum carbon": PARETO_COLORS[2],
        "Balanced": PARETO_COLORS[4],
    }
    for label, idx in reps.items():
        x, y = objectives[idx]
        handles.append(ax.scatter(
            x,
            y,
            c=rep_styles[label],
            marker="o",
            s=75,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
            label=f"{label}: policy {idx}",
        ))

    style_pareto_axis(ax, objectives[:, 0], objectives[:, 1], handles)

    stem = output_dir / f"pareto_front_{run_name}"
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, reps, front


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    objectives = load_objectives(run_dir)
    png_path, pdf_path, reps, front = plot_pareto(objectives, output_dir, run_dir.name)

    print(f"Loaded objectives: {len(objectives)}")
    print(f"Unique front points: {len(front)}")
    for label, idx in reps.items():
        cost, carbon = objectives[idx]
        print(f"{label}: EP_policy_{idx}.pt, cost={cost:.6f}, carbon={carbon:.6f}")
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
