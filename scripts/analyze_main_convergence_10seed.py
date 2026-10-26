import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
RESULT_ROOT = PROJECT_ROOT / "outputs" / "main_convergence_10seed"
FIGURE_ROOT = PROJECT_ROOT / "outputs" / "figures"
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
METHODS = [
    ("all_evolving_kappa0", "All-task Mean-HV E-MADDPG"),
    ("maddpg", "Fixed-weight MADDPG"),
]
CHECKPOINTS = np.arange(12, 973, 12)


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def update_archive(archive, value):
    value = np.asarray(value, dtype=np.float64)
    if any(dominates(point, value) or np.allclose(point, value, atol=1e-9) for point in archive):
        return archive
    return [point for point in archive if not dominates(value, point)] + [value]


def sequence(method, seed):
    path = RUN_ROOT / f"strict_v3_{method}_seed_{seed}"
    if method == "all_evolving_kappa0":
        frame = pd.read_csv(path / "episode_log.csv")
        values = frame[["eval_cost", "eval_carbon"]].to_numpy(dtype=np.float64)
    else:
        frames = [pd.read_csv(path / f"weight_{index}_episodes.csv") for index in range(6)]
        values = np.vstack(
            [frame[["eval_cost", "eval_carbon"]].to_numpy(dtype=np.float64) for frame in frames]
        )
    values = values[np.all(np.isfinite(values), axis=1)]
    if len(values) != 972:
        raise ValueError(f"Expected 972 evaluations for {method} seed={seed}, got {len(values)}")
    return values


def final_archive(values):
    archive = []
    for value in values:
        archive = update_archive(archive, value)
    return np.asarray(archive)


def igd(front, reference):
    distances = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def elapsed(method, seed):
    path = RUN_ROOT / f"strict_v3_{method}_seed_{seed}" / "launcher_elapsed_seconds.txt"
    return float(path.read_text(encoding="utf-8").strip())


def main():
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    sequences = {
        (method, seed): sequence(method, seed)
        for method, _ in METHODS
        for seed in SEEDS
    }
    final_fronts = {
        key: final_archive(values) for key, values in sequences.items()
    }
    reference = final_archive(np.vstack(list(final_fronts.values())))
    ideal = reference.min(axis=0)
    nadir = np.vstack(list(final_fronts.values())).max(axis=0)
    scale = np.maximum(nadir - ideal, 1e-12)
    reference_norm = (reference - ideal) / scale
    indicator = Hypervolume(ref_point=np.array([1.1, 1.1]))

    rows = []
    for method, label in METHODS:
        for seed in SEEDS:
            archive = []
            checkpoint_set = set(CHECKPOINTS.tolist())
            run_elapsed = elapsed(method, seed)
            for episode, value in enumerate(sequences[(method, seed)], start=1):
                archive = update_archive(archive, value)
                if episode not in checkpoint_set:
                    continue
                normalized = (np.asarray(archive) - ideal) / scale
                rows.append(
                    {
                        "method_key": method,
                        "method": label,
                        "seed": seed,
                        "evaluations": episode,
                        "estimated_hours": run_elapsed * episode / 972.0 / 3600.0,
                        "normalized_hv": float(indicator.do(normalized)),
                        "igd": igd(normalized, reference_norm),
                        "pareto_points": len(archive),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["method_key", "method", "evaluations"], as_index=False)
        .agg(
            hours_mean=("estimated_hours", "mean"),
            hv_mean=("normalized_hv", "mean"),
            hv_std=("normalized_hv", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            points_mean=("pareto_points", "mean"),
            points_std=("pareto_points", "std"),
        )
    )
    auc_rows = []
    for (method, label, seed), group in detail.groupby(["method_key", "method", "seed"]):
        group = group.sort_values("evaluations")
        x = group.evaluations.to_numpy(dtype=np.float64)
        auc_rows.append(
            {
                "method_key": method,
                "method": label,
                "seed": seed,
                "hv_auc": float(np.trapz(group.normalized_hv, x) / (x[-1] - x[0])),
                "igd_auc": float(np.trapz(group.igd, x) / (x[-1] - x[0])),
            }
        )
    auc_detail = pd.DataFrame(auc_rows)
    auc_summary = auc_detail.groupby(["method_key", "method"], as_index=False).agg(
        hv_auc_mean=("hv_auc", "mean"),
        hv_auc_std=("hv_auc", "std"),
        igd_auc_mean=("igd_auc", "mean"),
        igd_auc_std=("igd_auc", "std"),
    )
    detail.to_csv(RESULT_ROOT / "detail.csv", index=False)
    summary.to_csv(RESULT_ROOT / "summary.csv", index=False)
    auc_detail.to_csv(RESULT_ROOT / "auc_detail.csv", index=False)
    auc_summary.to_csv(RESULT_ROOT / "auc_summary.csv", index=False)
    (RESULT_ROOT / "protocol.json").write_text(
        json.dumps(
            {
                "seeds": SEEDS,
                "checkpoints": CHECKPOINTS.tolist(),
                "normalization_ideal": ideal.tolist(),
                "normalization_nadir": nadir.tolist(),
                "reference_front_points": len(reference),
                "note": "Curves use deterministic 1440-hour training evaluations and equal evaluation budgets.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4), constrained_layout=True)
    colors = {METHODS[0][0]: "#C44E52", METHODS[1][0]: "#F28E2B"}
    for method, label in METHODS:
        frame = summary[summary.method_key == method].sort_values("evaluations")
        x = frame.evaluations.to_numpy()
        for ax, mean, std, ylabel in (
            (axes[0, 0], "hv_mean", "hv_std", "Normalized HV"),
            (axes[0, 1], "igd_mean", "igd_std", "IGD"),
            (axes[1, 0], "points_mean", "points_std", "Pareto points"),
        ):
            y = frame[mean].to_numpy()
            error = frame[std].fillna(0.0).to_numpy()
            ax.plot(x, y, color=colors[method], linewidth=1.8, label=label)
            ax.fill_between(x, y - error, y + error, color=colors[method], alpha=0.14)
            ax.set_xlabel("Deterministic evaluations")
            ax.set_ylabel(ylabel)
            ax.grid(linestyle="--", linewidth=0.5, alpha=0.6)
        axes[1, 1].plot(
            frame.hours_mean,
            frame.hv_mean,
            color=colors[method],
            linewidth=1.8,
            label=label,
        )
    axes[1, 1].set_xlabel("Estimated training time (h)")
    axes[1, 1].set_ylabel("Normalized HV")
    axes[1, 1].grid(linestyle="--", linewidth=0.5, alpha=0.6)
    for ax in axes.flat:
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGURE_ROOT / "main_vs_maddpg_convergence_10seed.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(auc_summary.to_string(index=False))
    print("\nFinal checkpoints:\n" + summary[summary.evaluations == 972].to_string(index=False))


if __name__ == "__main__":
    main()
