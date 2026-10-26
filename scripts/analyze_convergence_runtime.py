from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

GROUPS = {
    "carbon_price": [
        ("price_0.00", RUN_ROOT / "carbon_cda_contribution_price_0_00"),
        ("price_0.02", RUN_ROOT / "carbon_cda_contribution_price_0_02"),
        ("price_0.05", RUN_ROOT / "carbon_cda_contribution_price_0_05"),
        ("price_0.08", RUN_ROOT / "carbon_cda_contribution_price_0_08"),
        ("price_0.10", RUN_ROOT / "carbon_cda_contribution_price_0_10"),
    ],
    "seed11_methods": [
        ("Standard CDA + Prediction", RUN_ROOT / "seed_stability_standard_prediction_seed_11"),
        ("Carbon CDA Only", RUN_ROOT / "seed_stability_carbon_cda_only_seed_11"),
        ("Standard CDA + Contribution", RUN_ROOT / "seed_stability_standard_contribution_seed_11"),
        ("Carbon CDA + Contribution", RUN_ROOT / "seed_stability_carbon_contribution_seed_11"),
    ],
}


def load_objectives(path):
    objectives = np.loadtxt(path, delimiter=",")
    return np.atleast_2d(objectives)


def pareto_indices(objectives):
    dominated = np.zeros(len(objectives), dtype=bool)
    for i in range(len(objectives)):
        for j in range(len(objectives)):
            if i == j:
                continue
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                dominated[i] = True
                break
    return np.where(~dominated)[0]


def unique_front(objectives, decimals=2):
    front = objectives[pareto_indices(objectives)]
    rounded = np.round(front, decimals=decimals)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    front = front[np.sort(idx)]
    return front[np.argsort(front[:, 0])]


def generation_objectives(run_dir):
    items = []
    for child in run_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        obj_path = child / "ep" / "objs.txt"
        if obj_path.exists():
            items.append((int(child.name), load_objectives(obj_path)))
    final_path = run_dir / "final" / "objs.txt"
    if final_path.exists():
        final_generation = max([gen for gen, _ in items], default=0) + 1
        items.append((final_generation, load_objectives(final_path)))
    return sorted(items, key=lambda item: item[0])


def read_elapsed(run_dir):
    elapsed_path = run_dir / "elapsed_seconds.txt"
    if elapsed_path.exists():
        return float(elapsed_path.read_text(encoding="utf-8").strip())
    log_path = run_dir / "train.log"
    if log_path.exists():
        return max(log_path.stat().st_mtime - log_path.stat().st_ctime, 0.0)
    return np.nan


def analyze_group(group_name, runs):
    series = []
    all_objectives = []
    for label, run_dir in runs:
        generations = generation_objectives(run_dir)
        for generation, objectives in generations:
            all_objectives.append(objectives)
            series.append((label, run_dir, generation, objectives))
    if not series:
        return pd.DataFrame()

    stacked = np.vstack(all_objectives)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    ranges = np.where(maxs - mins <= 1e-12, 1.0, maxs - mins)

    rows = []
    for label, run_dir, generation, objectives in series:
        front = unique_front(objectives)
        hv = float(Hypervolume(ref_point=np.array([1.1, 1.1])).do((front - mins) / ranges))
        rows.append(
            {
                "group": group_name,
                "label": label,
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                "generation": generation,
                "normalized_hypervolume": hv,
                "num_front_points": len(front),
            }
        )
    return pd.DataFrame(rows)


def plot_group(group_name, data):
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    for label, part in data.groupby("label"):
        part = part.sort_values("generation")
        ax.plot(part["generation"], part["normalized_hypervolume"], marker="o", linewidth=1.4, markersize=3, label=label)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Normalized HV")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUTPUT_DIR / f"convergence_{group_name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )

    convergence_frames = []
    runtime_rows = []
    for group_name, runs in GROUPS.items():
        group_data = analyze_group(group_name, runs)
        if not group_data.empty:
            group_data.to_csv(OUTPUT_DIR / f"convergence_{group_name}.csv", index=False)
            plot_group(group_name, group_data)
            convergence_frames.append(group_data)
        for label, run_dir in runs:
            final_path = run_dir / "final" / "objs.txt"
            runtime_rows.append(
                {
                    "group": group_name,
                    "label": label,
                    "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                    "has_final": final_path.exists(),
                    "elapsed_seconds": read_elapsed(run_dir),
                    "elapsed_hours": read_elapsed(run_dir) / 3600.0,
                }
            )

    if convergence_frames:
        pd.concat(convergence_frames, ignore_index=True).to_csv(OUTPUT_DIR / "convergence_all.csv", index=False)
    runtime = pd.DataFrame(runtime_rows)
    runtime_path = OUTPUT_DIR / "runtime_summary.csv"
    runtime.to_csv(runtime_path, index=False)

    print(runtime.to_string(index=False))
    print(f"Saved: {runtime_path}")


if __name__ == "__main__":
    main()
