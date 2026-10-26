from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume

from analyze_v2_pilot import igd, unique_front


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
RUNS = (
    (
        "v2",
        "Carbon CDA V2 + HV-Ensemble",
        RUN_ROOT / "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02",
    ),
    (
        "current_main",
        "Current main method",
        RUN_ROOT / "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02",
    ),
    ("fixed_mappo", "Fixed-weight MAPPO", RUN_ROOT / "baseline_fixed_weight_mappo"),
)


def main():
    records = []
    pool = []
    final_pool = []
    for key, label, path in RUNS:
        final = np.atleast_2d(np.loadtxt(path / "final" / "objs.txt", delimiter=","))
        final_pool.append(final)
        for generation in range(1, 13):
            objectives = np.atleast_2d(
                np.loadtxt(path / str(generation) / "ep" / "objs.txt", delimiter=",")
            )
            records.append((key, label, generation, objectives))
            pool.append(objectives)

    all_objectives = np.vstack(pool)
    minimum = all_objectives.min(axis=0)
    ranges = np.maximum(all_objectives.max(axis=0) - minimum, 1e-12)
    reference = unique_front(np.vstack(final_pool))
    reference_normalized = (reference - minimum) / ranges
    indicator = Hypervolume(ref_point=np.array([1.1, 1.1]))
    rows = []
    for key, label, generation, objectives in records:
        front = unique_front(objectives)
        normalized = (front - minimum) / ranges
        rows.append(
            {
                "method_key": key,
                "method": label,
                "generation": generation,
                "normalized_hv": float(indicator.do(normalized)),
                "igd": igd(normalized, reference_normalized),
                "front_points": len(front),
            }
        )
    detail = pd.DataFrame(rows)
    detail_path = OUTPUT_DIR / "v2_pilot_convergence.csv"
    detail.to_csv(detail_path, index=False)

    colors = {"v2": "#4C78A8", "current_main": "#F28E2B", "fixed_mappo": "#59A14F"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), constrained_layout=True)
    for key, label, _ in RUNS:
        selected = detail[detail.method_key == key]
        axes[0].plot(selected.generation, selected.normalized_hv, marker="o", ms=3, label=label, color=colors[key])
        axes[1].plot(selected.generation, selected.igd, marker="o", ms=3, label=label, color=colors[key])
    axes[0].set_ylabel("Normalized HV")
    axes[1].set_ylabel("IGD")
    for axis in axes:
        axis.set_xlabel("Generation")
        axis.set_xticks(range(1, 13))
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=True, fontsize=8)
    figure_path = OUTPUT_DIR / "v2_pilot_convergence.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(detail[detail.generation == 12].to_string(index=False))
    print(f"Saved: {detail_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
