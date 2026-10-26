from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.emomarl.arguments import get_parser
from src.emomarl.population import predict_hyperbolic_ensemble
from analyze_ensemble_uncertainty_validity import (
    candidate_weights,
    load_array,
    parent_candidates,
    parse_optgraph,
    unique_front,
)


RUN_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "runs"
    / "carbon_cda_v2_hv_ensemble_extreme_capped32_kappa_0_10_price_0_02"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
NUM_TASKS = 6
UPDATES_PER_TASK = 12


def reconstruct():
    args = get_parser().parse_args([])
    args.seed = 44
    args.elite_ensemble_size = 5
    args.elite_ensemble_sample_ratio = 0.80
    args.elite_ensemble_max_samples = 32
    rows = []
    pool = []
    for generation in range(1, 12):
        generation_dir = RUN_DIR / str(generation)
        graph, population_ids = parse_optgraph(generation_dir / "population" / "optgraph.txt")
        population_objectives = load_array(generation_dir / "population" / "objs.txt")
        elites = load_array(generation_dir / "elites" / "elites.txt")
        weights = load_array(generation_dir / "elites" / "weights.txt")
        saved_predictions = load_array(generation_dir / "elites" / "predictions.txt")
        offspring = load_array(RUN_DIR / str(generation + 1) / "elites" / "offsprings.txt")
        actual = offspring.reshape(NUM_TASKS, UPDATES_PER_TASK, 2)[:, -1, :]
        ep = load_array(generation_dir / "ep" / "objs.txt")
        pool.extend([elites, saved_predictions, actual, ep])
        cache = {}
        for task_id, (elite, weight, saved, realized) in enumerate(
            zip(elites, weights, saved_predictions, actual)
        ):
            matches = []
            for optgraph_id in parent_candidates(elite, population_objectives, population_ids):
                if optgraph_id not in cache:
                    candidates = candidate_weights(args, graph, optgraph_id)
                    if len(candidates) == 0:
                        continue
                    cache[optgraph_id] = (
                        candidates,
                        predict_hyperbolic_ensemble(args, graph, optgraph_id, candidates.copy()),
                    )
                candidates, result = cache[optgraph_id]
                distances = np.linalg.norm(candidates - weight, axis=1)
                index = int(np.argmin(distances))
                prediction = np.asarray(result["predictions"][index], dtype=float)
                score = distances[index] + 1e-6 * np.linalg.norm(prediction - saved)
                matches.append((score, optgraph_id, index, result))
            if not matches:
                raise RuntimeError(
                    f"No candidate match for generation={generation}, task={task_id}"
                )
            _, optgraph_id, index, result = min(matches, key=lambda item: item[0])
            prediction = np.asarray(result["predictions"][index], dtype=float)
            members = np.asarray(result["ensemble_predictions"], dtype=float)[:, index, :]
            rows.append(
                {
                    "generation": generation,
                    "task_id": task_id,
                    "optgraph_id": optgraph_id,
                    "predicted": prediction,
                    "saved": saved,
                    "members": members,
                    "actual": realized,
                    "ep": ep,
                }
            )
    return rows, np.vstack(pool)


def hv_gain(indicator, ep, point):
    base = indicator.do(unique_front(ep))
    extended = indicator.do(unique_front(np.vstack([ep, point])))
    return float(max(0.0, extended - base))


def main():
    raw, pool = reconstruct()
    minimum = pool.min(axis=0)
    ranges = np.maximum(pool.max(axis=0) - minimum, 1e-12)
    indicator = Hypervolume(ref_point=np.array([1.1, 1.1]))
    rows = []
    for record in raw:
        ep = (record["ep"] - minimum) / ranges
        members = (record["members"] - minimum) / ranges
        actual = (record["actual"] - minimum) / ranges
        prediction = (record["predicted"] - minimum) / ranges
        saved = (record["saved"] - minimum) / ranges
        member_gains = np.asarray([hv_gain(indicator, ep, point) for point in members])
        actual_gain = hv_gain(indicator, ep, actual)
        rows.append(
            {
                "generation": record["generation"],
                "task_id": record["task_id"],
                "optgraph_id": record["optgraph_id"],
                "hv_gain_mean": float(member_gains.mean()),
                "hv_gain_std": float(member_gains.std()),
                "actual_hv_gain": actual_gain,
                "hv_gain_prediction_error": float(abs(actual_gain - member_gains.mean())),
                "coordinate_prediction_error": float(np.linalg.norm(actual - prediction)),
                "coordinate_prediction_std": float(np.linalg.norm(np.std(members, axis=0))),
                "saved_prediction_difference": float(np.linalg.norm(saved - prediction)),
                "positive_actual_hv_gain": actual_gain > 1e-12,
            }
        )
    detail = pd.DataFrame(rows)
    detail_path = OUTPUT_DIR / "v2_hv_uncertainty_detail.csv"
    detail.to_csv(detail_path, index=False)

    comparisons = (
        ("hv_space", "hv_gain_std", "hv_gain_prediction_error"),
        ("coordinate_space", "coordinate_prediction_std", "coordinate_prediction_error"),
    )
    correlation_rows = []
    for name, x_column, y_column in comparisons:
        x = detail[x_column].to_numpy()
        y = detail[y_column].to_numpy()
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        correlation_rows.append(
            {
                "uncertainty_space": name,
                "samples": len(detail),
                "pearson_r": pearson.statistic,
                "pearson_p": pearson.pvalue,
                "spearman_rho": spearman.statistic,
                "spearman_p": spearman.pvalue,
                "near_zero_uncertainty_rate": float((x < 1e-12).mean()),
            }
        )
    correlations = pd.DataFrame(correlation_rows)
    correlation_path = OUTPUT_DIR / "v2_hv_uncertainty_correlation.csv"
    correlations.to_csv(correlation_path, index=False)

    detail["uncertainty_quartile"] = pd.qcut(
        detail["hv_gain_std"].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    quartiles = detail.groupby("uncertainty_quartile", observed=True, as_index=False).agg(
        samples=("hv_gain_prediction_error", "size"),
        mean_uncertainty=("hv_gain_std", "mean"),
        mean_prediction_error=("hv_gain_prediction_error", "mean"),
        positive_hv_rate=("positive_actual_hv_gain", "mean"),
        mean_actual_hv_gain=("actual_hv_gain", "mean"),
    )
    quartile_path = OUTPUT_DIR / "v2_hv_uncertainty_quartiles.csv"
    quartiles.to_csv(quartile_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), constrained_layout=True)
    axes[0].scatter(detail.hv_gain_std, detail.hv_gain_prediction_error, s=25, alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("Predicted HV disagreement")
    axes[0].set_ylabel("Actual HV-gain prediction error")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    axes[1].bar(quartiles.uncertainty_quartile.astype(str), quartiles.mean_prediction_error, color="#4C78A8", edgecolor="black", linewidth=0.6)
    axes[1].set_xlabel("HV-disagreement quartile")
    axes[1].set_ylabel("Mean HV-gain prediction error")
    axes[1].grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    figure_path = OUTPUT_DIR / "v2_hv_uncertainty_validity.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(correlations.to_string(index=False))
    print(quartiles.to_string(index=False))
    print("max saved prediction difference:", detail.saved_prediction_difference.max())
    for path in (detail_path, correlation_path, quartile_path, figure_path):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
