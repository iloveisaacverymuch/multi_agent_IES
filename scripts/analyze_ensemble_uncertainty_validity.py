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
from src.emomarl.opt_graph import OptGraph
from src.emomarl.population import predict_hyperbolic_ensemble


RUN_ROOT = PROJECT_ROOT / "outputs" / "runs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
SEEDS = [11, 22, 44]
NUM_TASKS = 6
UPDATES_PER_TASK = 12


def run_dir(seed):
    base = "carbon_cda_ensemble_uncertainty_capped32_kappa_0_20_price_0_02"
    return RUN_ROOT / (base if seed == 44 else f"{base}_seed_{seed}")


def load_array(path):
    return np.atleast_2d(np.loadtxt(path, delimiter=","))


def parse_optgraph(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    node_count = int(lines[0])
    graph = OptGraph()
    for line in lines[1 : node_count + 1]:
        weight_text, objective_text, parent_text = line.split(";")
        graph.weights.append(np.fromstring(weight_text, sep=","))
        graph.objs.append(np.fromstring(objective_text, sep=","))
        graph.prev.append(int(parent_text))
        graph.succ.append([])
    for index, parent in enumerate(graph.prev):
        if parent < 0:
            graph.delta_objs.append(np.zeros_like(graph.objs[index]))
        else:
            graph.succ[parent].append(index)
            graph.delta_objs.append(graph.objs[index] - graph.objs[parent])
    population_count = int(lines[node_count + 1])
    population_ids = [int(value) for value in lines[node_count + 2 : node_count + 2 + population_count]]
    return graph, population_ids


def candidate_weights(args, graph, optgraph_id):
    center = graph.weights[optgraph_id]
    angle_center = np.arctan2(center[1], center[0])
    weights = []
    for index in range(args.num_weight_candidates):
        angle = angle_center - np.pi / 4 + (np.pi / 2) / (args.num_weight_candidates - 1) * index
        weight = np.array([np.cos(angle), np.sin(angle)])
        if weight[0] < -1e-7 or weight[1] < -1e-7:
            continue
        if any(np.linalg.norm(weight - graph.weights[child]) < 1e-3 for child in graph.succ[optgraph_id]):
            continue
        weights.append(weight / weight.sum())
    return np.asarray(weights)


def parent_candidates(elite_objective, population_objectives, population_ids):
    distances = np.linalg.norm(population_objectives - elite_objective, axis=1)
    best = distances.min()
    return [population_ids[index] for index in np.where(distances <= best + 1e-5)[0]]


def unique_front(objectives):
    rounded = np.round(objectives, decimals=2)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    points = objectives[np.sort(indices)]
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    keep = []
    best_carbon = np.inf
    for index, point in enumerate(points):
        if point[1] < best_carbon:
            keep.append(index)
            best_carbon = point[1]
    return points[keep]


def bootstrap_correlation(x, y, statistic, seed=20260915, samples=5000):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(x), size=len(x))
        if np.unique(x[indices]).size < 2 or np.unique(y[indices]).size < 2:
            continue
        value = statistic(x[indices], y[indices]).statistic
        if np.isfinite(value):
            values.append(value)
    return np.percentile(values, [2.5, 97.5]) if values else np.array([np.nan, np.nan])


def reconstruct_records():
    raw_records = []
    objective_pool = []
    for seed in SEEDS:
        args = get_parser().parse_args([])
        args.seed = seed
        args.elite_ensemble_size = 5
        args.elite_ensemble_sample_ratio = 0.80
        args.elite_ensemble_max_samples = 32
        path = run_dir(seed)
        for generation in range(1, 12):
            generation_dir = path / str(generation)
            graph, population_ids = parse_optgraph(generation_dir / "population" / "optgraph.txt")
            population_objectives = load_array(generation_dir / "population" / "objs.txt")
            elite_objectives = load_array(generation_dir / "elites" / "elites.txt")
            selected_weights = load_array(generation_dir / "elites" / "weights.txt")
            saved_predictions = load_array(generation_dir / "elites" / "predictions.txt")
            next_offspring = load_array(path / str(generation + 1) / "elites" / "offsprings.txt")
            actual_objectives = next_offspring.reshape(NUM_TASKS, UPDATES_PER_TASK, 2)[:, -1, :]
            ep_objectives = load_array(generation_dir / "ep" / "objs.txt")
            objective_pool.extend([elite_objectives, saved_predictions, actual_objectives, ep_objectives])

            prediction_cache = {}
            for task_id, (elite, weight, saved_prediction, actual) in enumerate(
                zip(elite_objectives, selected_weights, saved_predictions, actual_objectives)
            ):
                matches = []
                for optgraph_id in parent_candidates(elite, population_objectives, population_ids):
                    if optgraph_id not in prediction_cache:
                        weights = candidate_weights(args, graph, optgraph_id)
                        if len(weights) == 0:
                            continue
                        result = predict_hyperbolic_ensemble(args, graph, optgraph_id, weights.copy())
                        prediction_cache[optgraph_id] = (weights, result)
                    weights, result = prediction_cache[optgraph_id]
                    weight_distances = np.linalg.norm(weights - weight, axis=1)
                    candidate_index = int(np.argmin(weight_distances))
                    prediction = np.asarray(result["predictions"][candidate_index], dtype=float)
                    matching_score = weight_distances[candidate_index] + 1e-6 * np.linalg.norm(
                        prediction - saved_prediction
                    )
                    matches.append((matching_score, optgraph_id, candidate_index, result, weights))
                if not matches:
                    raise RuntimeError(f"No reconstructed candidate for seed={seed}, generation={generation}, task={task_id}")
                _, optgraph_id, candidate_index, result, weights = min(matches, key=lambda item: item[0])
                prediction = np.asarray(result["predictions"][candidate_index], dtype=float)
                prediction_std = np.asarray(result["prediction_std"][candidate_index], dtype=float)
                raw_records.append(
                    {
                        "seed": seed,
                        "generation": generation,
                        "task_id": task_id,
                        "optgraph_id": optgraph_id,
                        "weight_cost": weight[0],
                        "weight_carbon": weight[1],
                        "parent_cost": elite[0],
                        "parent_carbon": elite[1],
                        "predicted_cost": prediction[0],
                        "predicted_carbon": prediction[1],
                        "saved_prediction_cost": saved_prediction[0],
                        "saved_prediction_carbon": saved_prediction[1],
                        "prediction_std_cost": prediction_std[0],
                        "prediction_std_carbon": prediction_std[1],
                        "uncertainty_score": float(result["uncertainty"][candidate_index]),
                        "actual_cost": actual[0],
                        "actual_carbon": actual[1],
                        "ep_objectives": ep_objectives,
                    }
                )
    return raw_records, np.vstack(objective_pool)


def main():
    raw_records, objective_pool = reconstruct_records()
    minimum = objective_pool.min(axis=0)
    ranges = objective_pool.max(axis=0) - minimum
    ranges = np.where(ranges <= 1e-12, 1.0, ranges)
    hypervolume = Hypervolume(ref_point=np.array([1.1, 1.1]))

    rows = []
    for record in raw_records:
        parent = np.array([record["parent_cost"], record["parent_carbon"]])
        prediction = np.array([record["predicted_cost"], record["predicted_carbon"]])
        saved_prediction = np.array(
            [record["saved_prediction_cost"], record["saved_prediction_carbon"]]
        )
        actual = np.array([record["actual_cost"], record["actual_carbon"]])
        prediction_std = np.array(
            [record["prediction_std_cost"], record["prediction_std_carbon"]]
        )
        weight = np.array([record["weight_cost"], record["weight_carbon"]])
        ep_normalized = (record.pop("ep_objectives") - minimum) / ranges
        actual_normalized = (actual - minimum) / ranges
        base_hv = hypervolume.do(unique_front(ep_normalized))
        actual_hv = hypervolume.do(unique_front(np.vstack([ep_normalized, actual_normalized])))
        record.update(
            {
                "prediction_error_normalized": float(np.linalg.norm((prediction - actual) / ranges)),
                "prediction_std_normalized": float(np.linalg.norm(prediction_std / ranges)),
                "saved_prediction_difference_normalized": float(
                    np.linalg.norm((saved_prediction - prediction) / ranges)
                ),
                "scalarized_improvement_normalized": float(np.sum(weight * (parent - actual) / ranges)),
                "actual_hv_gain": float(max(0.0, actual_hv - base_hv)),
            }
        )
        record["successful_improvement"] = record["scalarized_improvement_normalized"] > 0.0
        record["positive_hv_gain"] = record["actual_hv_gain"] > 1e-12
        rows.append(record)

    detail = pd.DataFrame(rows)
    detail_path = OUTPUT_DIR / "ensemble_uncertainty_validity_detail.csv"
    detail.to_csv(detail_path, index=False)

    x = detail["prediction_std_normalized"].to_numpy()
    y = detail["prediction_error_normalized"].to_numpy()
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    pearson_ci = bootstrap_correlation(x, y, pearsonr)
    spearman_ci = bootstrap_correlation(x, y, spearmanr)

    correlation_rows = [
        {
            "scope": "all",
            "samples": len(detail),
            "pearson_r": pearson.statistic,
            "pearson_p": pearson.pvalue,
            "pearson_ci_low": pearson_ci[0],
            "pearson_ci_high": pearson_ci[1],
            "spearman_rho": spearman.statistic,
            "spearman_p": spearman.pvalue,
            "spearman_ci_low": spearman_ci[0],
            "spearman_ci_high": spearman_ci[1],
        }
    ]
    for seed, selected in detail.groupby("seed"):
        seed_pearson = pearsonr(selected["prediction_std_normalized"], selected["prediction_error_normalized"])
        seed_spearman = spearmanr(selected["prediction_std_normalized"], selected["prediction_error_normalized"])
        correlation_rows.append(
            {
                "scope": f"seed_{seed}",
                "samples": len(selected),
                "pearson_r": seed_pearson.statistic,
                "pearson_p": seed_pearson.pvalue,
                "spearman_rho": seed_spearman.statistic,
                "spearman_p": seed_spearman.pvalue,
            }
        )
    correlation = pd.DataFrame(correlation_rows)
    correlation_path = OUTPUT_DIR / "ensemble_uncertainty_validity_correlation.csv"
    correlation.to_csv(correlation_path, index=False)

    detail["uncertainty_quartile"] = pd.qcut(
        detail["prediction_std_normalized"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    )
    quartiles = (
        detail.groupby("uncertainty_quartile", observed=True, as_index=False)
        .agg(
            samples=("prediction_error_normalized", "size"),
            mean_uncertainty=("prediction_std_normalized", "mean"),
            mean_prediction_error=("prediction_error_normalized", "mean"),
            std_prediction_error=("prediction_error_normalized", "std"),
            scalarized_success_rate=("successful_improvement", "mean"),
            positive_hv_rate=("positive_hv_gain", "mean"),
            mean_hv_gain=("actual_hv_gain", "mean"),
        )
    )
    quartile_path = OUTPUT_DIR / "ensemble_uncertainty_validity_quartiles.csv"
    quartiles.to_csv(quartile_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5), constrained_layout=True)
    colors = {11: "#4C78A8", 22: "#F28E2B", 44: "#59A14F"}
    for seed, selected in detail.groupby("seed"):
        axes[0].scatter(
            selected["prediction_std_normalized"],
            selected["prediction_error_normalized"],
            s=18,
            alpha=0.65,
            color=colors[seed],
            label=f"Seed {seed}",
        )
    coefficients = np.polyfit(x, y, deg=1)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[0].plot(x_line, np.polyval(coefficients, x_line), color="black", linewidth=1.2)
    axes[0].set_xlabel("Predicted uncertainty")
    axes[0].set_ylabel("Actual normalized prediction error")
    axes[0].legend(frameon=False)

    positions = np.arange(len(quartiles))
    axes[1].bar(
        positions,
        quartiles["mean_prediction_error"],
        yerr=quartiles["std_prediction_error"],
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
        capsize=4,
    )
    axes[1].set_xticks(positions, quartiles["uncertainty_quartile"])
    axes[1].set_xlabel("Uncertainty quartile")
    axes[1].set_ylabel("Prediction error")

    width = 0.36
    axes[2].bar(
        positions - width / 2,
        quartiles["scalarized_success_rate"],
        width,
        color="#59A14F",
        edgecolor="black",
        linewidth=0.6,
        label="Scalarized improvement",
    )
    axes[2].bar(
        positions + width / 2,
        quartiles["positive_hv_rate"],
        width,
        color="#E15759",
        edgecolor="black",
        linewidth=0.6,
        label="Positive HV gain",
    )
    axes[2].set_xticks(positions, quartiles["uncertainty_quartile"])
    axes[2].set_xlabel("Uncertainty quartile")
    axes[2].set_ylabel("Success rate")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    figure_path = OUTPUT_DIR / "ensemble_uncertainty_validity.png"
    fig.savefig(figure_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(correlation.to_string(index=False))
    print(quartiles.to_string(index=False))
    print(f"Saved: {detail_path}")
    print(f"Saved: {correlation_path}")
    print(f"Saved: {quartile_path}")
    print(f"Saved: {figure_path}")


if __name__ == "__main__":
    main()
