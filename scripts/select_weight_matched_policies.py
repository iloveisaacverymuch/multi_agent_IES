from pathlib import Path

import numpy as np
import pandas as pd


PREFERENCE_ORDER = ("cost_priority", "balanced", "carbon_priority")
TARGET_WEIGHTS = {
    "cost_priority": np.array([0.8, 0.2], dtype=np.float64),
    "balanced": np.array([0.5, 0.5], dtype=np.float64),
    "carbon_priority": np.array([0.2, 0.8], dtype=np.float64),
}


def nondominated_mask(objectives):
    dominated = np.zeros(len(objectives), dtype=bool)
    for index, value in enumerate(objectives):
        dominated[index] = np.any(
            np.all(objectives <= value, axis=1)
            & np.any(objectives < value, axis=1)
        )
    return ~dominated


def _load_policy_table(run_dir):
    run_dir = Path(run_dir)
    final = pd.read_csv(run_dir / "final" / "metadata.csv")
    strict = pd.read_csv(run_dir / "strict_test" / "metadata.csv")
    required_final = {
        "policy_file",
        "task_id",
        "role",
        "generation",
        "episode",
        "weight_cost",
        "weight_carbon",
    }
    required_strict = {"policy_file", "pareto_index", "cost", "carbon"}
    if missing := required_final - set(final.columns):
        raise ValueError(f"Missing final metadata columns: {sorted(missing)}")
    if missing := required_strict - set(strict.columns):
        raise ValueError(f"Missing strict-test columns: {sorted(missing)}")

    provenance = final[list(required_final)]
    table = strict[list(required_strict)].merge(
        provenance, on="policy_file", how="inner", validate="one_to_one"
    )
    if len(table) != len(strict):
        raise ValueError("Strict-test and final policy metadata do not match.")
    return table


def _preference_score(preference, objectives, normalized, index):
    if preference == "cost_priority":
        return (objectives[index, 0], objectives[index, 1])
    if preference == "carbon_priority":
        return (objectives[index, 1], objectives[index, 0])
    center_distance = float(np.linalg.norm(normalized[index] - 0.5))
    return (center_distance, float(normalized[index].sum()))


def select_weight_matched_policies(run_dir):
    table = _load_policy_table(run_dir)
    objectives = table[["cost", "carbon"]].to_numpy(dtype=np.float64)
    weights = table[["weight_cost", "weight_carbon"]].to_numpy(dtype=np.float64)
    is_nondominated = nondominated_mask(objectives)
    front = objectives[is_nondominated]
    ideal = front.min(axis=0)
    span = np.maximum(front.max(axis=0) - ideal, 1e-12)
    normalized = (objectives - ideal) / span

    selected = {}
    for preference in PREFERENCE_ORDER:
        target = TARGET_WEIGHTS[preference]
        distances = np.linalg.norm(weights - target, axis=1)
        minimum_distance = float(distances.min())
        nearest = np.flatnonzero(
            np.isclose(distances, minimum_distance, rtol=1e-7, atol=1e-10)
        )
        nondominated_nearest = nearest[is_nondominated[nearest]]
        eligible = nondominated_nearest if len(nondominated_nearest) else nearest
        chosen = min(
            eligible,
            key=lambda index: (
                *_preference_score(preference, objectives, normalized, index),
                str(table.iloc[index]["policy_file"]),
            ),
        )
        row = table.iloc[chosen]
        selected[preference] = {
            "preference": preference,
            "policy_file": str(row["policy_file"]),
            "pareto_index": int(row["pareto_index"]),
            "cost": float(row["cost"]),
            "carbon": float(row["carbon"]),
            "target_weight_cost": float(target[0]),
            "target_weight_carbon": float(target[1]),
            "weight_cost": float(row["weight_cost"]),
            "weight_carbon": float(row["weight_carbon"]),
            "weight_distance": float(distances[chosen]),
            "strict_nondominated": bool(is_nondominated[chosen]),
            "task_id": int(row["task_id"]),
            "role": str(row["role"]),
            "generation": int(row["generation"]),
            "episode": int(row["episode"]),
        }

    policy_files = [row["policy_file"] for row in selected.values()]
    if len(set(policy_files)) != len(policy_files):
        raise RuntimeError("Representative preferences selected duplicate policies.")
    return selected
