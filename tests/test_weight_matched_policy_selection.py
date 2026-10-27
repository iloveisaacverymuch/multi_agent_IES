from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.select_weight_matched_policies import select_weight_matched_policies


def test_weight_matching_prefers_nondominated_then_preference_metric(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "final").mkdir(parents=True)
    (run_dir / "strict_test").mkdir()
    policy_rows = [
        ("cost_best.pt", 0.8, 0.2, 1.0, 9.0),
        ("cost_dominated.pt", 0.8, 0.2, 2.0, 10.0),
        ("balanced_center.pt", 0.4, 0.6, 5.0, 5.0),
        ("balanced_off_center.pt", 0.6, 0.4, 6.0, 4.0),
        ("carbon_best.pt", 0.2, 0.8, 9.0, 1.0),
    ]
    final_rows = []
    strict_rows = []
    for index, (policy, weight_cost, weight_carbon, cost, carbon) in enumerate(
        policy_rows
    ):
        final_rows.append(
            {
                "policy_file": policy,
                "task_id": index,
                "role": "evolving",
                "generation": 1,
                "episode": index,
                "weight_cost": weight_cost,
                "weight_carbon": weight_carbon,
            }
        )
        strict_rows.append(
            {
                "policy_file": policy,
                "pareto_index": index,
                "cost": cost,
                "carbon": carbon,
            }
        )
    pd.DataFrame(final_rows).to_csv(run_dir / "final" / "metadata.csv", index=False)
    pd.DataFrame(strict_rows).to_csv(
        run_dir / "strict_test" / "metadata.csv", index=False
    )

    selected = select_weight_matched_policies(run_dir)

    assert selected["cost_priority"]["policy_file"] == "cost_best.pt"
    assert selected["balanced"]["policy_file"] == "balanced_center.pt"
    assert selected["carbon_priority"]["policy_file"] == "carbon_best.pt"
    assert all(row["strict_nondominated"] for row in selected.values())
