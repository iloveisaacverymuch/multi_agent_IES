import numpy as np
from pymoo.indicators.hv import Hypervolume

from src.emomarl.population import Population
from src.utils import get_ep_indices


def test_analytic_2d_hv_gain_matches_pymoo():
    rng = np.random.default_rng(20260915)
    indicator = Hypervolume(ref_point=np.array([1.0, 1.0]))
    for _ in range(200):
        points = rng.uniform(0.0, 0.95, size=(20, 2))
        front = points[get_ep_indices(points)]
        prediction = rng.uniform(0.0, 0.95, size=2)
        expected = indicator.do(np.vstack([front, prediction])) - indicator.do(front)
        actual = Population._hypervolume_gain_2d(front, prediction, ref_point=(1.0, 1.0))
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_extreme_protection_selects_both_preference_endpoints():
    candidates = [
        {"weight": np.array([0.9, 0.1])},
        {"weight": np.array([0.5, 0.5])},
        {"weight": np.array([0.1, 0.9])},
    ]
    scores = [1.0, 100.0, 2.0]
    mask = [True, True, True]

    first = Population._select_extreme_or_best(candidates, mask, scores, 0, enabled=True)
    mask[first] = False
    second = Population._select_extreme_or_best(candidates, mask, scores, 1, enabled=True)

    assert first == 0
    assert second == 2
    assert Population._select_extreme_or_best(
        candidates, [True, True, True], scores, 0, enabled=False
    ) == 1
