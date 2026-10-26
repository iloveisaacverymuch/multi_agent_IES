import numpy as np

from src.envs.market import (
    CDA,
    carbon_aware_CDA,
    carbon_dividend_settlement,
    volume_preserving_carbon_aware_CDA,
)


def traded_volume(records):
    return sum(float(record[2]) for record in records)


def test_v2_does_not_block_a_standard_feasible_trade():
    standard = CDA(10.0, 0.50, -10.0, 0.60, 0.0, 0.55)[2]
    old_carbon = carbon_aware_CDA(
        10.0, 0.50, 10.0,
        -10.0, 0.60, 0.0,
        0.0, 0.55, 0.0,
        carbon_price=0.02,
    )[2]
    v2 = volume_preserving_carbon_aware_CDA(
        10.0, 0.50, 10.0,
        -10.0, 0.60, 0.0,
        0.0, 0.55, 0.0,
        carbon_price=0.02,
    )[2]

    assert traded_volume(standard) == 10.0
    assert traded_volume(old_carbon) == 0.0
    assert traded_volume(v2) == traded_volume(standard)


def test_v2_prefers_the_cleaner_feasible_seller():
    records = volume_preserving_carbon_aware_CDA(
        10.0, 0.40, 5.0,
        10.0, 0.40, 0.1,
        -5.0, 0.80, 0.0,
        carbon_price=0.02,
    )[2]

    assert len(records) == 1
    assert records[0][1] == 2
    assert traded_volume(records) == 5.0


def test_v2_never_reduces_standard_volume_on_random_markets():
    rng = np.random.default_rng(20260915)
    for _ in range(500):
        quantities = rng.uniform(-20.0, 20.0, size=3)
        prices = rng.uniform(0.2, 1.0, size=3)
        carbon = rng.uniform(0.0, 5.0, size=3)
        standard = CDA(
            quantities[0], prices[0], quantities[1], prices[1], quantities[2], prices[2]
        )[2]
        v2 = volume_preserving_carbon_aware_CDA(
            quantities[0], prices[0], carbon[0],
            quantities[1], prices[1], carbon[1],
            quantities[2], prices[2], carbon[2],
            carbon_price=0.02,
        )[2]
        assert traded_volume(v2) + 1e-9 >= traded_volume(standard)


def test_v3_dividend_is_budget_balanced_and_rewards_clean_seller():
    levies, credits, avoided = carbon_dividend_settlement(
        trading_record=[[3, 1, 10.0, 0.5], [3, 2, 10.0, 0.5]],
        carbon_intensities={1: 0.1, 2: 0.5, 3: 0.0},
        grid_imports={1: 0.0, 2: 0.0, 3: 20.0},
        grid_emission_factor=0.6,
        carbon_price=0.02,
    )

    assert np.isclose(sum(levies.values()), sum(credits.values()), atol=1e-12)
    assert credits[1] > credits[2] > 0.0
    assert avoided[1] > avoided[2] > 0.0


def test_v3_returns_equal_dividend_without_eligible_peer_trade():
    levies, credits, avoided = carbon_dividend_settlement(
        trading_record=[],
        carbon_intensities={1: 0.0, 2: 0.0, 3: 0.0},
        grid_imports={1: 10.0, 2: 5.0, 3: 0.0},
        grid_emission_factor=0.6,
        carbon_price=0.02,
    )

    assert np.isclose(sum(levies.values()), sum(credits.values()), atol=1e-12)
    assert np.isclose(credits[1], credits[2])
    assert np.isclose(credits[2], credits[3])
    assert sum(avoided.values()) == 0.0


def test_v3_equal_ablation_ignores_eligible_seller_contribution():
    levies, credits, avoided = carbon_dividend_settlement(
        trading_record=[[3, 1, 10.0, 0.5], [3, 2, 10.0, 0.5]],
        carbon_intensities={1: 0.1, 2: 0.5, 3: 0.0},
        grid_imports={1: 0.0, 2: 0.0, 3: 20.0},
        grid_emission_factor=0.6,
        carbon_price=0.02,
        dividend_mode="equal",
    )

    assert np.isclose(sum(levies.values()), sum(credits.values()), atol=1e-12)
    assert np.isclose(credits[1], credits[2])
    assert np.isclose(credits[2], credits[3])
    assert avoided[1] > avoided[2] > 0.0
