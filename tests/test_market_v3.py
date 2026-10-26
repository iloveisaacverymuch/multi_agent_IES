import numpy as np

from src.envs.ies_env import env


def test_v3_preserves_standard_dispatch_and_system_cost():
    action = np.array(
        [
            [-0.09300422, -0.73191661, -0.19377403, -0.59308952, -0.47537332],
            [0.50072935, -0.43918248, -0.02961805, 0.96147440, 0.92331439],
            [0.44957988, 0.08245371, -0.44621759, -0.67869598, 0.93985083],
        ]
    )
    initial = env()
    state = np.concatenate(initial.reset_train(), axis=0)
    standard = env(market_mechanism="standard")
    dividend = env(market_mechanism="carbon_dividend_v3", cda_carbon_price=0.02)

    standard_output = standard.step(action, state, step=0, flag=0)
    dividend_output = dividend.step(action, state, step=0, flag=0)

    assert np.allclose(standard.trade_record_e, dividend.trade_record_e)
    assert np.allclose(
        [
            standard.buy_from_grid_residential,
            standard.buy_from_grid_commercial,
            standard.buy_from_grid_industry,
        ],
        [
            dividend.buy_from_grid_residential,
            dividend.buy_from_grid_commercial,
            dividend.buy_from_grid_industry,
        ],
    )
    standard_costs = np.array([item[0] for item in standard_output[4:7]])
    dividend_costs = np.array([item[0] for item in dividend_output[4:7]])
    assert dividend.carbon_dividend_fund > 0.0
    assert abs(dividend.carbon_dividend_balance_error) <= 1e-12
    assert not np.allclose(standard_costs, dividend_costs)
    assert np.isclose(standard_costs.sum(), dividend_costs.sum(), atol=1e-9)


def test_v3_equal_ablation_preserves_dispatch_and_budget_balance():
    action = np.array(
        [
            [-0.09300422, -0.73191661, -0.19377403, -0.59308952, -0.47537332],
            [0.50072935, -0.43918248, -0.02961805, 0.96147440, 0.92331439],
            [0.44957988, 0.08245371, -0.44621759, -0.67869598, 0.93985083],
        ]
    )
    initial = env()
    state = np.concatenate(initial.reset_train(), axis=0)
    standard = env(market_mechanism="standard")
    equal = env(market_mechanism="carbon_dividend_equal_v3", cda_carbon_price=0.02)

    standard_output = standard.step(action, state, step=0, flag=0)
    equal_output = equal.step(action, state, step=0, flag=0)

    assert np.allclose(standard.trade_record_e, equal.trade_record_e)
    assert np.isclose(equal.carbon_credit[1], equal.carbon_credit[2])
    assert np.isclose(equal.carbon_credit[2], equal.carbon_credit[3])
    assert abs(equal.carbon_dividend_balance_error) <= 1e-12
    standard_cost = sum(item[0] for item in standard_output[4:7])
    equal_cost = sum(item[0] for item in equal_output[4:7])
    assert np.isclose(standard_cost, equal_cost, atol=1e-9)
