import numpy as np
import pandas as pd
import pytest

from src.decision_evaluation import DemandDraws, draw_demand, evaluate_plan, recommend


def test_evaluation_hand_computed_and_zero_demand():
    stock = pd.DataFrame([["S01", "K01", 2]], columns=["store_id", "sku_id", "opening_units"])
    plan = dict(checker=dict(passed=True), shipments=[dict(store_id="S01", sku_id="K01", units=3)],
                totals=dict(freight=6, shipped_units=3, unmet_units=7))
    draws = DemandDraws((("S01", "K01"),), np.array([[0], [4], [10]]), 42)
    result = evaluate_plan(plan, stock, draws)
    # Available 5. Costs: 6+15, 6+3, 6+60. Fulfillment: 1, 1, 0.5.
    assert result["realized_cost_mean"] == pytest.approx(32)
    assert result["realized_cost_p90"] == pytest.approx(57)
    assert result["fulfillment_mean"] == pytest.approx(2.5 / 3)
    assert result["fulfillment_p10"] == pytest.approx(.6)
    assert result["simulated_lost_sales_mean"] == pytest.approx(5 / 3)
    assert result["unmet_units"] == 7  # target shortage is not simulated lost sales
    before = draws.values.copy()
    assert evaluate_plan(plan, stock, draws) == result
    np.testing.assert_array_equal(before, draws.values)
    plan["checker"]["passed"] = False
    with pytest.raises(ValueError):
        evaluate_plan(plan, stock, draws)


def test_negative_binomial_parameterization_and_reproducibility():
    truth = pd.DataFrame([["S01", "K01", 53, 20., 5.]],
                         columns=["store_id", "sku_id", "week", "true_mean", "dispersion"])
    a = draw_demand(truth, [("S01", "K01")])
    b = draw_demand(truth, [("S01", "K01")])
    expected = np.random.default_rng(42).negative_binomial([5.], [5. / 25.], size=(200, 1))
    np.testing.assert_array_equal(a.values, expected)
    np.testing.assert_array_equal(a.values, b.values)
    assert a.digest == b.digest
    assert not a.values.flags.writeable
    truth.loc[0, "true_mean"] = 0
    assert not draw_demand(truth, a.pairs).values.any()
    truth.loc[0, "dispersion"] = 0
    with pytest.raises(ValueError):
        draw_demand(truth, a.pairs)


def test_draw_pair_validation():
    truth = pd.DataFrame([["S01", "K01", 53, 20., 5.]] * 2,
                         columns=["store_id", "sku_id", "week", "true_mean", "dispersion"])
    with pytest.raises(ValueError):
        draw_demand(truth, [("S01", "K01")])
    with pytest.raises(ValueError):
        draw_demand(truth.iloc[:1], [("S02", "K01")])
    with pytest.raises(ValueError):
        draw_demand(truth.iloc[:1], [("S01", "K01")] * 2)


def test_recommendation_uses_cost_not_level_order():
    rows = [dict(level="P50", fulfillment_p10=.8, realized_cost_mean=50),
            dict(level="P60", fulfillment_p10=.92, realized_cost_mean=100),
            dict(level="P70", fulfillment_p10=.91, realized_cost_mean=90),
            dict(level="P80", fulfillment_p10=.85, realized_cost_mean=20)]
    assert recommend(rows, .9) == "P70"
    assert recommend(rows, .99) is None
    with pytest.raises(ValueError):
        recommend(rows, 90)


def test_network_fulfillment_is_demand_weighted():
    stock = pd.DataFrame([["S01", "K01", 1], ["S02", "K01", 0]],
                         columns=["store_id", "sku_id", "opening_units"])
    plan = dict(checker=dict(passed=True), shipments=[], totals=dict(freight=0, shipped_units=0, unmet_units=0))
    draws = DemandDraws((("S01", "K01"), ("S02", "K01")), np.array([[1, 9]]), 42)
    assert evaluate_plan(plan, stock, draws)["fulfillment_mean"] == pytest.approx(.1)
