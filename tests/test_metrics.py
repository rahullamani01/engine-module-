import numpy as np
import pandas as pd
import pytest

from src.metrics import (evaluate_quantile_forecasts, merge_with_actuals, observed_coverage,
                         pinball_loss, wape)


def test_pinball_hand_values():
    assert pinball_loss([10], [8], 0.9) == pytest.approx(1.8)    # under-forecast: tau * 2
    assert pinball_loss([10], [12], 0.9) == pytest.approx(0.2)   # over-forecast: (1 - tau) * 2
    assert pinball_loss([10, 4], [10, 4], 0.7) == 0.0
    assert pinball_loss([10, 0], [8, 3], 0.5) == pytest.approx((1.0 + 1.5) / 2)


def test_pinball_rejects_bad_inputs():
    with pytest.raises(ValueError):
        pinball_loss([1], [1], 1.0)
    with pytest.raises(ValueError):
        pinball_loss([1, 2], [1], 0.5)
    with pytest.raises(ValueError):
        pinball_loss([1], [np.nan], 0.5)


def test_coverage_and_wape():
    assert observed_coverage([1, 2, 3, 4], [1, 1, 5, 5]) == 0.75
    assert wape([10, 10], [8, 13]) == pytest.approx(5 / 20)
    assert np.isnan(wape([0, 0], [1, 1]))


def _frame(pred):
    return pd.DataFrame({"store_id": ["S01"] * 2, "sku_id": ["K01"] * 2, "planning_week": [1, 2],
                         **{c: pred for c in ("P50", "P60", "P70", "P80", "P90", "P95")}})


def test_evaluate_and_missing_actuals():
    sales = pd.DataFrame({"store_id": ["S01"] * 2, "sku_id": ["K01"] * 2, "week": [1, 2], "demand": [10, 20]})
    table, summary = evaluate_quantile_forecasts(_frame([10.0, 20.0]), sales)
    assert summary["n_observations"] == 2 and summary["mean_pinball_loss"] == 0.0
    assert (table["observed_coverage"] == 1.0).all()
    with pytest.raises(ValueError):
        merge_with_actuals(_frame([1.0, 1.0]).assign(planning_week=[1, 9]), sales)


def test_miscalibration_flag_uses_ten_point_tolerance():
    sales = pd.DataFrame({"store_id": ["S01"] * 10, "sku_id": ["K01"] * 10,
                          "week": range(1, 11), "demand": range(1, 11)})
    # Forecast = 10 everywhere: every actual <= forecast => coverage 1.0, far above P50/P60/P70 nominal.
    fc = pd.DataFrame({"store_id": ["S01"] * 10, "sku_id": ["K01"] * 10, "planning_week": range(1, 11),
                       **{c: 10.0 for c in ("P50", "P60", "P70", "P80", "P90", "P95")}})
    table, _ = evaluate_quantile_forecasts(fc, sales)
    flags = dict(zip(table["column"], table["miscalibrated"]))
    assert flags["P50"] and flags["P60"] and flags["P70"]   # gaps 0.50, 0.40, 0.30 > 0.10
    assert flags["P80"]                                       # gap 0.20
    assert not flags["P90"] and not flags["P95"]              # gaps 0.10 (boundary, allowed) and 0.05
