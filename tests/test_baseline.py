import numpy as np
import pandas as pd
import pytest

from src import config
from src.baseline import (MA4ResidualQuantileForecaster, build_wide, evaluate_holdout,
                          forecast_planning_week, ma4_point_forecast)
from src.data_generator import generate_dataset
from src.schemas import enforce_quantile_rules, validate_forecast_frame

_cache = {}


def sales_normal():
    if "s" not in _cache:
        _cache["s"] = generate_dataset("normal").weekly
    return _cache["s"]


def linear_sales(slopes, weeks=30, skus=None):
    """Series i: demand_t = slope_i * t  (so MA4 residual is exactly 2.5*slope)."""
    rows = []
    for i, m in enumerate(slopes):
        for t in range(1, weeks + 1):
            rows.append(("S%02d" % (i + 1), (skus or ["K01"] * len(slopes))[i], t, m * t))
    return pd.DataFrame(rows, columns=["store_id", "sku_id", "week", "demand"])


# --- MA4 -------------------------------------------------------------------
def test_ma4_hand_calculation():
    wide = build_wide(linear_sales([10]))
    assert ma4_point_forecast(wide, 5)[0] == pytest.approx(25.0)    # mean(10,20,30,40)
    assert ma4_point_forecast(wide, 29)[0] == pytest.approx(265.0)  # mean(250..280)


def test_ma4_needs_four_complete_prior_weeks():
    wide = build_wide(linear_sales([10]))
    with pytest.raises(ValueError):
        ma4_point_forecast(wide, 4)
    wide.iloc[0, 9] = np.nan
    with pytest.raises(ValueError):
        ma4_point_forecast(wide, 12)  # window 8..11 contains the missing week 10


def test_ma4_plus_residual_quantile_hand_calculation():
    # demand = 10t => MA4(t) = 10t - 25, residual = 25 for every quantile => forecast = 10t.
    sales = linear_sales([10])
    model = MA4ResidualQuantileForecaster().fit(sales, (5, 28))  # 24 residuals >= 20
    fc = model.predict(sales, [29])
    assert not fc["used_pooled_residuals"].any()
    for col in config.QUANTILE_COLUMNS:
        assert fc[col].iloc[0] == pytest.approx(290.0)


def test_pooled_fallback_under_20_residuals_and_flagged():
    sales = linear_sales([10, 20])  # same SKU; residuals 25 and 50
    model = MA4ResidualQuantileForecaster().fit(sales, (5, 20))  # 16 residuals < 20
    assert model.used_pooled_.all()
    fc = model.predict(sales, [21]).set_index("store_id")
    assert fc["used_pooled_residuals"].all()
    # pooled P50 = median of 16x25 and 16x50 = 37.5; MA4 = 10*21 - 25 = 185
    assert fc.loc["S01", "P50"] == pytest.approx(222.5)


# --- past-only residuals and leakage -----------------------------------------
def test_residual_quantiles_ignore_data_after_fit_window():
    s = sales_normal()
    changed = s.copy()
    changed.loc[changed["week"] > 40, "demand"] = 99999
    a = MA4ResidualQuantileForecaster().fit(s, config.TRAIN_WEEKS)
    b = MA4ResidualQuantileForecaster().fit(changed, config.TRAIN_WEEKS)
    np.testing.assert_array_equal(a.residual_quantiles_, b.residual_quantiles_)


def test_forecast_does_not_change_when_target_or_future_demand_changes():
    s = sales_normal()
    model = MA4ResidualQuantileForecaster().fit(s, config.TRAIN_WEEKS)
    base = model.predict(s, [41, 47])
    changed = s.copy()
    changed.loc[changed["week"] >= 47, "demand"] = 123456  # week 47 and later altered
    after = model.predict(changed, [41, 47])
    pd.testing.assert_frame_equal(base, after)  # week 47 forecast ignores week 47+ demand


def test_planning_week_forecast_ignores_week_53_data():
    s = sales_normal()
    extra = s[s["week"] == 52].assign(week=53, demand=10**6)
    leaked = pd.concat([s, extra], ignore_index=True)
    a, _ = forecast_planning_week(s)
    b, _ = forecast_planning_week(leaked)
    pd.testing.assert_frame_equal(a, b)


# --- output contract -------------------------------------------------------
def test_holdout_windows_and_row_count():
    res = evaluate_holdout(sales_normal())
    assert res.metadata["fit_weeks"] == [5, 40] and res.metadata["eval_weeks"] == [41, 52]
    assert len(res.forecasts) == 480 == res.summary["n_observations"]
    assert sorted(res.forecasts["planning_week"].unique()) == list(range(41, 53))
    assert res.metadata["residuals_per_series_min"] == 36  # weeks 5..40
    assert res.metadata["n_series_using_pooled_residuals"] == 0


def test_final_refit_window_and_planning_week():
    fc, model = forecast_planning_week(sales_normal())
    assert model.fit_weeks_ == (5, 52) and len(fc) == 40
    assert (fc["planning_week"] == 53).all() and (fc["method"] == "MA4").all()
    assert model.n_residuals_.min() == 48  # weeks 5..52


def test_forecasts_nonnegative_ordered_and_valid_for_all_scenarios():
    for name in config.SCENARIOS:
        s = generate_dataset(name).weekly
        for frame in (evaluate_holdout(s).forecasts, forecast_planning_week(s)[0]):
            assert validate_forecast_frame(frame) == []
            assert (frame[list(config.QUANTILE_COLUMNS)].to_numpy() >= 0).all()


def test_results_reproducible():
    a, b = evaluate_holdout(sales_normal()), evaluate_holdout(sales_normal())
    pd.testing.assert_frame_equal(a.forecasts, b.forecasts)
    pd.testing.assert_frame_equal(a.per_quantile, b.per_quantile)


def test_clamp_and_quantile_crossing_correction():
    frame = pd.DataFrame([[-3.0, 2.0, 1.0, 5.0, 4.0, 6.0], [1, 2, 3, 4, 5, 6]], columns=list(config.QUANTILE_COLUMNS))
    fixed, n = enforce_quantile_rules(frame)
    assert n == 1
    assert fixed.iloc[0].tolist() == [0.0, 1.0, 2.0, 4.0, 5.0, 6.0]
    assert fixed.iloc[1].tolist() == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError):
        enforce_quantile_rules(frame.assign(P50=np.nan))


def test_forecaster_requires_fit_and_rejects_unknown_series():
    s = sales_normal()
    with pytest.raises(RuntimeError):
        MA4ResidualQuantileForecaster().predict(s, [41])
    model = MA4ResidualQuantileForecaster().fit(s, config.TRAIN_WEEKS)
    with pytest.raises(ValueError):
        model.predict(s[s["store_id"] != "S01"], [41])  # missing series -> no fabricated forecast
