import numpy as np
import pandas as pd
import pytest

from src import config
from src.baseline import MA4ResidualQuantileForecaster
from src.data_generator import generate_dataset
from src.quantile_model import GradientBoostingQuantileForecaster
from src.schemas import FORECAST_COLUMNS, validate_forecast_frame

_cache = {}


def sales():
    if "s" not in _cache:
        _cache["s"] = generate_dataset("normal").weekly
    return _cache["s"]


def fitted():
    if "m" not in _cache:
        _cache["m"] = GradientBoostingQuantileForecaster().fit(sales(), config.TRAIN_WEEKS)
    return _cache["m"]


def test_one_model_per_quantile_with_matching_alpha():
    m = fitted()
    assert sorted(m.models_) == list(config.QUANTILES)
    for q, model in m.models_.items():
        assert model.loss == "quantile" and model.alpha == q
    assert m.n_train_rows_ == 40 * 36  # weeks 5..40


def test_same_interface_and_output_contract_as_baseline():
    fc = fitted().predict(sales(), config.weeks_in(config.EVAL_WEEKS))
    base = MA4ResidualQuantileForecaster().fit(sales(), config.TRAIN_WEEKS).predict(
        sales(), config.weeks_in(config.EVAL_WEEKS))
    assert list(fc.columns) == list(FORECAST_COLUMNS) == list(base.columns)
    assert len(fc) == 480 and (fc["method"] == "GB").all()
    assert fc[["store_id", "sku_id", "planning_week"]].equals(base[["store_id", "sku_id", "planning_week"]])
    assert validate_forecast_frame(fc) == []
    assert (fc[list(config.QUANTILE_COLUMNS)].to_numpy() >= 0).all()


def test_training_window_excludes_later_data():
    changed = sales().copy()
    changed.loc[changed["week"] > 40, "demand"] = 99999
    a = fitted().predict(sales(), [41])
    b = GradientBoostingQuantileForecaster().fit(changed, config.TRAIN_WEEKS)
    # Same training weeks => identical models; compare on unchanged history for week 41.
    pd.testing.assert_frame_equal(a, b.predict(sales(), [41]))


def test_forecast_ignores_target_and_future_demand():
    m = fitted()
    base = m.predict(sales(), [41, 47])
    changed = sales().copy()
    changed.loc[changed["week"] >= 47, "demand"] = 123456
    pd.testing.assert_frame_equal(base, m.predict(changed, [41, 47]))


def test_planning_week_forecast_ignores_week_53_data():
    leaked = pd.concat([sales(), sales()[sales()["week"] == 52].assign(week=53, demand=10**6)], ignore_index=True)
    a = GradientBoostingQuantileForecaster().fit(sales(), config.FINAL_FIT_WEEKS).predict(sales(), [53])
    b = GradientBoostingQuantileForecaster().fit(leaked, config.FINAL_FIT_WEEKS).predict(leaked, [53])
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 40 and (a["planning_week"] == 53).all()


def test_reproducible_under_fixed_seed():
    a = GradientBoostingQuantileForecaster().fit(sales(), config.TRAIN_WEEKS).predict(sales(), [41, 52])
    pd.testing.assert_frame_equal(a, fitted().predict(sales(), [41, 52]))


def test_requires_fit_and_rejects_missing_series():
    with pytest.raises(RuntimeError):
        GradientBoostingQuantileForecaster().predict(sales(), [41])
    with pytest.raises(ValueError):
        fitted().predict(sales()[sales()["store_id"] != "S01"], [41])  # no fabricated forecast