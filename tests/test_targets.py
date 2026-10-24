"""Tests for forecast-to-optimizer target calculation."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.targets import build_targets
from src.engine import load_forecast


@pytest.fixture
def forecast():
    return load_forecast("normal")


@pytest.fixture
def opening_stock(forecast):
    stock = forecast[["store_id", "sku_id"]].copy()
    stock["opening_units"] = 20
    return stock


def test_target_calculation(forecast, opening_stock):
    result = build_targets(
        forecast,
        opening_stock,
        level="P90",
    )

    expected = np.maximum(
        0,
        np.ceil(forecast["P90"].to_numpy()) - 20,
    ).astype(int)

    assert np.array_equal(
        result["target_units"].to_numpy(),
        expected,
    )


def test_targets_are_nonnegative_integers(
    forecast,
    opening_stock,
):
    result = build_targets(
        forecast,
        opening_stock,
        "P90",
    )

    assert len(result) == config.N_SERIES
    assert (result["target_units"] >= 0).all()
    assert pd.api.types.is_integer_dtype(
        result["target_units"]
    )


def test_excess_opening_stock_gives_zero_target(
    forecast,
    opening_stock,
):
    opening_stock["opening_units"] = 100000

    result = build_targets(
        forecast,
        opening_stock,
        "P90",
    )

    assert (result["target_units"] == 0).all()


def test_missing_stock_rejected(forecast, opening_stock):
    incomplete = opening_stock.iloc[:-1]

    with pytest.raises(ValueError):
        build_targets(forecast, incomplete)


def test_duplicate_stock_rejected(forecast, opening_stock):
    duplicate = pd.concat(
        [opening_stock, opening_stock.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError):
        build_targets(forecast, duplicate)


def test_negative_stock_rejected(forecast, opening_stock):
    opening_stock.loc[0, "opening_units"] = -5

    with pytest.raises(ValueError):
        build_targets(forecast, opening_stock)


def test_invalid_planning_level_rejected(
    forecast,
    opening_stock,
):
    with pytest.raises(ValueError):
        build_targets(
            forecast,
            opening_stock,
            "P99",
        )


def test_higher_quantile_never_reduces_target(
    forecast,
    opening_stock,
):
    p50 = build_targets(
        forecast,
        opening_stock,
        "P50",
    )

    p90 = build_targets(
        forecast,
        opening_stock,
        "P90",
    )

    assert (
        p90["target_units"].to_numpy()
        >= p50["target_units"].to_numpy()
    ).all()