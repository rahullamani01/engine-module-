"""Tests for the Byte Works forecasting engine."""

import json

import pytest

from src import config
from src import engine


@pytest.fixture(scope="module")
def completed_pipeline(tmp_path_factory):
    """Run the actual forecasting pipeline once for these tests."""

    root = tmp_path_factory.mktemp("engine")

    result = engine.run_pipeline(
        scenario="normal",
        data_dir=root / "data",
        output_dir=root / "outputs",
    )

    return result, root


def test_pipeline_generates_artifacts(completed_pipeline):
    result, root = completed_pipeline

    output = root / "outputs" / "normal"

    expected = [
        "model_comparison.csv",
        "calibration.csv",
        "holdout_forecasts.csv",
        "planning_forecast.csv",
        "selection.json",
    ]

    for filename in expected:
        assert (output / filename).is_file()

    assert result.scenario == "normal"
    assert result.forecast_rows == 40
    assert result.planning_week == 53
    assert result.selected_method in ("MA4", "GB")
    assert result.ml_driven == (result.selected_method == "GB")


def test_saved_forecast_contract(completed_pipeline):
    _, root = completed_pipeline

    forecast = engine.load_forecast(
        "normal",
        output_dir=root / "outputs",
    )

    assert len(forecast) == config.N_SERIES
    assert forecast["planning_week"].eq(53).all()

    assert not forecast.duplicated(
        ["store_id", "sku_id"]
    ).any()

    quantiles = forecast[
        list(config.QUANTILE_COLUMNS)
    ].to_numpy()

    assert (quantiles >= 0).all()

    assert (quantiles[:, 1:] >= quantiles[:, :-1]).all()


def test_selection_metadata(completed_pipeline):
    result, root = completed_pipeline

    metadata = engine.load_selection(
        "normal",
        output_dir=root / "outputs",
    )

    assert metadata["selected"] == result.selected_method
    assert metadata["ml_driven"] == result.ml_driven
    assert metadata["seed"] == config.SEED
    assert metadata["synthetic_data"] is True


def test_invalid_data_stops_pipeline(monkeypatch, tmp_path):
    def reject_data(dataset):
        raise ValueError("Invalid dataset")

    monkeypatch.setattr(
        engine,
        "validate_or_raise",
        reject_data,
    )

    with pytest.raises(ValueError, match="Invalid dataset"):
        engine.run_pipeline(
            scenario="normal",
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "outputs",
        )

    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "outputs").exists()