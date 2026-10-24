"""Tests for the frontend forecast JSON adapter."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.forecast_adapter import (
    build_forecast_json,
    export_forecast_json,
)


@pytest.fixture
def sample_data():
    rows = []

    for store in range(1, 9):
        for sku in range(1, 6):
            row = {
                "store_id": f"S{store:02d}",
                "sku_id": f"K{sku:02d}",
                "planning_week": 53,
                "method": "GB",
                "used_pooled_residuals": False,
            }

            for level, value in zip(
                config.QUANTILE_COLUMNS,
                [50.2, 60.2, 70.2, 80.2, 90.2, 95.2],
            ):
                row[level] = value

            rows.append(row)

    forecast = pd.DataFrame(rows)

    stock = forecast[["store_id", "sku_id"]].copy()
    stock["opening_units"] = 20

    return forecast, stock


def test_forecast_json_contains_40_rows(sample_data):
    forecast, stock = sample_data

    payload = build_forecast_json(forecast, stock)

    assert len(payload["rows"]) == 40


def test_frontend_field_names(sample_data):
    forecast, stock = sample_data

    payload = build_forecast_json(forecast, stock)

    row = payload["rows"][0]

    assert "p50" in row
    assert "p90" in row
    assert "target_p50" in row
    assert "target_p90" in row

    assert "P90" not in row


def test_targets_match_forecast_minus_stock(sample_data):
    forecast, stock = sample_data

    payload = build_forecast_json(forecast, stock)

    row = payload["rows"][0]

    assert row["opening_units"] == 20
    assert row["target_p90"] == 71

    expected = max(0, int(np.ceil(row["p90"])) - 20)

    assert row["target_p90"] == expected


def test_quantiles_ordered(sample_data):
    forecast, stock = sample_data

    payload = build_forecast_json(forecast, stock)

    for row in payload["rows"]:
        values = [
            row[level.lower()]
            for level in config.QUANTILE_COLUMNS
        ]

        assert values == sorted(values)


def test_missing_stock_rejected(sample_data):
    forecast, stock = sample_data

    with pytest.raises(ValueError):
        build_forecast_json(
            forecast,
            stock.iloc[:-1],
        )


def test_does_not_overwrite_existing_result(
    sample_data, monkeypatch, tmp_path
):
    forecast, stock = sample_data

    monkeypatch.setattr(
        "src.forecast_adapter.load_forecast",
        lambda scenario, output_dir: forecast.copy(),
    )

    destination = tmp_path / "results" / "normal"
    destination.mkdir(parents=True)

    existing = destination / "forecast.json"
    existing.write_text('{"mock": true}')

    with pytest.raises(FileExistsError):
        export_forecast_json(
            "normal",
            stock,
            output_dir=tmp_path,
            results_dir=tmp_path / "results",
        )

    assert existing.read_text() == '{"mock": true}'