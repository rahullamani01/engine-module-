"""Byte Works — Frontend Bridge Integration Tests.

Tests integration between Stream A's forecasting engine
and Stream C's Streamlit frontend.

Verifies:
    - Scenario-name translation.
    - Canonical scenario directories.
    - Frontend forecast JSON contract.
    - Inventory-adjusted shipment targets.
    - Invalid inventory rejection.
    - Protection against overwriting existing results.
    - Protection of the frontend demo directory.

Uses synthetic fixtures.
No model training or optimization is performed.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config
from src import frontend_bridge as bridge
from src.schemas import FORECAST_COLUMNS


# ============================================================
# TEST FIXTURE
# ============================================================

@pytest.fixture
def sample_data(monkeypatch):
    """Create 40 valid forecasts and corresponding opening stock."""

    rows = []

    for store in range(1, 9):
        for sku in range(1, 6):

            row = {
                "store_id": f"S{store:02d}",
                "sku_id": f"K{sku:02d}",
                "planning_week": config.PLANNING_WEEK,
                "method": "GB",
                "used_pooled_residuals": False,
            }

            quantile_values = [
                50.2,
                60.2,
                70.2,
                80.2,
                90.2,
                95.2,
            ]

            for level, value in zip(
                config.QUANTILE_COLUMNS,
                quantile_values,
            ):
                row[level] = value

            rows.append(row)

    # Follow the official forecasting column contract.
    forecast = pd.DataFrame(rows)[list(FORECAST_COLUMNS)]

    # Create synthetic opening inventory.
    stock = forecast[
        ["store_id", "sku_id"]
    ].copy()

    stock["opening_units"] = 20

    # Prevent tests from loading production forecast files.
    monkeypatch.setattr(
        bridge,
        "load_forecast",
        lambda scenario, output_dir: forecast.copy(),
    )

    return stock


# ============================================================
# TEST 1 — SCENARIO TRANSLATION
# ============================================================

def test_scenario_translation():

    assert bridge.normalize_scenario("normal") == "normal"

    assert bridge.normalize_scenario("scarce") == "scarce_supply"

    assert bridge.normalize_scenario("expiry") == "expiry_trap"


# ============================================================
# TEST 2 — CANONICAL SCENARIO NAMES
# ============================================================

def test_canonical_scenario_names():

    assert bridge.normalize_scenario(
        "scarce_supply"
    ) == "scarce_supply"

    assert bridge.normalize_scenario(
        "expiry_trap"
    ) == "expiry_trap"


# ============================================================
# TEST 3 — REVERSE SCENARIO TRANSLATION
# ============================================================

def test_reverse_scenario_translation():

    assert bridge.frontend_scenario("normal") == "normal"

    assert bridge.frontend_scenario(
        "scarce_supply"
    ) == "scarce"

    assert bridge.frontend_scenario(
        "expiry_trap"
    ) == "expiry"


# ============================================================
# TEST 4 — INVALID SCENARIO REJECTION
# ============================================================

def test_invalid_scenario_rejected():

    with pytest.raises(
        ValueError,
        match="Unknown scenario",
    ):
        bridge.normalize_scenario("invalid")


# ============================================================
# TEST 5 — FORECAST ROW COUNT
# ============================================================

def test_frontend_forecast_has_40_rows(sample_data):

    result = bridge.prepare_frontend_forecast(
        "normal",
        sample_data,
    )

    assert isinstance(result, dict)

    assert "rows" in result

    assert len(result["rows"]) == config.N_SERIES


# ============================================================
# TEST 6 — REQUIRED FRONTEND FIELDS
# ============================================================

def test_frontend_forecast_fields(sample_data):

    result = bridge.prepare_frontend_forecast(
        "normal",
        sample_data,
    )

    row = result["rows"][0]

    required_fields = {
        "store_id",
        "sku_id",
        "opening_units",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
        "p95",
        "target_p50",
        "target_p90",
    }

    assert required_fields.issubset(row.keys())

    assert "P50" not in row
    assert "P90" not in row


# ============================================================
# TEST 7 — SHIPMENT TARGET CALCULATION
# ============================================================

def test_target_calculation(sample_data):

    result = bridge.prepare_frontend_forecast(
        "normal",
        sample_data,
    )

    row = result["rows"][0]

    assert row["opening_units"] == 20

    assert row["p50"] == pytest.approx(50.2)
    assert row["p90"] == pytest.approx(90.2)

    # ceil(50.2) - 20 = 31
    assert row["target_p50"] == 31

    # ceil(90.2) - 20 = 71
    assert row["target_p90"] == 71


# ============================================================
# TEST 8 — CANONICAL EXPORT DIRECTORY
# ============================================================

def test_export_to_correct_frontend_directory(
    sample_data,
    tmp_path,
):

    destination = bridge.export_frontend_forecast(
        "scarce",
        sample_data,
        results_dir=tmp_path,
    )

    assert destination == (
        tmp_path / "scarce_supply" / "forecast.json"
    )

    assert destination.is_file()

    payload = json.loads(
        destination.read_text(encoding="utf-8")
    )

    assert "rows" in payload

    assert len(payload["rows"]) == config.N_SERIES


# ============================================================
# TEST 9 — NORMAL SCENARIO EXPORT
# ============================================================

def test_normal_scenario_export(
    sample_data,
    tmp_path,
):

    destination = bridge.export_frontend_forecast(
        "normal",
        sample_data,
        results_dir=tmp_path,
    )

    assert destination == (
        tmp_path / "normal" / "forecast.json"
    )

    assert destination.is_file()


# ============================================================
# TEST 10 — EXPIRY SCENARIO EXPORT
# ============================================================

def test_expiry_scenario_export(
    sample_data,
    tmp_path,
):

    destination = bridge.export_frontend_forecast(
        "expiry",
        sample_data,
        results_dir=tmp_path,
    )

    assert destination == (
        tmp_path / "expiry_trap" / "forecast.json"
    )

    assert destination.is_file()


# ============================================================
# TEST 11 — EXISTING RESULTS PROTECTED
# ============================================================

def test_existing_results_never_overwritten(
    sample_data,
    tmp_path,
):

    destination = (
        tmp_path / "normal" / "forecast.json"
    )

    destination.parent.mkdir(parents=True)

    destination.write_text(
        '{"existing": true}',
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):

        bridge.export_frontend_forecast(
            "normal",
            sample_data,
            results_dir=tmp_path,
        )

    assert json.loads(
        destination.read_text(encoding="utf-8")
    ) == {"existing": True}


# ============================================================
# TEST 12 — MISSING OPENING STOCK
# ============================================================

def test_missing_stock_rejected(sample_data):

    incomplete = sample_data.iloc[:-1]

    with pytest.raises(ValueError):

        bridge.prepare_frontend_forecast(
            "normal",
            incomplete,
        )


# ============================================================
# TEST 13 — NEGATIVE OPENING STOCK
# ============================================================

def test_negative_stock_rejected(sample_data):

    invalid = sample_data.copy()

    invalid.loc[0, "opening_units"] = -10

    with pytest.raises(ValueError):

        bridge.prepare_frontend_forecast(
            "normal",
            invalid,
        )


# ============================================================
# TEST 14 — FRACTIONAL OPENING STOCK
# FIXED FOR PANDAS 3.0.2
# ============================================================

def test_fractional_stock_rejected(sample_data):
    """Fractional inventory must be rejected by our validator."""

    invalid = sample_data.copy()

    # IMPORTANT:
    # pandas 3.x does not allow setting 12.5
    # directly into an int64 column.
    #
    # Convert the column to float first.
    invalid["opening_units"] = (
        invalid["opening_units"].astype(float)
    )

    # Introduce deliberately invalid inventory.
    invalid.loc[0, "opening_units"] = 12.5

    # Now the actual application validator should reject it.
    with pytest.raises(
        ValueError,
        match="whole units",
    ):
        bridge.prepare_frontend_forecast(
            "normal",
            invalid,
        )


# ============================================================
# TEST 15 — DUPLICATE OPENING STOCK
# ============================================================

def test_duplicate_stock_rejected(sample_data):

    duplicate = pd.concat(
        [
            sample_data,
            sample_data.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError):

        bridge.prepare_frontend_forecast(
            "normal",
            duplicate,
        )


# ============================================================
# TEST 16 — DEMO DIRECTORY PROTECTION
# ============================================================

def test_demo_directory_protected(
    sample_data,
    tmp_path,
):

    demo_dir = tmp_path / "results"

    with pytest.raises(ValueError):

        bridge.export_frontend_forecast(
            "normal",
            sample_data,
            results_dir=demo_dir,
        )

    assert not (
        demo_dir / "normal" / "forecast.json"
    ).exists()


# ============================================================
# TEST 17 — EXPORTED JSON TARGETS
# ============================================================

def test_exported_json_targets(
    sample_data,
    tmp_path,
):

    destination = bridge.export_frontend_forecast(
        "normal",
        sample_data,
        results_dir=tmp_path,
    )

    payload = json.loads(
        destination.read_text(encoding="utf-8")
    )

    for row in payload["rows"]:

        assert row["target_p50"] == 31

        assert row["target_p90"] == 71

        assert row["opening_units"] == 20


# ============================================================
# TEST 18 — QUANTILE ORDERING
# ============================================================

def test_quantiles_remain_ordered(sample_data):

    result = bridge.prepare_frontend_forecast(
        "normal",
        sample_data,
    )

    for row in result["rows"]:

        quantiles = [
            row["p50"],
            row["p60"],
            row["p70"],
            row["p80"],
            row["p90"],
            row["p95"],
        ]

        assert quantiles == sorted(quantiles)