"""Tests for saved Byte Works forecasting artifact auditing.

These tests are self-contained:
- No model training required.
- No dependency on existing output files.
- All artifacts are created in temporary directories.
- Test values are synthetic fixtures, not measured results.
"""

import json

import pandas as pd
import pytest

from src import config
from src.forecast_audit import audit_forecast_artifacts
from src.schemas import FORECAST_COLUMNS


@pytest.fixture
def artifacts(tmp_path):
    """Create a valid set of forecasting artifacts for testing."""

    directory = tmp_path / "normal"
    directory.mkdir()

    # --------------------------------------------------
    # 1. Create 40 valid store-SKU forecast records.
    # --------------------------------------------------

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

            quantile_values = [50, 60, 70, 80, 90, 95]

            for level, value in zip(
                config.QUANTILE_COLUMNS,
                quantile_values,
            ):
                row[level] = float(value)

            rows.append(row)

    # IMPORTANT:
    # Enforce the official forecast column order.
    forecast = pd.DataFrame(rows)
    forecast = forecast[list(FORECAST_COLUMNS)]

    forecast.to_csv(
        directory / "planning_forecast.csv",
        index=False,
    )

    # --------------------------------------------------
    # 2. Create model comparison metadata.
    # --------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "method": "MA4",
                "mean_pinball_loss": 5.0,
                "selected": False,
                "n_observations": 480,
            },
            {
                "method": "GB",
                "mean_pinball_loss": 4.0,
                "selected": True,
                "n_observations": 480,
            },
        ]
    )

    comparison.to_csv(
        directory / "model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------
    # 3. Create selection metadata.
    # --------------------------------------------------

    metadata = {
        "selected": "GB",
        "ml_driven": True,
        "seed": config.SEED,
        "synthetic_data": True,
    }

    (directory / "selection.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # 4. Create remaining required artifacts.
    #
    # The current auditor checks their existence only.
    # Their full contents are not validated yet.
    # --------------------------------------------------

    (directory / "holdout_forecasts.csv").write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    (directory / "calibration.csv").write_text(
        "placeholder\n",
        encoding="utf-8",
    )

    return tmp_path, directory


# ======================================================
# TEST 1 — Valid artifacts must pass
# ======================================================

def test_valid_forecast_passes(artifacts):

    root, _ = artifacts

    result = audit_forecast_artifacts(
        "normal",
        output_dir=root,
    )

    assert result["valid"] is True, result["errors"]

    assert result["forecast_rows"] == config.N_SERIES

    assert result["selected_method"] == "GB"

    assert result["errors"] == []


# ======================================================
# TEST 2 — Missing artifacts must be rejected
# ======================================================

def test_missing_artifact_rejected(artifacts):

    root, directory = artifacts

    (directory / "selection.json").unlink()

    result = audit_forecast_artifacts(
        "normal",
        output_dir=root,
    )

    assert result["valid"] is False

    assert any(
        "Missing artifact" in error
        for error in result["errors"]
    )


# ======================================================
# TEST 3 — Wrong planning week must be rejected
# ======================================================

def test_wrong_planning_week_rejected(artifacts):

    root, directory = artifacts

    path = directory / "planning_forecast.csv"

    forecast = pd.read_csv(path)

    forecast["planning_week"] = 52

    forecast.to_csv(
        path,
        index=False,
    )

    result = audit_forecast_artifacts(
        "normal",
        output_dir=root,
    )

    assert result["valid"] is False

    assert any(
        "planning week" in error.lower()
        for error in result["errors"]
    )


# ======================================================
# TEST 4 — Duplicate store-SKU forecasts rejected
# ======================================================

def test_duplicate_forecast_rejected(artifacts):

    root, directory = artifacts

    path = directory / "planning_forecast.csv"

    forecast = pd.read_csv(path)

    # Make the second row duplicate the first row's IDs.
    forecast.loc[1, "store_id"] = forecast.loc[0, "store_id"]

    forecast.loc[1, "sku_id"] = forecast.loc[0, "sku_id"]

    forecast.to_csv(
        path,
        index=False,
    )

    result = audit_forecast_artifacts(
        "normal",
        output_dir=root,
    )

    assert result["valid"] is False

    assert any(
        "duplicate" in error.lower()
        for error in result["errors"]
    )


# ======================================================
# TEST 5 — Incorrect model selection rejected
# ======================================================

def test_incorrect_model_selection_rejected(artifacts):

    root, directory = artifacts

    path = directory / "selection.json"

    metadata = json.loads(
        path.read_text(encoding="utf-8")
    )

    # Claim MA4 was selected, even though GB scored better.
    metadata["selected"] = "MA4"

    metadata["ml_driven"] = False

    path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    result = audit_forecast_artifacts(
        "normal",
        output_dir=root,
    )

    assert result["valid"] is False

    assert result["errors"]


# ======================================================
# TEST 6 — Unknown scenario must be rejected
# ======================================================

def test_unknown_scenario_rejected(tmp_path):

    with pytest.raises(
        ValueError,
        match="Unknown scenario",
    ):
        audit_forecast_artifacts(
            "unknown",
            output_dir=tmp_path,
        )


# ======================================================
# TEST 7 — Negative forecasts must be rejected
# ======================================================

def test_negative_forecast_rejected(artifacts):

    root, directory = artifacts

    path = directory / "planning_forecast.csv"

    forecast = pd.read_csv(path)

    forecast.loc[0, "P50"] = -10

    forecast.to_csv(
        path,
        index=False,
    )

    result = audit_forecast_artifacts(
        "normal",
        output_dir=root,
    )

    assert result["valid"] is False

    assert result["errors"]


# ======================================================
# TEST 8 — Crossing quantiles must be rejected
# ======================================================

def test_crossing_quantiles_rejected(artifacts):

    root, directory = artifacts

    path = directory / "planning_forecast.csv"

    forecast = pd.read_csv(path)

    # P90 cannot be smaller than P80.
    forecast.loc[0, "P90"] = 10

    forecast.to_csv(
        path,
        index=False,
    )

    result = audit_forecast_artifacts(
        "normal",
        output_dir=root,
    )

    assert result["valid"] is False

    assert result["errors"]


# ======================================================
# TEST 9 — Missing forecast column rejected
# ======================================================

def test_missing_forecast_column_rejected(artifacts):

    root, directory = artifacts

    path = directory / "planning_forecast.csv"

    forecast = pd.read_csv(path)

    forecast = forecast.drop(columns=["P90"])

    forecast.to_csv(
        path,
        index=False,
    )

    result = audit_forecast_artifacts(
        "normal",
        output_dir=root,
    )

    assert result["valid"] is False

    assert result["errors"]


# ======================================================
# TEST 10 — Wrong random seed rejected
# ======================================================

def test_wrong_seed_rejected(artifacts):

    root, directory = artifacts

    path = directory / "selection.json"

    metadata = json.loads(
        path.read_text(encoding="utf-8")
    )

    metadata["seed"] = 999

    path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    result = audit_forecast_artifacts(
        "normal",
        output_dir=root,
    )

    assert result["valid"] is False

    assert any(
        "seed" in error.lower()
        for error in result["errors"]
    )