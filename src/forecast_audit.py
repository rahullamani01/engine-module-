"""Independent checks for saved Byte Works forecasting artifacts.

This module reads existing artifacts only.

It does not:
- retrain forecasting models
- generate synthetic datasets
- run optimization
- modify existing artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.schemas import FORECAST_COLUMNS, validate_forecast_frame


REQUIRED_ARTIFACTS = (
    "planning_forecast.csv",
    "holdout_forecasts.csv",
    "model_comparison.csv",
    "calibration.csv",
    "selection.json",
)


def audit_forecast_artifacts(
    scenario: str,
    output_dir: Path = config.OUTPUT_DIR,
) -> dict:
    """Verify that saved forecast artifacts satisfy the handoff contract."""

    if scenario not in config.SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    directory = Path(output_dir) / scenario
    errors = []

    # 1. Check all required artifacts.
    for filename in REQUIRED_ARTIFACTS:
        if not (directory / filename).is_file():
            errors.append(f"Missing artifact: {filename}")

    if errors:
        return {
            "scenario": scenario,
            "valid": False,
            "errors": errors,
        }

    # 2. Read saved artifacts.
    try:
        forecast = pd.read_csv(directory / "planning_forecast.csv")
        comparison = pd.read_csv(directory / "model_comparison.csv")
        metadata = json.loads(
            (directory / "selection.json").read_text()
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "scenario": scenario,
            "valid": False,
            "errors": [f"Unable to read artifacts: {exc}"],
        }

    # 3. Validate forecast structure.
    if list(forecast.columns) != list(FORECAST_COLUMNS):
        errors.append("Forecast columns do not match the contract.")

    if len(forecast) != config.N_SERIES:
        errors.append(
            f"Expected {config.N_SERIES} forecasts, "
            f"found {len(forecast)}."
        )

    if {"store_id", "sku_id"}.issubset(forecast.columns):
        if forecast.duplicated(["store_id", "sku_id"]).any():
            errors.append("Duplicate store-SKU forecasts.")

    if "planning_week" in forecast.columns:
        if not forecast["planning_week"].eq(
            config.PLANNING_WEEK
        ).all():
            errors.append("Incorrect planning week.")

    # 4. Run the existing forecast schema validator.
    try:
        errors.extend(
            str(error)
            for error in validate_forecast_frame(forecast)
        )
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"Forecast validation failed: {exc}")

    # 5. Validate saved model-selection information.
    selected = metadata.get("selected")

    if selected not in ("MA4", "GB"):
        errors.append("Invalid selected forecasting method.")

    if metadata.get("synthetic_data") is not True:
        errors.append("Synthetic-data marker missing or incorrect.")

    if metadata.get("seed") != config.SEED:
        errors.append("Incorrect random seed.")

    if metadata.get("ml_driven") != (selected == "GB"):
        errors.append("ML-driven flag contradicts selected method.")

    if "method" in forecast.columns:
        if not forecast["method"].eq(selected).all():
            errors.append(
                "Forecast method disagrees with selection.json."
            )

    # 6. Verify model comparison.
    required_comparison_columns = {
        "method",
        "mean_pinball_loss",
        "selected",
        "n_observations",
    }

    if not required_comparison_columns.issubset(comparison.columns):
        errors.append("Model comparison columns are incomplete.")
    else:
        if comparison["method"].duplicated().any():
            errors.append("Duplicate methods in comparison.")

        if len(comparison) not in (1, 2):
            errors.append("Unexpected number of compared methods.")

        selected_rows = comparison[
            comparison["selected"].astype(str).str.lower() == "true"
        ]

        if (
            len(selected_rows) != 1
            or selected_rows.iloc[0]["method"] != selected
        ):
            errors.append("Model-selection metadata mismatch.")

        if not comparison["n_observations"].eq(
            config.N_EVAL_OBSERVATIONS
        ).all():
            errors.append("Incorrect holdout observation count.")

        losses = pd.to_numeric(
            comparison["mean_pinball_loss"],
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(losses).all() or (losses < 0).any():
            errors.append("Invalid mean pinball loss.")

        if (
            len(comparison) == 2
            and set(comparison["method"]) == {"MA4", "GB"}
            and np.isfinite(losses).all()
        ):
            scores = dict(
                zip(comparison["method"], losses)
            )

            expected_method = (
                "GB"
                if scores["GB"] < scores["MA4"]
                else "MA4"
            )

            if selected != expected_method:
                errors.append(
                    "Selected method contradicts pinball-loss rule."
                )

    return {
        "scenario": scenario,
        "valid": len(errors) == 0,
        "selected_method": selected,
        "forecast_rows": len(forecast),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit saved Byte Works forecasts"
    )

    parser.add_argument(
        "scenario",
        choices=config.SCENARIOS,
    )

    args = parser.parse_args()

    result = audit_forecast_artifacts(args.scenario)

    print("\nBYTE WORKS — FORECAST AUDIT")
    print("--------------------------------")
    print(f"Scenario: {result['scenario']}")
    print(f"Valid: {result['valid']}")

    for error in result["errors"]:
        print(f"ERROR: {error}")

    if not result["valid"]:
        raise SystemExit(1)

    print(f"Selected method: {result['selected_method']}")
    print(f"Forecast rows: {result['forecast_rows']}")
    print("Status: Basic forecast contract checks passed")


if __name__ == "__main__":
    main()