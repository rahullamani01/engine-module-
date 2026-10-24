"""Byte Works pipeline orchestration.

Current scope:
    Synthetic data generation
    Data validation
    Forecasting
    Model selection
    Forecast artifact export

Optimization, simulation and frontend JSON export
are separate integration stages.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import config
from src.data_generator import generate_dataset, save_dataset
from src.data_validation import validate_or_raise
from src.evaluation import select_and_forecast, save_results
from src.schemas import FORECAST_COLUMNS, validate_forecast_frame


@dataclass(frozen=True)
class PipelineResult:
    scenario: str
    selected_method: str
    ml_driven: bool
    forecast_rows: int
    planning_week: int
    output_directory: Path


def load_forecast(
    scenario: str,
    output_dir: Path = config.OUTPUT_DIR,
) -> pd.DataFrame:
    """Load and validate a previously generated planning forecast."""

    if scenario not in config.SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    path = Path(output_dir) / scenario / "planning_forecast.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Planning forecast does not exist: {path}"
        )

    forecast = pd.read_csv(path)

    if list(forecast.columns) != list(FORECAST_COLUMNS):
        raise ValueError("Forecast columns do not match the contract.")

    errors = validate_forecast_frame(forecast)

    if errors:
        raise ValueError(f"Invalid forecast: {errors}")

    if len(forecast) != config.N_SERIES:
        raise ValueError(
            f"Expected {config.N_SERIES} forecasts, "
            f"received {len(forecast)}"
        )

    if not forecast["planning_week"].eq(config.PLANNING_WEEK).all():
        raise ValueError("Incorrect planning week.")

    if forecast.duplicated(["store_id", "sku_id"]).any():
        raise ValueError("Duplicate store-SKU forecast.")

    return forecast


def load_selection(
    scenario: str,
    output_dir: Path = config.OUTPUT_DIR,
) -> dict:
    """Load previously saved model-selection metadata."""

    if scenario not in config.SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    path = Path(output_dir) / scenario / "selection.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"Model selection does not exist: {path}"
        )

    return json.loads(path.read_text())


def run_pipeline(
    scenario: str = "normal",
    data_dir: Path = config.DATA_DIR,
    output_dir: Path = config.OUTPUT_DIR,
) -> PipelineResult:
    """Run the validated forecasting pipeline for one scenario."""

    if scenario not in config.SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    # 1. Generate deterministic synthetic data.
    dataset = generate_dataset(scenario)

    # 2. Reject invalid data before performing any output writes.
    validate_or_raise(dataset)

    # 3. Compare forecasters and generate the week-53 forecast.
    selection = select_and_forecast(dataset.weekly)

    # 4. Validate the selected forecast before saving artifacts.
    forecast = selection.planning_forecast

    errors = validate_forecast_frame(forecast)

    if errors:
        raise ValueError(f"Invalid planning forecast: {errors}")

    if list(forecast.columns) != list(FORECAST_COLUMNS):
        raise ValueError("Unexpected planning forecast columns.")

    if len(forecast) != config.N_SERIES:
        raise ValueError(
            f"Expected {config.N_SERIES} rows, got {len(forecast)}."
        )

    if not forecast["planning_week"].eq(config.PLANNING_WEEK).all():
        raise ValueError("Planning week must be 53.")

    if forecast.duplicated(["store_id", "sku_id"]).any():
        raise ValueError("Duplicate store-SKU forecasts.")

    # 5. Save validated synthetic input data.
    save_dataset(dataset, Path(data_dir) / scenario)

    # 6. Save forecasting and evaluation artifacts.
    destination = Path(output_dir) / scenario
    save_results(selection, destination)

    # 7. Return metadata for the caller.
    return PipelineResult(
        scenario=scenario,
        selected_method=selection.selected,
        ml_driven=selection.ml_driven,
        forecast_rows=len(forecast),
        planning_week=config.PLANNING_WEEK,
        output_directory=destination,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Byte Works forecasting pipeline"
    )

    parser.add_argument(
        "scenario",
        nargs="?",
        default="normal",
        choices=config.SCENARIOS,
    )

    args = parser.parse_args()

    result = run_pipeline(args.scenario)

    print("\nBYTE WORKS — PIPELINE RESULT")
    print("----------------------------")
    print(f"Scenario: {result.scenario}")
    print(f"Selected method: {result.selected_method}")
    print(f"ML-driven: {result.ml_driven}")
    print(f"Planning week: {result.planning_week}")
    print(f"Forecast rows: {result.forecast_rows}")
    print(f"Output directory: {result.output_directory}")
    print("Status: Forecast artifacts generated")
    print("Optimization: Not integrated")


if __name__ == "__main__":
    main()