"""Byte Works — Genuine forecasting integration reports.

Generates:
    engine_results/<scenario>/meta.json
    engine_results/<scenario>/validation.json

Uses the existing synthetic dataset, model-selection results,
computed forecasts, and validated opening inventory.

Does not create solver results, fake metrics, or activation flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import config
from src.data_generator import load_dataset
from src.data_validation import validate_or_raise
from src.engine import load_forecast, load_selection
from src.forecast_adapter import build_forecast_json


SCENARIO_NAMES = {
    "normal": "Normal",
    "scarce_supply": "Scarce Supply",
    "expiry_trap": "Expiry Trap",
}


def save_json(path: Path, payload: dict) -> None:
    """Write a new JSON file without overwriting existing artifacts."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            allow_nan=False,
        )


def generate_reports(
    scenario: str,
    stock_path: Path,
    results_dir: Path = Path("engine_results"),
) -> tuple[Path, Path]:

    if scenario not in config.SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    stock_path = Path(stock_path)

    if not stock_path.is_file():
        raise FileNotFoundError(
            f"Opening inventory not found: {stock_path}"
        )

    # Load actual saved synthetic dataset.
    dataset = load_dataset(
        config.DATA_DIR / scenario
    )

    # Run existing data-validation rules.
    validate_or_raise(dataset)

    # Load actual week-53 forecast and model selection.
    forecast = load_forecast(scenario)
    selection = load_selection(scenario)

    # Read and validate Stream B's opening inventory.
    opening_stock = pd.read_csv(stock_path)

    # The existing adapter checks inventory completeness,
    # quantity constraints, IDs, and shipment targets.
    forecast_payload = build_forecast_json(
        forecast=forecast,
        opening_stock=opening_stock,
    )

    if len(forecast_payload["rows"]) != config.N_SERIES:
        raise ValueError("Expected exactly 40 forecast rows.")

    # Read actual saved holdout model scores.
    comparison_path = (
        config.OUTPUT_DIR / scenario / "model_comparison.csv"
    )

    comparison = pd.read_csv(comparison_path)

    required_score_columns = {
        "method",
        "mean_pinball_loss",
    }

    if not required_score_columns.issubset(comparison.columns):
        raise ValueError(
            "Model comparison file has missing columns."
        )

    if comparison["method"].duplicated().any():
        raise ValueError("Duplicate model comparison entries.")

    scores = {
        str(row["method"]): float(row["mean_pinball_loss"])
        for _, row in comparison.iterrows()
    }

    if not scores:
        raise ValueError("Model comparison contains no scores.")

    if not all(
        pd.notna(value) and value >= 0
        for value in scores.values()
    ):
        raise ValueError("Invalid model comparison scores.")

    selected_method = str(selection["selected"])

    if selected_method not in scores:
        raise ValueError(
            "Selected model is absent from model comparison."
        )

    # ------------------------------
    # Genuine forecasting metadata
    # ------------------------------

    meta = {
        "scenario_id": scenario,
        "scenario_name": SCENARIO_NAMES[scenario],
        "planning_week": config.PLANNING_WEEK,
        "seed": int(config.SEED),
        "synthetic_data": True,
        "forecaster_used": selected_method,
        "forecaster_scores": scores,
        "levels": list(config.QUANTILE_COLUMNS),
        "default_level": "P50",
        "network": {
            "warehouses": config.N_WAREHOUSES,
            "stores": config.N_STORES,
            "skus": config.N_SKUS,
        },
        "latest_arrival_day": config.LATEST_ARRIVAL_DAY,
        "min_shelf_life_days": (
            config.MIN_SHELF_LIFE_ON_ARRIVAL_DAYS
        ),
    }

    # Solver status and solve time are intentionally omitted.
    # Stream B must provide them after actual optimization.

    # ------------------------------
    # Genuine validation report
    # ------------------------------

    source_tables = {
        "locations.csv": dataset.locations,
        "products.csv": dataset.products,
        "demand_history.csv": dataset.demand_history,
        "hidden_truth.csv": dataset.hidden_truth,
        "opening_stock.csv": opening_stock,
        "planning_forecast.csv": forecast,
    }

    files = [
        {
            "name": name,
            "rows": int(len(frame)),
            "status": "ok",
            "errors": 0,
            "warnings": 0,
        }
        for name, frame in source_tables.items()
    ]

    checks = [
        {
            "id": "synthetic_data_valid",
            "label": "Synthetic demand dataset validates",
            "passed": True,
            "detail": "Existing dataset validation completed successfully.",
        },
        {
            "id": "forecast_contract",
            "label": "Planning forecast contract validates",
            "passed": True,
            "detail": (
                f"{len(forecast)} valid store-SKU forecasts "
                f"for week {config.PLANNING_WEEK}."
            ),
        },
        {
            "id": "opening_inventory",
            "label": "Opening inventory validates",
            "passed": True,
            "detail": (
                f"{len(opening_stock)} validated store-SKU "
                "opening inventory records."
            ),
        },
        {
            "id": "model_selection",
            "label": "Selected model has holdout scores",
            "passed": True,
            "detail": (
                f"Selected {selected_method} using saved "
                "holdout model comparison."
            ),
        },
    ]

    validation = {
        "ok": True,
        "files": files,
        "checks": checks,
        "errors": [],
        "warnings": [],
    }

    destination = Path(results_dir) / scenario

    meta_path = destination / "meta.json"
    validation_path = destination / "validation.json"

    # Refuse to overwrite either existing report.
    if meta_path.exists() or validation_path.exists():
        raise FileExistsError(
            "Existing integration reports detected. "
            "Review them before regenerating."
        )

    save_json(meta_path, meta)
    save_json(validation_path, validation)

    return meta_path, validation_path


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Generate genuine Byte Works integration reports."
    )

    parser.add_argument(
        "scenario",
        choices=config.SCENARIOS,
    )

    parser.add_argument(
        "--stock",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    meta_path, validation_path = generate_reports(
        scenario=args.scenario,
        stock_path=args.stock,
    )

    print("\nBYTE WORKS — INTEGRATION REPORTS")
    print("--------------------------------")
    print(f"Scenario: {args.scenario}")
    print(f"Metadata: {meta_path}")
    print(f"Validation: {validation_path}")
    print("Input validation: PASSED")
    print("Forecasting metadata: GENERATED")
    print("Optimization: PENDING")
    print("Risk Dial: PENDING")
    print("Full planner activation: DISABLED")


if __name__ == "__main__":
    main()