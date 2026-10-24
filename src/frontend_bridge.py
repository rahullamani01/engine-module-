"""Byte Works — Backend-to-Frontend Integration Bridge.

Connects Stream A's forecasting engine to Stream C's Streamlit app.

Responsibilities:
    1. Translate frontend scenario aliases to canonical backend names.
    2. Load validated forecasting artifacts.
    3. Combine forecasts with Stream B's opening inventory.
    4. Generate frontend-compatible forecast JSON.
    5. Export to engine_results/<canonical_scenario>/forecast.json.

This module does NOT:
    - Generate opening inventory.
    - Run warehouse optimization.
    - Run Monte Carlo simulations.
    - Generate fake plans or evaluation metrics.
    - Modify the frontend's demo results.
    - Activate the full-planner integration gate.

All forecast results use synthetic data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import config
from src.engine import load_forecast
from src.forecast_adapter import build_forecast_json


# ============================================================
# SCENARIO MAPPING
# ============================================================

# Frontend aliases -> canonical backend scenario identifiers.
SCENARIO_MAP = {
    "normal": "normal",
    "scarce": "scarce_supply",
    "expiry": "expiry_trap",
}

# Canonical backend identifiers -> frontend aliases.
REVERSE_SCENARIO_MAP = {
    value: key
    for key, value in SCENARIO_MAP.items()
}


# ============================================================
# SCENARIO NORMALIZATION
# ============================================================

def normalize_scenario(scenario: str) -> str:
    """Convert a frontend alias into a canonical backend scenario.

    Examples:
        normal        -> normal
        scarce        -> scarce_supply
        expiry        -> expiry_trap
        scarce_supply -> scarce_supply
        expiry_trap   -> expiry_trap
    """

    if scenario in SCENARIO_MAP:
        return SCENARIO_MAP[scenario]

    if scenario in config.SCENARIOS:
        return scenario

    raise ValueError(
        f"Unknown scenario: {scenario}. "
        f"Supported scenarios: {list(config.SCENARIOS)}"
    )


def frontend_scenario(scenario: str) -> str:
    """Return the short frontend alias for a scenario.

    This function is retained for compatibility with existing tests.

    Note:
        Export directories always use CANONICAL scenario names.
    """

    backend_scenario = normalize_scenario(scenario)

    return REVERSE_SCENARIO_MAP[backend_scenario]


# ============================================================
# PREPARE FRONTEND FORECAST
# ============================================================

def prepare_frontend_forecast(
    scenario: str,
    opening_stock: pd.DataFrame,
    output_dir: Path = config.OUTPUT_DIR,
) -> dict:
    """Build frontend-compatible forecast data.

    Reads actual saved forecasting artifacts.

    Requires valid opening stock from Stream B.

    Returns:
        {
            "rows": [
                {
                    "store_id": "...",
                    "sku_id": "...",
                    "opening_units": ...,
                    "p50": ...,
                    "p60": ...,
                    "p70": ...,
                    "p80": ...,
                    "p90": ...,
                    "p95": ...,
                    "target_p50": ...,
                    ...
                    "target_p95": ...
                }
            ]
        }
    """

    backend_scenario = normalize_scenario(scenario)

    # Load the actual computed forecast.
    forecast = load_forecast(
        backend_scenario,
        output_dir=output_dir,
    )

    # Reuse the existing validated adapter.
    # This also calculates inventory-adjusted shipment targets.
    payload = build_forecast_json(
        forecast=forecast,
        opening_stock=opening_stock,
    )

    return payload


# ============================================================
# EXPORT FRONTEND FORECAST
# ============================================================

def export_frontend_forecast(
    scenario: str,
    opening_stock: pd.DataFrame,
    output_dir: Path = config.OUTPUT_DIR,
    results_dir: Path = Path("engine_results"),
) -> Path:
    """Export a computed forecast into Stream C's engine directory.

    Destination:
        engine_results/<canonical_scenario>/forecast.json

    Examples:
        normal -> engine_results/normal/forecast.json

        scarce -> engine_results/scarce_supply/forecast.json

        expiry -> engine_results/expiry_trap/forecast.json

    Existing files are never overwritten automatically.
    """

    backend_scenario = normalize_scenario(scenario)

    # Prepare and validate everything before writing files.
    payload = prepare_frontend_forecast(
        scenario=backend_scenario,
        opening_stock=opening_stock,
        output_dir=output_dir,
    )

    # Always use the canonical scenario name for output.
    destination = (
        Path(results_dir)
        / backend_scenario
        / "forecast.json"
    )

    # Protect the demo directory.
    if (
        "results" in destination.parts
        and "engine_results" not in destination.parts
    ):
        raise ValueError(
            "Refusing to write into the demo results directory."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Exclusive creation:
    # Raises FileExistsError if forecast.json already exists.
    # This prevents accidental replacement of existing artifacts.
    with destination.open(
        "x",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            allow_nan=False,
        )

    return destination


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def main() -> None:
    """Run the frontend forecast exporter from the terminal."""

    parser = argparse.ArgumentParser(
        description=(
            "Byte Works — Export computed forecasts "
            "for the Streamlit frontend"
        )
    )

    # --------------------------------------------------------
    # Scenario argument
    # --------------------------------------------------------

    supported_scenarios = tuple(
        dict.fromkeys(
            list(SCENARIO_MAP)
            + list(config.SCENARIOS)
        )
    )

    parser.add_argument(
        "scenario",
        choices=supported_scenarios,
        help="Scenario to export",
    )

    # --------------------------------------------------------
    # Opening inventory argument
    # --------------------------------------------------------

    parser.add_argument(
        "--stock",
        required=True,
        type=Path,
        help="Path to validated opening-stock CSV from Stream B",
    )

    # --------------------------------------------------------
    # Output directory argument
    # --------------------------------------------------------

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("engine_results"),
        help=(
            "Destination for computed engine artifacts "
            "(default: engine_results)"
        ),
    )

    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate stock file
    # --------------------------------------------------------

    if not args.stock.is_file():
        parser.error(
            f"Opening stock file not found: {args.stock}"
        )

    try:
        opening_stock = pd.read_csv(args.stock)

    except (OSError, pd.errors.ParserError) as exc:
        parser.error(
            f"Unable to read opening stock CSV: {exc}"
        )

    # --------------------------------------------------------
    # Generate frontend-compatible JSON
    # --------------------------------------------------------

    destination = export_frontend_forecast(
        scenario=args.scenario,
        opening_stock=opening_stock,
        results_dir=args.results_dir,
    )

    # --------------------------------------------------------
    # Report actual execution result
    # --------------------------------------------------------

    print()
    print("BYTE WORKS — FRONTEND INTEGRATION")
    print("========================================")
    print(
        f"Backend scenario: "
        f"{normalize_scenario(args.scenario)}"
    )
    print(
        f"Frontend alias: "
        f"{frontend_scenario(args.scenario)}"
    )
    print(f"Forecast JSON: {destination}")
    print()
    print("Status: Forecast JSON exported successfully")
    print("Data source: Synthetic forecasting engine")
    print("Optimization: Not included")
    print("Risk evaluation: Not included")
    print("Full planner activation: Disabled")
    print("Demo results: Unchanged")


# ============================================================
# MODULE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()