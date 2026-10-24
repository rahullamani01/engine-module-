"""Forecast-to-optimizer handoff.

Converts saved planning forecasts into inventory-adjusted
integer shipment targets for all six planning levels.

Opening inventory is supplied externally by Stream B.

This module does NOT:
- generate supply inventory
- calculate eligible warehouse routes
- allocate shipments
- solve the linear program
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.engine import load_forecast
from src.targets import build_targets


def prepare_handoff(
    scenario: str,
    opening_stock: pd.DataFrame,
    output_dir: Path = config.OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Build all six shipment-target tables without saving them."""

    forecast = load_forecast(
        scenario,
        output_dir=output_dir,
    )

    results = {}

    for level in config.QUANTILE_COLUMNS:
        targets = build_targets(
            forecast=forecast,
            opening_stock=opening_stock,
            level=level,
        )

        results[level] = targets

    return results


def export_handoff(
    scenario: str,
    opening_stock: pd.DataFrame,
    output_dir: Path = config.OUTPUT_DIR,
) -> dict[str, Path]:
    """Validate all planning levels before writing target CSVs."""

    results = prepare_handoff(
        scenario=scenario,
        opening_stock=opening_stock,
        output_dir=output_dir,
    )

    destination = Path(output_dir) / scenario

    destination.mkdir(parents=True, exist_ok=True)

    paths = {}

    for level, targets in results.items():

        path = destination / f"targets_{level}.csv"

        targets.to_csv(
            path,
            index=False,
        )

        paths[level] = path

    return paths


def main():

    parser = argparse.ArgumentParser(
        description="Byte Works optimizer handoff"
    )

    parser.add_argument(
        "scenario",
        choices=config.SCENARIOS,
    )

    parser.add_argument(
        "--stock",
        required=True,
        type=Path,
        help="Path to the opening-stock CSV",
    )

    args = parser.parse_args()

    if not args.stock.is_file():
        parser.error(
            f"Opening stock file not found: {args.stock}"
        )

    opening_stock = pd.read_csv(args.stock)

    paths = export_handoff(
        scenario=args.scenario,
        opening_stock=opening_stock,
    )

    print("\nBYTE WORKS — OPTIMIZER HANDOFF")
    print("--------------------------------")

    for level, path in paths.items():
        print(f"{level}: {path}")

    print("\nStatus: Shipment targets exported")
    print("Optimizer: Not executed")


if __name__ == "__main__":
    main()