"""Convert validated forecasting results into frontend JSON.

Requires opening inventory from Stream B.
Does not generate inventory or fabricate optimization results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config
from src.engine import load_forecast
from src.targets import build_targets


def build_forecast_json(
    forecast: pd.DataFrame,
    opening_stock: pd.DataFrame,
) -> dict:
    """Build the frontend forecast contract from computed results."""

    levels = list(config.QUANTILE_COLUMNS)

    # Validate every level before constructing JSON.
    targets = {
        level: build_targets(forecast, opening_stock, level)
        for level in levels
    }

    # Associate inventory and targets by identifiers, not row position.
    base = forecast.set_index(["store_id", "sku_id"]).sort_index()

    first = targets[levels[0]].set_index(
        ["store_id", "sku_id"]
    ).sort_index()

    rows = []

    for (store_id, sku_id), forecast_row in base.iterrows():

        key = (store_id, sku_id)

        row = {
            "store_id": str(store_id),
            "sku_id": str(sku_id),
            "opening_units": int(first.loc[key, "opening_units"]),
        }

        for level in levels:
            row[level.lower()] = float(forecast_row[level])

            target_frame = targets[level].set_index(
                ["store_id", "sku_id"]
            )

            row[f"target_{level.lower()}"] = int(
                target_frame.loc[key, "target_units"]
            )

        rows.append(row)

    return {"rows": rows}


def export_forecast_json(
    scenario: str,
    opening_stock: pd.DataFrame,
    output_dir: Path = config.OUTPUT_DIR,
    results_dir: Path = Path("results"),
) -> Path:
    """Export real forecast results without replacing unrelated files."""

    forecast = load_forecast(scenario, output_dir)

    payload = build_forecast_json(
        forecast,
        opening_stock,
    )

    destination = Path(results_dir) / scenario / "forecast.json"

    # Refuse to overwrite existing frontend/mock artifacts.
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents accidental replacement.
    with destination.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)

    return destination