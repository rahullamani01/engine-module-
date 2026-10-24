"""Convert demand forecasts into inventory-adjusted shipment targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.schemas import validate_forecast_frame


STOCK_COLUMNS = [
    "store_id",
    "sku_id",
    "opening_units",
]


def build_targets(
    forecast: pd.DataFrame,
    opening_stock: pd.DataFrame,
    level: str = "P90",
) -> pd.DataFrame:
    """
    Calculate shipment targets for a selected demand quantile.

    target_units = max(0, ceil(forecast_quantile) - opening_units)

    Opening stock must contain exactly one nonnegative integer
    inventory value per store-SKU forecast.
    """

    # 1. Validate selected quantile.
    if level not in config.QUANTILE_COLUMNS:
        raise ValueError(
            f"Invalid planning level: {level}. "
            f"Allowed: {config.QUANTILE_COLUMNS}"
        )

    # 2. Validate forecast.
    errors = validate_forecast_frame(forecast)

    if errors:
        raise ValueError(f"Invalid forecast: {errors}")

    if forecast.duplicated(["store_id", "sku_id"]).any():
        raise ValueError("Duplicate store-SKU forecast rows.")

    if len(forecast) != config.N_SERIES:
        raise ValueError(
            f"Expected {config.N_SERIES} forecasts, "
            f"received {len(forecast)}."
        )

    if not forecast["planning_week"].eq(
        config.PLANNING_WEEK
    ).all():
        raise ValueError("Incorrect planning week.")

    # 3. Validate opening stock columns.
    missing = set(STOCK_COLUMNS) - set(opening_stock.columns)

    if missing:
        raise ValueError(
            f"Opening stock missing columns: {sorted(missing)}"
        )

    stock = opening_stock[STOCK_COLUMNS].copy()

    # 4. Reject duplicate stock records.
    if stock.duplicated(["store_id", "sku_id"]).any():
        raise ValueError("Duplicate opening stock records.")

    if stock[["store_id", "sku_id"]].isna().any().any():
        raise ValueError("Missing store or SKU identifier.")

    if stock["opening_units"].isna().any():
        raise ValueError("Missing opening inventory.")

    # 5. Validate inventory values.
    units = pd.to_numeric(
        stock["opening_units"],
        errors="coerce",
    )

    if not np.isfinite(units.to_numpy(dtype=float)).all():
        raise ValueError("Opening stock must contain finite numbers.")

    if (units < 0).any():
        raise ValueError("Opening stock cannot be negative.")

    if (units % 1 != 0).any():
        raise ValueError("Opening stock must contain whole units.")

    stock["opening_units"] = units.astype("int64")

    # 6. Match forecasts with inventory.
    result = forecast[
        ["store_id", "sku_id", "planning_week", "method", level]
    ].merge(
        stock,
        on=["store_id", "sku_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if (result["_merge"] != "both").any():
        raise ValueError(
            "Opening inventory missing for one or more forecasts."
        )

    if len(stock) != len(result):
        raise ValueError(
            "Opening inventory contains unexpected store-SKU records."
        )

    result = result.drop(columns=["_merge"])

    # 7. Calculate nonnegative integer shipment targets.
    result["target_units"] = np.maximum(
        0,
        np.ceil(result[level].to_numpy())
        - result["opening_units"].to_numpy(),
    ).astype("int64")

    result["planning_level"] = level

    return result[
        [
            "store_id",
            "sku_id",
            "planning_week",
            "method",
            "planning_level",
            level,
            "opening_units",
            "target_units",
        ]
    ]