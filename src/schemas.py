"""Table schemas and the shared forecast-output contract.

The forecast frame produced by every forecasting method (MA4 now, gradient
boosting in G2) follows FORECAST_COLUMNS so downstream code never needs to
know which method generated the numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import QUANTILE_COLUMNS

# --- Input tables (PRD section 3.1) -----------------------------------------
LOCATION_COLUMNS = ("location_id", "type", "region")
PRODUCT_COLUMNS = ("sku_id", "shelf_life_days", "lost_sale_cost", "overstock_cost")
DEMAND_HISTORY_COLUMNS = ("week_start", "store_id", "sku_id", "units_sold")
# Hidden truth is for simulation/inspection only and is never a forecaster input.
# `dispersion` is the negative-binomial size k: Var = mu + mu^2 / k.
HIDDEN_TRUTH_COLUMNS = ("store_id", "sku_id", "week", "true_mean", "dispersion")
WEEKLY_COLUMNS = ("store_id", "sku_id", "week", "demand")  # internal forecasting view


def to_weekly(demand_history: pd.DataFrame) -> pd.DataFrame:
    """PRD demand_history (week_start, units_sold) -> internal (week, demand) view.

    Week 1 is the earliest week_start in the table; later weeks are 7-day steps.
    Run data validation first: this raises on unparseable or off-grid dates.
    """
    dates = pd.to_datetime(demand_history["week_start"], format="%Y-%m-%d")
    offset = (dates - dates.min()).dt.days
    if (offset % 7 != 0).any():
        raise ValueError("week_start values are not on a 7-day grid; run data validation first.")
    out = pd.DataFrame({
        "store_id": demand_history["store_id"].to_numpy(),
        "sku_id": demand_history["sku_id"].to_numpy(),
        "week": (offset // 7 + 1).to_numpy(),
        "demand": demand_history["units_sold"].to_numpy(),
    })
    return out.sort_values(["store_id", "sku_id", "week"]).reset_index(drop=True)


# --- Forecast output -------------------------------------------------------
FORECAST_KEY_COLUMNS = ("store_id", "sku_id", "planning_week", "method")
# Extra (beyond the PRD record) flag: True when SKU-pooled residuals were used.
FORECAST_FLAG_COLUMNS = ("used_pooled_residuals",)
FORECAST_COLUMNS = FORECAST_KEY_COLUMNS + QUANTILE_COLUMNS + FORECAST_FLAG_COLUMNS


def enforce_quantile_rules(
    frame: pd.DataFrame, quantile_columns: tuple[str, ...] = QUANTILE_COLUMNS
) -> tuple[pd.DataFrame, int]:
    """Make quantile columns nonnegative and non-decreasing across levels.

    Documented correction:
      1. clamp negatives to 0;
      2. if quantile crossing remains, sort each row's values ascending
         (monotone rearrangement), so P50 <= P60 <= ... <= P95.
    Non-finite values are rejected rather than silently filled.

    Returns (corrected_copy, number_of_rows_that_were_changed).
    """
    cols = list(quantile_columns)
    values = frame[cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Forecast quantiles contain NaN/inf; refusing to fabricate values.")
    fixed = np.sort(np.clip(values, 0.0, None), axis=1)
    n_changed = int((fixed != values).any(axis=1).sum())
    out = frame.copy()
    out[cols] = fixed
    return out, n_changed


def validate_forecast_frame(
    frame: pd.DataFrame, quantile_columns: tuple[str, ...] = QUANTILE_COLUMNS
) -> list[str]:
    """Return a list of contract violations (empty list means the frame is valid)."""
    problems: list[str] = []
    required = list(FORECAST_KEY_COLUMNS) + list(quantile_columns)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return [f"missing columns: {missing}"]
    values = frame[list(quantile_columns)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        problems.append("non-finite quantile values")
        return problems
    if (values < 0).any():
        problems.append("negative quantile values")
    if (np.diff(values, axis=1) < 0).any():
        problems.append("quantile crossing: values not non-decreasing across P50..P95")
    if frame.duplicated(["store_id", "sku_id", "planning_week", "method"]).any():
        problems.append("duplicate store/sku/planning_week/method records")
    return problems
