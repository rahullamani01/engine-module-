"""Forecast evaluation metrics: pinball loss, quantile coverage, WAPE."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def pinball_loss(y_true, y_pred, tau: float) -> float:
    """Mean pinball (quantile) loss: mean(max(tau*(y-q), (tau-1)*(y-q)))."""
    if not 0 < tau < 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    y, q = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y.shape != q.shape or y.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same shape")
    if not (np.isfinite(y).all() and np.isfinite(q).all()):
        raise ValueError("non-finite values in pinball loss inputs")
    diff = y - q
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def observed_coverage(y_true, y_pred) -> float:
    """Share of actuals at or below the predicted quantile."""
    y, q = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(y <= q))


def wape(y_true, y_pred) -> float:
    """sum|y - yhat| / sum(y); NaN where undefined (total actual demand is zero)."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    total = y.sum()
    return float("nan") if total == 0 else float(np.abs(y - p).sum() / total)


def merge_with_actuals(forecasts: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Join forecasts to realised demand on (store, sku, week); every forecast needs an actual."""
    f = forecasts.rename(columns={"planning_week": "week"})
    merged = f.merge(sales[["store_id", "sku_id", "week", "demand"]],
                     on=["store_id", "sku_id", "week"], how="left", validate="one_to_one")
    if merged["demand"].isna().any():
        raise ValueError(f"{int(merged['demand'].isna().sum())} forecast rows have no realised demand")
    return merged


def evaluate_quantile_forecasts(forecasts: pd.DataFrame, sales: pd.DataFrame,
                                quantiles=config.QUANTILES, columns=config.QUANTILE_COLUMNS):
    """Return (per_quantile_table, summary_dict). WAPE uses the P50 forecast as the point forecast."""
    merged = merge_with_actuals(forecasts, sales)
    y = merged["demand"].to_numpy(float)
    rows = []
    for tau, col in zip(quantiles, columns):
        rows.append({
            "quantile": tau, "column": col,
            "pinball_loss": pinball_loss(y, merged[col], tau),
            "observed_coverage": observed_coverage(y, merged[col]),
            "nominal_coverage": tau,
        })
    table = pd.DataFrame(rows)
    table["coverage_gap"] = table["observed_coverage"] - table["nominal_coverage"]
    # PRD 6.1: flag levels whose coverage is more than 10 points from nominal.
    table["miscalibrated"] = table["coverage_gap"].abs() > config.CALIBRATION_TOLERANCE + 1e-12
    summary = {
        "n_observations": int(len(merged)),
        "mean_pinball_loss": float(table["pinball_loss"].mean()),
        "wape_p50": wape(y, merged[columns[0]]),
    }
    return table, summary
