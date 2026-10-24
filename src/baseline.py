"""MA4 point forecast + empirical residual quantiles (the forecasting baseline).

For planning week t:  MA4(t) = mean(demand[t-1], ..., demand[t-4]).
Residual(t) = actual(t) - MA4(t), each MA4 built only from weeks before t.
Quantile forecast  = max(0, MA4(t) + empirical residual quantile), then sorted
so quantiles are non-decreasing. Series with < MIN_SERIES_RESIDUALS residuals
use SKU-pooled residuals and are flagged `used_pooled_residuals=True`.

Leakage guard: `predict` filters history to weeks < t BEFORE computing anything,
and `fit` filters to weeks <= the end of the fit window.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.metrics import evaluate_quantile_forecasts
from src.schemas import FORECAST_COLUMNS, enforce_quantile_rules

METHOD_NAME = "MA4"


def build_wide(sales: pd.DataFrame, max_week: int | None = None) -> pd.DataFrame:
    """Pivot long sales to a (store, sku) x week matrix (weeks 1..max_week, NaN if absent)."""
    if sales.duplicated(["store_id", "sku_id", "week"]).any():
        raise ValueError("Duplicate store-SKU-week rows; run data validation first.")
    if max_week is None:
        max_week = int(sales["week"].max())
    wide = sales.pivot(index=["store_id", "sku_id"], columns="week", values="demand")
    return wide.reindex(columns=range(1, max_week + 1)).sort_index().astype(float)


def _ma4_nan_aware(wide: pd.DataFrame, week: int, window: int) -> np.ndarray:
    first = week - window
    if first < 1:
        raise ValueError(f"Week {week} needs {window} prior weeks of history.")
    cols = list(range(first, week))
    if not set(cols) <= set(wide.columns):
        raise ValueError(f"History for weeks {cols} is not available.")
    return wide.loc[:, cols].to_numpy(float).mean(axis=1)  # NaN propagates if any week missing


def ma4_point_forecast(wide: pd.DataFrame, week: int, window: int = config.MA_WINDOW) -> np.ndarray:
    """MA4 for `week` using ONLY weeks week-4..week-1. Raises if any needed value is missing."""
    point = _ma4_nan_aware(wide, week, window)
    if np.isnan(point).any():
        raise ValueError(f"Incomplete history for MA{window} at week {week}; not fabricating a forecast.")
    return point


def historical_residuals(wide: pd.DataFrame, target_weeks, window: int = config.MA_WINDOW) -> np.ndarray:
    """(n_series, n_target_weeks) of actual - MA4, each MA4 built from past weeks only."""
    cols = []
    for t in target_weeks:
        cols.append(wide[t].to_numpy(float) - _ma4_nan_aware(wide, t, window))
    return np.column_stack(cols)


class MA4ResidualQuantileForecaster:
    method_name = METHOD_NAME

    def __init__(self, quantiles=config.QUANTILES, window: int = config.MA_WINDOW,
                 min_residuals: int = config.MIN_SERIES_RESIDUALS):
        self.quantiles = tuple(quantiles)
        self.window = window
        self.min_residuals = min_residuals
        self.fit_weeks_: tuple[int, int] | None = None

    def fit(self, sales: pd.DataFrame, fit_weeks=config.TRAIN_WEEKS):
        start, end = fit_weeks
        if end < start:
            raise ValueError("fit_weeks end must be >= start")
        wide = build_wide(sales[sales["week"] <= end], max_week=end)  # nothing after `end` is visible
        resid = historical_residuals(wide, range(start, end + 1), self.window)
        finite = np.isfinite(resid)
        n_res = finite.sum(axis=1)

        skus = wide.index.get_level_values("sku_id").to_numpy()
        pooled = {}
        for sku in np.unique(skus):
            vals = resid[skus == sku][finite[skus == sku]]
            if vals.size == 0:
                raise ValueError(f"No usable residuals for SKU {sku}.")
            pooled[sku] = np.quantile(vals, self.quantiles, method="linear")

        rq = np.empty((len(wide), len(self.quantiles)))
        use_pooled = n_res < self.min_residuals
        for i in range(len(wide)):
            rq[i] = pooled[skus[i]] if use_pooled[i] else np.quantile(resid[i][finite[i]], self.quantiles, method="linear")

        self.series_index_ = wide.index
        self.residual_quantiles_ = rq
        self.used_pooled_ = use_pooled
        self.n_residuals_ = n_res
        self.fit_weeks_ = (start, end)
        return self

    def predict(self, sales: pd.DataFrame, weeks) -> pd.DataFrame:
        if self.fit_weeks_ is None:
            raise RuntimeError("Call fit() before predict().")
        frames = []
        for t in sorted(set(weeks)):
            history = sales[sales["week"] < t]  # leakage guard: week t and later are dropped
            wide = build_wide(history, max_week=t - 1).reindex(self.series_index_)
            point = ma4_point_forecast(wide, t, self.window)
            raw = np.maximum(point[:, None] + self.residual_quantiles_, 0.0)
            frame = pd.DataFrame({
                "store_id": self.series_index_.get_level_values("store_id"),
                "sku_id": self.series_index_.get_level_values("sku_id"),
                "planning_week": t, "method": self.method_name,
                **{col: raw[:, j] for j, col in enumerate(config.QUANTILE_COLUMNS)},
                "used_pooled_residuals": self.used_pooled_,
            })
            frames.append(enforce_quantile_rules(frame)[0])
        return pd.concat(frames, ignore_index=True)[list(FORECAST_COLUMNS)]

    def metadata(self) -> dict:
        return {
            "method": self.method_name, "fit_weeks": list(self.fit_weeks_), "window": self.window,
            "quantiles": list(self.quantiles), "min_series_residuals": self.min_residuals,
            "n_series": int(len(self.series_index_)),
            "n_series_using_pooled_residuals": int(self.used_pooled_.sum()),
            "residuals_per_series_min": int(self.n_residuals_.min()),
            "residuals_per_series_max": int(self.n_residuals_.max()),
            "residual_quantile_method": "numpy linear interpolation",
        }


@dataclass
class HoldoutResult:
    forecasts: pd.DataFrame
    per_quantile: pd.DataFrame
    summary: dict
    metadata: dict


def evaluate_holdout(sales, fit_weeks=config.TRAIN_WEEKS, eval_weeks=config.EVAL_WEEKS) -> HoldoutResult:
    """Fit on fit_weeks, forecast each eval week from its own past, score on actuals."""
    model = MA4ResidualQuantileForecaster().fit(sales, fit_weeks)
    forecasts = model.predict(sales, config.weeks_in(eval_weeks))
    per_q, summary = evaluate_quantile_forecasts(forecasts, sales)
    meta = model.metadata() | {"eval_weeks": list(eval_weeks)}
    return HoldoutResult(forecasts, per_q, summary, meta)


def forecast_planning_week(sales, fit_weeks=config.FINAL_FIT_WEEKS, planning_week=config.PLANNING_WEEK):
    """Refit on fit_weeks (after holdout evaluation) and forecast the planning week."""
    model = MA4ResidualQuantileForecaster().fit(sales, fit_weeks)
    return model.predict(sales, [planning_week]), model


def run(scenario: str = "normal", data_dir: Path = config.DATA_DIR, out_dir: Path = config.OUTPUT_DIR) -> None:
    """Generate -> validate -> holdout evaluation -> refit -> planning-week forecast -> CSV/JSON."""
    from src.data_generator import generate_dataset, save_dataset
    from src.data_validation import validate_or_raise

    ds = generate_dataset(scenario)
    validate_or_raise(ds)
    save_dataset(ds, Path(data_dir) / scenario)
    weekly = ds.weekly
    res = evaluate_holdout(weekly)
    plan, model = forecast_planning_week(weekly)
    out = Path(out_dir) / scenario
    out.mkdir(parents=True, exist_ok=True)
    res.forecasts.to_csv(out / "ma4_holdout_forecasts.csv", index=False)
    res.per_quantile.to_csv(out / "ma4_evaluation.csv", index=False)
    plan.to_csv(out / "ma4_planning_week_forecast.csv", index=False)
    (out / "ma4_metadata.json").write_text(json.dumps(
        {"holdout": res.metadata, "holdout_summary": res.summary, "final_fit": model.metadata(),
         "seed": config.SEED, "synthetic_data": True}, indent=2))
    print(f"[{scenario}] n_obs={res.summary['n_observations']} "
          f"mean_pinball={res.summary['mean_pinball_loss']:.4f} wape_p50={res.summary['wape_p50']:.4f}")
    print(res.per_quantile.round(4).to_string(index=False))


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "normal")
