"""Gradient-boosting quantile forecaster (the ML candidate).

One GradientBoostingRegressor(loss="quantile", alpha=q) per planning quantile.
Same fit/predict/metadata interface and output contract as the MA4 baseline.

Leakage guards:
  * fit() only sees weeks <= the end of the fit window; training rows are target
    weeks start..end, each built from the four weeks before it.
  * predict() filters history to weeks < t BEFORE building features for week t.
Hyperparameters are fixed a priori in config.GB_PARAMS (no tuning on the holdout).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src import config
from src.baseline import build_wide
from src.features import FeatureBuilder
from src.schemas import FORECAST_COLUMNS, enforce_quantile_rules

METHOD_NAME = "GB"


class GradientBoostingQuantileForecaster:
    method_name = METHOD_NAME

    def __init__(self, quantiles=config.QUANTILES, params: dict | None = None):
        self.quantiles = tuple(quantiles)
        self.params = dict(config.GB_PARAMS if params is None else params)
        self.params.setdefault("random_state", config.SEED)
        self.fit_weeks_: tuple[int, int] | None = None

    def fit(self, sales: pd.DataFrame, fit_weeks=config.TRAIN_WEEKS):
        start, end = fit_weeks
        if end < start:
            raise ValueError("fit_weeks end must be >= start")
        wide = build_wide(sales[sales["week"] <= end], max_week=end)  # nothing after `end` is visible
        builder = FeatureBuilder().fit(
            wide.index.get_level_values("store_id"), wide.index.get_level_values("sku_id")
        )
        X, _, y = builder.transform(wide, range(start, end + 1), with_target=True)

        self.models_ = {}
        for q in self.quantiles:
            model = GradientBoostingRegressor(loss="quantile", alpha=q, **self.params)
            model.fit(X, y)
            self.models_[q] = model

        self.builder_ = builder
        self.series_index_ = wide.index
        self.n_train_rows_ = int(len(X))
        self.fit_weeks_ = (start, end)
        return self

    def predict(self, sales: pd.DataFrame, weeks) -> pd.DataFrame:
        if self.fit_weeks_ is None:
            raise RuntimeError("Call fit() before predict().")
        frames = []
        for t in sorted(set(weeks)):
            history = sales[sales["week"] < t]  # leakage guard: week t and later are dropped
            wide = build_wide(history, max_week=t - 1).reindex(self.series_index_)
            X, _, _ = self.builder_.transform(wide, [t])  # raises if any series has missing history
            raw = np.column_stack([self.models_[q].predict(X) for q in self.quantiles])
            raw = np.maximum(raw, 0.0)
            frame = pd.DataFrame({
                "store_id": self.series_index_.get_level_values("store_id"),
                "sku_id": self.series_index_.get_level_values("sku_id"),
                "planning_week": t, "method": self.method_name,
                **{col: raw[:, j] for j, col in enumerate(config.QUANTILE_COLUMNS)},
                "used_pooled_residuals": False,
            })
            frames.append(enforce_quantile_rules(frame)[0])  # clamp >= 0, sort so quantiles never cross
        return pd.concat(frames, ignore_index=True)[list(FORECAST_COLUMNS)]

    def metadata(self) -> dict:
        return {
            "method": self.method_name, "fit_weeks": list(self.fit_weeks_),
            "quantiles": list(self.quantiles), "params": self.params,
            "n_series": int(len(self.series_index_)), "n_train_rows": self.n_train_rows_,
            "features": list(self.builder_.feature_names_),
            "n_series_using_pooled_residuals": 0,
        }
