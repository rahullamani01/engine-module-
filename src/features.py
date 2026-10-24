"""Leakage-safe features for the gradient-boosting forecaster.

Features for target week t use ONLY weeks t-1 ... t-4:
    lag_1..lag_4, roll_mean / roll_std (over those same 4 weeks; std is sample, ddof=1),
    one-hot store, one-hot SKU.
The 4-week rolling window matches the PRD warm-up (weeks 1-4), so weeks 5+ are usable.
Categories are learned from training data only; an unseen store or SKU raises.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

LAGS = (1, 2, 3, 4)
NUMERIC_FEATURES = tuple(f"lag_{k}" for k in LAGS) + ("roll_mean", "roll_std")


class FeatureBuilder:
    def fit(self, store_ids, sku_ids) -> "FeatureBuilder":
        self.stores_ = sorted(set(store_ids))
        self.skus_ = sorted(set(sku_ids))
        self.feature_names_ = (
            list(NUMERIC_FEATURES)
            + [f"store_{s}" for s in self.stores_]
            + [f"sku_{k}" for k in self.skus_]
        )
        return self

    def transform(self, wide: pd.DataFrame, target_weeks, with_target: bool = False):
        """Build one row per (series, target week). Returns (X, keys, y_or_None)."""
        if not hasattr(self, "stores_"):
            raise RuntimeError("Call fit() first.")
        store = wide.index.get_level_values("store_id").to_numpy()
        sku = wide.index.get_level_values("sku_id").to_numpy()
        unseen = (set(store) - set(self.stores_)) | (set(sku) - set(self.skus_))
        if unseen:
            raise ValueError(f"Unseen store/SKU ids (not in training data): {sorted(unseen)}")

        frames, keys, targets = [], [], []
        for t in target_weeks:
            lag_cols = [t - k for k in LAGS]
            if min(lag_cols) < 1 or not set(lag_cols) <= set(wide.columns):
                raise ValueError(f"Week {t} needs weeks {sorted(lag_cols)} of history.")
            lags = np.column_stack([wide[c].to_numpy(float) for c in lag_cols])  # only weeks < t
            if np.isnan(lags).any():
                raise ValueError(f"Missing history in weeks {sorted(lag_cols)}; not fabricating features.")
            block = pd.DataFrame(lags, columns=[f"lag_{k}" for k in LAGS])
            block["roll_mean"] = lags.mean(axis=1)
            block["roll_std"] = lags.std(axis=1, ddof=1)
            for s in self.stores_:
                block[f"store_{s}"] = (store == s).astype(float)
            for k in self.skus_:
                block[f"sku_{k}"] = (sku == k).astype(float)
            frames.append(block)
            keys.append(pd.DataFrame({"store_id": store, "sku_id": sku, "week": t}))
            if with_target:
                y = wide[t].to_numpy(float)
                if np.isnan(y).any():
                    raise ValueError(f"Missing target demand in week {t}.")
                targets.append(y)
        X = pd.concat(frames, ignore_index=True)[self.feature_names_]
        return X, pd.concat(keys, ignore_index=True), (np.concatenate(targets) if with_target else None)