import numpy as np
import pandas as pd
import pytest

from src.baseline import build_wide
from src.features import FeatureBuilder


def linear_sales(weeks=12):
    rows = [("S01", "K01", t, 10 * t) for t in range(1, weeks + 1)]
    rows += [("S02", "K01", t, 5 * t) for t in range(1, weeks + 1)]
    return pd.DataFrame(rows, columns=["store_id", "sku_id", "week", "demand"])


def builder_for(wide):
    return FeatureBuilder().fit(wide.index.get_level_values("store_id"), wide.index.get_level_values("sku_id"))


def test_lag_and_rolling_features_hand_calculation():
    wide = build_wide(linear_sales())
    X, keys, y = builder_for(wide).transform(wide, [7], with_target=True)
    row = X[(keys["store_id"] == "S01").to_numpy()].iloc[0]
    assert [row["lag_1"], row["lag_2"], row["lag_3"], row["lag_4"]] == [60, 50, 40, 30]
    assert row["roll_mean"] == pytest.approx(45.0)
    assert row["roll_std"] == pytest.approx(np.std([30, 40, 50, 60], ddof=1))
    assert y[(keys["store_id"] == "S01").to_numpy()][0] == 70  # target is week 7 itself, kept out of X
    assert row["store_S01"] == 1 and row["store_S02"] == 0 and row["sku_K01"] == 1


def test_features_use_only_past_weeks():
    base = linear_sales()
    changed = base.copy()
    changed.loc[changed["week"] >= 7, "demand"] = 99999  # target week 7 and later
    wa, wb = build_wide(base), build_wide(changed)
    b = builder_for(wa)
    Xa, _, _ = b.transform(wa, [7])
    Xb, _, _ = b.transform(wb, [7])
    pd.testing.assert_frame_equal(Xa, Xb)  # features for week 7 ignore week 7+
    changed2 = base.copy()
    changed2.loc[changed2["week"] == 6, "demand"] = 1  # a PAST week must matter
    Xc, _, _ = b.transform(build_wide(changed2), [7])
    assert not Xa.equals(Xc)


def test_needs_four_weeks_of_history_and_rejects_gaps():
    wide = build_wide(linear_sales())
    b = builder_for(wide)
    with pytest.raises(ValueError):
        b.transform(wide, [4])  # would need week 0
    wide.iloc[0, 5] = np.nan  # week 6 missing for S01
    with pytest.raises(ValueError):
        b.transform(wide, [8])


def test_unseen_categories_rejected_and_column_order_stable():
    wide = build_wide(linear_sales())
    b = builder_for(wide)
    other = linear_sales().assign(store_id=lambda d: d["store_id"].replace({"S02": "S99"}))
    with pytest.raises(ValueError):
        b.transform(build_wide(other), [7])
    X, _, _ = b.transform(wide, [7])
    assert list(X.columns) == b.feature_names_