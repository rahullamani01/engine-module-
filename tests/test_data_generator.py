import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_generator import SCENARIO_SPECS, generate_dataset, get_scenario, load_dataset, save_dataset
from src.schemas import DEMAND_HISTORY_COLUMNS, to_weekly

_cache = {}


def ds(name="normal"):
    if name not in _cache:
        _cache[name] = generate_dataset(name)
    return _cache[name]


def test_config_windows_and_scenarios_match_prd():
    assert config.WARMUP_WEEKS[1] + 1 == config.TRAIN_WEEKS[0]
    assert config.TRAIN_WEEKS[1] + 1 == config.EVAL_WEEKS[0]
    assert config.EVAL_WEEKS[1] == config.N_HISTORY_WEEKS
    assert config.PLANNING_WEEK == config.N_HISTORY_WEEKS + 1
    assert config.N_SERIES == 40 and config.N_EVAL_OBSERVATIONS == 480
    assert config.QUANTILE_COLUMNS == ("P50", "P60", "P70", "P80", "P90", "P95")
    assert config.SCENARIOS == ("normal", "scarce_supply", "expiry_trap")
    assert tuple(SCENARIO_SPECS) == config.SCENARIOS


def test_network_dimensions():
    d = ds()
    loc = d.locations
    assert (loc["type"] == "warehouse").sum() == 3 and (loc["type"] == "store").sum() == 8
    assert loc["region"].nunique() == 2 and len(d.products) == 5
    assert len(d.demand_history) == 8 * 5 * 52
    assert list(d.demand_history.columns) == list(DEMAND_HISTORY_COLUMNS)  # PRD demand_history.csv
    assert d.demand_history["week_start"].nunique() == 52
    assert sorted(d.weekly["week"].unique()) == list(range(1, 53))  # no planning-week demand
    assert len(d.hidden_truth) == 8 * 5 * 53


def test_weekly_view_maps_dates_to_week_numbers():
    w = ds().weekly
    first = ds().demand_history["week_start"].min()
    assert (w.loc[w["week"] == 1].shape[0]) == 40
    assert ds().demand_history.loc[ds().demand_history["week_start"] == first, "units_sold"].sum() == \
        w.loc[w["week"] == 1, "demand"].sum()
    bad = ds().demand_history.copy()
    bad.loc[0, "week_start"] = "2025-01-08"
    with pytest.raises(ValueError):
        to_weekly(bad)


def test_reproducible_with_seed_42():
    a, b = generate_dataset("normal"), generate_dataset("normal")
    pd.testing.assert_frame_equal(a.demand_history, b.demand_history)
    pd.testing.assert_frame_equal(a.hidden_truth, b.hidden_truth)


def test_different_seed_changes_data():
    assert not generate_dataset("normal", seed=1).demand_history["units_sold"].equals(
        ds().demand_history["units_sold"])


def test_demand_is_nonnegative_integer():
    s = ds().demand_history["units_sold"]
    assert (s >= 0).all() and np.issubdtype(s.dtype, np.integer)


def test_referential_integrity():
    d = ds()
    stores, skus = set(d.store_ids), set(d.products["sku_id"])
    for table in (d.demand_history, d.hidden_truth):
        assert set(table["store_id"]) <= stores and set(table["sku_id"]) <= skus
    assert set(d.locations["location_id"]).isdisjoint(skus)


def test_scenarios_share_demand_history_because_they_differ_in_supply():
    for name in ("scarce_supply", "expiry_trap"):
        pd.testing.assert_frame_equal(ds("normal").demand_history, ds(name).demand_history)
        assert ds(name).scenario == name


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        get_scenario("demand_spike")
    with pytest.raises(ValueError):
        generate_dataset("nope")


def test_save_and_load_roundtrip_with_prd_config_json(tmp_path):
    save_dataset(ds(), tmp_path / "x")
    back = load_dataset(tmp_path / "x")
    assert back.scenario == "normal" and back.seed == 42
    for name in ("locations", "products", "demand_history", "hidden_truth"):
        pd.testing.assert_frame_equal(getattr(back, name), getattr(ds(), name))
    cfg = json.loads((tmp_path / "x" / "config.json").read_text())
    for key in ("planning_week", "quantiles", "latest_arrival_day", "min_shelf_life_days", "seed"):
        assert key in cfg
    assert cfg["planning_week"] == 53 and cfg["latest_arrival_day"] == 2 and cfg["min_shelf_life_days"] == 3
