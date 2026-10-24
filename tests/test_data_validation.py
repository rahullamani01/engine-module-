import dataclasses

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_generator import generate_dataset
from src.data_validation import DataValidationError, validate_dataset, validate_or_raise


def fresh(**overrides):
    d = generate_dataset("normal")
    copies = {n: getattr(d, n).copy() for n in ("locations", "products", "demand_history", "hidden_truth")}
    return dataclasses.replace(d, **copies | overrides)


def issues_for(d, table=None, column=None):
    return [i for i in validate_dataset(d).issues
            if (table is None or i.table == table) and (column is None or i.column == column)]


def test_all_prd_scenarios_validate_clean():
    for name in config.SCENARIOS:
        assert validate_dataset(generate_dataset(name)).ok, name


def test_missing_required_column():
    d = fresh()
    d = fresh(demand_history=d.demand_history.drop(columns=["units_sold"]))
    found = issues_for(d, "demand_history", "units_sold")
    assert found and "missing" in found[0].reason


def test_missing_value_reports_row_and_column():
    d = fresh()
    d.demand_history.loc[7, "units_sold"] = np.nan
    found = issues_for(d, "demand_history", "units_sold")
    assert [i.row for i in found] == [7] and "missing" in found[0].reason


def test_negative_demand_rejected():
    d = fresh()
    d.demand_history.loc[3, "units_sold"] = -5
    assert any(i.row == 3 and "below minimum" in i.reason for i in issues_for(d, "demand_history", "units_sold"))


def test_infinite_and_non_numeric_demand_rejected():
    d = fresh()
    d.demand_history["units_sold"] = d.demand_history["units_sold"].astype(float)
    d.demand_history.loc[1, "units_sold"] = np.inf
    assert any(i.row == 1 and "infinite" in i.reason for i in issues_for(d, "demand_history", "units_sold"))
    d2 = fresh()
    d2.demand_history["units_sold"] = d2.demand_history["units_sold"].astype(object)
    d2.demand_history.loc[2, "units_sold"] = "abc"
    assert any(i.row == 2 and "non-numeric" in i.reason for i in issues_for(d2, "demand_history", "units_sold"))


def test_duplicate_store_sku_week_rejected():
    d = fresh()
    d = fresh(demand_history=pd.concat([d.demand_history, d.demand_history.iloc[[0]]], ignore_index=True))
    assert any("duplicate store-SKU-week" in i.reason for i in issues_for(d, "demand_history"))


def test_duplicate_ids_rejected_in_locations_and_products():
    d = fresh()
    d.locations.loc[1, "location_id"] = d.locations.loc[0, "location_id"]
    d.products.loc[1, "sku_id"] = d.products.loc[0, "sku_id"]
    assert len([i for i in issues_for(d, "locations", "location_id") if "duplicate" in i.reason]) == 2
    assert len([i for i in issues_for(d, "products", "sku_id") if "duplicate" in i.reason]) == 2


def test_bad_location_type_and_product_values_rejected():
    d = fresh()
    d.locations.loc[0, "type"] = "depot"
    d.products.loc[0, "shelf_life_days"] = 0
    d.products.loc[1, "lost_sale_cost"] = -1
    assert any(i.row == 0 and "type must be" in i.reason for i in issues_for(d, "locations", "type"))
    assert any(i.row == 0 for i in issues_for(d, "products", "shelf_life_days"))
    assert any(i.row == 1 for i in issues_for(d, "products", "lost_sale_cost"))


def test_invalid_date_rejected():
    d = fresh()
    d.demand_history.loc[4, "week_start"] = "2025-13-45"
    assert any(i.row == 4 and "invalid date" in i.reason for i in issues_for(d, "demand_history", "week_start"))


def test_off_grid_date_rejected():
    d = fresh()
    d.demand_history.loc[5, "week_start"] = "2025-03-12"  # a Wednesday, not on the weekly grid
    assert any(i.row == 5 and "weekly grid" in i.reason for i in issues_for(d, "demand_history", "week_start"))


def test_week_beyond_history_rejected():
    d = fresh()
    d.demand_history.loc[6, "week_start"] = "2026-01-05"  # week 53 of the grid
    assert any(i.row == 6 and "beyond history" in i.reason for i in issues_for(d, "demand_history", "week_start"))


def test_orphan_references_rejected():
    d = fresh()
    d.demand_history.loc[0, "store_id"] = "S99"
    d.demand_history.loc[1, "sku_id"] = "K99"
    assert any(i.row == 0 and "orphan" in i.reason for i in issues_for(d, "demand_history", "store_id"))
    assert any(i.row == 1 and "orphan" in i.reason for i in issues_for(d, "demand_history", "sku_id"))


def test_warehouse_id_is_not_a_valid_store_reference():
    d = fresh()
    d.demand_history.loc[0, "store_id"] = "W01"
    assert any(i.row == 0 and "orphan" in i.reason for i in issues_for(d, "demand_history", "store_id"))


def test_missing_weeks_reported_per_series():
    d = fresh()
    h = d.demand_history
    gap_date = h["week_start"].unique()[9]
    h = h[~((h.store_id == "S01") & (h.sku_id == "K01") & (h.week_start == gap_date))]
    assert any("S01/K01" in i.reason and "missing 1 week" in i.reason
               for i in issues_for(fresh(demand_history=h), "demand_history"))


def test_validate_or_raise_rejects_and_does_not_mutate_input():
    d = fresh()
    d.demand_history.loc[0, "units_sold"] = -1
    before = d.demand_history.copy()
    with pytest.raises(DataValidationError) as exc:
        validate_or_raise(d)
    assert "demand_history" in str(exc.value) and not exc.value.report.ok
    pd.testing.assert_frame_equal(d.demand_history, before)  # reported, never silently repaired
