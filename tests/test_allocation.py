"""Stream B tests use original supply and intentionally corrupt checked plans."""
import datetime as dt

import pandas as pd
import pytest

from src.allocation import allocate, assemble_plan, target_map
from src.data_generator import generate_dataset
from src.plan_checker import check_plan
from src.supply import (BATCH_COLUMNS, ROUTE_COLUMNS, Supply, eligible_options,
                        generate_supply, planning_date, validate_supply)


@pytest.fixture
def tiny():
    date = (planning_date() + dt.timedelta(days=10)).isoformat()
    batches = pd.DataFrame([["B1", "W01", "K01", 60, date],
                            ["B2", "W02", "K01", 80, date]], columns=BATCH_COLUMNS)
    routes = pd.DataFrame([[w, s, c, 100, 1, True] for w, c in [("W01", 2), ("W02", 5)]
                           for s in ("S01", "S02")], columns=ROUTE_COLUMNS)
    stock = pd.DataFrame([[s, "K01", 0] for s in ("S01", "S02")],
                         columns=["store_id", "sku_id", "opening_units"])
    return Supply(batches, routes, stock), {("S01", "K01"): 100, ("S02", "K01"): 40}


def test_hand_solved_two_by_two(tiny):
    supply, targets = tiny
    plan = allocate(supply, targets)
    assert plan["solver"]["status"] == "optimal"
    assert plan["solver"]["continuous_objective"] == pytest.approx(520)
    assert plan["planned_objective"] == pytest.approx(520)
    assert plan["totals"]["shipped_units"] == 140
    assert check_plan(plan, supply, targets)["passed"]


@pytest.mark.parametrize("scenario", ["normal", "scarce_supply", "expiry_trap"])
def test_generated_supply_and_baselines(scenario):
    dataset = generate_dataset(scenario)
    stock = pd.DataFrame([[s, k, 7] for s in dataset.store_ids for k in dataset.products.sku_id],
                         columns=["store_id", "sku_id", "opening_units"])
    supply = generate_supply(dataset.locations, dataset.products, stock, scenario)
    again = generate_supply(dataset.locations, dataset.products, stock, scenario)
    pd.testing.assert_frame_equal(supply.batches, again.batches)
    pd.testing.assert_frame_equal(supply.routes, again.routes)
    pd.testing.assert_frame_equal(supply.opening_stock, stock)
    assert validate_supply(supply, dataset.locations, dataset.products).ok
    options, blocked = eligible_options(supply)
    assert all(v > 0 for v in blocked.values())
    assert len(options) + sum(blocked.values()) == len(supply.batches) * 8
    targets = {(s, k): 100 for s, k in stock[["store_id", "sku_id"]].itertuples(index=False, name=None)}
    lp = allocate(supply, targets)
    for method in ("lp", "pro_rata", "greedy"):
        plan = allocate(supply, targets, method)
        assert check_plan(plan, supply, targets)["passed"]
        assert lp["solver"]["continuous_objective"] <= plan["solver"]["continuous_objective"] + 1e-7
    if scenario == "scarce_supply":
        assert lp["totals"]["unmet_units"] > 0


@pytest.mark.parametrize("field,value", [("available_units", -1), ("available_units", 0.5),
                                          ("available_units", float("inf")), ("available_units", True),
                                          ("expiry_date", "not-date"), ("expiry_date", "2026-1-1"),
                                          ("warehouse_id", "missing"), ("sku_id", "missing"),
                                          ("batch_id", "")])
def test_invalid_batch_row(field, value):
    dataset = generate_dataset()
    stock = pd.DataFrame([[s, k, 0] for s in dataset.store_ids for k in dataset.products.sku_id],
                         columns=["store_id", "sku_id", "opening_units"])
    supply = generate_supply(dataset.locations, dataset.products, stock)
    supply.batches[field] = supply.batches[field].astype(object)
    supply.batches.loc[0, field] = value
    report = validate_supply(supply, dataset.locations, dataset.products)
    assert not report.ok
    assert any(i.table == "inventory_batches.csv" and i.row == 2 and i.column == field for i in report.issues)


@pytest.mark.parametrize("table,field,value", [("routes", "capacity_units", -1),
                                               ("routes", "unit_cost", float("nan")),
                                               ("routes", "lead_time_days", 1.5),
                                               ("routes", "is_active", "false"),
                                               ("routes", "store_id", "orphan"),
                                               ("opening_stock", "opening_units", 0.5)])
def test_invalid_other_row(table, field, value):
    dataset = generate_dataset()
    stock = pd.DataFrame([[s, k, 0] for s in dataset.store_ids for k in dataset.products.sku_id],
                         columns=["store_id", "sku_id", "opening_units"])
    supply = generate_supply(dataset.locations, dataset.products, stock)
    frame = getattr(supply, table)
    frame[field] = frame[field].astype(object)
    frame.loc[0, field] = value
    assert not validate_supply(supply, dataset.locations, dataset.products).ok


def test_duplicates_missing_columns_and_pairs(tiny):
    dataset = generate_dataset()
    stock = pd.DataFrame([[s, k, 0] for s in dataset.store_ids for k in dataset.products.sku_id],
                         columns=["store_id", "sku_id", "opening_units"])
    supply = generate_supply(dataset.locations, dataset.products, stock)
    supply.batches.loc[1, "batch_id"] = supply.batches.loc[0, "batch_id"]
    supply.routes.loc[1] = supply.routes.loc[0]
    supply.opening_stock.loc[1] = supply.opening_stock.loc[0]
    report = validate_supply(supply, dataset.locations, dataset.products)
    assert sum("duplicate" in i.reason for i in report.issues) == 6
    assert any("complete_pairs" in i.reason for i in report.issues)
    supply.routes.drop(columns="unit_cost", inplace=True)
    assert any(i.column == "unit_cost" for i in validate_supply(supply, dataset.locations, dataset.products).issues)


def test_eligibility_boundaries_and_precedence(tiny):
    supply, _ = tiny
    supply.routes["lead_time_days"] = 2
    supply.batches["expiry_date"] = (planning_date() + dt.timedelta(days=5)).isoformat()
    assert len(eligible_options(supply)[0]) == 4
    supply.routes.loc[0, "is_active"] = False
    supply.routes.loc[0, "lead_time_days"] = 3
    supply.routes.loc[1, "lead_time_days"] = 3
    supply.batches.loc[1, "expiry_date"] = (planning_date() + dt.timedelta(days=4)).isoformat()
    assert eligible_options(supply)[1] == {"route": 1, "lead_time": 1, "shelf_life": 2}
    supply.routes.drop(index=0, inplace=True)
    assert eligible_options(supply)[1]["route"] == 1


def test_shared_route_capacity_across_skus(tiny):
    supply, targets = tiny
    supply.batches.loc[2] = ["B3", "W01", "K02", 100, supply.batches.loc[0, "expiry_date"]]
    supply.opening_stock.loc[2] = ["S01", "K02", 0]
    supply.opening_stock.loc[3] = ["S02", "K02", 0]
    targets[("S01", "K02")] = 100
    targets[("S02", "K02")] = 0
    supply.routes.loc[0, "capacity_units"] = 50
    for method in ("lp", "greedy", "pro_rata"):
        plan = allocate(supply, targets, method)
        assert check_plan(plan, supply, targets)["passed"]
        assert sum(s["units"] for s in plan["shipments"] if (s["warehouse_id"], s["store_id"]) == ("W01", "S01")) <= 50


def test_flooring_recomputes_unmet(tiny):
    supply, targets = tiny
    options, blocked = eligible_options(supply)
    plan = assemble_plan(options, [2.9, 3.1, 0., 0.], targets, blocked, {})
    assert [s["units"] for s in plan["shipments"]] == [2, 3]
    assert plan["totals"]["unmet_units"] == 135
    assert plan["totals"]["freight"] == 10
    assert check_plan(plan, supply, targets)["passed"]


@pytest.mark.parametrize("violation", ["negative", "fractional", "nan", "batch", "identity", "inactive", "late",
                                       "short_life", "oversold", "capacity", "target", "unmet", "cost", "totals", "objective"])
def test_checker_rejects_injected_violations(tiny, violation):
    supply, targets = tiny
    plan = allocate(supply, targets)
    shipment = plan["shipments"][0]
    bi = supply.batches.index[supply.batches.batch_id == shipment["batch_id"]][0]
    ri = supply.routes.index[(supply.routes.warehouse_id == shipment["warehouse_id"]) &
                             (supply.routes.store_id == shipment["store_id"])][0]
    if violation == "negative": shipment["units"] = -1
    elif violation == "fractional": shipment["units"] = 0.5
    elif violation == "nan": shipment["units"] = float("nan")
    elif violation == "batch": shipment["batch_id"] = "unknown"
    elif violation == "identity": shipment["sku_id"] = "K02"
    elif violation == "inactive": supply.routes.loc[ri, "is_active"] = False
    elif violation == "late": supply.routes.loc[ri, "lead_time_days"] = 3
    elif violation == "short_life": supply.batches.loc[bi, "expiry_date"] = planning_date().isoformat()
    elif violation == "oversold": supply.batches.loc[bi, "available_units"] = 0
    elif violation == "capacity": supply.routes.loc[ri, "capacity_units"] = 0
    elif violation == "target": targets[(shipment["store_id"], shipment["sku_id"])] = 0
    elif violation == "unmet": plan["unmet"][0]["unmet_units"] += 1
    elif violation == "cost": shipment["unit_cost"] = 0
    elif violation == "totals": plan["totals"]["freight"] += 1
    elif violation == "objective": plan["planned_objective"] += 1
    assert not check_plan(plan, supply, targets)["passed"]


def test_no_eligible_options_and_zero_targets(tiny):
    supply, targets = tiny
    supply.routes["is_active"] = False
    plan = allocate(supply, targets)
    assert plan["totals"]["unmet_units"] == 140
    assert check_plan(plan, supply, targets)["passed"]
    targets = dict.fromkeys(targets, 0)
    plan = allocate(supply, targets)
    assert plan["planned_objective"] == 0
    assert check_plan(plan, supply, targets)["passed"]


def test_solver_failure_does_not_create_plan(tiny, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected solver failure")
    monkeypatch.setattr("src.allocation.linprog", fail)
    supply, targets = tiny
    result = allocate(supply, targets)
    assert result["solver"]["status"] == "error"
    assert "shipments" not in result
    assert not check_plan(result, supply, targets)["passed"]


def test_targets_all_levels():
    for level in ("P50", "P60", "P70", "P80", "P90", "P95"):
        assert target_map([dict(store_id="S01", sku_id="K01", **{f"target_{level.lower()}": 4})], level) == {("S01", "K01"): 4}
    with pytest.raises(ValueError):
        target_map([dict(store_id="S01", sku_id="K01", target_p50=0.5)], "P50")
