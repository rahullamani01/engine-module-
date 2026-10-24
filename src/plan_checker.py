"""Independent feasibility checker. No solver or eligibility imports.

Checks shipment identities against original supply, and recomputes date rules,
shared capacities, target identities, display fields, costs and totals.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict

from src import config


def check_plan(plan, supply, targets, ship_date=None):
    ship_date = ship_date or config.HISTORY_START_DATE + dt.timedelta(weeks=config.PLANNING_WEEK - 1)
    violations = []

    def fail(rule, detail, row=None):
        violations.append(dict(rule=rule, message=detail, row=row))

    def number(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    def equal(value, expected):
        return number(value) and math.isclose(value, expected, rel_tol=0, abs_tol=1e-7)

    batches = {b.batch_id: b for b in supply.batches.itertuples(index=False)}
    routes = {(r.warehouse_id, r.store_id): r for r in supply.routes.itertuples(index=False)}
    batch_units, route_units, pair_units = defaultdict(float), defaultdict(float), defaultdict(float)
    freight = 0.
    shipments = plan.get("shipments")
    if not isinstance(shipments, list):
        fail("shipments", "Missing shipment list")
        shipments = []
    for i, s in enumerate(shipments):
        units = s.get("units")
        if not number(units) or units < 0 or int(units) != units:
            fail("whole_units", "Shipment must contain nonnegative integer units", i)
            continue
        batch = batches.get(s.get("batch_id"))
        route_key = (s.get("warehouse_id"), s.get("store_id"))
        pair = (s.get("store_id"), s.get("sku_id"))
        route = routes.get(route_key)
        if batch is None:
            fail("batch_reference", "Unknown batch", i)
            continue
        if (s.get("warehouse_id"), s.get("sku_id")) != (batch.warehouse_id, batch.sku_id):
            fail("batch_identity", "Warehouse/SKU does not match original batch", i)
        batch_units[batch.batch_id] += units
        route_units[route_key] += units
        pair_units[pair] += units
        if pair not in targets:
            fail("target_reference", "Unknown store-SKU", i)
        if route is None:
            fail("route", "Route missing", i)
            continue
        if not route.is_active:
            fail("route", "Route inactive", i)
        day = int(route.lead_time_days)
        life = (dt.date.fromisoformat(batch.expiry_date) - ship_date).days - day
        if day > config.LATEST_ARRIVAL_DAY:
            fail("lead_time", "Arrival exceeds cutoff", i)
        if life < config.MIN_SHELF_LIFE_ON_ARRIVAL_DAYS:
            fail("shelf_life", "Insufficient shelf life at arrival", i)
        cost = float(route.unit_cost)
        freight += units * cost
        for field, expected in (("unit_cost", cost), ("freight", units * cost), ("arrival_day", day),
                                ("shelf_life_on_arrival", life)):
            if not equal(s.get(field), expected):
                fail("shipment_fields", f"Incorrect {field}", i)
        if s.get("expiry_date") != batch.expiry_date:
            fail("shipment_fields", "Incorrect expiry_date", i)
    for key, units in batch_units.items():
        if units > float(batches[key].available_units) + 1e-7:
            fail("batch_stock", f"Oversold batch {key}")
    for key, units in route_units.items():
        if key in routes and units > float(routes[key].capacity_units) + 1e-7:
            fail("route_capacity", f"Exceeded shared capacity {key}")
    for key, units in pair_units.items():
        if key in targets and units > targets[key] + 1e-7:
            fail("over_target", f"Over-shipped {key}")
    unmet_rows = plan.get("unmet", [])
    seen = set()
    for i, row in enumerate(unmet_rows):
        key = (row.get("store_id"), row.get("sku_id"))
        if key in seen or key not in targets:
            fail("unmet_identity", "Duplicate or unknown target pair", i)
            continue
        seen.add(key)
        for field, expected in (("target_units", targets[key]), ("shipped_units", pair_units[key]),
                                ("unmet_units", targets[key] - pair_units[key])):
            if not equal(row.get(field), expected):
                fail("unmet_identity", f"Incorrect {field}", i)
    if seen != set(targets):
        fail("unmet_identity", "Unmet rows must cover every target pair")
    shortage = sum(targets.values()) - sum(pair_units.values())
    totals = dict(target_units=sum(targets.values()), shipped_units=sum(pair_units.values()),
                  unmet_units=shortage, freight=freight, shipments=len(shipments))
    for field, expected in totals.items():
        if not equal(plan.get("totals", {}).get(field), expected):
            fail("totals", f"Incorrect {field}")
    if not equal(plan.get("planned_objective"), freight + config.LOST_SALE_COST * shortage):
        fail("objective", "Incorrect freight plus unmet penalty")
    return dict(passed=not violations, violations=violations)
