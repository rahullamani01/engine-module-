"""Stream B supply inputs; validation never repairs or replaces caller data.

Rows in issues are CSV line numbers (header is line 1). Synthetic assumptions
are fixed in SUPPLY_ASSUMPTIONS, independently of forecasts and hidden truth.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.data_validation import ValidationReport, validate_locations, validate_products


BATCH_COLUMNS = ("batch_id", "warehouse_id", "sku_id", "available_units", "expiry_date")
ROUTE_COLUMNS = ("warehouse_id", "store_id", "unit_cost", "capacity_units", "lead_time_days", "is_active")
STOCK_COLUMNS = ("store_id", "sku_id", "opening_units")
SUPPLY_ASSUMPTIONS = {
    "version": 1,
    "seed": config.SEED,
    "batches_per_warehouse_sku": 2,
    "units_per_batch": {"normal": 160, "scarce_supply": 35, "expiry_trap": 160},
    "route_capacity": {"normal": 1000, "scarce_supply": 120, "expiry_trap": 1000},
    "costs": "seeded integer 1..6; expiry_trap W01 costs 1",
    "traps": "W01-S01 inactive; W02-S02 lead 3; W01-K01-B1 expires day 2",
    "expiry_trap": "all W01 batches expire day 2",
    "expiry": "other batches use product shelf life plus batch offset (0 or 1 day)",
    "blocked_counts": "first failure: route (including absent), lead_time, shelf_life",
}


def planning_date() -> dt.date:
    return config.HISTORY_START_DATE + dt.timedelta(weeks=config.PLANNING_WEEK - 1)


@dataclass(frozen=True)
class Supply:
    batches: pd.DataFrame
    routes: pd.DataFrame
    opening_stock: pd.DataFrame


def generate_supply(locations, products, opening_stock, scenario="normal", seed=config.SEED) -> Supply:
    """Generate only supply; accepted store stock is copied, never regenerated."""
    if scenario not in config.SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    rng = np.random.default_rng(seed)
    stores = sorted(locations.loc[locations.type == "store", "location_id"])
    warehouses = sorted(locations.loc[locations.type == "warehouse", "location_id"])
    batches, routes = [], []
    for w in warehouses:
        for p in products.sort_values("sku_id").itertuples(index=False):
            for b in range(2):
                days = int(p.shelf_life_days) + b
                if (w == "W01" and p.sku_id == "K01" and b == 0) or (scenario == "expiry_trap" and w == "W01"):
                    days = 2
                batches.append([f"{w}-{p.sku_id}-B{b + 1}", w, p.sku_id,
                                SUPPLY_ASSUMPTIONS["units_per_batch"][scenario],
                                (planning_date() + dt.timedelta(days=days)).isoformat()])
        for s in stores:
            cost = int(rng.integers(1, 7))
            if scenario == "expiry_trap" and w == "W01":
                cost = 1
            lead = 3 if (w, s) == ("W02", "S02") else int(rng.integers(0, 3))
            routes.append([w, s, cost, SUPPLY_ASSUMPTIONS["route_capacity"][scenario],
                           lead, (w, s) != ("W01", "S01")])
    return Supply(pd.DataFrame(batches, columns=BATCH_COLUMNS),
                  pd.DataFrame(routes, columns=ROUTE_COLUMNS), opening_stock.copy(deep=True))


def validate_supply(supply: Supply, locations, products) -> ValidationReport:
    report = ValidationReport()
    stores, warehouses = validate_locations(locations, report)
    skus = validate_products(products, report)
    refs = {"store_id": stores, "warehouse_id": warehouses, "sku_id": skus}
    tables = (("inventory_batches.csv", supply.batches, BATCH_COLUMNS, ["batch_id"]),
              ("routes.csv", supply.routes, ROUTE_COLUMNS, ["warehouse_id", "store_id"]),
              ("opening_stock.csv", supply.opening_stock, STOCK_COLUMNS, ["store_id", "sku_id"]))
    for file, frame, columns, keys in tables:
        missing = set(columns) - set(frame.columns)
        for col in sorted(missing):
            report.add(file, None, col, "required: missing column")
        if missing:
            continue
        for pos, (_, row) in enumerate(frame.iterrows(), start=2):
            for col in columns:
                v = row[col]
                if pd.isna(v) or (isinstance(v, str) and not v.strip()):
                    report.add(file, pos, col, "required: missing value")
                    continue
                if col in refs and v not in refs[col]:
                    report.add(file, pos, col, "reference: unknown ID")
                if col == "batch_id" and (not isinstance(v, str) or v != v.strip()):
                    report.add(file, pos, col, "identifier: expected nonempty unpadded text")
                if col in ("available_units", "capacity_units", "opening_units", "lead_time_days", "unit_cost"):
                    try:
                        n = float(v)
                        valid = not isinstance(v, (bool, np.bool_)) and np.isfinite(n) and n >= 0
                        valid = valid and (col == "unit_cost" or n.is_integer())
                    except (TypeError, ValueError, OverflowError):
                        valid = False
                    if not valid:
                        report.add(file, pos, col, "quantity: expected finite nonnegative number" +
                                   ("" if col == "unit_cost" else " in whole units"))
                if col == "is_active" and not (isinstance(v, (bool, np.bool_)) or
                                                isinstance(v, (int, np.integer)) and v in (0, 1)):
                    report.add(file, pos, col, "active_flag: expected boolean or integer 0/1")
                if col == "expiry_date":
                    try:
                        if not isinstance(v, str) or dt.date.fromisoformat(v).isoformat() != v:
                            raise ValueError()
                    except (ValueError, TypeError):
                        report.add(file, pos, col, "date: expected YYYY-MM-DD")
            if frame.duplicated(keys, keep=False).iloc[pos - 2]:
                report.add(file, pos, ",".join(keys), "unique: duplicate key")
        if file == "opening_stock.csv":
            seen = set(frame[["store_id", "sku_id"]].itertuples(index=False, name=None))
            if seen != {(s, k) for s in stores for k in skus}:
                report.add(file, None, "store_id,sku_id", "complete_pairs: expected every store-SKU pair exactly once")
    return report


def eligible_options(supply: Supply, ship_date: dt.date | None = None):
    """Validated input only. Count every batch × store once, using first failure."""
    ship_date = ship_date or planning_date()
    routes = {(r.warehouse_id, r.store_id): r for r in supply.routes.itertuples(index=False)}
    stores = sorted(supply.opening_stock.store_id.unique())
    blocked = dict.fromkeys(("route", "lead_time", "shelf_life"), 0)
    options = []
    for b in supply.batches.sort_values("batch_id").itertuples(index=False):
        for s in stores:
            r = routes.get((b.warehouse_id, s))
            if r is None or not r.is_active:
                blocked["route"] += 1
                continue
            day = int(r.lead_time_days)
            if day > config.LATEST_ARRIVAL_DAY:
                blocked["lead_time"] += 1
                continue
            life = (dt.date.fromisoformat(b.expiry_date) - ship_date).days - day
            if life < config.MIN_SHELF_LIFE_ON_ARRIVAL_DAYS:
                blocked["shelf_life"] += 1
                continue
            options.append(dict(batch_id=b.batch_id, warehouse_id=b.warehouse_id, store_id=s,
                                sku_id=b.sku_id, unit_cost=float(r.unit_cost), arrival_day=day,
                                shelf_life_on_arrival=life, expiry_date=b.expiry_date))
    return options, blocked
