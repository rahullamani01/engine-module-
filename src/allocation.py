"""Eligible continuous HiGHS allocation and feasible allocation baselines.

This module does not certify its own output: callers must run plan_checker.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from src import config
from src.supply import Supply, eligible_options


def target_map(rows, level):
    if level not in config.QUANTILE_COLUMNS:
        raise ValueError(f"Unknown level: {level}")
    result = {}
    for row in rows:
        key = (row["store_id"], row["sku_id"])
        value = row[f"target_{level.lower()}"]
        if key in result or isinstance(value, bool) or not math.isfinite(value) or value < 0 or int(value) != value:
            raise ValueError("Targets must be unique nonnegative whole units")
        result[key] = int(value)
    return result


def assemble_plan(options, quantities, targets, blocked, solver):
    """Floor without rounding up; recompute every post-floor total."""
    shipments = []
    shipped = dict.fromkeys(targets, 0)
    for option, value in zip(options, quantities, strict=True):
        units = math.floor(float(value))
        if units < 0:
            raise ValueError("Negative allocation cannot be floored to a plan")
        if units:
            shipments.append({**option, "units": units, "freight": units * option["unit_cost"]})
            shipped[(option["store_id"], option["sku_id"])] += units
    unmet = [dict(store_id=s, sku_id=k, target_units=t, shipped_units=shipped[(s, k)],
                  unmet_units=t - shipped[(s, k)]) for (s, k), t in sorted(targets.items())]
    freight = sum(s["freight"] for s in shipments)
    shortage = sum(r["unmet_units"] for r in unmet)
    return dict(shipments=shipments, unmet=unmet, blocked=blocked,
                totals=dict(target_units=sum(targets.values()), shipped_units=sum(shipped.values()),
                            unmet_units=shortage, freight=freight, shipments=len(shipments)),
                planned_objective=freight + config.LOST_SALE_COST * shortage,
                solver=solver, whole_unit_optimality_proven=False)


def allocate(supply: Supply, targets, method="lp"):
    """Supply must already be validated; no hidden truth or forecasting is read."""
    options, blocked = eligible_options(supply)
    expected = set(supply.opening_stock[["store_id", "sku_id"]].itertuples(index=False, name=None))
    if set(targets) != expected or any(isinstance(t, bool) or not math.isfinite(t) or t < 0 or int(t) != t for t in targets.values()):
        raise ValueError("Targets must match inventory pairs and be nonnegative whole units")
    batches = {b.batch_id: float(b.available_units) for b in supply.batches.itertuples(index=False)}
    routes = {(r.warehouse_id, r.store_id): float(r.capacity_units) for r in supply.routes.itertuples(index=False)}
    start = time.perf_counter()
    if method == "lp":
        pairs = sorted(targets)
        batch_keys, route_keys = sorted(batches), sorted(routes)
        n = len(options)
        eq = lil_matrix((len(pairs), n + len(pairs)))
        ub = lil_matrix((len(batches) + len(routes), n + len(pairs)))
        pi = {k: i for i, k in enumerate(pairs)}
        bi = {k: i for i, k in enumerate(batch_keys)}
        ri = {k: i + len(batches) for i, k in enumerate(route_keys)}
        for j, o in enumerate(options):
            eq[pi[(o["store_id"], o["sku_id"] )], j] = 1
            ub[bi[o["batch_id"]], j] = 1
            ub[ri[(o["warehouse_id"], o["store_id"] )], j] = 1
        for i in range(len(pairs)):
            eq[i, n + i] = 1
        try:
            result = linprog([o["unit_cost"] for o in options] + [config.LOST_SALE_COST] * len(pairs),
                             A_ub=ub.tocsr(), b_ub=[batches[k] for k in batch_keys] + [routes[k] for k in route_keys],
                             A_eq=eq.tocsr(), b_eq=[targets[k] for k in pairs], bounds=(0, None), method="highs")
        except Exception as exc:
            return {"solver": dict(status="error", message=str(exc), solve_time_seconds=time.perf_counter() - start)}
        solver = dict(status="optimal" if result.success else ("infeasible" if result.status == 2 else "error"),
                      raw_status=int(result.status), message=result.message,
                      solve_time_seconds=time.perf_counter() - start)
        if not result.success:
            return {"solver": solver}
        solver["continuous_objective"] = float(result.fun)
        quantities = result.x[:n]
    elif method in ("greedy", "pro_rata"):
        quantities = np.zeros(len(options))
        remaining = {k: float(v) for k, v in targets.items()}

        def take(j, amount):
            o = options[j]
            pair, route, batch = (o["store_id"], o["sku_id"]), (o["warehouse_id"], o["store_id"]), o["batch_id"]
            q = max(0., min(amount, remaining[pair], routes[route], batches[batch]))
            quantities[j] += q
            remaining[pair] -= q
            routes[route] -= q
            batches[batch] -= q

        if method == "greedy":
            for j in sorted(range(len(options)), key=lambda j: (options[j]["store_id"], options[j]["sku_id"],
                                                                options[j]["unit_cost"], options[j]["batch_id"])):
                o = options[j]
                take(j, remaining[(o["store_id"], o["sku_id"])])
        else:
            # Each batch: proportional split of starting remaining need, capped
            # by route capacity. Unused shares stay in the warehouse (no topping up).
            for batch in sorted(batches):
                indices = [j for j, o in enumerate(options) if o["batch_id"] == batch]
                need = [remaining[(options[j]["store_id"], options[j]["sku_id"])] for j in indices]
                total = sum(need)
                stock = batches[batch]
                if total:
                    for j, value in zip(indices, need):
                        take(j, stock * value / total)
        continuous = sum(q * o["unit_cost"] for q, o in zip(quantities, options)) + config.LOST_SALE_COST * sum(remaining.values())
        solver = dict(status="feasible", method=method, continuous_objective=float(continuous),
                      solve_time_seconds=time.perf_counter() - start)
    else:
        raise ValueError(f"Unknown allocation method: {method}")
    return assemble_plan(options, quantities, targets, blocked, solver)
