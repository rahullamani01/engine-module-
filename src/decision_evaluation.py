"""Stream D only: hidden-truth draws and scoring of fixed, checked plans.

Demand is shared by passing the same DemandDraws instance to every comparison.
Zero-total-demand draws have fulfillment 1 (nothing was requested).
"""
from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass

import numpy as np

from src import config


@dataclass(frozen=True)
class DemandDraws:
    pairs: tuple
    values: np.ndarray
    seed: int

    @property
    def digest(self):
        return hashlib.sha256(self.values.astype("<i8").tobytes()).hexdigest()


def draw_demand(hidden_truth, pairs, seed=config.SEED, draws=config.N_DEMAND_DRAWS):
    truth = hidden_truth.loc[hidden_truth.week == config.PLANNING_WEEK].copy()
    pairs = tuple(sorted(pairs))
    if len(pairs) != len(set(pairs)):
        raise ValueError("Evaluated pairs must be unique")
    keys = list(truth[["store_id", "sku_id"]].itertuples(index=False, name=None))
    if len(keys) != len(set(keys)) or set(keys) != set(pairs):
        raise ValueError("Planning hidden truth must match every evaluated pair exactly once")
    truth = truth.set_index(["store_id", "sku_id"]).loc[list(pairs)]
    mean = truth.true_mean.to_numpy(dtype=float)
    size = truth.dispersion.to_numpy(dtype=float)
    if not np.isfinite(mean).all() or (mean < 0).any() or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("Hidden truth requires finite mean >= 0 and NB size k > 0")
    if not isinstance(draws, int) or isinstance(draws, bool) or draws <= 0:
        raise ValueError("Draw count must be a positive integer")
    values = np.random.default_rng(seed).negative_binomial(size, size / (size + mean), size=(draws, len(pairs)))
    values.setflags(write=False)
    return DemandDraws(pairs, values, seed)


def evaluate_plan(plan, opening_stock, demand: DemandDraws):
    if plan.get("checker", {}).get("passed") is not True:
        raise ValueError("Only independently checked fixed plans may be evaluated")
    stock = {(r.store_id, r.sku_id): float(r.opening_units) for r in opening_stock.itertuples(index=False)}
    if len(stock) != len(opening_stock) or set(stock) != set(demand.pairs):
        raise ValueError("Opening stock must match evaluated pairs exactly once")
    if not all(np.isfinite(v) and v >= 0 and v.is_integer() for v in stock.values()):
        raise ValueError("Opening stock must contain nonnegative whole units")
    available = stock.copy()
    for s in plan["shipments"]:
        available[(s["store_id"], s["sku_id"])] += s["units"]
    available = np.array([available[k] for k in demand.pairs])
    values = demand.values
    if values.ndim != 2 or values.shape[1] != len(demand.pairs) or len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Demand draws must be a nonnegative finite draw-by-pair matrix")
    short = np.maximum(0, values - available)
    excess = np.maximum(0, available - values)
    freight = float(plan["totals"]["freight"])
    costs = freight + config.LOST_SALE_COST * short.sum(axis=1) + config.OVERSTOCK_COST * excess.sum(axis=1)
    total = values.sum(axis=1)
    fulfillment = np.divide(np.minimum(values, available).sum(axis=1), total,
                            out=np.ones(len(total), dtype=float), where=total != 0)
    return dict(freight=freight, realized_cost_mean=float(costs.mean()),
                realized_cost_p90=float(np.quantile(costs, .90)),
                fulfillment_mean=float(fulfillment.mean()), fulfillment_p10=float(np.quantile(fulfillment, .10)),
                shipped_units=int(plan["totals"]["shipped_units"]), unmet_units=int(plan["totals"]["unmet_units"]),
                simulated_lost_sales_mean=float(short.sum(axis=1).mean()))


def recommend(levels, threshold=config.DEFAULT_P10_FULFILLMENT_THRESHOLD):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a fraction in [0, 1]")
    qualifying = [r for r in levels if r["fulfillment_p10"] >= threshold]
    return min(qualifying, key=lambda r: r["realized_cost_mean"])["level"] if qualifying else None


def expiry_risk_units(plan, batches):
    """Unshipped stock expiring by week-end + 7 days, including already expired.

    Week end is Sunday (Monday + 6 days). This is a proxy, not measured waste.
    """
    end = config.HISTORY_START_DATE + dt.timedelta(weeks=config.PLANNING_WEEK - 1, days=13)
    shipped = {}
    for s in plan["shipments"]:
        shipped[s["batch_id"]] = shipped.get(s["batch_id"], 0) + s["units"]
    return int(sum(float(b.available_units) - shipped.get(b.batch_id, 0)
                   for b in batches.itertuples(index=False) if dt.date.fromisoformat(b.expiry_date) <= end))
