"""Deterministic synthetic data generator (SYNTHETIC - not real business data).

Follows PRD section 3.1 schemas. G1 scope: locations.csv, products.csv,
demand_history.csv (weeks 1-52), hidden_truth.csv (weeks 1-53) and config.json.
inventory_batches.csv, store_inventory.csv and routes.csv arrive in G3.

Demand model per store-SKU series:
    mean_t = base * (1 + amp * sin(2*pi*(t + phase)/52)) * (1 + trend * t)
    demand_t ~ Poisson(mean_t * Gamma(shape=k, scale=1/k))  == NegBin(mean_t, size=k)
Planning-week (53) demand is NOT generated; only its true mean/spread is recorded
in the hidden truth, for drawing test demand in G4.

PRD section 6.3 scenarios differ in SUPPLY (stock levels, batch expiry), not in
demand history, so all three scenarios share the same demand_history at seed 42.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.schemas import to_weekly

# Placeholder shelf lives (days) per SKU; G3 batch expiry dates are built from these.
SKU_SHELF_LIFE_DAYS = (7, 10, 14, 21, 30)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str


SCENARIO_SPECS: dict[str, ScenarioSpec] = {
    "normal": ScenarioSpec("normal", "Ample stock, few blocked options; optimizer may only tie simple rules."),
    "scarce_supply": ScenarioSpec(
        "scarce_supply",
        "Eligible stock below P90 need for several store-SKU pairs (stock tables added in G3)."),
    "expiry_trap": ScenarioSpec(
        "expiry_trap",
        "Cheapest warehouse holds batches failing the shelf-life or lead-time rule (batch/route tables added in G3)."),
}


@dataclass(frozen=True)
class SyntheticDataset:
    scenario: str
    seed: int
    locations: pd.DataFrame
    products: pd.DataFrame
    demand_history: pd.DataFrame  # PRD columns; weeks 1..52 only
    hidden_truth: pd.DataFrame  # weeks 1..53; never a forecaster input

    @property
    def weekly(self) -> pd.DataFrame:
        """Internal (store_id, sku_id, week, demand) view used by forecasting code."""
        return to_weekly(self.demand_history)

    @property
    def store_ids(self) -> list[str]:
        return self.locations.loc[self.locations["type"] == "store", "location_id"].tolist()


def get_scenario(name: str) -> ScenarioSpec:
    if name not in SCENARIO_SPECS:
        raise ValueError(f"Unknown scenario {name!r}; choose from {sorted(SCENARIO_SPECS)}")
    return SCENARIO_SPECS[name]


def generate_dataset(scenario: str = "normal", seed: int = config.SEED) -> SyntheticDataset:
    get_scenario(scenario)
    rng = np.random.default_rng(seed)
    ns, nk, nw = config.N_STORES, config.N_SKUS, config.N_WAREHOUSES

    store_ids = [f"S{i + 1:02d}" for i in range(ns)]
    sku_ids = [f"K{i + 1:02d}" for i in range(nk)]
    wh_ids = [f"W{i + 1:02d}" for i in range(nw)]
    regions = [f"R{i + 1}" for i in range(config.N_REGIONS)]

    locations = pd.DataFrame({
        "location_id": wh_ids + store_ids,
        "type": ["warehouse"] * nw + ["store"] * ns,
        "region": [regions[i % config.N_REGIONS] for i in range(nw)]
                  + [regions[0] if i < ns // 2 else regions[1] for i in range(ns)],
    })
    products = pd.DataFrame({
        "sku_id": sku_ids,
        "shelf_life_days": SKU_SHELF_LIFE_DAYS[:nk],
        "lost_sale_cost": config.LOST_SALE_COST,
        "overstock_cost": config.OVERSTOCK_COST,
    })

    # Parameter draws (fixed order => reproducible).
    sku_level = rng.uniform(20, 60, nk)
    store_scale = rng.uniform(0.6, 1.4, ns)
    amplitude = rng.uniform(0.05, 0.25, (ns, nk))
    phase = rng.uniform(0, 52, (ns, nk))
    trend = rng.uniform(-0.002, 0.004, (ns, nk))
    dispersion = rng.uniform(8, 20, (ns, nk))

    all_weeks = np.arange(1, config.PLANNING_WEEK + 1)  # 1..53
    base = store_scale[:, None] * sku_level[None, :]
    seasonal = 1 + amplitude[:, :, None] * np.sin(2 * np.pi * (all_weeks[None, None, :] + phase[:, :, None]) / 52)
    growth = 1 + trend[:, :, None] * all_weeks[None, None, :]
    true_mean = base[:, :, None] * seasonal * growth

    n_hist = config.N_HISTORY_WEEKS
    noise = rng.gamma(shape=dispersion[:, :, None], scale=1 / dispersion[:, :, None], size=(ns, nk, n_hist))
    demand = rng.poisson(true_mean[:, :, :n_hist] * noise)

    week_dates = [(config.HISTORY_START_DATE + dt.timedelta(days=7 * (int(w) - 1))).isoformat() for w in all_weeks]

    def _long(n_weeks: int):
        s, k, w = np.meshgrid(np.arange(ns), np.arange(nk), np.arange(n_weeks), indexing="ij")
        return s.ravel(), k.ravel(), w.ravel()

    s, k, w = _long(n_hist)
    demand_history = pd.DataFrame({
        "week_start": np.array(week_dates)[w],
        "store_id": np.array(store_ids)[s],
        "sku_id": np.array(sku_ids)[k],
        "units_sold": demand[s, k, w].astype(int),
    })
    s, k, w = _long(len(all_weeks))
    hidden_truth = pd.DataFrame({
        "store_id": np.array(store_ids)[s],
        "sku_id": np.array(sku_ids)[k],
        "week": w + 1,
        "true_mean": true_mean[s, k, w],
        "dispersion": dispersion[s, k],
    })
    return SyntheticDataset(scenario, seed, locations, products, demand_history, hidden_truth)


_TABLES = ("locations", "products", "demand_history", "hidden_truth")


def save_dataset(dataset: SyntheticDataset, directory: Path) -> None:
    """Write PRD CSVs plus config.json (PRD fields + scenario metadata) into `directory`."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in _TABLES:
        getattr(dataset, name).to_csv(directory / f"{name}.csv", index=False)
    cfg = {
        "planning_week": config.PLANNING_WEEK,
        "quantiles": list(config.QUANTILES),
        "latest_arrival_day": config.LATEST_ARRIVAL_DAY,
        "min_shelf_life_days": config.MIN_SHELF_LIFE_ON_ARRIVAL_DAYS,
        "seed": dataset.seed,
        "scenario": dataset.scenario,
        "scenario_description": get_scenario(dataset.scenario).description,
        "synthetic": True,
    }
    (directory / "config.json").write_text(json.dumps(cfg, indent=2))


def load_dataset(directory: Path) -> SyntheticDataset:
    directory = Path(directory)
    meta = json.loads((directory / "config.json").read_text())
    tables = {name: pd.read_csv(directory / f"{name}.csv") for name in _TABLES}
    return SyntheticDataset(meta["scenario"], meta["seed"], **tables)
