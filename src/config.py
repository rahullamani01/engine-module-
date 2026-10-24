"""Centralised configuration for the Adaptive Distribution Intelligence Engine.

Every PRD parameter lives here. Other modules import from this file and must
not invent their own network sizes, windows, seeds or cost parameters.

Weeks are 1-indexed, matching the PRD (history = weeks 1..52, planning = week 53).
Ranges written as (start, end) tuples are INCLUSIVE on both ends.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --- Reproducibility -------------------------------------------------------
SEED = 42

# --- Network size (PRD) ----------------------------------------------------
N_WAREHOUSES = 3
N_STORES = 8
N_SKUS = 5
N_REGIONS = 2
N_SERIES = N_STORES * N_SKUS  # 40 store-SKU time series

# --- Time windows (PRD) ----------------------------------------------------
N_HISTORY_WEEKS = 52
PLANNING_WEEK = 53
WARMUP_WEEKS = (1, 4)
TRAIN_WEEKS = (5, 40)
EVAL_WEEKS = (41, 52)
FINAL_FIT_WEEKS = (5, 52)  # refit window used for the planning-week forecast
N_EVAL_OBSERVATIONS = N_SERIES * (EVAL_WEEKS[1] - EVAL_WEEKS[0] + 1)  # 480
HISTORY_START_DATE = dt.date(2025, 1, 6)  # Monday of week 1 (synthetic calendar)

# --- Forecasting -----------------------------------------------------------
MA_WINDOW = 4
MIN_SERIES_RESIDUALS = 20  # below this, fall back to SKU-pooled residuals
QUANTILES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
QUANTILE_COLUMNS = tuple(f"P{round(q * 100)}" for q in QUANTILES)  # P50 ... P95

# --- Simulation and costs (used from G3/G4 onwards) -------------------------
N_DEMAND_DRAWS = 200
LOST_SALE_COST = 12.0  # INR per unit, default
OVERSTOCK_COST = 3.0  # INR per unit, default
FREIGHT_COST_RANGE = (1.0, 6.0)  # INR per unit
LEAD_TIME_RANGE_DAYS = (0, 3)
LATEST_ARRIVAL_DAY = 2
MIN_SHELF_LIFE_ON_ARRIVAL_DAYS = 3
DEFAULT_P10_FULFILLMENT_THRESHOLD = 0.90

# --- Scenarios -------------------------------------------------------------
SCENARIOS = ("normal", "scarce_supply", "expiry_trap")  # PRD section 6.3
LOCATION_TYPES = ("warehouse", "store")
CALIBRATION_TOLERANCE = 0.10  # PRD 6.1: coverage within +/-10 points of nominal, else flagged


def weeks_in(window: tuple[int, int]) -> range:
    """All week numbers in an inclusive (start, end) window."""
    return range(window[0], window[1] + 1)

# --- Gradient-boosting quantile model (G2) ----------------------------------
# Fixed a priori. NOT tuned on weeks 41-52: tuning on the scoring window would
# bias the ML-vs-baseline comparison.
GB_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 10,
    "subsample": 0.8,
}