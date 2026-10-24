# DistriSense — Forecasting Engine

**Team Byte Works · Triwizathon 2026 · Problem Statement 2**

Adaptive Distribution Intelligence Engine backend: completed Stream A forecasting,
plus a Stream B/D normal candidate with supply validation, checked allocation,
allocation baselines and simulated demand evaluation.

**All data is synthetic (seed 42). Results are not evidence of real-world
performance. P90 is a planning quantile, not a 90% fulfillment guarantee.**

## Current status

**Stream A complete; normal-scenario A → C forecast integration accepted.**
The complete distribution planner is **not jointly accepted**: the normal B/D
candidate is implemented and checked locally; frontend acceptance and full
scarce/expiry exported runs remain pending.

| Area | Status |
|---|---|
| Synthetic demand generation and input validation | Implemented |
| MA4 residual-quantile baseline | Implemented |
| Gradient Boosting quantile forecasting | Implemented |
| Holdout model selection, forecast metrics and calibration | Implemented |
| Six quantile forecasts and inventory-adjusted targets | Implemented |
| Artifact audit, JSON adapter and frontend bridge | Implemented |
| Normal forecast displayed in DistriSense | Accepted by Stream C |
| Warehouse supply, routes, optimizer and independent checker | Implemented; normal candidate pending joint acceptance |
| 200-draw Risk Dial and allocation comparisons | Implemented; normal candidate pending joint acceptance |

### Verification record

- Stream A reports **112 tests passed in 19.59 seconds** at forecasting commit
  `5c22573eda8901c8abc3e308bf746d734446fb4e`.
- Stream A reports all three scenario audits passed, with 40 week-53 forecasts
  per scenario and Gradient Boosting selected for this frozen synthetic dataset.
- Commit `20ad5f1` subsequently added documentation and the normal forecast JSON.
- Stream C verified all 40 pairs against the supplied opening stock, all six
  quantile/target calculations, and actual forecast rendering. Its separate
  app suite passed **37 tests** at integration commit `04e338e`.

These are separate backend and frontend runs, **not** one combined test suite or
full-planner acceptance. Model selection is measured, not hardcoded: the baseline
must be used if it scores better on another run.

## Configuration

| Parameter | Value |
|---|---|
| Network | 3 warehouses, 8 stores, 5 SKUs, 2 regions |
| History / planning | Weeks 1–52 / week 53 |
| Fit / holdout / final refit | Weeks 5–40 / 41–52 / 5–52 |
| Planning levels | P50, P60, P70, P80, P90, P95 |
| Seed | 42 |
| Runtime | Python 3.11+, CPU only |

Canonical scenarios are `normal`, `scarce_supply`, and `expiry_trap`. Historical
demand is shared, so forecasts are identical across these supply scenarios.
Their eventual differences come from supply, routes, and batch expiry—not from
changing demand history to improve results.

## Setup and run

From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q

python -m src.engine normal
python -m src.forecast_audit normal
```

Repeat the engine and audit commands with `scarce_supply` or `expiry_trap` for
the other scenarios. Generated source data is stored under `data/<scenario>/`
and forecasting artifacts under `outputs/<scenario>/` (normally git-ignored).

## Forecast contract

`outputs/<scenario>/planning_forecast.csv` contains **40 unique store–SKU pairs**:

```text
store_id,sku_id,planning_week,method,P50,P60,P70,P80,P90,P95,used_pooled_residuals
```

- IDs follow the actual generated network: `S01`–`S08` and `K01`–`K05`.
- `planning_week` is 53; forecasts are finite, nonnegative and nondecreasing.
- Targets use `max(0, ceil(selected_quantile) - opening_units)`.
- Targets are requests to the optimizer, **not confirmed shipments**.

## Export to the Streamlit app

The app lives in a separate repository:
**[kushwahayashh/distrisense](https://github.com/kushwahayashh/distrisense)**.
It reads saved artifacts; it does not train models or solve LPs per click.

Supply a validated opening-stock CSV with columns:

```text
store_id,sku_id,opening_units
```

It must contain exactly the forecast's 40 unique pairs, with nonnegative integer
inventory. The accepted normal-scenario stock file and its synthetic assumptions
are in the app repo's `data/normal/` directory. Reuse that same file for subsequent
normal targets and plans; do not invent replacement stock during integration.

After generating the scenario forecast, export with:

```bash
python -m src.frontend_bridge normal --stock data/normal/opening_stock.csv
```

This writes `engine_results/normal/forecast.json`, with a `rows` envelope,
preserved IDs, `opening_units`, lowercase `p50`–`p95`, and `target_p50`–`target_p95`.
The bridge accepts `scarce` / `expiry` aliases but always exports to canonical
`scarce_supply` / `expiry_trap` directories.

**Existing JSON files are not overwritten.** This repo already includes the
accepted normal artifact, so exporting to the default destination may raise
`FileExistsError`. To generate a fresh candidate without deleting accepted data:

```bash
python -m src.frontend_bridge normal \
  --stock data/normal/opening_stock.csv \
  --results-dir candidate_engine_results
```

Transfer the validated JSON to the app repository's matching `engine_results/`
directory, coordinating any replacement. Select **Engine results → Normal →
Data & Validation** in Streamlit. Keep mock files in the app's separate `results/`
directory untouched.

## Normal allocation and evaluation candidate

Stream B/D now has a separate saved-artifact pipeline. It **does not rerun
forecasting**, overwrite accepted artifacts, or enable the frontend gate:

```bash
source .venv/bin/activate
python -m pytest -q
python -m src.planner --destination candidate_engine_results
```

Requires the existing `data/normal/{locations,products,opening_stock,hidden_truth}.csv`,
saved normal forecasting outputs, and accepted `engine_results/normal/forecast.json`.
The accepted forecast must exactly match the saved CSV plus opening stock.
The destination scenario directory must not exist; use a different destination
for another run, rather than deleting accepted or unrelated files.

`candidate_engine_results/normal/` contains all seven UI-facing JSON files.
`audit/` additionally contains original supply/stock, all 18 checked plans,
all-level comparisons, the shared demand matrix, input/code hashes, dependency
versions, backend HEAD and working-tree status. An uncommitted implementation
is explicitly recorded as such, not attributed to HEAD alone.

Supply assumptions are fixed in `src/supply.py::SUPPLY_ASSUMPTIONS`, independently
of forecasts/hidden truth: 30 batches, seed 42, 160 units per normal batch,
1,000-unit shared route capacities, seeded costs, and planted eligibility traps.
Blocked counts use first-failure precedence: absent/inactive route → lead time
→ shelf life. Validation rows are CSV line numbers including the header.
Pro-rata processes batches in ID order, caps proportional shares by route capacity,
and leaves unused shares at the warehouse; no redistribution is added. Greedy
processes store/SKU IDs in order, then cost and batch ID. All methods floor
shipments and recompute unmet/costs. Continuous LP objectives are retained separately.

The evaluator alone reads hidden truth, after fixed plans have been checked.
Negative-binomial dispersion is size k. Every method/level shares the same 200
seed-42 draws. Zero-total-demand draws have fulfillment 1. The expiry proxy
includes unshipped stock expiring on/before planning Sunday + 7 days (including
already expired stock); it is not measured waste. The default recommendation
uses a 0.90 P10 threshold, and may be null. UI-facing baseline comparisons remain
P50 only; all six are available in `audit/all_level_comparisons.json`.

First normal candidate, using the accepted forecast and pinned dependencies:

- All six LP solves optimal; all 18 plans independently checked.
- P50: 1,577 target/shipped units, zero unmet, ₹4,027 freight.
- LP and greedy **tie** at P50 mean simulated cost ₹8,168.965; pro-rata ₹10,135.465.
- P80 is cheapest among LP levels meeting P10 fulfillment ≥ 0.90
  (P10 ≈ 0.91496; mean simulated cost ₹8,362.64).
- Backend verification: **162 tests passed in 20.64 s**; this is not frontend
  verification or joint acceptance, and these synthetic results imply no real savings.

Remaining: Stream C must validate/render the candidate and jointly accept normal.
No frontend activation index is written. Scarce/expiry supplies have feasibility,
blocked-rule and baseline-bound tests, but their full exported runs are deliberately
deferred until normal acceptance. Keep demo artifacts and accepted forecast intact.

## Remaining work and ownership

| Stream | Responsibility |
|---|---|
| A | Demand data, forecasting, model selection, targets, forecast adapter/bridge |
| B | Supply/batch/route validation, eligibility, HiGHS LP, flooring, independent checker, allocation baselines |
| C | Streamlit app, saved-result display, verified-plan CSV export |
| D | 200 seeded negative-binomial demand draws, Risk Dial, decision evaluation and comparisons |

Next milestone: **one complete, verified normal scenario**, then the other two.
Besides `forecast.json`, the frontend needs genuine `meta.json`,
`validation.json`, `plan.json`, `risk_dial.json`, `baselines.json`, and
`evaluation.json`. Follow the app repository's `CONTRACT.md` and
`engine_results/README.md` for their shapes and activation rules.

Do **not** set `mock: false` and `integration_validated: true` in the app's engine
index until the full run is jointly accepted. Forecast viewing does not require
full-planner activation. Checker status must come from actual independent
checking, never a placeholder.

## Limitations and claims

- Synthetic data, single-week planning, no live dispatch or ERP/POS integration.
- Lower holdout pinball loss determines the selected forecaster; no general
  ML-superiority or real-world forecast-accuracy claim is made.
- No measured real-business savings or waste reduction is claimed.
- Planned objective and simulated realized cost are different metrics. The LP
  may tie or lose to allocation baselines on realized cost.
- Whole-unit shipments will be floored after optimization; they must not be
  described as proven integer-optimal.
