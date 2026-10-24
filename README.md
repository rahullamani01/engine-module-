# DistriSense — Forecasting Engine

**Team Byte Works · Triwizathon 2026 · Problem Statement 2**

Stream A of the Adaptive Distribution Intelligence Engine: synthetic demand
generation, validation, probabilistic forecasting, model selection, and
inventory-adjusted targets for weekly FMCG replenishment.

**All data is synthetic (seed 42). Results are not evidence of real-world
performance. P90 is a planning quantile, not a 90% fulfillment guarantee.**

## Current status

**Stream A complete; normal-scenario A → C forecast integration accepted.**
The complete distribution planner is **not finished**: shipment optimization,
independent feasibility checking, and demand-draw evaluation remain separate work.

| Area | Status |
|---|---|
| Synthetic demand generation and input validation | Implemented |
| MA4 residual-quantile baseline | Implemented |
| Gradient Boosting quantile forecasting | Implemented |
| Holdout model selection, forecast metrics and calibration | Implemented |
| Six quantile forecasts and inventory-adjusted targets | Implemented |
| Artifact audit, JSON adapter and frontend bridge | Implemented |
| Normal forecast displayed in DistriSense | Accepted by Stream C |
| Warehouse supply, routes, optimizer and independent checker | Pending Stream B |
| 200-draw Risk Dial and allocation comparisons | Pending Stream D / B |

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
