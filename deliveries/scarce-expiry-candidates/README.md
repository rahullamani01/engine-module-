# Scarce Supply and Expiry Trap candidates — partner review

**Status: computed synthetic backend candidates, NOT jointly accepted or activated.**

Implementation commit: `e94e05f` (full hash in each `audit/provenance.json`).
This delivery packages artifacts; it does not modify the accepted Normal forecast,
opening stock, mock `results/`, or frontend activation index.

## What changed

- The candidate builder now accepts a scenario and defaults to `normal`, so the
  accepted Normal path is unchanged.
- Forecast equality is checked semantically: CSV parsing can move a quantile by
  one floating-point unit, so quantiles are compared with a tight tolerance while
  identifiers, opening stock and every integer target must match exactly.
- Two complete scenario candidates were built from the same frozen demand,
  accepted forecast and opening stock as Normal. Only supply, routes and batch
  expiry differ, as required by the PRD.

## Scenario conditions

Shared planted traps:

- `W01 → S01` route is inactive.
- `W02 → S02` has a 3-day lead time and is ineligible.
- `W01-K01-B1` expires 2 days after the planning Monday.

| Scenario | Supply condition | Intended effect |
|---|---|---|
| `scarce_supply` | 35 units per warehouse–SKU batch (1,050 total); route capacity 120 units | Eligible stock is below P90 need; shortages are explicit |
| `expiry_trap` | All `W01` batches expire after 2 days; `W01` freight is cheapest | Cheap stock is ineligible on shelf life; feasible plans cost more |

The six quantile forecasts are identical to the accepted Normal forecast. Historical
demand is shared across scenarios by design; no forecaster was rerun. Opening stock
is the accepted store stock, copied without regeneration.

## What to review

`scarce_supply/` and `expiry_trap/` each contain the seven UI-facing JSON files.
`audit/` contains the original supply, all 18 checked plans, all-level comparisons,
the shared demand draws and provenance. `inputs/` freezes the source data, saved
forecast outputs and accepted forecast so the verifier is self-contained.

Do **not** copy `audit/` or `inputs/` into the frontend, overwrite mock `results/`,
or enable activation before frontend review and explicit acceptance.

## Verify from the repository root

Use the backend's pinned environment, separately from the frontend environment:

```bash
python -m pytest -q
python deliveries/scarce-expiry-candidates/verify.py scarce_supply
python deliveries/scarce-expiry-candidates/verify.py expiry_trap
```

The verifier independently:

- rechecks every level and every baseline plan against the original batches,
  routes and targets;
- regenerates supply and compares every batch, route and stock row;
- reproduces the shared 200 × 40 demand matrix and every reported evaluation;
- confirms source and code hashes;
- checks the LP continuous bound, P50 baseline comparison and Risk Dial result;
- confirms no activation index exists.

Recorded output is under `verification/`.

## Measured synthetic results

### Scarce Supply

| Level | Target | Shipped | Unmet | Freight | Mean realized cost | P10 fulfillment |
|---|---:|---:|---:|---:|---:|---:|
| P50 | 1,577 | 969 | 608 | ₹2,344 | ₹12,288.64 | 56.57% |
| P80 | 2,092 | 1,015 | 1,077 | ₹2,389 | ₹13,436.07 | 53.72% |
| P90 | 2,440 | 1,015 | 1,425 | ₹2,343 | ₹14,231.34 | 50.78% |
| P95 | 2,743 | 1,015 | 1,728 | ₹2,325 | ₹14,982.62 | 48.05% |

At the default 90% P10 threshold, **no level qualifies**; the Risk Dial reports
that honestly rather than selecting a level. P10 fulfillment falls as the target
rises because the fixed eligible supply cannot cover the additional target.

P50 baselines, identical inputs and shared draws:

| Method | Planned objective | Mean realized cost | P10 fulfillment | Unmet |
|---|---:|---:|---:|---:|
| LP | ₹9,640 | ₹12,288.64 | 56.57% | 608 |
| Cheapest eligible greedy | ₹10,274 | ₹12,963.67 | 56.72% | 608 |
| Pro-rata | ₹11,789 | ₹13,196.02 | 55.40% | 709 |

LP has the lowest planned objective and mean realized cost here. Greedy has a
slightly higher P10 fulfillment because it distributes the limited stock
differently; this is reported, not hidden.

### Expiry Trap

| Level | Target | Shipped | Unmet | Freight | Mean realized cost | P10 fulfillment |
|---|---:|---:|---:|---:|---:|---:|
| P50 | 1,577 | 1,577 | 0 | ₹4,609 | ₹8,750.97 | 82.10% |
| P80 | 2,092 | 2,092 | 0 | ₹6,238 | ₹9,312.64 | 91.50% |
| P90 | 2,440 | 2,440 | 0 | ₹7,528 | ₹10,837.54 | 94.70% |
| P95 | 2,743 | 2,696 | 47 | ₹8,473 | ₹12,404.82 | 95.57% |

At the default 90% P10 threshold, **P80** is the cheapest qualifying level
(P10 91.50%, mean realized cost ₹9,312.64). Shelf-life blocks rise from **7** in
Normal to **70** here; the cheapest warehouse's stock is excluded.

P50 baselines:

| Method | Planned objective | Mean realized cost | P10 fulfillment | Unmet |
|---|---:|---:|---:|---:|
| LP | ₹4,609 | ₹8,750.97 | 82.10% | 0 |
| Cheapest eligible greedy | ₹4,609 | ₹8,750.97 | 82.10% | 0 |
| Pro-rata | ₹6,568 | ₹10,461.49 | 81.23% | 33 |

LP and greedy tie; pro-rata is worse and leaves 33 target units unmet. Expiry-risk
units at P50 are **2,258** (a proxy for unshipped stock expiring soon, not measured
waste), versus 1,298 in Normal.

## Rebuild from frozen inputs

```bash
python - <<'PY'
from pathlib import Path
from src.planner import build_candidate

root = Path('deliveries/scarce-expiry-candidates/inputs')
destination = Path('candidate_engine_results/partner-review')
for scenario in ('scarce_supply', 'expiry_trap'):
    print(build_candidate(
        destination,
        data_dir=root / 'data' / scenario,
        output_dir=root / 'outputs' / scenario,
        accepted_forecast=root / 'engine_results' / scenario / 'forecast.json',
        scenario=scenario,
    ))
PY
```

The destination must not already contain either scenario. Solver timings and
absolute provenance paths will differ; supply, shared draws and numerical results
should agree.

## Frontend acceptance checklist

- All seven files load for both scenarios.
- All six levels display only after `checker.passed` is true.
- Shipment CSV reimports with the same values.
- Scarce Supply shows unmet units and a “no level meets the threshold” state.
- Expiry Trap shows the shelf-life block count and recommends P80 at 90%.
- P50 LP/greedy tie and pro-rata loss are visible for Expiry Trap.
- No Normal artifact or activation flag is changed before acceptance.

Synthetic, single-week prototype only. Unmet target units are not simulated lost
sales; floored plans are not proven integer-optimal; expiry risk is not measured
waste.
