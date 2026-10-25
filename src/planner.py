"""Build an unactivated scenario candidate from accepted, saved Stream A outputs.

Never runs forecasting, replaces accepted artifacts, or writes a frontend index.
Usage: python -m src.planner --destination candidate_engine_results --scenario scarce_supply
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import scipy
import sklearn

from src import config
from src.allocation import allocate, target_map
from src.decision_evaluation import draw_demand, evaluate_plan, expiry_risk_units, recommend
from src.forecast_adapter import build_forecast_json
from src.plan_checker import check_plan
from src.supply import SUPPLY_ASSUMPTIONS, generate_supply, planning_date, validate_supply


def _save(path, payload):
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)


def _forecasts_agree(saved, recomputed):
    """Compare semantics, not float repr: CSV parsing can move a value by 1 ulp."""
    if saved.keys() != recomputed.keys():
        return False
    left, right = saved["rows"], recomputed["rows"]
    if len(left) != len(right):
        return False
    for a, b in zip(left, right):
        if a.keys() != b.keys():
            return False
        if (a["store_id"], a["sku_id"], a["opening_units"]) != (
                b["store_id"], b["sku_id"], b["opening_units"]):
            return False
        for level in config.QUANTILE_COLUMNS:
            key = level.lower()
            if not math.isclose(a[key], b[key], rel_tol=1e-12, abs_tol=1e-9):
                return False
            if a[f"target_{key}"] != b[f"target_{key}"]:
                return False
    return True


SCENARIO_NAMES = {"normal": "Normal", "scarce_supply": "Scarce Supply", "expiry_trap": "Expiry Trap"}


def build_candidate(destination, data_dir=None, output_dir=None, accepted_forecast=None,
                    scenario="normal"):
    """Build one scenario. Supply differs; accepted forecast and stock are shared."""
    if scenario not in config.SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    source = Path(data_dir) if data_dir is not None else config.DATA_DIR / scenario
    outputs = Path(output_dir) if output_dir is not None else config.OUTPUT_DIR / scenario
    forecast_path = Path(accepted_forecast) if accepted_forecast is not None else config.PROJECT_ROOT / f"engine_results/{scenario}/forecast.json"
    destination = Path(destination) / scenario
    if destination.exists():
        raise FileExistsError(f"Candidate directory already exists: {destination}")
    source_paths = [source / name for name in ("locations.csv", "products.csv", "opening_stock.csv")]
    source_paths += [outputs / name for name in ("planning_forecast.csv", "model_comparison.csv", "selection.json", "calibration.csv")]
    source_paths += [forecast_path]
    locations, products, stock = [pd.read_csv(p) for p in source_paths[:3]]
    forecast = json.loads(forecast_path.read_text())
    # The existing adapter validates forecast quantiles, planning week, pairs,
    # opening stock and every target. Reject dependency/artifact drift.
    recomputed = build_forecast_json(pd.read_csv(outputs / "planning_forecast.csv"), stock)
    if not _forecasts_agree(forecast, recomputed):
        raise ValueError("Saved forecasts/stock disagree with accepted forecast; coordinate rerun, do not overwrite")
    supply = generate_supply(locations, products, stock, scenario)
    report = validate_supply(supply, locations, products)
    report.raise_if_invalid()
    comparison = pd.read_csv(outputs / "model_comparison.csv")
    selection = json.loads((outputs / "selection.json").read_text())
    mapping = {"MA4": "baseline_ma4", "GB": "ml_gbm"}
    selected = selection["selected"]
    if comparison.method.duplicated().any() or selected not in mapping or not comparison.method.isin(mapping).all():
        raise ValueError("Invalid saved forecaster selection")
    if not pd.read_csv(outputs / "planning_forecast.csv").method.eq(selected).all():
        raise ValueError("Saved forecast method disagrees with selected forecaster")
    scores = {mapping[r.method]: float(r.mean_pinball_loss) for r in comparison.itertuples(index=False)}
    if mapping[selected] not in scores or not all(np.isfinite(v) and v >= 0 for v in scores.values()):
        raise ValueError("Invalid saved holdout scores")
    calibration = pd.read_csv(outputs / "calibration.csv")
    selected_calibration = calibration.loc[calibration.method == selected]
    if list(selected_calibration.column) != list(config.QUANTILE_COLUMNS):
        raise ValueError("Selected calibration must have all six ordered levels")
    calibration_rows = []
    for r, quantile in zip(selected_calibration.itertuples(index=False), config.QUANTILES):
        nominal, observed = float(r.nominal_coverage), float(r.observed_coverage)
        if not (np.isfinite(nominal) and np.isfinite(observed) and nominal == quantile and 0 <= observed <= 1):
            raise ValueError("Invalid calibration fractions")
        calibration_rows.append(dict(level=r.column, nominal=nominal, observed=observed,
                                     within_tolerance=abs(observed - nominal) <= config.CALIBRATION_TOLERANCE))
    plans, all_plans = {}, {}
    for level in config.QUANTILE_COLUMNS:
        targets = target_map(forecast["rows"], level)
        methods = {}
        for method in ("lp", "pro_rata", "greedy"):
            plan = allocate(supply, targets, method)
            if "shipments" not in plan:
                raise RuntimeError(f"{level}/{method} solver failed: {plan['solver']}")
            plan["checker"] = check_plan(plan, supply, targets)
            if not plan["checker"]["passed"]:
                raise RuntimeError(f"{level}/{method} checker failed: {plan['checker']}")
            methods[method] = plan
        for method in ("pro_rata", "greedy"):
            if methods["lp"]["solver"]["continuous_objective"] > methods[method]["solver"]["continuous_objective"] + 1e-6:
                raise RuntimeError(f"Continuous LP bound failed for {level}/{method}")
        plans[level] = methods["lp"]
        all_plans[level] = methods
    # Hidden truth first enters here, after all fixed plans have been constructed
    # and checked. No generator, forecaster or allocator receives this table.
    truth_path = source / "hidden_truth.csv"
    source_paths.append(truth_path)
    draws = draw_demand(pd.read_csv(truth_path), target_map(forecast["rows"], "P50"))
    risk, comparisons = [], {}
    names = {"lp": "LP optimizer", "pro_rata": "Pro-rata", "greedy": "Cheapest eligible first"}
    for level, methods in all_plans.items():
        metrics = {method: evaluate_plan(plan, stock, draws) for method, plan in methods.items()}
        risk.append(dict(level=level, **metrics["lp"]))
        comparisons[level] = [dict(id=m, name=names[m], planned_objective=p["planned_objective"],
                                   continuous_objective=p["solver"]["continuous_objective"],
                                   checker_passed=p["checker"]["passed"], **metrics[m]) for m, p in methods.items()]
    statuses = {level: p["solver"]["status"] for level, p in plans.items()}
    meta = dict(scenario_id=scenario, scenario_name=SCENARIO_NAMES[scenario], planning_week=config.PLANNING_WEEK,
                planning_date=planning_date().isoformat(), seed=config.SEED,
                forecaster_used=mapping[selected], original_forecaster=selected, forecaster_scores=scores,
                solver_status="optimal" if all(s == "optimal" for s in statuses.values()) else "error",
                solver_status_by_level=statuses, solve_time_seconds=sum(p["solver"]["solve_time_seconds"] for p in plans.values()),
                levels=list(config.QUANTILE_COLUMNS), default_level="P50",
                network=dict(warehouses=int(locations.type.eq("warehouse").sum()),
                             stores=int(locations.type.eq("store").sum()), skus=len(products)), latest_arrival_day=config.LATEST_ARRIVAL_DAY,
                min_shelf_life_days=config.MIN_SHELF_LIFE_ON_ARRIVAL_DAYS,
                synthetic=True, joint_acceptance_pending=True)
    files = [("locations.csv", locations), ("products.csv", products), ("opening_stock.csv", stock),
             ("inventory_batches.csv", supply.batches), ("routes.csv", supply.routes)]
    validation = dict(ok=True, files=[dict(name=n, rows=len(f), status="ok", errors=0, warnings=0) for n, f in files],
                      checks=[dict(id=k, label=label, passed=True, detail=detail) for k, label, detail in [
                          ("supply", "Supply and references validate", "Strict row-level validation executed; CSV row numbers include header."),
                          ("accepted_forecast", "Accepted forecast matches saved forecast and stock", "Existing adapter rerun without fitting; all six targets match."),
                          ("checker", "Independent check passes", "All six levels and all three allocation methods checked."),
                          ("lp_bound", "Continuous LP objective bound passes", "Compared against both feasible baselines at all six levels.")]],
                      errors=[], warnings=[])
    provenance = dict(backend_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.PROJECT_ROOT, text=True).strip(),
                      working_tree_status=subprocess.check_output(["git", "status", "--short"], cwd=config.PROJECT_ROOT, text=True).splitlines(),
                      versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                                    scipy=scipy.__version__, sklearn=sklearn.__version__),
                      source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                      code_sha256={str(p.relative_to(config.PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((config.PROJECT_ROOT / "src").glob("*.py"))},
                      demand=dict(seed=draws.seed, draws=len(draws.values), sha256=draws.digest,
                                  pairs=draws.pairs, distribution="negative_binomial", dispersion="size k",
                                  zero_demand_fulfillment=1.0, shared_across_all_levels_and_methods=True),
                      supply_assumptions=SUPPLY_ASSUMPTIONS)
    payloads = {"meta": meta, "validation": validation, "forecast": forecast,
                "plan": dict(scenario_id=scenario, levels=plans),
                "risk_dial": dict(draws=len(draws.values), levels=risk,
                                  default_threshold=config.DEFAULT_P10_FULFILLMENT_THRESHOLD,
                                  recommended_level=recommend(risk)),
                "baselines": dict(levels=["P50"], methods=comparisons["P50"]),
                "evaluation": dict(pinball=dict(scored_weeks="41-52", selected=mapping[selected], **scores),
                                   calibration=calibration_rows, expiry_risk_units=expiry_risk_units(plans["P50"], supply.batches),
                                   expiry_risk_units_by_level={level: expiry_risk_units(p, supply.batches) for level, p in plans.items()},
                                   expiry_risk_definition="Unshipped units with expiry <= planning Sunday + 7 days, including already expired; proxy only")}
    # Reserve a NEW scenario directory only after every computation passes.
    destination.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        _save(destination / f"{name}.json", payload)
    audit = destination / "audit"
    audit.mkdir()
    _save(audit / "provenance.json", provenance)
    _save(audit / "all_level_comparisons.json", comparisons)
    _save(audit / "all_plans.json", all_plans)
    for name, frame in files:
        frame.to_csv(audit / name, index=False)
    np.save(audit / "demand_draws.npy", draws.values, allow_pickle=False)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--scenario", choices=config.SCENARIOS, default="normal")
    args = parser.parse_args()
    print(f"Unactivated candidate written: {build_candidate(args.destination, scenario=args.scenario)}")


if __name__ == "__main__":
    main()
