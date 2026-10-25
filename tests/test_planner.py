import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src.allocation import target_map
from src.data_generator import generate_dataset
from src.decision_evaluation import DemandDraws, evaluate_plan, expiry_risk_units
from src.forecast_adapter import build_forecast_json
from src.plan_checker import check_plan
from src.planner import build_candidate
from src.supply import Supply


@pytest.fixture
def sources(tmp_path):
    data, outputs = tmp_path / "data", tmp_path / "outputs"
    data.mkdir()
    outputs.mkdir()
    dataset = generate_dataset()
    dataset.locations.to_csv(data / "locations.csv", index=False)
    dataset.products.to_csv(data / "products.csv", index=False)
    dataset.hidden_truth.to_csv(data / "hidden_truth.csv", index=False)
    stock = pd.DataFrame([[s, k, 7] for s in dataset.store_ids for k in dataset.products.sku_id],
                         columns=["store_id", "sku_id", "opening_units"])
    stock.to_csv(data / "opening_stock.csv", index=False)
    forecast = stock[["store_id", "sku_id"]].copy()
    forecast["planning_week"] = 53
    forecast["method"] = "GB"
    forecast["used_pooled_residuals"] = False
    for i, level in enumerate(config.QUANTILE_COLUMNS):
        forecast[level] = 30. + 5 * i
    forecast.to_csv(outputs / "planning_forecast.csv", index=False)
    (outputs / "selection.json").write_text(json.dumps(dict(selected="GB")))
    pd.DataFrame([dict(method=m, mean_pinball_loss=v) for m, v in [("MA4", 5.), ("GB", 4.)]]).to_csv(outputs / "model_comparison.csv", index=False)
    pd.DataFrame([dict(method="GB", column=level, nominal_coverage=q, observed_coverage=q)
                  for q, level in zip(config.QUANTILES, config.QUANTILE_COLUMNS)]).to_csv(outputs / "calibration.csv", index=False)
    accepted = tmp_path / "accepted.json"
    accepted.write_text(json.dumps(build_forecast_json(forecast, stock)))
    return data, outputs, accepted


def test_export_round_trip_all_plans_and_common_draws(tmp_path, sources):
    data, outputs, accepted = sources
    path = build_candidate(tmp_path / "candidate", data, outputs, accepted)
    load = lambda name: json.loads((path / name).read_text())
    forecast = load("forecast.json")
    assert forecast == json.loads(accepted.read_text())
    assert not (path.parent / "index.json").exists()
    plan = load("plan.json")
    assert list(plan["levels"]) == list(config.QUANTILE_COLUMNS)
    assert load("baselines.json")["levels"] == ["P50"]
    assert len(load("risk_dial.json")["levels"]) == 6
    audit = path / "audit"
    supply = Supply(pd.read_csv(audit / "inventory_batches.csv"), pd.read_csv(audit / "routes.csv"),
                    pd.read_csv(audit / "opening_stock.csv"))
    provenance = load("audit/provenance.json")
    draws = DemandDraws(tuple(map(tuple, provenance["demand"]["pairs"])), np.load(audit / "demand_draws.npy"), 42)
    assert draws.digest == provenance["demand"]["sha256"]
    all_plans = load("audit/all_plans.json")
    for level, methods in all_plans.items():
        targets = target_map(forecast["rows"], level)
        for method, p in methods.items():
            assert check_plan(p, supply, targets)["passed"]
            result = evaluate_plan(p, supply.opening_stock, draws)
            saved = next(m for m in load("audit/all_level_comparisons.json")[level] if m["id"] == method)
            assert saved["realized_cost_mean"] == result["realized_cost_mean"]
        assert plan["levels"][level] == methods["lp"]
    assert load("evaluation.json")["expiry_risk_units"] == expiry_risk_units(plan["levels"]["P50"], supply.batches)
    with pytest.raises(FileExistsError):
        build_candidate(tmp_path / "candidate", data, outputs, accepted)


@pytest.mark.parametrize("scenario", ["scarce_supply", "expiry_trap"])
def test_build_candidate_other_scenarios(tmp_path, sources, scenario):
    data, outputs, accepted = sources
    path = build_candidate(tmp_path / "candidate", data, outputs, accepted, scenario=scenario)
    assert path.name == scenario
    meta = json.loads((path / "meta.json").read_text())
    plan = json.loads((path / "plan.json").read_text())
    assert meta["scenario_id"] == scenario
    assert plan["scenario_id"] == scenario
    assert all(node["checker"]["passed"] for node in plan["levels"].values())
    assert json.loads((path / "forecast.json").read_text()) == json.loads(accepted.read_text())


def test_scarce_and_expiry_plant_expected_traps(tmp_path, sources):
    data, outputs, accepted = sources
    scarce = build_candidate(tmp_path / "scarce", data, outputs, accepted, scenario="scarce_supply")
    expiry = build_candidate(tmp_path / "expiry", data, outputs, accepted, scenario="expiry_trap")
    normal = build_candidate(tmp_path / "normal", data, outputs, accepted)
    scarce_plan = json.loads((scarce / "plan.json").read_text())["levels"]["P50"]
    expiry_plan = json.loads((expiry / "plan.json").read_text())["levels"]["P50"]
    normal_plan = json.loads((normal / "plan.json").read_text())["levels"]["P50"]
    assert scarce_plan["totals"]["unmet_units"] > 0
    assert expiry_plan["blocked"]["shelf_life"] > normal_plan["blocked"]["shelf_life"]
    assert expiry_plan["totals"]["freight"] > normal_plan["totals"]["freight"]


def test_forecast_drift_rejected_without_output(tmp_path, sources):
    data, outputs, accepted = sources
    payload = json.loads(accepted.read_text())
    payload["rows"][0]["target_p50"] += 1
    accepted.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="disagree"):
        build_candidate(tmp_path / "candidate", data, outputs, accepted)
    assert not (tmp_path / "candidate").exists()


def test_hidden_truth_changes_scores_not_allocations(tmp_path, sources):
    data, outputs, accepted = sources
    a = build_candidate(tmp_path / "first", data, outputs, accepted)
    truth = pd.read_csv(data / "hidden_truth.csv")
    truth.loc[truth.week == 53, "true_mean"] = 1.
    truth.to_csv(data / "hidden_truth.csv", index=False)
    b = build_candidate(tmp_path / "second", data, outputs, accepted)
    left = json.loads((a / "audit/all_plans.json").read_text())
    right = json.loads((b / "audit/all_plans.json").read_text())
    for level in config.QUANTILE_COLUMNS:
        for method in ("lp", "pro_rata", "greedy"):
            left[level][method]["solver"].pop("solve_time_seconds")
            right[level][method]["solver"].pop("solve_time_seconds")
    assert left == right
    assert json.loads((a / "risk_dial.json").read_text()) != json.loads((b / "risk_dial.json").read_text())


def test_invalid_supply_blocks_candidate_before_hidden_truth(tmp_path, sources):
    data, outputs, accepted = sources
    products = pd.read_csv(data / "products.csv")
    products.loc[0, "lost_sale_cost"] = -1
    products.to_csv(data / "products.csv", index=False)
    # If validation is skipped, the missing truth would trigger FileNotFoundError.
    (data / "hidden_truth.csv").unlink()
    with pytest.raises(ValueError, match="lost_sale_cost"):
        build_candidate(tmp_path / "candidate", data, outputs, accepted)
    assert not (tmp_path / "candidate").exists()
