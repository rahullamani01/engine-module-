"""Recheck this frozen delivery without fitting forecasts or solving allocations."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config
from src.decision_evaluation import draw_demand, evaluate_plan, recommend
from src.plan_checker import check_plan
from src.supply import Supply, generate_supply, validate_supply


def main():
    delivery = Path(__file__).resolve().parent
    candidate = delivery / "normal"
    load = lambda name: json.loads((candidate / name).read_text())
    provenance = load("audit/provenance.json")
    for relative, digest in provenance["code_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest, relative
    for original, digest in provenance["source_sha256"].items():
        parts = Path(original).parts
        start = next(i for i, part in enumerate(parts) if part in ("data", "outputs", "engine_results"))
        snapshot = delivery / "inputs" / Path(*parts[start:])
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest, snapshot
    data = delivery / "inputs/data/normal"
    locations, products, stock = [pd.read_csv(data / f"{name}.csv") for name in ("locations", "products", "opening_stock")]
    audit = candidate / "audit"
    supply = Supply(pd.read_csv(audit / "inventory_batches.csv"), pd.read_csv(audit / "routes.csv"),
                    pd.read_csv(audit / "opening_stock.csv"))
    validate_supply(supply, locations, products).raise_if_invalid()
    regenerated = generate_supply(locations, products, stock)
    for name in ("batches", "routes", "opening_stock"):
        pd.testing.assert_frame_equal(getattr(supply, name), getattr(regenerated, name))
    forecast = load("forecast.json")
    assert forecast == json.loads((delivery / "inputs/engine_results/normal/forecast.json").read_text())
    pairs = [(r["store_id"], r["sku_id"]) for r in forecast["rows"]]
    draws = draw_demand(pd.read_csv(data / "hidden_truth.csv"), pairs)
    np.testing.assert_array_equal(draws.values, np.load(audit / "demand_draws.npy", allow_pickle=False))
    assert draws.digest == provenance["demand"]["sha256"]
    all_plans, comparisons = load("audit/all_plans.json"), load("audit/all_level_comparisons.json")
    plan, risk = load("plan.json"), load("risk_dial.json")
    assert list(all_plans) == list(config.QUANTILE_COLUMNS)
    count = 0
    for level, methods in all_plans.items():
        targets = {(r["store_id"], r["sku_id"]): r[f"target_{level.lower()}"] for r in forecast["rows"]}
        assert set(methods) == {"lp", "greedy", "pro_rata"}
        for method, p in methods.items():
            assert check_plan(p, supply, targets)["passed"], (level, method)
            actual = evaluate_plan(p, stock, draws)
            saved = next(row for row in comparisons[level] if row["id"] == method)
            assert all(actual[key] == saved[key] for key in actual), (level, method)
            assert methods["lp"]["solver"]["continuous_objective"] <= p["solver"]["continuous_objective"] + 1e-6
            if method == "lp":
                assert p == plan["levels"][level]
                saved_risk = next(row for row in risk["levels"] if row["level"] == level)
                assert all(actual[key] == saved_risk[key] for key in actual)
            count += 1
    assert load("baselines.json")["methods"] == comparisons["P50"]
    assert recommend(risk["levels"], risk["default_threshold"]) == risk["recommended_level"]
    assert not (candidate.parent / "index.json").exists()
    print(f"PASS: {count} independently checked plans; shared 200 x 40 demand draws; source/code hashes match.")
    print("PASS: all-level evaluations, LP bounds, P50 comparisons and Risk Dial agree.")
    print("Not frontend acceptance: activation remains off.")


if __name__ == "__main__":
    main()
