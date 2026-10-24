"""Model comparison, automatic selection and planning-week forecast (G2).

Order of operations (PRD 3.3):
  1. fit MA4 and GB on weeks 5-40;
  2. score both on the same 480 held-out observations (weeks 41-52);
  3. select the lower mean pinball loss (GB must be strictly better, else MA4);
  4. ONLY THEN refit the selected method on weeks 5-52 and forecast week 53.
If GB fails or does not win, MA4 ships and no ML claim is made.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src import config
from src.baseline import MA4ResidualQuantileForecaster
from src.metrics import evaluate_quantile_forecasts
from src.quantile_model import GradientBoostingQuantileForecaster

BASELINE, ML = MA4ResidualQuantileForecaster.method_name, GradientBoostingQuantileForecaster.method_name


@dataclass
class MethodResult:
    name: str
    forecasts: pd.DataFrame
    per_quantile: pd.DataFrame
    summary: dict
    metadata: dict


@dataclass
class SelectionResult:
    selected: str
    ml_driven: bool
    reason: str
    results: dict[str, MethodResult]
    planning_forecast: pd.DataFrame
    final_metadata: dict
    ml_error: str | None = None


def evaluate_method(forecaster, sales, fit_weeks=config.TRAIN_WEEKS, eval_weeks=config.EVAL_WEEKS) -> MethodResult:
    forecaster.fit(sales, fit_weeks)
    forecasts = forecaster.predict(sales, config.weeks_in(eval_weeks))
    per_q, summary = evaluate_quantile_forecasts(forecasts, sales)
    return MethodResult(forecaster.method_name, forecasts, per_q, summary,
                        forecaster.metadata() | {"eval_weeks": list(eval_weeks)})


def choose_method(scores: dict[str, float | None]) -> tuple[str, str]:
    """Lower mean pinball wins; GB must be STRICTLY lower; missing GB score => baseline."""
    base, ml = scores[BASELINE], scores.get(ML)
    if ml is None:
        return BASELINE, f"{ML} unavailable; {BASELINE} used (no ML claim)."
    if ml < base:
        return ML, f"{ML} mean pinball {ml:.4f} < {BASELINE} {base:.4f} on weeks 41-52."
    return BASELINE, (f"{ML} did not beat {BASELINE} ({ml:.4f} vs {base:.4f}) on weeks 41-52; "
                      f"{BASELINE} used (no ML-superiority claim).")


def select_and_forecast(sales, ml_factory=GradientBoostingQuantileForecaster) -> SelectionResult:
    results = {BASELINE: evaluate_method(MA4ResidualQuantileForecaster(), sales)}
    ml_error = None
    try:
        results[ML] = evaluate_method(ml_factory(), sales)
        keys = ["store_id", "sku_id", "planning_week"]
        if not results[ML].forecasts[keys].equals(results[BASELINE].forecasts[keys]):
            raise RuntimeError("Methods were not scored on identical observations.")
    except Exception as exc:  # reported, never hidden
        results.pop(ML, None)
        ml_error = f"{type(exc).__name__}: {exc}"

    scores = {name: r.summary["mean_pinball_loss"] for name, r in results.items()}
    selected, reason = choose_method(scores)
    if ml_error:
        reason += f" ({ml_error})"

    final = MA4ResidualQuantileForecaster() if selected == BASELINE else ml_factory()
    final.fit(sales, config.FINAL_FIT_WEEKS)  # refit only AFTER the held-out evaluation above
    planning = final.predict(sales, [config.PLANNING_WEEK])
    return SelectionResult(selected, selected == ML, reason, results, planning, final.metadata(), ml_error)


def comparison_tables(sel: SelectionResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame([
        {"method": n, "mean_pinball_loss": r.summary["mean_pinball_loss"], "wape_p50": r.summary["wape_p50"],
         "n_observations": r.summary["n_observations"], "selected": n == sel.selected}
        for n, r in sel.results.items()])
    calibration = pd.concat([r.per_quantile.assign(method=n) for n, r in sel.results.items()], ignore_index=True)
    return summary, calibration[["method"] + [c for c in calibration.columns if c != "method"]]


def save_results(sel: SelectionResult, out_dir: Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, calibration = comparison_tables(sel)
    summary.to_csv(out / "model_comparison.csv", index=False)
    calibration.to_csv(out / "calibration.csv", index=False)
    pd.concat([r.forecasts for r in sel.results.values()], ignore_index=True).to_csv(
        out / "holdout_forecasts.csv", index=False)
    sel.planning_forecast.to_csv(out / "planning_forecast.csv", index=False)
    (out / "selection.json").write_text(json.dumps({
        "selected": sel.selected, "ml_driven": sel.ml_driven, "reason": sel.reason, "ml_error": sel.ml_error,
        "holdout_metadata": {n: r.metadata for n, r in sel.results.items()},
        "final_fit_metadata": sel.final_metadata, "seed": config.SEED, "synthetic_data": True}, indent=2))


def run(scenario: str = "normal", out_dir: Path = config.OUTPUT_DIR) -> SelectionResult:
    from src.data_generator import generate_dataset
    from src.data_validation import validate_or_raise

    ds = generate_dataset(scenario)
    validate_or_raise(ds)
    sel = select_and_forecast(ds.weekly)
    save_results(sel, Path(out_dir) / scenario)
    summary, calibration = comparison_tables(sel)
    print(f"[{scenario}]\n{summary.round(4).to_string(index=False)}\n\n{calibration.round(4).to_string(index=False)}")
    print(f"\nSelected: {sel.selected} | ML-driven: {sel.ml_driven}\n{sel.reason}")
    return sel


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "normal")