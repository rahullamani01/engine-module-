import pandas as pd

from src import config
from src.data_generator import generate_dataset
from src.evaluation import BASELINE, ML, choose_method, save_results, select_and_forecast
from src.schemas import validate_forecast_frame

_cache = {}


def sel():
    if "sel" not in _cache:
        _cache["sel"] = select_and_forecast(generate_dataset("normal").weekly)
    return _cache["sel"]


def test_choose_method_rules():
    assert choose_method({BASELINE: 5.0, ML: 4.0})[0] == ML
    assert choose_method({BASELINE: 5.0, ML: 5.0})[0] == BASELINE      # tie => baseline
    name, reason = choose_method({BASELINE: 5.0, ML: 6.0})
    assert name == BASELINE and "no ML-superiority claim" in reason
    assert choose_method({BASELINE: 5.0})[0] == BASELINE                # ML missing


def test_both_methods_scored_on_same_480_observations_and_lower_loss_selected():
    s = sel()
    assert set(s.results) == {BASELINE, ML}
    for r in s.results.values():
        assert r.summary["n_observations"] == 480
        assert r.metadata["fit_weeks"] == [5, 40] and r.metadata["eval_weeks"] == [41, 52]
    scores = {n: r.summary["mean_pinball_loss"] for n, r in s.results.items()}
    assert (s.selected == ML) == (scores[ML] < scores[BASELINE])  # GB only if strictly lower
    assert s.ml_driven == (s.selected == ML)


def test_final_forecast_refit_on_weeks_5_to_52_for_week_53():
    s = sel()
    assert s.final_metadata["fit_weeks"] == [5, 52]
    assert len(s.planning_forecast) == 40 and (s.planning_forecast["planning_week"] == 53).all()
    assert (s.planning_forecast["method"] == s.selected).all()
    assert validate_forecast_frame(s.planning_forecast) == []


def test_ml_failure_falls_back_to_baseline_and_is_reported():
    class Broken:
        method_name = ML
        def fit(self, *a, **k):
            raise RuntimeError("boom")
    s = select_and_forecast(generate_dataset("normal").weekly, ml_factory=Broken)
    assert s.selected == BASELINE and not s.ml_driven and "boom" in s.ml_error and "boom" in s.reason
    assert (s.planning_forecast["method"] == BASELINE).all()


def test_selection_is_reproducible():
    again = select_and_forecast(generate_dataset("normal").weekly)
    pd.testing.assert_frame_equal(again.planning_forecast, sel().planning_forecast)


def test_save_results_writes_all_artifacts(tmp_path):
    save_results(sel(), tmp_path)
    for name in ("model_comparison.csv", "calibration.csv", "holdout_forecasts.csv",
                 "planning_forecast.csv", "selection.json"):
        assert (tmp_path / name).exists(), name
    assert len(pd.read_csv(tmp_path / "holdout_forecasts.csv")) == 960  # 480 per method
    