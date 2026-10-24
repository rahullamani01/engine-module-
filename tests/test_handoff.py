"""Tests for forecast-to-optimizer handoff."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src import handoff


@pytest.fixture
def example_data(monkeypatch):

    stores = [
        f"S{i:02d}"
        for i in range(1, config.N_STORES + 1)
    ]

    skus = [
        f"K{i:02d}"
        for i in range(1, config.N_SKUS + 1)
    ]

    rows = []

    for store in stores:
        for sku in skus:

            row = {
                "store_id": store,
                "sku_id": sku,
                "planning_week": config.PLANNING_WEEK,
                "method": "MA4",
                "used_pooled_residuals": False,
            }

            for level, value in zip(
                config.QUANTILE_COLUMNS,
                [50, 60, 70, 80, 90, 95],
            ):
                row[level] = value

            rows.append(row)

    forecast = pd.DataFrame(rows)

    stock = forecast[
        ["store_id", "sku_id"]
    ].copy()

    stock["opening_units"] = 20

    monkeypatch.setattr(
        handoff,
        "load_forecast",
        lambda scenario, output_dir: forecast.copy(),
    )

    return stock


def test_generates_six_levels(example_data):

    results = handoff.prepare_handoff(
        "normal",
        example_data,
    )

    assert set(results) == set(
        config.QUANTILE_COLUMNS
    )


def test_each_level_has_40_rows(example_data):

    results = handoff.prepare_handoff(
        "normal",
        example_data,
    )

    for targets in results.values():
        assert len(targets) == 40


def test_p90_target_calculation(example_data):

    results = handoff.prepare_handoff(
        "normal",
        example_data,
    )

    targets = results["P90"]

    assert (targets["target_units"] == 70).all()


def test_higher_level_never_decreases_target(example_data):

    results = handoff.prepare_handoff(
        "normal",
        example_data,
    )

    previous = None

    for level in config.QUANTILE_COLUMNS:

        current = results[level][
            "target_units"
        ].to_numpy()

        if previous is not None:
            assert np.all(current >= previous)

        previous = current


def test_exports_six_files(example_data, tmp_path):

    paths = handoff.export_handoff(
        "normal",
        example_data,
        output_dir=tmp_path,
    )

    assert len(paths) == 6

    for path in paths.values():
        assert path.is_file()

        frame = pd.read_csv(path)

        assert len(frame) == 40


def test_missing_inventory_rejected(
    example_data,
    tmp_path,
):

    incomplete = example_data.iloc[:-1]

    with pytest.raises(ValueError):
        handoff.export_handoff(
            "normal",
            incomplete,
            output_dir=tmp_path,
        )

    assert not (
        tmp_path / "normal"
    ).exists()