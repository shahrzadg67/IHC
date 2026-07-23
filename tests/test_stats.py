"""Unit tests for Stage 10 aggregation + internal distance test. CPU-only."""
import numpy as np
import pandas as pd
import pytest

from src.s10_aggregate_stats import distance_tests, per_image_summary


def _cells():
    return pd.DataFrame({
        "field_id": ["F"] * 15,
        "panel": ["B"] * 15, "sample": ["1"] * 15, "magnification": ["40x"] * 15,
        "condition": ["CFA"] * 15,
        "F4-80_pos": [True] * 5 + [False] * 10,
        "distance_to_nerve_px": [1, 2, 3, 4, 5] + [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        "nerve_associated": [True] * 5 + [False] * 10,
    })


def test_per_image_density_and_pct():
    nerve = pd.DataFrame({"field_id": ["F"], "area_fraction_pct": [5.0]})
    pi = per_image_summary(_cells(), nerve)
    row = pi.iloc[0]
    assert row["n_cells"] == 15
    assert row["density_per_Mpx"] == pytest.approx(15 / (1024 * 1024 / 1e6), abs=0.1)
    assert row["F4-80_pct_pos"] == pytest.approx(100 * 5 / 15, abs=0.1)
    assert row["nerve_area_pct"] == 5.0
    assert row["pct_nerve_associated"] == pytest.approx(100 * 5 / 15, abs=0.1)


def test_distance_test_detects_positive_closer():
    dt = distance_tests(_cells())
    r = dt[dt["marker"] == "F4-80"].iloc[0]
    assert r["median_pos"] < r["median_neg"]     # positives nearer the nerve
    assert r["median_diff"] < 0
    assert r["fields_pos_closer"] == 1 and r["fields_tested"] == 1
    assert float(r["p_pooled"]) < 0.05


def test_distance_test_skips_sparse_marker():
    cells = _cells()
    cells["RARE_pos"] = [True] + [False] * 14        # only 1 positive -> skipped
    dt = distance_tests(cells)
    assert "RARE" not in set(dt["marker"])
