"""Unit tests for Stage 5/6/7 logic (features, positivity, phenotyping). CPU-only."""
import numpy as np
import pandas as pd
import pytest

from src.s0_config import load_config
from src.s5_features import field_markers
from src.s6_positivity import compute_threshold, positivity_markers
from src.s7_phenotype import cooccurrence, panel_markers, phenotype_label

CFG = load_config("config/config.yaml")


# --- Stage 5 ---
def test_field_markers_from_manifest():
    manifest = pd.read_csv(CFG["paths"]["manifest"], dtype={"sample": str})
    a = set(field_markers(manifest, "A_CFA_40x"))
    b = set(field_markers(manifest, "B_1_40x"))
    assert a == {"DAPI", "CD11c", "CD207", "SYK"}
    assert b == {"DAPI", "F4-80", "PGP9.5", "SYK"}
    assert "overlay" not in a and "overlay" not in b


# --- Stage 6 ---
def test_compute_threshold_separates_bimodal():
    rng = np.random.default_rng(0)
    low = rng.normal(10, 3, 800)
    high = rng.normal(80, 5, 200)
    vals = np.clip(np.concatenate([low, high]), 0, 255)
    thr = compute_threshold(vals, "otsu", 99.0)
    assert 15 < thr < 70          # lands between the two modes (~10 and ~80)
    frac_pos = (vals > thr).mean()
    assert 0.10 < frac_pos < 0.35  # ~the 200/1000 high-mode cells


def test_compute_threshold_percentile():
    vals = np.arange(0, 100, dtype=float)
    assert compute_threshold(vals, "percentile", 90.0) == pytest.approx(89.1, abs=1.0)


def test_positivity_markers_excludes_nuclear_and_fiber():
    cols = ["DAPI_mean", "CD11c_mean", "SYK_mean", "PGP9.5_mean", "F4-80_mean"]
    cells = pd.DataFrame({c: [1.0] for c in cols})
    markers = positivity_markers(CFG, cells)
    assert "DAPI" not in markers        # nuclear anchor
    assert "PGP9.5" not in markers       # fiber marker
    assert set(markers) == {"CD11c", "SYK", "F4-80"}


# --- Stage 7 ---
def _cells(pos):
    df = pd.DataFrame(pos)
    df["panel"] = "B"
    return df


def test_phenotype_label():
    df = pd.DataFrame({"F4-80_pos": [True, True, False, False],
                       "SYK_pos": [True, False, True, False]})
    labels = phenotype_label(df, ["F4-80", "SYK"]).tolist()
    assert labels == ["F4-80+/SYK+", "F4-80+", "SYK+", "none"]


def test_cooccurrence_diagonal_is_pct_positive():
    df = pd.DataFrame({"F4-80_pos": [True, True, False, False],
                       "SYK_pos":   [True, False, True, False]})
    co = cooccurrence(df, ["F4-80", "SYK"])
    assert co.loc["F4-80", "F4-80"] == pytest.approx(50.0)   # 2/4 positive
    assert co.loc["SYK", "SYK"] == pytest.approx(50.0)
    assert co.loc["F4-80", "SYK"] == pytest.approx(25.0)     # 1/4 double-positive


def test_panel_markers_from_pos_cols():
    df = pd.DataFrame({"F4-80_pos": [True, False], "SYK_pos": [True, True],
                       "CD11c_pos": [pd.NA, pd.NA]})
    assert panel_markers(df) == ["F4-80", "SYK"]   # CD11c all-NA -> excluded
