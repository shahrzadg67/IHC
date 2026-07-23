"""Unit tests for Stage 0 parsing and Stage 1 channel extraction.

Runs without the raw data: uses tiny synthetic RGB arrays and filename strings.
    pytest tests/
"""
import numpy as np
import pytest

from src.s0_config import load_config, normalize_marker, parse_filename
from src.utils.io import dominant_channel, nonzero_fraction, rgb_max_project

CFG = load_config("config/config.yaml")


# --------------------------------------------------------------------------- #
# Stage 1 core: max-projection recovers signal for single-colour AND yellow
# --------------------------------------------------------------------------- #
def _solid(color):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = color
    return img


def test_max_project_blue_dapi():
    gray = rgb_max_project(_solid((0, 0, 200)))  # DAPI = blue
    assert gray.shape == (8, 8) and gray.dtype == np.uint8
    assert gray.max() == 200


def test_max_project_yellow_pgp95():
    gray = rgb_max_project(_solid((150, 150, 0)))  # PGP9.5 = red+green (yellow)
    assert gray.max() == 150  # per-pixel max recovers the yellow signal


def test_max_project_passthrough_2d():
    g = np.arange(16, dtype=np.uint8).reshape(4, 4)
    assert np.array_equal(rgb_max_project(g), g)


def test_max_project_rejects_bad_shape():
    with pytest.raises(ValueError):
        rgb_max_project(np.zeros((8, 8, 2), dtype=np.uint8))


def test_dominant_channel_codes():
    assert dominant_channel(_solid((0, 0, 200))) == "B"       # DAPI
    assert dominant_channel(_solid((200, 0, 0))) == "R"       # F4-80 / CD11c
    assert dominant_channel(_solid((0, 200, 0))) == "G"       # SYK
    assert dominant_channel(_solid((150, 150, 0))) == "RG"    # yellow: PGP9.5/CD207
    assert dominant_channel(_solid((120, 120, 120))) == "RGB"  # overlay/merge
    assert dominant_channel(np.zeros((8, 8, 3), np.uint8)) == "none"


def test_nonzero_fraction():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[:5, :] = 1
    assert nonzero_fraction(img) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Stage 0 parsing: the tricky, inconsistent real filenames
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name, panel, sample, mag, marker, role", [
    ("2026-07-10_CFA 20x CD11C.tif", "A", "CFA", "20x", "CD11c", "marker"),
    ("2026-07-10_CFA 40x DAPI.tif",  "A", "CFA", "40x", "DAPI", "marker"),
    ("2026-07-10_CFA 40x 6_overlay.tif", "A", "CFA", "40x", "overlay", "overlay"),
    ("CFA 1 40X DAPI.tif",   "B", "1",   "40x", "DAPI",  "marker"),
    ("CFA 1.2 40X F480.tif", "B", "1.2", "40x", "F4-80", "marker"),
    ("CFA X 40X PGP9.5.tif", "B", "X",   "40x", "PGP9.5", "marker"),
    ("CFA 1.1 40X merge.tif", "B", "1.1", "40x", "overlay", "overlay"),
])
def test_parse_filename(name, panel, sample, mag, marker, role):
    rec = parse_filename(name, CFG)
    assert rec["panel"] == panel
    assert rec["sample"] == sample
    assert rec["magnification"] == mag
    assert rec["marker"] == marker
    assert rec["role"] == role
    assert rec["notes"] == ""


def test_field_id_groups_channels():
    a = parse_filename("CFA 1.2 40X F480.tif", CFG)
    b = parse_filename("CFA 1.2 40X DAPI.tif", CFG)
    assert a["field_id"] == b["field_id"] == "B_1.2_40x"


def test_unknown_marker_is_flagged_not_crashed():
    rec = parse_filename("CFA 1 40X FOXP3.tif", CFG)
    assert rec["marker"] == "FOXP3"          # raw kept for review
    assert "unrecognised marker" in rec["notes"]


def test_missing_mag_is_flagged():
    rec = parse_filename("random_file.tif", CFG)
    assert rec["magnification"] is None
    assert "magnification" in rec["notes"]


def test_marker_alias_case_insensitive():
    assert normalize_marker("cd11c", CFG["marker_aliases"]) == "CD11c"
    assert normalize_marker("F480", CFG["marker_aliases"]) == "F4-80"
    assert normalize_marker("MERGE", CFG["marker_aliases"]) == "overlay"
