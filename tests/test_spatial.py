"""Unit tests for Stage 8/9 logic (nerve mask, distance-to-nerve, skeleton). CPU-only."""
import numpy as np
import pytest

from src.s8_nerve import nerve_mask, skeleton_metrics
from src.s9_spatial import distance_to_nerve


def test_nerve_mask_thresholds_bright_region():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[20:30, 20:30] = 200                       # a bright 10x10 nerve-like patch
    mask = nerve_mask(img, "otsu", smooth_sigma=0.0, min_object_px=5)
    assert 80 < mask.sum() < 140                  # ~the 100-pixel patch
    assert mask[25, 25]                           # centre is nerve
    assert not mask[5, 5]                          # background is not


def test_nerve_mask_removes_small_specks():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[20:30, 20:30] = 200                       # big patch (kept)
    img[2, 2] = 200                               # 1-px speck (removed)
    mask = nerve_mask(img, "otsu", smooth_sigma=0.0, min_object_px=10)
    assert not mask[2, 2]
    assert mask[25, 25]


def test_distance_to_nerve_line():
    nerve = np.zeros((20, 20), dtype=bool)
    nerve[:, 10] = True                            # vertical nerve at column x=10
    cy = np.array([5.0, 5.0, 5.0])
    cx = np.array([0.0, 10.0, 7.0])
    d = distance_to_nerve(nerve, cy, cx)
    assert d[0] == pytest.approx(10.0)             # 10 px away
    assert d[1] == pytest.approx(0.0)              # on the nerve
    assert d[2] == pytest.approx(3.0)              # |7-10| = 3


def test_distance_to_nerve_empty_mask_is_nan():
    nerve = np.zeros((10, 10), dtype=bool)
    d = distance_to_nerve(nerve, np.array([1.0]), np.array([1.0]))
    assert np.isnan(d[0])


def test_skeleton_metrics_nonzero_for_shape():
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 18:22] = True                      # a bar -> skeleton is a line
    m = skeleton_metrics(mask)
    assert m["skeleton_length_px"] > 0
