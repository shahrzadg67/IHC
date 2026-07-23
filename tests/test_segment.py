"""Unit tests for Stage 3/4 helpers that don't require a GPU or Cellpose.

Covers the config-driven eval kwargs, nucleus->cell expansion, the magnification
lookup, and the QC overlay writer.  pytest tests/test_segment.py
"""
from pathlib import Path

import numpy as np
import pytest
from skimage.segmentation import expand_labels

from src.s0_config import load_config
from src.s3_segment_nuclei import eval_kwargs_from_cfg
from src.s4_cell_bodies import field_magnification
from src.utils.viz import overlay_labels

CFG = load_config("config/config.yaml")


def test_eval_kwargs_maps_config():
    kw = eval_kwargs_from_cfg(CFG["segmentation"])
    assert kw["flow_threshold"] == 0.4
    assert kw["cellprob_threshold"] == 0.0
    assert kw["normalize"] is True
    assert kw["min_size"] == 15
    # diameter is null in config -> omitted (let Cellpose-SAM auto-scale)
    assert "diameter" not in kw


def test_eval_kwargs_includes_diameter_when_set():
    cfg = dict(CFG["segmentation"])
    cfg["diameter"] = 30
    assert eval_kwargs_from_cfg(cfg)["diameter"] == 30.0


def test_expand_labels_grows_without_overlap():
    # two adjacent nuclei; expansion must not let one overwrite the other
    lab = np.zeros((20, 20), dtype=np.uint16)
    lab[5, 5] = 1
    lab[5, 14] = 2
    grown = expand_labels(lab, distance=3)
    assert grown.max() == 2
    assert (grown == 1).sum() > 1 and (grown == 2).sum() > 1   # both expanded
    # label 1's region and label 2's region stay disjoint
    assert not ((grown == 1) & (grown == 2)).any()


def test_field_magnification_from_manifest():
    # These field_ids exist in the generated manifest.
    assert field_magnification(CFG, "A_CFA_20x") == "20x"
    assert field_magnification(CFG, "B_1_40x") == "40x"


def test_overlay_labels_writes_png(tmp_path):
    gray = (np.random.default_rng(0).random((32, 32)) * 255).astype(np.uint8)
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[8:16, 8:16] = 1
    labels[20:28, 20:28] = 2
    out = overlay_labels(gray, labels, "test", tmp_path / "ov.png")
    assert Path(out).exists() and Path(out).stat().st_size > 0
