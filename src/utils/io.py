"""I/O and channel helpers shared across pipeline stages.

Kept deliberately small and dependency-light (numpy + tifffile only) so Stage 0
and Stage 1 can run on a CPU-only login node without the heavy segmentation stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import tifffile


# --------------------------------------------------------------------------- #
# TIFF read / write
# --------------------------------------------------------------------------- #
def read_tiff(path: str | Path) -> np.ndarray:
    """Read a TIFF into a numpy array (shape as stored, e.g. (H, W, 3) for RGB)."""
    return tifffile.imread(str(path))


def write_tiff(path: str | Path, arr: np.ndarray) -> None:
    """Write a 2-D (or 3-D) array as a TIFF, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), arr)


# --------------------------------------------------------------------------- #
# Channel statistics (for dominant-channel detection / QC cross-checks)
# --------------------------------------------------------------------------- #
def channel_means(rgb: np.ndarray, sample_step: int = 4) -> Dict[str, float]:
    """Mean intensity per R/G/B channel, subsampling every `sample_step` pixels.

    Accepts an (H, W, 3+) RGB(A) array. For a 2-D grayscale array, returns the
    same mean under all three keys (so callers stay uniform).
    """
    arr = np.asarray(rgb)
    if arr.ndim == 2:  # already grayscale
        m = float(arr[::sample_step, ::sample_step].mean())
        return {"R": m, "G": m, "B": m}
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected an RGB image (H, W, >=3); got shape {arr.shape}")
    sub = arr[::sample_step, ::sample_step, :3].astype(np.float64)
    return {"R": float(sub[..., 0].mean()),
            "G": float(sub[..., 1].mean()),
            "B": float(sub[..., 2].mean())}


def dominant_channel(rgb: np.ndarray, sample_step: int = 4,
                     populated_frac: float = 0.4) -> str:
    """Return a short code for which channel(s) carry signal.

    Single-colour -> "R"/"G"/"B"; yellow (red+green) -> "RG"; a fully populated
    display/overlay image -> "RGB". A channel counts as "populated" if its mean
    is >= populated_frac * (max channel mean). All-black images return "none".
    """
    means = channel_means(rgb, sample_step=sample_step)
    mx = max(means.values())
    if mx <= 0:
        return "none"
    populated = [ch for ch in ("R", "G", "B") if means[ch] >= populated_frac * mx]
    return "".join(populated)


# --------------------------------------------------------------------------- #
# RGB -> grayscale extraction (Stage 1 core)
# --------------------------------------------------------------------------- #
def rgb_max_project(rgb: np.ndarray) -> np.ndarray:
    """Collapse a pseudo-coloured RGB export to a single 2-D intensity image.

    Uses a per-pixel max across the R/G/B channels. This robustly recovers the
    signal whether a marker is encoded in one channel (e.g. DAPI=blue, SYK=green)
    or two (yellow = red+green, e.g. PGP9.5, CD207), without needing to know the
    LUT in advance. Output preserves the input dtype (uint8 for these exports).
    """
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected an RGB image (H, W, >=3); got shape {arr.shape}")
    return arr[..., :3].max(axis=2).astype(arr.dtype)


def nonzero_fraction(arr: np.ndarray) -> float:
    """Fraction of pixels with signal > 0 (a quick QC on extraction sanity)."""
    a = np.asarray(arr)
    return float((a > 0).mean())
