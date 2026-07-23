"""Stage 6 - Marker positivity calling.

For each panel and each positivity marker (all measured markers except the nuclear anchor
DAPI and the fiber markers), a threshold is computed on the pooled per-cell mean intensity
and each cell is called positive/negative. Thresholds are logged; a QC histogram with the
cutoff line is written per panel x marker; and a per-image %-positive summary is produced.

NB: these are 8-bit RGB exports with per-image LUTs, so intensities are only relatively
comparable -> thresholds are pooled per panel (documented caveat). Method is config-driven.

CLI:  python -m src.s6_positivity --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu, threshold_triangle

from src.s0_config import load_config


def positivity_markers(cfg: Dict[str, Any], cells: pd.DataFrame) -> List[str]:
    """Measured markers minus the nuclear anchor and fiber markers."""
    nuclear = cfg.get("nuclear_marker", "DAPI")
    fibers = set(cfg.get("fiber_markers", []))
    measured = [c[:-5] for c in cells.columns if c.endswith("_mean")]
    return [m for m in measured if m != nuclear and m not in fibers]


def compute_threshold(values: np.ndarray, method: str, percentile: float) -> float:
    """Return an intensity cutoff for one marker using the configured method."""
    v = values[np.isfinite(values)]
    if method == "otsu":
        return float(threshold_otsu(v))
    if method == "triangle":
        return float(threshold_triangle(v))
    if method == "percentile":
        return float(np.percentile(v, percentile))
    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        g = GaussianMixture(2, random_state=0).fit(v.reshape(-1, 1))
        hi = int(np.argmax(g.means_.ravel()))
        xs = np.linspace(float(v.min()), float(v.max()), 512).reshape(-1, 1)
        post = g.predict_proba(xs)[:, hi]
        idx = int(np.argmax(post >= 0.5))
        return float(xs[idx, 0])
    raise ValueError(f"Unknown positivity method '{method}'")


def run(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pos_cfg = cfg.get("positivity", {})
    method = pos_cfg.get("method", "otsu")
    percentile = float(pos_cfg.get("percentile", 99.0))
    min_cells = int(pos_cfg.get("min_cells", 30))

    tables_dir = Path(cfg["paths"]["masks_dir"]).parent / "tables"
    qc_dir = Path(cfg["paths"]["masks_dir"]).parent / "qc_quant"
    cells = pd.read_csv(tables_dir / "cells.csv", dtype={"sample": str})

    markers = positivity_markers(cfg, cells)
    thresholds: List[Dict[str, Any]] = []

    for panel in sorted(cells["panel"].dropna().unique()):
        pmask = cells["panel"] == panel
        for marker in markers:
            col = f"{marker}_mean"
            if col not in cells.columns:
                continue
            vals = cells.loc[pmask, col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size < min_cells:
                continue  # marker not in this panel (or too few cells)

            thr = compute_threshold(vals, method, percentile)
            call = pmask & (cells[col] > thr)
            cells.loc[pmask, f"{marker}_pos"] = call[pmask].astype("boolean")
            pct = 100.0 * call[pmask].mean()
            thresholds.append({"panel": panel, "marker": marker, "method": method,
                               "threshold": round(thr, 3), "n_cells": int(vals.size),
                               "pct_positive": round(pct, 1)})
            print(f"  panel {panel} {marker:7s}: thr={thr:7.2f}  {pct:5.1f}% positive "
                  f"({int(vals.size)} cells)")

            # QC histogram with the cutoff line.
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(vals, bins=60, color="#9aa7b5")
            ax.axvline(thr, color="crimson", lw=2, label=f"{method} thr={thr:.1f}")
            ax.set_yscale("log")
            ax.set_title(f"{panel} · {marker} ({pct:.1f}% +)", fontsize=10)
            ax.set_xlabel(f"{marker} mean intensity"); ax.legend(fontsize=8)
            fig.tight_layout()
            qc_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(qc_dir / f"positivity_{panel}_{marker}.png", dpi=110)
            plt.close(fig)

    thr_df = pd.DataFrame(thresholds)
    thr_df.to_csv(tables_dir / "positivity_thresholds.csv", index=False)
    cells.to_csv(tables_dir / "cells_positive.csv", index=False)

    # Per-image %-positive summary (biologically the reportable unit).
    pos_cols = [c for c in cells.columns if c.endswith("_pos")]
    per_image = (cells.groupby(["panel", "field_id"])[pos_cols]
                 .apply(lambda g: 100.0 * g.mean(numeric_only=True)).reset_index())
    per_image.to_csv(tables_dir / "positivity_per_image.csv", index=False)
    return thr_df, per_image


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 6: marker positivity calling.")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    thr_df, _ = run(cfg)
    tables_dir = Path(cfg["paths"]["masks_dir"]).parent / "tables"
    print(f"\nStage 6 done: {len(thr_df)} panel×marker thresholds.")
    print(f"Tables -> {tables_dir} (positivity_thresholds.csv, cells_positive.csv, positivity_per_image.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
