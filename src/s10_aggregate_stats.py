"""Stage 10 - Aggregation & (descriptive) statistics.

Rolls the single-cell table up to per-image and per-sample summaries (cell density,
%-positive per marker, nerve area fraction, distance-to-nerve), and runs the internal
tests that are valid WITHOUT a control group: marker⁺ vs marker⁻ distance-to-nerve
(pooled Mann–Whitney, exploratory) plus a per-field consistency count (how many fields
show marker⁺ cells closer to nerve — a replicate-level statement).

⚠️ All images are the CFA condition → this is DESCRIPTIVE, not a between-group comparison.

CLI:  python -m src.s10_aggregate_stats --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.s0_config import load_config

FIELD_PX = 1024 * 1024  # all fields are 1024x1024


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def per_image_summary(cells: pd.DataFrame, nerve: pd.DataFrame) -> pd.DataFrame:
    pos_cols = [c for c in cells.columns if c.endswith("_pos")]
    rows: List[Dict[str, Any]] = []
    for fid, g in cells.groupby("field_id"):
        rec: Dict[str, Any] = {
            "field_id": fid, "panel": g["panel"].iloc[0], "sample": str(g["sample"].iloc[0]),
            "magnification": g["magnification"].iloc[0], "condition": g["condition"].iloc[0],
            "n_cells": len(g), "density_per_Mpx": round(len(g) / (FIELD_PX / 1e6), 1),
        }
        for pc in pos_cols:
            b = _to_bool(g[pc])
            if g[pc].notna().any():   # marker present in this panel
                rec[f"{pc[:-4]}_pct_pos"] = round(100.0 * b.mean(), 1)
        if g["distance_to_nerve_px"].notna().any():
            rec["mean_dist_nerve_px"] = round(g["distance_to_nerve_px"].mean(), 1)
            rec["median_dist_nerve_px"] = round(g["distance_to_nerve_px"].median(), 1)
            rec["pct_nerve_associated"] = round(100.0 * _to_bool(g["nerve_associated"]).mean(), 1)
        rows.append(rec)
    df = pd.DataFrame(rows)
    if not nerve.empty:
        df = df.merge(nerve[["field_id", "area_fraction_pct"]]
                      .rename(columns={"area_fraction_pct": "nerve_area_pct"}), on="field_id", how="left")
    return df


def per_sample_summary(per_image: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to the biological-replicate level (mean ± SD across a sample's fields)."""
    num = per_image.select_dtypes("number").columns
    grp = per_image.groupby(["panel", "sample"])
    mean = grp[num].mean().add_suffix("_mean")
    std = grp[num].std().add_suffix("_sd")
    out = pd.concat([mean, std], axis=1).reset_index()
    out.insert(2, "n_fields", grp.size().values)
    return out.round(2)


def distance_tests(cells: pd.DataFrame) -> pd.DataFrame:
    """Marker⁺ vs marker⁻ distance-to-nerve: pooled MWU + per-field consistency."""
    b = cells[cells["distance_to_nerve_px"].notna()].copy()
    markers = [c[:-4] for c in b.columns if c.endswith("_pos") and b[c].notna().any()]
    rows: List[Dict[str, Any]] = []
    for m in markers:
        pos = b.loc[_to_bool(b[f"{m}_pos"]), "distance_to_nerve_px"]
        neg = b.loc[~_to_bool(b[f"{m}_pos"]), "distance_to_nerve_px"]
        if len(pos) < 5 or len(neg) < 5:
            continue
        U, p = mannwhitneyu(pos, neg, alternative="two-sided")
        # per-field: does marker⁺ have a smaller median distance than marker⁻?
        closer = 0; nfields = 0
        for _, g in b.groupby("field_id"):
            gp = g.loc[_to_bool(g[f"{m}_pos"]), "distance_to_nerve_px"]
            gn = g.loc[~_to_bool(g[f"{m}_pos"]), "distance_to_nerve_px"]
            if len(gp) >= 3 and len(gn) >= 3:
                nfields += 1
                closer += int(gp.median() < gn.median())
        rows.append({"marker": m, "n_pos": len(pos), "n_neg": len(neg),
                     "median_pos": round(pos.median(), 1), "median_neg": round(neg.median(), 1),
                     "median_diff": round(pos.median() - neg.median(), 1),
                     "mannwhitney_U": float(U), "p_pooled": p,
                     "fields_pos_closer": closer, "fields_tested": nfields})
    df = pd.DataFrame(rows)
    if not df.empty:
        from statsmodels.stats.multitest import multipletests
        df["p_fdr"] = multipletests(df["p_pooled"], method="fdr_bh")[1]
        df["p_pooled"] = df["p_pooled"].map(lambda x: f"{x:.2e}")
        df["p_fdr"] = df["p_fdr"].map(lambda x: f"{x:.2e}")
    return df


def run(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    tables_dir = Path(cfg["paths"]["masks_dir"]).parent / "tables"
    cells = pd.read_csv(tables_dir / "cells_spatial.csv", dtype={"sample": str})
    nerve = pd.read_csv(tables_dir / "nerve_metrics.csv") if (tables_dir / "nerve_metrics.csv").exists() else pd.DataFrame()

    per_img = per_image_summary(cells, nerve)
    per_smp = per_sample_summary(per_img)
    tests = distance_tests(cells)

    per_img.to_csv(tables_dir / "per_image_summary.csv", index=False)
    per_smp.to_csv(tables_dir / "per_sample_summary.csv", index=False)
    tests.to_csv(tables_dir / "internal_tests.csv", index=False)

    print("Per-image summary:")
    print(per_img.to_string(index=False))
    print("\nInternal test — marker⁺ vs marker⁻ distance-to-nerve (exploratory, cell-level):")
    print(tests.to_string(index=False) if not tests.empty else "  (none)")
    print("\n⚠️ Descriptive only (no control group). Pooled MWU is cell-level (pseudoreplication);"
          "\n   'fields_pos_closer' is the replicate-level consistency check.")
    return {"per_image": per_img, "per_sample": per_smp, "tests": tests}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 10: aggregation + descriptive stats.")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    run(cfg)
    print(f"\nStage 10 done. Tables -> {Path(cfg['paths']['masks_dir']).parent / 'tables'} "
          "(per_image_summary.csv, per_sample_summary.csv, internal_tests.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
