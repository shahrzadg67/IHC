"""Stage 4 - Cell-body approximation.

No membrane/cytoplasm marker is present, so cell bodies are approximated by expanding
each nucleus label outward a fixed distance (magnification-dependent, from config) into
neighbouring background without overlapping other cells. Both nucleus and cell masks are
kept; downstream feature extraction can measure each marker in whichever compartment is
biologically appropriate (e.g. surface markers in the expanded ring).

CLI:  python -m src.s4_cell_bodies --config config/config.yaml [--fields B_1_40x]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from skimage.segmentation import expand_labels

from src.s0_config import load_config
from src.utils.io import read_tiff, write_tiff
from src.utils.viz import overlay_labels


def field_magnification(cfg: Dict[str, Any], field_id: str) -> str:
    """Look up a field's magnification from the manifest (for the expansion distance)."""
    manifest = pd.read_csv(cfg["paths"]["manifest"], dtype={"sample": str})
    mags = manifest.loc[manifest["field_id"] == field_id, "magnification"].unique()
    return str(mags[0]) if len(mags) else "40x"


def run(cfg: Dict[str, Any], only_fields: Optional[List[str]] = None) -> pd.DataFrame:
    masks_dir = Path(cfg["paths"]["masks_dir"])
    qc_seg_dir = Path(cfg["paths"]["qc_seg_dir"])
    gray_dir = Path(cfg["paths"]["gray_dir"])
    dist_map = cfg.get("cell_expansion", {}).get("distance_px", {})

    nuclei_masks = sorted(masks_dir.glob("*__nuclei.tif"))
    if only_fields:
        nuclei_masks = [p for p in nuclei_masks
                        if p.name.replace("__nuclei.tif", "") in only_fields]
    if not nuclei_masks:
        raise SystemExit("No nucleus masks found (run Stage 3 first).")

    report: List[Dict[str, Any]] = []
    for mp in nuclei_masks:
        fid = mp.name.replace("__nuclei.tif", "")
        nuclei = read_tiff(mp)
        mag = field_magnification(cfg, fid)
        dist = int(dist_map.get(mag, 6))

        cells = expand_labels(nuclei, distance=dist)
        cell_path = masks_dir / f"{fid}__cells.tif"
        write_tiff(cell_path, cells.astype(np.uint16))

        # QC overlay: nuclei (yellow) + expanded cell bodies (cyan) on DAPI.
        dapi = read_tiff(gray_dir / f"{fid}__DAPI.tif")
        overlay_labels(dapi, nuclei, f"{fid} — nuclei + cells (expand {dist}px @ {mag})",
                       qc_seg_dir / f"{fid}__cells.png", second_labels=cells)

        report.append({"field_id": fid, "magnification": mag, "expand_px": dist,
                       "n_cells": int(cells.max()), "cells_mask": cell_path.name})
        print(f"  {fid:12s} expand {dist}px @ {mag} -> {int(cells.max()):5d} cells")

    df = pd.DataFrame(report)
    tables_dir = masks_dir.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "cell_counts.csv", index=False)
    return df


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 4: expand nuclei into cell bodies.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--fields", nargs="*", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    df = run(cfg, only_fields=args.fields)
    print(f"\nStage 4 done: {len(df)} fields. Cell masks -> {cfg['paths']['masks_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
