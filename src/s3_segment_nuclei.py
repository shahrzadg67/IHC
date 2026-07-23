"""Stage 3 - Nuclear segmentation with Cellpose-SAM.

Runs Cellpose-SAM on each field's DAPI grayscale image (from Stage 1) to produce
integer nucleus label masks, saves them to `outputs/masks/{field_id}__nuclei.tif`,
writes a QC overlay per field, and records per-field nucleus counts.

Needs a GPU in practice -> run via the sbatch job (jobs/run_segmentation.sbatch),
not on the login node.

CLI:  python -m src.s3_segment_nuclei --config config/config.yaml [--fields A_CFA_40x B_1_40x]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.s0_config import load_config
from src.utils.io import read_tiff, write_tiff
from src.utils.viz import overlay_labels


def build_model(seg_cfg: Dict[str, Any]):
    """Instantiate a Cellpose-SAM model; log whether a GPU is actually in use."""
    from cellpose import core, models

    want_gpu = bool(seg_cfg.get("gpu", True))
    gpu_ok = core.use_gpu() if want_gpu else False
    print(f"[cellpose] requested gpu={want_gpu} | GPU available/used={gpu_ok}")

    pretrained = seg_cfg.get("pretrained_model")
    if pretrained:  # explicit model name / path
        model = models.CellposeModel(gpu=gpu_ok, pretrained_model=pretrained)
        print(f"[cellpose] model = {pretrained}")
    else:  # built-in Cellpose-SAM default (cpsam_v2 in cellpose 4.2.x)
        model = models.CellposeModel(gpu=gpu_ok)
        name = Path(str(getattr(model, "pretrained_model", "") or "cpsam_v2")).name
        print(f"[cellpose] model = built-in Cellpose-SAM default ({name})")
    return model


def eval_kwargs_from_cfg(seg_cfg: Dict[str, Any]) -> Dict[str, Any]:
    kw: Dict[str, Any] = dict(
        batch_size=int(seg_cfg.get("batch_size", 8)),
        flow_threshold=float(seg_cfg.get("flow_threshold", 0.4)),
        cellprob_threshold=float(seg_cfg.get("cellprob_threshold", 0.0)),
        normalize=bool(seg_cfg.get("normalize", True)),
    )
    if seg_cfg.get("diameter") is not None:
        kw["diameter"] = float(seg_cfg["diameter"])
    if seg_cfg.get("min_size") is not None:
        kw["min_size"] = int(seg_cfg["min_size"])
    return kw


def dapi_fields(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Return manifest rows for the DAPI channel of each (non-excluded) field."""
    manifest = pd.read_csv(cfg["paths"]["manifest"], dtype={"sample": str})
    nuclear = cfg.get("nuclear_marker", "DAPI")
    return manifest[(~manifest["exclude"].astype(bool)) &
                    (manifest["marker"] == nuclear)].copy()


def run(cfg: Dict[str, Any], only_fields: Optional[List[str]] = None) -> pd.DataFrame:
    seg_cfg = cfg.get("segmentation", {})
    gray_dir = Path(cfg["paths"]["gray_dir"])
    masks_dir = Path(cfg["paths"]["masks_dir"])
    qc_seg_dir = Path(cfg["paths"]["qc_seg_dir"])

    rows = dapi_fields(cfg)
    if only_fields:
        rows = rows[rows["field_id"].isin(only_fields)]
    if rows.empty:
        raise SystemExit("No DAPI fields to segment (check manifest / --fields).")

    model = build_model(seg_cfg)
    kw = eval_kwargs_from_cfg(seg_cfg)
    print(f"[cellpose] eval kwargs = {kw}\n")

    report: List[Dict[str, Any]] = []
    for _, r in rows.iterrows():
        fid = str(r["field_id"])
        gray_path = gray_dir / f"{fid}__DAPI.tif"
        img = read_tiff(gray_path)

        t0 = time.time()
        out = model.eval(img, **kw)          # (masks, flows, styles) in Cellpose v4
        masks = np.asarray(out[0])
        dt = time.time() - t0

        n = int(masks.max())
        mask_path = masks_dir / f"{fid}__nuclei.tif"
        write_tiff(mask_path, masks.astype(np.uint16))
        overlay_labels(img, masks, f"{fid} — nuclei", qc_seg_dir / f"{fid}__nuclei.png")

        report.append({"field_id": fid, "n_nuclei": n, "seconds": round(dt, 1),
                       "mask": mask_path.name})
        print(f"  {fid:12s} -> {n:5d} nuclei  ({dt:.1f}s)")

    df = pd.DataFrame(report)
    tables_dir = masks_dir.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "nuclei_counts.csv", index=False)
    return df


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 3: Cellpose-SAM nuclear segmentation.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--fields", nargs="*", default=None,
                    help="limit to these field_ids (e.g. for tuning); default = all")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    df = run(cfg, only_fields=args.fields)
    print(f"\nStage 3 done: {len(df)} fields, {int(df['n_nuclei'].sum())} nuclei total.")
    print(f"Masks -> {cfg['paths']['masks_dir']} | QC overlays -> {cfg['paths']['qc_seg_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
