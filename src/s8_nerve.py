"""Stage 8 - Nerve (PGP9.5) analysis.

PGP9.5 is fiber-like, so it is quantified as a **field**, not as cells: threshold the
PGP9.5 grayscale to a nerve mask, report the **area fraction** (% of field covered), and
optionally **skeletonize** for total fiber length and branch count. The nerve mask is
saved for Stage 9 (immune–nerve distances). Panel B fields only (they carry PGP9.5).

CLI:  python -m src.s8_nerve --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu, threshold_triangle
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries

from src.s0_config import load_config
from src.utils.io import read_tiff, write_tiff


def nerve_mask(gray: np.ndarray, method: str, smooth_sigma: float, min_object_px: int) -> np.ndarray:
    """Threshold a PGP9.5 grayscale image into a cleaned boolean nerve mask."""
    img = gaussian(gray.astype(np.float64), sigma=smooth_sigma) if smooth_sigma else gray.astype(np.float64)
    thr = threshold_triangle(img) if method == "triangle" else threshold_otsu(img)
    mask = img > thr
    if min_object_px:  # drop connected components smaller than min_object_px (version-robust)
        lab, _ = ndi.label(mask)
        counts = np.bincount(lab.ravel())
        too_small = counts < int(min_object_px)
        too_small[0] = False  # never remove background
        mask = mask & ~too_small[lab]
    return mask


def skeleton_metrics(mask: np.ndarray) -> Dict[str, Any]:
    """Skeleton length (px) and branch count (via skan if available)."""
    skel = skeletonize(mask)
    out: Dict[str, Any] = {"skeleton_length_px": int(skel.sum()), "n_branches": np.nan}
    if skel.sum() == 0:
        return out
    try:
        from skan import Skeleton, summarize
        summary = summarize(Skeleton(skel), separator="-")
        out["n_branches"] = int(len(summary))
    except Exception:
        pass  # skan optional; length is still reported
    return out


def nerve_overlay(gray: np.ndarray, mask: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(gray, cmap="gray")
    ov = np.zeros((*gray.shape, 4))
    ov[find_boundaries(mask, mode="outer")] = (1.0, 0.2, 0.2, 1.0)  # red nerve outline
    ax.imshow(ov)
    ax.set_title(title, fontsize=11); ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


def run(cfg: Dict[str, Any]) -> pd.DataFrame:
    ncfg = cfg.get("nerve", {})
    marker = ncfg.get("marker", "PGP9.5")
    gray_dir = Path(cfg["paths"]["gray_dir"])
    masks_dir = Path(cfg["paths"]["masks_dir"])
    qc_dir = Path(cfg["paths"]["masks_dir"]).parent / "qc_quant"

    manifest = pd.read_csv(cfg["paths"]["manifest"], dtype={"sample": str})
    fields = sorted(manifest.loc[(manifest["marker"] == marker) &
                                 (~manifest["exclude"].astype(bool)), "field_id"].unique())
    if not fields:
        raise SystemExit(f"No fields carry the nerve marker '{marker}'.")

    report: List[Dict[str, Any]] = []
    for fid in fields:
        gray = read_tiff(gray_dir / f"{fid}__{marker}.tif")
        mask = nerve_mask(gray, ncfg.get("threshold_method", "otsu"),
                          float(ncfg.get("smooth_sigma", 1.0)),
                          int(ncfg.get("min_object_px", 30)))
        write_tiff(masks_dir / f"{fid}__nerve.tif", mask.astype(np.uint8))

        rec = {"field_id": fid, "area_fraction_pct": round(100.0 * mask.mean(), 2),
               "nerve_objects": int(ndi.label(mask)[1])}
        if ncfg.get("skeleton", True):
            rec.update(skeleton_metrics(mask))
        report.append(rec)

        nerve_overlay(gray, mask, f"{fid} — {marker} nerve mask ({rec['area_fraction_pct']}% area)",
                      qc_dir / f"nerve_{fid}.png")
        print(f"  {fid:12s}: {rec['area_fraction_pct']:5.2f}% area | "
              f"{rec['nerve_objects']} objects | skel {rec.get('skeleton_length_px','-')}px")

    df = pd.DataFrame(report)
    tables_dir = masks_dir.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "nerve_metrics.csv", index=False)
    return df


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 8: PGP9.5 nerve area / skeleton.")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    df = run(cfg)
    print(f"\nStage 8 done: {len(df)} nerve fields. Metrics -> "
          f"{Path(cfg['paths']['masks_dir']).parent / 'tables' / 'nerve_metrics.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
