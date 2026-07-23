"""Stage 5 - Per-cell feature extraction -> single-cell table (+ AnnData).

For every field, measures per cell: centroid, morphology (area, eccentricity, solidity,
perimeter) and, for each marker channel, the mean / median / integrated intensity inside
the chosen compartment (expanded cell body by default; config `features.compartment`).
Rows are cells; metadata columns (panel, sample, field, magnification, condition, animal)
are carried through. Writes `outputs/tables/cells.csv` and one AnnData per panel
(`adata_<panel>.h5ad`) for the downstream spatial analysis (Stage 9).

CLI:  python -m src.s5_features --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops_table

from src.s0_config import load_config
from src.utils.io import read_tiff

MORPHOLOGY = ["label", "centroid", "area", "eccentricity", "solidity", "perimeter"]


def field_markers(manifest: pd.DataFrame, field_id: str) -> List[str]:
    """Marker channels present for a field (non-overlay), e.g. [DAPI, F4-80, PGP9.5, SYK]."""
    m = manifest[(manifest["field_id"] == field_id) & (~manifest["exclude"].astype(bool))]
    return list(m["marker"].unique())


def field_metadata(manifest: pd.DataFrame, field_id: str) -> Dict[str, Any]:
    r = manifest[manifest["field_id"] == field_id].iloc[0]
    return {"panel": r["panel"], "sample": str(r["sample"]),
            "magnification": r["magnification"], "condition": r.get("group", ""),
            "animal": r.get("animal", "")}


def measure_field(cfg: Dict[str, Any], manifest: pd.DataFrame, field_id: str) -> pd.DataFrame:
    """Return a per-cell DataFrame for one field."""
    compartment = cfg.get("features", {}).get("compartment", "cell")
    masks_dir = Path(cfg["paths"]["masks_dir"])
    gray_dir = Path(cfg["paths"]["gray_dir"])
    suffix = "cells" if compartment == "cell" else "nuclei"
    mask = read_tiff(masks_dir / f"{field_id}__{suffix}.tif")

    # Morphology (one row per labelled object).
    props = regionprops_table(mask, properties=MORPHOLOGY)
    df = pd.DataFrame(props).rename(columns={
        "centroid-0": "centroid_y", "centroid-1": "centroid_x"})
    labels = df["label"].to_numpy()
    if labels.size == 0:
        return df

    # Per-marker intensity inside the compartment (mean / median / integrated).
    for marker in field_markers(manifest, field_id):
        gray = read_tiff(gray_dir / f"{field_id}__{marker}.tif").astype(np.float64)
        df[f"{marker}_mean"] = ndi.mean(gray, mask, index=labels)
        df[f"{marker}_median"] = ndi.median(gray, mask, index=labels)
        df[f"{marker}_integrated"] = ndi.sum(gray, mask, index=labels)

    meta = field_metadata(manifest, field_id)
    df.insert(0, "cell_id", [f"{field_id}_{int(l)}" for l in labels])
    df.insert(1, "field_id", field_id)
    for k, v in meta.items():
        df[k] = v
    df["compartment"] = compartment
    return df


def build_anndata(cells: pd.DataFrame, panel: str, out_path: Path) -> Optional[Path]:
    """Write a per-panel AnnData: X = per-marker mean intensity, obsm['spatial'] = centroids."""
    import anndata as ad

    sub = cells[cells["panel"] == panel].copy()
    markers = [c[:-5] for c in sub.columns if c.endswith("_mean")]
    markers = [m for m in markers if sub[f"{m}_mean"].notna().any()]
    if sub.empty or not markers:
        return None
    X = sub[[f"{m}_mean" for m in markers]].to_numpy(dtype=np.float32)
    obs = sub[["cell_id", "field_id", "sample", "magnification", "condition", "animal",
               "area", "eccentricity", "solidity", "perimeter"]].reset_index(drop=True)
    obs.index = sub["cell_id"].to_numpy().astype(str)
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = markers
    adata.obsm["spatial"] = sub[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    return out_path


def run(cfg: Dict[str, Any]) -> pd.DataFrame:
    manifest = pd.read_csv(cfg["paths"]["manifest"], dtype={"sample": str})
    fields = sorted(manifest.loc[~manifest["exclude"].astype(bool), "field_id"].dropna().unique())

    per_field = [measure_field(cfg, manifest, fid) for fid in fields]
    cells = pd.concat(per_field, ignore_index=True)

    tables_dir = Path(cfg["paths"]["masks_dir"]).parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(tables_dir / "cells.csv", index=False)

    for panel in sorted(cells["panel"].dropna().unique()):
        out = build_anndata(cells, panel, tables_dir / f"adata_{panel}.h5ad")
        n = int((cells["panel"] == panel).sum())
        print(f"  panel {panel}: {n} cells -> {out.name if out else '(no AnnData)'}")

    return cells


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 5: per-cell feature extraction.")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cells = run(cfg)
    intensity_cols = [c for c in cells.columns if c.endswith("_mean")]
    print(f"\nStage 5 done: {len(cells)} cells across {cells['field_id'].nunique()} fields.")
    print(f"Markers measured: {[c[:-5] for c in intensity_cols]}")
    print(f"Table -> {Path(cfg['paths']['masks_dir']).parent / 'tables' / 'cells.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
