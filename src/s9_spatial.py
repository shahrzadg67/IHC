"""Stage 9 - Immune–nerve spatial analysis (Panel B).

For every cell, computes the distance (px) from its centroid to the nearest nerve pixel
(distance transform of the Stage-8 nerve mask), flags "nerve-associated" cells within a
config distance, summarizes distances per phenotype, and — as a control-free internal test —
runs squidpy neighborhood enrichment (observed vs a permutation null) on the phenotypes.

Distances are in PIXELS until the pixel size (µm/px) is known (plan Sec.7 Q2).

CLI:  python -m src.s9_spatial --config config/config.yaml
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

from src.s0_config import load_config
from src.utils.io import read_tiff
from src.s7_phenotype import panel_markers, phenotype_label


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "1.0"])


def distance_to_nerve(nerve: np.ndarray, cy: np.ndarray, cx: np.ndarray) -> np.ndarray:
    """Distance (px) from each centroid to the nearest nerve pixel."""
    if nerve.sum() == 0:
        return np.full(cy.shape, np.nan)
    dist = ndi.distance_transform_edt(~nerve.astype(bool))
    H, W = nerve.shape
    yy = np.clip(np.round(cy).astype(int), 0, H - 1)
    xx = np.clip(np.round(cx).astype(int), 0, W - 1)
    return dist[yy, xx]


def run(cfg: Dict[str, Any]) -> pd.DataFrame:
    scfg = cfg.get("spatial", {})
    assoc_px = float(scfg.get("association_distance_px", 20))
    masks_dir = Path(cfg["paths"]["masks_dir"])
    tables_dir = masks_dir.parent / "tables"
    qc_dir = masks_dir.parent / "qc_quant"

    cells = pd.read_csv(tables_dir / "cells_positive.csv", dtype={"sample": str})
    cells["distance_to_nerve_px"] = np.nan
    cells["nerve_associated"] = pd.NA
    cells["phenotype"] = pd.NA

    # Fields that have a nerve mask (Panel B).
    nerve_fields = sorted(p.name.replace("__nerve.tif", "")
                          for p in masks_dir.glob("*__nerve.tif"))
    if not nerve_fields:
        raise SystemExit("No nerve masks found (run Stage 8 first).")

    for fid in nerve_fields:
        nerve = read_tiff(masks_dir / f"{fid}__nerve.tif").astype(bool)
        sel = cells["field_id"] == fid
        d = distance_to_nerve(nerve, cells.loc[sel, "centroid_y"].to_numpy(),
                              cells.loc[sel, "centroid_x"].to_numpy())
        cells.loc[sel, "distance_to_nerve_px"] = d
        cells.loc[sel, "nerve_associated"] = d <= assoc_px

    # Phenotype label per Panel-B cell (reuse Stage 7 logic). Determine the panel's
    # markers from the RAW columns (absent markers are all-NA -> correctly excluded);
    # only then coerce the present markers to clean booleans.
    panelB = cells[cells["panel"] == "B"].copy()
    markers = panel_markers(panelB)
    for m in markers:
        panelB[f"{m}_pos"] = _to_bool(panelB[f"{m}_pos"])
    cells.loc[cells["panel"] == "B", "phenotype"] = phenotype_label(panelB, markers).to_numpy()

    cells.to_csv(tables_dir / "cells_spatial.csv", index=False)

    # Distance summary per phenotype (Panel B).
    b = cells[cells["panel"] == "B"].copy()
    summ = (b.groupby("phenotype")["distance_to_nerve_px"]
            .agg(n="count", mean_dist="mean", median_dist="median").round(1))
    summ["pct_nerve_associated"] = (b.groupby("phenotype")["nerve_associated"]
                                    .apply(lambda s: 100.0 * s.astype(float).mean()).round(1))
    summ = summ.reset_index()
    summ.to_csv(tables_dir / "distance_by_phenotype.csv", index=False)
    print("Distance-to-nerve by phenotype (px):")
    print(summ.to_string(index=False))

    # Per-image nerve-association %.
    per_img = (b.groupby("field_id")["nerve_associated"]
               .apply(lambda s: round(100.0 * s.astype(float).mean(), 1)).reset_index(name="pct_nerve_associated"))
    per_img.to_csv(tables_dir / "nerve_association_per_image.csv", index=False)

    _distance_histograms(b, markers, assoc_px, qc_dir)
    if scfg.get("run_squidpy", True):
        _neighborhood_enrichment(b, int(scfg.get("n_neighs", 6)), tables_dir, qc_dir)
    return summ


def _distance_histograms(b: pd.DataFrame, markers: List[str], assoc_px: float, qc_dir: Path) -> None:
    """Distance-to-nerve distributions split by each marker's positivity (the key internal test)."""
    fig, axes = plt.subplots(1, len(markers), figsize=(4.5 * len(markers), 3.2), squeeze=False)
    bins = np.linspace(0, np.nanpercentile(b["distance_to_nerve_px"], 99), 40)
    for ax, m in zip(axes[0], markers):
        pos = b.loc[_to_bool(b[f"{m}_pos"]), "distance_to_nerve_px"].dropna()
        neg = b.loc[~_to_bool(b[f"{m}_pos"]), "distance_to_nerve_px"].dropna()
        if len(neg):
            ax.hist(neg, bins=bins, alpha=0.6, density=True, label=f"{m}-", color="#9aa7b5")
        if len(pos):
            ax.hist(pos, bins=bins, alpha=0.6, density=True, label=f"{m}+", color="#cc5555")
        ax.axvline(assoc_px, color="k", ls="--", lw=1, label=f"assoc {assoc_px:.0f}px")
        mp = pos.median() if len(pos) else float("nan")
        mn = neg.median() if len(neg) else float("nan")
        ax.set_title(f"{m}: dist-to-nerve (med {mp:.0f} vs {mn:.0f})", fontsize=9)
        ax.set_xlabel("distance to nerve (px)"); ax.legend(fontsize=7)
    fig.tight_layout()
    qc_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(qc_dir / "distance_to_nerve_B.png", dpi=120); plt.close(fig)


def _neighborhood_enrichment(b: pd.DataFrame, n_neighs: int, tables_dir: Path, qc_dir: Path) -> None:
    """squidpy neighborhood enrichment (observed vs permutation null) on phenotypes."""
    try:
        import anndata as ad
        import squidpy as sq
        import seaborn as sns
        sub = b.dropna(subset=["distance_to_nerve_px"]).copy()
        adata = ad.AnnData(X=np.zeros((len(sub), 1), dtype=np.float32))
        adata.obs["phenotype"] = pd.Categorical(sub["phenotype"].astype(str).to_numpy())
        adata.obs["field_id"] = pd.Categorical(sub["field_id"].astype(str).to_numpy())
        adata.obsm["spatial"] = sub[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
        sq.gr.spatial_neighbors(adata, library_key="field_id", coord_type="generic", n_neighs=n_neighs)
        sq.gr.nhood_enrichment(adata, cluster_key="phenotype", seed=0, show_progress_bar=False)

        z = adata.uns["phenotype_nhood_enrichment"]["zscore"]
        cats = list(adata.obs["phenotype"].cat.categories)
        zdf = pd.DataFrame(z, index=cats, columns=cats)
        zdf.to_csv(tables_dir / "nhood_enrichment_B.csv")
        fig, ax = plt.subplots(figsize=(1.1 * len(cats) + 2, 1.0 * len(cats) + 1.5))
        sns.heatmap(zdf, annot=True, fmt=".1f", cmap="coolwarm", center=0, ax=ax,
                    cbar_kws={"label": "enrichment z-score"})
        ax.set_title("Panel B — phenotype neighborhood enrichment (z vs permutation null)", fontsize=9)
        fig.tight_layout(); fig.savefig(qc_dir / "nhood_enrichment_B.png", dpi=120); plt.close(fig)
        print(f"\nsquidpy neighborhood enrichment written ({len(cats)} phenotypes).")
    except Exception as exc:  # squidpy optional / may fail on tiny categories
        print(f"\n[squidpy] neighborhood enrichment skipped: {exc}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 9: immune–nerve spatial analysis.")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run(cfg)
    print(f"\nStage 9 done. Tables -> {Path(cfg['paths']['masks_dir']).parent / 'tables'} "
          f"(cells_spatial.csv, distance_by_phenotype.csv, nerve_association_per_image.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
