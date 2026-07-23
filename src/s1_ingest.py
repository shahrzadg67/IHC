"""Stage 1 - Ingest & channel extraction.

For every non-overlay file in the manifest, extract a single-channel 2-D grayscale
intensity image and save it as `outputs/gray/{field_id}__{marker}.tif`. For the
current 8-bit RGB exports we max-project across R/G/B (robust to single-colour and
yellow encodings). A per-field QC contact sheet (all marker grayscales, labelled)
is written to `outputs/qc/` so nuclei/signal can be eyeballed.

CLI:  python -m src.s1_ingest --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless / no-display cluster node
import matplotlib.pyplot as plt
import pandas as pd

from src.s0_config import load_config
from src.utils.io import nonzero_fraction, read_tiff, rgb_max_project, write_tiff


def extract_gray(rgb, input_mode: str):
    """Return a 2-D grayscale intensity image according to the input mode."""
    if input_mode == "rgb_export":
        return rgb_max_project(rgb)
    if input_mode in ("single_channel", "ome_tiff"):
        raise NotImplementedError(
            f"input_mode='{input_mode}' (raw 16-bit path) is planned for milestone M6; "
            "M1 implements 'rgb_export' only.")
    raise ValueError(f"Unknown input_mode '{input_mode}'")


def _safe_name(s: str) -> str:
    """Filesystem-safe token for output filenames."""
    return str(s).replace("/", "-").replace(" ", "_")


def write_qc_contact_sheet(field_id: str, entries: List[Dict[str, Any]], qc_dir: Path) -> Path:
    """Write one PNG per field: each marker's grayscale image side by side."""
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), squeeze=False)
    for ax, e in zip(axes[0], entries):
        ax.imshow(e["image"], cmap="gray")
        ax.set_title(f"{e['marker']}\nnz={e['nonzero_frac']:.1%}", fontsize=9)
        ax.axis("off")
    fig.suptitle(field_id, fontsize=11)
    fig.tight_layout()
    out = qc_dir / f"{_safe_name(field_id)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def run(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Execute Stage 1; returns a small report DataFrame of what was written."""
    input_mode = cfg.get("input_mode", "rgb_export")
    manifest = pd.read_csv(cfg["paths"]["manifest"], dtype={"sample": str})
    gray_dir = Path(cfg["paths"]["gray_dir"])
    qc_dir = Path(cfg["paths"]["qc_dir"])

    todo = manifest[~manifest["exclude"].astype(bool)].copy()
    report_rows: List[Dict[str, Any]] = []
    per_field: Dict[str, List[Dict[str, Any]]] = {}

    for _, row in todo.iterrows():
        rgb = read_tiff(row["file_path"])
        gray = extract_gray(rgb, input_mode)
        field_id = str(row["field_id"])
        marker = str(row["marker"])
        out_path = gray_dir / f"{_safe_name(field_id)}__{_safe_name(marker)}.tif"
        write_tiff(out_path, gray)

        nz = nonzero_fraction(gray)
        report_rows.append({
            "field_id": field_id, "marker": marker,
            "shape": "x".join(map(str, gray.shape)), "dtype": str(gray.dtype),
            "nonzero_frac": round(nz, 4), "out": out_path.name,
        })
        per_field.setdefault(field_id, []).append(
            {"marker": marker, "image": gray, "nonzero_frac": nz})

    # QC contact sheet per field (markers ordered with DAPI first for easy review).
    for field_id, entries in per_field.items():
        entries.sort(key=lambda e: (e["marker"] != "DAPI", e["marker"]))
        write_qc_contact_sheet(field_id, entries, qc_dir)

    return pd.DataFrame(report_rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1: RGB -> grayscale extraction + QC.")
    ap.add_argument("--config", default="config/config.yaml", help="path to config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    report = run(cfg)

    print(f"\nStage 1: wrote {len(report)} grayscale images -> {cfg['paths']['gray_dir']}")
    print(f"         wrote {report['field_id'].nunique()} QC contact sheets -> {cfg['paths']['qc_dir']}\n")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(report.sort_values(["field_id", "marker"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
