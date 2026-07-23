"""run_all.py - reproduce the whole IF pipeline in one command (plan Stage 12).

Runs Stage 0 → 11 in order, timing each stage and writing a provenance snapshot
(timestamp + package versions + config) for reproducibility.

Segmentation (Stages 3–4) uses Cellpose and wants a GPU: run this on a GPU node
(`sbatch jobs/run_all.sbatch`) or pass --skip-segmentation to reuse existing masks
(e.g. after `sbatch jobs/run_segmentation.sbatch`).

The manifest is treated as human-reviewed: Stage 0 only rebuilds it when it is
missing, or when --rebuild-manifest is given (which would overwrite manual edits).

Usage:
    python run_all.py --config config/config.yaml
    python run_all.py --skip-segmentation          # CPU-only; reuse masks
    python run_all.py --rebuild-manifest            # regenerate manifest (drops edits)
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as md
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

from src.s0_config import build_manifest, load_config, validate_manifest
from src import (s1_ingest, s3_segment_nuclei, s4_cell_bodies, s5_features,
                 s6_positivity, s7_phenotype, s8_nerve, s9_spatial,
                 s10_aggregate_stats, s11_report)

_PKGS = ["numpy", "pandas", "tifffile", "scikit-image", "scipy", "torch", "cellpose",
         "anndata", "scikit-learn", "squidpy", "skan", "statsmodels", "matplotlib"]


def stage0(cfg, rebuild: bool) -> None:
    """Build/validate the manifest — but preserve a reviewed one unless --rebuild-manifest."""
    manifest = Path(cfg["paths"]["manifest"])
    if manifest.exists() and not rebuild:
        print(f"  manifest exists, kept as-is ({manifest}); use --rebuild-manifest to regenerate")
        return
    df = build_manifest(cfg)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)
    warns = validate_manifest(df, cfg)
    print(f"  wrote manifest ({len(df)} files); {len(warns)} validation warning(s)")


def write_provenance(cfg, out_dir: Path) -> None:
    versions = {}
    for p in _PKGS:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = "—"
    lines = [f"IF pipeline run — {dt.datetime.now().isoformat(timespec='seconds')}",
             f"python {sys.version.split()[0]}", "", "package versions:"]
    lines += [f"  {k:14s} {v}" for k, v in versions.items()]
    lines += ["", "config:", Path(cfg["_repo_root"], "config", "config.yaml").read_text()]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_provenance.txt").write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the whole IF pipeline (Stage 0–11).")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--skip-segmentation", action="store_true",
                    help="skip Stages 3–4 (reuse existing masks; CPU-only run)")
    ap.add_argument("--rebuild-manifest", action="store_true",
                    help="regenerate the manifest (overwrites manual edits)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    stages: List[Tuple[str, Callable]] = [
        ("0  manifest", lambda: stage0(cfg, args.rebuild_manifest)),
        ("1  ingest", lambda: s1_ingest.run(cfg)),
    ]
    if not args.skip_segmentation:
        stages += [("3  segment (GPU)", lambda: s3_segment_nuclei.run(cfg)),
                   ("4  cell bodies", lambda: s4_cell_bodies.run(cfg))]
    stages += [
        ("5  features", lambda: s5_features.run(cfg)),
        ("6  positivity", lambda: s6_positivity.run(cfg)),
        ("7  phenotype", lambda: s7_phenotype.run(cfg)),
        ("8  nerve", lambda: s8_nerve.run(cfg)),
        ("9  spatial", lambda: s9_spatial.run(cfg)),
        ("10 aggregate/stats", lambda: s10_aggregate_stats.run(cfg)),
        ("11 report", lambda: s11_report.run(cfg)),
    ]

    print(f"=== IF pipeline: {len(stages)} stages "
          f"({'skip' if args.skip_segmentation else 'incl.'} segmentation) ===")
    t_all = time.time()
    for name, fn in stages:
        print(f"\n--- Stage {name} ---")
        t0 = time.time()
        fn()
        print(f"    [{time.time() - t0:.1f}s]")

    out_dir = Path(cfg["paths"]["masks_dir"]).parent
    write_provenance(cfg, out_dir)
    print(f"\n=== DONE in {time.time() - t_all:.1f}s. "
          f"Report: {out_dir / 'report' / 'report.html'} | provenance: {out_dir / 'run_provenance.txt'} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
