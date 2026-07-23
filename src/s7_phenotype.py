"""Stage 7 - Co-expression / phenotyping.

From the per-cell positivity calls (Stage 6), assigns each cell a phenotype label
(the combination of markers it is positive for), tabulates per-image phenotype
composition, builds a marker co-occurrence matrix + heatmap per panel, and optionally
applies named-phenotype rules from config.

CLI:  python -m src.s7_phenotype --config config/config.yaml
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
import seaborn as sns

from src.s0_config import load_config


def panel_markers(cells_panel: pd.DataFrame) -> List[str]:
    """Positivity markers that actually have calls in this panel subset."""
    return sorted(c[:-4] for c in cells_panel.columns
                  if c.endswith("_pos") and cells_panel[c].notna().any())


def phenotype_label(bools: pd.DataFrame, markers: List[str]) -> pd.Series:
    """Per-cell label = '/'.join of markers the cell is positive for (or 'none')."""
    def lab(row):
        pos = [m for m in markers if bool(row[f"{m}_pos"])]
        return "/".join(f"{m}+" for m in pos) if pos else "none"
    return bools.apply(lab, axis=1)


def apply_named_rules(bools: pd.DataFrame, markers: List[str],
                      rules: List[Dict[str, Any]]) -> pd.Series:
    """Assign the first matching named phenotype rule (else 'unclassified')."""
    def classify(row):
        for rule in rules:
            all_of = rule.get("all_of", [])
            any_of = rule.get("any_of", [])
            none_of = rule.get("none_of", [])
            if all(bool(row.get(f"{m}_pos", False)) for m in all_of) and \
               (not any_of or any(bool(row.get(f"{m}_pos", False)) for m in any_of)) and \
               all(not bool(row.get(f"{m}_pos", False)) for m in none_of):
                return rule["name"]
        return "unclassified"
    return bools.apply(classify, axis=1)


def cooccurrence(sub: pd.DataFrame, markers: List[str]) -> pd.DataFrame:
    """Marker x marker matrix: diagonal = % positive, off-diagonal = % double-positive."""
    M = len(markers)
    mat = np.zeros((M, M))
    for i, mi in enumerate(markers):
        pi = sub[f"{mi}_pos"].fillna(False).astype(bool)
        for j, mj in enumerate(markers):
            pj = sub[f"{mj}_pos"].fillna(False).astype(bool)
            mat[i, j] = 100.0 * (pi & pj).mean() if i != j else 100.0 * pi.mean()
    return pd.DataFrame(mat, index=markers, columns=markers)


def run(cfg: Dict[str, Any]) -> pd.DataFrame:
    tables_dir = Path(cfg["paths"]["masks_dir"]).parent / "tables"
    qc_dir = Path(cfg["paths"]["masks_dir"]).parent / "qc_quant"
    rules = cfg.get("phenotype", {}).get("rules", []) or []
    cells = pd.read_csv(tables_dir / "cells_positive.csv", dtype={"sample": str})

    comps: List[pd.DataFrame] = []
    for panel in sorted(cells["panel"].dropna().unique()):
        sub = cells[cells["panel"] == panel].copy()
        markers = panel_markers(sub)
        if not markers:
            continue
        for m in markers:  # ensure clean bool for logic (CSV round-trips bools as strings/NaN)
            sub[f"{m}_pos"] = sub[f"{m}_pos"].astype(str).str.lower().isin(["true", "1", "1.0"])

        sub["phenotype"] = phenotype_label(sub, markers)
        if rules:
            sub["named_phenotype"] = apply_named_rules(sub, markers, rules)

        # Per-image phenotype composition (counts + % of cells; totals reconcile with Stage 5).
        totals = sub.groupby("field_id").size()
        comp = (sub.groupby(["field_id", "phenotype"]).size()
                .reset_index(name="n_cells"))
        comp["panel"] = panel
        comp["pct_of_field"] = comp.apply(
            lambda r: round(100.0 * r["n_cells"] / totals[r["field_id"]], 1), axis=1)
        comps.append(comp)

        # Co-occurrence heatmap.
        co = cooccurrence(sub, markers)
        fig, ax = plt.subplots(figsize=(1.4 * len(markers) + 1.5, 1.2 * len(markers) + 1))
        sns.heatmap(co, annot=True, fmt=".1f", cmap="magma", cbar_kws={"label": "% of cells"},
                    ax=ax, vmin=0)
        ax.set_title(f"Panel {panel} — marker co-occurrence (%)\n"
                     f"diagonal = % positive, off-diagonal = % double-positive", fontsize=9)
        fig.tight_layout()
        qc_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(qc_dir / f"cooccurrence_{panel}.png", dpi=120)
        plt.close(fig)
        co.to_csv(tables_dir / f"cooccurrence_{panel}.csv")
        print(f"  panel {panel}: markers {markers} | {sub['phenotype'].nunique()} phenotype combos")

    composition = pd.concat(comps, ignore_index=True) if comps else pd.DataFrame()
    composition.to_csv(tables_dir / "phenotype_composition.csv", index=False)
    return composition


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 7: phenotyping / co-expression.")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    comp = run(cfg)
    tables_dir = Path(cfg["paths"]["masks_dir"]).parent / "tables"
    print(f"\nStage 7 done: {len(comp)} (field, phenotype) rows.")
    print(f"Tables -> {tables_dir} (phenotype_composition.csv, cooccurrence_<panel>.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
