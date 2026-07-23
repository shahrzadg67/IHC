"""Stage 11 - Visualization & QC report.

Bundles the pipeline's results into a single self-contained HTML report: overview
stats, summary figures (density, positivity, phenotype composition), the immune–nerve
spatial results, and QC thumbnails (segmentation + nerve overlays). All images are
embedded as base64 so the report is one portable file.

CLI:  python -m src.s11_report --config config/config.yaml
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.s0_config import load_config


def _fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _png_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _img(b64: str, width: str = "100%") -> str:
    return f'<img loading="lazy" style="max-width:{width};height:auto" src="data:image/png;base64,{b64}">'


def _fig_density(per_img: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(per_img["field_id"], per_img["density_per_Mpx"], color="#4477aa")
    ax.set_ylabel("cells / Mpx"); ax.set_title("Cell density per field")
    plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    return _fig_b64(fig)


def _fig_positivity(thr: pd.DataFrame) -> str:
    labels = [f"{r.panel}:{r.marker}" for r in thr.itertuples()]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(labels, thr["pct_positive"], color="#cc7755")
    ax.set_ylabel("% positive"); ax.set_title("Marker positivity (per panel)")
    plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    return _fig_b64(fig)


def _fig_phenotype(comp: pd.DataFrame) -> str:
    pivot = comp.pivot_table(index="field_id", columns="phenotype",
                             values="pct_of_field", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20", width=0.8)
    ax.set_ylabel("% of cells"); ax.set_title("Phenotype composition per field")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right"); fig.tight_layout()
    return _fig_b64(fig)


def _section(title: str, body: str) -> str:
    return f'<section><h2>{title}</h2>{body}</section>'


def _thumbs(paths: List[Path], width: str = "48%") -> str:
    return "".join(f'<figure>{_img(_png_b64(p), width="100%")}<figcaption>{p.stem}</figcaption></figure>'
                   for p in paths)


def build_html(cfg: Dict[str, Any]) -> str:
    tables = Path(cfg["paths"]["masks_dir"]).parent / "tables"
    qc_seg = Path(cfg["paths"]["qc_seg_dir"])
    qc_q = Path(cfg["paths"]["masks_dir"]).parent / "qc_quant"

    def rd(name):
        p = tables / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    per_img = rd("per_image_summary.csv")
    thr = rd("positivity_thresholds.csv")
    comp = rd("phenotype_composition.csv")
    tests = rd("internal_tests.csv")
    nerve = rd("nerve_metrics.csv")
    cells = rd("cells.csv")

    n_fields = per_img["field_id"].nunique() if not per_img.empty else 0
    n_cells = int(per_img["n_cells"].sum()) if not per_img.empty else 0
    panels = ", ".join(sorted(per_img["panel"].unique())) if not per_img.empty else "-"

    parts: List[str] = []
    parts.append(_section("Overview", f"""
      <p><b>{n_cells}</b> cells across <b>{n_fields}</b> fields (panels {panels}).
      Condition: <b>CFA</b> (single group → descriptive analysis, no between-group comparison).</p>
      <div class="tbl">{per_img.to_html(index=False)}</div>"""))

    figs = f'<div class="row">{_img(_fig_density(per_img), "49%")}'
    if not thr.empty:
        figs += _img(_fig_positivity(thr), "49%")
    figs += "</div>"
    if not comp.empty:
        figs += _img(_fig_phenotype(comp), "80%")
    body = figs
    if not thr.empty:
        body += f'<h3>Positivity thresholds</h3><div class="tbl">{thr.to_html(index=False)}</div>'
    parts.append(_section("Cell density, positivity & phenotypes", body))

    # Co-occurrence heatmaps.
    co = sorted(qc_q.glob("cooccurrence_*.png"))
    if co:
        parts.append(_section("Marker co-occurrence", f'<div class="row">{_thumbs(co)}</div>'))

    # Nerve & immune–nerve spatial.
    spatial_body = ""
    if not nerve.empty:
        spatial_body += f'<h3>Nerve metrics (Panel B)</h3><div class="tbl">{nerve.to_html(index=False)}</div>'
    dh = sorted(qc_q.glob("distance_to_nerve_*.png"))
    nh = sorted(qc_q.glob("nhood_enrichment_*.png"))
    if dh:
        spatial_body += "<h3>Distance-to-nerve by marker positivity</h3>" + "".join(_img(_png_b64(p), "90%") for p in dh)
    if not tests.empty:
        spatial_body += ('<h3>Internal test — marker⁺ vs marker⁻ distance-to-nerve</h3>'
                         f'<div class="tbl">{tests.to_html(index=False)}</div>'
                         '<p class="note">Exploratory, cell-level (pseudoreplication); '
                         '<code>fields_pos_closer</code> is the replicate-level consistency check.</p>')
    if nh:
        spatial_body += "<h3>Phenotype neighborhood enrichment (z vs null)</h3>" + "".join(_img(_png_b64(p), "70%") for p in nh)
    nerve_ov = sorted(qc_q.glob("nerve_*.png"))
    if nerve_ov:
        spatial_body += f'<h3>Nerve masks</h3><div class="row">{_thumbs(nerve_ov)}</div>'
    if spatial_body:
        parts.append(_section("Nerve &amp; immune–nerve spatial", spatial_body))

    # Segmentation QC thumbnails.
    seg = sorted(qc_seg.glob("*__nuclei.png"))
    if seg:
        parts.append(_section("Segmentation QC (nuclei overlays)", f'<div class="row">{_thumbs(seg)}</div>'))

    parts.append(_section("Caveats", """
      <ul>
        <li>Inputs are <b>8-bit RGB figure exports</b> → relative/morphological analysis only;
            intensities are not cross-image comparable. Rerun on raw 16-bit for absolute numbers (M6).</li>
        <li><b>No control group</b> (all CFA) → descriptive + internal/spatial tests, not between-group.</li>
        <li>Distances/areas are in <b>pixels</b> until pixel size (µm/px) is provided.</li>
        <li>Replicate structure of CFA 1/1.1/1.2/X unconfirmed → stats unit may need revising.</li>
      </ul>"""))

    today = dt.date.today().isoformat()
    css = """
      body{font-family:system-ui,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1a1a1a}
      h1{margin-bottom:0} .sub{color:#666;margin-top:4px}
      section{border-top:2px solid #eee;padding-top:8px;margin-top:28px}
      h2{color:#33475b} h3{color:#4a5a6a;margin-bottom:6px}
      .row{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start}
      figure{margin:0;flex:1 1 45%} figcaption{font-size:11px;color:#777;text-align:center}
      .tbl{overflow-x:auto} table{border-collapse:collapse;font-size:12px}
      th,td{border:1px solid #ddd;padding:3px 7px;text-align:right} th{background:#f4f6f8}
      .note{font-size:12px;color:#777} code{background:#f0f0f0;padding:1px 4px;border-radius:3px}
    """
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>IF pipeline report</title>"
            f"<style>{css}</style></head><body>"
            f"<h1>IF Cell Segmentation & Quantification — Report</h1>"
            f"<p class='sub'>Generated {today} · descriptive analysis (CFA, no control group)</p>"
            + "".join(parts) + "</body></html>")


def run(cfg: Dict[str, Any]) -> Path:
    repo_root = Path(cfg["_repo_root"])
    rep_dir = Path(cfg.get("report", {}).get("dir", "outputs/report"))
    if not rep_dir.is_absolute():
        rep_dir = repo_root / rep_dir
    rep_dir.mkdir(parents=True, exist_ok=True)
    out = rep_dir / "report.html"
    out.write_text(build_html(cfg))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 11: build the HTML report.")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = run(cfg)
    print(f"Stage 11 done. Report -> {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
