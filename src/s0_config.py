"""Stage 0 - Config loading + manifest construction.

Scans the raw TIFF folder and builds a human-reviewable `manifest.csv` mapping
every file -> panel / sample / magnification / field / marker / channel / role.
Filename parsing is deliberately *tolerant* (anchored on the magnification token,
not a brittle whole-name regex) so the two inconsistent naming schemes both work,
and anything unparsed is flagged in a `notes` column rather than crashing the run.

CLI:  python -m src.s0_config --config config/config.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.utils.io import dominant_channel, read_tiff

# A magnification token looks like "20x", "40X", "10x", etc.
_MAG_RE = re.compile(r"^(\d+)[xX]$")
# A date prefix like "2026-07-10_" marks a Panel-A filename.
_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load the YAML config and resolve output paths relative to the repo root.

    Relative paths in the config are resolved against the repo root (the parent
    of `config/`), so the CLI works regardless of the current directory.
    """
    config_path = Path(config_path).resolve()
    repo_root = config_path.parent.parent
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    paths = cfg.setdefault("paths", {})
    # raw_dir is absolute in the shipped config; leave absolute paths untouched.
    # Resolve every relative output path against the repo root so stages/notebooks
    # work regardless of the current working directory.
    for key in ("manifest", "gray_dir", "qc_dir", "masks_dir", "qc_seg_dir"):
        if key in paths and not Path(paths[key]).is_absolute():
            paths[key] = str(repo_root / paths[key])
    if "raw_dir" in paths and not Path(paths["raw_dir"]).is_absolute():
        paths["raw_dir"] = str((repo_root / paths["raw_dir"]).resolve())
    cfg["_repo_root"] = str(repo_root)
    return cfg


# --------------------------------------------------------------------------- #
# Filename parsing
# --------------------------------------------------------------------------- #
def normalize_marker(token: str, aliases: Dict[str, str]) -> Optional[str]:
    """Map a raw filename marker token to its canonical name via the alias map."""
    return aliases.get(token.strip().lower())


def parse_filename(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Parse one TIFF filename into manifest fields.

    Strategy: split the stem on whitespace and anchor on the magnification token
    (matches `\\d+x`). Everything before it is the sample part; everything after
    is the marker. This tolerates both `2026-07-10_CFA 20x CD11C` and
    `CFA 1.1 40X F480` without a per-scheme regex.

    Returns a dict with panel/sample/magnification/marker/... and a `notes` field
    describing any parse problem (never raises on odd names).
    """
    aliases = cfg["marker_aliases"]
    overlay_markers = set(cfg.get("overlay_markers", ["overlay"]))
    stem = Path(name).stem
    tokens = stem.split()

    out: Dict[str, Any] = {
        "filename": name, "panel": None, "sample": None, "magnification": None,
        "marker": None, "marker_raw": None, "role": "marker",
        "field_id": None, "notes": "",
    }

    # Locate the magnification token.
    mag_idx = next((i for i, t in enumerate(tokens) if _MAG_RE.match(t)), None)
    if mag_idx is None:
        out["notes"] = "no magnification token found; parse manually"
        return out
    out["magnification"] = tokens[mag_idx].lower()

    # Marker = everything after the magnification token (usually one token).
    marker_tokens = tokens[mag_idx + 1:]
    if not marker_tokens:
        out["notes"] = "no marker token after magnification; parse manually"
        return out
    marker_raw = " ".join(marker_tokens)
    out["marker_raw"] = marker_raw
    canonical = normalize_marker(marker_raw, aliases)
    if canonical is None:
        out["notes"] = f"unrecognised marker '{marker_raw}'; add to marker_aliases"
        out["marker"] = marker_raw  # keep raw so the row is still reviewable
    else:
        out["marker"] = canonical
    if out["marker"] in overlay_markers:
        out["role"] = "overlay"

    # Panel + sample from the tokens before the magnification.
    pre = tokens[:mag_idx]
    if pre and _DATE_PREFIX_RE.match(pre[0]):
        # Panel A: date-prefixed, e.g. "2026-07-10_CFA" -> sample "CFA".
        out["panel"] = "A"
        out["sample"] = _DATE_PREFIX_RE.sub("", pre[0]) or "CFA"
    elif pre and pre[0].upper() == "CFA":
        # Panel B: "CFA <sample>" -> sample is the token after CFA.
        out["panel"] = "B"
        out["sample"] = pre[1] if len(pre) > 1 else "CFA"
    else:
        out["notes"] = (out["notes"] + "; " if out["notes"] else "") + \
            "could not assign panel/sample; parse manually"
        out["sample"] = pre[-1] if pre else None

    if out["panel"] and out["sample"] and out["magnification"]:
        out["field_id"] = f"{out['panel']}_{out['sample']}_{out['magnification']}"
    return out


# --------------------------------------------------------------------------- #
# Manifest build + validation
# --------------------------------------------------------------------------- #
def build_manifest(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Scan raw_dir and build the manifest DataFrame (unsorted rows -> sorted)."""
    raw_dir = Path(cfg["paths"]["raw_dir"])
    overlay_markers = set(cfg.get("overlay_markers", ["overlay"]))
    stats_cfg = cfg.get("channel_stats", {})
    step = int(stats_cfg.get("sample_step", 4))
    frac = float(stats_cfg.get("populated_frac", 0.4))

    files = sorted(raw_dir.glob("*.tif")) + sorted(raw_dir.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files found in {raw_dir}")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        rec = parse_filename(fp.name, cfg)
        # Dominant colour channel (cheap QC cross-check vs the marker alias).
        try:
            rec["channel_dominant"] = dominant_channel(read_tiff(fp), step, frac)
        except Exception as exc:  # pragma: no cover - unreadable file
            rec["channel_dominant"] = "error"
            rec["notes"] = (rec["notes"] + "; " if rec["notes"] else "") + f"read error: {exc}"
        rec["file_path"] = str(fp)
        rec["exclude"] = rec["role"] == "overlay" or rec["marker"] in overlay_markers
        # Editable metadata columns for later stages (left blank for the user to
        # fill; `group` is the experimental condition for stats -- everything is
        # currently "CFA" and a control/naive group is still needed, see plan Sec.7).
        rec["animal"] = ""
        rec["group"] = ""
        rows.append(rec)

    cols = ["filename", "file_path", "panel", "sample", "magnification", "field_id",
            "marker", "marker_raw", "role", "channel_dominant", "exclude",
            "animal", "group", "notes"]
    df = pd.DataFrame(rows)[cols]
    return df.sort_values(["panel", "sample", "magnification", "marker"],
                          na_position="last").reset_index(drop=True)


def validate_manifest(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable validation warnings (empty == all good)."""
    warnings: List[str] = []
    nuclear = cfg.get("nuclear_marker", "DAPI")

    unparsed = df[df["notes"].str.len() > 0]
    for _, r in unparsed.iterrows():
        warnings.append(f"[parse] {r['filename']}: {r['notes']}")

    # Every non-overlay field must have exactly one nuclear (DAPI) entry.
    quant = df[~df["exclude"] & df["field_id"].notna()]
    for fid, grp in quant.groupby("field_id"):
        n_dapi = int((grp["marker"] == nuclear).sum())
        if n_dapi != 1:
            warnings.append(f"[field] {fid}: expected exactly 1 {nuclear}, found {n_dapi}")
        dups = grp["marker"].value_counts()
        for marker, count in dups.items():
            if count > 1:
                warnings.append(f"[field] {fid}: marker {marker} appears {count}x")

    # Cross-check: overlay files should have all 3 channels populated.
    for _, r in df[df["exclude"]].iterrows():
        if r["channel_dominant"] not in ("RGB", "error"):
            warnings.append(f"[qc] {r['filename']}: flagged overlay but channels="
                            f"{r['channel_dominant']} (expected RGB)")
    return warnings


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact human summary of the manifest to stdout."""
    print(f"\nManifest: {len(df)} files, {df['field_id'].nunique(dropna=True)} fields, "
          f"{int((~df['exclude']).sum())} quantifiable / {int(df['exclude'].sum())} excluded (overlay)\n")
    view = df[["field_id", "marker", "magnification", "channel_dominant", "role", "exclude"]]
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(view.to_string(index=False))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 0: build the file manifest.")
    ap.add_argument("--config", default="config/config.yaml", help="path to config.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    df = build_manifest(cfg)

    manifest_path = Path(cfg["paths"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    print(f"Wrote manifest -> {manifest_path}")

    print_summary(df)
    warnings = validate_manifest(df, cfg)
    if warnings:
        print(f"\n{len(warnings)} validation warning(s):")
        for w in warnings:
            print("  - " + w)
        print("\nReview/correct manifest.csv before running Stage 1 if needed.")
    else:
        print("\nValidation: OK (all fields have exactly one DAPI; overlays flagged).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
