# IF Cell Segmentation & Quantification Pipeline

Reproducible pipeline for segmenting cells and quantifying immunofluorescence (IF)
markers in CFA microscopy images. Full design lives in [`../plan.md`](../plan.md);
this folder implements it stage by stage.

## Status — Milestone M1 (Stage 0 + Stage 1) ✅
- **Stage 0** — builds a reviewable `data/manifest.csv` mapping every raw TIFF →
  panel / sample / magnification / field / marker / channel / role, and validates it.
- **Stage 1** — extracts a clean 2-D grayscale image per marker per field
  (max-projection across R/G/B of the 8-bit pseudo-coloured exports) plus per-field
  QC contact sheets.

Later milestones (M2 segmentation with Cellpose-SAM, M3 quantification, M4 nerve/spatial,
M5 stats/report, M6 raw-16-bit rerun) are not yet implemented — see `../plan.md` §6, §8.

## Setup (SickKids `hpf`)
```bash
module load python/3.11.15
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run M1
```bash
cd "/hpf/projects/msalter/sghazis/Imaging/ED/For Shahrzad/if-pipeline"
source .venv/bin/activate

python -m src.s0_config --config config/config.yaml   # -> data/manifest.csv (review it)
python -m src.s1_ingest --config config/config.yaml   # -> outputs/gray/*.tif + outputs/qc/*.png

pytest tests/                                          # unit tests
```

## Review the results
A methods-documented review notebook renders the manifest, all 6 QC contact sheets, per-marker
intensity histograms, and a step-by-step channel-extraction demo:
```bash
# Option A — just look (no Jupyter needed): open the pre-rendered, self-contained HTML
notebooks/review_M1.html

# Option B — interactive: launch JupyterLab and pick the "Python (if-pipeline)" kernel
module load python/3.11.15 && source .venv/bin/activate
jupyter lab notebooks/review_M1.ipynb        # or: jupyter notebook
# To re-execute headless after editing:
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=if-pipeline notebooks/review_M1.ipynb
```

## Layout
```
config/config.yaml   paths, pixel sizes, marker aliases, panel defs, input_mode
src/utils/io.py      TIFF read/write, RGB max-projection, channel stats
src/s0_config.py     Stage 0: config + manifest builder + validation  (CLI)
src/s1_ingest.py     Stage 1: RGB→grayscale extraction + QC sheets     (CLI)
data/manifest.csv    generated; review/edit before downstream stages
outputs/gray/        {field_id}__{marker}.tif  (2-D 8-bit)
outputs/qc/          per-field contact-sheet PNGs
notebooks/           review_M1.ipynb + review_M1.html (methods + manifest + QC review)
tests/               pytest unit tests (parser + projection)
```

## Data caveats (read `../plan.md` §3)
The current TIFFs are **8-bit RGB figure exports** → good for developing the pipeline
and relative/morphological analysis, **not** for publication-grade intensity numbers.
Rerun on raw 16-bit single-channel/OME-TIFF data (config `input_mode`) at M6.

### Manifest columns you may need to edit
- `animal`, `group` — biological replicate / condition labels. `group` is set to `CFA` (the single
  condition present); `animal` is blank pending confirmation of whether `CFA 1/1.1/1.2/X` are separate
  animals or ROIs of one (sets the replicate unit for dispersion — see `../plan.md` §7).
- `marker`, `exclude` — override if any file was mis-parsed (Stage 0 flags these in `notes`).

## Analysis design — descriptive (no control group)
All images are the **CFA** condition; there is no naive/vehicle/contralateral comparison group. The
analysis is therefore **descriptive**, not between-group:
- **Characterize** CFA tissue — cell density, %-positive per marker, phenotype fractions, PGP9.5 nerve
  area fraction, and immune–nerve distances — reported **per biological replicate** with dispersion.
- **Internal / spatial tests that need no second group** (still valid statistics): immune cells
  **near vs far** from nerve, marker⁺ vs marker⁻ distance distributions, and spatial enrichment /
  co-occurrence tested against a **permutation / random null** (`squidpy`).
- If a contralateral (CFA is usually unilateral) or naive/vehicle group is added later, standard
  between-group stats (Stage 10) become possible — revisit then.
