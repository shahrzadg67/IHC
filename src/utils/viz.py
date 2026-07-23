"""Visualization helpers for segmentation QC (mask outlines on the source image)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless cluster node
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import find_boundaries


def overlay_labels(gray: np.ndarray, labels: np.ndarray, title: str, out_path: str | Path,
                   second_labels: Optional[np.ndarray] = None) -> Path:
    """Save a QC PNG: grayscale image with label-mask boundaries drawn on top.

    `labels` boundaries are drawn in one colour; optional `second_labels` (e.g. expanded
    cell bodies) in a second colour, so nucleus vs cell outlines can be compared.
    The subtitle reports the object count.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(gray, cmap="gray")

    if second_labels is not None:
        b2 = find_boundaries(second_labels, mode="outer")
        ov2 = np.zeros((*gray.shape, 4))
        ov2[b2] = (0.1, 0.6, 1.0, 1.0)  # cyan = expanded cell bodies
        ax.imshow(ov2)

    b1 = find_boundaries(labels, mode="outer")
    ov1 = np.zeros((*gray.shape, 4))
    ov1[b1] = (1.0, 0.85, 0.0, 1.0)      # yellow = nuclei
    ax.imshow(ov1)

    n = int(labels.max())
    sub = f"{n} nuclei" + (f"  /  cells outlined in cyan" if second_labels is not None else "")
    ax.set_title(f"{title}\n{sub}", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
