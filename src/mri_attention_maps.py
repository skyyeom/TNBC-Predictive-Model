"""MRI attention-map utilities for trained PyTorch imaging encoders.

This module intentionally provides reusable scaffolding rather than fabricated maps.
Use it after training a real MRI encoder on image tensors.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def normalize_heatmap(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmin(x)
    denom = np.nanmax(x) + 1e-8
    return x / denom


def save_attention_overlay_placeholder(out_path: str | Path) -> None:
    """Create a text marker explaining why no real attention map is bundled."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "MRI attention maps are generated only after a real MRI encoder has been trained "
        "on patient image tensors. Use Grad-CAM, attention rollout, or occlusion sensitivity "
        "on the trained encoder and save overlays here.\n",
        encoding="utf-8",
    )
