"""Radiomics extraction interface.

Use PyRadiomics for real experiments. This file keeps the repository importable
without requiring heavy medical imaging dependencies by default.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict


def extract_radiomics_placeholder(image_path: str | Path, mask_path: str | Path) -> Dict[str, float]:
    raise NotImplementedError(
        "For real data, install pyradiomics and call featureextractor.RadiomicsFeatureExtractor().execute(image, mask)."
    )
