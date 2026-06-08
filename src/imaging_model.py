"""Deep MRI encoder scaffold.

The real implementation can use MONAI models such as DenseNet, SegResNet, SwinUNETR,
or a custom 3D CNN. The minimal fallback below produces deterministic placeholder
embeddings from tabular MRI measurements for testing the fusion pipeline.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def make_placeholder_deep_mri_embeddings(df: pd.DataFrame, n_features: int = 8, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    base_cols = [c for c in ["mri_ld_baseline", "mri_ld_early_ac", "mri_ld_interreg", "mri_ld_presurg"] if c in df]
    if not base_cols:
        arr = rng.normal(size=(len(df), n_features))
    else:
        base = df[base_cols].fillna(df[base_cols].median()).to_numpy()
        weights = rng.normal(size=(base.shape[1], n_features))
        arr = base @ weights
        arr = (arr - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-6)
    return pd.DataFrame(arr, columns=[f"deep_mri_{i:02d}" for i in range(n_features)], index=df.index)
