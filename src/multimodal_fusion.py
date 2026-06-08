"""Fusion utilities for clinical, radiomics, deep MRI, and optional omics features."""
from __future__ import annotations
from typing import List
import pandas as pd


def collect_feature_columns(df: pd.DataFrame, include_omics: bool = True) -> List[str]:
    prefixes = ("rad_", "deep_mri_")
    base = [
        "age", "tumor_size_cm", "node_positive", "grade",
        "er_negative", "pr_negative", "her2_negative",
        "mri_ld_baseline", "mri_ld_early_ac", "mri_ld_interreg", "mri_ld_presurg",
        "mri_ld_delta_t1_t2", "mri_ld_delta_t1_t3", "mri_ld_delta_t1_t4",
    ]
    omics = ["tp53_mut", "pik3ca_mut", "hrd_score", "tmb"] if include_omics else []
    columns = [c for c in base + omics if c in df.columns]
    columns += [c for c in df.columns if c.startswith(prefixes)]
    return list(dict.fromkeys(columns))
