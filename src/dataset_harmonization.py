"""Dataset harmonization utilities for ISPY1, ACRIN6698, ISPY2, Duke, and CBIS-DDSM.

These functions define the expected mapping. Real public datasets must be downloaded
separately according to their access rules and then converted into this schema.
"""
from __future__ import annotations
import pandas as pd

TNBC_RULE = "ER negative AND PR negative AND HER2 negative"

TIMEPOINT_MAP = {
    "ISPY1": {
        "T1": "baseline_pre_treatment",
        "T2": "early_treatment_24_96h_after_AC_start",
        "T3": "inter_regimen_after_AC_before_taxane",
        "T4": "pre_surgery_after_chemotherapy",
    },
    "ISPY2": {
        "T0": "baseline",
        "T1": "early_treatment",
        "T2": "inter_regimen",
        "T3": "pre_surgery",
    },
}

DATASET_ROLES = {
    "ispy1": "longitudinal DCE-MRI pCR cohort",
    "acrin6698": "DWI/ADC MRI response biomarker cohort",
    "ispy2": "primary DCE-MRI pCR training cohort",
    "duke": "DCE-MRI imaging pretraining and radiomics cohort",
    "cbis_ddsm": "auxiliary mammography lesion pretraining cohort",
}

def filter_tnbc(df: pd.DataFrame) -> pd.DataFrame:
    required = ["er_negative", "pr_negative", "her2_negative"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing TNBC subtype columns: {missing}")
    return df[(df["er_negative"] == 1) & (df["pr_negative"] == 1) & (df["her2_negative"] == 1)].copy()

def add_longitudinal_mri_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    baseline = out["mri_ld_baseline"].replace(0, pd.NA)
    out["mri_ld_delta_t1_t2"] = (out["mri_ld_baseline"] - out["mri_ld_early_ac"]) / baseline
    out["mri_ld_delta_t1_t3"] = (out["mri_ld_baseline"] - out["mri_ld_interreg"]) / baseline
    out["mri_ld_delta_t1_t4"] = (out["mri_ld_baseline"] - out["mri_ld_presurg"]) / baseline
    return out
