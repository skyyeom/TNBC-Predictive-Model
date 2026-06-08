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

_NEGATIVE_VALUES = {"negative", "neg", "0", "false", "f", "no", "n", 0, False}
_POSITIVE_VALUES = {"positive", "pos", "1", "true", "t", "yes", "y", 1, True}


def _as_negative_indicator(series: pd.Series) -> pd.Series:
    def convert(v):
        if pd.isna(v):
            return pd.NA
        lv = str(v).strip().lower() if not isinstance(v, (int, float, bool)) else v
        if lv in _NEGATIVE_VALUES:
            return 1
        if lv in _POSITIVE_VALUES:
            return 0
        return pd.NA
    return series.map(convert).astype("Int64")


def ensure_tnbc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping = {
        "er_negative": "er_status",
        "pr_negative": "pr_status",
        "her2_negative": "her2_status",
    }
    for indicator, status_col in mapping.items():
        if indicator not in out.columns and status_col in out.columns:
            out[indicator] = _as_negative_indicator(out[status_col])
    return out


def filter_tnbc(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_tnbc_indicators(df)
    required = ["er_negative", "pr_negative", "her2_negative"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing TNBC subtype columns or source status columns: {missing}")
    return out[(out["er_negative"] == 1) & (out["pr_negative"] == 1) & (out["her2_negative"] == 1)].copy()


def add_longitudinal_mri_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Support both older synthetic names and clearer template names.
    aliases = {
        "mri_ld_baseline": ["mri_t1_longest_diameter_mm"],
        "mri_ld_early_ac": ["mri_t2_longest_diameter_mm"],
        "mri_ld_interreg": ["mri_t3_longest_diameter_mm"],
        "mri_ld_presurg": ["mri_t4_longest_diameter_mm"],
    }
    for canonical, candidates in aliases.items():
        if canonical not in out.columns:
            for c in candidates:
                if c in out.columns:
                    out[canonical] = out[c]
                    break
    needed = ["mri_ld_baseline", "mri_ld_early_ac", "mri_ld_interreg", "mri_ld_presurg"]
    if not all(c in out.columns for c in needed):
        return out
    baseline = out["mri_ld_baseline"].replace(0, pd.NA)
    out["delta_mri_ld_t1_t2"] = (out["mri_ld_baseline"] - out["mri_ld_early_ac"]) / baseline
    out["delta_mri_ld_t1_t3"] = (out["mri_ld_baseline"] - out["mri_ld_interreg"]) / baseline
    out["delta_mri_ld_t1_t4"] = (out["mri_ld_baseline"] - out["mri_ld_presurg"]) / baseline
    # Preserve previous synthetic names for compatibility.
    out["mri_ld_delta_t1_t2"] = out["delta_mri_ld_t1_t2"]
    out["mri_ld_delta_t1_t3"] = out["delta_mri_ld_t1_t3"]
    out["mri_ld_delta_t1_t4"] = out["delta_mri_ld_t1_t4"]
    return out
