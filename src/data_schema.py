"""Canonical schema for TNBC pCR multimodal prediction."""
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class Columns:
    patient_id: str = "patient_id"
    dataset: str = "dataset"
    pcr: str = "pcr"
    er_negative: str = "er_negative"
    pr_negative: str = "pr_negative"
    her2_negative: str = "her2_negative"

CLINICAL_COLUMNS: List[str] = [
    "age", "tumor_size_cm", "node_positive", "grade",
    "er_negative", "pr_negative", "her2_negative",
]

MRI_LD_COLUMNS: List[str] = [
    "mri_ld_baseline", "mri_ld_early_ac", "mri_ld_interreg", "mri_ld_presurg",
    "mri_ld_delta_t1_t2", "mri_ld_delta_t1_t3", "mri_ld_delta_t1_t4",
]

OPTIONAL_OMICS_COLUMNS: List[str] = ["tp53_mut", "pik3ca_mut", "hrd_score", "tmb"]
