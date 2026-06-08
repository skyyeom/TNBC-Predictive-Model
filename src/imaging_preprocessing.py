"""MRI preprocessing placeholders.

Real MRI experiments should convert DICOM to NIfTI, resample spacing, normalize
intensity, crop around breast/tumor regions, and save aligned volumes per patient
and timepoint.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict


def expected_mri_paths(root: str | Path, patient_id: str) -> Dict[str, Path]:
    root = Path(root)
    return {
        "baseline": root / patient_id / "T1_baseline.nii.gz",
        "early_ac": root / patient_id / "T2_early_ac.nii.gz",
        "inter_regimen": root / patient_id / "T3_inter_regimen.nii.gz",
        "pre_surgery": root / patient_id / "T4_pre_surgery.nii.gz",
    }


def preprocess_volume_placeholder(input_path: str, output_path: str) -> None:
    raise NotImplementedError(
        "Install nibabel/SimpleITK and implement resampling, bias correction, "
        "z-score normalization, and breast/tumor cropping for real MRI data."
    )
