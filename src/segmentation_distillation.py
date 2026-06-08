"""Teacher-student segmentation loss definition.

This module documents the segmentation extension. It is intentionally dependency-light.
For real training, implement the same loss with PyTorch/MONAI tensors.
"""
from __future__ import annotations

DISTILLATION_OBJECTIVE = "L_total = L_dice + L_bce + lambda_kd * L_soft_teacher"

TEACHER_MODELS = ["nnU-Net", "Swin UNETR", "MedSAM/SAM-Med3D-style encoder"]
STUDENT_MODELS = ["Compact 3D U-Net", "Efficient UNet", "Mobile 3D CNN"]


def describe_distillation() -> str:
    return (
        "Train a high-capacity teacher on available tumor masks, then train a compact "
        "student using both ground-truth masks and teacher soft probabilities. The resulting "
        "masks support tumor volume, longest diameter, radiomics, and deep MRI embeddings."
    )
