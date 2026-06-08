# Efficiency Plan

## Image-Level Efficiency

1. Convert DICOM to NIfTI once and cache outputs.
2. Save tumor masks and cropped tumor volumes for each MRI timepoint.
3. Extract radiomics once and save them as structured tables.
4. Save deep MRI embeddings separately from the final prediction model.

## Model-Level Efficiency

1. Begin with the Sammut-style classical ensemble because it trains quickly.
2. Freeze the MRI encoder during early experiments.
3. Use mixed precision for 3D segmentation and MRI encoder training.
4. Use smaller student segmentation models for deployment and repeated inference.
5. Use ablation studies to justify adding omics or deep imaging features.

## Data-Level Efficiency

1. Maintain dataset manifests with one row per patient and timepoint.
2. Keep all patient IDs de-identified.
3. Track missing modalities explicitly instead of silently dropping patients.
4. Use external validation splits, not only random splits.
