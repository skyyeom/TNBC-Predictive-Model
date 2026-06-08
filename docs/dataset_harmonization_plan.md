# Dataset Harmonization Plan

## I-SPY1 / ACRIN 6657

Use as a longitudinal DCE-MRI pCR cohort. Map T1, T2, T3, and T4 MRI scans into the canonical timepoint fields. Extract MRI longest diameter, tumor volume, enhancement kinetics, and radiomics features.

## ACRIN 6698 / I-SPY2 Breast DWI

Use as a DWI/ADC feature source. ADC histogram statistics, diffusion signal change, and DWI-derived tumor features can be added to the MRI branch. It is best used as external validation or as an additional imaging modality when linked pCR labels are available.

## I-SPY2 DCE-MRI

Use as the primary training cohort because it is a large multi-center neoadjuvant therapy MRI dataset. Harmonize DCE-MRI timepoints, treatment variables, subtype labels, and pCR outcome.

## Duke Breast Cancer MRI

Use for imaging representation learning, tumor segmentation, and radiomics pretraining. It is not treated as the primary pCR cohort unless verified pCR outcomes are available.

## CBIS-DDSM

Use only as auxiliary pretraining for breast lesion representation. Since it is mammography rather than MRI and does not directly measure neoadjuvant pCR, it should not be mixed into the final pCR supervised training set. A defensible use is self-supervised or supervised lesion pretraining followed by fine-tuning on MRI pCR cohorts.
