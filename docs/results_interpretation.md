# Results Interpretation Guide

This repository currently reports synthetic demo results only. The numbers confirm that the pipeline executes, but they are not clinical claims.

## What to Report for Real Data

For each dataset split, report:

- number of TNBC patients
- number of pCR and residual disease cases
- train, validation, and external-test split definitions
- AUC with confidence interval
- accuracy, sensitivity, specificity, PPV, and NPV
- calibration curve or Brier score
- selected features after univariable filtering and correlation reduction
- ablation results for clinical-only, MRI-only, clinical + MRI, and full multimodal models

## Required Caution

Do not mix synthetic results with real public-dataset results. Keep them in separate tables and clearly label data source and validation strategy.
