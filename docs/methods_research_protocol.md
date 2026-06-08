# Methods Research Protocol

## Study Goal

Develop and validate a TNBC-specific model for predicting pathological complete response after neoadjuvant therapy using clinical features and MRI-derived features, with optional omics features when available.

## Cohort Definition

Patients are included when breast cancer subtype satisfies:

```text
ER negative, PR negative, HER2 negative
```

Patients must have a known pCR or residual disease endpoint. MRI inclusion depends on experiment type:

- baseline-only MRI experiment: requires pre-treatment MRI
- longitudinal MRI experiment: requires at least baseline plus one on-treatment or pre-surgery timepoint
- DWI experiment: requires DWI or ADC maps from ACRIN 6698 or linked I-SPY2 imaging

## Feature Pipeline

The modeling pipeline follows Sammut et al.:

1. Curate clinical and imaging variables.
2. Impute missing values.
3. Perform univariable association screening against pCR.
4. Remove collinear features.
5. Train an unweighted ensemble consisting of logistic regression, SVM, and random forest.
6. Average model probabilities to obtain the final pCR score.

## MRI Feature Extraction

MRI features are generated from two sources:

1. Tumor size and response variables, including longest diameter and volume changes.
2. Image-derived features from segmentations, including radiomics and deep embeddings.

## Segmentation and Distillation

A high-capacity teacher segmentation model is trained on available tumor masks. The student model receives both hard segmentation labels and soft teacher outputs. The student then generates masks for radiomics extraction and tumor-localized deep feature extraction.

## Evaluation

Metrics:

- ROC-AUC
- accuracy
- sensitivity
- specificity
- calibration curve
- decision curve analysis when clinically relevant

Recommended validation:

- train on I-SPY2 and validate on I-SPY1
- train on I-SPY1 plus I-SPY2 and validate on ACRIN 6698
- use Duke MRI for pretraining rather than direct pCR evaluation if pCR labels are not available

## Limitations

Dataset labels and acquisition protocols are heterogeneous. Public imaging collections may not provide all clinical variables needed for direct harmonization. Therefore, all reported performance must be based only on patients with verified labels, matched imaging, and documented preprocessing.
