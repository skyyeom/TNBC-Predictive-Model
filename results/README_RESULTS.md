# Multimodal TNBC pCR Prediction Results

This run uses the uploaded Duke Breast MRI clinical/imaging feature spreadsheets and the uploaded I-SPY2 / ACRIN6698 clinical and longitudinal MRI feature spreadsheets. The pipeline uses feature harmonization, univariable feature selection, and an unweighted ensemble of logistic regression, linear SVM, and random forest classifiers.

## Cohorts Used

| Cohort | Data used | TNBC definition | Outcome | N | pCR rate |
|---|---|---|---|---:|---:|
| Duke Breast MRI | Clinical variables + MRI-derived radiomic features | Mol Subtype = triple negative or ER-/PR-/HER2- | Complete response vs not complete response | 78 | 0.295 |
| I-SPY2 / ACRIN6698 Cohort 1 | Clinical variables + longitudinal MRI features: FTV, LD, sphericity, BPE changes | HR=0 and HER2=0 | pCR | 132 | 0.379 |

## Results

| dataset              | model          |   n_total |   n_test |   pcr_rate |   auc |   pr_auc |   accuracy |   precision |   recall |    f1 |   tn |   fp |   fn |   tp |   n_features_raw |
|:---------------------|:---------------|----------:|---------:|-----------:|------:|---------:|-----------:|------------:|---------:|------:|-----:|-----:|-----:|-----:|-----------------:|
| Duke_TNBC            | Clinical only  |        78 |       20 |      0.295 | 0.571 |    0.368 |      0.650 |       0.429 |    0.500 | 0.462 |   10 |    4 |    3 |    3 |               10 |
| Duke_TNBC            | MRI only       |        78 |       20 |      0.295 | 0.690 |    0.536 |      0.600 |       0.333 |    0.333 | 0.333 |   10 |    4 |    4 |    2 |              529 |
| Duke_TNBC            | Clinical + MRI |        78 |       20 |      0.295 | 0.774 |    0.600 |      0.700 |       0.500 |    0.667 | 0.571 |   10 |    4 |    2 |    4 |              539 |
| ISPY2_ACRIN6698_TNBC | Clinical only  |       132 |       33 |      0.379 | 0.425 |    0.364 |      0.455 |       0.200 |    0.167 | 0.182 |   13 |    8 |   10 |    2 |                5 |
| ISPY2_ACRIN6698_TNBC | MRI only       |       132 |       33 |      0.379 | 0.754 |    0.676 |      0.636 |       0.500 |    0.750 | 0.600 |   12 |    9 |    3 |    9 |               28 |
| ISPY2_ACRIN6698_TNBC | Clinical + MRI |       132 |       33 |      0.379 | 0.778 |    0.747 |      0.636 |       0.500 |    0.750 | 0.600 |   12 |    9 |    3 |    9 |               33 |

## Output Files

- `harmonized_tnbc_pcr.csv`
- `results_metrics.csv`
- `*_roc.png`
- `*_confusion_matrix.png`
- `*_feature_importance.csv/png`
- `ablation_auc_summary.png`
