# Real Results Protocol

This repository now contains the code needed to run the real-data TNBC pCR experiment, but real performance values must only be reported after the actual public cohort tables and MRI-derived features are available locally.

## Cohort plan

| Dataset | Role | Use in final experiment |
|---|---|---|
| ISPY1 | Training / internal validation | TNBC subset, longitudinal MRI + clinical pCR labels |
| ISPY2 | Training / internal validation | TNBC subset, MRI + treatment + clinical pCR labels |
| ACRIN6698 | External validation | DCE-MRI response biomarker validation |
| Duke Breast Cancer MRI | Imaging pretraining / optional validation | MRI representation learning and segmentation/radiomics features |

## Generated artifacts after running on real data

The command below generates publication-style outputs:

```bash
python scripts/run_real_multicohort_pipeline.py \
  --table data/real/harmonized_tnbc_pcr.csv \
  --train-datasets ISPY1 ISPY2 \
  --external-datasets ACRIN6698 DUKE
```

Outputs:

- ROC curve
- Precision-recall curve
- Confusion matrix
- AUC / PR-AUC / accuracy / sensitivity / specificity / F1
- Permutation feature importance plot
- Ablation table
- Saved ensemble model

## Reporting rule

Do not replace the demo values with real AUCs until the above command has been run on the downloaded, harmonized public data.

Use this wording in the README until then:

> Real-data results will be reported after completion of cohort harmonization and external validation. The repository already includes the executable scripts that generate ROC curves, confusion matrices, AUC values, feature-importance plots, MRI-derived feature ablations, and saved model artifacts once the protected public datasets are placed in `data/real/`.
