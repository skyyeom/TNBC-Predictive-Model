# Real-data folder

Place harmonized public cohort files here after obtaining them through the official access mechanisms.

Expected main file:

```text
data/real/harmonized_tnbc_pcr.csv
```

Required columns:

| Column | Meaning |
|---|---|
| `patient_id` | De-identified patient identifier |
| `dataset` | One of `ISPY1`, `ISPY2`, `ACRIN6698`, `DUKE` |
| `pcr` | Binary pCR label, 1 = pCR, 0 = residual disease |
| `er_status` | ER status; TNBC filter expects negative/0 values |
| `pr_status` | PR status; TNBC filter expects negative/0 values |
| `her2_status` | HER2 status; TNBC filter expects negative/0 values |

Feature columns should be numeric and start with one of these prefixes:

- `clinical_`
- `treatment_`
- `mri_`
- `delta_`
- `radiomics_`
- `deep_`
- `omics_`

Run:

```bash
python scripts/run_real_multicohort_pipeline.py \
  --table data/real/harmonized_tnbc_pcr.csv \
  --train-datasets ISPY1 ISPY2 \
  --external-datasets ACRIN6698 DUKE
```

The script will create:

- `outputs/real/real_metrics.json`
- `outputs/real/real_ablation_results.csv`
- `outputs/real/roc_curve.png`
- `outputs/real/precision_recall_curve.png`
- `outputs/real/confusion_matrix.png`
- `outputs/real/feature_importance.png`
- `outputs/real/feature_importance.csv`
- `outputs/real/real_pcr_ensemble.joblib`
