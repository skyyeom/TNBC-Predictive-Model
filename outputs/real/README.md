# Real outputs

This folder is intentionally empty until the real-data pipeline is run on harmonized public cohort data.

Run:

```bash
python scripts/run_real_multicohort_pipeline.py \
  --table data/real/harmonized_tnbc_pcr.csv \
  --train-datasets ISPY1 ISPY2 \
  --external-datasets ACRIN6698 DUKE
```
