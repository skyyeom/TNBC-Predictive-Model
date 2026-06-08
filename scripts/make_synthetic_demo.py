"""Create a synthetic TNBC multimodal table for testing the pipeline.

This does not represent real patient data. It only verifies that the code runs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 240
patient_id = [f"SYN_{i:04d}" for i in range(n)]
age = rng.normal(49, 10, n).clip(25, 80)
tumor_size = rng.gamma(3, 1.2, n).clip(1.0, 10.0)
node_positive = rng.binomial(1, 0.48, n)
grade = rng.choice([2, 3], n, p=[0.25, 0.75])
er_negative = np.ones(n, dtype=int)
pr_negative = np.ones(n, dtype=int)
her2_negative = np.ones(n, dtype=int)
baseline = rng.normal(45, 12, n).clip(10, 100)
shrink_factor = rng.beta(2.2, 2.0, n)
early = baseline * (1 - 0.18 * shrink_factor + rng.normal(0, 0.04, n))
inter = baseline * (1 - 0.46 * shrink_factor + rng.normal(0, 0.06, n))
presurg = baseline * (1 - 0.78 * shrink_factor + rng.normal(0, 0.08, n))
tp53 = rng.binomial(1, 0.75, n)
pik3ca = rng.binomial(1, 0.12, n)
hrd = rng.normal(42, 18, n).clip(0, 100)
tmb = rng.gamma(2, 1.2, n)
rad_entropy = rng.normal(0, 1, n) + 0.4 * shrink_factor
rad_glcm = rng.normal(0, 1, n) + 0.3 * hrd / 50
logit = -0.9 + 2.2 * shrink_factor + 0.018 * hrd + 0.4 * tp53 - 0.5 * node_positive - 0.08 * tumor_size + 0.25 * rad_entropy
prob = 1 / (1 + np.exp(-logit))
pcr = rng.binomial(1, prob)

df = pd.DataFrame({
    "patient_id": patient_id,
    "dataset": rng.choice(["ispy1", "ispy2", "acrin6698"], n, p=[0.25, 0.55, 0.20]),
    "age": age,
    "tumor_size_cm": tumor_size,
    "node_positive": node_positive,
    "grade": grade,
    "er_negative": er_negative,
    "pr_negative": pr_negative,
    "her2_negative": her2_negative,
    "mri_ld_baseline": baseline,
    "mri_ld_early_ac": early.clip(0, None),
    "mri_ld_interreg": inter.clip(0, None),
    "mri_ld_presurg": presurg.clip(0, None),
    "tp53_mut": tp53,
    "pik3ca_mut": pik3ca,
    "hrd_score": hrd,
    "tmb": tmb,
    "rad_firstorder_entropy": rad_entropy,
    "rad_glcm_contrast": rad_glcm,
    "pcr": pcr,
})
for i in range(8):
    df[f"deep_mri_{i:02d}"] = rng.normal(0, 1, n) + (0.25 + i * 0.02) * shrink_factor

out = Path("data/processed")
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "synthetic_tnbc_multimodal.csv", index=False)
print(f"Wrote {out / 'synthetic_tnbc_multimodal.csv'} with {len(df)} rows")
