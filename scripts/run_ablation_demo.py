"""Run synthetic ablation experiments for the TNBC pCR demo.

This uses synthetic data only. It is intended to verify the multimodal
pipeline structure and produce README-ready example tables.
"""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.dataset_harmonization import filter_tnbc, add_longitudinal_mri_deltas
from src.feature_engineering import select_features
from src.modeling import UnweightedPCRensemble

RANDOM_STATE = 42
TARGET = "pcr"
ID_COL = "patient_id"
DATA_PATH = Path("data/processed/synthetic_tnbc_multimodal.csv")
OUT_PATH = Path("outputs/ablation_results.csv")


def existing(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def group_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    clinical = existing(df, [
        "age", "tumor_size_cm", "node_positive", "grade",
        "er_negative", "pr_negative", "her2_negative",
    ])
    mri_ld = existing(df, [
        "mri_ld_baseline", "mri_ld_early_ac", "mri_ld_interreg", "mri_ld_presurg",
        "mri_ld_delta_t1_t2", "mri_ld_delta_t1_t3", "mri_ld_delta_t1_t4",
    ])
    radiomics = [c for c in df.columns if c.startswith("rad_")]
    deep_mri = [c for c in df.columns if c.startswith("deep_mri_")]
    omics = existing(df, ["tp53_mut", "pik3ca_mut", "hrd_score", "tmb"])
    return {
        "Clinical only": clinical,
        "MRI only": mri_ld + radiomics + deep_mri,
        "Clinical + MRI longest diameter": clinical + mri_ld,
        "Clinical + radiomics": clinical + mri_ld + radiomics,
        "Clinical + deep MRI": clinical + mri_ld + deep_mri,
        "Clinical + MRI + omics": clinical + mri_ld + radiomics + deep_mri + omics,
        "Full multimodal ensemble": clinical + mri_ld + radiomics + deep_mri + omics,
    }


def evaluate_group(df: pd.DataFrame, name: str, features: list[str]) -> dict:
    train_df, test_df = train_test_split(
        df, test_size=0.25, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    _, selected = select_features(
        train_df[[*features, TARGET]], target=TARGET, exclude=[TARGET],
        p_threshold=0.20, corr_threshold=0.85,
    )
    if not selected:
        selected = features
    model = UnweightedPCRensemble(random_state=RANDOM_STATE)
    model.fit(train_df[selected], train_df[TARGET])
    prob = model.predict_proba(test_df[selected])[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "model": name,
        "data_used": "synthetic demo",
        "n_test": len(test_df),
        "auc": round(float(roc_auc_score(test_df[TARGET], prob)), 4),
        "accuracy": round(float(accuracy_score(test_df[TARGET], pred)), 4),
        "n_selected_features": len(selected),
        "selected_features": "; ".join(selected),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = add_longitudinal_mri_deltas(df)
    df = filter_tnbc(df)
    results = [evaluate_group(df, name, features) for name, features in group_columns(df).items()]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
