#!/usr/bin/env python
"""Run real multicohort TNBC pCR experiments.

Expected input: a harmonized CSV with one row per patient/study timepoint summary.
The script does not download protected TCIA/NBIA data. Place your harmonized table at
`data/real/harmonized_tnbc_pcr.csv` or pass --table.

Required columns:
patient_id,dataset,pcr,er_status,pr_status,her2_status

Recommended feature prefixes:
clinical_, mri_, delta_, radiomics_, deep_, omics_, treatment_
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_harmonization import filter_tnbc, add_longitudinal_mri_deltas
from src.feature_engineering import select_features
from src.modeling import UnweightedPCRensemble
from src.evaluation_plots import (
    classification_metrics, save_roc, save_pr, save_confusion,
    save_feature_importance, write_metrics,
)

FEATURE_PREFIXES = ("clinical_", "mri_", "delta_", "radiomics_", "deep_", "omics_", "treatment_")
DEFAULT_TRAIN = ["ISPY1", "ISPY2"]
DEFAULT_EXTERNAL = ["ACRIN6698", "DUKE"]


def collect_features(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c.startswith(FEATURE_PREFIXES):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
    return cols


def validate_input(df: pd.DataFrame) -> None:
    required = {"patient_id", "dataset", "pcr", "er_status", "pr_status", "her2_status"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    features = collect_features(df)
    if len(features) < 2:
        raise ValueError("No usable numeric feature columns found. Use prefixes clinical_, mri_, delta_, radiomics_, deep_, omics_, treatment_.")
    for d in sorted(df["dataset"].dropna().unique()):
        sub = df[df["dataset"] == d]
        if sub["pcr"].nunique() < 2:
            print(f"WARNING: dataset {d} has only one outcome class and cannot produce ROC-AUC by itself.")


def train_eval(train_df, test_df, features, label, outdir):
    # Feature selection only inside training data, following Sammut-style filtering.
    _, selected = select_features(
        train_df[[*features, label]], target=label, exclude=[label],
        p_threshold=0.2, corr_threshold=0.85,
    )
    if not selected:
        selected = features
    model = UnweightedPCRensemble(random_state=42)
    model.fit(train_df[selected], train_df[label])
    prob = model.predict_proba(test_df[selected])[:, 1]
    metrics = classification_metrics(test_df[label], prob)
    preds = test_df[["patient_id", "dataset", label]].copy()
    preds["pcr_probability"] = prob
    preds["pcr_predicted"] = (prob >= 0.5).astype(int)
    preds.to_csv(outdir / f"predictions_{'_'.join(test_df['dataset'].astype(str).unique())}.csv", index=False)
    save_roc(test_df[label], prob, outdir / "roc_curve.png", "External validation ROC curve")
    save_pr(test_df[label], prob, outdir / "precision_recall_curve.png", "External validation precision-recall curve")
    save_confusion(test_df[label], prob, outdir / "confusion_matrix.png", "External validation confusion matrix")
    save_feature_importance(model, test_df[selected], test_df[label], outdir / "feature_importance.csv", outdir / "feature_importance.png", "Permutation feature importance")
    joblib.dump({"model": model, "features": selected}, outdir / "real_pcr_ensemble.joblib")
    metrics["selected_features"] = selected
    metrics["n_selected_features"] = len(selected)
    write_metrics(metrics, outdir / "real_metrics.json")
    return metrics


def run_ablation(df, train_datasets, external_datasets, label, outdir):
    groups = {
        "Clinical only": ["clinical_", "treatment_"],
        "MRI only": ["mri_", "delta_", "radiomics_", "deep_"],
        "Clinical + MRI": ["clinical_", "treatment_", "mri_", "delta_", "radiomics_", "deep_"],
        "Clinical + MRI + Omics": ["clinical_", "treatment_", "mri_", "delta_", "radiomics_", "deep_", "omics_"],
    }
    rows = []
    train_df = df[df["dataset"].isin(train_datasets)]
    test_df = df[df["dataset"].isin(external_datasets)]
    for name, prefixes in groups.items():
        feats = [c for c in df.columns if any(c.startswith(p) for p in prefixes) and pd.api.types.is_numeric_dtype(df[c])]
        if len(feats) < 2:
            rows.append({"model": name, "status": "skipped: insufficient features"})
            continue
        try:
            model = UnweightedPCRensemble(random_state=42)
            _, selected = select_features(train_df[[*feats, label]], target=label, exclude=[label], p_threshold=0.2, corr_threshold=0.85)
            selected = selected or feats
            model.fit(train_df[selected], train_df[label])
            prob = model.predict_proba(test_df[selected])[:, 1]
            m = classification_metrics(test_df[label], prob)
            rows.append({"model": name, "status": "complete", **{k: v for k, v in m.items() if k in ["n", "auc", "pr_auc", "accuracy", "f1"]}, "n_features": len(selected)})
        except Exception as e:
            rows.append({"model": name, "status": f"failed: {e}"})
    pd.DataFrame(rows).to_csv(outdir / "real_ablation_results.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default="data/real/harmonized_tnbc_pcr.csv")
    p.add_argument("--outdir", default="outputs/real")
    p.add_argument("--train-datasets", nargs="+", default=DEFAULT_TRAIN)
    p.add_argument("--external-datasets", nargs="+", default=DEFAULT_EXTERNAL)
    p.add_argument("--no-tnbc-filter", action="store_true")
    args = p.parse_args()
    table = ROOT / args.table if not Path(args.table).is_absolute() else Path(args.table)
    outdir = ROOT / args.outdir if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not table.exists():
        raise FileNotFoundError(f"Real harmonized table not found: {table}. See data/real/README.md and data/real/harmonized_tnbc_pcr_template.csv")
    df = pd.read_csv(table)
    validate_input(df)
    df = add_longitudinal_mri_deltas(df)
    if not args.no_tnbc_filter:
        df = filter_tnbc(df)
    features = collect_features(df)
    train_df = df[df["dataset"].isin(args.train_datasets)].copy()
    test_df = df[df["dataset"].isin(args.external_datasets)].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Train or external validation split is empty. Check dataset names in the harmonized CSV.")
    if train_df["pcr"].nunique() < 2 or test_df["pcr"].nunique() < 2:
        raise ValueError("Train and external validation splits must each contain both pCR and residual disease cases.")
    metrics = train_eval(train_df, test_df, features, "pcr", outdir)
    run_ablation(df, args.train_datasets, args.external_datasets, "pcr", outdir)
    print("Real-data experiment complete. Metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
