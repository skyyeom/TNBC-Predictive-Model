"""Train the TNBC pCR ensemble model."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from .dataset_harmonization import filter_tnbc, add_longitudinal_mri_deltas
from .feature_engineering import select_features
from .modeling import UnweightedPCRensemble
from .multimodal_fusion import collect_feature_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    path = Path(cfg["paths"]["processed_table"])
    outdir = Path(cfg["paths"]["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    df = add_longitudinal_mri_deltas(df)
    if cfg.get("subtype_filter", {}).get("require_tnbc", True):
        df = filter_tnbc(df)

    target = cfg["outcome"]
    exclude = [target, cfg["id_column"], "dataset"]
    candidate_features = collect_feature_columns(df)
    uni, selected = select_features(
        df[[*candidate_features, target]], target=target, exclude=[target],
        p_threshold=cfg["selection"]["univariable_p_threshold"],
        corr_threshold=cfg["selection"]["correlation_threshold"],
    )
    if not selected:
        selected = candidate_features

    train_df, test_df = train_test_split(
        df, test_size=cfg["validation"]["test_size"], stratify=df[target],
        random_state=cfg["random_state"],
    )

    model = UnweightedPCRensemble(random_state=cfg["random_state"])
    model.fit(train_df[selected], train_df[target])
    prob = model.predict_proba(test_df[selected])[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "n_total_tnbc": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "auc": float(roc_auc_score(test_df[target], prob)),
        "accuracy": float(accuracy_score(test_df[target], pred)),
        "univariable_selected": uni,
        "final_features": selected,
    }
    pd.DataFrame({"patient_id": test_df[cfg["id_column"]], "y_true": test_df[target], "pcr_probability": prob}).to_csv(outdir / "predictions.csv", index=False)
    pd.Series(metrics, dtype="object").to_json(outdir / "metrics.json", indent=2)
    joblib.dump({"model": model, "features": selected, "config": cfg}, outdir / "pcr_ensemble.joblib")
    print(metrics)

if __name__ == "__main__":
    main()
