"""Evaluate pCR predictions."""
from __future__ import annotations
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="outputs/predictions.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    y = df["y_true"]
    p = df["pcr_probability"]
    pred = (p >= 0.5).astype(int)
    print({
        "auc": roc_auc_score(y, p),
        "accuracy": accuracy_score(y, pred),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    })

if __name__ == "__main__":
    main()
