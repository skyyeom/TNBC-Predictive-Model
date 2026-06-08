"""Evaluation plots for real TNBC pCR experiments."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def save_roc(y_true, y_prob, out_png: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = auc(fpr, tpr)
    plt.figure(figsize=(6, 5), dpi=180)
    plt.plot(fpr, tpr, label=f"AUC = {score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_pr(y_true, y_prob, out_png: Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    score = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 5), dpi=180)
    plt.plot(recall, precision, label=f"AP = {score:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_confusion(y_true, y_prob, out_png: Path, title: str, threshold: float = 0.5) -> None:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Residual disease", "pCR"])
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=180)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def save_feature_importance(model, X, y, out_csv: Path, out_png: Path, title: str, n_top: int = 20) -> None:
    """Manual permutation importance using model.predict_proba.

    This avoids estimator-tag compatibility issues across scikit-learn versions.
    """
    rng = np.random.default_rng(42)
    base_prob = model.predict_proba(X)[:, 1]
    base_auc = roc_auc_score(y, base_prob) if len(np.unique(y)) == 2 else np.nan
    rows = []
    for col in X.columns:
        scores = []
        for _ in range(20):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            prob = model.predict_proba(Xp)[:, 1]
            score = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan
            scores.append(base_auc - score)
        rows.append({
            "feature": col,
            "importance_mean": float(np.nanmean(scores)),
            "importance_std": float(np.nanstd(scores)),
        })
    imp = pd.DataFrame(rows).sort_values("importance_mean", ascending=False)
    imp.to_csv(out_csv, index=False)
    top = imp.head(n_top).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.28 * len(top))), dpi=180)
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Permutation importance, ROC-AUC decrease")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def write_metrics(metrics: dict, out_json: Path) -> None:
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
