"""Feature selection matching the Sammut et al. style pipeline."""
from __future__ import annotations
from typing import Iterable, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr


def numeric_feature_columns(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    excluded = set(exclude)
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def univariable_selection(df: pd.DataFrame, features: List[str], target: str, p_threshold: float = 0.20) -> List[str]:
    selected = []
    y = df[target].astype(float)
    for col in features:
        x = df[col].astype(float)
        valid = x.notna() & y.notna()
        if valid.sum() < 10 or x[valid].nunique() < 2:
            continue
        try:
            _, p = pointbiserialr(y[valid], x[valid])
        except Exception:
            p = 1.0
        if np.isfinite(p) and p <= p_threshold:
            selected.append(col)
    return selected


def remove_correlated_features(df: pd.DataFrame, features: List[str], threshold: float = 0.85) -> List[str]:
    if not features:
        return []
    corr = df[features].corr(method="spearman").abs()
    keep = []
    dropped = set()
    for col in corr.columns:
        if col in dropped:
            continue
        keep.append(col)
        correlated = corr.index[(corr[col] > threshold) & (corr.index != col)].tolist()
        dropped.update(correlated)
    return keep


def select_features(df: pd.DataFrame, target: str, exclude: Iterable[str], p_threshold: float, corr_threshold: float) -> Tuple[List[str], List[str]]:
    candidates = numeric_feature_columns(df, exclude=exclude)
    uni = univariable_selection(df, candidates, target, p_threshold)
    final = remove_correlated_features(df, uni, corr_threshold)
    return uni, final
