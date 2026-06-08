"""Unweighted ensemble classifier: elastic-net LR + SVM + RF."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_lr(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5,
            max_iter=5000, class_weight="balanced", random_state=random_state,
        )),
    ])


def make_svm(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
    ])


def make_rf(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            random_state=random_state,
        )),
    ])

@dataclass
class UnweightedPCRensemble(BaseEstimator, ClassifierMixin):
    random_state: int = 42
    _estimator_type: str = "classifier"

    def __post_init__(self):
        self.models_ = [make_lr(self.random_state), make_svm(self.random_state), make_rf(self.random_state)]

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.models_ = [make_lr(self.random_state), make_svm(self.random_state), make_rf(self.random_state)]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        probs = np.stack([m.predict_proba(X)[:, 1] for m in self.models_], axis=1).mean(axis=1)
        return np.vstack([1 - probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
