"""Model factory and cross-validation helper."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor

RANDOM_STATE = 42


def get_models() -> dict[str, Pipeline]:
    """Return name → Pipeline(preprocessor, classifier)."""
    raw_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=5,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor()), ("model", clf)])
        for name, clf in raw_models.items()
    }


def run_cv(models: dict[str, Pipeline], X, y, cv_splits: int = 5) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in models.items():
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring=["roc_auc", "f1", "accuracy"],
            n_jobs=-1, return_train_score=False,
        )
        rows.append({
            "Model": name,
            "ROC-AUC mean": scores["test_roc_auc"].mean(),
            "ROC-AUC std": scores["test_roc_auc"].std(),
            "F1 mean": scores["test_f1"].mean(),
            "Accuracy mean": scores["test_accuracy"].mean(),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC mean", ascending=False).reset_index(drop=True)
