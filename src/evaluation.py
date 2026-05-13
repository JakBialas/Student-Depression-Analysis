"""Evaluation metrics and plots."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc,
)


def metrics_table(models: dict, X_test, y_test) -> pd.DataFrame:
    """Compute Accuracy/Precision/Recall/F1/ROC-AUC for each fitted model."""
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_proba),
        })
    return pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def plot_confusion_matrices(models: dict, X_test, y_test, ncols: int = 3):
    n = len(models)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, (name, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_title(name)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for ax in axes[len(models):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_roc_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend()
    return fig


def plot_pr_curve(model, X_test, y_test, model_name: str):
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {model_name}")
    return fig


def feature_importance_df(pipeline, top_n: int = 15) -> pd.DataFrame:
    """Extract top-N feature importances from a fitted Pipeline(preprocessor, tree-model)."""
    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    names = pre.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        raise ValueError("Model has neither feature_importances_ nor coef_")
    return (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def plot_feature_importance(pipeline, model_name: str, top_n: int = 15, ax=None):
    df = feature_importance_df(pipeline, top_n=top_n)
    ax = ax or plt.gca()
    sns.barplot(data=df, y="feature", x="importance", ax=ax, color="#4C72B0")
    ax.set_title(f"Top {top_n} features — {model_name}")
    return ax
