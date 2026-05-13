"""Reusable plotting helpers for EDA."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"0": "#4C72B0", "1": "#C44E52"}  # not-depressed / depressed


def plot_numeric_distribution(df: pd.DataFrame, column: str, ax=None):
    ax = ax or plt.gca()
    sns.histplot(df[column].dropna(), kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"Distribution of {column}")
    return ax


def plot_categorical_counts(df: pd.DataFrame, column: str, top_n: int | None = None, ax=None):
    ax = ax or plt.gca()
    order = df[column].value_counts().index
    if top_n:
        order = order[:top_n]
    sns.countplot(data=df[df[column].isin(order)], x=column, order=order, ax=ax, color="#4C72B0")
    ax.set_title(f"Counts of {column}" + (f" (top {top_n})" if top_n else ""))
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_numeric_vs_target(df: pd.DataFrame, column: str, target: str = "Depression", ax=None):
    ax = ax or plt.gca()
    sns.boxplot(data=df, x=target, y=column, ax=ax, palette=PALETTE)
    ax.set_title(f"{column} by {target}")
    return ax


def plot_categorical_vs_target(df: pd.DataFrame, column: str, target: str = "Depression", ax=None):
    """Stacked bar of % depressed per category."""
    ax = ax or plt.gca()
    grouped = (
        df.groupby(column)[target]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    sns.barplot(data=grouped, x=column, y=target, ax=ax, color="#C44E52")
    ax.set_title(f"Share with {target}=1 by {column}")
    ax.set_ylabel(f"P({target}=1)")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_correlation_heatmap(df: pd.DataFrame, columns: list[str], ax=None):
    ax = ax or plt.gca()
    corr = df[columns].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    return ax


def save_fig(fig, name: str, figures_dir: Path | str = "reports/figures"):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=120, bbox_inches="tight")
