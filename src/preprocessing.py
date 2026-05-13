"""Preprocessing helpers: encoders and the full ColumnTransformer factory."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

SLEEP_DURATION_MAP = {
    "Less than 5 hours": 4.5,
    "5-6 hours": 5.5,
    "7-8 hours": 7.5,
    "More than 8 hours": 8.5,
}

DIETARY_MAP = {"Unhealthy": 0, "Moderate": 1, "Healthy": 2}


def encode_sleep_duration(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.strip("'").str.strip('"')
    return cleaned.map(SLEEP_DURATION_MAP).astype("float64")


def encode_dietary_habits(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    return cleaned.map(DIETARY_MAP).astype("float64")


def encode_yes_no(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype("float64")


def encode_gender(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({"male": 0, "female": 1}).astype("float64")


def group_rare_categories(series: pd.Series, min_count: int = 50, other_label: str = "Other") -> pd.Series:
    counts = series.value_counts()
    keep = counts[counts >= min_count].index
    return series.where(series.isin(keep), other_label)
