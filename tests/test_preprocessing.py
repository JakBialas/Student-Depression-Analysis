import pandas as pd
import numpy as np
from src.data_loader import load_raw
from src.preprocessing import (
    encode_sleep_duration,
    encode_dietary_habits,
    group_rare_categories,
    prepare_features,
    build_preprocessor,
)


def test_encode_sleep_duration_known_buckets():
    s = pd.Series(["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
    out = encode_sleep_duration(s)
    assert list(out) == [4.5, 5.5, 7.5, 8.5]


def test_encode_sleep_duration_unknown_to_nan():
    s = pd.Series(["Others", "weird value"])
    out = encode_sleep_duration(s)
    assert out.isna().all()


def test_encode_dietary_habits_ordinal():
    s = pd.Series(["Healthy", "Moderate", "Unhealthy", "Others"])
    out = encode_dietary_habits(s)
    assert list(out[:3]) == [2, 1, 0]
    assert pd.isna(out.iloc[3])


def test_group_rare_categories_threshold():
    s = pd.Series(["A"] * 10 + ["B"] * 10 + ["C"] * 2 + ["D"] * 1)
    out = group_rare_categories(s, min_count=5)
    assert set(out.unique()) == {"A", "B", "Other"}
    assert (out == "Other").sum() == 3


def test_prepare_features_shapes_and_dtypes():
    df = load_raw()
    X, y = prepare_features(df)
    assert "Depression" not in X.columns
    assert "id" not in X.columns
    assert y.dtype.kind == "i"
    assert len(X) == len(y) == len(df)
    # binary-ish columns are numeric now
    assert X["Have you ever had suicidal thoughts ?"].dropna().isin([0.0, 1.0]).all()


def test_build_preprocessor_runs_end_to_end():
    df = load_raw()
    X, y = prepare_features(df)
    pre = build_preprocessor()
    Xt = pre.fit_transform(X, y)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] > len(X.columns)  # one-hot expanded
