import pandas as pd
import numpy as np
from src.preprocessing import encode_sleep_duration, encode_dietary_habits, group_rare_categories


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
