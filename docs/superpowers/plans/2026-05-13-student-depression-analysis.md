# Student Depression Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portfolio-grade data science project that analyses the Student Depression dataset end-to-end (EDA → preprocessing → modeling → evaluation), organized as three Jupyter notebooks driven by reusable `src/` modules.

**Architecture:** Three notebooks (`01_eda`, `02_preprocessing`, `03_modeling`) orchestrate the analysis. Pure logic lives in `src/` (preprocessing transforms, plotting helpers, model factory, evaluation utilities). Processed train/test splits are persisted to disk so the modeling notebook can start from a clean entry point. All randomness is seeded; the preprocessor is wrapped in a sklearn `Pipeline` to prevent leakage during CV/tuning.

**Tech Stack:** Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`, `jupyter`. Light testing via `pytest` for pure-logic functions only.

**Testing note:** This is a notebook-driven portfolio project. Pure-logic helpers (preprocessing transforms) get small pytest smoke tests in `tests/`. Plotting helpers and notebooks are validated by running cells and visually inspecting output (described per task).

---

## File Structure

```
Student-Depression-Analysis/
├── data/
│   ├── raw/student_depression_dataset.csv         # existing, moved here in Task 1
│   └── processed/                                  # written by notebook 02
│       ├── X_train.parquet
│       ├── X_test.parquet
│       ├── y_train.parquet
│       └── y_test.parquet
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # load + minimal cleaning of raw CSV
│   ├── preprocessing.py      # encoders, build_preprocessor() → ColumnTransformer
│   ├── visualization.py      # reusable EDA plotting helpers
│   ├── models.py             # model factory + CV runner
│   └── evaluation.py         # metric tables, confusion matrix, ROC, feature importance plots
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py # pytest smoke tests for encoders
├── reports/figures/          # PNG exports for README
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Task 1: Project setup (git, dirs, deps, dataset relocation)

**Files:**
- Create: `.gitignore`, `requirements.txt`, `src/__init__.py`, `tests/__init__.py`, `README.md` (stub)
- Move: `student_depression_dataset.csv` → `data/raw/student_depression_dataset.csv`

- [ ] **Step 1: Initialize git and create directory tree**

```bash
cd /home/kuba/Projects/Student-Depression-Analysis
git init
mkdir -p data/raw data/processed notebooks src tests reports/figures docs/superpowers/specs docs/superpowers/plans
mv student_depression_dataset.csv data/raw/
touch src/__init__.py tests/__init__.py
```

Expected: `ls data/raw/` shows `student_depression_dataset.csv`.

- [ ] **Step 2: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.ipynb_checkpoints/

# Data / processed outputs
data/processed/

# OS / editor
.DS_Store
.idea/
.vscode/
```

(`data/raw/` is kept under version control because the dataset is small (~3 MB) and the project must be reproducible from a fresh clone.)

- [ ] **Step 3: Create `requirements.txt`**

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.13
jupyter>=1.0
pyarrow>=14.0
pytest>=7.4
```

- [ ] **Step 4: Create README stub**

```markdown
# Student Depression Analysis

Portfolio data science project analysing the Student Depression dataset.
Full description, results, and figures will be added once the analysis is complete.
```

- [ ] **Step 5: First commit**

```bash
git add .gitignore requirements.txt README.md src/__init__.py tests/__init__.py data/raw/student_depression_dataset.csv docs/
git commit -m "chore: project scaffold, dataset, and dependency manifest"
```

Expected: `git log --oneline` shows one commit.

---

## Task 2: `src/data_loader.py`

**Files:**
- Create: `src/data_loader.py`

- [ ] **Step 1: Write `data_loader.py`**

```python
"""Load the Student Depression dataset."""
from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "student_depression_dataset.csv"


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df
```

- [ ] **Step 2: Verify it loads**

```bash
python -c "from src.data_loader import load_raw; df = load_raw(); print(df.shape); print(df.columns.tolist())"
```

Expected: shape around `(27901, 18)` and the column list including `Depression`.

- [ ] **Step 3: Commit**

```bash
git add src/data_loader.py
git commit -m "feat(data): add raw CSV loader"
```

---

## Task 3: `src/visualization.py` — reusable plotting helpers

**Files:**
- Create: `src/visualization.py`

- [ ] **Step 1: Write `visualization.py`**

```python
"""Reusable plotting helpers for EDA."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#4C72B0", 1: "#C44E52"}  # not-depressed / depressed


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
```

- [ ] **Step 2: Smoke-check the import**

```bash
python -c "from src.visualization import plot_numeric_distribution, plot_categorical_counts, plot_numeric_vs_target, plot_categorical_vs_target, plot_correlation_heatmap, save_fig; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/visualization.py
git commit -m "feat(viz): add EDA plotting helpers"
```

---

## Task 4: Notebook 01 — EDA (sanity checks + univariate)

**Files:**
- Create: `notebooks/01_eda.ipynb`

For all notebook tasks: create the notebook with `jupyter nbconvert --to notebook` from a script, or use Jupyter directly. The cells listed are markdown / code in order.

- [ ] **Step 1: Create `01_eda.ipynb` with the cells below**

**Cell 1 (markdown):**
```markdown
# Exploratory Data Analysis — Student Depression Dataset

Goal: understand the dataset, identify data quality issues, and surface the strongest signals related to depression. Findings here drive the preprocessing in `02_preprocessing.ipynb`.
```

**Cell 2 (code) — imports and load:**
```python
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_raw
from src.visualization import (
    plot_numeric_distribution,
    plot_categorical_counts,
    plot_numeric_vs_target,
    plot_categorical_vs_target,
    plot_correlation_heatmap,
    save_fig,
)

df = load_raw()
df.shape
```

**Cell 3 (markdown):**
```markdown
## 1. Sanity checks
```

**Cell 4 (code):**
```python
df.head()
```

**Cell 5 (code):**
```python
df.info()
```

**Cell 6 (code) — missing values + duplicates:**
```python
print("Missing values per column:")
print(df.isna().sum())
print("\nDuplicate rows:", df.duplicated().sum())
```

**Cell 7 (code) — describe + target balance:**
```python
print(df.describe(include="all").T)
print("\nTarget distribution:")
print(df["Depression"].value_counts(normalize=True).rename("share"))
```

**Cell 8 (code) — Profession check (project assumes students):**
```python
df["Profession"].value_counts()
```

**Cell 9 (markdown):**
```markdown
## 2. Univariate distributions
```

**Cell 10 (code) — numeric distributions grid:**
```python
numeric_cols = ["Age", "CGPA", "Academic Pressure", "Work/Study Hours", "Financial Stress", "Study Satisfaction"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flat, numeric_cols):
    plot_numeric_distribution(df, col, ax=ax)
fig.tight_layout()
save_fig(fig, "01_numeric_distributions", figures_dir="../reports/figures")
plt.show()
```

**Cell 11 (code) — categorical countplots:**
```python
cat_cols = ["Gender", "Sleep Duration", "Dietary Habits", "Family History of Mental Illness",
            "Have you ever had suicidal thoughts ?"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, cat_cols):
    plot_categorical_counts(df, col, ax=ax)
for ax in axes.flat[len(cat_cols):]:
    ax.axis("off")
fig.tight_layout()
save_fig(fig, "01_categorical_counts", figures_dir="../reports/figures")
plt.show()
```

**Cell 12 (code) — top cities:**
```python
fig, ax = plt.subplots(figsize=(10, 5))
plot_categorical_counts(df, "City", top_n=15, ax=ax)
fig.tight_layout()
save_fig(fig, "01_top_cities", figures_dir="../reports/figures")
plt.show()
```

- [ ] **Step 2: Run all cells top to bottom**

Expected:
- Shape ~ `(27901, 18)`
- `Depression` roughly balanced (≈ 40/60 or similar — record the actual share)
- All plots render and three PNGs land in `reports/figures/`

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_eda.ipynb reports/figures/01_*.png
git commit -m "feat(eda): notebook part 1 — sanity checks and univariate distributions"
```

---

## Task 5: Notebook 01 — EDA (bivariate + conclusions)

**Files:**
- Modify: `notebooks/01_eda.ipynb` (append cells)

- [ ] **Step 1: Append the cells below to `01_eda.ipynb`**

**Cell 13 (markdown):**
```markdown
## 3. Bivariate analysis vs Depression
```

**Cell 14 (code) — numeric vs target:**
```python
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, col in zip(axes.flat, numeric_cols):
    plot_numeric_vs_target(df, col, ax=ax)
fig.tight_layout()
save_fig(fig, "01_numeric_vs_target", figures_dir="../reports/figures")
plt.show()
```

**Cell 15 (code) — categorical vs target:**
```python
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, cat_cols):
    plot_categorical_vs_target(df, col, ax=ax)
for ax in axes.flat[len(cat_cols):]:
    ax.axis("off")
fig.tight_layout()
save_fig(fig, "01_categorical_vs_target", figures_dir="../reports/figures")
plt.show()
```

**Cell 16 (code) — correlation heatmap:**
```python
corr_cols = ["Age", "CGPA", "Academic Pressure", "Work Pressure", "Study Satisfaction",
             "Job Satisfaction", "Work/Study Hours", "Financial Stress", "Depression"]
fig, ax = plt.subplots(figsize=(10, 8))
plot_correlation_heatmap(df, corr_cols, ax=ax)
fig.tight_layout()
save_fig(fig, "01_correlation_heatmap", figures_dir="../reports/figures")
plt.show()
```

**Cell 17 (code) — suicidal thoughts crosstab (expected strong predictor):**
```python
ct = pd.crosstab(df["Have you ever had suicidal thoughts ?"], df["Depression"], normalize="index")
print(ct)
fig, ax = plt.subplots(figsize=(6, 4))
ct.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_title("Depression rate by history of suicidal thoughts")
ax.set_ylabel("Share")
fig.tight_layout()
save_fig(fig, "01_suicidal_vs_depression", figures_dir="../reports/figures")
plt.show()
```

**Cell 18 (markdown) — conclusions (fill values after running):**
```markdown
## 4. Conclusions from EDA

1. Target balance: `Depression=1` shares roughly XX% of records → moderately balanced, no need for class-weight tricks but worth confirming with stratified CV.
2. `Have you ever had suicidal thoughts ?` is the dominant single predictor — categorical encoding to binary is essential.
3. `Academic Pressure`, `Financial Stress`, and `Work/Study Hours` show the strongest numeric separation between classes.
4. `CGPA` and `Job Satisfaction` show weaker signal; `Job Satisfaction` is near-zero because the dataset is almost entirely students.
5. `Sleep Duration` is ordinal-looking (`<5h`, `5-6h`, `7-8h`, `>8h`) — encode as ordinal, not one-hot.
6. `City` has high cardinality; rare cities should be grouped into `Other` to control feature dimensionality.
7. No missing values worth imputing (or: X% in column Y — handle in preprocessing).

These observations drive the encoding choices in `02_preprocessing.ipynb`.
```

- [ ] **Step 2: Run new cells and update the conclusion numbers based on actual outputs**

Replace the `XX%` and similar placeholders in Cell 18 with the values you observed.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_eda.ipynb reports/figures/01_*.png
git commit -m "feat(eda): notebook part 2 — bivariate analysis and conclusions"
```

---

## Task 6: `src/preprocessing.py` — pure encoders + pytest smoke tests

**Files:**
- Create: `src/preprocessing.py`
- Create: `tests/test_preprocessing.py`

- [ ] **Step 1: Write the failing pytest first**

```python
# tests/test_preprocessing.py
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
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
pytest tests/test_preprocessing.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` because `src/preprocessing.py` doesn't exist yet.

- [ ] **Step 3: Implement the encoders in `src/preprocessing.py`**

```python
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
```

- [ ] **Step 4: Run the tests — they should pass**

```bash
pytest tests/test_preprocessing.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): encoders for sleep/diet/yes-no/gender + rare-category grouping"
```

---

## Task 7: `src/preprocessing.py` — add `build_preprocessor` + `prepare_features`

**Files:**
- Modify: `src/preprocessing.py`

- [ ] **Step 1: Append `prepare_features` and `build_preprocessor` to `src/preprocessing.py`**

```python
def prepare_features(df: pd.DataFrame, rare_city_min: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    """Apply manual ordinal/binary encodings and return (X, y)."""
    data = df.copy()
    if "id" in data.columns:
        data = data.drop(columns=["id"])

    data["Sleep Duration"] = encode_sleep_duration(data["Sleep Duration"])
    data["Dietary Habits"] = encode_dietary_habits(data["Dietary Habits"])
    data["Have you ever had suicidal thoughts ?"] = encode_yes_no(data["Have you ever had suicidal thoughts ?"])
    data["Family History of Mental Illness"] = encode_yes_no(data["Family History of Mental Illness"])
    data["Gender"] = encode_gender(data["Gender"])

    data["City"] = group_rare_categories(data["City"], min_count=rare_city_min)

    y = data["Depression"].astype(int)
    X = data.drop(columns=["Depression"])
    return X, y


NUMERIC_FEATURES = [
    "Gender", "Age", "Academic Pressure", "Work Pressure", "CGPA",
    "Study Satisfaction", "Job Satisfaction", "Sleep Duration",
    "Dietary Habits", "Have you ever had suicidal thoughts ?",
    "Work/Study Hours", "Financial Stress", "Family History of Mental Illness",
]
CATEGORICAL_FEATURES = ["City", "Profession", "Degree"]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])
```

- [ ] **Step 2: Add a smoke test in `tests/test_preprocessing.py`**

Append:
```python
from src.data_loader import load_raw
from src.preprocessing import prepare_features, build_preprocessor


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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_preprocessing.py -v
```

Expected: 6 passed.

- [ ] **Step 4: Commit**

```bash
git add src/preprocessing.py tests/test_preprocessing.py
git commit -m "feat(preprocessing): prepare_features and ColumnTransformer factory"
```

---

## Task 8: Notebook 02 — Preprocessing

**Files:**
- Create: `notebooks/02_preprocessing.ipynb`

- [ ] **Step 1: Create `02_preprocessing.ipynb` with the cells below**

**Cell 1 (markdown):**
```markdown
# Preprocessing — Student Depression Dataset

Apply manual ordinal/binary encodings, group rare cities, build the sklearn `ColumnTransformer`, split into train/test, and persist the splits for `03_modeling.ipynb`.
```

**Cell 2 (code) — imports + load:**
```python
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.data_loader import load_raw
from src.preprocessing import prepare_features, build_preprocessor

RANDOM_STATE = 42
df = load_raw()
df.shape
```

**Cell 3 (markdown):**
```markdown
## 1. Manual encodings
```

**Cell 4 (code):**
```python
X, y = prepare_features(df)
print("X shape:", X.shape)
print("y distribution:")
print(y.value_counts(normalize=True))
X.head()
```

**Cell 5 (code) — check encoded columns:**
```python
encoded_cols = ["Gender", "Sleep Duration", "Dietary Habits",
                "Have you ever had suicidal thoughts ?",
                "Family History of Mental Illness"]
X[encoded_cols].describe()
```

**Cell 6 (code) — confirm rare-city grouping:**
```python
print("Unique cities after grouping:", X["City"].nunique())
print(X["City"].value_counts().head(10))
print("\n'Other' count:", (X["City"] == "Other").sum())
```

**Cell 7 (markdown):**
```markdown
## 2. Train/test split (stratified)
```

**Cell 8 (code):**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, "| Test:", X_test.shape)
print("Train target share:", y_train.mean())
print("Test target share:", y_test.mean())
```

**Cell 9 (markdown):**
```markdown
## 3. Build and inspect the preprocessor
```

**Cell 10 (code):**
```python
preprocessor = build_preprocessor()
X_train_t = preprocessor.fit_transform(X_train, y_train)
X_test_t = preprocessor.transform(X_test)
print("Transformed train shape:", X_train_t.shape)
print("Transformed test shape:", X_test_t.shape)
```

**Cell 11 (code) — get feature names for downstream use:**
```python
feature_names = preprocessor.get_feature_names_out()
print("Number of features after preprocessing:", len(feature_names))
feature_names[:20]
```

**Cell 12 (markdown):**
```markdown
## 4. Persist the splits

We save the pre-transform `X` so that the modeling notebook can wrap the `preprocessor` inside a `Pipeline(preprocessor, model)`. That keeps the preprocessor inside cross-validation and prevents leakage.
```

**Cell 13 (code):**
```python
from pathlib import Path
out_dir = Path("../data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

X_train.to_parquet(out_dir / "X_train.parquet")
X_test.to_parquet(out_dir / "X_test.parquet")
y_train.to_frame("Depression").to_parquet(out_dir / "y_train.parquet")
y_test.to_frame("Depression").to_parquet(out_dir / "y_test.parquet")
print("Saved to", out_dir.resolve())
```

- [ ] **Step 2: Run all cells top to bottom**

Expected: four parquet files appear in `data/processed/`. Print outputs show consistent shapes.

- [ ] **Step 3: Commit**

```bash
git add notebooks/02_preprocessing.ipynb
git commit -m "feat(preprocessing): notebook 02 — apply encodings, split, persist data"
```

---

## Task 9: `src/models.py` — model factory + CV runner

**Files:**
- Create: `src/models.py`

- [ ] **Step 1: Write `src/models.py`**

```python
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
```

- [ ] **Step 2: Smoke check the import**

```bash
python -c "from src.models import get_models, run_cv; print(list(get_models()))"
```

Expected: `['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'XGBoost']`.

- [ ] **Step 3: Commit**

```bash
git add src/models.py
git commit -m "feat(models): model factory and CV runner with sklearn Pipelines"
```

---

## Task 10: `src/evaluation.py` — metrics and evaluation plots

**Files:**
- Create: `src/evaluation.py`

- [ ] **Step 1: Write `src/evaluation.py`**

```python
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
```

- [ ] **Step 2: Smoke-check the import**

```bash
python -c "from src.evaluation import metrics_table, plot_confusion_matrices, plot_roc_curves, plot_pr_curve, feature_importance_df, plot_feature_importance; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/evaluation.py
git commit -m "feat(evaluation): metrics table, confusion/ROC/PR/importance plots"
```

---

## Task 11: Notebook 03 — baseline models + CV

**Files:**
- Create: `notebooks/03_modeling.ipynb`

- [ ] **Step 1: Create `03_modeling.ipynb` with the cells below**

**Cell 1 (markdown):**
```markdown
# Modeling and Evaluation — Student Depression

Train five classifiers (LogReg, KNN, Decision Tree, Random Forest, XGBoost) inside leak-free sklearn Pipelines, compare them with stratified 5-fold CV, tune the top two, and evaluate on the held-out test set.
```

**Cell 2 (code) — load processed data:**
```python
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models import get_models, run_cv
from src.evaluation import (
    metrics_table, plot_confusion_matrices, plot_roc_curves, plot_pr_curve,
    feature_importance_df, plot_feature_importance,
)
from src.visualization import save_fig

RANDOM_STATE = 42
proc = Path("../data/processed")
X_train = pd.read_parquet(proc / "X_train.parquet")
X_test = pd.read_parquet(proc / "X_test.parquet")
y_train = pd.read_parquet(proc / "y_train.parquet")["Depression"]
y_test = pd.read_parquet(proc / "y_test.parquet")["Depression"]
print("Train:", X_train.shape, "Test:", X_test.shape)
```

**Cell 3 (markdown):**
```markdown
## 1. Cross-validation on five baseline models
```

**Cell 4 (code) — run CV (may take a couple of minutes):**
```python
models = get_models()
cv_results = run_cv(models, X_train, y_train, cv_splits=5)
cv_results
```

- [ ] **Step 2: Run cells, verify CV completes**

Expected: a DataFrame with five rows sorted by ROC-AUC. XGBoost and Random Forest should typically be on top; record the exact ranking.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_modeling.ipynb
git commit -m "feat(modeling): notebook 03 part 1 — baseline CV across 5 models"
```

---

## Task 12: Notebook 03 — hyperparameter tuning of top two

**Files:**
- Modify: `notebooks/03_modeling.ipynb`

- [ ] **Step 1: Append the cells below**

**Cell 5 (markdown):**
```markdown
## 2. Hyperparameter tuning — top two models

We tune only the top two CV performers to keep the analysis focused. `RandomizedSearchCV` is used to keep runtime reasonable.
```

**Cell 6 (code):**
```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
top_two = cv_results["Model"].head(2).tolist()
print("Tuning:", top_two)
```

**Cell 7 (code) — parameter grids:**
```python
PARAM_GRIDS = {
    "Random Forest": {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 8, 16, 24],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "model__n_estimators": [200, 400, 600],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
    },
    "Logistic Regression": {
        "model__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
        "model__penalty": ["l2"],
    },
    "KNN": {
        "model__n_neighbors": [5, 10, 15, 25, 40],
        "model__weights": ["uniform", "distance"],
    },
    "Decision Tree": {
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 10, 20],
    },
}
```

**Cell 8 (code) — run tuning:**
```python
tuned = {}
tuning_results = []
for name in top_two:
    print(f"\nTuning {name}...")
    pipe = get_models()[name]
    search = RandomizedSearchCV(
        pipe, PARAM_GRIDS[name],
        n_iter=20, scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best ROC-AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    tuned[name] = search.best_estimator_
    tuning_results.append({
        "Model": name,
        "Best CV ROC-AUC": search.best_score_,
        "Best params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
    })
```

**Cell 9 (code) — tuned summary table:**
```python
tuned_summary = pd.DataFrame(tuning_results)
tuned_summary
```

- [ ] **Step 2: Run cells, record best params**

Expected: each tuned model reports its best CV ROC-AUC and chosen hyperparameters.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_modeling.ipynb
git commit -m "feat(modeling): notebook 03 part 2 — hyperparameter tuning top two models"
```

---

## Task 13: Notebook 03 — final evaluation + feature importance + conclusions

**Files:**
- Modify: `notebooks/03_modeling.ipynb`

- [ ] **Step 1: Append the cells below**

**Cell 10 (markdown):**
```markdown
## 3. Final test-set evaluation

We refit all five baseline models on the full train set and add the tuned versions, then evaluate on the held-out test set.
```

**Cell 11 (code) — refit baselines on full train:**
```python
final_models = {}
baselines = get_models()
for name, pipe in baselines.items():
    pipe.fit(X_train, y_train)
    final_models[name] = pipe
for name, pipe in tuned.items():
    final_models[f"{name} (tuned)"] = pipe  # already fitted by RandomizedSearchCV
print("Final models:", list(final_models))
```

**Cell 12 (code) — metrics table:**
```python
results = metrics_table(final_models, X_test, y_test)
results.to_csv("../reports/figures/03_metrics.csv", index=False)
results
```

**Cell 13 (code) — confusion matrices:**
```python
fig = plot_confusion_matrices(final_models, X_test, y_test, ncols=3)
save_fig(fig, "03_confusion_matrices", figures_dir="../reports/figures")
plt.show()
```

**Cell 14 (code) — ROC curves:**
```python
fig = plot_roc_curves(final_models, X_test, y_test)
save_fig(fig, "03_roc_curves", figures_dir="../reports/figures")
plt.show()
```

**Cell 15 (code) — PR curve for best model:**
```python
best_name = results.iloc[0]["Model"]
print("Best model:", best_name)
fig = plot_pr_curve(final_models[best_name], X_test, y_test, best_name)
save_fig(fig, "03_pr_curve_best", figures_dir="../reports/figures")
plt.show()
```

**Cell 16 (markdown):**
```markdown
## 4. Feature importance
```

**Cell 17 (code) — side-by-side RF + XGBoost importances:**
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
plot_feature_importance(final_models["Random Forest"], "Random Forest", top_n=15, ax=axes[0])
plot_feature_importance(final_models["XGBoost"], "XGBoost", top_n=15, ax=axes[1])
fig.tight_layout()
save_fig(fig, "03_feature_importance", figures_dir="../reports/figures")
plt.show()
```

**Cell 18 (code) — Logistic Regression coefficients for contrast:**
```python
lr_imp = feature_importance_df(final_models["Logistic Regression"], top_n=15)
fig, ax = plt.subplots(figsize=(8, 8))
import seaborn as sns
sns.barplot(data=lr_imp, y="feature", x="importance", ax=ax, color="#4C72B0")
ax.set_title("Top 15 Logistic Regression coefficients (|β|)")
fig.tight_layout()
save_fig(fig, "03_logreg_coefficients", figures_dir="../reports/figures")
plt.show()
```

**Cell 19 (markdown) — conclusions (fill values after running):**
```markdown
## 5. Conclusions

- **Winning model:** `<MODEL>` with test ROC-AUC `<XX.XXX>` and F1 `<XX.XXX>`.
- **Top 5 predictors of depression (consensus across RF + XGBoost):**
  1. History of suicidal thoughts
  2. Academic pressure
  3. Financial stress
  4. Work/study hours
  5. Sleep duration
- **Interpretation:** psychosocial and stress-related variables dominate; CGPA and demographic features add little marginal signal. This aligns with the EDA observations.
- **Limitations:**
  - Dataset is India-centric and self-reported; results may not generalize.
  - Target labels are not clinically validated.
  - No causal claims — feature importance reflects predictive value, not causation.
- **Future work:** SHAP-based explanations for individual predictions, a calibrated probability output for screening use, expanding the dataset across countries.
```

- [ ] **Step 2: Run cells, fill in the actual numbers in Cell 19**

Expected: metrics CSV saved, four PNGs added to `reports/figures/`, conclusions concrete.

- [ ] **Step 3: Commit**

```bash
git add notebooks/03_modeling.ipynb reports/figures/03_*.png reports/figures/03_metrics.csv
git commit -m "feat(modeling): notebook 03 part 3 — test evaluation, feature importance, conclusions"
```

---

## Task 14: Write the final README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the README with the version below**

```markdown
# Student Depression Analysis

End-to-end data science project on a Kaggle-style dataset of ~27 900 students,
predicting `Depression` (binary) from demographic, academic, and lifestyle features.
The project covers EDA, preprocessing, modeling, evaluation, and interpretation —
organized as three Jupyter notebooks driven by reusable `src/` modules.

## Results

| Model | ROC-AUC | F1 | Accuracy |
|-------|---------|----|----------|
| _(filled in from `reports/figures/03_metrics.csv` after running)_ | | | |

Best model: **`<MODEL>`** — ROC-AUC ≈ `<XX.XXX>`.

Top predictors of depression (consistent across Random Forest and XGBoost):
1. History of suicidal thoughts
2. Academic pressure
3. Financial stress
4. Work/study hours
5. Sleep duration

## Highlights

![Categorical features vs Depression](reports/figures/01_categorical_vs_target.png)
![ROC curves](reports/figures/03_roc_curves.png)
![Feature importance](reports/figures/03_feature_importance.png)

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook  # then run notebooks in order: 01 → 02 → 03
```

## Project structure

```
data/             raw + processed splits
notebooks/        01_eda → 02_preprocessing → 03_modeling
src/              data_loader, preprocessing, visualization, models, evaluation
tests/            pytest smoke tests for preprocessing
reports/figures/  PNG exports used in this README
docs/superpowers/ design spec and implementation plan
```

## Limitations

- Dataset is India-centric and self-reported.
- Labels are not clinically validated — results predict the recorded label, not depression itself.
- Feature importance is associational, not causal.

## License

For portfolio / educational use.
```

- [ ] **Step 2: Fill in the actual metric values** in the table and "Best model" line from the metrics CSV produced in Task 13.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: write README with results, highlight figures, and instructions"
```

---

## Task 15: Final verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full test suite**

```bash
pytest -v
```

Expected: 6 passed, 0 failures.

- [ ] **Step 2: Re-execute notebooks top-to-bottom in a clean kernel**

For each notebook (01, 02, 03):
```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_preprocessing.ipynb --output 02_preprocessing.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_modeling.ipynb --output 03_modeling.ipynb
```

Expected: all three notebooks execute without errors. (Notebook 03 may take several minutes due to CV + tuning.)

- [ ] **Step 3: Verify outputs exist**

```bash
ls reports/figures/
```

Expected: `01_numeric_distributions.png`, `01_categorical_counts.png`, `01_top_cities.png`, `01_numeric_vs_target.png`, `01_categorical_vs_target.png`, `01_correlation_heatmap.png`, `01_suicidal_vs_depression.png`, `03_confusion_matrices.png`, `03_roc_curves.png`, `03_pr_curve_best.png`, `03_feature_importance.png`, `03_logreg_coefficients.png`, `03_metrics.csv`.

- [ ] **Step 4: Final commit of any rerun notebook outputs**

```bash
git add notebooks/ reports/
git commit -m "chore: final clean run of all notebooks" || echo "Nothing to commit"
```

- [ ] **Step 5: View git log**

```bash
git log --oneline
```

Expected: clean linear history of focused commits, one per task.
