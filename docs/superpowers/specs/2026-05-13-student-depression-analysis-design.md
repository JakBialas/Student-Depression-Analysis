# Student Depression Analysis — Design Document

**Date:** 2026-05-13
**Type:** Portfolio data science project
**Dataset:** `student_depression_dataset.csv` (~27 900 rows, 18 columns)

## 1. Goal

Build a complete, presentable data science project on the Student Depression dataset, intended for a public portfolio (GitHub). The project should demonstrate:

- A clean, professional code organization (notebooks orchestrating reusable `src/` modules)
- Solid exploratory data analysis with clear visual storytelling
- A reproducible preprocessing pipeline (no data leakage)
- A comparison of classical and gradient-boosted ML models
- Interpretable conclusions about predictors of depression

The project is **portfolio-driven**, not research-driven — readability, structure and presentation matter as much as model accuracy.

## 2. Dataset

Columns: `id`, `Gender`, `Age`, `City`, `Profession`, `Academic Pressure`, `Work Pressure`, `CGPA`, `Study Satisfaction`, `Job Satisfaction`, `Sleep Duration`, `Dietary Habits`, `Degree`, `Have you ever had suicidal thoughts ?`, `Work/Study Hours`, `Financial Stress`, `Family History of Mental Illness`, `Depression`.

**Target:** `Depression` (binary 0/1).

## 3. Repository Structure

```
Student-Depression-Analysis/
├── data/
│   ├── raw/student_depression_dataset.csv     # original dataset
│   └── processed/                              # cleaned + encoded data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── models.py
│   └── evaluation.py
├── reports/figures/                            # saved PNGs for README
├── requirements.txt
└── README.md
```

Notebooks orchestrate and present; reusable code lives in `src/`. Three notebooks (EDA / preprocessing / modeling+evaluation) — modeling and evaluation are kept together so the full "data → metrics" flow can be read in one place.

## 4. Notebook 01 — EDA

### Sanity checks
- Shape, dtypes, missing values, duplicates
- `describe()` summary
- Target balance (`Depression` 0 vs 1)
- Inspect `Profession` column (dataset is "students" but column exists — verify there is no filtering needed)

### Univariate plots
- Histograms for numeric features: `Age`, `CGPA`, `Academic Pressure`, `Work/Study Hours`, `Financial Stress`, `Study Satisfaction`
- Countplots for categoricals: `Gender`, `Sleep Duration`, `Dietary Habits`, `Degree`, `Family History of Mental Illness`, `Have you ever had suicidal thoughts ?`
- Top-N `City` countplot

### Bivariate plots (vs `Depression`)
- Box / violin plots: academic pressure, CGPA, work/study hours, financial stress split by depression
- Stacked / grouped barplots for categorical features vs target (% depressed per group)
- Correlation heatmap of numeric features
- Key plot: suicidal thoughts vs depression (expected to be a strong predictor)

### Conclusions section
A markdown cell at the end summarizing 5–7 observations from EDA that will drive preprocessing and modeling decisions (class balance, important features, features needing engineering).

### Code organization
Plotting functions go into `src/visualization.py` (e.g. `plot_categorical_vs_target`, `plot_numeric_distribution`) so the notebook stays readable.

## 5. Notebook 02 — Preprocessing

### Cleaning
- Drop `id` column
- Handle missing values: median (numeric) / mode (categorical) if any are found
- Inspect outliers (`Age`, `CGPA`) and decide whether to keep them

### Feature transformations
- `Sleep Duration` (`'<5 hours'`, `'5-6 hours'`, `'7-8 hours'`, `'>8 hours'`) → ordinal encoding using bucket midpoints (preserves order, single column)
- `Have you ever had suicidal thoughts ?` → binary 0/1
- `Family History of Mental Illness` → binary 0/1
- `Gender` → binary
- `Dietary Habits` (Healthy / Moderate / Unhealthy) → ordinal encoding
- `City`, `Degree`, `Profession` → one-hot encoding (`drop_first=True`); rare cities grouped into `Other` to control dimensionality
- Scaling (`StandardScaler`) applied via pipeline — needed for LogReg/KNN, harmless for trees; kept uniform for simplicity

### Train/test split
- `train_test_split` 80/20 with `stratify=y`
- `random_state=42` everywhere for reproducibility
- Save processed splits to `data/processed/` so `03_modeling.ipynb` has a clean entry point

### Architecture
`src/preprocessing.py` exposes `build_preprocessor()` returning a sklearn `ColumnTransformer`. This allows wrapping the whole flow in `Pipeline(preprocessor, model)`, which prevents data leakage during cross-validation and tuning — an important "professional" touch for the portfolio.

## 6. Notebook 03 — Modeling and Evaluation

### Models compared (each wrapped in `Pipeline(preprocessor, model)`)
1. Logistic Regression — baseline, interpretable coefficients
2. K-Nearest Neighbors — simple, illustrates the value of scaling
3. Decision Tree — interpretable, overfitting-prone
4. Random Forest — robust, gives feature importance
5. XGBoost — typically best on tabular data

### Procedure
- Train each model on the train split with sensible defaults
- 5-fold Stratified Cross-Validation on train: ROC-AUC and F1 (mean ± std)
- Hyperparameter tuning (`GridSearchCV` or `RandomizedSearchCV`) only for the top 2 models — keeps the project focused while demonstrating the skill
- Final evaluation on the held-out test set only for the tuned models

### Evaluation metrics and plots (functions in `src/evaluation.py`)
- Comparison table: Accuracy, Precision, Recall, F1, ROC-AUC (printed and saved as CSV)
- Confusion matrix heatmaps (subplot per model)
- ROC curves for all models on a single plot
- Precision-Recall curve for the best model
- Feature importance: top 15 features for Random Forest and XGBoost (side-by-side barplots)
- Logistic Regression coefficients for comparison

### Conclusions (markdown section)
- Which model wins and why
- Top 5 predictors of depression with interpretation — the core insight of the portfolio piece
- Limitations: India-centric dataset, self-reported, no clinical validation
- Future work: SHAP analysis, deep learning, more data

## 7. Visualization Stack

Static plots only: `matplotlib` + `seaborn`. Reasons:
- Standard for DS portfolios
- Renders perfectly in GitHub-rendered notebooks
- Easy to save as PNG for the README

## 8. README.md

A short README at the project root:
- One-paragraph project description
- Headline results (best model + ROC-AUC)
- 2–3 most interesting figures embedded from `reports/figures/`
- How to run (install requirements, notebook order)
- Limitations and credits

## 9. Reproducibility

- Single Python environment described in `requirements.txt`
- `random_state=42` fixed wherever randomness occurs
- Notebooks runnable top-to-bottom in order: 01 → 02 → 03
- Processed data persisted between notebooks 02 and 03

## 10. Out of Scope

The following are intentionally excluded to keep the project tight:
- Interactive dashboards (Streamlit, Plotly Dash)
- Interactive plots (Plotly)
- SVM, Naive Bayes, neural networks
- SHAP / advanced explainability (mentioned only as future work)
- Deployment / API / model serving
- Geographic visualizations (map of India)
- Dimensionality-reduction visualizations (PCA, t-SNE)

These can be follow-ups if the project is later extended.
