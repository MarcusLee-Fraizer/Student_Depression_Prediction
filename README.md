# Student Depression Prediction

A binary classification project that explores which student-related factors are
associated with depression, and evaluates how well those factors can predict it.

> **Note on scope:** This is an educational / exploratory project. It is **not**
> a diagnostic tool and must not be used to make decisions about individuals.
> See [Ethical Considerations](#ethical-considerations) before interpreting any
> results.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [Important Caveats](#important-caveats)
- [Ethical Considerations](#ethical-considerations)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Limitations & Next Steps](#limitations--next-steps)
- [References](#references)

---

## Overview

Depression among students is a significant and growing concern. This project
uses a public survey-style dataset to:

1. Explore correlations between demographic, academic, lifestyle, and
   mental-health-history variables and self-reported depression.
2. Build and compare classification models to predict depression.
3. Identify the most influential predictors via model coefficients,
   odds ratios, and permutation importance.

The analysis is implemented as a scikit-learn pipeline with a dummy baseline,
feature selection, and cross-validation.

---

## Dataset

- **Source:** [Kaggle — Student Depression Dataset](https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset)
- **Target variable:** `Depression` (binary: 0 = No, 1 = Yes)
- **Rows:** [PLACEHOLDER — e.g. ~27,900]
- **Features:** [PLACEHOLDER — e.g. 17 input features]

The raw dataset is downloaded at runtime by the script (see
[Reproducing the Analysis](#reproducing-the-analysis)); it is not committed to
the repository.

### Feature groups

| Group | Features |
|---|---|
| **Numeric** | `Age`, `CGPA`, `Work/Study Hours` |
| **Ordinal** | `Academic Pressure`, `Study Satisfaction`, `Financial Stress` |
| **Binary** | `Gender`, `Have you ever had suicidal thoughts ?`, `Family History of Mental Illness` |
| **Categorical** | `Profession`, `Sleep Duration`, `Dietary Habits`, `Degree` |
| **Dropped** | `id` (identifier), `City` (high cardinality), `Work Pressure` & `Job Satisfaction` (zero variance) |

---

## Project Structure
.
├── README.md
├── model.py # Full analysis script (exported from notebook)
├── model.ipynb # Original notebook
└── figures/ # Generated Plots

---

## Methodology

### 1. Data cleaning
- Drop identifier and zero-variance columns.
- Handle missing values (median for numeric, mode for categorical).
- Remove rows with malformed `Financial Stress` values (`'?'`).

### 2. Exploratory analysis
- Correlation matrix for mental-health-history variables vs. `Depression`.
- Correlation matrix across academic / lifestyle / demographic features.
- Distribution plots of numeric features split by depression status.

<!-- IMAGE: Correlation heatmap — Category 1 (family history, suicidal thoughts, depression) -->
<!-- IMAGE: Distribution plots of numeric features by depression status -->

### 3. Preprocessing
A `ColumnTransformer` applies:
- `StandardScaler` → numeric features
- passthrough → ordinal features
- `OneHotEncoder` → binary and categorical features

<!-- IMAGE: (optional) Diagram of preprocessing pipeline -->

### 4. Modeling
- **Baseline:** `DummyClassifier` (for reference).
- **Model:** `LogisticRegression` with `SelectFromModel(RandomForestClassifier)`
  feature selection, all wrapped in a single `Pipeline`.
- **Validation:** 5-fold `cross_validate` on the training split; final metrics on
  a held-out test set.

### 5. Interpretation
- Coefficients and **odds ratios** for the selected features.
- **Permutation importance** to confirm which features the model relies on.

<!-- IMAGE: Feature importance table (coefficients / odds ratios) -->
<!-- IMAGE: Permutation importance boxplot -->

---

## Results

Metrics on the held-out test set:

| Metric | Dummy Baseline | Logistic Regression |
|---|---|---|
| Accuracy  | 0.5835 | 0.8502 |
| Precision | [PLACEHOLDER] | 0.8638 |
| Recall    | [PLACEHOLDER] | 0.8877 |
| F1-score  | [PLACEHOLDER] | 0.8756 |

<!-- IMAGE: Confusion matrix -->
<!-- IMAGE: ROC curve -->

---

## Key Findings

- **Prior suicidal ideation** (`Have you ever had suicidal thoughts ?`) shows the
  strongest association with depression (correlation ≈ 0.55). **See the caveat
  below — this feature is likely a symptom of the target, not a precursor.**
- **Financial stress** and **academic pressure** are consistently among the most
  influential predictors.
- **Family history of mental illness** is positively but weakly correlated with
  depression (≈ 0.05), and its relative importance is lower than financial and
  academic stressors in the full model.
- Lifestyle factors — particularly **sleep duration** and **dietary habits** —
  contribute meaningful signal.

---

## Important Caveats

### ⚠️ Target leakage: "suicidal thoughts"
`Have you ever had suicidal thoughts ?` is a **core symptom** of depression
(it is one of the DSM-5 diagnostic criteria), not a precursor to it. Including it
as a predictor is **target leakage**: it makes the model look far more accurate
than it would be in a genuine pre-assessment / screening setting.

The reported metrics are therefore **optimistic**. To obtain a model that reflects
realistic predictive performance, this feature should be excluded. A
leakage-free variant is [included / planned — update as applicable] and reports
**[PLACEHOLDER metrics]**.

This feature's inclusion is defensible only if the task is explicitly reframed as
*"predict depression given that the person has already reported ideation"* —
which is not the screening use case this project aims at.

### Other caveats
- The dataset is **survey-based and self-reported**, so labels may be noisy.
- The data is [PLACEHOLDER — describe geographic/demographic scope, e.g. "from
  Indian university students"], so results may not generalize.
- Correlation ≠ causation. These are associations, not causal claims.
- `Financial Stress` is treated as ordinal but stored as a string with a `'?'`
  sentinel; cleaning is necessary for correct encoding.

---

## Ethical Considerations

Predicting mental-health outcomes carries real risks:

- **This is not a diagnostic tool.** It must never be used to label, screen, or
  make decisions about real people without clinical oversight.
- **False negatives are dangerous.** A model that misses at-risk individuals can
  cause harm; metrics should be evaluated with that asymmetry in mind (e.g.,
  prioritize recall).
- **Privacy.** Even "anonymized" mental-health data can be sensitive; avoid
  re-identification and store data responsibly.
- **Bias.** Survey data may under-represent certain groups; a model trained on it
  may perform unevenly across populations.

If you or someone you know is struggling, please contact a local mental-health
service or crisis line.

---

## Reproducing the Analysis

### Requirements
python >= 3.9
numpy
pandas
matplotlib
seaborn
scipy
scikit-learn

Install with:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

Run
```bash
python model.py
```

The script downloads the dataset from Kaggle automatically. If the download
fails (e.g., rate limiting), download it manually from the
dataset page
and place student_depression_dataset.csv in the expected path.

The !wget / !unzip lines are Colab shell magics. When running as a plain
.py script, replace them with subprocess calls or run the download
manually. See Limitations.

## Limitations & Next Steps
- Remove the leakage feature (suicidal thoughts) or report both
leakage and non-leakage variants.
- Consolidate the two coexisting pipelines into one clean pipeline; remove
dead/commented code.
- Nest SelectFromModel properly inside cross-validation to avoid optimistic
CV scores.
- Replace the !wget/!unzip magics with portable Python.
- Add a proper model comparison (Logistic Regression vs. Random Forest vs.
gradient boosting) with consistent evaluation.
- Report calibration and a confusion matrix / ROC curve.
- Handle the Sleep Duration quote artifacts and Financial Stress '?'
sentinel in the test split as well as train.
- Drop or justify the odds-ratio "confidence interval" (currently computed on
the mean of odds ratios, which is not a meaningful interval).

## References
Dataset: Student Depression Dataset — Kaggle

scikit-learn documentation: https://scikit-learn.org

DSM-5 diagnostic criteria for major depressive disorder (for context on the
symptom-leakage caveat).
