# US News 8-Model Causal Ranking Analysis — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This plan supersedes any earlier `[NEW]us_news_model_script_v2.ipynb`-derived plan.

**Goal:** Recover and compare the implicit feature weights US News uses for its business-school ranking, by fitting eight linear models over the 2025/2026 dataset (every combination of `ProfessionSalaryRank` in/out × `GMAT Final 1` / `GMAT Final 2` × 2026-only / both years), then producing a structured causal-interpretation report that lets the user pick the model whose feature-driver story is most credible.

**Architecture:**
- The target `OverallScore` is computed by US News from a (publicly undisclosed but approximately linear-on-transformed-inputs) weighted formula. So the modelling problem is *interpretation*, not *prediction*: we want unbiased coefficients with stable signs, not the lowest possible RMSE.
- **Primary estimator: bootstrapped OLS** on standardised features after `log1p` / `logit` / inverse-normal transforms. OLS is unbiased — unlike Ridge/Lasso/ElasticNet, which shrink coefficients toward zero and therefore distort percentage-contribution decompositions.
- **Secondary estimator: bootstrapped ElasticNetCV** is reported in parallel as a stability/robustness check; sign disagreement between OLS and ElasticNet is a multicollinearity warning.
- **Three independent importance decompositions** (`|β_std|` share, drop-one ΔR², Shapley/dominance share of R²) are reported side-by-side so the user can see when they agree (robust) vs disagree (multicollinearity).
- **Multicollinearity is the central enemy of causal interpretation**, so VIF, condition number, and pairwise correlations are first-class outputs, not afterthoughts.
- **Cross-model robustness panel** answers the user's headline question — "which features drive ranking, and how confident am I?" — by comparing coefficient signs / magnitudes / significance *across* the 8 configs. Features whose direction and importance don't move when modelling choices change are the real drivers.
- **Sensitivity matrices** (the user's explicit ±1 %, ±5 %, ±10 % ask) are computed on raw values with safe-domain clipping; AcceptanceRate and ProfessionSalaryRank are flagged as reverse-direction features and a derived "improvement-direction" Δ rank view is reported alongside the raw view.
- **Final deliverable** is `outputs/summary_comparison.xlsx` (machine-readable cross-model comparison) plus `outputs/final_report.md` (my opinion on which of M1–M8 is the best causal model, with the reasoning grounded in the actual numbers).

**Tech Stack:** Python 3.x, pandas, numpy, scikit-learn (`LinearRegression`, `ElasticNetCV`, `KNNImputer`, `StandardScaler`, `KFold`, `GroupKFold`), statsmodels (`OLS`, `variance_inflation_factor`), scipy (`norm`, `spearmanr`, `rankdata`), matplotlib + seaborn, joblib (`Parallel`), openpyxl (Excel writer), `nbformat` (notebook scaffolding). No new dependencies relative to the reference notebook except `statsmodels` (likely already present in the user's Anaconda).

---

## Methodology — Why each design choice

This section is the rationale the user asked for ("present your opinions ... with sufficient details"). The implementation tasks below reference these choices by number.

### M1. OLS, not ElasticNet, for the canonical coefficients
ElasticNet shrinks coefficients toward zero. That's good for prediction but actively bad for "% contribution" — it overstates the dominance of strong features and understates moderate ones, especially under multicollinearity. With **N ≈ 122 and K ≤ 9**, plain OLS is well-conditioned and unbiased: the variance penalty isn't needed. ElasticNet is still reported, but as a **robustness check** — agreement between OLS and ElasticNet signs is evidence the result is real; disagreement is a multicollinearity flag.

### M2. Bootstrap for confidence intervals, not for point estimates
The reference notebook used the bootstrap **mean** of ElasticNetCV coefficients as its point estimate. That bakes in two layers of bias (sample resampling × Lasso shrinkage). We instead use **OLS on the full data for the point estimate**, and the bootstrap distribution only for CIs and stability metrics. This is the standard approach in inferential statistics.

### M3. Multiple importance measures, not just `|β_std|`
A standardised-coefficient share answers "if I changed each input by one standard deviation, which would move the prediction most?" But under collinearity it can wildly over- or under-credit a feature. We compute three measures and report them side-by-side:
- **`|β_std|` share** — the user's original request; what was in the reference notebook.
- **Drop-one ΔR²** — refit without each feature; how much explanatory power is lost?
- **Shapley share of R²** — average R² gain from adding the feature across all subset orderings; the only measure that is collinearity-invariant. Computationally feasible (2⁹ = 512 fits per config).

When all three agree, the contribution number is trustworthy. When they disagree, the disagreement is itself the finding.

### M4. Multicollinearity diagnostics as a first-class output
For each config we report VIF per feature, the design matrix's condition number, and the pairwise Spearman-correlation heatmap. Schools with low acceptance rates also tend to have high peer scores, high GMAT, etc. — these features carry overlapping information. The user *needs* to see how much.

### M5. Year stability test for the both-years configs
For M2/M4/M6/M8 we additionally fit the same model on **2025 alone** and **2026 alone** and report the coefficient delta. If a feature's effect is fundamentally different between years, pooling them masks an important story.

### M6. Sign-coherence audit
For each feature, we record the **expected sign** from domain knowledge (positive for "more is better" features, negative for AcceptanceRate / ProfessionSalaryRank) and flag any model in which the **observed sign on the transformed feature** disagrees. Sign flips under collinearity are a classic interpretation pitfall and need to be visible.

### M7. Sensitivity perturbation = `(1 + δ) × raw_value` with safe-domain clipping
The user said "10 %, 5 %, 1 % in both directions". Multiplicative perturbation on the raw value is the natural reading. We clip to the legal domain per feature (`[1e-3, 1−1e-3]` for rates, `[0, 4]` for MedianGPA, `[200, 800]` for GMAT, `≥ 1` for ProfessionSalaryRank). We **also** report each cell in two views:
- **Δ score** (signed; positive = predicted-score rose) — the literal change.
- **Δ rank in improvement direction** (signed so that *positive always means the school moved up the table*, regardless of which feature; for reverse-direction features the sign is flipped). This is the form the user actually wants to read.

### M8. CV is reported but not used for model selection
The user explicitly said "I don't care much about the model performance as much as I care about the model causality." So predictive metrics (`R²`, `Spearman`, `MAE`, `RMSE`) are reported for completeness but **don't drive the recommendation**. The recommendation is driven by the causal-quality scorecard (M9).

For the CV metrics we use **5-fold KFold** on single-year configs and **GroupKFold by school** on both-years configs (so the same school can't appear in both train and test in a given fold — that would be leakage). Imputation is done **once on the full frame** because for this analysis we are not making out-of-sample predictions; the imputed values are part of the training reality. We note this caveat in the report.

### M9. The recommendation scorecard
For each of the 8 models we compute a composite **Causal-Quality Score** with five components, each rescaled to `[0, 1]` then averaged with equal weight:

| Component | What it measures | Lower or higher better |
|-----------|------------------|------------------------|
| `coef_stability` | Mean of `1 − CV(β)` across features in the bootstrap (CV = std/|mean|) | Higher |
| `sign_coherence` | Fraction of features whose observed sign matches the expected sign | Higher |
| `vif_clean` | `1 − min(1, max(VIF) / 10)`; saturates at VIF ≥ 10 | Higher |
| `importance_agreement` | Mean rank-correlation across the three importance measures (`|β|` share, drop-1 ΔR², Shapley share) | Higher |
| `bootstrap_sign_stability` | Mean fraction of bootstrap iterations in which each coefficient kept the sign of its mean | Higher |

Predictive metrics (`R²`, `Spearman`, CV-R²) are reported but **not** in the score. The user can override the scorecard if a single dimension matters more to them.

### M10. The final report
`outputs/final_report.md` is a written narrative with: per-feature driver call-outs, cross-model robustness summary, multicollinearity findings, my opinion on the best model with the specific evidence, and any caveats the user should know about. This is what the user reads first.

---

## Background and Source Material

**Reference notebook:** `D:\work\US news\notebooks\[NEW]us_news_model_script_v2.ipynb` — used **only** for the data-cleaning recipe (column renames, the `"GMAT_New" / "GMAT_Old"` semantics, MedianGPA min-fill) and the transform groups (`log1p` on salary & GMAT, `logit` on rates, inverse-normal on `ProfessionSalaryRank`). The modelling approach is rebuilt from scratch per the requirements above.

**New dataset:** `D:\work\US news\notebooks\new_models\US News Data 2025 - 2026.xlsx`. Two sheets: `2026` (122 rows × 15 cols), `2025` (121 × 15). Column names match across sheets; `GMAT_Old` / `GMAT_New` are in different physical orders — we read by name only.

**Column rename map** (raw → short):

| Raw | Short |
|-----|-------|
| `school_info.school_name` | `School` |
| `school_info.us_news_rank` | `Rank` |
| `school_info.us_news_overall_score` | `OverallScore` |
| `admissions_and_enrollment.acceptance_rate` | `AcceptanceRate` |
| `ranking_scores_two_year_averages.avg_starting_salary_and_bonus_two_yr_avg` | `AvgSalaryBonus` |
| `ranking_scores_two_year_averages.salaries_by_profession_indicator_rank` | `ProfessionSalaryRank` |
| `ranking_scores_two_year_averages.fulltime_employed_3_months_after_two_yr_avg` | `Employed3Mo` |
| `ranking_scores_two_year_averages.fulltime_employed_at_graduation_two_yr_avg` | `EmployedAtGrad` |
| `ranking_scores_two_year_averages.peer_assessment_score_out_of_5` | `PeerScore` |
| `ranking_scores_two_year_averages.recruiter_assessment_score_out_of_5` | `RecruiterScore` |
| `ranking_scores_two_year_averages.median_undergraduate_gpa` | `MedianGPA` |
| `ranking_scores_two_year_averages.median_gmat_score_fulltime_new` | `GMAT_New` |
| `ranking_scores_two_year_averages.median_gmat_score_fulltime_old` | `GMAT_Old` |
| `GMAT Final 1` | `GMAT_Final_1` |
| `GMAT Final 2` | `GMAT_Final_2` |

**GMAT feature interpretation:**
- `GMAT_Final_1` is fully numeric, no missing — but 66/122 (2026) and 62/121 (2025) rows sit at the dataset minimum (465 / 475), a placeholder for unreported schools. Used as-is.
- `GMAT_Final_2` contains the literal string `"KNN Imputed Value"` in 65 (2026) / 61 (2025) rows. We coerce → NaN, then KNN-impute.

**Missing-value treatment** (matches reference notebook intentions):
- `MedianGPA` → fill with `min(MedianGPA)` (floor; reference cell 24).
- Everything else with NaN (`AvgSalaryBonus`, `ProfessionSalaryRank`, `Employed3Mo`, `EmployedAtGrad`, `PeerScore`, `RecruiterScore`, and `GMAT_Final_2`) → KNN-impute using the other numeric features as the distance basis.

**The 8 model configurations:**

| ID | ProfessionSalaryRank | GMAT feature | Year(s) | N |
|----|---------------------|--------------|---------|---|
| M1 | Included | `GMAT_Final_1` | 2026 only | 122 |
| M2 | Included | `GMAT_Final_1` | 2025 + 2026 | 243 |
| M3 | Included | `GMAT_Final_2` (KNN) | 2026 only | 122 |
| M4 | Included | `GMAT_Final_2` (KNN) | 2025 + 2026 | 243 |
| M5 | Excluded | `GMAT_Final_1` | 2026 only | 122 |
| M6 | Excluded | `GMAT_Final_1` | 2025 + 2026 | 243 |
| M7 | Excluded | `GMAT_Final_2` (KNN) | 2026 only | 122 |
| M8 | Excluded | `GMAT_Final_2` (KNN) | 2025 + 2026 | 243 |

**Feature transformation map:**

| Feature | Transformation | Expected raw-direction sign | Expected sign on transformed feature |
|---------|---------------|----------------------------|--------------------------------------|
| `PeerScore` | none | + | + |
| `RecruiterScore` | none | + | + |
| `MedianGPA` | none | + | + |
| `AvgSalaryBonus` | `log1p` | + | + |
| `GMAT_Combined` | `log1p` | + | + |
| `EmployedAtGrad` | `logit` | + | + |
| `Employed3Mo` | `logit` | + | + |
| `AcceptanceRate` | `logit` | − (lower is better) | − |
| `ProfessionSalaryRank` | inverse-normal with sign flip (rank 1 → high positive z) | − (lower rank# is better) | **+** (because the transform already flipped the meaning) |

The expected-sign column is what the sign-coherence audit (M6) checks against.

---

## File Structure

```
D:\work\US news\notebooks\new_models\
├── US News Data 2025 - 2026.xlsx                              # Input (exists)
├── us_news_8_models.ipynb                                     # NEW — main notebook
├── plans\
│   └── 2026-05-16-eight-model-ranking-build.md                # This plan
└── outputs\                                                   # Created at runtime
    ├── per_model\
    │   ├── M1\
    │   │   ├── coefficients_ols.csv          # OLS bootstrap point estimates + CIs + signs
    │   │   ├── coefficients_enet.csv         # ElasticNet bootstrap (robustness)
    │   │   ├── importance.csv                # |β|-share / drop-1 ΔR² / Shapley share
    │   │   ├── vif.csv                       # VIF per feature
    │   │   ├── sensitivity_score.csv         # Δ score table (signed)
    │   │   ├── sensitivity_rank.csv          # Δ rank table (signed)
    │   │   ├── sensitivity_rank_improve.csv  # Δ rank flipped so + is always better
    │   │   ├── diagnostics.json              # R², adj-R², Spearman, CV-R², BIC, condition #
    │   │   ├── model_ols.pkl
    │   │   └── training_frame.parquet
    │   └── M2 ... M8\                        # same shape
    ├── summary_comparison.xlsx               # Cross-model master workbook
    └── final_report.md                       # My opinion on the best model + reasoning
```

Single notebook because the work is iterative analysis, not a service. The notebook drives a runner over all 8 configs and writes the per-model artifacts; the workbook + the markdown report are the cross-model deliverables.

---

## Output Specifications

### Per-model files (each in `outputs/per_model/<id>/`)

**`coefficients_ols.csv`** — one row per feature plus an `Intercept` row:

| Feature | Beta_OLS | Std_Bootstrap | Lower_95_CI | Upper_95_CI | Sign_Stability_Pct | Is_Significant | Expected_Sign | Observed_Sign | Sign_Matches |
|---------|---------:|--------------:|------------:|------------:|-------------------:|---------------:|:--------------|:--------------|:------------:|

- `Beta_OLS` — OLS point estimate on the full training frame, on **standardised transformed features** so magnitudes are directly comparable.
- `Std_Bootstrap` / CIs — from 2000 bootstrap OLS refits.
- `Sign_Stability_Pct` — % of bootstrap iterations in which `sign(β_iter) == sign(β_mean)`. A coefficient with 99 % stability is rock-solid; 60 % is unreliable.
- `Is_Significant` — CI doesn't cross zero.
- `Sign_Matches` — boolean cross-check vs the M-6 expected-sign table.

**`coefficients_enet.csv`** — same shape but estimator is ElasticNetCV bootstrap (matches reference notebook). Only used to spot OLS vs ElasticNet sign disagreements.

**`importance.csv`** — one row per feature:

| Feature | Beta_Share_Pct | DropOne_DeltaR2 | Shapley_R2_Share_Pct | Rank_By_Beta | Rank_By_DropOne | Rank_By_Shapley |
|---------|---------------:|----------------:|---------------------:|-------------:|----------------:|----------------:|

- `Beta_Share_Pct` — `100 × |Beta_OLS| / sum(|Beta_OLS|)`; the user's original "percentage contribution" request.
- `DropOne_DeltaR2` — `R²(full) − R²(without this feature)`; absolute units, not percent.
- `Shapley_R2_Share_Pct` — average marginal R² gain across all 2⁹ subset orderings, expressed as % of total R². The gold-standard collinearity-invariant measure.
- Rank columns let the user spot disagreement (e.g., `Rank_By_Beta=1` but `Rank_By_Shapley=4` is a multicollinearity warning).

**`vif.csv`** — `Feature`, `VIF`. Computed on the standardised transformed feature matrix.

**`sensitivity_score.csv`** — features × `[-10%, -5%, -1%, +1%, +5%, +10%]`; cells are median Δ score across all schools after multiplicative perturbation of that one feature.

**`sensitivity_rank.csv`** — same layout, cells are median Δ rank position (negative = moved up = better).

**`sensitivity_rank_improve.csv`** — same layout, but signs for `AcceptanceRate` and `ProfessionSalaryRank` rows are pre-flipped so that the column header `+10%` always means "this feature changed in the direction US News considers favourable, by 10 %". Negative cells = ranking improved.

**`diagnostics.json`** —

```json
{
  "n_schools": 122,
  "r2_train": 0.987,
  "adj_r2_train": 0.985,
  "spearman_train": 0.991,
  "mae_train": 1.23,
  "rmse_train": 1.78,
  "cv_r2_mean": 0.972, "cv_r2_std": 0.018,
  "cv_spearman_mean": 0.981, "cv_spearman_std": 0.012,
  "bic": 412.5,
  "condition_number": 14.7,
  "max_vif": 3.2,
  "bootstrap_iterations": 2000,
  "estimator": "OLS",
  "year_stability": {                             # only populated for both-years configs
    "delta_beta_l2": 0.45,
    "max_abs_delta": 0.18
  }
}
```

### Cross-model workbook (`outputs/summary_comparison.xlsx`)

| Sheet | Contents |
|-------|----------|
| `overview` | 1 row per model × { config, N, R², adj-R², Spearman, CV-R²_mean, CV-Spearman_mean, max VIF, condition #, BIC, causal-quality score } |
| `quality_scorecard` | The five M9 components for each model + final composite |
| `coefficients_wide` | features × models, cells = `Beta_OLS` |
| `coef_stability_wide` | features × models, cells = `Sign_Stability_Pct` |
| `significance_wide` | features × models, cells = `Is_Significant` (boolean) |
| `pct_contribution_wide` | features × models, cells = `Beta_Share_Pct` |
| `shapley_wide` | features × models, cells = `Shapley_R2_Share_Pct` |
| `sign_audit_wide` | features × models, cells = `Sign_Matches` (`✓` / `✗`) |
| `vif_wide` | features × models, cells = `VIF` |
| `enet_sign_check` | features × models, cells = `+` / `−` / `0` from ElasticNet — to spot OLS↔ENet disagreement |
| `sens_score_<p>` | features × models for each `p ∈ {−10, −5, −1, +1, +5, +10}` |
| `sens_rank_<p>` | same layout, rank Δ |
| `sens_rank_improve_<p>` | same layout, improvement-direction Δ |
| `cross_model_drivers` | One row per feature: how many models give it sig-positive / sig-negative / non-sig; min / max / mean coefficient across models; verdict ("robust positive driver" / "robust negative driver" / "unstable" / "weak") |

### Final report (`outputs/final_report.md`)

A markdown narrative with these sections, **filled in by the runner using actual numbers** at execution time:

1. **Executive summary** — one paragraph, names the recommended model and the three features identified as the most robust drivers.
2. **Per-feature driver call-outs** — one bullet per feature: "PeerScore is a robust positive driver in 8/8 models; mean β = +0.42 (sd 0.06); Shapley share 18%."
3. **Cross-model robustness** — features whose sign is stable across all 8 → "core drivers"; features whose sign flips → "ambiguous, likely confounded".
4. **Multicollinearity findings** — which features have VIF > 5 in any model; which OLS↔ENet disagreements happened.
5. **Year stability findings** (for M2/M4/M6/M8) — coefficients that changed materially between 2025 and 2026 fits.
6. **Recommendation** — which of M1–M8 is the best causal model and why, with the M9 scorecard table inline. Includes a "if you care more about X, pick model Y instead" sensitivity statement.
7. **Caveats** — leakage in KNN imputation, school-year independence assumption in stacked configs, ProfessionSalaryRank's overlap with AvgSalaryBonus.

---

## Task list

> Reproducibility: `np.random.seed(42)` at the top of the notebook; every estimator that accepts `random_state` is passed `42`.

### Task 1: Notebook skeleton

**Files:**
- Create: `D:\work\US news\notebooks\new_models\us_news_8_models.ipynb`

- [ ] **Step 1: Create the notebook with a title cell**

Run-once script (don't commit to the repo):

```python
import nbformat as nbf
nb = nbf.v4.new_notebook()
nb['cells'].append(nbf.v4.new_markdown_cell(
    "# US News Rankings — Eight-Model Causal Comparison\n\n"
    "Recovers the implicit feature weights US News uses for its business-school ranking. "
    "Fits eight OLS models (bootstrap CIs + ElasticNet robustness check) across the 2×2×2 grid "
    "of ProfessionSalaryRank in/out × GMAT Final 1 / Final 2 × 2026-only / both years. "
    "Outputs per-model artifacts to `outputs/per_model/`, a master comparison to "
    "`outputs/summary_comparison.xlsx`, and a written recommendation to `outputs/final_report.md`."
))
nbf.write(nb, r'D:\work\US news\notebooks\new_models\us_news_8_models.ipynb')
```

- [ ] **Step 2: Verify the notebook opens cleanly**

Run: `python -c "import nbformat; nbformat.read(r'D:\work\US news\notebooks\new_models\us_news_8_models.ipynb', 4)"`. Expected: no exception.

---

### Task 2: Imports + paths cell

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: Imports cell**

```python
import os, json, pickle
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from scipy.stats import norm, spearmanr, rankdata
from joblib import Parallel, delayed

NEW_MODELS_DIR = Path(r"D:\work\US news\notebooks\new_models")
DATA_FILE = NEW_MODELS_DIR / "US News Data 2025 - 2026.xlsx"
OUTPUT_DIR = NEW_MODELS_DIR / "outputs"
PER_MODEL_DIR = OUTPUT_DIR / "per_model"
PER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
CV_FOLDS = 5

np.random.seed(RANDOM_SEED)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
```

- [ ] **Step 2: Run the cell, confirm `outputs/per_model/` exists**

---

### Task 3: Data loading + standardisation

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: Rename map + loader**

```python
RENAME_MAP = {
    'school_info.school_name': 'School',
    'school_info.us_news_rank': 'Rank',
    'school_info.us_news_overall_score': 'OverallScore',
    'admissions_and_enrollment.acceptance_rate': 'AcceptanceRate',
    'ranking_scores_two_year_averages.avg_starting_salary_and_bonus_two_yr_avg': 'AvgSalaryBonus',
    'ranking_scores_two_year_averages.salaries_by_profession_indicator_rank': 'ProfessionSalaryRank',
    'ranking_scores_two_year_averages.fulltime_employed_3_months_after_two_yr_avg': 'Employed3Mo',
    'ranking_scores_two_year_averages.fulltime_employed_at_graduation_two_yr_avg': 'EmployedAtGrad',
    'ranking_scores_two_year_averages.peer_assessment_score_out_of_5': 'PeerScore',
    'ranking_scores_two_year_averages.recruiter_assessment_score_out_of_5': 'RecruiterScore',
    'ranking_scores_two_year_averages.median_undergraduate_gpa': 'MedianGPA',
    'ranking_scores_two_year_averages.median_gmat_score_fulltime_new': 'GMAT_New',
    'ranking_scores_two_year_averages.median_gmat_score_fulltime_old': 'GMAT_Old',
    'GMAT Final 1': 'GMAT_Final_1',
    'GMAT Final 2': 'GMAT_Final_2',
}

def load_year(year: int) -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE, sheet_name=str(year))
    df = df.rename(columns=RENAME_MAP)
    df['GMAT_Final_2'] = pd.to_numeric(df['GMAT_Final_2'], errors='coerce')   # "KNN Imputed Value" -> NaN
    df['GMAT_Final_1'] = pd.to_numeric(df['GMAT_Final_1'], errors='coerce')
    df['Year'] = year
    return df

df_2026 = load_year(2026)
df_2025 = load_year(2025)
print(f"2026: shape={df_2026.shape}, GMAT_Final_2 NaN={df_2026['GMAT_Final_2'].isna().sum()}")
print(f"2025: shape={df_2025.shape}, GMAT_Final_2 NaN={df_2025['GMAT_Final_2'].isna().sum()}")
```

- [ ] **Step 2: Run; expect `2026: shape=(122, 16)`, `GMAT_Final_2 NaN=65` and `2025: shape=(121, 16)`, `NaN=61`**

---

### Task 4: KNN imputation helper

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: Add `knn_impute_missing`** (same as v1 plan — it's not the bottleneck for causality)

```python
def knn_impute_missing(df: pd.DataFrame, impute_cols: List[str],
                       reference_cols: List[str], n_neighbors: int = 5) -> pd.DataFrame:
    """Standard-scale, KNN-impute, inverse-scale. Only `impute_cols` are written back."""
    work = list(dict.fromkeys(impute_cols + reference_cols))
    sub = df[work].astype(float).copy()
    scaler = StandardScaler()
    sub_scaled = pd.DataFrame(
        scaler.fit_transform(sub.fillna(sub.median(numeric_only=True))),
        columns=work, index=sub.index,
    )
    for c in impute_cols:
        sub_scaled.loc[sub[c].isna(), c] = np.nan
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
    imputed_scaled = pd.DataFrame(imputer.fit_transform(sub_scaled), columns=work, index=sub.index)
    imputed = pd.DataFrame(scaler.inverse_transform(imputed_scaled), columns=work, index=sub.index)
    out = df.copy()
    for c in impute_cols:
        out[c] = imputed[c]
    return out
```

- [ ] **Step 2: Run + scratch-test against `df_2026` to confirm no residual NaN on `GMAT_Final_2`**

```python
_test = knn_impute_missing(df_2026, ['GMAT_Final_2'],
                           ['PeerScore', 'RecruiterScore', 'AvgSalaryBonus',
                            'EmployedAtGrad', 'Employed3Mo', 'MedianGPA', 'AcceptanceRate'])
assert _test['GMAT_Final_2'].isna().sum() == 0
print(_test.loc[df_2026['GMAT_Final_2'].isna(), 'GMAT_Final_2'].describe())
```

Expected: imputed range falls roughly within `[500, 750]`. Delete this scratch cell.

---

### Task 5: Feature transformer (functional, not a class)

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

The reference notebook wrapped transformers in sklearn classes for pipeline compatibility. For our causal analysis we never `pipeline.predict()`; we need direct access to the transformed matrix for VIF, Shapley, etc. So functions are simpler.

- [ ] **Step 1: Add the transform functions**

```python
LOG_VARS = ['AvgSalaryBonus', 'GMAT_Combined']
LOGIT_VARS = ['EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate']
INV_NORM_VARS = ['ProfessionSalaryRank']

# Expected sign in raw-feature direction (does increasing the raw value help the rank?).
EXPECTED_SIGN_RAW = {
    'PeerScore': '+', 'RecruiterScore': '+', 'MedianGPA': '+',
    'AvgSalaryBonus': '+', 'GMAT_Combined': '+',
    'EmployedAtGrad': '+', 'Employed3Mo': '+',
    'AcceptanceRate': '-', 'ProfessionSalaryRank': '-',
}
# Expected sign on the *transformed* feature (this is what the regression coefficient sees).
# log1p preserves sign. logit preserves sign. inverse-normal-with-flip on ProfessionSalaryRank
# *reverses* sign because rank 1 -> high positive z, so the transformed feature is "higher = better".
EXPECTED_SIGN_TRANSFORMED = {
    'PeerScore': '+', 'RecruiterScore': '+', 'MedianGPA': '+',
    'AvgSalaryBonus': '+', 'GMAT_Combined': '+',
    'EmployedAtGrad': '+', 'Employed3Mo': '+',
    'AcceptanceRate': '-',
    'ProfessionSalaryRank': '+',     # NOTE: flipped vs raw
}


def transform_features(df: pd.DataFrame, features: List[str],
                       rank_n: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Apply log1p / logit / inverse-normal transforms to the feature columns.

    Returns (transformed_df, rank_counts). Pass the returned rank_counts back in for
    out-of-sample / fold-internal transforms so the percentile denominator stays consistent.
    """
    out = df[features].copy().astype(float)
    rank_counts = dict(rank_n) if rank_n else {}

    for col in LOG_VARS:
        if col in out.columns:
            out[col] = np.log1p(out[col])

    for col in LOGIT_VARS:
        if col in out.columns:
            p = out[col].clip(1e-3, 1 - 1e-3)
            out[col] = np.log(p / (1 - p))

    for col in INV_NORM_VARS:
        if col in out.columns:
            if col not in rank_counts:
                rank_counts[col] = float(out[col].max())
            N = rank_counts[col]
            pct = ((out[col] - 0.5) / N).clip(1e-3, 1 - 1e-3)
            out[col] = -1.0 * norm.ppf(pct)        # rank 1 -> positive z

    return out, rank_counts


def cap_outliers(df: pd.DataFrame, features: List[str], lo: float = 0.05, hi: float = 0.05) -> pd.DataFrame:
    """5/95 winsorisation, applied to raw values before transformation."""
    out = df.copy()
    for col in features:
        if col in out.columns:
            lo_v, hi_v = out[col].quantile(lo), out[col].quantile(1 - hi)
            out[col] = out[col].clip(lo_v, hi_v)
    return out
```

- [ ] **Step 2: Run and confirm the cell evaluates with no errors**

---

### Task 6: Per-model dataset prep

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: `prepare_model_frame`**

```python
ALWAYS_KEEP = ['PeerScore', 'RecruiterScore', 'MedianGPA',
               'AvgSalaryBonus', 'EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate']

def prepare_model_frame(*, years: List[int], gmat_col: str,
                        include_profession_rank: bool) -> Tuple[pd.DataFrame, List[str]]:
    """Build (clean df, feature list). Returns the *raw* (untransformed, uncapped) features so
    downstream callers can apply transforms / caps inside CV folds or in sensitivity analysis."""
    assert gmat_col in ('GMAT_Final_1', 'GMAT_Final_2')

    frames = [load_year(y) for y in years]
    df = pd.concat(frames, axis=0, ignore_index=True)
    df['MedianGPA'] = df['MedianGPA'].fillna(df['MedianGPA'].min())   # floor-fill

    feats = list(ALWAYS_KEEP)
    if include_profession_rank:
        feats.append('ProfessionSalaryRank')
    feats.append('GMAT_Combined')
    df['GMAT_Combined'] = df[gmat_col]

    df = df[['School', 'Year', 'Rank', 'OverallScore'] + feats].copy()

    impute_targets = [c for c in feats if df[c].isna().any()]
    if impute_targets:
        reference_cols = [c for c in feats if c not in impute_targets] or feats
        df = knn_impute_missing(df, impute_cols=impute_targets,
                                reference_cols=reference_cols, n_neighbors=5)
    if 'ProfessionSalaryRank' in feats:
        df['ProfessionSalaryRank'] = df['ProfessionSalaryRank'].round().clip(lower=1)
    assert df[feats].isna().sum().sum() == 0
    return df, feats
```

- [ ] **Step 2: Smoke-test on M3 config and inspect**

```python
_df, _f = prepare_model_frame(years=[2026], gmat_col='GMAT_Final_2', include_profession_rank=True)
print("Shape:", _df.shape, "Features:", _f)
print(_df.describe().T.round(3))
```

Expected `(122, 13)`. Delete scratch cell.

---

### Task 7: Build standardised design matrix + utilities

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: Add helpers**

```python
def build_design_matrix(df: pd.DataFrame, features: List[str]
                        ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Cap outliers -> transform -> standardise. Returns (X_std, y, fit_params).

    `fit_params` captures everything needed to repeat the same transformation on perturbed /
    out-of-sample rows: rank_counts (for ProfessionSalaryRank inv-normal), per-feature outlier
    caps, the scaler's mean/std.
    """
    df_capped = cap_outliers(df, features)
    df_trans, rank_counts = transform_features(df_capped, features)
    scaler = StandardScaler()
    X_std = pd.DataFrame(
        scaler.fit_transform(df_trans), columns=features, index=df.index,
    )
    fit_params = {
        'rank_counts': rank_counts,
        'caps': {c: (df[c].quantile(0.05), df[c].quantile(0.95)) for c in features},
        'scaler_mean': dict(zip(features, scaler.mean_)),
        'scaler_scale': dict(zip(features, scaler.scale_)),
    }
    return X_std, df['OverallScore'].astype(float), fit_params


def transform_like(df_raw: pd.DataFrame, features: List[str],
                   fit_params: Dict[str, Any]) -> pd.DataFrame:
    """Apply the same cap/transform/standardise as the training fit, on new raw rows.
    Used for sensitivity analysis perturbations."""
    out = df_raw[features].copy().astype(float)
    for c, (lo, hi) in fit_params['caps'].items():
        if c in out.columns:
            out[c] = out[c].clip(lo, hi)
    out, _ = transform_features(out, features, rank_n=fit_params['rank_counts'])
    for c in features:
        out[c] = (out[c] - fit_params['scaler_mean'][c]) / fit_params['scaler_scale'][c]
    return out
```

- [ ] **Step 2: Run**

---

### Task 8: OLS fit + bootstrap

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: `fit_ols_bootstrap`**

```python
def _ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (coefs, intercept) from OLS on a possibly-resampled (X, y)."""
    lr = LinearRegression().fit(X, y)
    return lr.coef_, lr.intercept_


def fit_ols_bootstrap(X: pd.DataFrame, y: pd.Series,
                      n_iter: int = N_BOOTSTRAP, n_jobs: int = -1,
                      seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """OLS on the full data for point estimates; bootstrap resample for CIs."""
    point_coef, point_intercept = _ols_fit(X.values, y.values)

    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**31 - 1, size=n_iter)

    def _one(s):
        rs = np.random.RandomState(int(s))
        idx = rs.choice(len(X), size=len(X), replace=True)
        return _ols_fit(X.values[idx], y.values[idx])

    out = Parallel(n_jobs=n_jobs)(delayed(_one)(s) for s in seeds)
    boot_coefs = np.array([o[0] for o in out])
    boot_inter = np.array([o[1] for o in out])

    return {
        'beta': pd.Series(point_coef, index=X.columns),
        'intercept': float(point_intercept),
        'boot_coefs': pd.DataFrame(boot_coefs, columns=X.columns),
        'boot_intercept': boot_inter,
    }
```

- [ ] **Step 2: Add `summarise_ols`**

```python
def summarise_ols(ols_result: Dict[str, Any]) -> pd.DataFrame:
    boot = ols_result['boot_coefs']
    beta = ols_result['beta']
    lo = boot.quantile(0.025)
    hi = boot.quantile(0.975)
    std = boot.std()
    same_sign = boot.apply(lambda col: float((np.sign(col) == np.sign(beta[col.name])).mean()))
    report = pd.DataFrame({
        'Beta_OLS': beta,
        'Std_Bootstrap': std,
        'Lower_95_CI': lo,
        'Upper_95_CI': hi,
        'Sign_Stability_Pct': (same_sign * 100).round(1),
    })
    report['Is_Significant'] = ~((report['Lower_95_CI'] <= 0) & (report['Upper_95_CI'] >= 0))
    report['Observed_Sign'] = np.where(report['Beta_OLS'] >= 0, '+', '-')
    report['Expected_Sign'] = report.index.map(EXPECTED_SIGN_TRANSFORMED.get)
    report['Sign_Matches'] = report['Observed_Sign'] == report['Expected_Sign']
    return report.reset_index().rename(columns={'index': 'Feature'})
```

- [ ] **Step 3: Smoke-test**

```python
_X, _y, _fp = build_design_matrix(_df, _f)
_ols = fit_ols_bootstrap(_X, _y, n_iter=200, n_jobs=-1)
print(summarise_ols(_ols))
```

Expect `PeerScore`, `RecruiterScore` to come back with `Beta_OLS > 0`; `AcceptanceRate` with `Beta_OLS < 0`. If not, abort and audit transforms before continuing. Delete scratch.

---

### Task 9: ElasticNet robustness fit

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: `fit_enet_bootstrap`**

```python
def _enet_fit(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    enet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                        cv=5, max_iter=10000, n_jobs=1, random_state=seed)
    enet.fit(X, y)
    return enet.coef_


def fit_enet_bootstrap(X: pd.DataFrame, y: pd.Series,
                       n_iter: int = 500, n_jobs: int = -1,
                       seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """Lower n_iter than OLS (ENet is ~50x slower per fit)."""
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**31 - 1, size=n_iter)

    def _one(s):
        rs = np.random.RandomState(int(s))
        idx = rs.choice(len(X), size=len(X), replace=True)
        return _enet_fit(X.values[idx], y.values[idx], int(s))

    boot = np.array(Parallel(n_jobs=n_jobs)(delayed(_one)(s) for s in seeds))
    boot_df = pd.DataFrame(boot, columns=X.columns)
    return {
        'beta': boot_df.mean(),                 # bootstrap mean (matches reference convention)
        'boot_coefs': boot_df,
    }


def summarise_enet(enet_result: Dict[str, Any]) -> pd.DataFrame:
    boot = enet_result['boot_coefs']
    beta = enet_result['beta']
    return pd.DataFrame({
        'Feature': beta.index,
        'Beta_ENet': beta.values,
        'Sign': np.where(beta.values >= 0, '+', '-'),
        'Sign_Stability_Pct': (boot.apply(
            lambda c: (np.sign(c) == np.sign(beta[c.name])).mean()) * 100).round(1).values,
    })
```

- [ ] **Step 2: Run**

---

### Task 10: Multicollinearity diagnostics

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: VIF + condition number**

```python
def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """VIF on the design matrix (already standardised, transformed)."""
    Xc = sm.add_constant(X.values)
    vif = []
    for i, name in enumerate(X.columns):
        vif.append({'Feature': name, 'VIF': float(variance_inflation_factor(Xc, i + 1))})
    return pd.DataFrame(vif)


def condition_number(X: pd.DataFrame) -> float:
    s = np.linalg.svd(X.values, compute_uv=False)
    return float(s.max() / s.min())
```

- [ ] **Step 2: Smoke-test**

```python
print(compute_vif(_X))
print("Condition #:", condition_number(_X))
```

Expected: VIF ≤ ~5 for most features; `ProfessionSalaryRank` and `AvgSalaryBonus` plausibly higher (they share variance via the salary signal). Condition # under 30 is comfortable.

---

### Task 11: Importance decompositions

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: `compute_importance`**

```python
def compute_importance(X: pd.DataFrame, y: pd.Series, beta: pd.Series,
                       max_shapley_features: int = 12) -> pd.DataFrame:
    """Three importance measures, plus rank-by-each so disagreement is visible."""
    feats = list(X.columns)
    # 1) |beta| share
    beta_share = beta.abs() / beta.abs().sum() * 100

    # 2) Drop-one delta R^2
    full = LinearRegression().fit(X, y)
    r2_full = r2_score(y, full.predict(X))
    drop_one = {}
    for f in feats:
        Xm = X.drop(columns=[f])
        m = LinearRegression().fit(Xm, y)
        drop_one[f] = r2_full - r2_score(y, m.predict(Xm))

    # 3) Shapley (= dominance analysis): average marginal R^2 over all subset orderings.
    # Only feasible if K is small.
    if len(feats) > max_shapley_features:
        shapley = {f: np.nan for f in feats}
    else:
        # Pre-compute R^2 for every subset
        r2_cache = {(): 0.0}
        for k in range(1, len(feats) + 1):
            for combo in combinations(feats, k):
                Xc = X[list(combo)].values
                r2_cache[tuple(sorted(combo))] = r2_score(y, LinearRegression().fit(Xc, y).predict(Xc))
        # Shapley = mean over permutations of (R^2 with f minus R^2 without f).
        # Equivalent closed form via combinatorial weights:
        from math import comb
        K = len(feats)
        shapley = {}
        for f in feats:
            other = [g for g in feats if g != f]
            total = 0.0
            for k in range(0, K):
                weight = 1.0 / (K * comb(K - 1, k))
                for subset in combinations(other, k):
                    with_f = tuple(sorted(list(subset) + [f]))
                    without_f = tuple(sorted(subset))
                    total += weight * (r2_cache[with_f] - r2_cache[without_f])
            shapley[f] = total
        # Express as % of total R^2.
        total_r2 = sum(shapley.values())
        shapley = {f: (v / total_r2 * 100 if total_r2 > 0 else 0.0) for f, v in shapley.items()}

    df = pd.DataFrame({
        'Feature': feats,
        'Beta_Share_Pct': beta_share.values,
        'DropOne_DeltaR2': [drop_one[f] for f in feats],
        'Shapley_R2_Share_Pct': [shapley[f] for f in feats],
    })
    df['Rank_By_Beta'] = df['Beta_Share_Pct'].rank(ascending=False, method='min').astype(int)
    df['Rank_By_DropOne'] = df['DropOne_DeltaR2'].rank(ascending=False, method='min').astype(int)
    df['Rank_By_Shapley'] = df['Shapley_R2_Share_Pct'].rank(ascending=False, method='min').astype(int)
    return df
```

- [ ] **Step 2: Smoke-test**

```python
print(compute_importance(_X, _y, _ols['beta']))
```

Expect a 9-row table; rank columns should be integers 1–9. Delete scratch.

---

### Task 12: Cross-validated performance metrics

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: `cv_metrics` with year-aware folds**

```python
def cv_metrics(df_raw: pd.DataFrame, features: List[str], n_splits: int = CV_FOLDS,
               seed: int = RANDOM_SEED) -> Dict[str, float]:
    """5-fold CV; if multiple Years present, group by School so the same school can't appear
    in both train and test in a fold."""
    y = df_raw['OverallScore'].values
    if df_raw['Year'].nunique() > 1:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(df_raw, y, groups=df_raw['School']))
    else:
        splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                       .split(df_raw))

    r2_list, sp_list, mae_list, rmse_list = [], [], [], []
    for tr, te in splits:
        df_tr, df_te = df_raw.iloc[tr], df_raw.iloc[te]
        X_tr, y_tr, fp = build_design_matrix(df_tr, features)
        X_te = transform_like(df_te[features], features, fp)
        y_te = df_te['OverallScore'].values
        lr = LinearRegression().fit(X_tr, y_tr)
        p = lr.predict(X_te)
        r2_list.append(r2_score(y_te, p))
        sp_list.append(spearmanr(y_te, p).statistic)
        mae_list.append(mean_absolute_error(y_te, p))
        rmse_list.append(float(np.sqrt(mean_squared_error(y_te, p))))

    return {
        'cv_r2_mean': float(np.mean(r2_list)), 'cv_r2_std': float(np.std(r2_list)),
        'cv_spearman_mean': float(np.mean(sp_list)), 'cv_spearman_std': float(np.std(sp_list)),
        'cv_mae_mean': float(np.mean(mae_list)),
        'cv_rmse_mean': float(np.mean(rmse_list)),
    }
```

- [ ] **Step 2: Smoke-test on M3**

```python
print(cv_metrics(_df, _f))
```

Single-year M3: expect CV R² ≥ 0.9. Delete scratch.

---

### Task 13: Year-stability diagnostic (both-years only)

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: `year_stability`**

```python
def year_stability(df_raw: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """Fit OLS on each year separately, return the standardised-beta delta."""
    if df_raw['Year'].nunique() < 2:
        return {}
    coefs_by_year = {}
    for y_ in sorted(df_raw['Year'].unique()):
        sub = df_raw[df_raw['Year'] == y_]
        X_, y_arr, _ = build_design_matrix(sub, features)
        lr = LinearRegression().fit(X_, y_arr)
        coefs_by_year[int(y_)] = pd.Series(lr.coef_, index=features)
    diffs = (coefs_by_year[max(coefs_by_year)] - coefs_by_year[min(coefs_by_year)]).abs()
    return {
        'per_year_betas': {k: v.to_dict() for k, v in coefs_by_year.items()},
        'max_abs_delta': float(diffs.max()),
        'max_delta_feature': str(diffs.idxmax()),
        'delta_beta_l2': float(np.sqrt((diffs ** 2).sum())),
    }
```

- [ ] **Step 2: Run cell**

---

### Task 14: Sensitivity matrices

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: Add helpers**

```python
PERTURBATIONS = [-0.10, -0.05, -0.01, 0.01, 0.05, 0.10]
REVERSE_DIRECTION_FEATURES = {'AcceptanceRate', 'ProfessionSalaryRank'}


def _safe_perturb(values: np.ndarray, feature: str, delta: float) -> np.ndarray:
    new = values * (1.0 + delta)
    if feature in ('AcceptanceRate', 'EmployedAtGrad', 'Employed3Mo'):
        new = np.clip(new, 1e-3, 1 - 1e-3)
    elif feature == 'MedianGPA':
        new = np.clip(new, 0.0, 4.0)
    elif feature == 'GMAT_Combined':
        new = np.clip(new, 200.0, 800.0)
    elif feature == 'ProfessionSalaryRank':
        new = np.clip(np.round(new), 1, None)
    return new


def predict_from_raw(df_raw: pd.DataFrame, features: List[str],
                     fit_params: Dict[str, Any], beta: pd.Series, intercept: float) -> np.ndarray:
    X = transform_like(df_raw, features, fit_params)
    return X.values @ beta.values + intercept


def compute_sensitivity(df_raw: pd.DataFrame, features: List[str],
                        fit_params: Dict[str, Any], beta: pd.Series, intercept: float
                        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_pred = predict_from_raw(df_raw, features, fit_params, beta, intercept)
    base_rank = pd.Series(rankdata(-base_pred, method='min'), index=df_raw.index)

    score_rows, rank_rows, rank_imp_rows = [], [], []
    cols = [f'{int(d*100):+d}%' for d in PERTURBATIONS]

    for f in features:
        s_row, r_row, ri_row = {'Feature': f}, {'Feature': f}, {'Feature': f}
        for d, col in zip(PERTURBATIONS, cols):
            df_pert = df_raw.copy()
            df_pert[f] = _safe_perturb(df_pert[f].values, f, d)
            new_pred = predict_from_raw(df_pert, features, fit_params, beta, intercept)
            new_rank = pd.Series(rankdata(-new_pred, method='min'), index=df_pert.index)
            ds = float(np.median(new_pred - base_pred))
            dr = float((new_rank - base_rank).median())
            s_row[col] = ds
            r_row[col] = dr
            # improvement-direction Δrank: flip sign for reverse-direction features
            sign_flip = -1.0 if f in REVERSE_DIRECTION_FEATURES else 1.0
            # Δrank where negative = improvement; multiply by -1 so "+ always means better rank"
            # then if the feature is reverse-direction, the perturbation direction is also flipped
            ri_row[col] = -dr * sign_flip
        score_rows.append(s_row); rank_rows.append(r_row); rank_imp_rows.append(ri_row)
    return (pd.DataFrame(score_rows).set_index('Feature'),
            pd.DataFrame(rank_rows).set_index('Feature'),
            pd.DataFrame(rank_imp_rows).set_index('Feature'))
```

- [ ] **Step 2: Smoke-test**

```python
ss, sr, sri = compute_sensitivity(_df, _f, _fp, _ols['beta'], _ols['intercept'])
print("Score sensitivity:\n", ss.round(3))
print("\nRank sensitivity:\n", sr.round(2))
print("\nRank-improvement sensitivity (+ = better):\n", sri.round(2))
```

Sanity check: in `sri`, every cell in the `+10%` column should be ≥ 0 for *every* feature (because "+10% in improvement direction" = the favourable change). If any cell is negative, the model's coefficient sign contradicts domain expectation — flag in the report.

Delete scratch cell.

---

### Task 15: Model configs + runner

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: Add `MODEL_CONFIGS` and `run_one_model`**

```python
MODEL_CONFIGS = [
    {'id': 'M1', 'include_profession_rank': True,  'gmat_col': 'GMAT_Final_1', 'years': [2026]},
    {'id': 'M2', 'include_profession_rank': True,  'gmat_col': 'GMAT_Final_1', 'years': [2025, 2026]},
    {'id': 'M3', 'include_profession_rank': True,  'gmat_col': 'GMAT_Final_2', 'years': [2026]},
    {'id': 'M4', 'include_profession_rank': True,  'gmat_col': 'GMAT_Final_2', 'years': [2025, 2026]},
    {'id': 'M5', 'include_profession_rank': False, 'gmat_col': 'GMAT_Final_1', 'years': [2026]},
    {'id': 'M6', 'include_profession_rank': False, 'gmat_col': 'GMAT_Final_1', 'years': [2025, 2026]},
    {'id': 'M7', 'include_profession_rank': False, 'gmat_col': 'GMAT_Final_2', 'years': [2026]},
    {'id': 'M8', 'include_profession_rank': False, 'gmat_col': 'GMAT_Final_2', 'years': [2025, 2026]},
]


def run_one_model(cfg: Dict[str, Any]) -> Dict[str, Any]:
    df_raw, feats = prepare_model_frame(
        years=cfg['years'], gmat_col=cfg['gmat_col'],
        include_profession_rank=cfg['include_profession_rank'])
    X, y, fp = build_design_matrix(df_raw, feats)

    ols = fit_ols_bootstrap(X, y, n_iter=N_BOOTSTRAP, n_jobs=-1)
    enet = fit_enet_bootstrap(X, y, n_iter=500, n_jobs=-1)
    coef_report = summarise_ols(ols)
    enet_report = summarise_enet(enet)
    importance = compute_importance(X, y, ols['beta'])
    vif = compute_vif(X)

    train_pred = X.values @ ols['beta'].values + ols['intercept']
    diagnostics = {
        'config': cfg,
        'n_schools': int(len(df_raw)),
        'n_features': len(feats),
        'features': feats,
        'r2_train': float(r2_score(y, train_pred)),
        'adj_r2_train': float(1 - (1 - r2_score(y, train_pred))
                              * (len(y) - 1) / (len(y) - len(feats) - 1)),
        'spearman_train': float(spearmanr(y, train_pred).statistic),
        'mae_train': float(mean_absolute_error(y, train_pred)),
        'rmse_train': float(np.sqrt(mean_squared_error(y, train_pred))),
        'intercept': float(ols['intercept']),
        'condition_number': condition_number(X),
        'max_vif': float(vif['VIF'].max()),
        'bootstrap_iterations': N_BOOTSTRAP,
        **cv_metrics(df_raw, feats),
        **({'year_stability': year_stability(df_raw, feats)} if len(cfg['years']) > 1 else {}),
    }

    sens_score, sens_rank, sens_rank_improve = compute_sensitivity(df_raw, feats, fp,
                                                                   ols['beta'], ols['intercept'])

    out_dir = PER_MODEL_DIR / cfg['id']
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_report.to_csv(out_dir / 'coefficients_ols.csv', index=False)
    enet_report.to_csv(out_dir / 'coefficients_enet.csv', index=False)
    importance.to_csv(out_dir / 'importance.csv', index=False)
    vif.to_csv(out_dir / 'vif.csv', index=False)
    sens_score.to_csv(out_dir / 'sensitivity_score.csv')
    sens_rank.to_csv(out_dir / 'sensitivity_rank.csv')
    sens_rank_improve.to_csv(out_dir / 'sensitivity_rank_improve.csv')
    with open(out_dir / 'diagnostics.json', 'w') as fp_:
        json.dump(diagnostics, fp_, indent=2, default=float)
    with open(out_dir / 'model_ols.pkl', 'wb') as fp_:
        pickle.dump({'beta': ols['beta'], 'intercept': ols['intercept'],
                     'fit_params': fp, 'features': feats}, fp_)
    df_raw.to_parquet(out_dir / 'training_frame.parquet')

    return {
        'config': cfg, 'features': feats, 'df_raw': df_raw,
        'coef_report': coef_report, 'enet_report': enet_report,
        'importance': importance, 'vif': vif,
        'sens_score': sens_score, 'sens_rank': sens_rank, 'sens_rank_improve': sens_rank_improve,
        'diagnostics': diagnostics,
        'ols': ols, 'enet': enet,
    }
```

- [ ] **Step 2: Run a single config end-to-end as a smoke test**

```python
m1 = run_one_model(MODEL_CONFIGS[0])
print("Diagnostics:", json.dumps({k: m1['diagnostics'][k] for k in
                                  ['n_schools','r2_train','spearman_train','cv_r2_mean',
                                   'max_vif','condition_number']}, indent=2, default=float))
```

Expected R² ≥ 0.95, CV R² ≥ 0.85, max VIF likely under 10.

---

### Task 16: Causal-quality scorecard

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: `compute_quality_score`**

```python
def compute_quality_score(res: Dict[str, Any]) -> Dict[str, float]:
    coef = res['coef_report'].set_index('Feature')
    # 1) coef_stability: 1 - mean CV of |beta| (clipped to [0, 1]; CV = std / |mean|)
    cv = (coef['Std_Bootstrap'] / coef['Beta_OLS'].abs().replace(0, np.nan)).fillna(2.0)
    coef_stability = float(np.clip(1 - cv.mean(), 0.0, 1.0))
    # 2) sign_coherence: fraction of features whose sign matches expected
    sign_coherence = float(coef['Sign_Matches'].mean())
    # 3) vif_clean: 1 - min(1, maxVIF/10)
    vif_clean = float(1 - min(1.0, res['diagnostics']['max_vif'] / 10.0))
    # 4) importance_agreement: mean Spearman rho across the three rank columns
    imp = res['importance'].set_index('Feature')[
        ['Rank_By_Beta', 'Rank_By_DropOne', 'Rank_By_Shapley']]
    if imp['Rank_By_Shapley'].isna().any():
        imp = imp[['Rank_By_Beta', 'Rank_By_DropOne']]
    rhos = []
    cols = imp.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = spearmanr(imp[cols[i]], imp[cols[j]]).statistic
            rhos.append(0.0 if np.isnan(r) else r)
    importance_agreement = float(np.mean(rhos)) if rhos else 0.0
    # 5) bootstrap_sign_stability: mean of Sign_Stability_Pct / 100
    bootstrap_sign_stability = float(coef['Sign_Stability_Pct'].mean() / 100.0)

    composite = float(np.mean([coef_stability, sign_coherence, vif_clean,
                               importance_agreement, bootstrap_sign_stability]))
    return {
        'coef_stability': coef_stability,
        'sign_coherence': sign_coherence,
        'vif_clean': vif_clean,
        'importance_agreement': importance_agreement,
        'bootstrap_sign_stability': bootstrap_sign_stability,
        'composite_score': composite,
    }
```

- [ ] **Step 2: Run + verify on M1**

```python
print(compute_quality_score(m1))
```

Composite should be in `(0, 1)`. Delete scratch.

---

### Task 17: Run all 8 models

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: Runner loop**

```python
results: Dict[str, Dict[str, Any]] = {}
quality: Dict[str, Dict[str, float]] = {}
for cfg in MODEL_CONFIGS:
    print(f"\n=== Running {cfg['id']}: profession={cfg['include_profession_rank']}, "
          f"gmat={cfg['gmat_col']}, years={cfg['years']} ===")
    res = run_one_model(cfg)
    results[cfg['id']] = res
    quality[cfg['id']] = compute_quality_score(res)
    print(f"  R²={res['diagnostics']['r2_train']:.4f}  "
          f"Spearman={res['diagnostics']['spearman_train']:.4f}  "
          f"CV-R²={res['diagnostics']['cv_r2_mean']:.4f}  "
          f"maxVIF={res['diagnostics']['max_vif']:.2f}  "
          f"composite={quality[cfg['id']]['composite_score']:.3f}")
```

- [ ] **Step 2: Run**

Expected: 8 progress lines, all artifacts written under `outputs/per_model/`.

---

### Task 18: Cross-model summary workbook

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: `build_summary_workbook`**

```python
def build_summary_workbook(results, quality, out_path):
    overview_rows = []
    for mid, r in results.items():
        d = r['diagnostics']
        q = quality[mid]
        overview_rows.append({
            'Model': mid,
            'ProfessionRank': r['config']['include_profession_rank'],
            'GMAT': r['config']['gmat_col'],
            'Years': '+'.join(str(y) for y in r['config']['years']),
            'N': d['n_schools'],
            'R2': round(d['r2_train'], 4),
            'AdjR2': round(d['adj_r2_train'], 4),
            'Spearman': round(d['spearman_train'], 4),
            'CV_R2_mean': round(d['cv_r2_mean'], 4),
            'CV_Spearman_mean': round(d['cv_spearman_mean'], 4),
            'MAE': round(d['mae_train'], 3),
            'RMSE': round(d['rmse_train'], 3),
            'MaxVIF': round(d['max_vif'], 2),
            'CondNum': round(d['condition_number'], 2),
            'Composite_Score': round(q['composite_score'], 4),
        })
    overview = pd.DataFrame(overview_rows).set_index('Model').sort_values(
        'Composite_Score', ascending=False)

    scorecard = pd.DataFrame({mid: q for mid, q in quality.items()}).T.round(4)
    scorecard.index.name = 'Model'

    def _wide(field, src='coef_report', from_index='Feature'):
        pieces = []
        for mid, r in results.items():
            sub = r[src].set_index(from_index)[[field]].rename(columns={field: mid})
            pieces.append(sub)
        return pd.concat(pieces, axis=1)

    coef_wide = _wide('Beta_OLS')
    stab_wide = _wide('Sign_Stability_Pct')
    sig_wide  = _wide('Is_Significant')
    sign_aud  = _wide('Sign_Matches')
    pct_wide  = _wide('Beta_Share_Pct', src='importance')
    shap_wide = _wide('Shapley_R2_Share_Pct', src='importance')
    vif_wide  = _wide('VIF', src='vif')
    enet_wide = _wide('Sign', src='enet_report')

    # Cross-model driver verdict
    drivers = []
    feats_union = sorted({f for r in results.values() for f in r['features']})
    for f in feats_union:
        models_with = [m for m, r in results.items() if f in r['features']]
        betas = [results[m]['coef_report'].set_index('Feature').loc[f, 'Beta_OLS']
                 for m in models_with]
        sigs = [results[m]['coef_report'].set_index('Feature').loc[f, 'Is_Significant']
                for m in models_with]
        n_pos = sum(1 for b, s in zip(betas, sigs) if s and b > 0)
        n_neg = sum(1 for b, s in zip(betas, sigs) if s and b < 0)
        n_ns  = sum(1 for s in sigs if not s)
        verdict = (
            'Robust positive driver' if n_pos == len(models_with) else
            'Robust negative driver' if n_neg == len(models_with) else
            'Unstable / sign-flips' if n_pos > 0 and n_neg > 0 else
            'Weak (often non-significant)'
        )
        drivers.append({
            'Feature': f, 'N_Models': len(models_with),
            'N_SigPositive': n_pos, 'N_SigNegative': n_neg, 'N_NonSig': n_ns,
            'Mean_Beta': float(np.mean(betas)),
            'Min_Beta': float(np.min(betas)),
            'Max_Beta': float(np.max(betas)),
            'Verdict': verdict,
        })
    drivers_df = pd.DataFrame(drivers).set_index('Feature')

    # Sensitivity wide pivots, one per perturbation
    pert_cols = [f'{int(d*100):+d}%' for d in PERTURBATIONS]
    def _sens_wide(key):
        out = {p: [] for p in pert_cols}
        for mid, r in results.items():
            for p in pert_cols:
                col = r[key][[p]].rename(columns={p: mid})
                out[p].append(col)
        return {p: pd.concat(out[p], axis=1) for p in pert_cols}

    score_wide = _sens_wide('sens_score')
    rank_wide = _sens_wide('sens_rank')
    rank_imp_wide = _sens_wide('sens_rank_improve')

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        overview.to_excel(writer, sheet_name='overview')
        scorecard.to_excel(writer, sheet_name='quality_scorecard')
        drivers_df.to_excel(writer, sheet_name='cross_model_drivers')
        coef_wide.to_excel(writer, sheet_name='coefficients_wide')
        stab_wide.to_excel(writer, sheet_name='coef_stability_wide')
        sig_wide.to_excel(writer, sheet_name='significance_wide')
        sign_aud.to_excel(writer, sheet_name='sign_audit_wide')
        pct_wide.to_excel(writer, sheet_name='pct_contribution_wide')
        shap_wide.to_excel(writer, sheet_name='shapley_wide')
        vif_wide.to_excel(writer, sheet_name='vif_wide')
        enet_wide.to_excel(writer, sheet_name='enet_sign_check')
        for p in pert_cols:
            tag = p.replace('+', 'p').replace('-', 'm')
            score_wide[p].to_excel(writer, sheet_name=f'sens_score_{tag}')
            rank_wide[p].to_excel(writer, sheet_name=f'sens_rank_{tag}')
            rank_imp_wide[p].to_excel(writer, sheet_name=f'sens_rank_imp_{tag}')

    print(f"Wrote {out_path}")


build_summary_workbook(results, quality, OUTPUT_DIR / 'summary_comparison.xlsx')
```

- [ ] **Step 2: Open the workbook and spot-check the `overview` and `cross_model_drivers` sheets**

---

### Task 19: Final report writer

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

This generates `outputs/final_report.md`, a written narrative that uses the actual numbers from `results` and `quality`.

- [ ] **Step 1: `write_final_report`**

```python
def write_final_report(results, quality, out_path):
    # Identify the recommended model: highest composite score, with ties broken by
    # (1) lower max VIF, (2) higher CV R^2.
    rank_df = pd.DataFrame({
        m: {
            'composite': quality[m]['composite_score'],
            'maxVIF': results[m]['diagnostics']['max_vif'],
            'cv_r2': results[m]['diagnostics']['cv_r2_mean'],
        } for m in results
    }).T
    rank_df = rank_df.sort_values(['composite', 'maxVIF', 'cv_r2'],
                                  ascending=[False, True, False])
    winner = rank_df.index[0]
    runner_up = rank_df.index[1]
    win_res = results[winner]

    # Per-feature cross-model summary (re-derive from results so the report is self-contained)
    feats_union = sorted({f for r in results.values() for f in r['features']})
    feature_lines = []
    for f in feats_union:
        models_with = [m for m, r in results.items() if f in r['features']]
        coefs = [(m, results[m]['coef_report'].set_index('Feature').loc[f]) for m in models_with]
        betas = np.array([row['Beta_OLS'] for _, row in coefs])
        sigs = np.array([row['Is_Significant'] for _, row in coefs])
        n = len(models_with)
        n_pos = int(((betas > 0) & sigs).sum())
        n_neg = int(((betas < 0) & sigs).sum())
        n_ns  = int((~sigs).sum())
        mean_b = betas.mean()
        sd_b = betas.std()
        feature_lines.append(
            f"- **{f}** — appears in {n}/8 models. "
            f"Significant positive in {n_pos}; significant negative in {n_neg}; "
            f"non-significant in {n_ns}. Mean β = {mean_b:+.3f} (sd {sd_b:.3f})."
        )

    # Multicollinearity findings
    high_vif = []
    for m, r in results.items():
        v = r['vif'].set_index('Feature')['VIF']
        for f, val in v.items():
            if val > 5:
                high_vif.append(f"  - {m} / {f}: VIF = {val:.2f}")

    # OLS ↔ ENet sign disagreements
    disagree = []
    for m, r in results.items():
        ols = r['coef_report'].set_index('Feature')['Beta_OLS']
        enet = r['enet_report'].set_index('Feature')['Beta_ENet']
        for f in ols.index:
            if np.sign(ols[f]) != np.sign(enet[f]):
                disagree.append(f"  - {m} / {f}: OLS β = {ols[f]:+.3f}, ENet β = {enet[f]:+.3f}")

    # Year stability findings (for both-years configs)
    year_stab = []
    for m, r in results.items():
        ys = r['diagnostics'].get('year_stability')
        if ys:
            year_stab.append(
                f"  - {m}: max |Δβ| between years = {ys['max_abs_delta']:.3f} "
                f"on `{ys['max_delta_feature']}`; L2 = {ys['delta_beta_l2']:.3f}"
            )

    lines = []
    lines.append(f"# US News 8-Model Ranking — Causal Comparison Report\n")
    lines.append("## Executive summary\n")
    cfg = win_res['config']
    lines.append(
        f"**Recommended model: `{winner}`** — ProfessionSalaryRank "
        f"{'included' if cfg['include_profession_rank'] else 'excluded'}, "
        f"GMAT feature = `{cfg['gmat_col']}`, years = {'+'.join(str(y) for y in cfg['years'])}.\n\n"
        f"It has the highest composite causal-quality score "
        f"({quality[winner]['composite_score']:.3f} vs runner-up `{runner_up}` at "
        f"{quality[runner_up]['composite_score']:.3f}). Predictive metrics: training R² "
        f"= {win_res['diagnostics']['r2_train']:.3f}, Spearman ρ = "
        f"{win_res['diagnostics']['spearman_train']:.3f}, CV-R² = "
        f"{win_res['diagnostics']['cv_r2_mean']:.3f}; max VIF = "
        f"{win_res['diagnostics']['max_vif']:.2f}.\n\n"
        "Top three drivers in this model (by Shapley R² share):\n"
    )
    top3 = win_res['importance'].sort_values('Shapley_R2_Share_Pct', ascending=False).head(3)
    for _, row in top3.iterrows():
        lines.append(f"- **{row['Feature']}** — Shapley share {row['Shapley_R2_Share_Pct']:.1f}%, "
                     f"|β| share {row['Beta_Share_Pct']:.1f}%.")
    lines.append("\n")

    lines.append("## Causal-quality scorecard\n")
    lines.append(pd.DataFrame({m: q for m, q in quality.items()}).T.round(3).to_markdown())
    lines.append("\n## Predictive metrics\n")
    pred_table = pd.DataFrame({m: {
        'R²': results[m]['diagnostics']['r2_train'],
        'Adj R²': results[m]['diagnostics']['adj_r2_train'],
        'Spearman': results[m]['diagnostics']['spearman_train'],
        'CV-R²': results[m]['diagnostics']['cv_r2_mean'],
        'CV-Spearman': results[m]['diagnostics']['cv_spearman_mean'],
        'MAE': results[m]['diagnostics']['mae_train'],
        'RMSE': results[m]['diagnostics']['rmse_train'],
        'Max VIF': results[m]['diagnostics']['max_vif'],
        'Cond #': results[m]['diagnostics']['condition_number'],
    } for m in results}).T.round(3)
    lines.append(pred_table.to_markdown())

    lines.append("\n## Per-feature driver summary across all 8 models\n")
    lines.extend(feature_lines)

    lines.append("\n## Multicollinearity findings (VIF > 5)\n")
    lines.append("\n".join(high_vif) if high_vif else "_No features had VIF > 5 in any model._")

    lines.append("\n\n## OLS ↔ ElasticNet sign disagreements\n")
    lines.append("\n".join(disagree) if disagree
                 else "_OLS and ElasticNet agreed on coefficient signs in every (model, feature) cell._")

    lines.append("\n\n## Year-stability (both-years configs)\n")
    lines.append("\n".join(year_stab) if year_stab else "_n/a — applies only to M2/M4/M6/M8._")

    lines.append("\n\n## Recommendation rationale\n")
    lines.append(
        f"`{winner}` wins because:\n"
        f"- **Coefficient stability**: bootstrap CV averaged across features = "
        f"{1 - quality[winner]['coef_stability']:.3f} (lower is better).\n"
        f"- **Sign coherence**: {int(quality[winner]['sign_coherence'] * len(win_res['features']))} "
        f"of {len(win_res['features'])} features have the expected sign on the transformed "
        f"feature.\n"
        f"- **Multicollinearity**: max VIF = {win_res['diagnostics']['max_vif']:.2f} "
        f"({'comfortable' if win_res['diagnostics']['max_vif'] <= 10 else 'elevated; interpret with care'}).\n"
        f"- **Importance agreement**: rank correlation across `|β|`-share, drop-one ΔR², and "
        f"Shapley share = {quality[winner]['importance_agreement']:.3f}.\n"
        f"- **Bootstrap sign stability**: average % of bootstrap iterations where each coefficient "
        f"kept its mean sign = {quality[winner]['bootstrap_sign_stability']*100:.1f}%.\n"
    )

    lines.append("\n## When to pick a different model\n")
    lines.append(
        "- If **predictive accuracy** matters more than coefficient stability, prefer the model "
        f"with the highest CV-R² (see the predictive-metrics table).\n"
        "- If **only 2026 reality** matters (e.g., decisions about next year's targets), prefer a "
        "2026-only config; the both-years configs blend two years of data.\n"
        "- If `ProfessionSalaryRank` is operationally hard to influence, prefer one of the "
        "`Excluded` configs (M5–M8) so the recommended sensitivity matrix focuses on features "
        "the user can actually move.\n"
    )

    lines.append("\n## Caveats\n")
    lines.append(
        "- KNN imputation was performed once on the full per-config frame, not inside CV folds. "
        "This is acceptable because we are not making out-of-sample predictions; the imputed "
        "values are part of the *training reality* for each config. For honest predictive "
        "evaluation in deployment, imputation must be re-fitted per fold.\n"
        "- Both-years configs treat each school-year as an independent observation. If you would "
        "prefer schools weighted equally across years (one row per school, year-averaged), that "
        "is a different model spec and would shift coefficients.\n"
        "- `GMAT_Final_1` has many rows at the placeholder floor value, which artificially "
        "compresses the GMAT distribution and may understate GMAT's true contribution in M1/M2/M5/M6 "
        "relative to the KNN-imputed `GMAT_Final_2` configs.\n"
    )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Wrote {out_path}")


write_final_report(results, quality, OUTPUT_DIR / 'final_report.md')
```

- [ ] **Step 2: Run + open `outputs/final_report.md`**

Verify it reads cleanly: executive summary names a specific model, the scorecard table renders, the per-feature lines all have numbers, and the recommendation rationale is coherent.

---

### Task 20: Per-model on-screen summary (optional but useful)

**Files:**
- Modify: `us_news_8_models.ipynb` — add code cell.

- [ ] **Step 1: Inline print helper**

```python
def display_model_summary(mid: str, r: Dict[str, Any], q: Dict[str, float]) -> None:
    print("=" * 80)
    print(f"  {mid}: profession={r['config']['include_profession_rank']}, "
          f"gmat={r['config']['gmat_col']}, years={'+'.join(str(y) for y in r['config']['years'])}")
    print(f"  Composite causal-quality score: {q['composite_score']:.3f}")
    print(f"  R²={r['diagnostics']['r2_train']:.4f}  Spearman={r['diagnostics']['spearman_train']:.4f}"
          f"  CV-R²={r['diagnostics']['cv_r2_mean']:.4f}  MaxVIF={r['diagnostics']['max_vif']:.2f}")
    print("=" * 80)
    print("\nCoefficients (OLS, standardised, with bootstrap CI):")
    print(r['coef_report'].to_string(index=False, float_format=lambda x: f'{x:+.3f}'))
    print("\nImportance decomposition:")
    print(r['importance'].to_string(index=False, float_format=lambda x: f'{x:+.3f}'))
    print("\nRank sensitivity (improvement direction; positive = better rank):")
    print(r['sens_rank_improve'].round(2).to_string())
    print()


for mid in sorted(results.keys()):
    display_model_summary(mid, results[mid], quality[mid])
```

- [ ] **Step 2: Run + visually confirm**

---

### Task 21: Final verification

- [ ] **Step 1: Confirm all 8 models wrote artifacts**

```python
for mid in [c['id'] for c in MODEL_CONFIGS]:
    p = PER_MODEL_DIR / mid
    files = ['coefficients_ols.csv', 'coefficients_enet.csv', 'importance.csv',
             'vif.csv', 'sensitivity_score.csv', 'sensitivity_rank.csv',
             'sensitivity_rank_improve.csv', 'diagnostics.json',
             'model_ols.pkl', 'training_frame.parquet']
    missing = [f for f in files if not (p / f).exists()]
    print(f"{mid}: {'OK' if not missing else 'MISSING ' + ', '.join(missing)}")
```

- [ ] **Step 2: Spot-check the cross-model `cross_model_drivers` sheet**

Open `outputs/summary_comparison.xlsx`. The `cross_model_drivers` sheet should classify each feature into one of `{Robust positive driver, Robust negative driver, Unstable / sign-flips, Weak}`. Read the `Verdict` column — features whose verdict surprises you (e.g., `AcceptanceRate` not robustly negative, or `PeerScore` not robustly positive) are signals worth investigating before trusting the report.

- [ ] **Step 3: Read `outputs/final_report.md` and validate the recommendation makes sense**

Specifically check:
- Does the recommended model's feature drivers align with the cross-model "robust driver" verdicts?
- Do the multicollinearity findings flag any feature that the report still treats as a top driver?
- Are there year-stability issues that contradict the recommendation?

If anything is inconsistent, look at the underlying CSVs/diagnostics rather than re-running.

---

## Self-Review

**Spec coverage**
- 8 models, 2×2×2 grid — `MODEL_CONFIGS` (Task 15).
- Causal interpretation over predictive accuracy — OLS bootstrap + Shapley + drop-one importance + sign coherence + VIF, all aggregated into a `causal-quality scorecard` (M9, Task 16). Predictive metrics are reported but explicitly not used for selection.
- Clean comparison report for all metrics the user asked for — sensitivity matrix (`sens_*.csv`), coefficient significance/direction/% contribution (`coefficients_ols.csv` + `importance.csv`), per-model artifacts under `outputs/per_model/`, cross-model `summary_comparison.xlsx`, narrative `final_report.md`.
- My opinion on the best model with sufficient details — `final_report.md`, sections "Executive summary" and "Recommendation rationale", with the M9 scorecard table inline and a "when to pick a different model" override panel.

**Placeholder scan**
- No `TODO`, `fill in details`, or unspecified function bodies. Every step is concrete code.

**Type / signature consistency**
- `prepare_model_frame` → `(df_raw, features)`. `build_design_matrix(df_raw, features)` → `(X_std, y, fit_params)`. `transform_like(df_raw, features, fit_params)` reuses `fit_params` to keep transforms identical. `fit_ols_bootstrap(X, y, ...)` and `fit_enet_bootstrap(X, y, ...)` consume the standardised matrix. `compute_sensitivity` operates on raw `df_raw` and re-applies `transform_like` per perturbation. The `EXPECTED_SIGN_TRANSFORMED` map is consulted by `summarise_ols` for the sign-coherence audit.

**Open risks (already flagged in `final_report.md` caveats)**
- KNN imputation is done once before CV → mild leakage in CV-R² reported. Acceptable for causal interpretation; explicitly noted to the user.
- School-year independence assumption in stacked configs is non-trivial; the year-stability check + the GroupKFold CV partially mitigate.
- `GMAT_Final_1`'s floor placeholder will likely show as a stable feature with relatively low Shapley share — the placeholder compresses variance. This is a real characteristic of the data, not a bug; flagged in the caveats.

---

## Execution Handoff

Plan saved to `D:\work\US news\notebooks\new_models\plans\2026-05-16-eight-model-ranking-build.md`. Two execution options:

1. **Subagent-driven (recommended)** — I dispatch a fresh subagent per task, review the result, then move to the next. Fast iteration, clean context per task.
2. **Inline execution** — I execute the tasks myself in this session using `superpowers:executing-plans`, with batched checkpoints for review.

Which approach?
