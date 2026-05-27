# M11 — Bounded Methodology-Weight Model: Implementation Plan

**Status:** Draft awaiting user confirmation on the weight table and constraint type.

**Goal:** Build M11, an alternative to M5 that uses the same 8 features but constrains each feature's **standardized-β contribution percentage** to lie within a corridor around US News's **published methodology weights**. Cross-validated. Adds a "methodology-anchored" view to the dashboard alongside the OLS-driven M1-M10.

**Why this is a meaningfully different model:**
- M1-M10 fit β freely; the importance distribution emerges from the data.
- M11 fits β with hard constraints; the importance distribution is forced to track US News's published weights.
- That gives the user a sanity-check view: "If US News really uses *these* weights, can the model still fit the OverallScore well?" The answer is a direct test of whether their published methodology, applied via a linear model on the (transformed, standardized) features we have, can reproduce the score they publish.

---

## 1. Definition of "contribution percentage"

For a fitted linear model on standardized transformed features:

```
contribution_i = |β_i| / Σ_j |β_j| × 100
```

This is the same `Beta_Share_Pct` metric already in `outputs/per_model/*/importance.csv` and shown on the dashboard's Overview tab. It captures the per-feature share of total coefficient magnitude.

**Why this is the right target:** US News's published weights apply to *normalized sub-scores* (each raw input mapped to a 0-100 scale before weighting). Our transformed-and-standardized features are functionally equivalent to those normalized sub-scores — both are centered, unit-variance, monotonic in the underlying signal. So when US News says "PeerScore is weighted at 25%," it means a one-σ change in PeerScore drives ~25% of the score-change magnitude — which is exactly what `|β_i| / Σ|β_j|` measures in our linear model on standardized features.

---

## 2. Proposed US News methodology weights (FOR USER CONFIRMATION)

The 2026 published methodology has not been re-extracted live (WebFetch timed out). Below are the weights I'd use unless you correct them — they reflect the standard structure of the recent US News Business School methodology, **renormalized** to sum to 100% over the 8 features in M5 (since M5 excludes `ProfessionSalaryRank`, which absorbs a few percent in US News's full methodology).

### Option A — full canonical weights (with ProfessionSalaryRank)

| Feature | Sub-category | Weight |
|---------|--------------|-------:|
| PeerScore | Quality assessment | 25.00% |
| RecruiterScore | Quality assessment | 15.00% |
| AvgSalaryBonus | Placement success | 14.00% |
| Employed3Mo | Placement success | 14.00% |
| EmployedAtGrad | Placement success | 7.00% |
| GMAT_Combined | Student selectivity | 16.25% |
| MedianGPA | Student selectivity | 7.50% |
| AcceptanceRate | Student selectivity | 1.25% |
| **Subtotal (M5 features)** | | **99.00%** |
| _ProfessionSalaryRank_ | _Placement success_ | _~1.00%_ (M5 doesn't include) |
| **Total** | | **100.00%** |

### Option B — M5 features renormalized to 100%

| Feature | M5-only weight (renormalized) |
|---------|------------------------------:|
| PeerScore | 25.25% |
| RecruiterScore | 15.15% |
| AvgSalaryBonus | 14.14% |
| Employed3Mo | 14.14% |
| EmployedAtGrad | 7.07% |
| GMAT_Combined | 16.41% |
| MedianGPA | 7.58% |
| AcceptanceRate | 1.26% |
| **Total** | **100.00%** |

**Decision needed from user:** confirm Option B (renormalized — recommended so the target sums to exactly 100%), or override any specific weight, or paste the official numbers from the 2026 methodology page.

---

## 3. Constraint design — three options

### Option C1 — Hard equality (most restrictive)

Force `|β_i| / Σ|β_j| = w_i` exactly. With expected signs fixed and the ratios fixed, the model has only **one free parameter** (the overall scale `α`) plus an intercept:

```
score_pred = α · (Σ_i s_i · w_i · z_i) + c
```

where `s_i = ±1` is the expected sign, `z_i` is the standardized transformed feature, `w_i` is the US News weight, `α` and `c` are fit by least-squares.

**Pros:** Maximally interpretable — coefficients map directly to published weights.
**Cons:** Only 2 free parameters → strong assumption, likely lower R² than M5. R² becomes a one-number summary of "how well does the *published methodology* explain the score?"

### Option C2 — Soft bandwidth (recommended)

Constrain `|β_i| / Σ|β_j| ∈ [w_i − ε, w_i + ε]` for each feature i, where ε is a tolerance (e.g., ±2 percentage points). Plus expected signs.

This is a quadratic programming problem solvable with `scipy.optimize.minimize` (method `trust-constr` or SLSQP). Free parameters: K + 1 (intercept), with K-1 binding constraints in tight regions of the corridor.

**Pros:** Lets the data adjust within methodology bounds. R² closer to M5 if methodology is approximately right.
**Cons:** Need to pick ε. Solver convergence can be finicky for very tight ε.

### Option C3 — Penalty (soft constraint, no bounds)

Standard OLS objective + penalty term that pulls contributions toward methodology weights:

```
loss = ||y - Xβ||² + λ · Σ_i ( |β_i|/Σ|β_j| − w_i )²
```

with sign penalties for any wrong sign. λ controls how strongly the model respects methodology vs data.

**Pros:** Smoothly interpolates between "free OLS" (λ=0) and "methodology-exact" (λ→∞). Easy to fit.
**Cons:** λ is a tuning parameter (could pick via CV); contribution percentages aren't strictly bounded — they're just pulled toward targets.

### My recommendation: C2 (soft bandwidth)

The user said "**bounded** cross cv model that **maintains** contribution within US News methodology weights" — those words map cleanly to C2. The bandwidth ε is the only knob to tune; I'll default to **ε = 2 percentage points**, which lets:
- PeerScore contribution be in [23%, 27%]
- AcceptanceRate contribution be in [0%, 3.26%] (clipped to ≥ 0)
- etc.

If the user wants something stricter, ε=0.5pp gets close to C1; if looser, ε=5pp behaves nearly free.

---

## 4. Algorithm details

### 4.1 Optimization variable
Let `β ∈ ℝ^K` (K=8). Decision variables: `β` plus intercept `c`.

### 4.2 Sign handling
We separate magnitude from sign:
```
β_i = s_i · m_i,  m_i ≥ 0
```
where `s_i` is the expected sign from `EXPECTED_SIGN_TRANSFORMED` (fixed) and `m_i` is a non-negative magnitude variable. The optimization runs over `m_i` and `c`.

### 4.3 Objective
Squared loss over the standardized transformed design matrix:
```
minimize  ||y − X β − c||²
       =  ||y − X (S m) − c||²       (S is diag(signs))
```

### 4.4 Constraints
For each i in 1..K:
```
m_i / Σ_j m_j  ∈  [max(0, w_i − ε), w_i + ε]
m_i ≥ 0
```

Equivalent linear form (avoiding division):
```
m_i ≥ (w_i − ε) · Σ_j m_j     # lower bound
m_i ≤ (w_i + ε) · Σ_j m_j     # upper bound
```

Both are linear in `m`, so the full problem is a quadratic program with linear constraints. SLSQP or trust-constr from scipy handles this directly.

### 4.5 Solver setup

```python
from scipy.optimize import minimize, LinearConstraint

K = len(features)
S = np.diag([+1 if EXPECTED_SIGN_TRANSFORMED[f] == '+' else -1 for f in features])
X_signed = X.values @ S  # absorbed signs into design matrix
y = y.values

# decision: theta = [m_1, ..., m_K, c]
def loss(theta):
    m = theta[:K]
    c = theta[K]
    resid = y - X_signed @ m - c
    return float(resid @ resid)

# constraints: for each i, lower and upper bound rows
def make_constraints(weights, eps):
    A_lo = np.zeros((K, K + 1))
    A_hi = np.zeros((K, K + 1))
    for i in range(K):
        # m_i - (w_i - eps) * sum(m_j) >= 0  =>  row in A_lo
        A_lo[i, :K] = -(weights[i] - eps)
        A_lo[i, i] += 1.0
        # (w_i + eps) * sum(m_j) - m_i >= 0  =>  row in A_hi
        A_hi[i, :K] = +(weights[i] + eps)
        A_hi[i, i] -= 1.0
    A = np.vstack([A_lo, A_hi])
    return LinearConstraint(A, 0, np.inf)

# magnitude non-negativity
bounds = [(0, None)] * K + [(None, None)]  # c unbounded

theta0 = np.concatenate([weights * (np.std(y) / np.sqrt(K)), [np.mean(y)]])  # warm start
res = minimize(loss, theta0, method='trust-constr',
               bounds=bounds, constraints=make_constraints(weights, eps))

m_hat = res.x[:K]
c_hat = res.x[K]
beta_hat = pd.Series(S @ m_hat, index=features)
```

The warm-start with `weights × scale` makes convergence reliable.

### 4.6 Cross-validation
Wrap the solver in 5-fold CV (`KFold(n_splits=5, shuffle=True, random_state=42)`). For each fold:
1. Fit constrained model on training fold.
2. Predict on test fold (apply the same `transform_like` using fold-internal `fit_params`).
3. Record fold R², Spearman, MAE, RMSE.

Report `cv_r2_mean ± cv_r2_std` (etc.) in `diagnostics.json`.

### 4.7 Bootstrap CIs
For each bootstrap iteration (N_BOOTSTRAP = 1000 — fewer than the 2000 used elsewhere because each constrained fit is slower):
1. Resample (X, y) with replacement.
2. Re-fit the constrained model.
3. Record β.

Compute Lower/Upper 95% CI, sign stability %, etc. — same `summarise_ols`-style report.

---

## 5. Output artifacts (match M1-M10 format exactly)

Under `outputs/per_model/M11/`:

| File | Contents |
|------|----------|
| `coefficients_ols.csv` | Constrained β with Std_Bootstrap, 95% CI, Sign_Stability_Pct, Is_Significant, Expected_Sign, Sign_Matches |
| `coefficients_enet.csv` | _Skip for M11_ (placeholder with one row noting "n/a — constrained model") |
| `importance.csv` | Same three measures: Beta_Share_Pct (by construction ≈ US News weights), DropOne_DeltaR2, Shapley_R2_Share_Pct |
| `vif.csv` | Same as M5 (identical features) |
| `sensitivity_score.csv` | Median Δ score per feature × perturbation |
| `sensitivity_rank.csv` | Median Δ rank per feature × perturbation |
| `sensitivity_rank_improve.csv` | Reverse-flipped Δ rank |
| `diagnostics.json` | R², adj-R², Spearman, CV-R², CV-Spearman, MAE, RMSE, condition_number, max_vif, intercept, constraint metadata (weights, ε) |
| `model_ols.pkl` | `{'beta': pd.Series, 'intercept': float, 'fit_params': dict, 'features': list, 'constraint_meta': {'weights': dict, 'eps': float}}` |
| `training_frame.parquet` | Same as M5 |

The dashboard reads each artifact by name and doesn't care that M11's β came from a constrained fit — the pickle layout is the same shape.

---

## 6. Dashboard integration

### 6.1 MODEL_CONFIGS entry
```python
'M11': {
    'include_profession_rank': False,
    'gmat_col': 'GMAT_Final_1',
    'years': [2026],
    'extras': [],
    'kind': 'methodology_bounded',  # new optional field
},
```

### 6.2 Sidebar
Add a brief help-text tag: "M11 = M5 with coefficient contributions bounded to US News methodology weights (±2pp)."

### 6.3 Composite quality score
M11 will score differently:
- `sign_coherence` → **1.000** (signs forced).
- `vif_clean` → same as M5 (0.353), since features are identical.
- `coef_stability` → likely **worse** than M5: constrained bootstrap fits have tighter coefficient distributions but the constraint itself forces β toward fixed ratios, which suppresses some genuine variation. CV of |β| could be artificially low (always near weights) or artificially high if the optimizer struggles near a binding constraint. Expect 0.5-0.8.
- `importance_agreement` → could go either way. By construction, `Beta_Share_Pct` ≈ weights. But `Shapley_R2_Share_Pct` is collinearity-invariant and reflects the *actual* explanatory contribution, which can diverge from weights (e.g., AcceptanceRate has weight 1.25% but Shapley might give it more). Disagreement here is informative — it tells you which weights US News "should" use given the data.
- `bootstrap_sign_stability` → **1.000** (signs are fixed in the optimization).

Net: M11's composite will likely be in the 0.65-0.75 range. The number isn't the point — the *story* is: "Did the data accept methodology weights gracefully (high composite) or fight them (lower composite, big Shapley/β-share gap)?"

### 6.4 Overview tab
The Coefficient plot will show all bootstrap CIs hugging the constraint corridor — visually obvious. Add a caption explaining this is by design.

### 6.5 Cross-Model tab
M11 will sit alongside M1-M10 in the composite chart and β heatmap. The driver-verdict table will likely call every feature a "Robust positive (or negative) driver" because signs are forced — note this in the verdict logic if it's misleading.

---

## 7. Validation strategy — how do we know the model is sensible?

1. **R² gap to M5:** If M5 has R² = 0.973 and M11 has R² < 0.90 with ε=2pp, that's a *finding* — US News methodology weights do not optimally fit their own published scores. If R² is within ~0.01 of M5, the weights are essentially data-optimal.
2. **CV-R² should be close to train-R²:** Constrained models are more parsimonious, less prone to overfitting. CV-R² should be ~CV-R² of M5 or better.
3. **Constraint activity:** Track which constraints are *binding* at the optimum (i.e., contribution at the edge of the corridor) vs slack. A heavily-binding model means the data is fighting the methodology; lots of slack means the methodology is being conservative.
4. **Shapley vs |β|-share gap:** Compute the L1 distance between Shapley shares and US News weights. If small, the methodology weights match the data's true marginal contributions. If large, the methodology is mis-weighted relative to what the data implies.

These four numbers should appear in the on-screen output and the report.

---

## 8. Task breakdown (after user approves the plan)

- [ ] **T1.** Define `US_NEWS_WEIGHTS` dict (Option B, M5-renormalized) with user confirmation.
- [ ] **T2.** Write `_fit_m11.py` — new standalone script with `fit_bounded_methodology(X, y, weights, eps)` function. ~250 lines.
- [ ] **T3.** Add bootstrap wrapper around the constrained fit (1000 iters).
- [ ] **T4.** Run on the 2026 sheet (same data as M5).
- [ ] **T5.** Save all 10 artifact files in `outputs/per_model/M11/`.
- [ ] **T6.** Add 'M11' to `MODEL_CONFIGS` in `dashboard.py` with the new `kind` field.
- [ ] **T7.** Optional: tweak the driver-verdict logic in the Cross-Model tab to flag M11 separately (since its signs are forced, not learned).
- [ ] **T8.** AppTest smoke test for all 11 models.
- [ ] **T9.** Commit + push.

Estimated implementation time: ~45-60 minutes once the plan is approved.

---

## 9. Open questions for the user (please confirm before implementation)

1. **Weight table:** Use Option B (M5 features renormalized to 100%) or do you have specific weights from the published 2026 methodology page you want me to use? If the latter, please paste them — I want to avoid guessing.

2. **Bandwidth ε:** Default ±2 percentage points. Want stricter (±1pp), looser (±5pp), or exact equality (ε=0, which collapses to a 1-parameter model)?

3. **Sign enforcement:** I plan to force expected signs (e.g., AcceptanceRate is negative, PeerScore is positive). Do you want this, or should signs be free within the magnitude bounds?

4. **Bootstrap iterations:** 1000 (half of M1-M10's 2000) to keep runtime under 2 minutes. OK?

5. **Naming:** `M11` is the obvious next ID. If you'd prefer a more descriptive name in the dashboard label (e.g., "M11 (methodology-bounded)"), let me know.

Once you reply on these, I'll execute the build.
