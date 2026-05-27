"""Fit M11: methodology-bounded constrained-LS model.

M11 uses the same 8 features as M5, but constrains each feature's
|β| / Σ|β| contribution percentage to lie within ±5 percentage points of
the US News published methodology weight for that feature. Expected signs
are enforced. 2000 bootstrap iterations + 5-fold CV.

Run with:
    python _fit_m11.py
"""
from __future__ import annotations
import json
import pickle
from itertools import combinations
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import LinearConstraint, minimize
from scipy.stats import norm, rankdata, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


NEW_MODELS_DIR = Path(__file__).parent
DATA_FILE = NEW_MODELS_DIR / "US News Data 2025 - 2026.xlsx"
PER_MODEL_DIR = NEW_MODELS_DIR / "outputs" / "per_model"

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
CV_FOLDS = 5

LOG_VARS = ['AvgSalaryBonus', 'GMAT_Combined']
LOGIT_VARS = ['EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate']
INV_NORM_VARS = ['ProfessionSalaryRank']

EXPECTED_SIGN_TRANSFORMED = {
    'PeerScore': '+', 'RecruiterScore': '+', 'MedianGPA': '+',
    'AvgSalaryBonus': '+', 'GMAT_Combined': '+',
    'EmployedAtGrad': '+', 'Employed3Mo': '+',
    'AcceptanceRate': '-',
    'ProfessionSalaryRank': '+',
}

PERTURBATIONS = [-0.10, -0.05, -0.01, 0.01, 0.05, 0.10]
REVERSE_DIRECTION = {'AcceptanceRate', 'ProfessionSalaryRank'}

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
    'GMAT Final 1': 'GMAT_Final_1',
}

# === US News methodology weights and per-feature boundaries ===
# These are the published US News weights for the 2026 Best Business Schools
# ranking (M5's 8 features; "Salary by profession" at 10% is excluded from M5).
#
# We keep the weights at their published values (which sum to 0.90, not 1.0).
# The fitted |beta|/sum(|beta|) contributions ALWAYS sum to 1.0 by construction,
# so the missing 10pp must come from somewhere within the per-feature corridors
# below — concentrated in the features with the loosest boundary widths.
US_NEWS_WEIGHTS = {
    'AvgSalaryBonus':  0.20,
    'Employed3Mo':     0.13,
    'PeerScore':       0.125,
    'RecruiterScore':  0.125,
    'MedianGPA':       0.10,
    'EmployedAtGrad':  0.07,
    'AcceptanceRate':  0.02,
    'GMAT_Combined':   0.13,
}
assert abs(sum(US_NEWS_WEIGHTS.values()) - 0.90) < 1e-9, \
    "Published weights for M5 features should sum to 0.90 (Salary by profession excluded)"

# Per-feature boundary widths in percentage points (from the user-supplied table).
# Each fitted contribution must lie within w_i +/- EPSILONS_PP[f] / 100.
EPSILONS_PP = {
    'AvgSalaryBonus':  1.0,
    'Employed3Mo':     5.0,
    'PeerScore':       3.0,
    'RecruiterScore':  3.0,
    'MedianGPA':       3.0,
    'EmployedAtGrad':  5.0,
    'AcceptanceRate':  2.0,
    'GMAT_Combined':   3.0,
}


# =================== Feature transforms (mirror notebook) ===================

def transform_features(df, features, rank_n=None):
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
            out[col] = -1.0 * norm.ppf(pct)
    return out, rank_counts


def cap_outliers(df, features, lo=0.05, hi=0.05):
    out = df.copy()
    for col in features:
        if col in out.columns:
            lo_v, hi_v = out[col].quantile(lo), out[col].quantile(1 - hi)
            out[col] = out[col].clip(lo_v, hi_v)
    return out


def build_design_matrix(df, features):
    df_c = cap_outliers(df, features)
    df_t, rank_counts = transform_features(df_c, features)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df_t), columns=features, index=df.index)
    fp = {
        'rank_counts': rank_counts,
        'caps': {c: (df[c].quantile(0.05), df[c].quantile(0.95)) for c in features},
        'scaler_mean': dict(zip(features, scaler.mean_)),
        'scaler_scale': dict(zip(features, scaler.scale_)),
    }
    return X, df['OverallScore'].astype(float), fp


def transform_like(df_raw, features, fp):
    out = df_raw[features].copy().astype(float)
    for c, (lo, hi) in fp['caps'].items():
        if c in out.columns:
            out[c] = out[c].clip(lo, hi)
    out, _ = transform_features(out, features, rank_n=fp['rank_counts'])
    for c in features:
        out[c] = (out[c] - fp['scaler_mean'][c]) / fp['scaler_scale'][c]
    return out


def predict_from_raw(df_raw, features, fp, beta, intercept):
    X = transform_like(df_raw, features, fp)
    return X.values @ beta.reindex(features).values + intercept


# =================== Constrained least squares ===================

def _signs_vector(features):
    """Return ±1 vector matching EXPECTED_SIGN_TRANSFORMED."""
    return np.array(
        [+1.0 if EXPECTED_SIGN_TRANSFORMED[f] == '+' else -1.0 for f in features]
    )


def _make_bandwidth_constraint(weights_arr, eps_arr, K):
    """Linear constraint enforcing  w_i - eps_i  <=  m_i / sum(m_j)  <=  w_i + eps_i.

    Theta is [m_1, ..., m_K, c]. Both inequalities become linear in theta:
      m_i - (w_i - eps_i) * sum(m_j) >= 0    (lower bound, trivial when w_i - eps_i < 0)
      (w_i + eps_i) * sum(m_j) - m_i >= 0    (upper bound)
    """
    A_lo = np.zeros((K, K + 1))
    A_hi = np.zeros((K, K + 1))
    for i in range(K):
        A_lo[i, :K] = -(weights_arr[i] - eps_arr[i])
        A_lo[i, i] += 1.0
        A_hi[i, :K] = +(weights_arr[i] + eps_arr[i])
        A_hi[i, i] -= 1.0
    A = np.vstack([A_lo, A_hi])  # 2K x (K+1)
    return LinearConstraint(A, 0.0, np.inf)


def fit_constrained_one(X_vals, y_vals, signs, weights_arr, eps_arr,
                        maxiter=300, x0=None):
    """One constrained-LS fit. Returns (beta_array, intercept, info)."""
    K = X_vals.shape[1]
    # Pre-multiply X by signs so optimization is over m only.
    X_signed = X_vals * signs[None, :]   # column-wise scale

    def loss(theta):
        m = theta[:K]
        c = theta[K]
        pred = X_signed @ m + c
        resid = y_vals - pred
        return float(resid @ resid)

    def grad(theta):
        m = theta[:K]
        c = theta[K]
        pred = X_signed @ m + c
        resid = y_vals - pred
        g_m = -2.0 * (X_signed.T @ resid)
        g_c = -2.0 * resid.sum()
        return np.concatenate([g_m, [g_c]])

    bounds = [(0.0, None)] * K + [(None, None)]
    cons = _make_bandwidth_constraint(weights_arr, eps_arr, K)

    if x0 is None:
        # Warm start: use published weights renormalized so initial ratios sum
        # to 1.0 (which satisfies the unit-sum invariant of the contribution
        # space). trust-constr handles small infeasibility at start gracefully.
        renorm = weights_arr / weights_arr.sum()
        scale = float(np.linalg.norm(y_vals - y_vals.mean())) / max(K, 1)
        x0 = np.concatenate([renorm * scale, [y_vals.mean()]])

    res = minimize(
        loss, x0, jac=grad,
        method='trust-constr',
        bounds=bounds, constraints=cons,
        options={'maxiter': maxiter, 'gtol': 1e-8, 'xtol': 1e-10, 'verbose': 0},
    )
    m_hat = res.x[:K]
    c_hat = float(res.x[K])
    beta = signs * m_hat
    return beta, c_hat, {'success': bool(res.success), 'fun': float(res.fun),
                        'niter': int(res.nit)}


# =================== Bootstrap ===================

def fit_bootstrap(X, y, features, weights, eps_pp_map, n_iter=N_BOOTSTRAP,
                  seed=RANDOM_SEED, n_jobs=-1):
    K = len(features)
    signs = _signs_vector(features)
    weights_arr = np.array([weights[f] for f in features])
    eps_arr = np.array([eps_pp_map[f] / 100.0 for f in features])

    # Point estimate on full data
    point_beta, point_c, info = fit_constrained_one(
        X.values, y.values, signs, weights_arr, eps_arr,
    )
    if not info['success']:
        print(f"  WARNING: point estimate optimizer did not converge: {info}")

    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**31 - 1, size=n_iter)

    def _one(s):
        rs = np.random.RandomState(int(s))
        idx = rs.choice(len(X), size=len(X), replace=True)
        beta, _c, _info = fit_constrained_one(
            X.values[idx], y.values[idx], signs, weights_arr, eps_arr,
        )
        return beta

    boot = np.array(Parallel(n_jobs=n_jobs)(delayed(_one)(s) for s in seeds))
    return {
        'beta': pd.Series(point_beta, index=features),
        'intercept': point_c,
        'boot_coefs': pd.DataFrame(boot, columns=features),
        'boot_intercept': None,
        'info': info,
    }


def summarise_ols_style(result):
    boot = result['boot_coefs']
    beta = result['beta']
    lo = boot.quantile(0.025)
    hi = boot.quantile(0.975)
    std = boot.std()
    same_sign = boot.apply(
        lambda col: float((np.sign(col) == np.sign(beta[col.name])).mean()),
    )
    rep = pd.DataFrame({
        'Beta_OLS': beta, 'Std_Bootstrap': std,
        'Lower_95_CI': lo, 'Upper_95_CI': hi,
        'Sign_Stability_Pct': (same_sign * 100).round(1),
    })
    rep['Is_Significant'] = ~((rep['Lower_95_CI'] <= 0) & (rep['Upper_95_CI'] >= 0))
    rep['Observed_Sign'] = np.where(rep['Beta_OLS'] >= 0, '+', '-')
    rep['Expected_Sign'] = rep.index.map(EXPECTED_SIGN_TRANSFORMED.get)
    rep['Sign_Matches'] = rep['Observed_Sign'] == rep['Expected_Sign']
    return rep.reset_index().rename(columns={'index': 'Feature'})


# =================== Diagnostics (VIF, importance, CV, sensitivity) ===================

def compute_vif(X):
    Xc = sm.add_constant(X.values)
    return pd.DataFrame(
        [{'Feature': name,
          'VIF': float(variance_inflation_factor(Xc, i + 1))}
         for i, name in enumerate(X.columns)]
    )


def condition_number(X):
    s = np.linalg.svd(X.values, compute_uv=False)
    return float(s.max() / s.min())


def compute_importance(X, y, beta, max_shapley=12):
    feats = list(X.columns)
    beta_share = beta.abs() / beta.abs().sum() * 100
    full = LinearRegression().fit(X, y)
    r2_full = r2_score(y, full.predict(X))
    drop_one = {}
    for f in feats:
        Xm = X.drop(columns=[f])
        m = LinearRegression().fit(Xm, y)
        drop_one[f] = r2_full - r2_score(y, m.predict(Xm))
    if len(feats) > max_shapley:
        shap = {f: np.nan for f in feats}
    else:
        r2_cache = {(): 0.0}
        for k in range(1, len(feats) + 1):
            for combo in combinations(feats, k):
                Xc = X[list(combo)].values
                r2_cache[tuple(sorted(combo))] = r2_score(
                    y, LinearRegression().fit(Xc, y).predict(Xc),
                )
        K = len(feats)
        shap = {}
        for f in feats:
            other = [g for g in feats if g != f]
            total = 0.0
            for k in range(0, K):
                weight = 1.0 / (K * comb(K - 1, k))
                for subset in combinations(other, k):
                    with_f = tuple(sorted(list(subset) + [f]))
                    without_f = tuple(sorted(subset))
                    total += weight * (r2_cache[with_f] - r2_cache[without_f])
            shap[f] = total
        total_r2 = sum(shap.values())
        shap = {f: (v / total_r2 * 100 if total_r2 > 0 else 0.0)
                for f, v in shap.items()}
    df = pd.DataFrame({
        'Feature': feats,
        'Beta_Share_Pct': beta_share.values,
        'DropOne_DeltaR2': [drop_one[f] for f in feats],
        'Shapley_R2_Share_Pct': [shap[f] for f in feats],
    })
    df['Rank_By_Beta'] = df['Beta_Share_Pct'].rank(
        ascending=False, method='min').astype(int)
    df['Rank_By_DropOne'] = df['DropOne_DeltaR2'].rank(
        ascending=False, method='min').astype(int)
    df['Rank_By_Shapley'] = df['Shapley_R2_Share_Pct'].rank(
        ascending=False, method='min').astype(int)
    return df


def cv_metrics_constrained(df_raw, features, weights, eps_pp_map,
                           n_splits=CV_FOLDS, seed=RANDOM_SEED):
    """5-fold CV; each fold re-fits the constrained model."""
    y = df_raw['OverallScore'].values
    splits = list(KFold(n_splits=n_splits, shuffle=True,
                        random_state=seed).split(df_raw))
    signs = _signs_vector(features)
    weights_arr = np.array([weights[f] for f in features])
    eps_arr = np.array([eps_pp_map[f] / 100.0 for f in features])

    r2l, spl, mael, rmsel = [], [], [], []
    for tr, te in splits:
        df_tr, df_te = df_raw.iloc[tr], df_raw.iloc[te]
        X_tr, y_tr, fp = build_design_matrix(df_tr, features)
        beta_arr, c, _ = fit_constrained_one(
            X_tr.values, y_tr.values, signs, weights_arr, eps_arr,
        )
        X_te = transform_like(df_te[features], features, fp)
        y_te = df_te['OverallScore'].values
        p = X_te.values @ beta_arr + c
        r2l.append(r2_score(y_te, p))
        spl.append(spearmanr(y_te, p).statistic)
        mael.append(mean_absolute_error(y_te, p))
        rmsel.append(float(np.sqrt(mean_squared_error(y_te, p))))
    return {
        'cv_r2_mean': float(np.mean(r2l)), 'cv_r2_std': float(np.std(r2l)),
        'cv_spearman_mean': float(np.mean(spl)),
        'cv_spearman_std': float(np.std(spl)),
        'cv_mae_mean': float(np.mean(mael)),
        'cv_rmse_mean': float(np.mean(rmsel)),
    }


def _safe_perturb(values, feature, delta):
    new = values * (1.0 + delta)
    if feature in ('AcceptanceRate', 'EmployedAtGrad', 'Employed3Mo'):
        return np.clip(new, 1e-3, 1 - 1e-3)
    if feature == 'MedianGPA':
        return np.clip(new, 0.0, 4.0)
    if feature == 'GMAT_Combined':
        return np.clip(new, 200.0, 800.0)
    if feature == 'ProfessionSalaryRank':
        return np.clip(np.round(new), 1, None)
    return new


def compute_sensitivity(df_raw, features, fp, beta, intercept):
    base_pred = predict_from_raw(df_raw, features, fp, beta, intercept)
    base_rank = pd.Series(rankdata(-base_pred, method='min'),
                          index=df_raw.index)
    cols = [f'{int(d*100):+d}%' for d in PERTURBATIONS]
    score_rows, rank_rows, ri_rows = [], [], []
    for f in features:
        s, r, ri = {'Feature': f}, {'Feature': f}, {'Feature': f}
        for d, c in zip(PERTURBATIONS, cols):
            df_p = df_raw.copy()
            df_p[f] = df_p[f].astype(float)
            df_p[f] = _safe_perturb(df_p[f].values, f, d)
            new_p = predict_from_raw(df_p, features, fp, beta, intercept)
            new_r = pd.Series(rankdata(-new_p, method='min'), index=df_p.index)
            ds = float(np.median(new_p - base_pred))
            dr = float((new_r - base_rank).median())
            s[c] = ds; r[c] = dr
            sign_flip = -1.0 if f in REVERSE_DIRECTION else 1.0
            ri[c] = -dr * sign_flip
        score_rows.append(s); rank_rows.append(r); ri_rows.append(ri)
    return (pd.DataFrame(score_rows).set_index('Feature'),
            pd.DataFrame(rank_rows).set_index('Feature'),
            pd.DataFrame(ri_rows).set_index('Feature'))


# =================== Data prep — same as M5 ===================

def load_m5_frame():
    df = pd.read_excel(DATA_FILE, sheet_name='2026')
    df = df.rename(columns=RENAME_MAP)
    df['GMAT_Final_1'] = pd.to_numeric(df['GMAT_Final_1'], errors='coerce')
    df['Year'] = 2026
    df['MedianGPA'] = df['MedianGPA'].fillna(df['MedianGPA'].min())
    df['GMAT_Combined'] = df['GMAT_Final_1']
    features = ['PeerScore', 'RecruiterScore', 'MedianGPA',
                'AvgSalaryBonus', 'EmployedAtGrad', 'Employed3Mo',
                'AcceptanceRate', 'GMAT_Combined']
    keep = ['School', 'Year', 'Rank', 'OverallScore'] + features
    df = df[keep].copy()
    # KNN-impute any residual NaN (mirror M5 behaviour) — use SimpleImputer-like
    # via the same knn_impute_missing function in the notebook. Keep it inline
    # to avoid the dependency:
    impute_targets = [c for c in features if df[c].isna().any()]
    if impute_targets:
        from sklearn.impute import KNNImputer
        ref_cols = [c for c in features if c not in impute_targets] or features
        work = list(dict.fromkeys(impute_targets + ref_cols))
        sub = df[work].astype(float).copy()
        sc = StandardScaler()
        ss = pd.DataFrame(
            sc.fit_transform(sub.fillna(sub.median(numeric_only=True))),
            columns=work, index=sub.index,
        )
        for c in impute_targets:
            ss.loc[sub[c].isna(), c] = np.nan
        imp = pd.DataFrame(
            KNNImputer(n_neighbors=5, weights='distance').fit_transform(ss),
            columns=work, index=sub.index,
        )
        out = pd.DataFrame(sc.inverse_transform(imp), columns=work,
                           index=sub.index)
        for c in impute_targets:
            df[c] = out[c]
    assert df[features].isna().sum().sum() == 0
    return df, features


# =================== Runner ===================

def main():
    np.random.seed(RANDOM_SEED)
    print("Loading M5 frame (2026)...")
    df_raw, features = load_m5_frame()
    print(f"Frame: {df_raw.shape}, features ({len(features)}): {features}")
    print("Per-feature constraint corridors (centered on published US News weight):")
    sum_w = 0.0
    sum_hi = 0.0
    sum_lo = 0.0
    for f in features:
        w = US_NEWS_WEIGHTS[f] * 100
        e = EPSILONS_PP[f]
        lo = max(0, w - e)
        hi = w + e
        sum_w += w; sum_hi += hi; sum_lo += lo
        print(f"  {f:22s}: w={w:5.2f}%, eps=+/-{e:.1f}pp, "
              f"corridor=[{lo:5.2f}%, {hi:5.2f}%]")
    print(f"  {'TOTAL':22s}: sum(w)={sum_w:5.2f}%, "
          f"sum(corridor)=[{sum_lo:.2f}%, {sum_hi:.2f}%], target sum=100%")

    X, y, fp = build_design_matrix(df_raw, features)

    print(f"\nFitting constrained model (bootstrap n_iter={N_BOOTSTRAP})...")
    result = fit_bootstrap(X, y, features, US_NEWS_WEIGHTS, EPSILONS_PP,
                           n_iter=N_BOOTSTRAP, n_jobs=-1)
    beta = result['beta']
    intercept = result['intercept']

    coef_report = summarise_ols_style(result)
    importance = compute_importance(X, y, beta)
    vif = compute_vif(X)
    cv = cv_metrics_constrained(df_raw, features, US_NEWS_WEIGHTS, EPSILONS_PP)

    # Verify constraint compliance for the point estimate
    contributions = (beta.abs() / beta.abs().sum()) * 100
    constraint_check = pd.DataFrame({
        'feature': features,
        'us_news_pct': [US_NEWS_WEIGHTS[f] * 100 for f in features],
        'epsilon_pp': [EPSILONS_PP[f] for f in features],
        'fitted_pct': contributions.values,
        'corridor_lo': [max(0, US_NEWS_WEIGHTS[f] * 100 - EPSILONS_PP[f])
                        for f in features],
        'corridor_hi': [US_NEWS_WEIGHTS[f] * 100 + EPSILONS_PP[f]
                        for f in features],
    })
    constraint_check['in_corridor'] = (
        (constraint_check['fitted_pct'] >= constraint_check['corridor_lo'] - 1e-6)
        & (constraint_check['fitted_pct'] <= constraint_check['corridor_hi'] + 1e-6)
    )
    print("\nConstraint compliance (point estimate):")
    print(constraint_check.to_string(index=False,
                                     float_format=lambda v: f"{v:6.2f}"))

    train_pred = X.values @ beta.values + intercept
    diag = {
        'config': {'id': 'M11', 'include_profession_rank': False,
                   'gmat_col': 'GMAT_Final_1', 'years': [2026],
                   'extras': [], 'kind': 'methodology_bounded',
                   'label': 'M11 (methodology-bounded)'},
        'n_schools': int(len(df_raw)),
        'n_features': len(features),
        'features': features,
        'r2_train': float(r2_score(y, train_pred)),
        'adj_r2_train': float(1 - (1 - r2_score(y, train_pred))
                              * (len(y) - 1) / (len(y) - len(features) - 1)),
        'spearman_train': float(spearmanr(y, train_pred).statistic),
        'mae_train': float(mean_absolute_error(y, train_pred)),
        'rmse_train': float(np.sqrt(mean_squared_error(y, train_pred))),
        'intercept': float(intercept),
        'condition_number': condition_number(X),
        'max_vif': float(vif['VIF'].max()),
        'bootstrap_iterations': N_BOOTSTRAP,
        **cv,
        'constraint_meta': {
            'kind': 'methodology_bounded',
            'weights_pct': {f: round(US_NEWS_WEIGHTS[f] * 100, 4)
                            for f in features},
            'epsilons_pp': {f: EPSILONS_PP[f] for f in features},
            'weights_sum_pct': round(sum(US_NEWS_WEIGHTS.values()) * 100, 4),
            'signs_enforced': True,
            'sign_map': {f: EXPECTED_SIGN_TRANSFORMED[f] for f in features},
        },
        'constraint_compliance': constraint_check.to_dict(orient='records'),
    }

    sens_score, sens_rank, sens_rank_improve = compute_sensitivity(
        df_raw, features, fp, beta, intercept,
    )

    out_dir = PER_MODEL_DIR / 'M11'
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_report.to_csv(out_dir / 'coefficients_ols.csv', index=False)
    # ENet not meaningful for a constrained model — write a placeholder
    pd.DataFrame([{'Feature': f, 'Beta_ENet': float('nan'), 'Sign': '',
                   'Sign_Stability_Pct': float('nan')} for f in features]
                 ).to_csv(out_dir / 'coefficients_enet.csv', index=False)
    importance.to_csv(out_dir / 'importance.csv', index=False)
    vif.to_csv(out_dir / 'vif.csv', index=False)
    sens_score.to_csv(out_dir / 'sensitivity_score.csv')
    sens_rank.to_csv(out_dir / 'sensitivity_rank.csv')
    sens_rank_improve.to_csv(out_dir / 'sensitivity_rank_improve.csv')
    with open(out_dir / 'diagnostics.json', 'w') as f:
        json.dump(diag, f, indent=2, default=float)
    with open(out_dir / 'model_ols.pkl', 'wb') as f:
        pickle.dump({
            'beta': beta, 'intercept': intercept,
            'fit_params': fp, 'features': features,
            'constraint_meta': diag['constraint_meta'],
        }, f)
    df_raw.to_parquet(out_dir / 'training_frame.parquet')

    print(f"\n=== M11 results ===")
    print(f"  R2_train         = {diag['r2_train']:.4f}")
    print(f"  CV-R2 (mean+/-sd)= {diag['cv_r2_mean']:.4f} +/- "
          f"{diag['cv_r2_std']:.4f}")
    print(f"  Spearman_train   = {diag['spearman_train']:.4f}")
    print(f"  CV-Spearman      = {diag['cv_spearman_mean']:.4f}")
    print(f"  Max VIF          = {diag['max_vif']:.2f}")
    print(f"  Intercept        = {diag['intercept']:.4f}")
    print(f"\nFitted standardised betas:")
    for _, row in coef_report.iterrows():
        print(f"  {row['Feature']:22s} beta = {row['Beta_OLS']:+.4f}  "
              f"contrib = {(abs(row['Beta_OLS']) / coef_report['Beta_OLS'].abs().sum() * 100):5.2f}%  "
              f"target = {US_NEWS_WEIGHTS[row['Feature']]*100:5.2f}%  "
              f"95% CI=[{row['Lower_95_CI']:+.3f}, {row['Upper_95_CI']:+.3f}]")
    print(f"\nArtifacts written to {out_dir}")


if __name__ == '__main__':
    main()
