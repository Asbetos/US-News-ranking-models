"""Fit M9 and M10 from the new dataset and write per-model artifacts in the
same format as M1..M8 (outputs/per_model/M9/, M10/).

  M9  = M5 + GRE_Final feature (no profession rank, GMAT_Final_1, 2026 only,
        plus the new GRE_Final from column R).
  M10 = M5 with GMAT_Final_1 replaced by GMAT_Final_3 (the GRE-imputed GMAT;
        15 "KNN Imputed Value" cells are KNN-filled here using the other
        non-GRE features as distance basis, so the imputed GMAT is not a
        proxy for GRE).

Run with:
    python _fit_m9_m10.py
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
from scipy.stats import norm, rankdata, spearmanr
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


NEW_MODELS_DIR = Path(__file__).parent
DATA_FILE = NEW_MODELS_DIR / "US News Data 2025 - 2026 Final 5.24.26.xlsx"
PER_MODEL_DIR = NEW_MODELS_DIR / "outputs" / "per_model"

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
N_ENET_BOOTSTRAP = 500
CV_FOLDS = 5

# Transformation groups — `GRE_Final` is log1p'd (same family of treatment as
# `GMAT_Combined`: a strictly positive test score whose effect tends to be
# concave in raw units).
LOG_VARS = ['AvgSalaryBonus', 'GMAT_Combined', 'GRE_Final']
LOGIT_VARS = ['EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate']
INV_NORM_VARS = ['ProfessionSalaryRank']

EXPECTED_SIGN_TRANSFORMED = {
    'PeerScore': '+', 'RecruiterScore': '+', 'MedianGPA': '+',
    'AvgSalaryBonus': '+', 'GMAT_Combined': '+', 'GRE_Final': '+',
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
    'GMAT Final 3': 'GMAT_Final_3',
    'GRE Final 1': 'GRE_Final',
}


# ====================== Feature transformations ======================

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


# ====================== KNN imputation ======================

def knn_impute_missing(df, impute_cols, reference_cols, n_neighbors=5):
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
    out_scaled = pd.DataFrame(imputer.fit_transform(sub_scaled), columns=work, index=sub.index)
    out = pd.DataFrame(scaler.inverse_transform(out_scaled), columns=work, index=sub.index)
    res = df.copy()
    for c in impute_cols:
        res[c] = out[c]
    return res


# ====================== OLS bootstrap ======================

def _ols_one(X, y):
    lr = LinearRegression().fit(X, y)
    return lr.coef_, lr.intercept_


def fit_ols_bootstrap(X, y, n_iter=N_BOOTSTRAP, seed=RANDOM_SEED, n_jobs=-1):
    pc, pi = _ols_one(X.values, y.values)
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**31 - 1, size=n_iter)

    def _one(s):
        rs = np.random.RandomState(int(s))
        idx = rs.choice(len(X), size=len(X), replace=True)
        return _ols_one(X.values[idx], y.values[idx])

    out = Parallel(n_jobs=n_jobs)(delayed(_one)(s) for s in seeds)
    coefs = np.array([o[0] for o in out])
    inters = np.array([o[1] for o in out])
    return {
        'beta': pd.Series(pc, index=X.columns),
        'intercept': float(pi),
        'boot_coefs': pd.DataFrame(coefs, columns=X.columns),
        'boot_intercept': inters,
    }


def summarise_ols(ols):
    boot = ols['boot_coefs']
    beta = ols['beta']
    lo = boot.quantile(0.025)
    hi = boot.quantile(0.975)
    std = boot.std()
    same_sign = boot.apply(lambda col: float((np.sign(col) == np.sign(beta[col.name])).mean()))
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


# ====================== ElasticNet bootstrap (robustness check) ======================

def _enet_one(X, y, seed):
    en = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                      cv=5, max_iter=10000, n_jobs=1, random_state=seed)
    en.fit(X, y)
    return en.coef_


def fit_enet_bootstrap(X, y, n_iter=N_ENET_BOOTSTRAP, seed=RANDOM_SEED, n_jobs=-1):
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 2**31 - 1, size=n_iter)

    def _one(s):
        rs = np.random.RandomState(int(s))
        idx = rs.choice(len(X), size=len(X), replace=True)
        return _enet_one(X.values[idx], y.values[idx], int(s))

    boot = np.array(Parallel(n_jobs=n_jobs)(delayed(_one)(s) for s in seeds))
    df = pd.DataFrame(boot, columns=X.columns)
    return {'beta': df.mean(), 'boot_coefs': df}


def summarise_enet(enet):
    boot = enet['boot_coefs']
    beta = enet['beta']
    return pd.DataFrame({
        'Feature': beta.index,
        'Beta_ENet': beta.values,
        'Sign': np.where(beta.values >= 0, '+', '-'),
        'Sign_Stability_Pct': (boot.apply(
            lambda c: (np.sign(c) == np.sign(beta[c.name])).mean()) * 100).round(1).values,
    })


# ====================== VIF, importance, CV, sensitivity ======================

def compute_vif(X):
    Xc = sm.add_constant(X.values)
    return pd.DataFrame(
        [{'Feature': name,
          'VIF': float(variance_inflation_factor(Xc, i + 1))}
         for i, name in enumerate(X.columns)])


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
                r2_cache[tuple(sorted(combo))] = r2_score(y, LinearRegression().fit(Xc, y).predict(Xc))
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
        shap = {f: (v / total_r2 * 100 if total_r2 > 0 else 0.0) for f, v in shap.items()}
    df = pd.DataFrame({
        'Feature': feats,
        'Beta_Share_Pct': beta_share.values,
        'DropOne_DeltaR2': [drop_one[f] for f in feats],
        'Shapley_R2_Share_Pct': [shap[f] for f in feats],
    })
    df['Rank_By_Beta'] = df['Beta_Share_Pct'].rank(ascending=False, method='min').astype(int)
    df['Rank_By_DropOne'] = df['DropOne_DeltaR2'].rank(ascending=False, method='min').astype(int)
    df['Rank_By_Shapley'] = df['Shapley_R2_Share_Pct'].rank(ascending=False, method='min').astype(int)
    return df


def cv_metrics(df_raw, features, n_splits=CV_FOLDS, seed=RANDOM_SEED):
    y = df_raw['OverallScore'].values
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df_raw))
    r2l, spl, mael, rmsel = [], [], [], []
    for tr, te in splits:
        df_tr, df_te = df_raw.iloc[tr], df_raw.iloc[te]
        X_tr, y_tr, fp = build_design_matrix(df_tr, features)
        X_te = transform_like(df_te[features], features, fp)
        y_te = df_te['OverallScore'].values
        lr = LinearRegression().fit(X_tr, y_tr)
        p = lr.predict(X_te)
        r2l.append(r2_score(y_te, p)); spl.append(spearmanr(y_te, p).statistic)
        mael.append(mean_absolute_error(y_te, p))
        rmsel.append(float(np.sqrt(mean_squared_error(y_te, p))))
    return {
        'cv_r2_mean': float(np.mean(r2l)), 'cv_r2_std': float(np.std(r2l)),
        'cv_spearman_mean': float(np.mean(spl)), 'cv_spearman_std': float(np.std(spl)),
        'cv_mae_mean': float(np.mean(mael)), 'cv_rmse_mean': float(np.mean(rmsel)),
    }


def _safe_perturb(values, feature, delta):
    new = values * (1.0 + delta)
    if feature in ('AcceptanceRate', 'EmployedAtGrad', 'Employed3Mo'):
        return np.clip(new, 1e-3, 1 - 1e-3)
    if feature == 'MedianGPA':
        return np.clip(new, 0.0, 4.0)
    if feature == 'GMAT_Combined':
        return np.clip(new, 200.0, 800.0)
    if feature == 'GRE_Final':
        return np.clip(new, 260.0, 340.0)
    if feature == 'ProfessionSalaryRank':
        return np.clip(np.round(new), 1, None)
    return new


def compute_sensitivity(df_raw, features, fp, beta, intercept):
    base_pred = predict_from_raw(df_raw, features, fp, beta, intercept)
    base_rank = pd.Series(rankdata(-base_pred, method='min'), index=df_raw.index)
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


# ====================== Data prep for M9 / M10 ======================

def load_new_2026():
    df = pd.read_excel(DATA_FILE, sheet_name='2026')
    df = df.rename(columns=RENAME_MAP)
    # GMAT_Final_3 has "KNN Imputed Value" strings -> coerce to NaN
    df['GMAT_Final_3'] = pd.to_numeric(df['GMAT_Final_3'], errors='coerce')
    df['GMAT_Final_1'] = pd.to_numeric(df['GMAT_Final_1'], errors='coerce')
    df['GRE_Final'] = pd.to_numeric(df['GRE_Final'], errors='coerce')
    df['Year'] = 2026
    return df


def prepare_m9_frame():
    """M9: M5 + GRE_Final. Features in M5 order plus GRE_Final at the end."""
    df = load_new_2026()
    df['MedianGPA'] = df['MedianGPA'].fillna(df['MedianGPA'].min())
    df['GMAT_Combined'] = df['GMAT_Final_1']
    features = ['PeerScore', 'RecruiterScore', 'MedianGPA',
                'AvgSalaryBonus', 'EmployedAtGrad', 'Employed3Mo',
                'AcceptanceRate', 'GMAT_Combined', 'GRE_Final']
    keep = ['School', 'Year', 'Rank', 'OverallScore'] + features
    df = df[keep].copy()
    # KNN-impute residual missing features (RecruiterScore + others might still have NaN)
    impute_targets = [c for c in features if df[c].isna().any()]
    if impute_targets:
        ref = [c for c in features if c not in impute_targets] or features
        df = knn_impute_missing(df, impute_cols=impute_targets, reference_cols=ref)
    assert df[features].isna().sum().sum() == 0
    return df, features


def prepare_m10_frame():
    """M10: M5 but GMAT_Final_1 -> GMAT_Final_3 (KNN-impute the 15 GRE-only rows
    using non-GRE features as distance basis, so GMAT_Final_3 doesn't become a
    proxy for GRE)."""
    df = load_new_2026()
    df['MedianGPA'] = df['MedianGPA'].fillna(df['MedianGPA'].min())

    # KNN-impute GMAT_Final_3 using only non-GRE features, BEFORE we slice columns.
    # That keeps GMAT_Final_3 epistemically distinct from GRE.
    if df['GMAT_Final_3'].isna().any():
        reference_cols_for_gmat = [
            'PeerScore', 'RecruiterScore', 'MedianGPA',
            'AvgSalaryBonus', 'EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate',
        ]
        # Some of these may themselves be NaN — impute jointly.
        impute_targets = ['GMAT_Final_3'] + [c for c in reference_cols_for_gmat if df[c].isna().any()]
        ref = [c for c in reference_cols_for_gmat if c not in impute_targets] or reference_cols_for_gmat
        df = knn_impute_missing(df, impute_cols=impute_targets, reference_cols=ref)

    df['GMAT_Combined'] = df['GMAT_Final_3']
    features = ['PeerScore', 'RecruiterScore', 'MedianGPA',
                'AvgSalaryBonus', 'EmployedAtGrad', 'Employed3Mo',
                'AcceptanceRate', 'GMAT_Combined']
    keep = ['School', 'Year', 'Rank', 'OverallScore'] + features
    df = df[keep].copy()
    assert df[features].isna().sum().sum() == 0
    return df, features


# ====================== Runner ======================

def run_one(mid, cfg, df_raw, features):
    print(f"\n=== {mid}: {cfg} ===")
    X, y, fp = build_design_matrix(df_raw, features)
    ols = fit_ols_bootstrap(X, y, n_iter=N_BOOTSTRAP, n_jobs=-1)
    enet = fit_enet_bootstrap(X, y, n_iter=N_ENET_BOOTSTRAP, n_jobs=-1)
    coef_report = summarise_ols(ols)
    enet_report = summarise_enet(enet)
    importance = compute_importance(X, y, ols['beta'])
    vif = compute_vif(X)
    train_pred = X.values @ ols['beta'].values + ols['intercept']
    diag = {
        'config': cfg,
        'n_schools': int(len(df_raw)),
        'n_features': len(features),
        'features': features,
        'r2_train': float(r2_score(y, train_pred)),
        'adj_r2_train': float(1 - (1 - r2_score(y, train_pred))
                              * (len(y) - 1) / (len(y) - len(features) - 1)),
        'spearman_train': float(spearmanr(y, train_pred).statistic),
        'mae_train': float(mean_absolute_error(y, train_pred)),
        'rmse_train': float(np.sqrt(mean_squared_error(y, train_pred))),
        'intercept': float(ols['intercept']),
        'condition_number': condition_number(X),
        'max_vif': float(vif['VIF'].max()),
        'bootstrap_iterations': N_BOOTSTRAP,
        **cv_metrics(df_raw, features),
    }
    sens_score, sens_rank, sens_rank_improve = compute_sensitivity(
        df_raw, features, fp, ols['beta'], ols['intercept'])

    out_dir = PER_MODEL_DIR / mid
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_report.to_csv(out_dir / 'coefficients_ols.csv', index=False)
    enet_report.to_csv(out_dir / 'coefficients_enet.csv', index=False)
    importance.to_csv(out_dir / 'importance.csv', index=False)
    vif.to_csv(out_dir / 'vif.csv', index=False)
    sens_score.to_csv(out_dir / 'sensitivity_score.csv')
    sens_rank.to_csv(out_dir / 'sensitivity_rank.csv')
    sens_rank_improve.to_csv(out_dir / 'sensitivity_rank_improve.csv')
    with open(out_dir / 'diagnostics.json', 'w') as f:
        json.dump(diag, f, indent=2, default=float)
    with open(out_dir / 'model_ols.pkl', 'wb') as f:
        pickle.dump({'beta': ols['beta'], 'intercept': ols['intercept'],
                     'fit_params': fp, 'features': features}, f)
    df_raw.to_parquet(out_dir / 'training_frame.parquet')

    print(f"  R²={diag['r2_train']:.4f}  Spearman={diag['spearman_train']:.4f}  "
          f"CV-R²={diag['cv_r2_mean']:.4f}  maxVIF={diag['max_vif']:.2f}")
    print("  Top 3 by |beta|: " + ', '.join(
        coef_report.assign(abs_beta=coef_report['Beta_OLS'].abs())
                   .sort_values('abs_beta', ascending=False).head(3)
                   .apply(lambda r: f"{r['Feature']} ({r['Beta_OLS']:+.2f})", axis=1)))


def main():
    np.random.seed(RANDOM_SEED)
    print("Loading new 2026 dataset...")
    df_m9, feats_m9 = prepare_m9_frame()
    df_m10, feats_m10 = prepare_m10_frame()
    print(f"M9 frame: {df_m9.shape}, features ({len(feats_m9)}): {feats_m9}")
    print(f"M10 frame: {df_m10.shape}, features ({len(feats_m10)}): {feats_m10}")

    cfg_m9 = {'id': 'M9', 'include_profession_rank': False,
              'gmat_col': 'GMAT_Final_1', 'years': [2026], 'extras': ['GRE_Final']}
    cfg_m10 = {'id': 'M10', 'include_profession_rank': False,
               'gmat_col': 'GMAT_Final_3', 'years': [2026], 'extras': []}

    run_one('M9', cfg_m9, df_m9, feats_m9)
    run_one('M10', cfg_m10, df_m10, feats_m10)
    print("\nDone.")


if __name__ == '__main__':
    main()
