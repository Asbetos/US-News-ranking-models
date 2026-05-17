"""Regenerate outputs/gwu_sensitivity_per_model.xlsx with per-year rank
for both-years configs (matches the dashboard fix).

Loads saved model artifacts directly — no notebook re-execution needed.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

NEW_MODELS_DIR = Path(__file__).parent
PER_MODEL_DIR = NEW_MODELS_DIR / "outputs" / "per_model"
OUT_CANONICAL = NEW_MODELS_DIR / "outputs" / "gwu_sensitivity_per_model.xlsx"
OUT_FALLBACK = NEW_MODELS_DIR / "outputs" / "gwu_sensitivity_per_model_v2.xlsx"

# If Excel has the canonical file open (lock file present), write to the fallback.
LOCK = NEW_MODELS_DIR / "outputs" / "~$gwu_sensitivity_per_model.xlsx"
OUT = OUT_FALLBACK if LOCK.exists() else OUT_CANONICAL

LOG_VARS = ['AvgSalaryBonus', 'GMAT_Combined']
LOGIT_VARS = ['EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate']
INV_NORM_VARS = ['ProfessionSalaryRank']
PERTURBATIONS = [-0.10, -0.05, -0.01, 0.01, 0.05, 0.10]

MODEL_CONFIGS = {
    'M1': {'include_profession_rank': True,  'gmat_col': 'GMAT_Final_1', 'years': [2026]},
    'M2': {'include_profession_rank': True,  'gmat_col': 'GMAT_Final_1', 'years': [2025, 2026]},
    'M3': {'include_profession_rank': True,  'gmat_col': 'GMAT_Final_2', 'years': [2026]},
    'M4': {'include_profession_rank': True,  'gmat_col': 'GMAT_Final_2', 'years': [2025, 2026]},
    'M5': {'include_profession_rank': False, 'gmat_col': 'GMAT_Final_1', 'years': [2026]},
    'M6': {'include_profession_rank': False, 'gmat_col': 'GMAT_Final_1', 'years': [2025, 2026]},
    'M7': {'include_profession_rank': False, 'gmat_col': 'GMAT_Final_2', 'years': [2026]},
    'M8': {'include_profession_rank': False, 'gmat_col': 'GMAT_Final_2', 'years': [2025, 2026]},
}


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


def safe_perturb(values, feature, delta):
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


def per_year_rank(preds, df, idx):
    """Rank within the same year as df.at[idx, 'Year']. Falls back to global rank
    when the frame has only one year."""
    if 'Year' in df.columns and df['Year'].nunique() > 1:
        target_year = df.at[idx, 'Year']
        mask = (df['Year'] == target_year).values
    else:
        mask = np.ones(len(df), dtype=bool)
    sub = preds[mask]
    masked_index = df.index[mask].tolist()
    pos = masked_index.index(idx)
    return int(rankdata(-sub, method='min')[pos]), int(mask.sum())


def compute_gwu_analysis(mid, school='George Washington University'):
    d = PER_MODEL_DIR / mid
    df_raw = pd.read_parquet(d / 'training_frame.parquet')
    with open(d / 'model_ols.pkl', 'rb') as f:
        m = pickle.load(f)
    features, beta, intercept, fp = m['features'], m['beta'], m['intercept'], m['fit_params']
    coef_report = pd.read_csv(d / 'coefficients_ols.csv').set_index('Feature')
    importance = pd.read_csv(d / 'importance.csv').set_index('Feature')

    mask = df_raw['School'] == school
    if not mask.any():
        return None
    sub = df_raw[mask]
    if 'Year' in df_raw.columns and sub['Year'].nunique() > 1:
        idx = sub[sub['Year'] == sub['Year'].max()].index[0]
    else:
        idx = sub.index[0]

    gwu_row = df_raw.loc[[idx]]

    # Section 1: global model
    global_table = coef_report.copy()
    global_table['Pct_Contribution'] = importance['Beta_Share_Pct'].round(2)
    global_table = global_table.reset_index()

    # Section 2: GWU feature contributions
    gwu_X = transform_like(gwu_row[features], features, fp)
    gwu_std = gwu_X[features].values[0]
    contributions = beta.reindex(features).values * gwu_std
    total_abs = float(np.abs(contributions).sum()) or 1.0
    gwu_contribution = pd.DataFrame({
        'Feature': features,
        'GWU_Raw_Value': [float(gwu_row[f].values[0]) for f in features],
        'GWU_Standardized': np.round(gwu_std, 3),
        'Beta_OLS': [round(float(beta[f]), 3) for f in features],
        'GWU_Contribution': np.round(contributions, 3),
        'Pct_Contribution_GWU': np.round(np.abs(contributions) / total_abs * 100, 2),
        'Is_Significant': [bool(coef_report.loc[f, 'Is_Significant']) for f in features],
        'Direction': ['+' if v >= 0 else '-' for v in contributions],
    })

    # Section 3: GWU score sensitivity
    base_score = float(predict_from_raw(gwu_row, features, fp, beta, intercept)[0])
    cols = [f'{int(d*100):+d}%' for d in PERTURBATIONS]
    score_rows = []
    for f in features:
        row = {'Feature': f}
        for d, c in zip(PERTURBATIONS, cols):
            pert = gwu_row.copy()
            pert[f] = safe_perturb(pert[f].values, f, d)
            new_score = float(predict_from_raw(pert, features, fp, beta, intercept)[0])
            row[c] = round(new_score - base_score, 4)
        score_rows.append(row)
    score_sens = pd.DataFrame(score_rows)

    # Section 4: GWU rank sensitivity — within-year for both-years configs
    df_raw_float = df_raw.copy()
    for f in features:
        df_raw_float[f] = df_raw_float[f].astype(float)
    base_all_preds = predict_from_raw(df_raw_float, features, fp, beta, intercept)
    base_rank, pool_size = per_year_rank(base_all_preds, df_raw_float, idx)

    rank_rows = []
    for f in features:
        row = {'Feature': f}
        for d, c in zip(PERTURBATIONS, cols):
            df_p = df_raw_float.copy()
            current = np.array([df_raw_float.at[idx, f]], dtype=float)
            df_p.at[idx, f] = float(safe_perturb(current, f, d)[0])
            new_preds = predict_from_raw(df_p, features, fp, beta, intercept)
            new_rank, _ = per_year_rank(new_preds, df_p, idx)
            row[c] = new_rank - base_rank
        rank_rows.append(row)
    rank_sens = pd.DataFrame(rank_rows)

    return {
        'global': global_table,
        'gwu_contribution': gwu_contribution,
        'gwu_score_sensitivity': score_sens,
        'gwu_rank_sensitivity': rank_sens,
        'gwu_metadata': {
            'school': school,
            'idx': int(idx),
            'year_used': int(df_raw.at[idx, 'Year']) if 'Year' in df_raw.columns else None,
            'actual_rank': int(df_raw.at[idx, 'Rank']),
            'actual_score': float(df_raw.at[idx, 'OverallScore']),
            'predicted_rank': base_rank,
            'predicted_score': round(base_score, 3),
            'rank_pool_size': pool_size,
        }
    }


def main():
    with pd.ExcelWriter(OUT, engine='openpyxl') as writer:
        for mid in sorted(MODEL_CONFIGS.keys()):
            a = compute_gwu_analysis(mid)
            if a is None:
                pd.DataFrame([{'Note': 'School not found'}]).to_excel(writer, sheet_name=mid, index=False)
                continue
            cfg = MODEL_CONFIGS[mid]
            md = a['gwu_metadata']
            header_df = pd.DataFrame([
                {'Item': 'Model ID', 'Value': mid},
                {'Item': 'ProfessionSalaryRank included', 'Value': cfg['include_profession_rank']},
                {'Item': 'GMAT feature', 'Value': cfg['gmat_col']},
                {'Item': 'Years used', 'Value': '+'.join(str(y) for y in cfg['years'])},
                {'Item': 'School', 'Value': md['school']},
                {'Item': 'GWU row year', 'Value': md['year_used']},
                {'Item': 'GWU actual rank', 'Value': md['actual_rank']},
                {'Item': 'GWU predicted rank',
                 'Value': f"{md['predicted_rank']} of {md['rank_pool_size']}"},
                {'Item': 'GWU actual score', 'Value': md['actual_score']},
                {'Item': 'GWU predicted score', 'Value': md['predicted_score']},
            ])
            startrow = 0
            header_df.to_excel(writer, sheet_name=mid, index=False, startrow=startrow)
            startrow += len(header_df) + 2
            sections = [
                ('1. Global model: coefficient, significance, direction, % contribution',
                 a['global']),
                ('2. GWU feature contributions (beta x standardized GWU value)',
                 a['gwu_contribution']),
                ('3. GWU score sensitivity (delta predicted score for GWU)',
                 a['gwu_score_sensitivity']),
                ('4. GWU rank sensitivity (delta rank for GWU; negative = moved up; '
                 'rank computed within GWU\'s year)',
                 a['gwu_rank_sensitivity']),
            ]
            for label, table in sections:
                pd.DataFrame([{'Section': label}]).to_excel(
                    writer, sheet_name=mid, index=False, header=False, startrow=startrow)
                startrow += 1
                table.to_excel(writer, sheet_name=mid, index=False, startrow=startrow)
                startrow += len(table) + 3
            print(f"  {mid}: actual rank {md['actual_rank']} | "
                  f"predicted {md['predicted_rank']} of {md['rank_pool_size']}")
    print(f"Wrote {OUT}")


if __name__ == '__main__':
    main()
