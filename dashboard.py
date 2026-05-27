"""
US News 8-Model Interactive Dashboard

Run with:
    streamlit run dashboard.py

Tabs:
  - Overview:     model config, predictive metrics, causal-quality scorecard.
  - Sensitivity:  live "what-if" sliders per feature for any school; sees
                  the school's rank/score update in real time.
  - EDA:          feature distributions, correlations, scatter vs OverallScore
                  for the selected model's training frame.
  - Diagnostics:  predicted-vs-actual, residuals, QQ plot, coefficient CIs,
                  importance decompositions (|beta|-share / drop-1 / Shapley).
  - Cross-Model:  beta heatmap across 8 models, driver verdicts, composite scorecard.

Loads pre-computed artifacts from outputs/per_model/. No re-fitting.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, rankdata, spearmanr

# =================== Paths & constants ===================
NEW_MODELS_DIR = Path(__file__).parent
PER_MODEL_DIR = NEW_MODELS_DIR / "outputs" / "per_model"

LOG_VARS = ['AvgSalaryBonus', 'GMAT_Combined', 'GRE_Final']
LOGIT_VARS = ['EmployedAtGrad', 'Employed3Mo', 'AcceptanceRate']
INV_NORM_VARS = ['ProfessionSalaryRank']
REVERSE_DIRECTION = {'AcceptanceRate', 'ProfessionSalaryRank'}

EXPECTED_SIGN_TRANSFORMED = {
    'PeerScore': '+', 'RecruiterScore': '+', 'MedianGPA': '+',
    'AvgSalaryBonus': '+', 'GMAT_Combined': '+', 'GRE_Final': '+',
    'EmployedAtGrad': '+', 'Employed3Mo': '+',
    'AcceptanceRate': '-',
    'ProfessionSalaryRank': '+',
}

MODEL_CONFIGS = {
    'M1':  {'include_profession_rank': True,  'gmat_col': 'GMAT_Final_1', 'years': [2026],       'extras': []},
    'M2':  {'include_profession_rank': True,  'gmat_col': 'GMAT_Final_1', 'years': [2025, 2026], 'extras': []},
    'M3':  {'include_profession_rank': True,  'gmat_col': 'GMAT_Final_2', 'years': [2026],       'extras': []},
    'M4':  {'include_profession_rank': True,  'gmat_col': 'GMAT_Final_2', 'years': [2025, 2026], 'extras': []},
    'M5':  {'include_profession_rank': False, 'gmat_col': 'GMAT_Final_1', 'years': [2026],       'extras': []},
    'M6':  {'include_profession_rank': False, 'gmat_col': 'GMAT_Final_1', 'years': [2025, 2026], 'extras': []},
    'M7':  {'include_profession_rank': False, 'gmat_col': 'GMAT_Final_2', 'years': [2026],       'extras': []},
    'M8':  {'include_profession_rank': False, 'gmat_col': 'GMAT_Final_2', 'years': [2025, 2026], 'extras': []},
    'M9':  {'include_profession_rank': False, 'gmat_col': 'GMAT_Final_1', 'years': [2026],       'extras': ['GRE_Final']},
    'M10': {'include_profession_rank': False, 'gmat_col': 'GMAT_Final_3', 'years': [2026],       'extras': []},
    'M11': {'include_profession_rank': False, 'gmat_col': 'GMAT_Final_1', 'years': [2026],       'extras': [],
            'kind': 'methodology_bounded',
            'label': 'M11 (methodology-bounded)'},
}


def _model_label(mid: str) -> str:
    """Return the user-facing label for a model id (falls back to the id)."""
    return MODEL_CONFIGS[mid].get('label', mid)

st.set_page_config(page_title="US News 8-Model Dashboard", layout="wide")


# =================== Feature transforms (mirror notebook) ===================
def transform_features(df: pd.DataFrame, features: List[str],
                       rank_n: Dict[str, float] | None = None
                       ) -> Tuple[pd.DataFrame, Dict[str, float]]:
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


def transform_like(df_raw: pd.DataFrame, features: List[str],
                   fit_params: Dict[str, Any]) -> pd.DataFrame:
    out = df_raw[features].copy().astype(float)
    for c, (lo, hi) in fit_params['caps'].items():
        if c in out.columns:
            out[c] = out[c].clip(lo, hi)
    out, _ = transform_features(out, features, rank_n=fit_params['rank_counts'])
    for c in features:
        out[c] = (out[c] - fit_params['scaler_mean'][c]) / fit_params['scaler_scale'][c]
    return out


def predict_from_raw(df_raw: pd.DataFrame, features: List[str],
                     fit_params: Dict[str, Any],
                     beta: pd.Series, intercept: float) -> np.ndarray:
    X = transform_like(df_raw, features, fit_params)
    return X.values @ beta.reindex(features).values + intercept


# =================== Data loader (cached) ===================
@st.cache_data(show_spinner="Loading model artifacts...")
def load_all_models() -> Dict[str, Dict[str, Any]]:
    models: Dict[str, Dict[str, Any]] = {}
    for mid in MODEL_CONFIGS.keys():
        d = PER_MODEL_DIR / mid
        m: Dict[str, Any] = {}
        m['coef'] = pd.read_csv(d / 'coefficients_ols.csv')
        m['enet'] = pd.read_csv(d / 'coefficients_enet.csv')
        m['importance'] = pd.read_csv(d / 'importance.csv')
        m['vif'] = pd.read_csv(d / 'vif.csv')
        m['sens_score'] = pd.read_csv(d / 'sensitivity_score.csv', index_col=0)
        m['sens_rank'] = pd.read_csv(d / 'sensitivity_rank.csv', index_col=0)
        m['sens_rank_improve'] = pd.read_csv(d / 'sensitivity_rank_improve.csv', index_col=0)
        with open(d / 'diagnostics.json') as fp:
            m['diagnostics'] = json.load(fp)
        with open(d / 'model_ols.pkl', 'rb') as fp:
            m['model_state'] = pickle.load(fp)
        m['training_frame'] = pd.read_parquet(d / 'training_frame.parquet')
        models[mid] = m
    return models


@st.cache_data(show_spinner=False)
def compute_quality_score_cached(mid: str, _coef_df: pd.DataFrame,
                                 _imp_df: pd.DataFrame, max_vif: float,
                                 sign_stab_mean: float) -> Dict[str, float]:
    coef = _coef_df.set_index('Feature')
    cv = (coef['Std_Bootstrap'] / coef['Beta_OLS'].abs().replace(0, np.nan)).fillna(2.0)
    coef_stability = float(np.clip(1 - cv.mean(), 0.0, 1.0))
    sign_coherence = float(coef['Sign_Matches'].mean())
    vif_clean = float(1 - min(1.0, max_vif / 10.0))
    imp = _imp_df.set_index('Feature')[
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
    bootstrap_sign_stability = float(sign_stab_mean / 100.0)
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


def get_quality(mid: str, models: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    m = models[mid]
    return compute_quality_score_cached(
        mid, m['coef'], m['importance'],
        max_vif=float(m['diagnostics']['max_vif']),
        sign_stab_mean=float(m['coef']['Sign_Stability_Pct'].mean()),
    )


# =================== Sidebar controls ===================
models = load_all_models()
school_names = sorted(models['M5']['training_frame']['School'].unique().tolist())

with st.sidebar:
    st.markdown("## US News 8-Model Dashboard")
    st.caption("All model artifacts are pre-computed in `outputs/per_model/`. "
               "Switching models is instant; sensitivity sliders recompute predictions live.")
    selected_model = st.selectbox(
        "Model",
        options=list(MODEL_CONFIGS.keys()),
        index=4,  # M5 default
        format_func=_model_label,
        help="M5 has the highest causal-quality composite among the original "
             "8 OLS models. M9 = M5 + GRE_Final. M10 = M5 with GMAT_Final_3 "
             "(KNN-imputed GMAT for GRE-only schools). "
             "M11 = M5 with coefficient contributions bounded to "
             "+/-5pp of US News methodology weights.",
    )
    cfg = MODEL_CONFIGS[selected_model]
    st.markdown(
        f"**Config**\n\n"
        f"- ProfessionSalaryRank: `{'in' if cfg['include_profession_rank'] else 'out'}`\n"
        f"- GMAT: `{cfg['gmat_col']}`\n"
        f"- Years: `{'+'.join(str(y) for y in cfg['years'])}`"
    )
    st.divider()
    default_idx = school_names.index('George Washington University') \
        if 'George Washington University' in school_names else 0
    selected_school = st.selectbox(
        "Focus school (for Sensitivity + Diagnostics)",
        options=school_names,
        index=default_idx,
    )

# Convenient handles
M = models[selected_model]
df_raw: pd.DataFrame = M['training_frame']
features: List[str] = M['model_state']['features']
beta: pd.Series = M['model_state']['beta']
intercept: float = M['model_state']['intercept']
fit_params: Dict[str, Any] = M['model_state']['fit_params']
coef_df: pd.DataFrame = M['coef']
imp_df: pd.DataFrame = M['importance']
diag: Dict[str, Any] = M['diagnostics']
quality = get_quality(selected_model, models)


def get_school_row(school: str) -> Tuple[pd.DataFrame, int]:
    sub = df_raw[df_raw['School'] == school]
    if sub.empty:
        return df_raw.iloc[[0]], df_raw.index[0]
    if 'Year' in df_raw.columns and sub['Year'].nunique() > 1:
        idx = sub[sub['Year'] == sub['Year'].max()].index[0]
    else:
        idx = sub.index[0]
    return df_raw.loc[[idx]], idx


def school_rank_within_year(preds: np.ndarray, df: pd.DataFrame, idx
                            ) -> Dict[str, Any]:
    """Per-year predicted rank for the row at `idx`.

    For both-years configs, the published US News rank is per-year (1..N for each
    year). Ranking across the stacked 2025+2026 frame would double the rank
    numbers. This helper restricts the rank pool to the focus school's own year,
    so the predicted rank is directly comparable to the published rank.
    """
    if 'Year' in df.columns and df['Year'].nunique() > 1:
        target_year = df.at[idx, 'Year']
        mask = (df['Year'] == target_year).values
    else:
        mask = np.ones(len(df), dtype=bool)

    sub_preds = preds[mask]
    masked_index_list = df.index[mask].tolist()
    pos_in_subset = masked_index_list.index(idx)

    sub_ranks = rankdata(-sub_preds, method='min')
    school_rank = int(sub_ranks[pos_in_subset])
    school_score = float(sub_preds[pos_in_subset])

    sorted_desc = np.sort(sub_preds)[::-1]
    n = len(sorted_desc)
    score_above = float(sorted_desc[school_rank - 2]) if school_rank >= 2 else None
    score_below = float(sorted_desc[school_rank]) if school_rank < n else None

    return {
        'rank': school_rank,
        'score': school_score,
        'gap_to_above': (score_above - school_score) if score_above is not None else None,
        'gap_to_below': (school_score - score_below) if score_below is not None else None,
        'n_in_pool': n,
        'year': int(df.at[idx, 'Year']) if 'Year' in df.columns else None,
    }


# =================== Top header (always visible) ===================
st.title("US News 8-Model Causal Comparison — Dashboard")
top_cols = st.columns([1, 1, 1, 1, 1, 1])
top_cols[0].metric("Model", selected_model,
                   help="Selected from sidebar")
top_cols[1].metric("R² (train)", f"{diag['r2_train']:.3f}")
top_cols[2].metric("CV-R²", f"{diag['cv_r2_mean']:.3f}",
                   delta=f"±{diag['cv_r2_std']:.3f}", delta_color="off")
top_cols[3].metric("Spearman ρ", f"{diag['spearman_train']:.3f}")
top_cols[4].metric("Max VIF", f"{diag['max_vif']:.2f}",
                   delta="comfortable" if diag['max_vif'] <= 10 else "elevated",
                   delta_color="normal" if diag['max_vif'] <= 10 else "inverse")
top_cols[5].metric("Composite causal score", f"{quality['composite_score']:.3f}",
                   help="0 to 1, higher = more trustworthy causal interpretation")

# Tabs
tab_overview, tab_sens, tab_eda, tab_diag, tab_cross = st.tabs([
    "Overview", "Sensitivity Explorer", "EDA", "Diagnostics", "Cross-Model"
])


# =================== Tab: Overview ===================
with tab_overview:
    left, right = st.columns([2, 3])

    with left:
        st.subheader("Causal-quality scorecard")
        scorecard_df = pd.DataFrame({
            'Component': [
                'coef_stability', 'sign_coherence', 'vif_clean',
                'importance_agreement', 'bootstrap_sign_stability', 'composite_score'],
            'Value': [
                quality['coef_stability'], quality['sign_coherence'], quality['vif_clean'],
                quality['importance_agreement'], quality['bootstrap_sign_stability'],
                quality['composite_score']],
        })
        bar = go.Figure(data=[go.Bar(
            x=scorecard_df['Value'],
            y=scorecard_df['Component'],
            orientation='h',
            marker_color=['#2563eb' if c != 'composite_score' else '#16a34a'
                          for c in scorecard_df['Component']],
            text=[f"{v:.3f}" for v in scorecard_df['Value']],
            textposition='outside',
        )])
        bar.update_layout(
            xaxis=dict(range=[0, 1.05]), height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange='reversed'),
        )
        st.plotly_chart(bar, use_container_width=True)

        st.subheader("Predictive metrics")
        st.dataframe(pd.DataFrame({
            'Metric': ['R² (train)', 'Adj R² (train)', 'Spearman ρ (train)',
                       'CV R² (mean)', 'CV Spearman (mean)', 'MAE (train)',
                       'RMSE (train)', 'Condition number', 'Max VIF',
                       'Intercept', '#Schools', 'Bootstrap iterations'],
            'Value': [
                f"{diag['r2_train']:.4f}",
                f"{diag['adj_r2_train']:.4f}",
                f"{diag['spearman_train']:.4f}",
                f"{diag['cv_r2_mean']:.4f} ± {diag['cv_r2_std']:.4f}",
                f"{diag['cv_spearman_mean']:.4f} ± {diag['cv_spearman_std']:.4f}",
                f"{diag['mae_train']:.3f}",
                f"{diag['rmse_train']:.3f}",
                f"{diag['condition_number']:.2f}",
                f"{diag['max_vif']:.2f}",
                f"{diag['intercept']:.4f}",
                f"{diag['n_schools']}",
                f"{diag['bootstrap_iterations']}",
            ],
        }), hide_index=True, use_container_width=True)

    with right:
        st.subheader("Coefficients (OLS, bootstrap CIs)")
        ord_df = coef_df.copy()
        ord_df['abs_beta'] = ord_df['Beta_OLS'].abs()
        ord_df = ord_df.sort_values('abs_beta', ascending=True)
        coef_fig = go.Figure()
        coef_fig.add_trace(go.Scatter(
            x=ord_df['Beta_OLS'],
            y=ord_df['Feature'],
            mode='markers',
            error_x=dict(
                type='data', symmetric=False,
                array=ord_df['Upper_95_CI'] - ord_df['Beta_OLS'],
                arrayminus=ord_df['Beta_OLS'] - ord_df['Lower_95_CI'],
                thickness=2,
            ),
            marker=dict(
                size=12,
                color=['#16a34a' if s else '#9ca3af' for s in ord_df['Is_Significant']],
                line=dict(color='black', width=1),
            ),
            hovertemplate=(
                "<b>%{y}</b><br>β = %{x:+.3f}<br>"
                "95% CI: [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]<br>"
                "Sign-stability: %{customdata[2]:.1f}%<extra></extra>"
            ),
            customdata=ord_df[['Lower_95_CI', 'Upper_95_CI', 'Sign_Stability_Pct']].values,
        ))
        coef_fig.add_vline(x=0, line=dict(color='red', dash='dash'))
        coef_fig.update_layout(
            height=420,
            xaxis_title="Standardised β (95% bootstrap CI)",
            yaxis_title="",
            margin=dict(l=10, r=10, t=10, b=40),
            showlegend=False,
        )
        st.plotly_chart(coef_fig, use_container_width=True)
        st.caption("Green markers = 95% CI excludes zero (significant). "
                   "Grey = not significant.")

    # Importance decomposition
    st.subheader("Feature importance — three independent measures")
    imp = imp_df.set_index('Feature')[
        ['Beta_Share_Pct', 'Shapley_R2_Share_Pct']].copy()
    imp['DropOne_DeltaR2_x100'] = imp_df.set_index('Feature')['DropOne_DeltaR2'] * 100
    imp_long = imp.reset_index().melt(
        id_vars='Feature', var_name='Measure', value_name='Value')
    fig_imp = px.bar(
        imp_long, x='Feature', y='Value', color='Measure', barmode='group',
        labels={'Value': 'Importance (%)', 'Feature': ''},
    )
    fig_imp.update_layout(height=420, legend_title="")
    st.plotly_chart(fig_imp, use_container_width=True)
    st.caption(
        "**Beta_Share_Pct**: 100·|β|/Σ|β| — fast but biased by multicollinearity. "
        "**Shapley_R2_Share_Pct**: collinearity-invariant decomposition of R². "
        "**DropOne_DeltaR2 (×100)**: R² drop when this feature is removed (×100 to share axis). "
        "When the three disagree, multicollinearity is the usual culprit."
    )

    if 'year_stability' in diag and diag['year_stability']:
        st.subheader("Year stability (both-years configs only)")
        ys = diag['year_stability']
        st.caption(
            f"Max |Δβ| between years: **{ys['max_abs_delta']:.3f}** on `{ys['max_delta_feature']}` — "
            f"L2 norm of all Δβ: {ys['delta_beta_l2']:.3f}. "
            f"Larger values indicate the implicit US News weights shifted between years."
        )
        py = pd.DataFrame(ys['per_year_betas']).T
        py.index.name = 'Year'
        fig_y = go.Figure()
        for f in py.columns:
            fig_y.add_trace(go.Scatter(
                x=py.index.astype(int), y=py[f], mode='lines+markers', name=f))
        fig_y.update_layout(
            height=380, xaxis_title='Year',
            yaxis_title='Standardised β', legend_title='Feature',
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_y, use_container_width=True)


# =================== Tab: Sensitivity Explorer ===================
with tab_sens:
    st.subheader(f"Live what-if for {selected_school}")
    school_row, school_idx = get_school_row(selected_school)
    base_all_preds = predict_from_raw(df_raw, features, fit_params, beta, intercept)
    base_diag = school_rank_within_year(base_all_preds, df_raw, school_idx)
    base_school_rank = base_diag['rank']
    base_school_score = base_diag['score']
    base_school_actual_rank = int(df_raw.at[school_idx, 'Rank'])
    base_school_actual_score = float(df_raw.at[school_idx, 'OverallScore'])

    if base_diag['year'] is not None and df_raw['Year'].nunique() > 1:
        rank_pool_note = (f"Predicted rank computed within the **{base_diag['year']}** cohort "
                          f"({base_diag['n_in_pool']} schools) so it is directly comparable "
                          f"to the published rank.")
    else:
        rank_pool_note = (f"Predicted rank computed against all "
                          f"{base_diag['n_in_pool']} schools in this model's training frame.")

    info_cols = st.columns(4)
    info_cols[0].metric("Actual rank", base_school_actual_rank)
    info_cols[1].metric(
        "Predicted rank",
        f"{base_school_rank} of {base_diag['n_in_pool']}",
        delta=f"{base_school_rank - base_school_actual_rank:+d} vs actual",
        delta_color="inverse",
    )
    info_cols[2].metric("Actual score", f"{base_school_actual_score:.1f}")
    info_cols[3].metric("Predicted score", f"{base_school_score:.2f}")
    st.caption(rank_pool_note)

    st.markdown(
        "**Move the sliders below to perturb the school's features.** "
        "The school's new predicted score and rank update instantly. "
        "Sliders express **percent change** from the school's current raw value. "
        "Bounded features (rates, GMAT, GPA) are clipped to legal domains."
    )

    # Feature-clipping ranges (match _safe_perturb in the notebook)
    def safe_clip(value: float, feature: str) -> float:
        if feature in ('AcceptanceRate', 'EmployedAtGrad', 'Employed3Mo'):
            return float(np.clip(value, 1e-3, 1 - 1e-3))
        if feature == 'MedianGPA':
            return float(np.clip(value, 0.0, 4.0))
        if feature == 'GMAT_Combined':
            return float(np.clip(value, 200.0, 800.0))
        if feature == 'GRE_Final':
            return float(np.clip(value, 260.0, 340.0))
        if feature == 'ProfessionSalaryRank':
            return float(max(1.0, round(value)))
        return float(value)

    # Sliders
    slider_cols = st.columns(3)
    perturbations: Dict[str, float] = {}
    for i, feat in enumerate(features):
        with slider_cols[i % 3]:
            current = float(school_row[feat].iloc[0])
            arrow = " (↓ better)" if feat in REVERSE_DIRECTION else " (↑ better)"
            pct = st.slider(
                f"{feat}{arrow}",
                min_value=-50, max_value=50, value=0, step=1,
                # Key scoped to model+school so each (model, school) gets its
                # own persistent slider state and switching back restores it.
                key=f"sens_{selected_model}_{selected_school}_{feat}_v2",
                help=f"Current raw value: {current:g}",
            )
            new_val = safe_clip(current * (1 + pct / 100.0), feat)
            perturbations[feat] = new_val
            st.caption(f"Current: `{current:g}` → New: `{new_val:g}`")

    # Apply all perturbations at once
    df_pert = df_raw.copy()
    for f in features:
        df_pert[f] = df_pert[f].astype(float)
    for f, v in perturbations.items():
        df_pert.at[school_idx, f] = v

    new_all_preds = predict_from_raw(df_pert, features, fit_params, beta, intercept)
    new_diag = school_rank_within_year(new_all_preds, df_pert, school_idx)
    new_school_rank = new_diag['rank']
    new_school_score = new_diag['score']

    delta_score = new_school_score - base_school_score
    delta_rank = new_school_rank - base_school_rank
    rank_arrow = "↑" if delta_rank < 0 else ("↓" if delta_rank > 0 else "—")

    res_cols = st.columns(4)
    res_cols[0].metric(
        "New predicted score", f"{new_school_score:.2f}",
        delta=f"{delta_score:+.2f}")
    res_cols[1].metric(
        "New predicted rank", f"{new_school_rank} of {new_diag['n_in_pool']}",
        delta=f"{-delta_rank:+d} ({rank_arrow})",
        delta_color="normal" if delta_rank <= 0 else "inverse")
    res_cols[2].metric(
        "Gap to next rank up",
        f"+{new_diag['gap_to_above']:.3f}" if new_diag['gap_to_above'] is not None else "—",
        help="Score points needed to advance one more rank.")
    res_cols[3].metric(
        "Gap to next rank down",
        f"−{new_diag['gap_to_below']:.3f}" if new_diag['gap_to_below'] is not None else "—",
        help="Score points a competitor would need to overtake.")

    st.divider()
    st.subheader("Single-feature sweep (one feature at a time, ±50% range)")
    sweep_feat = st.selectbox(
        "Feature", options=features, index=0,
        key=f"sweep_feat_{selected_model}")
    sweep_deltas = np.arange(-50, 51, 2)
    sweep_scores, sweep_ranks = [], []
    base_val = float(school_row[sweep_feat].iloc[0])
    for d in sweep_deltas:
        new_val = safe_clip(base_val * (1 + d / 100.0), sweep_feat)
        df_s = df_raw.copy()
        df_s[sweep_feat] = df_s[sweep_feat].astype(float)
        df_s.at[school_idx, sweep_feat] = new_val
        ps = predict_from_raw(df_s, features, fit_params, beta, intercept)
        sweep_diag = school_rank_within_year(ps, df_s, school_idx)
        sweep_scores.append(sweep_diag['score'])
        sweep_ranks.append(sweep_diag['rank'])

    fig_sweep = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sweep.add_trace(go.Scatter(
        x=sweep_deltas, y=sweep_scores, mode='lines+markers',
        name='Predicted score', line=dict(color='#2563eb')),
        secondary_y=False)
    fig_sweep.add_trace(go.Scatter(
        x=sweep_deltas, y=sweep_ranks, mode='lines+markers',
        name='Predicted rank', line=dict(color='#ea580c')),
        secondary_y=True)
    fig_sweep.add_vline(x=0, line=dict(color='gray', dash='dash'))
    fig_sweep.update_xaxes(title=f"Perturbation of {sweep_feat} (%)")
    fig_sweep.update_yaxes(title_text="Predicted score", secondary_y=False)
    fig_sweep.update_yaxes(
        title_text="Predicted rank (lower = better)",
        secondary_y=True, autorange='reversed')
    fig_sweep.update_layout(
        height=380, hovermode='x unified',
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation='h', y=1.1),
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    # Per-feature contribution waterfall for the focus school (baseline values)
    st.subheader(f"Score-component breakdown for {selected_school}")
    school_X = transform_like(school_row, features, fit_params)
    components = (beta.reindex(features).values * school_X[features].values[0])
    waterfall_df = pd.DataFrame({
        'Feature': list(features) + ['Intercept'],
        'Contribution': np.concatenate([components, [intercept]]),
    })
    waterfall_df['abs'] = waterfall_df['Contribution'].abs()
    waterfall_df = waterfall_df.sort_values('abs', ascending=True)
    fig_wf = go.Figure(data=[go.Bar(
        x=waterfall_df['Contribution'],
        y=waterfall_df['Feature'],
        orientation='h',
        marker_color=['#16a34a' if v >= 0 else '#dc2626'
                      for v in waterfall_df['Contribution']],
        text=[f"{v:+.2f}" for v in waterfall_df['Contribution']],
        textposition='outside',
    )])
    fig_wf.update_layout(
        height=420, xaxis_title="Contribution to predicted score",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_wf, use_container_width=True)
    st.caption("Per-feature contribution = β × (standardised feature value). "
               "Sum + intercept = predicted score.")


# =================== Tab: EDA ===================
with tab_eda:
    st.subheader("Feature distributions for this model's training frame")
    n_features = len(features)
    cols = 3
    rows = (n_features + cols - 1) // cols
    fig_d = make_subplots(rows=rows, cols=cols, subplot_titles=features)
    for i, f in enumerate(features):
        r, c = i // cols + 1, i % cols + 1
        fig_d.add_trace(
            go.Histogram(x=df_raw[f], nbinsx=25, name=f,
                         marker_color='#3b82f6', showlegend=False),
            row=r, col=c)
        # Overlay outlier-capper bounds
        lo, hi = fit_params['caps'][f]
        fig_d.add_vline(x=lo, line=dict(color='red', dash='dot'), row=r, col=c)
        fig_d.add_vline(x=hi, line=dict(color='red', dash='dot'), row=r, col=c)
    fig_d.update_layout(height=260 * rows, margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=False)
    st.plotly_chart(fig_d, use_container_width=True)
    st.caption("Red dotted lines show the 5/95-percentile outlier cap learned by `OutlierCapper`. "
               "Values are clipped to that range before transforms.")

    # Correlation heatmap
    st.subheader("Spearman correlation — features + target")
    corr_cols = features + ['OverallScore']
    corr = df_raw[corr_cols].corr(method='spearman')
    fig_h = px.imshow(
        corr, text_auto='.2f', aspect='auto',
        color_continuous_scale='RdBu', zmin=-1, zmax=1, origin='lower',
    )
    fig_h.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_h, use_container_width=True)
    st.caption("Spearman is rank-correlation; immune to the non-linear transforms applied "
               "later in the pipeline. Strong off-diagonal correlations are a multicollinearity flag "
               "(see Diagnostics tab for VIF).")

    # Scatter vs OverallScore
    st.subheader("Each feature vs OverallScore")
    scatter_cols = 3
    scatter_rows = (n_features + scatter_cols - 1) // scatter_cols
    fig_s = make_subplots(rows=scatter_rows, cols=scatter_cols, subplot_titles=features)
    show_year = 'Year' in df_raw.columns and df_raw['Year'].nunique() > 1
    for i, f in enumerate(features):
        r, c = i // scatter_cols + 1, i % scatter_cols + 1
        if show_year:
            for y in sorted(df_raw['Year'].unique()):
                sub = df_raw[df_raw['Year'] == y]
                fig_s.add_trace(
                    go.Scatter(x=sub[f], y=sub['OverallScore'], mode='markers',
                               name=f'{y}', marker=dict(size=5),
                               showlegend=(i == 0)),
                    row=r, col=c)
        else:
            fig_s.add_trace(
                go.Scatter(x=df_raw[f], y=df_raw['OverallScore'],
                           mode='markers', marker=dict(size=5, color='#3b82f6'),
                           showlegend=False),
                row=r, col=c)
    fig_s.update_layout(height=260 * scatter_rows, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_s, use_container_width=True)


# =================== Tab: Diagnostics ===================
with tab_diag:
    # Predicted vs actual
    st.subheader("Predicted vs Actual")
    all_pred = predict_from_raw(df_raw, features, fit_params, beta, intercept)
    actual = df_raw['OverallScore'].values
    residuals = actual - all_pred
    school_row, school_idx = get_school_row(selected_school)
    school_pos = df_raw.index.get_loc(school_idx)

    fig_pa = go.Figure()
    other_mask = np.arange(len(df_raw)) != school_pos
    fig_pa.add_trace(go.Scatter(
        x=actual[other_mask], y=all_pred[other_mask],
        mode='markers',
        marker=dict(size=8, color='#3b82f6', opacity=0.65,
                    line=dict(width=0.5, color='black')),
        name='Other schools',
        text=df_raw.loc[~df_raw.index.isin([school_idx]), 'School'].values,
        hovertemplate="<b>%{text}</b><br>Actual: %{x:.1f}<br>Predicted: %{y:.2f}<extra></extra>",
    ))
    fig_pa.add_trace(go.Scatter(
        x=[actual[school_pos]], y=[all_pred[school_pos]],
        mode='markers', marker=dict(size=16, color='#dc2626', symbol='star'),
        name=selected_school,
        hovertemplate=f"<b>{selected_school}</b><br>Actual: %{{x:.1f}}<br>Predicted: %{{y:.2f}}<extra></extra>",
    ))
    lim = [min(actual.min(), all_pred.min()) - 2, max(actual.max(), all_pred.max()) + 2]
    fig_pa.add_trace(go.Scatter(
        x=lim, y=lim, mode='lines', line=dict(color='gray', dash='dash'),
        name='Perfect prediction'))
    fig_pa.update_layout(
        xaxis_title='Actual OverallScore', yaxis_title='Predicted OverallScore',
        height=450, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_pa, use_container_width=True)

    # Residuals
    res_l, res_r = st.columns(2)
    with res_l:
        st.subheader("Residual vs Predicted")
        fig_rs = go.Figure()
        fig_rs.add_trace(go.Scatter(
            x=all_pred, y=residuals, mode='markers',
            marker=dict(size=7, color='#7c3aed', opacity=0.7),
            text=df_raw['School'],
            hovertemplate="<b>%{text}</b><br>Pred: %{x:.2f}<br>Residual: %{y:+.2f}<extra></extra>",
        ))
        fig_rs.add_hline(y=0, line=dict(color='red', dash='dash'))
        fig_rs.update_layout(height=380, xaxis_title='Predicted',
                             yaxis_title='Residual (actual − predicted)',
                             margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_rs, use_container_width=True)

    with res_r:
        st.subheader("Residual distribution (QQ + histogram)")
        sorted_res = np.sort(residuals)
        n = len(sorted_res)
        theoretical = norm.ppf((np.arange(n) + 0.5) / n) * sorted_res.std()
        fig_q = make_subplots(rows=2, cols=1, subplot_titles=('QQ-plot', 'Histogram'),
                              vertical_spacing=0.15)
        fig_q.add_trace(go.Scatter(
            x=theoretical, y=sorted_res, mode='markers',
            marker=dict(size=6, color='#16a34a'), name='Residuals'),
            row=1, col=1)
        fig_q.add_trace(go.Scatter(
            x=theoretical, y=theoretical, mode='lines',
            line=dict(color='red', dash='dash'), name='Normal',
            showlegend=False),
            row=1, col=1)
        fig_q.add_trace(go.Histogram(x=residuals, nbinsx=25,
                                     marker_color='#16a34a',
                                     showlegend=False),
                        row=2, col=1)
        fig_q.update_xaxes(title_text='Theoretical quantile', row=1, col=1)
        fig_q.update_yaxes(title_text='Sample quantile', row=1, col=1)
        fig_q.update_xaxes(title_text='Residual', row=2, col=1)
        fig_q.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                            showlegend=False)
        st.plotly_chart(fig_q, use_container_width=True)

    # VIF + importance bars
    st.subheader("Multicollinearity (VIF) and importance comparison")
    vif_left, vif_right = st.columns(2)
    with vif_left:
        vif_sorted = M['vif'].sort_values('VIF', ascending=True)
        fig_v = go.Figure(data=[go.Bar(
            x=vif_sorted['VIF'], y=vif_sorted['Feature'],
            orientation='h',
            marker_color=['#dc2626' if v > 10 else
                          ('#f59e0b' if v > 5 else '#16a34a')
                          for v in vif_sorted['VIF']],
            text=[f"{v:.2f}" for v in vif_sorted['VIF']],
            textposition='outside',
        )])
        fig_v.add_vline(x=5, line=dict(color='#f59e0b', dash='dot'))
        fig_v.add_vline(x=10, line=dict(color='#dc2626', dash='dot'))
        fig_v.update_layout(
            height=380, xaxis_title='VIF',
            title='VIF per feature (5/10 are concern thresholds)',
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_v, use_container_width=True)

    with vif_right:
        # Coefficient ENet vs OLS comparison
        merged = coef_df.set_index('Feature')[['Beta_OLS']].join(
            M['enet'].set_index('Feature')[['Beta_ENet']])
        merged['Same_sign'] = np.sign(merged['Beta_OLS']) == np.sign(merged['Beta_ENet'])
        fig_e = go.Figure()
        for sign_match, sub in merged.groupby('Same_sign'):
            fig_e.add_trace(go.Scatter(
                x=sub['Beta_OLS'], y=sub['Beta_ENet'],
                mode='markers+text', text=sub.index, textposition='top center',
                marker=dict(size=12, color='#16a34a' if sign_match else '#dc2626'),
                name='OLS/ENet agree' if sign_match else 'OLS/ENet disagree',
            ))
        m_lim = max(merged.abs().max())
        fig_e.add_trace(go.Scatter(
            x=[-m_lim, m_lim], y=[-m_lim, m_lim], mode='lines',
            line=dict(color='gray', dash='dash'), showlegend=False))
        fig_e.update_layout(
            height=380, xaxis_title='OLS β', yaxis_title='ElasticNet β',
            title='OLS vs ElasticNet coefficient agreement',
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_e, use_container_width=True)


# =================== Tab: Cross-Model ===================
with tab_cross:
    st.subheader("Composite causal-quality across the 8 models")
    qrows = []
    for mid in MODEL_CONFIGS.keys():
        q = get_quality(mid, models)
        qrows.append({**q, 'Model': mid})
    qdf = pd.DataFrame(qrows).set_index('Model')
    # Build the stacked bar with explicit go.Bar traces so the stack mode is preserved
    # even after the Composite line trace is added on top. (Plotly Express's barmode
    # can get reset when extra trace types are appended on some versions.)
    component_order = [
        'coef_stability', 'sign_coherence', 'vif_clean',
        'importance_agreement', 'bootstrap_sign_stability',
    ]
    palette = px.colors.qualitative.Set2
    model_order = list(MODEL_CONFIGS.keys())
    qdf = qdf.reindex(model_order)

    fig_c = go.Figure()
    for i, comp in enumerate(component_order):
        fig_c.add_trace(go.Bar(
            x=model_order, y=qdf[comp].values,
            name=comp, marker_color=palette[i % len(palette)],
        ))
    fig_c.add_trace(go.Scatter(
        x=model_order, y=qdf['composite_score'].values,
        mode='lines+markers+text',
        name='Composite (mean)',
        text=[f"{v:.3f}" for v in qdf['composite_score'].values],
        textposition='top center', line=dict(color='black', width=2),
        marker=dict(size=10, color='black'),
    ))
    fig_c.update_layout(
        barmode='stack', height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title='Model', yaxis_title='Component value',
        legend=dict(orientation='v'),
    )
    st.plotly_chart(fig_c, use_container_width=True)
    st.caption("Stacked bars = the 5 components averaged into the composite score (top line). "
               "M5 leads; M1–M4 are pulled down by their elevated VIF (`vif_clean ≈ 0`).")

    # Coefficient heatmap
    st.subheader("Coefficient (β_OLS) heatmap across models")
    feats_union = sorted({f for m in models.values()
                          for f in m['model_state']['features']})
    grid = pd.DataFrame(index=feats_union, columns=list(MODEL_CONFIGS.keys()),
                        dtype=float)
    for mid, m in models.items():
        coef = m['coef'].set_index('Feature')['Beta_OLS']
        for f in coef.index:
            grid.loc[f, mid] = coef[f]
    fig_g = px.imshow(
        grid.astype(float), text_auto='.2f', aspect='auto',
        color_continuous_scale='RdBu',
        zmin=-grid.abs().max().max(), zmax=grid.abs().max().max(),
    )
    fig_g.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_g, use_container_width=True)
    st.caption("Read across each row to see how the feature's β shifts across the 8 model "
               "configurations. Stable rows = robust driver. Sign-flipping rows = unstable.")

    # Driver verdict
    st.subheader("Cross-model driver verdict")
    drivers = []
    for f in feats_union:
        models_with = [m for m, mm in models.items()
                       if f in mm['model_state']['features']]
        betas = np.array([models[m]['coef'].set_index('Feature').loc[f, 'Beta_OLS']
                          for m in models_with])
        sigs = np.array([models[m]['coef'].set_index('Feature').loc[f, 'Is_Significant']
                         for m in models_with])
        n_pos = int(((betas > 0) & sigs).sum())
        n_neg = int(((betas < 0) & sigs).sum())
        n_ns = int((~sigs).sum())
        verdict = (
            'Robust positive driver' if n_pos == len(models_with) else
            'Robust negative driver' if n_neg == len(models_with) else
            'Unstable / sign-flips' if n_pos > 0 and n_neg > 0 else
            'Weak (often non-significant)'
        )
        drivers.append({
            'Feature': f,
            'In models': len(models_with),
            'Sig +': n_pos, 'Sig −': n_neg, 'Non-sig': n_ns,
            'Mean β': float(np.mean(betas)),
            'Min β': float(np.min(betas)),
            'Max β': float(np.max(betas)),
            'Verdict': verdict,
        })
    drivers_df = pd.DataFrame(drivers)

    # Use an emoji prefix on the Verdict text so readers get an at-a-glance signal
    # without relying on pandas Styler / jinja2 (which is not installed here).
    verdict_emoji = {
        'Robust positive driver': '🟢',
        'Robust negative driver': '🔴',
        'Unstable / sign-flips': '🟡',
        'Weak (often non-significant)': '⚪',
    }
    drivers_df['Verdict'] = drivers_df['Verdict'].apply(
        lambda v: f"{verdict_emoji.get(v, '')} {v}")
    st.dataframe(
        drivers_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            'Feature': st.column_config.TextColumn(width='medium'),
            'In models': st.column_config.NumberColumn('In models', format='%d'),
            'Sig +': st.column_config.NumberColumn('Sig +', format='%d',
                help='Number of models where this feature is significant and positive'),
            'Sig −': st.column_config.NumberColumn('Sig −', format='%d',
                help='Number of models where this feature is significant and negative'),
            'Non-sig': st.column_config.NumberColumn('Non-sig', format='%d',
                help='Number of models where 95% CI crosses zero'),
            'Mean β': st.column_config.NumberColumn('Mean β', format='%+.3f',
                help='Mean standardised β across all models where the feature appears'),
            'Min β': st.column_config.NumberColumn('Min β', format='%+.3f'),
            'Max β': st.column_config.NumberColumn('Max β', format='%+.3f'),
            'Verdict': st.column_config.TextColumn('Verdict', width='large'),
        },
    )
    st.caption("🟢 robust positive driver · 🔴 robust negative driver · "
               "🟡 sign flips between models · ⚪ frequently non-significant")

    # Per-feature β across models with CIs
    st.subheader("Per-feature β across the 8 models (with 95% CI)")
    feat = st.selectbox("Feature", options=feats_union, key='cross_feat_persistent')
    rows = []
    for mid, m in models.items():
        c = m['coef'].set_index('Feature')
        if feat not in c.index:
            continue
        rows.append({
            'Model': mid,
            'Beta': c.loc[feat, 'Beta_OLS'],
            'Lower': c.loc[feat, 'Lower_95_CI'],
            'Upper': c.loc[feat, 'Upper_95_CI'],
            'Significant': c.loc[feat, 'Is_Significant'],
        })
    feat_df = pd.DataFrame(rows)
    fig_pf = go.Figure()
    fig_pf.add_trace(go.Scatter(
        x=feat_df['Model'], y=feat_df['Beta'],
        mode='markers',
        error_y=dict(type='data', symmetric=False,
                     array=feat_df['Upper'] - feat_df['Beta'],
                     arrayminus=feat_df['Beta'] - feat_df['Lower'],
                     thickness=2),
        marker=dict(size=14,
                    color=['#16a34a' if s else '#9ca3af'
                           for s in feat_df['Significant']],
                    line=dict(color='black', width=1)),
    ))
    fig_pf.add_hline(y=0, line=dict(color='red', dash='dash'))
    fig_pf.update_layout(
        height=380, yaxis_title=f"β ({feat})",
        margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
    )
    st.plotly_chart(fig_pf, use_container_width=True)
    st.caption(f"For `{feat}`: stable point estimate + tight CI across all 8 models = robust effect. "
               f"Widely varying β = effect is sensitive to modelling choices.")

st.sidebar.divider()
st.sidebar.caption(
    "Built from `outputs/per_model/` artifacts. Sliders perturb the school's feature "
    "values and recompute predictions in real time. Switch models in the sidebar; "
    "every tab updates accordingly."
)
