# Interactive Dashboard — Setup & Use

## What it is

`dashboard.py` is a [Streamlit](https://streamlit.io/) app that turns the 8-model
analysis into an interactive web UI. It reads the pre-computed artifacts in
`outputs/per_model/` (no model re-fitting) and exposes:

- **Overview** — predictive metrics, causal-quality scorecard, coefficient
  CIs, importance decompositions (|β|-share / drop-1 / Shapley), and a
  year-stability plot for the both-years configs.
- **Sensitivity Explorer** — live what-if sliders for any school's feature
  values. The school's predicted score and rank update as you drag. Includes a
  single-feature sweep view and a score-component waterfall.
- **EDA** — feature distributions (with outlier-cap markers), Spearman
  correlation heatmap, and feature-vs-target scatter.
- **Diagnostics** — predicted-vs-actual, residual scatter + QQ plot, VIF bar
  chart, and an OLS-vs-ElasticNet coefficient agreement plot.
- **Cross-Model** — stacked composite-quality bars across all 8 models, a β
  heatmap (features × models), a coloured driver-verdict table, and per-feature
  β with CIs across models.

## Prerequisites

```text
streamlit >= 1.30
plotly    >= 5.20
pandas, numpy, scipy, openpyxl, pyarrow
```

All are present in the Anaconda environment used to build the model artifacts.
If you're using a fresh environment:

```powershell
pip install streamlit plotly pandas numpy scipy openpyxl pyarrow
```

## Run

From this directory (`D:\work\US news\notebooks\new_models\`):

```powershell
streamlit run dashboard.py
```

The app opens automatically at `http://localhost:8501`. Use the **sidebar** to
pick a model (default M5, the recommended one) and a focus school (default
GWU). Each tab updates as you switch.

To run on a different port:

```powershell
streamlit run dashboard.py --server.port 8888
```

## Notes

- The dashboard caches the model load with `@st.cache_data`, so switching
  models is instant after the first load.
- Sensitivity sliders recompute predictions for the full ranking (~120-243
  rows) in real time; no perceptible lag.
- Each `(model, school, feature)` slider has its own persistent state, so
  switching models or schools doesn't erase scenarios you have set up in
  another (model, school) combination.
- If you change anything in `outputs/per_model/` (e.g. re-run the notebook
  to regenerate models), restart the dashboard or hit "Rerun" in the browser
  UI; the cache will refresh automatically.
