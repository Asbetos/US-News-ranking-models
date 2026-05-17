# GWSB — US News Ranking Models

Causal analysis of the US News business-school ranking. Eight bootstrapped-OLS
models recover the implicit feature weights US News uses, and an interactive
Streamlit dashboard lets you explore the result.

## What's in here

| File / directory | Purpose |
|------------------|---------|
| `dashboard.py` | Interactive Streamlit dashboard — 5 tabs (Overview, Sensitivity, EDA, Diagnostics, Cross-Model) |
| `us_news_8_models.ipynb` | The analytical notebook that fit all 8 models and produced the artifacts |
| `US News Data 2025 - 2026.xlsx` | Source dataset (2025 + 2026 sheets) |
| `outputs/final_report.md` | Written recommendation (M5 wins on causal-quality; M7 for school-specific GMAT scenarios) |
| `outputs/summary_comparison.xlsx` | 29-sheet cross-model comparison workbook |
| `outputs/gwu_sensitivity_per_model_v2.xlsx` | Per-model GWU sensitivity workbook (corrected per-year rank) |
| `outputs/per_model/M1..M8/` | Saved coefficients, importance, VIF, sensitivities, model pickle, training frame per model |
| `plans/2026-05-16-eight-model-ranking-build.md` | Implementation plan |
| `DASHBOARD_README.md` | Dashboard run instructions |
| `requirements.txt` / `runtime.txt` | Streamlit Cloud deployment configuration |

## The 8 models

| Model | ProfessionSalaryRank | GMAT feature | Year(s) | Causal-quality composite |
|-------|---------------------|--------------|---------|-------------------------:|
| M5 (recommended) | excluded | GMAT_Final_1 | 2026 | **0.720** |
| M7 | excluded | GMAT_Final_2 (KNN-imputed) | 2026 | 0.693 |
| M6 | excluded | GMAT_Final_1 | 2025+2026 | 0.614 |
| M8 | excluded | GMAT_Final_2 (KNN-imputed) | 2025+2026 | 0.603 |
| M1 | included | GMAT_Final_1 | 2026 | 0.587 |
| M3 | included | GMAT_Final_2 (KNN-imputed) | 2026 | 0.569 |
| M2 | included | GMAT_Final_1 | 2025+2026 | 0.520 |
| M4 | included | GMAT_Final_2 (KNN-imputed) | 2025+2026 | 0.470 |

See `outputs/final_report.md` for the full causal driver story and recommendation rationale.

## Run the dashboard locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. See `DASHBOARD_README.md` for details.

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Sign in at https://share.streamlit.io.
3. Click **New app**, choose this repo, set:
   - **Branch:** `main`
   - **Main file path:** `dashboard.py`
   - **Python version:** 3.12 (auto-detected from `runtime.txt`)
4. Deploy. Streamlit Cloud installs from `requirements.txt` and starts the app.

## Re-running the analysis

Open `us_news_8_models.ipynb` in Jupyter / VS Code and run all cells. The notebook
re-fits all 8 models (~5–8 minutes on a modern laptop) and rewrites every artifact
under `outputs/`. A standalone re-generator for the per-model GWU workbook is
provided at `_regen_gwu_workbook.py` — it loads the saved model pickles and rebuilds
`gwu_sensitivity_per_model.xlsx` without re-fitting (~30 seconds).
