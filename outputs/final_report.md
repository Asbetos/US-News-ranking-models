# US News 8-Model Ranking — Causal Comparison Report

## Executive summary

**Recommended model: `M5`** — ProfessionSalaryRank excluded, GMAT feature = `GMAT_Final_1`, years = 2026.

It has the highest composite causal-quality score (0.720 vs runner-up `M7` at 0.693). Predictive metrics: training R² = 0.973, Spearman ρ = 0.984, CV-R² = 0.962; max VIF = 6.47.

Top three drivers in this model (by Shapley R² share):

- **AvgSalaryBonus** — Shapley share 23.9%, |β| share 34.3%.
- **PeerScore** — Shapley share 22.7%, |β| share 18.6%.
- **GMAT_Combined** — Shapley share 18.0%, |β| share 4.3%.


## Causal-quality scorecard

|    |   coef_stability |   sign_coherence |   vif_clean |   importance_agreement |   bootstrap_sign_stability |   composite_score |
|:---|-----------------:|-----------------:|------------:|-----------------------:|---------------------------:|------------------:|
| M1 |            0.637 |            1     |       0     |                  0.322 |                      0.976 |             0.587 |
| M2 |            0.301 |            0.889 |       0     |                  0.483 |                      0.929 |             0.52  |
| M3 |            0.54  |            1     |       0     |                  0.344 |                      0.96  |             0.569 |
| M4 |            0.116 |            0.889 |       0     |                  0.417 |                      0.926 |             0.47  |
| M5 |            0.741 |            1     |       0.353 |                  0.508 |                      0.997 |             0.72  |
| M6 |            0.368 |            0.875 |       0.322 |                  0.571 |                      0.935 |             0.614 |
| M7 |            0.686 |            1     |       0.253 |                  0.54  |                      0.984 |             0.693 |
| M8 |            0.402 |            0.875 |       0.273 |                  0.524 |                      0.939 |             0.603 |

## Predictive metrics

|    |    R² |   Adj R² |   Spearman |   CV-R² |   CV-Spearman |   MAE |   RMSE |   Max VIF |   Cond # |
|:---|------:|---------:|-----------:|--------:|--------------:|------:|-------:|----------:|---------:|
| M1 | 0.973 |    0.971 |      0.984 |   0.962 |         0.975 | 2.46  |  3.315 |    17.071 |   10.632 |
| M2 | 0.906 |    0.902 |      0.946 |   0.877 |         0.932 | 3.83  |  6.469 |    16.282 |   10.226 |
| M3 | 0.973 |    0.971 |      0.985 |   0.961 |         0.97  | 2.491 |  3.317 |    16.962 |   10.639 |
| M4 | 0.927 |    0.924 |      0.961 |   0.903 |         0.946 | 3.719 |  5.713 |    15.807 |   10.091 |
| M5 | 0.973 |    0.971 |      0.984 |   0.962 |         0.971 | 2.472 |  3.332 |     6.467 |    6.113 |
| M6 | 0.904 |    0.901 |      0.945 |   0.88  |         0.932 | 3.842 |  6.53  |     6.784 |    6.484 |
| M7 | 0.972 |    0.97  |      0.983 |   0.96  |         0.969 | 2.498 |  3.364 |     7.466 |    6.74  |
| M8 | 0.926 |    0.924 |      0.96  |   0.906 |         0.947 | 3.762 |  5.733 |     7.274 |    6.795 |

## Per-feature driver summary across all 8 models

- **AcceptanceRate** — appears in 8/8 models. Significant positive in 0; significant negative in 4; non-significant in 4. Mean β = -0.633 (sd 0.616).
- **AvgSalaryBonus** — appears in 8/8 models. Significant positive in 8; significant negative in 0; non-significant in 0. Mean β = +8.835 (sd 0.892).
- **Employed3Mo** — appears in 8/8 models. Significant positive in 8; significant negative in 0; non-significant in 0. Mean β = +2.502 (sd 0.263).
- **EmployedAtGrad** — appears in 8/8 models. Significant positive in 8; significant negative in 0; non-significant in 0. Mean β = +2.138 (sd 0.577).
- **GMAT_Combined** — appears in 8/8 models. Significant positive in 3; significant negative in 0; non-significant in 5. Mean β = +1.015 (sd 0.867).
- **MedianGPA** — appears in 8/8 models. Significant positive in 8; significant negative in 0; non-significant in 0. Mean β = +2.577 (sd 0.241).
- **PeerScore** — appears in 8/8 models. Significant positive in 8; significant negative in 0; non-significant in 0. Mean β = +6.319 (sd 1.143).
- **ProfessionSalaryRank** — appears in 4/8 models. Significant positive in 0; significant negative in 0; non-significant in 4. Mean β = +1.327 (sd 0.679).
- **RecruiterScore** — appears in 8/8 models. Significant positive in 8; significant negative in 0; non-significant in 0. Mean β = +3.020 (sd 0.614).

## Multicollinearity findings (VIF > 5)

  - M1 / PeerScore: VIF = 6.58
  - M1 / AvgSalaryBonus: VIF = 17.07
  - M1 / ProfessionSalaryRank: VIF = 8.07
  - M2 / PeerScore: VIF = 6.69
  - M2 / AvgSalaryBonus: VIF = 16.28
  - M2 / ProfessionSalaryRank: VIF = 7.48
  - M3 / PeerScore: VIF = 6.60
  - M3 / AvgSalaryBonus: VIF = 16.96
  - M3 / ProfessionSalaryRank: VIF = 7.91
  - M3 / GMAT_Combined: VIF = 8.33
  - M4 / PeerScore: VIF = 6.50
  - M4 / AvgSalaryBonus: VIF = 15.81
  - M4 / ProfessionSalaryRank: VIF = 7.34
  - M4 / GMAT_Combined: VIF = 5.77
  - M5 / PeerScore: VIF = 6.47
  - M5 / AvgSalaryBonus: VIF = 6.23
  - M6 / PeerScore: VIF = 6.67
  - M6 / AvgSalaryBonus: VIF = 6.78
  - M7 / PeerScore: VIF = 6.52
  - M7 / AvgSalaryBonus: VIF = 6.71
  - M7 / GMAT_Combined: VIF = 7.47
  - M8 / PeerScore: VIF = 6.51
  - M8 / AvgSalaryBonus: VIF = 7.27
  - M8 / GMAT_Combined: VIF = 5.53


## OLS ↔ ElasticNet sign disagreements

  - M2 / GMAT_Combined: OLS β = -0.252, ENet β = +0.300
  - M4 / AcceptanceRate: OLS β = +0.133, ENet β = -0.121


## Year-stability (both-years configs)

  - M2: max |Δβ| between years = 4.737 on `AvgSalaryBonus`; L2 = 8.328
  - M4: max |Δβ| between years = 5.434 on `GMAT_Combined`; L2 = 8.578
  - M6: max |Δβ| between years = 4.671 on `PeerScore`; L2 = 7.328
  - M8: max |Δβ| between years = 6.276 on `GMAT_Combined`; L2 = 8.602


## Recommendation rationale

`M5` wins because:
- **Coefficient stability**: bootstrap CV averaged across features = 0.259 (lower is better).
- **Sign coherence**: 8 of 8 features have the expected sign on the transformed feature.
- **Multicollinearity**: max VIF = 6.47 (comfortable).
- **Importance agreement**: rank correlation across `|β|`-share, drop-one ΔR², and Shapley share = 0.508.
- **Bootstrap sign stability**: average % of bootstrap iterations where each coefficient kept its mean sign = 99.7%.


## School-specific analysis — George Washington University

The companion workbook `outputs/gwu_sensitivity_per_model.xlsx` gives a per-model view for GWU: one sheet per M1–M8, each containing the global coefficients, GWU's per-feature contribution decomposition, GWU's score-sensitivity matrix at ±1 % / ±5 % / ±10 %, and GWU's rank-sensitivity matrix at the same thresholds. The sensitivity tables perturb only GWU's row; every other school's features stay fixed.

**GWU in M5 (the recommended global model):** actual rank 69 / score 51; model-predicted rank 64 / score 50.8. Top actionable rank levers at +10 %:

| Feature | Δ rank | Notes |
|---------|-------:|-------|
| MedianGPA | **−14 spots** | GWU is 1.35 σ below mean; biggest single lever |
| PeerScore | −5 | |
| RecruiterScore | −5 | |
| AvgSalaryBonus | −5 | |
| Employed3Mo | −4 | GWU at 70.8 %, well below the 83 % field average |
| AcceptanceRate | 0 | GWU already at 0.25 — additional tightening doesn't move the needle |
| EmployedAtGrad | 0 | |
| GMAT_Combined | 0 | **Zero because GWU sits at the 465 floor placeholder in `GMAT_Final_1`** |

**Why M5 zeroes GMAT for GWU:** in `GMAT_Final_1`, GWU's value is the dataset minimum (465), set as a floor for unreported schools. The 5 / 95 outlier capper then pins this value, so downward perturbations are clipped back to 465 — i.e. GMAT has no measurable sensitivity for GWU in M5.

**M7 fixes this for GWU.** M7 uses `GMAT_Final_2` with KNN imputation, which gives GWU a plausible imputed GMAT of 614.6 instead of the floor. In M7, a +10 % GMAT perturbation moves GWU **−3 spots** instead of 0. M7's composite causal-quality score is 0.693 (very close to M5's 0.720), so swapping in M7 for GWU's actionable analysis sacrifices very little on the global causal story while giving GMAT a fair role.

**Practical recommendation for GWU specifically:** use M5 to read the global driver weights, and M7 to evaluate any scenario that involves changing GMAT.

## When to pick a different model

- If **predictive accuracy** matters more than coefficient stability, prefer the model with the highest CV-R² (see the predictive-metrics table).
- If **only 2026 reality** matters (e.g., decisions about next year's targets), prefer a 2026-only config; the both-years configs blend two years of data.
- If `ProfessionSalaryRank` is operationally hard to influence, prefer one of the `Excluded` configs (M5–M8) so the recommended sensitivity matrix focuses on features the user can actually move.
- If you are analysing a specific school (e.g. GWU) whose `GMAT_Final_1` value is the 465 placeholder floor, use **M7** instead of M5 so GMAT becomes an analysable feature for that school.


## Interactive dashboard

`dashboard.py` provides a Streamlit version of this report with live what-if
sliders for any school. Run with `streamlit run dashboard.py` from the
`new_models` directory. See `DASHBOARD_README.md` for full details.

## Caveats

- KNN imputation was performed once on the full per-config frame, not inside CV folds. This is acceptable because we are not making out-of-sample predictions; the imputed values are part of the *training reality* for each config. For honest predictive evaluation in deployment, imputation must be re-fitted per fold.
- Both-years configs treat each school-year as an independent observation. If you would prefer schools weighted equally across years (one row per school, year-averaged), that is a different model spec and would shift coefficients.
- `GMAT_Final_1` has many rows at the placeholder floor value, which artificially compresses the GMAT distribution and may understate GMAT's true contribution in M1/M2/M5/M6 relative to the KNN-imputed `GMAT_Final_2` configs.
