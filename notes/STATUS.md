# Project status — archived 2026-09-30 (end of session)

Point-level 311 dumping correction calibrated to the Litter Index. Plan:
`project-propose-plan-simplified-model-p-typed-grove.md`. This file is the
hand-off: what exists, what the numbers say, what changed relative to the
plan, and how to resume. Phase notes with full tables: `notes/day1_triage.md`,
`notes/phase3_reporting.md`, `notes/phase4_field.md`.

## Pipeline (run in this order; every step re-runnable from scratch)

| script | what | runtime | outputs (all under `output/`, gitignored) |
|---|---|---|---|
| `src/01_build_panel.py` | CBG × month panel + point tables | 15 s | `panel_cbg_month`, `cbg_static`, `cbg_geom`, `points_dumping` (116,772), `points_outage` (75,994), `poles_xy` (203,036) |
| `src/00_triage.py` | Day-1 gate: placebo, spline, variance budget, atoms, routing, spec curve | 2 min | `triage/*` |
| `src/02_reporting.py` | Stage 1: 05a refit (regression + production MCMC) + kernel ŵ_R surfaces | 8 min | `reporting_cbg`, `reporting_draws.npz` (2,000 joint draws), `wR_events`, `wR_survey_xy`, `wR_raster.npz`, `reporting_summary.json` |
| `src/03_litter_panel.py` | survey rows → model table, 40 CV blocks | 20 s | `litter_panel` (72,084 visits), `cbg_blocks` |
| `src/04_field.py` | Stage 2: fields at every visit, sweep, denominators, diagnostics, unit tests | 3.5 min | `visit_fields` (140 cols), `support_grid`, `field_sweep.csv`, `field_decoupled.csv`, `field_alt_denominators.csv`, `field_summary.json` |
| `src/05_calibrate.py` | Stage 3 step 1: ladder, profile, bootstrap, controls | ~75 min (bootstrap 58) | `calib/ladder.csv`, `profile_summary.csv`, `profile_curves.csv`, `bootstrap.csv`, `profile_num_h*_t120.npz`; **controls + `summary.json` + figures NOT yet written by a full run** (see below) |
| `src/06_cv.py` | Stage 3 step 2: buffered blocked nested CV, temporal holdout | ~45 min est. | **written and smoke-tested, not yet run in full** |
| `07_surface.py`, `08_figures.py` | citywide raster, publication figures | — | **not written** |

Shared modules: `src/kernels.py` (quartic kernel sums, space-time
accumulator, sparse W, Gamma-shrunk log), `src/ordinal.py` (proportional-odds
logit with analytic gradient, ridge, cluster-robust covariance, Somers' D;
matches statsmodels to 1e-4 in log-lik, 30× faster).

Environment: miniconda base, Python 3.12; `statsmodels 0.15.0` was installed
this session. JAX/NumPyro on CPU. Run logs of every step: `output/logs/`.
Predecessor repo `$SRC` untouched (its pre-existing dirty state is recorded in
the session scratchpad `predecessor_git_baseline.txt`; nothing new added).

## Decisions taken this session (deviations from the plan, all data-driven)

1. **311 records are filing events only** (user decision). `closed_dat` removed;
   no open-case stock, no closure controls. Cleanup/response confound on β̂
   becomes a stated limitation checked via the bulk outcome.
2. **Kernel scale.** The plan's "h = 200 m optimum" was in Gaussian-sd units.
   With the quartic kernel (sd = h/√7) the absolute-density optimum is radius
   500 m; the **ratio form's optimum is radius 1200 m** (sd ≈ CBG diameter),
   plateau 800–2000. Floor 300 m (geocoding atoms). Provisional default
   **h = 800** (on the plateau, keeps more reporting contrast: ρ(h) = 0.71).
   User asked about 50 m; rejected on evidence (39% zeros, −28% correlation,
   two-scale test adds +0.004 R²). Fine-scale component kept only as a check.
3. **Time kernel is a pooling window.** Placebo ties (future 90-day count
   predicts the rating as well as the past one; within-location past−future
   t ≈ 0.5). τ = 120 d fixed; τ sweep flat.
4. **Ratio form (÷ kernel-smoothed SafeGraph exposure) is primary.** Spearman
   .380 vs .306 absolute; exposure alone predicts nothing (+0.01 to +0.04);
   poles/population/street length as denominators do not do it (.30–.33).
5. **Framing softened twice.** Not "sub-CBG resolution": the optimal kernel is
   CBG-diameter scale. The contribution is a continuous relative-risk surface
   at exact coordinates/dates that beats the best polygon alternative
   (own + 1st-order neighbours, .346) by +0.034 Spearman.
6. **Stage 1b primary is `pre24_naive`** (2021–23 outage reports / all poles),
   not the plan's pole-time-at-risk form: LED conversion did not reduce
   outage reports (2024–25 vs 2021–23 ratio 0.52/0.50/0.50/0.55 across
   LED-share quartiles), so "non-LED pole-days at risk" is mis-specified. The
   +0.27 corr(w_R, led_share) predates conversion → rollout targeting, not
   contamination. h_w = 600 m (reliability 0.97).
7. **β₀ = 0.554 [0.428, 0.687]** from a production posterior (4 × 4000/8000,
   block-dense mass), not 05a's 0.579: 05a's single chain has a_T n_eff 7,
   β n_eff 18. Regression test with 05a's settings reproduces every number
   exactly. Production β/a_T R̂ 1.014/1.016 (strict < 1.01 missed; MC se 0.003).
8. File numbering: `03_litter_panel.py` before `04_field.py` (run order = file
   order); `05_calibrate.py` + `06_cv.py` split Stage 3.

## Results so far

**Stage 1a** (`reporting_summary.json`): regression PASS; production β₀ 0.554,
a_NT −1.079, a_T −5.262, σ_R 0.617, σ_T 1.491, 0 divergences, n 1,335;
corr(w_R, log R/P) 0.998 on R ≥ 5.

**Stage 1b**: kernel ŵ_R (600 m) vs CBG posterior: 0.80 at centroids, 0.79
averaged over each CBG's reports (resolution mismatch, not error).

**Field sweep** (Spearman with rating, τ = 120): ratio form .267/.331/.361/
.375/.380/.372 at h = 150/300/500/800/1200/2000; absolute peaks .306 at 500.
Zeros 0.35% at 800. Unit tests: sparse = dense to 1e-15; kernel mass 1.00000.

**Ladder** (`calib/ladder.csv`; ordinal logit, month + surveyor FE, block-clustered):

| h | ll S0 | Δll S2−S0 kernel / CBG | S1′ b | S1′ c | β̂ = −c/b | Fieller 95% | S1′ CBG β̂ | Mundlak β̂ |
|---|---|---|---|---|---|---|---|---|
| 500 | −49593.5 | +546 / +513 | 0.962 | −0.917 | 0.954 | [0.67, 1.28] | 0.991 | 0.908 |
| **800** | −49109.0 | +575 / +539 | 1.005 | −1.086 | 1.081 | [0.77, 1.44] | 1.178 | 1.043 |
| 1200 | −48851.9 | +581 / +526 | 1.065 | −1.236 | 1.161 | [0.83, 1.54] | 1.325 | 1.121 |

**Profile likelihood** (`calib/profile_summary.csv`; naive χ² CIs are ±0.05
because 72k rows are treated as independent — use the bootstrap):

| h | w_R | g | β̂ | naive 95% |
|---|---|---|---|---|
| 500 | kernel | spline | 1.089 | [1.04, 1.15] |
| **800** | **kernel** | **spline** | **1.037** | [0.99, 1.09] |
| 800 | kernel | linear | 1.114 | |
| 800 | CBG | spline | 1.007 | |
| 800 | kernel | spline, month FE only | 1.206 | |
| 800 | kernel | spline, α = 0.25 | 1.050 | |
| 1200 | kernel | spline | 0.979 | |

**Spatial block bootstrap** (100 resamples of 40 blocks, h = 800, kernel, spline;
`calib/bootstrap.csv`): profile β̂ sd 0.150, 95% pct [0.764, 1.289]; S1′ β̂ sd
0.179, [0.735, 1.465]. Bootstrap LR critical value 94 vs χ² 3.84 (the ~25×
independence inflation the plan predicted, effective units ~1,300). **Headline
so far: β̂ = 1.04, bootstrap-recalibrated 95% CI [0.79, 1.29]**, about twice
05a's 0.55 [0.43, 0.69]; the two intervals do not overlap.

**Controls** — the full run was stopped in this section; the numbers below are
from the smoke test, which used the SAME full-sample fits (only the profile
grid, bootstrap reps and permutation count were reduced), so they are valid
except where marked:
- Outage placebo field (h = 500): b = +0.185 (se 0.085), Δll vs FE +125 — vs
  dumping b = 0.888, Δll +3754. Small but non-zero generic component.
- Torus-shifted ŵ_R null for Δll(S2 − S0): shifted surfaces *worsen* the fit
  (≈ −400 with 2 shifts; needs the 50-shift run); actual +575.
- Dedup (h = 500): piles c = −0.932 vs all reports −0.917 — |c| does **not**
  shrink (plan predicted it would); b rises 0.96 → 1.14 so β̂ falls 0.95 → 0.82.
- Differential outcome: bulk b 0.79, Δll raw +664, correction +103 (15%);
  casual b 0.84, Δll +2407, correction +175 (7%). Relative gain larger for bulk.
- Partial-PO −c_k/b_k: 1.05 (≥2), 1.08 (≥3), 1.37 (≥4, n = 396). Single-sigmoid OK.
- NLS 1 + 3σ(η): RMSE 0.533 = ordinal 0.532; predictions span 58% of 1–4
  (p1–p99 [1.01, 2.76]); cutpoints [0.02, 2.85, 6.26] (not equally spaced).
- α sensitivity (S1′, h = 800): β̂ 1.16 (α 0.5) → 1.08 (1) → 0.98 (2) → 0.82 (5);
  S0 ll prefers smaller α. **α matters for β̂ — needs a principled choice
  (CV) or reporting as a spec dimension.**
- Level reconciliation: CBG-mean fitted E[rating] vs CBG-mean rating, Spearman
  0.83 (S0) / 0.85 (S2) with FE; field-only version not yet computed.

**CV smoke** (5 outer folds, h = 500, buffer 1.6 km — not the final config):
S2k beats S0 out of fold in 5/5 folds, Δll/visit +0.0128 ± 0.0021; S2c +0.0121;
S1 +0.012; fine-scale +0.000; S4 (covariates) +0.058; block calibration slope
0.98–1.00; temporal 2022→2023 S2k ≈ S0, 2023→2022 S2k better by 0.027.

## Pending

1. **Finish `05`**: the controls section, `summary.json`, figures. Profile
   numerators are cached, so ladder + profiles take ~6 min; the bootstrap is
   58 min — set `N_BOOT` lower or move the controls before the bootstrap if
   only the controls are needed. Consider making α a swept dimension.
2. **Run `06_cv.py`** in full (10 outer × 5 inner, buffer 1200 / inner 600 m;
   ~45 min). It reads `calib/profile_num_h*_t120.npz` (present).
3. **Write `07_surface.py`**: raster of E[rating] and P(rating ≥ 3) at 100 m
   from the chosen (h, β) model, FE at the population mean, exposure
   denominator on the support grid, ŵ_R raster for the correction; GeoTIFF via
   rasterio (installed). Flag 2024–25 as extrapolation.
4. **Write `08_figures.py`** and the paper tables.
5. **Two-stage uncertainty**: refit the profile over the 2,000 saved (β, w_R)
   draws for the CBG-w_R variant (plan: Rubin's rules) — not done.
6. **Location-scale ordinal check** (scale ~ surveyor) — not done.
7. Planning districts (leave-one-district-out) need a symlink to
   `$SRC/replication/data/Planning_Districts.geojson` — not added.

## Points for the PI

- The paper is spatial; τ is a pooling window (placebo).
- Optimal kernel sd ≈ CBG diameter; the gain over polygons is ~10% of the
  correlation; sell the continuous surface, not resolution.
- Ground-truth β̂ ≈ 1.0 [0.8, 1.3] vs 05a 0.55 [0.43, 0.69]: not casual-litter
  OVB (bulk gives the same), not sampling error. Candidates: cleanup/response
  channel (out of scope by decision), civic-engagement channel other than
  reporting, or 05a's prior-driven identification. Report as a bracket.
- The at-risk denominator and the "PSIP contamination" story do not survive
  the data; the reporting proxy predicts where the city converted.
- Surveyor effects explain 27% of rating variance and are identified; they are
  a leakage channel in spatial CV (surveyors are spatially organised) — stated.
