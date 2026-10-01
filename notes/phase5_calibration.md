# Phase 5 — calibration, β, validation, surface

Scripts: `src/05_calibrate.py` (ladder, profile, bootstrap, controls),
`src/06_cv.py` (buffered blocked nested CV), `src/07_surface.py` (raster),
`src/08_figures.py` (figure set + `output/figures/tables.md`), with
`src/ordinal.py` (proportional-odds logit, analytic gradient, ridge,
cluster-robust covariance; matches statsmodels' OrderedModel to 1e-4 in
log-likelihood and is ~30× faster after column standardisation).

Setup throughout: ratio-form field (÷ kernel-smoothed SafeGraph exposure),
τ = 120 d, Gamma shrinkage α = 1, month-of-sample + surveyor FE, natural
cubic spline g with 5 df (centred, one column dropped so the cutpoints carry
the intercept), SEs clustered on the 40 CV blocks. w_R: kernel surface at the
report (primary) or CBG posterior mean. β₀ = 0.554 (Stage 1a production).

## Ladder (`output/calib/ladder.csv`)

| h | ℓ FE-only | ℓ S0 | Δℓ S2−S0 kernel / CBG | S1′ b | S1′ c | β̂ = −c/b | Fieller 95% | S1′ CBG β̂ | Mundlak β̂ (between) |
|---|---|---|---|---|---|---|---|---|---|
| 500 | −53676.9 | −49593.5 | +546 / +513 | 0.962 | −0.917 | 0.954 | [0.67, 1.28] | 0.991 | 0.908 |
| **800** | −53676.9 | −49109.0 | +575 / +539 | 1.005 | −1.086 | 1.081 | [0.77, 1.44] | 1.178 | 1.043 |
| 1200 | −53676.9 | −48851.9 | +581 / +526 | 1.065 | −1.236 | 1.161 | [0.83, 1.54] | 1.325 | 1.121 |

The field adds ~4,500 log-likelihood points over the FE-only model; the
plug-in correction (S2 at β₀ = 0.554) adds a further ~575 with the same df.
The Mundlak split shows the between-location slope carries the signal
(within-location slope ≈ 0, as the placebo implied).

## Profile likelihood (`profile_summary.csv`, `profile_curves.csv`)

| h | w_R | g | FE | α | β̂ | naive χ² 95% |
|---|---|---|---|---|---|---|
| 500 | kernel | spline | month+surv | 1 | 1.089 | [1.04, 1.15] |
| **800** | **kernel** | **spline** | **month+surv** | **1** | **1.037** | [0.99, 1.09] |
| 800 | kernel | linear | month+surv | 1 | 1.114 | |
| 800 | CBG | spline | month+surv | 1 | 1.007 | |
| 800 | kernel | spline | month only | 1 | 1.206 | |
| 800 | kernel | spline | month+surv | 0.25 | 1.050 | |
| 1200 | kernel | spline | month+surv | 1 | 0.979 | |

The naive χ² intervals are ±0.05 because 72k rows are treated as independent.

**Spatial block bootstrap** (100 resamples of the 40 blocks, h = 800, kernel,
spline; `bootstrap.csv`): profile β̂ sd 0.150, percentile 95% [0.76, 1.29];
S1′ sd 0.179, [0.74, 1.47]. The bootstrap 95% critical value of the LR
statistic is **94 vs χ²'s 3.84**, i.e. the ~25× inflation the plan predicted
(~1,300 effective units behind 72k rows). **Headline: β̂ = 1.04, 95% CI
[0.79, 1.29]** (bootstrap-calibrated profile). 05a's β₀ = 0.55 [0.43, 0.69]
is outside it; all 100 bootstrap β̂ exceed 0.55.

Closed form vs profile: S1′ gives 1.08 (kernel) and the linear-g profile 1.11,
the spline profile 1.04. They need not agree exactly: S1′ is the first-order
linearisation and drops the β²·Var_K(w_R)/2 term, and the Gamma shrinkage
interacts with it. The profile uses the exact dependence and is the estimator.

## Controls and checks (`calib/summary.json`)

- **Permutation null** (50 torus shifts of the ŵ_R raster, autocorrelation
  preserved): shifted surfaces *worsen* the fit, Δℓ mean −268 (sd 231, q95
  +26, max +370); actual +575; p = 0.00. The correction is not "any smooth
  spatial covariate helps".
- **Outage placebo field** (h = 500): b = +0.185 (se 0.085), Δℓ vs FE +125;
  dumping b = 0.888, Δℓ +3,754. A small generic neighbourhood/reporting
  component exists (t ≈ 2); it is 3% of the dumping signal.
- **Dedup** (50 m / 14 d piles, h = 500): |c| does **not** shrink (0.932 vs
  0.917); b rises 0.96 → 1.14, so β̂ falls 0.95 → 0.82. Repeat calls at one
  address do not carry the reporting-propensity signal the plan expected;
  the pile field predicts the rating slightly better.
- **Differential outcome** (binary logits): bulk b 0.79, Δℓ raw +664,
  correction +103 (+15%); casual b 0.84, Δℓ +2,407, correction +175 (+7%).
  The correction helps the dumping-specific outcome relatively more, but the
  field predicts casual litter too; it is a neighbourhood-disorder signal
  with a dumping-specific component.
- **Partial proportional odds** −c_k/b_k: 1.05 (≥2), 1.08 (≥3), 1.37 (≥4,
  n = 396). The single-sigmoid framing holds for the two cutpoints that carry
  data.
- **Literal 1 + 3σ(η)** by NLS: RMSE 0.533 vs 0.532 for the ordinal E[rating];
  predictions p1–p99 [1.01, 2.76] = 58% of the 1–4 range; ordinal cutpoints
  [0.02, 2.85, 6.26], spacings 2.8 and 3.4 — not equal but not far off.
- **α sensitivity**: S1′ β̂ 1.29 / 1.23 / 1.16 / 1.08 / 0.98 / 0.82 for
  α = 0.1 / 0.25 / 0.5 / 1 / 2 / 5, and S0's in-sample fit prefers smaller α.
  The **profile β̂ is robust** (1.05 at α = 0.25 vs 1.04 at α = 1); the closed
  form is not. α is chosen out of fold in `06` (see below).
- **Level reconciliation**: CBG-mean fitted E[rating] vs CBG-mean rating,
  Spearman 0.83 (S0) / 0.85 (S2) with FE; **field only 0.68 / 0.71** (old
  CBG-level plan: 0.58 / 0.64).
- **Linearisation** (`figures/05_linearisation.png`): the raw-field curve
  flattens above the 8th decile; the corrected curve keeps rising. The
  correction straightens the top end, as the plan's replacement test #1 asked.

## Surface (`output/surface/`)

`07_surface.py`: 105,979 cells of 50 m on the built support, h = 800, β̂ =
1.037, S2 model refit at β̂, FE at the population mean. GeoTIFFs for
2023-06-30 (calibration window) and 2025-12-31 (extrapolation, flagged in the
file tags), bands E_rating_corr, E_rating_raw, P_ge3_corr, log_lam_corr,
log_lam_raw; CBG means in `surface_cbg.csv`.

| date | E[rating] p1 / p50 / p99 | correction Δ sd, p1 / p99 | CBG Spearman with observed 2022–23 mean: corrected / raw |
|---|---|---|---|
| 2023-06-30 | 1.08 / 1.25 / 2.02 | 0.099, −0.22 / +0.37 | 0.723 / 0.678 |
| 2025-12-31 | 1.08 / 1.27 / 1.95 | 0.100, −0.23 / +0.39 | 0.739 / 0.697 |

The index is on the report scale pushed through the sigmoid at the mean
month/surveyor effect: a relative surface, compressed toward the low end
(median 1.25 vs the survey mean 1.56) because the sigmoid is evaluated at a
typical cell and a typical FE. The correction moves cells by ±0.2–0.4 rating
points at the extremes (sd 0.10), i.e. ~0.15 sd of the rating — larger than
the plan's 0.07 sd guess because β̂ is twice β₀. The date barely matters
(0.72 vs 0.74), as the placebo predicted.

Known artefact: a single coordinate holding 167 reports produces a
disc-shaped feature of radius h around it (visible as a ring in the
correction map in the Northeast). The dedup field or a per-coordinate cap
would remove it; it is cosmetic for the calibration (one of 116,772 points).

## Validation (`output/cv/`, 65 min)

10 outer folds of 4 blocks; training drops visits within 1.2 km of a test
visit (19–23% of training lost per fold); month FE + deviation-coded ridged
surveyor effects; spline knots from the training fold.

**Out-of-fold log-likelihood per visit (τ = 120) and Somers' D**

| spec | h=500 | h=800 | h=1200 | Somers' D (800) |
|---|---|---|---|---|
| FE only | | −0.8145 | | 0.546 |
| S0 raw field | −0.7328 | −0.7249 | −0.7214 | 0.638 |
| S0 + fine 150 m component | −0.7331 | −0.7252 | −0.7213 | 0.637 |
| S1 raw + c·w̄ | −0.7188 | −0.7082 | −0.7047 | 0.655 |
| **S2 corrected, kernel ŵ_R, β₀** | −0.7215 | **−0.7130** | −0.7083 | 0.650 |
| S2 corrected, CBG w_R, β₀ | −0.7216 | −0.7135 | −0.7102 | 0.650 |
| S2 corrected, β chosen in fold | −0.7176 | −0.7094 | −0.7052 | 0.654 |
| S4 raw + CBG covariates | −0.6888 | −0.6869 | −0.6846 | 0.676 |

**Paired gain over S0** (per visit, ± fold SE, folds better): S2 kernel
+0.0120 ± 0.0024 (9/10) at h = 800, +0.0131 ± 0.0035 (9/10) at 1200; S2 CBG
+0.0114 ± 0.0024 (9/10); S1 +0.0166 ± 0.0066 (8/10); β-in-fold +0.0155 ±
0.0044 (9/10); fine-scale component −0.0003 ± 0.0002 (nothing); covariates
+0.038 ± 0.005 (10/10, prediction only).

**The headline verification holds: the correction beats the raw field on
held-out spatial blocks**, in 9 of 10 folds, with either w_R, and the gain
grows with the buffer (+0.0096 at 0 m, +0.0120 at 1.2 km, +0.0152 at 2.4 km —
the raw field profits more from spatial leakage).

**(h, τ) surface, S2 kernel**: best h = 1200, τ = 240 (−0.7067); the 1-SE rule
also lands on 1200/240 (800/120 is −0.0063 ± 0.0034 behind). So the CV prefers
the coarser kernel for *prediction*; the β headline stays at 800 where the
bootstrap was run (profile β̂ 0.98 at 1200 vs 1.04 at 800).

**Nested selection** (inner 5-fold on training blocks, buffer 600 m): picks
S2 kernel ×6 / S2 CBG ×4, h = 1200 ×9; nested OOF ℓ/visit −0.7118 vs the fixed
default S2 kernel h = 800 −0.7130 vs best-in-hindsight S4 −0.6846
(selection-biased). The honest number and the fixed default agree.

**Block-level calibration** (40 held-out blocks, h = 800): slope 1.02 / 1.02,
intercept −0.006 / +0.002, corr 0.86 / 0.88, block RMSE 0.145 / 0.136 for S0 /
S2. Calibrated at the block level, and the correction tightens it.

**Temporal holdout** (calendar-month FE, same locations): 2023 → 2022 the
correction helps (S0 −0.914, S2 −0.886, S1 −0.880; Somers' D 0.54 → 0.57);
2022 → 2023 it does not (S0 −0.795, S2 −0.798). Asymmetric; the 2022 training
year has fewer visits (27,901) and the correction's between-location gain
does not transfer in that direction. Stated as a limitation.

**α out of fold** (h = 800): monotone, smaller is better for both S0 and S2
(ℓ/visit −0.7222 at α = 0.1 vs −0.7249 at α = 1 for S0; −0.7102 vs −0.7130
for S2). CV picks α = 0.1 (the smallest tried). The profile β̂ is insensitive
to α; the plug-in and the surface should move to α = 0.1 in a final rerun.

### Configuration for the paper

- β: profile at h = 800, kernel ŵ_R, spline g, month + surveyor FE, α = 1 →
  **1.04, 95% CI [0.79, 1.29]** (block-bootstrap calibrated); 0.98 at h = 1200.
- Prediction / surface: CV-chosen h = 1200, τ = 240, α = 0.1 (gain over the
  800/120/1 default is +0.006 + 0.002 + 0.003 ≈ 0.011 ℓ/visit). The surface at
  h = 1200, τ = 120, α = 1 is produced alongside the 800 one
  (`07_surface.py --h 1200`); the τ = 240 / α = 0.1 variant needs the profile
  numerators recomputed at τ = 240 (05) — pending.
