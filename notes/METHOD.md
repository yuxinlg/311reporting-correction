# Method — point-level 311 dumping correction, calibrated to the Litter Index

One page. Every number here is produced by `src/01_data.py` … `src/06_surface.py`
from the raw inputs in `data/`; nothing is taken from the archived pipeline.

## The question

311 illegal-dumping reports are a cheap, citywide, continuous record of dumping,
but places differ in how readily residents call 311. We use street-light-outage
reports per street pole as a yardstick for that reporting propensity, reweight
each dumping report by it, smooth the reweighted reports into a continuous
field, map the field to the 1–4 Litter Index, and let the field survey
(ground truth) tell us how strong the reweighting should be.

## Six steps

1. **Reporting propensity from poles** (`02_reporting.py`)
   `w_R(s) = log[ Σ_outages K_hw(|s−s_j|) / Σ_poles K_hw(|s−s_p|) ] − mean over poles`
   Outage reports 2021–2023 per pole, kernel-smoothed, log scale, centred.
   Identifying assumption: true breakage per pole is uniform citywide, so spatial
   variation in outage reports per pole is reporting behaviour.
   *h_w rule:* smallest radius in {400, 600, 800, 1000} m whose Poisson-noise
   reliability at the survey visits is ≥ 0.95 → **h_w = 400 m** (reliability 0.951;
   600 m gives 0.969 and is the sensitivity value). Kernel w_R at block-group
   centroids agrees with each block group's own outage rate at Spearman 0.86.

2. **The 311 field** (`03_field.py`)
   `λ̂_β(s,t) = Σ_j e^{−β w_R(s_j)} K_h(|s−s_j|) e^{−(t−t_j)/τ} 1[t ≥ t_j] / Σ_c K_h(|s−c|) E_c`
   Quartic kernel of radius h (sd = h/√7), causal exponential in time, divided
   by kernel-smoothed SafeGraph exposure on the built support. β = 0 is the raw
   field; the correction counts each report e^{−β w_R} times. Zeros are handled
   by Gamma shrinkage toward a citywide rate computed on the support grid
   (α = 1 prior report).
   *h rule:* raw-field Spearman with the rating, smallest h within 0.01 of the
   maximum → **h = 800 m** (0.375 vs 0.380 at 1200; neighbours 500 and 1200 are
   the sensitivity values).
   *τ:* the sweep is flat (0.356 / 0.375 / 0.376 / 0.362 at 60 / 120 / 240 / 480 d)
   and a placebo ties: kernel-weighted reports in the 90 days *after* a visit
   predict its rating as well as reports in the 90 days *before* (past − future
   t = −0.5 without, +0.9 with month and location effects). The time kernel is a
   pooling window, not dynamics; **τ = 120 d** is fixed.

3. **The rating model** (`design.py`, `ordinal.py`)
   `P(rating ≥ k) = σ( g(log λ̂_β) + month-of-sample + surveyor − κ_k )`
   Proportional-odds logit: three sigmoids sharing one index. g is a 5-df
   natural spline; 17 month-of-sample and 46 surveyor effects (surveyors are
   deviation-coded with a small ridge so an unseen surveyor predicts at the mean).

4. **β from ground truth** (`04_calibrate.py`)
   Turn β from −0.2 to 2.0 in steps of 0.1, recompute the field, refit the
   rating model; ℓ(β) is the maximised log-likelihood. **β̂ = argmax ℓ(β)**
   (parabola through the best three points, then one exact refit). ℓ(0) is the
   raw-field baseline. CI: spatial block bootstrap, 200 resamples of the 40
   blocks, percentile interval.

5. **Checks** (`04_calibrate.py`, `05_validate.py`)
   - −c/b from one linear fit `b·log λ̂_0 + c·w̄` (the first-order version) must
     land near β̂.
   - Torus-shift null: 50 shifted copies of the w_R map, each allowed its best
     β; none should improve the fit like the real map.
   - Outage placebo: the same field built from outage reports instead of dumping
     reports should have a slope near zero.
   - Buffered, spatially blocked 10-fold CV (test = 4 whole blocks; training
     drops everything within 1.2 km of a test visit): raw field vs corrected
     field with β chosen on the training fold.

6. **The maps** (`06_surface.py`), on 50 m cells of the built support, kept as three separate objects:
   - **A. 311 dumping as filed:** long-term average report rate, all reports in a window kernel-smoothed
     and divided by kernel-smoothed exposure, in reports per year per 1,000 SafeGraph monthly stops.
     Windows: 2021–2025 (the deliverable) and 2022–2023 (the survey years).
   - **B. Corrected:** the same with every report weighted by e^{−β̂ w_R}. "Reports that would have been
     filed at the citywide-average reporting propensity." B / A is the correction factor.
   - **C. The Litter Index on its own:** ratings with month and surveyor effects removed by one OLS on the
     survey, kernel-smoothed at the visit locations with the same kernel; nothing from 311.
   The check is that B agrees with C better than A does. Appendix: the 120-day snapshot pushed through
   the calibrated sigmoid (E[rating], how a surveyor would score a cell) for 2023-06-30 and 2025-12-31.

## Results (run of 2026-10-05; `output/choices.json`, `calib/`, `cv/`, `surface/`)

**Table 1 — calibration** (`calib/table1.csv`; month + surveyor FE, α = 1, h_w = 400 unless stated;
72,084 visits; ℓ with FE only = −53,681)

| h | ℓ(0) raw | ℓ(β̂) | Δℓ | β̂ | −c/b |
|---|---|---|---|---|---|
| 500 | −49,492 | −48,888 | +605 | 0.88 | 0.81 |
| **800** | **−49,057** | **−48,447** | **+610** | **0.86, 95% CI [0.63, 1.08]** | 0.94 [0.66, 1.24] |
| 1200 | −48,835 | −48,229 | +606 | 0.83 | 1.04 |
| 800, month-only FE | −61,271 | −60,850 | +421 | 0.93 | 1.02 |
| 800, α = 0.25 | −49,007 | −48,385 | +622 | 0.87 | 1.08 |
| 800, h_w = 600 | −49,057 | −48,310 | +748 | 1.03 | 0.94 |

- The field itself adds ~4,600 log-likelihood points over the fixed effects; the correction adds a
  further ~610 with no extra parameters. β̂ is stable across h (0.83–0.88) and α; −c/b drifts with h
  and α as a first-order approximation should, and stays inside the bootstrap interval.
- CI: 200 block resamples, sd 0.12, percentile [0.63, 1.08]; every resample's maximum was bracketed.
- **Reading β:** a report from a place one sd above the citywide reporting propensity (sd of w_R at
  reports = 0.53) counts e^{−0.86 × 0.53} ≈ 0.63 of a report; one sd below counts ≈ 1.6. The smoother
  h_w = 600 surface gives β̂ = 1.03 with a larger fit gain (+748), but the same correction strength:
  β · sd(w_R) = 0.45 at 400 m vs 0.47 at 600 m. β is in units of the w_R scale; the product is the
  comparable quantity.
- **Torus-shift null:** the real map's Δℓ on the coarse grid is +592; the best of 50 shifted maps
  reaches +226 (mean +11, 95th percentile +51); p = 0.00. The gain comes from where the poles put
  w_R, not from adding any smooth map.
- **Outage placebo:** slope +0.25 (se 0.09) and Δℓ +218 for the field built from outage reports, vs
  +0.95 (se 0.08) and +4,269 for dumping. A small generic-311 component exists (5% of the dumping
  signal); the dumping field is not just 311 activity.

**Table 2 — out-of-sample, buffered spatially blocked 10-fold CV** (`cv/table2.csv`; 20–31% of
training visits dropped by the 1.2 km buffer per fold)

| spec, h = 800 | OOF ℓ / visit | Somers' D | gain over raw ± fold SE | folds better |
|---|---|---|---|---|
| FE only | −0.814 | 0.546 | | |
| raw field (S0) | −0.724 | 0.638 | — | — |
| linear raw + c·w̄ (S1) | −0.710 | 0.653 | +0.0138 ± 0.0061 | 8/10 |
| **corrected, β in-fold (S2)** | **−0.710** | **0.652** | **+0.0137 ± 0.0042** | **9/10** |

β chosen on the training folds: 0.8 in eight folds, 1.0 in two. At h = 500 / 1200 the raw field
scores −0.731 / −0.721 and S1 −0.719 / −0.707. Block-level calibration on the 40 held-out blocks:
slope 1.02 (raw) and 1.01 (corrected), correlation 0.86 → 0.88, block RMSE 0.144 → 0.136.

**Maps** (`surface/`, 105,979 cells of 50 m, h = 800, β̂ = 0.86; `fig3_maps.png`)

| | 2021–2025 (116,772 reports) | 2022–2023 (46,608 reports) |
|---|---|---|
| citywide rate, raw → corrected (per year per 1,000 stops) | 1.19 → 1.26 | 1.19 → 1.26 |
| cell rate p1 / p50 / p99, raw | 0.01 / 0.90 / 5.85 | 0.02 / 0.89 / 6.16 |
| cell rate p1 / p50 / p99, corrected | 0.01 / 0.89 / 8.62 | 0.02 / 0.86 / 8.84 |
| correction factor B/A, p1 / p50 / p99 | 0.49 / 0.97 / 3.88 | 0.50 / 0.97 / 3.73 |
| block-group Spearman with the survey map C, raw → corrected | 0.66 → 0.69 | 0.66 → 0.69 |
| block-group Spearman with observed mean rating, raw → corrected | 0.68 → 0.72 | 0.68 → 0.71 |
| cell-level Spearman with C (98.5% of cells), raw → corrected | 0.64 → 0.65 | 0.64 → 0.64 |

- The correction halves the reported rate in the highest-propensity cells (factor 0.49 at p1) and scales
  it up almost fourfold in the lowest (3.9 at p99); the median cell barely moves (0.97). The corrected
  citywide total is 6% above the raw one because the weights are not renormalised.
- The two windows give the same picture (cell-level correlation of the 2021–25 and 2022–23 corrected
  maps is high and the correction factors are nearly identical), so the five-year map is the deliverable
  and the two-year one the like-for-like comparison with the survey.
- C: month and surveyor effects explain 33% of rating variance; the adjusted, smoothed rating has
  p1 / p50 / p99 = 0.93 / 1.47 / 2.03 and covers 98.5% of the support.
- Appendix snapshot (120-day field through the sigmoid, `A1_rating_snapshot.png`): E[rating] median
  1.26 on 2023-06-30; block-group Spearman with observed 0.71 (raw 0.68); 0.74 / 0.70 on 2025-12-31.

**Checks that passed:** sparse-vs-dense unit test (1.6e-15); interior maximum of ℓ(β) at every h;
CI excludes 0; −c/b inside the CI; torus p = 0; CV gain positive in 9/10 folds and beyond its SE;
in-fold β inside the CI; block calibration slope ≈ 1; corrected surface ranks block groups better
than the raw one.

## Choices and where they are justified

| choice | value | rule / evidence | script |
|---|---|---|---|
| h_w | 400 m (alt 600) | smallest with reliability ≥ 0.95 | 02 |
| h | 800 m (500, 1200) | smallest within 0.01 Spearman of max | 03 |
| τ | 120 d | flat sweep + placebo tie | 03 |
| α | 1 prior report (alt 0.25) | zeros 0.35% at h = 800; sensitivity row | 03, 04 |
| FE | month-of-sample + surveyor (alt month only) | sensitivity row | 04 |
| spline df | 5 | fixed | design.py |
| blocks / buffer | 40 / 1.2 km | fixed | 01, 05 |

## Stated limitations

- w_R is estimated, not known; the bootstrap conditions on it (reliability 0.95). Its scale depends
  on h_w, so β is reported together with β · sd(w_R).
- The outage placebo shows a small generic-311 component in the dumping field (5% of its fit gain).
- Surveyors appear in CV training folds through their other blocks; paired
  differences between specs are immune, absolute out-of-fold levels are optimistic.
- Euclidean, not street-network, distances.
- The 2025 surface is an extrapolation of report rates through a sigmoid
  calibrated on 2022–23; it is not a validated litter measurement.

## Inputs

Raw data only, via `data/` symlinks: 311 shapefiles (dumping, outage), the pole
inventory, TIGER block groups, the Litter Index survey CSV, and the SafeGraph
stop count per block group (`alloc_avg_s_cnt` in `cov_cbg.geojson`) as the
exposure denominator.
