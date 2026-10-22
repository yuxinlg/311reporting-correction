# Method — point-level 311 dumping correction, with the Litter Index as the test

One page. Every number here is produced by `src/01_data.py` … `src/09_validate_divisions.py`
from the raw inputs in `data/`; nothing is taken from the archived pipeline.

## The question

311 illegal-dumping reports are a cheap, citywide, continuous record of dumping,
but places differ in how readily residents call 311. We use street-light-outage
reports per street pole as a yardstick for that reporting propensity, estimate
from the block groups alone how strongly dumping reports follow it (β), reweight
each dumping report by e^{−β w_R}, and smooth the reweighted reports into a
continuous field. The Litter Index field survey does not enter the estimate of β;
it is the test: does the correction at β̂ predict the ratings better than the raw
reports, and which β would the survey itself have chosen?

## Nine steps

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
   the sensitivity values). h is a smoothing bandwidth for the maps and the
   rating model; it does not enter β.
   *τ:* the sweep is flat (0.356 / 0.375 / 0.376 / 0.362 at 60 / 120 / 240 / 480 d)
   and a placebo ties: kernel-weighted reports in the 90 days *after* a visit
   predict its rating as well as reports in the 90 days *before* (past − future
   t = −0.5 without, +0.9 with month and location effects). The time kernel is a
   pooling window, not dynamics; **τ = 120 d** is fixed.

3. **β from the block groups, no survey** (`04_beta.py`)
   `T_i ~ Poisson( E_i · exp(a + β · w̄_R,i) )` over the 1,335 block groups: T_i
   the dumping reports 2021–2025, E_i the SafeGraph exposure (offset), w̄_R,i the
   mean kernel w_R over the block group's support cells. β is the elasticity of
   the report rate to reporting propensity, fitted by IRLS; CI from a spatial
   block bootstrap (200 resamples of the 40 blocks). *Identifying assumption:*
   true dumping per unit of exposure is unrelated to reporting propensity across
   block groups (the residual is independent of w_R) — the same assumption as
   the two-equation decomposition of the archived pipeline, which found 0.58 on
   its own w_R scale. Five other readings of the same slope are reported next
   to it: the survey years only; w_R at h_w = 600; least squares of the log
   rate (every block group weighted equally); the exposure elasticity left free
   (log E as a covariate); and 40 block fixed effects (neighbouring block groups
   only). Torus-shift null: 50 shifted copies of the w_R raster, each regressed
   the same way; the real map's deviance gain against theirs.

4. **The rating model** (`design.py`, `ordinal.py`)
   `P(rating ≥ k) = σ( g(log λ̂_β) + month-of-sample + surveyor − κ_k )`
   Proportional-odds logit: three sigmoids sharing one index. g is a 5-df
   natural spline; 17 month-of-sample and 46 surveyor effects (surveyors are
   deviation-coded with a small ridge so an unseen surveyor predicts at the mean).
   Used only to test the correction against the survey and, in the appendix, to
   translate the field into E[rating].

5. **The survey as the test** (`05_survey.py`)
   - In sample: turn β from −0.2 to 2.0 in steps of 0.1, recompute the field,
     refit the rating model; ℓ(β) is the maximised log-likelihood. One exact
     refit at β̂ gives **Δℓ(β̂) = ℓ(β̂) − ℓ(0)**, the correction's gain on the
     survey. **β_survey = argmax ℓ(β)** with its own block-bootstrap CI: the
     test is Δℓ(β̂) > 0 and β̂ inside the survey's interval; the loss
     ℓ(β_survey) − ℓ(β̂) is how much the survey would gain from its own value.
   - Outage placebo: the same field built from outage reports instead of dumping
     reports should have a slope near zero.
   - Out of sample: buffered, spatially blocked 10-fold CV (test = 4 whole
     blocks; training drops everything within 1.2 km of a test visit): raw
     field vs corrected field at β̂, nothing chosen in-fold.

6. **The maps** (`06_surface.py`), on 50 m cells of the built support, kept as three separate objects:
   - **A. 311 dumping as filed:** long-term average report rate, all reports in a window kernel-smoothed
     and divided by kernel-smoothed exposure, in reports per year per 1,000 SafeGraph monthly stops.
     Windows: 2021–2025 (the deliverable) and 2022–2023 (the survey years).
   - **B. Corrected:** the same with every report weighted by e^{−β̂ w_R}. "Reports that would have been
     filed at the citywide-average reporting propensity." B / A is the correction factor.
   - **C. The Litter Index on its own:** ratings with month and surveyor effects removed by one OLS on the
     survey, kernel-smoothed at the visit locations with the same kernel; nothing from 311.
   The check is that B agrees with C better than A does. B on its own, with the 18 planning districts
   outlined and named and the colorbar in multiples of the citywide rate and in absolute rates, is the
   conclusion map (`fig5_dumping_corrected.png`): the estimate of where dumping happens. Its companion
   `fig5b_correction_delta.png` is B − A in absolute units (band `rate_delta`): where the correction removed
   reports (chatty places, brown) and where it added them (quiet places, teal). Appendix: the 120-day
   snapshot pushed through the fitted sigmoid (E[rating], how a surveyor would score a cell) for
   2023-06-30 and 2025-12-31.

7. **District zooms** (`07_districts.py`): the same maps, β and survey check inside one City Planning
   Commission district (Center City = "Central", South Philadelphia = "South"; block groups assigned by their
   representative point, reports, visits and cells follow their block group). Per district: report rate as
   filed and corrected from the points and the district's exposure, as multiples of the citywide rate; mean
   w_R at its reports; block-group Spearman of A and B with C; β from the district's own block groups (the
   04 regression on them alone, robust SE); and the citywide rating model's log-likelihood gain on the
   district's visits, ℓ_d(β̂) − ℓ_d(0) with the citywide coefficients (the 18 districts sum to the citywide
   Δℓ(β̂)). The district figure shows A, B, C, w_R, the correction factor and the correction delta (B − A)
   on the citywide colour scales; the numbers sit in the title and the table.

8. **The plain comparison with the survey** (`08_validate.py`): no rating model, no kernel. Ground truth
   per block group = the mean rating of its own visits with month and surveyor effects removed (≥ 10
   visits; 1,290 block groups). 311 per block group = reports in the survey years per unit of exposure,
   as filed and with every report counted e^{−β̂ w_R} (and at the survey's own β, table only), Gamma-shrunk
   log rates. Four readings: the mean 311 rate in survey quintiles (cleanest → dirtiest); rank agreement
   (Spearman, the "dirtiest tenth" hit rate, with 40-block bootstrap intervals for the corrected − as filed
   difference); the block-group-by-block-group disagreement in percentile ranks, mapped as filed and
   corrected; and the chatty / middle / quiet w_R tertiles with 311 translated into rating points through
   its own straight line, so the survey's gap and 311's gap are on one scale. A second figure puts the
   before / after side by side: the rate as filed, corrected, their difference; the fit to the survey of
   each (311 read as a rating minus the survey's rating) and the change in the misfit.

9. **The published Litter Index as a second test** (`09_validate_divisions.py`): step 8 again on the city's own
   published index for 2023–2025 (`data/Litter_Index_Neighborhoods.geojson`: the mean visit rating per electoral
   division, `streets_division_score`, and the visits behind it, `blocks_averaged`), against 311 reports of the
   same three years. Ground truth per division = the three annual scores pooled by their visit counts (≥ 10 visits;
   1,699 divisions); no month or surveyor adjustment is possible, the file holds means. Exposure and w̄_R come from
   the support cells inside the division, the bootstrap block from the block group under its representative
   point. Extra rows: each year on its own (that year's reports against that year's score) and 06's kernel map
   rebuilt for 2023–25. The city's public visit table (`data/Litter_Index_Surveys.csv`, 2023 and 2024 only,
   pulled by `download_litter_surveys.py`) gives 08's own ground truth for those two years: the visits the city
   omitted from its index dropped, ratings with month + surveyor effects removed by one OLS, averaged per
   division; two more rows test the 2023–24 reports against that (adjusted, and the plain mean).

## Results (run of 2026-10-13; `output/choices.json`, `beta/`, `survey/`, `surface/`, `districts/`; step 9 run 2026-10-20)

**Table 1 — β from the block groups** (`beta/table1.csv`; 1,335 block groups, 116,751 dumping reports
2021–2025, w̄_R at h_w = 400 unless stated, sd(w̄_R) = 0.53; CI = percentile of 200 resamples of the 40 blocks)

| reading of the slope | β | robust se | block-bootstrap 95% CI | β · sd(w̄_R) |
|---|---|---|---|---|
| **main: Poisson, exposure offset, 2021–25** | **0.62** | 0.11 | **[0.28, 0.95]** | **0.33** |
| reports 2022–23 only | 0.65 | 0.11 | [0.29, 0.98] | 0.34 |
| w_R at h_w = 600 | 0.59 | 0.12 | [0.21, 0.98] | 0.29 |
| least squares of log(reports / exposure) | 0.85 | 0.10 | [0.38, 1.30] | 0.45 |
| exposure elasticity free (γ = 0.10) | −0.02 | 0.06 | [−0.32, 0.25] | −0.01 |
| 40 block fixed effects (within-block) | 1.06 | 0.08 | [0.80, 1.29] | 0.56 |

- **β̂ = 0.62 [0.28, 0.95].** A block group one sd chattier files e^{0.33} ≈ 1.4× the dumping reports at
  equal true dumping; in the maps a report from such a place counts 0.72 of a report, one from a place
  one sd quieter 1.4. The archived two-equation decomposition found 0.58 on its own (unsmoothed,
  shrunk) w_R scale.
- **The slope is not a stable quantity.** The six readings span −0.02 to 1.06 and the bootstrap interval
  of the main reading is three times wider than its robust se says (spatial clustering of the residual).
  Two facts drive this: dumping reports barely scale with SafeGraph stops across block groups
  (elasticity γ = 0.10 when freed, against the 1 the "per stop" rate imposes), and w̄_R is correlated
  with log exposure (−0.31). Imposing the offset gives 0.62; letting the elasticity float leaves no
  slope at all; comparing only neighbouring block groups (block fixed effects, broad contrasts removed)
  gives 1.06. The identifying assumption — true dumping per stop unrelated to w_R — therefore fails in a
  direction that depends on how the comparison is made, and nothing in 311 can adjudicate. This is the
  honest uncertainty of a survey-free β; the survey (Table 2) is the only external check.
- **Torus-shift null:** the real map's deviance gain is 16,410; the best of 50 shifted w_R rasters reaches
  8,779 (mean 1,530, 95th percentile 7,525), p = 0.00. The slope is specific to where the poles put w_R.

**Table 2 — the survey as the test** (`survey/summary.json`, `survey/table2.csv`; month + surveyor FE,
h = 800; 72,084 visits; ℓ with FE only = −53,681, ℓ(0) raw field = −49,057)

| in sample | |
|---|---|
| Δℓ(β̂ = 0.62) = ℓ(β̂) − ℓ(0) | **+559** (no extra parameters) |
| the survey's own β: argmax ℓ(β), block-bootstrap 95% CI | **0.86 [0.63, 1.09]** (0.88 / 0.83 at h = 500 / 1200) |
| loss ℓ(β_survey) − ℓ(β̂) | 51 |
| β̂ inside the survey's interval? | **no** — 0.62 sits just below the lower end, 0.63 |
| outage placebo: linear slope, Δℓ vs FE | outage +0.25 (se 0.09), +218  vs  dumping +0.95 (se 0.08), +4,269 |

| out of sample, buffered spatially blocked 10-fold CV (20–31% of training visits dropped by the 1.2 km buffer) | OOF ℓ / visit | Somers' D | gain over raw ± fold SE | folds better |
|---|---|---|---|---|
| FE only | −0.814 | 0.546 | | |
| raw field (S0), h = 800 | −0.724 | 0.638 | — | — |
| **corrected at β̂ = 0.62 (S2), nothing chosen in-fold** | **−0.712** | **0.651** | **+0.0123 ± 0.0029** | **9/10** |

- The correction at the block-group β̂ helps on the survey: +559 in sample (92% of the +610 the survey's own
  value reaches) and +0.0123 ± 0.0029 per visit out of sample, 9 of 10 folds, with nothing chosen in-fold.
- The survey prefers a stronger correction than 311 alone gives: 0.86 vs 0.62, and β̂ falls just outside the
  survey's interval. The direction is consistent with the within-block reading in Table 1 (1.06): comparing
  block groups citywide understates the slope, because quiet block groups with high exposure are also the
  cleaner ones per stop.
- Raw field at h = 500 / 1200: −0.731 / −0.721. Block-level calibration on the 40 held-out blocks: slope 1.02
  (raw) and 1.01 (corrected), correlation 0.86 → 0.88, block RMSE 0.144 → 0.136.
- A small generic-311 component exists (the outage field carries 5% of the dumping field's gain); the dumping
  field is not just 311 activity.

**Maps** (`surface/`, 105,979 cells of 50 m, h = 800, β̂ = 0.62; `fig3_maps.png`)

| | 2021–2025 (116,772 reports) | 2022–2023 (46,608 reports) |
|---|---|---|
| citywide rate, raw → corrected (per year per 1,000 stops) | 1.19 → 1.21 | 1.19 → 1.20 |
| cell rate p1 / p50 / p99, raw | 0.01 / 0.90 / 5.85 | 0.02 / 0.89 / 6.16 |
| cell rate p1 / p50 / p99, corrected | 0.01 / 0.88 / 7.27 | 0.02 / 0.86 / 7.43 |
| correction factor B/A, p1 / p50 / p99 | 0.59 / 0.97 / 2.58 | 0.60 / 0.97 / 2.47 |
| block-group Spearman with the survey map C, raw → corrected | 0.66 → 0.69 | 0.66 → 0.68 |
| block-group Spearman with observed mean rating, raw → corrected | 0.68 → 0.72 | 0.68 → 0.71 |
| cell-level Spearman with C (98.5% of cells), raw → corrected | 0.64 → 0.66 | 0.64 → 0.65 |

- The correction takes 40% off the reported rate in the highest-propensity cells (factor 0.59 at p1)
  and scales it up 2.6× in the lowest (p99); the median cell barely moves (0.97). The corrected
  citywide total is 2% above the raw one because the weights are not renormalised. The corrected map
  agrees with the survey better than the raw one at every level (block groups 0.66 → 0.69).
- The two windows give the same picture (the correction factors are nearly identical), so the
  five-year map is the deliverable and the two-year one the like-for-like comparison with the survey.
- C: month and surveyor effects explain 33% of rating variance; the adjusted, smoothed rating has
  p1 / p50 / p99 = 0.93 / 1.47 / 2.03 and covers 98.5% of the support.
- Appendix snapshot (120-day field through the sigmoid, `A1_rating_snapshot.png`): E[rating] median
  1.26 on 2023-06-30; block-group Spearman with observed 0.71 (raw 0.68); 0.73 / 0.70 on 2025-12-31.

**District zooms** (`districts/districts.csv`, `fig4_central.png`, `fig4_south.png`; 2021–25 window)

| | Center City (Central) | South Philadelphia (South) | citywide |
|---|---|---|---|
| block groups / reports / visits | 143 / 13,562 / 5,022 | 111 / 13,714 / 4,710 | 1,335 / 116,772 / 72,084 |
| mean w_R at reports (0 = citywide) | +0.22 | +0.52 | +0.05 |
| report rate as filed → corrected, × citywide | 0.61 → 0.54 | 1.71 → 1.24 | 1 → 1 |
| district correction factor; cell p10 / p50 / p90 | 0.91; 0.72 / 0.94 / 1.20 | 0.75; 0.66 / 0.74 / 1.21 | 1.03; 0.73 / 0.97 / 1.53 |
| survey: observed mean rating; map C mean | 1.25; 1.38 | 1.66; 1.51 | 1.56; 1.49 |
| block-group Spearman with C, raw → corrected | 0.05 → 0.02 | 0.22 → 0.26 | 0.66 → 0.69 |
| β from the district's own block groups (robust se) | 1.26 (0.19) | 1.07 (0.17) | 0.62 (0.11) |
| citywide rating model's gain on the district's visits (share of +559) | +60 (11%) | +101 (18%) | +559 |

- South Philadelphia is the chattiest large district (mean w_R +0.52; only Central Northeast is higher): it
  files 1.7× the citywide report rate per stop and the correction brings it to 1.24×. The factor is nearly
  uniform inside the district (p10–p50 0.66–0.74), so there the correction is a level shift; the citywide
  rating model's largest single gain comes from these visits (+101). Block-group agreement with the survey
  is weak on either map (0.22 → 0.26).
- Center City files 0.61× the citywide rate per stop (its SafeGraph exposure is the city's largest) and is
  also chatty (+0.22), so the correction lowers it to 0.54×. The survey has almost no spatial structure to
  rank inside Center City (adjusted mean 1.38; block-group Spearman ≈ 0 for both maps), so the zoom cannot
  show whether the correction improves the ranking there.
- **Within districts the slope is steeper than citywide.** β from a district's own block groups is above 0.8
  in 12 of the 18 districts (range −0.16 to 2.17; Center City 1.26, South 1.07, North 0.99, Lower North 1.12),
  the same message as the within-block reading of Table 1: the citywide 0.62 is pulled down by contrasts
  between districts, which is where exposure and true dumping co-vary with w_R.
- Closure: the 18 district gains sum to the citywide Δℓ(β̂) = +559 exactly. The gain is positive in 17
  districts; the largest are South (+101), River Wards (+98), North (+87) and Lower North (+83); Upper
  Northwest loses 38.

**Table 3 — the plain comparison with the survey** (`validate/table3.csv`, `validate/summary.json`,
`fig6_validation.png`; 1,290 block groups with ≥ 10 visits, reports 2022–23 per unit of exposure)

| block groups vs the survey | as filed | corrected (β̂ = 0.62) | at the survey's β (0.86) |
|---|---|---|---|
| Spearman with the adjusted mean rating | 0.510 | **0.542** (gain +0.032, 95% [+0.013, +0.054]) | 0.546 |
| same, with 06's kernel maps instead of counts | 0.581 | 0.608 | |
| dirtiest tenth per the survey caught by 311's top tenth | 28% | 30% | 31% |
| mean \|311 rank − survey rank\| (percentiles) | 0.222 | 0.217 | |
| block groups moved closer / further from the survey's rank | | 50% / 49% | |
| 311 rate in survey quintiles Q1 (cleanest) … Q5 (dirtiest), × citywide | 1.02, 1.22, 1.75, 2.52, 3.56 | 0.87, 1.01, 1.52, 2.49, 3.75 | 0.81, 0.92, 1.42, 2.45, 3.81 |
| dirtiest / cleanest quintile ratio | 3.5 | 4.3 | 4.7 |
| chatty − quiet tertile gap, rating points (survey: −0.10) | +0.02 | −0.03 | −0.05 |

- **Direction right, size modest.** Block groups the survey calls dirtier do file more reports, and the
  correction steepens that relation: it takes reports away from the cleanest quintiles (×1.02 → ×0.87)
  and adds them to the dirtiest (×3.56 → ×3.75), so the dirtiest-to-cleanest ratio rises from 3.5 to 4.3.
  Rank agreement improves by a small but definite amount (+0.03 Spearman, interval excludes 0); the
  2021–25 window gives the same (+0.037 [+0.017, +0.062]).
- **The bias it removes is the chatty-neighbourhood one.** The survey rates the chattiest tertile 0.10
  points *cleaner* than the quietest; 311 as filed, read as a rating, calls it 0.02 points dirtier; corrected
  it says 0.03 cleaner, at the survey's β 0.05 cleaner. The correction turns the sign the right way and
  closes about a third of the gap; the survey's own β closes half.
- **Most block groups barely move.** Half end closer to the survey's rank and half further; the mean
  absolute rank disagreement falls from 0.222 to 0.217. Where 311 still disagrees after the correction
  (panel E) is the same pattern as before, only paler: the correction is a modest, directionally right
  adjustment, not a reordering. The map version (06's kernel rasters) agrees better with the survey than
  the raw counts at both stages (0.58 → 0.61), as smoothing should.
- **Before / after in one picture** (`fig6b_before_after.png`): the block-group report rate as filed, corrected
  and their difference (top row), and the fit to the survey of each — 311 read as a rating through its own
  straight line (one log unit of report rate ≈ 0.10 rating points) minus what the survey found — and the
  change in that misfit (bottom row). Mean |misfit| 0.186 → 0.183 rating points; 56% of block groups end
  closer to the survey, 44% further. The misfit map keeps its shape: 311 reads South and West Philadelphia
  and the Lower Northwest dirtier than the survey found and the Northeast and North Delaware cleaner, before
  and after; the correction pales the first group and does nothing for the second.

**Table 3b — the published Litter Index 2023–25 as the test** (`validate_divisions/table3d.csv`, `summary.json`,
`fig7_divisions.png`, `fig7b_divisions_before_after.png`; 1,699 divisions with ≥ 10 visits, 67,386 reports 2023–25
per unit of exposure; the published streets score, no month or surveyor adjustment)

| divisions vs the published streets score | as filed | corrected (β̂ = 0.62) | at the survey's β (0.86) |
|---|---|---|---|
| Spearman, 2023–25 pooled | 0.563 | **0.624** (gain +0.061, 95% [+0.041, +0.091]) | 0.641 |
| same, with 06's kernel map rebuilt for 2023–25 | 0.645 | 0.710 | |
| Spearman, each year alone: 2023 / 2024 / 2025 | 0.563 / 0.341 / 0.393 | 0.602 / 0.373 / 0.445 (gains +0.039 / +0.033 / +0.052, every interval above 0) | 0.611 / 0.381 / 0.459 |
| Spearman, 2023–24 reports vs the published 2023–24 score / the city's visit table as a plain mean per division / the same with month + surveyor effects removed (08's ground truth; 58,029 visits, 1,653 divisions) | 0.554 / 0.539 / 0.535 | 0.605 / 0.589 / 0.570 (gains +0.051 / +0.050 / +0.035, every interval above 0) | 0.617 / 0.601 / 0.575 |
| dirtiest tenth per the index caught by 311's top tenth | 26% | 28% | 28% |
| mean \|311 rank − index rank\| (percentiles) | 0.210 | 0.196 | |
| divisions moved closer / further from the index's rank | | 58% [55%, 62%] / 41% | |
| 311 rate in index quintiles Q1 (cleanest) … Q5 (dirtiest), × citywide | 0.85, 1.19, 1.95, 2.63, 3.44 | 0.65, 0.98, 1.76, 2.58, 3.66 | 0.58, 0.89, 1.68, 2.53, 3.72 |
| dirtiest / cleanest quintile ratio | 4.0 | 5.6 | 6.4 |
| chatty − quiet tertile gap, rating points (index: −0.20) | −0.01 | −0.06 | −0.08 |
| mean \|misfit\| in rating points; divisions closer / further | 0.167 | 0.160; 63% / 37% | |

- **The published index says what the survey said, more strongly.** The gain in rank agreement is twice the
  block-group one (+0.061 vs +0.032), the interval sits well above 0, each of the three years on its own gains
  (+0.03 to +0.05), and the kernel map gains as much as the counts (0.645 → 0.710). The correction again takes
  reports away from the cleanest quintile (×0.85 → ×0.65) and adds them to the dirtiest (×3.44 → ×3.66).
- **The same chatty-neighbourhood bias.** The index rates the chattiest tertile of divisions 0.20 points cleaner
  than the quietest; 311 as filed sees no difference (−0.01), corrected −0.06, at the survey's β −0.08. The
  correction closes about a third of the gap, as on the survey; the survey's stronger β closes more.
- **Agreement depends on the year of the index.** As filed, 311 agrees with the 2023 index at 0.56 but with the
  2024 and 2025 indices only at 0.34–0.39; those are the years in which the published scale collapses toward 1
  (`output/litter_index_eda/summary.md`). The correction's gain holds in every year.
- **08's own ground truth from the visit table says the same.** With the city's omitted visits dropped and month
  + surveyor effects removed (they explain 35% of rating variance, as in the 2022–23 survey), the 2023–24 gain is
  +0.035 [+0.017, +0.061]; with the plain visit mean +0.050. The published score is that plain mean (division
  Spearman 0.94 with it, 0.74 with the adjusted mean), and the adjustment lowers agreement with 311 slightly:
  surveyors cover areas, so part of what their effects remove is geography.

**Sensitivity — without the visits the city omitted from its index** (`output/sens_no_omitted/`, run 2026-10-21: the
whole pipeline with `DROP_CITY_OMITTED=1`; 01 drops the 9,300 raw rows that match the 9,150 visits flagged
`omit_flag = 1` in the public visit table — three 2023 surveyors whose ratings were 97.5% "1" — leaving 62,790 visits)

| | main run (72,084 visits) | without the omitted visits (62,790) |
|---|---|---|
| h_w, h | 400, 800 | 400, 800 |
| β̂ (04), block-bootstrap CI | 0.62 [0.28, 0.95] | 0.62 [0.22, 0.99] (only the blocks change) |
| Δℓ(β̂) in sample; the survey's own β | +559; 0.86 [0.63, 1.09], β̂ just outside | +526; 0.85 [0.60, 1.06], **β̂ inside** |
| CV gain of the corrected field ± fold SE, folds better | +0.0123 ± 0.0029, 9/10 | +0.0155 ± 0.0051, 9/10 |
| Table 3 Spearman as filed → corrected (gain, CI) | 0.510 → 0.542 (+0.032 [+0.013, +0.054]) | 0.509 → 0.542 (+0.033 [+0.014, +0.055]) |
| chatty − quiet gap, rating points: survey / 311 as filed / corrected | −0.10 / +0.02 / −0.03 | −0.13 / +0.02 / −0.03 |
| block-group Spearman of the maps with survey map C, raw → corrected (2021–25) | 0.66 → 0.69 | 0.66 → 0.69 |

- Nothing moves. The surveyor effects had already absorbed those three surveyors' level, so β̂, the survey's β,
  the gains and the rank agreement are unchanged to the second decimal; the survey's interval shifts down a
  little and now contains β̂ (lower end 0.60 instead of 0.63). Out-of-fold levels are not comparable between the
  runs (the dropped visits were almost all rating 1, the easiest to predict); the paired gain is.

**Checks that passed:** sparse-vs-dense unit test (1.6e-15); β̂'s bootstrap CI excludes 0; torus p = 0;
Δℓ(β̂) = +559 > 0 on the survey; CV gain positive in 9/10 folds and four times its SE; block calibration
slope ≈ 1; corrected counts and surface rank block groups better than the raw ones (Table 3, interval
excludes 0); the chatty − quiet gap has the survey's sign after the correction; district gains close exactly;
the published 2023–25 index ranks the corrected counts closer than the raw ones, pooled and in each year (Table 3b).
**Checks that did not pass:** β̂ = 0.62 falls just outside the survey's interval [0.63, 1.09] (inside it,
[0.60, 1.06], once the visits the city omitted from its index are dropped); the six
readings of the block-group slope disagree (−0.02 to 1.06), so the survey-free β is a reading under an
assumption, not a measurement.

## Choices and where they are justified

| choice | value | rule / evidence | script |
|---|---|---|---|
| h_w | 400 m (alt 600) | smallest with reliability ≥ 0.95 | 02 |
| h | 800 m (500, 1200) | smallest within 0.01 Spearman of max (a map bandwidth; β does not depend on it) | 03 |
| τ | 120 d | flat sweep + placebo tie | 03 |
| α | 1 prior report | zeros 0.35% at h = 800 | 03 |
| β estimator | Poisson, exposure offset, no controls | the model's own likelihood with counts aggregated; the five other readings sit beside it | 04 |
| β window | 2021–2025 (alt 2022–23) | the deliverable map's window | 04 |
| FE | month-of-sample + surveyor | fixed | 05 |
| spline df | 5 | fixed | design.py |
| blocks / buffer | 40 / 1.2 km | fixed | 01, 05 |

## Stated limitations

- **β rests on an untestable-from-311 assumption** (true dumping per unit of exposure unrelated to w_R
  across block groups) and the six readings in Table 1 show how much the slope moves when that
  comparison is made differently (−0.02 to 1.06). The survey is the only external check; in this run it
  prefers a stronger correction than β̂ (Table 2).
- SafeGraph stops are a weak exposure for dumping at block-group level (elasticity 0.10); the rate "per
  stop" is a definition, not a fitted relation, and it enters β through the offset.
- w_R is estimated, not known; the bootstrap conditions on it (reliability 0.95). Its scale depends
  on h_w, so β is reported together with β · sd(w_R).
- The outage placebo shows a small generic-311 component in the dumping field (5% of its fit gain).
- Surveyors appear in CV training folds through their other blocks; paired
  differences between specs are immune, absolute out-of-fold levels are optimistic.
- Euclidean, not street-network, distances.
- The 2025 surface is an extrapolation of report rates through a sigmoid
  fitted on 2022–23; it is not a validated litter measurement.

## Inputs

Raw data only, via `data/` symlinks: 311 shapefiles (dumping, outage), the pole
inventory, TIGER block groups, the Litter Index survey CSV, the SafeGraph
stop count per block group (`alloc_avg_s_cnt` in `cov_cbg.geojson`) as the
exposure denominator, and the City Planning Commission's 18 planning districts
(`Planning_Districts.geojson`) for the district zooms. The city's published Litter Index division scores
(`Litter_Index_Neighborhoods.geojson`, a downloaded file, see `data/README.md`) enter only step 9.
