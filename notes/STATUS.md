# Project status

Point-level correction of 311 illegal-dumping reports for reporting propensity,
with street-light-outage reports per pole as the yardstick; the strength of the
correction is estimated from the block groups alone and the Litter Index field
survey is the test. **Read `notes/METHOD.md` first** — one page, the whole
method and the results.

## Pipeline (run in order; every step re-runs from the raw symlinks in `data/`)

| script | one sentence | runtime | writes to `output/` |
|---|---|---|---|
| `src/01_data.py` | raw files → point tables, poles, block groups, support grid, survey visits, 40 spatial blocks (every block group) | 20 s | `points_*.parquet`, `poles_xy`, `cbg*`, `support_grid`, `visits`, `cbg_blocks` |
| `src/02_reporting.py` | w_R = log(outage reports / poles) within h_w, centred; h_w chosen by a reliability rule; evaluated at reports, visits and support cells | 1 min | `wR_events`, `wR_visits`, `wR_support`, `wR_raster.npz`, `choices.json` |
| `src/03_field.py` | h chosen by a Spearman rule, τ checked by sweep + placebo, then the kernel sums the survey test needs | 4 min | `fields/h*.npz`, `field_sweep.csv`, `placebo.csv`, `choices.json` |
| `src/04_beta.py` | β from the block groups, no survey: Poisson regression of dumping reports on w̄_R with exposure as offset; block-bootstrap CI; five other readings of the slope (window, h_w, log-rate OLS, exposure elasticity free, within-block); torus-shift null | 10 s | `beta/table1.csv`, `beta/cbg.csv`, `beta/summary.json`, `choices.json`, `figures/fig1_beta.png` |
| `src/05_survey.py` | the survey as the test: its own profile of β with a block-bootstrap CI, Δℓ at β̂, outage placebo; buffered spatially blocked 10-fold CV of raw vs corrected at β̂ (nothing chosen in-fold) | 15 min (8 worker processes for the bootstrap) | `survey/profile_curves.csv`, `bootstrap.csv`, `fit_at_beta_hat.npz`, `table2.csv`, `summary.json`, `figures/fig2_survey.png`, `fig2b_cv.png` |
| `src/06_surface.py` | the maps: 311 as filed (A) and corrected (B) as long-term rates for 2021–25 and 2022–23, the survey on its own (C), their agreement; the conclusion map (B alone, planning districts named) and its delta B − A; appendix snapshot through the sigmoid | 30 s | `surface/dumping_*.tif`, `litter_survey.tif`, `litter_index_*.tif`, `surface_cbg.csv`, `figures/fig3_maps.png`, `fig5_dumping_corrected.png`, `fig5b_correction_delta.png`, `A1_*`, `A2_*` |
| `src/07_districts.py` | the same maps, β and survey check inside one planning district: Center City ("Central") and South Philadelphia ("South") get figures, all 18 districts get a table row (rates, w_R, agreement with the survey, β from the district's own block groups, the citywide rating model's gain on its visits) | 20 s | `districts/districts.csv`, `cbg_district.csv`, `summary.json`, `figures/fig4_central.png`, `fig4_south.png` |
| `src/08_validate.py` | the plain comparison: block groups ranked by their own survey visits vs their 311 report rate as filed and corrected — rate by survey quintile, rank agreement with bootstrap intervals, disagreement maps before and after, chatty vs quiet tertiles in rating points; a before / after figure (rate as filed, corrected, difference; fit to the survey of each, change in the misfit) | 8 s | `validate/table3.csv`, `cbg.csv`, `summary.json`, `figures/fig6_validation.png`, `fig6b_before_after.png` |

Headline (2026-10-13): β̂ = 0.62, block-bootstrap 95% CI [0.28, 0.95], from a Poisson regression of
dumping reports on w_R across the 1,335 block groups, no survey; the five other readings of that slope
span −0.02 to 1.06 (the identifying assumption is fragile). On the survey the correction at β̂ adds
Δℓ = +559 in sample and +0.0123 ± 0.0029 log-likelihood per visit out of sample (9/10 spatial folds);
the survey's own β is 0.86 [0.63, 1.09], so β̂ sits just below its interval. Torus-shift null p = 0.
Details in `notes/METHOD.md`.

Shared modules: `src/config.py` (every constant), `src/kernels.py` (quartic
kernel sums), `src/ordinal.py` (proportional-odds logit), `src/design.py`
(spline + fixed-effect design matrix). The data-driven choices (h_w, h, β̂, CI)
live in `output/choices.json`. Logs of each run: `output/logs/run_0X.log`.

```bash
cd /Users/yuxin/Dropbox/DDDI/311reporting-correction && for s in 01_data 02_reporting 03_field 04_beta 05_survey 06_surface 07_districts 08_validate; do python -u src/$s.py 2>&1 | tee output/logs/run_${s%%_*}.log; done
```

Environment: miniconda base, Python 3.12; numpy, pandas, geopandas, scipy,
patsy, rasterio, matplotlib. No JAX / NumPyro / statsmodels.

## Paper items the pipeline produces

- Table 1 `output/beta/table1.csv`: β̂ and its block-bootstrap CI, with the five
  other readings of the slope (window, h_w, log-rate OLS, exposure elasticity
  free, within-block); the torus null in `beta/summary.json`.
- Table 2 `output/survey/table2.csv`: out-of-fold log-likelihood per visit,
  Somers' D, paired gain of the corrected field over the raw one, folds better;
  the survey's own β and its CI, Δℓ(β̂) and the outage placebo in `survey/summary.json`.
- Fig 1 the block-group regression, bootstrap and torus null; Fig 2 the survey's
  profile with β̂ marked, its bootstrap, E[rating | log λ̂] raw vs corrected; Fig 2b the CV;
  Fig 3 the three maps: 311 as filed (A), corrected (B), survey on its own (C);
  Fig 5 the conclusion map, B alone with the planning districts named and the
  colorbar in multiples of the citywide rate and in absolute rates; Fig 5b the
  correction delta B − A in absolute rates, same layout;
  appendix A1 the sigmoid snapshot, A2 the two-year window, correction factor, w_R.
- Table 3 `output/validate/table3.csv`, Fig 6 `fig6_validation.png` and Fig 6b
  `fig6b_before_after.png`: the plain comparison of block groups' 311 report rates,
  as filed and corrected, with their own survey visits (quintiles, rank agreement,
  disagreement maps, chatty vs quiet; the before / after maps of the rate and of the
  fit to the survey).
- District table `output/districts/districts.csv` and Fig 4 `fig4_central.png`, `fig4_south.png`:
  the maps, the district's own β and the survey check inside Center City and South
  Philadelphia (planning districts), one table row per district for all 18.

## History

The first, larger pipeline (2026-09-30 → 10-01; nine scripts, 4,300 lines,
2.5 h) is preserved unchanged in `archive/` with its notes and outputs
(`output/archive/`), reproducible at git tag `v0-full-pipeline`. The live
pipeline does not read or cite it. See `archive/README.md`.

An earlier version of this pipeline estimated β on the survey itself (profile
likelihood of the rating model). That approach, with its notes and results, lives
on the branch `litter-survey-approach`; `main` carries none of it. Here the
survey's own β appears only as one of the checks in `05_survey.py`.
