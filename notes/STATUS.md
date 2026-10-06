# Project status

Point-level correction of 311 illegal-dumping reports for reporting propensity,
with street-light-outage reports per pole as the yardstick, calibrated to the
Litter Index field survey. **Read `notes/METHOD.md` first** — one page, the
whole method and the results.

## Pipeline (run in order; every step re-runs from the raw symlinks in `data/`)

| script | one sentence | runtime | writes to `output/` |
|---|---|---|---|
| `src/01_data.py` | raw files → point tables, poles, block groups, support grid, survey visits, 40 spatial blocks | 20 s | `points_*.parquet`, `poles_xy`, `cbg*`, `support_grid`, `visits`, `cbg_blocks` |
| `src/02_reporting.py` | w_R = log(outage reports / poles) within h_w, centred; h_w chosen by a reliability rule | 40 s | `wR_events`, `wR_visits`, `wR_raster.npz`, `choices.json` |
| `src/03_field.py` | h chosen by a Spearman rule, τ checked by sweep + placebo, then every kernel sum the statistics need | 5 min | `fields/h*.npz`, `field_sweep.csv`, `placebo.csv`, `choices.json` |
| `src/04_calibrate.py` | profile likelihood for β, block-bootstrap CI, −c/b check, torus-shift null, outage placebo, sensitivity rows | 10 min first run (8 worker processes), 2 min with caches | `calib/table1.csv`, `calib/summary.json`, `figures/fig1_*`, `fig2_*` |
| `src/05_validate.py` | buffered spatially blocked 10-fold CV: raw vs corrected (β in-fold) | 13 min | `cv/table2.csv`, `cv/summary.json`, `figures/05_cv.png` |
| `src/06_surface.py` | the maps: 311 as filed (A) and corrected (B) as long-term rates for 2021–25 and 2022–23, the survey on its own (C), their agreement; the conclusion map (B alone, planning districts named) and its delta B − A; appendix snapshot through the sigmoid | 30 s | `surface/dumping_*.tif`, `litter_survey.tif`, `litter_index_*.tif`, `surface_cbg.csv`, `figures/fig3_maps.png`, `fig5_dumping_corrected.png`, `fig5b_correction_delta.png`, `A1_*`, `A2_*` |
| `src/07_districts.py` | the same maps and the same calibration inside one planning district: Center City ("Central") and South Philadelphia ("South") get figures, all 18 districts get a table row (rates, w_R, agreement with the survey, the citywide model's gain on the district's visits, the district's own β̂) | 1 min | `districts/districts.csv`, `profile_curves.csv`, `cbg_district.csv`, `summary.json`, `figures/fig4_central.png`, `fig4_south.png` |

Headline (2026-10-05): β̂ = 0.86, 95% CI [0.63, 1.08]; the correction adds Δℓ = +610 in sample and
+0.0137 ± 0.0042 log-likelihood per visit out of sample (9/10 spatial folds); torus-shift null p = 0.
Details in `notes/METHOD.md`.

Shared modules: `src/config.py` (every constant), `src/kernels.py` (quartic
kernel sums), `src/ordinal.py` (proportional-odds logit), `src/design.py`
(spline + fixed-effect design matrix). The data-driven choices (h_w, h, β̂, CI)
live in `output/choices.json`. Logs of each run: `output/logs/run_0X.log`.

```bash
cd /Users/yuxin/Dropbox/DDDI/311reporting-correction && for s in 01_data 02_reporting 03_field 04_calibrate 05_validate 06_surface 07_districts; do python -u src/$s.py 2>&1 | tee output/logs/run_${s%%_*}.log; done
```

Environment: miniconda base, Python 3.12; numpy, pandas, geopandas, scipy,
patsy, rasterio, matplotlib. No JAX / NumPyro / statsmodels.

## Paper items the pipeline produces

- Table 1 `output/calib/table1.csv`: per h (500 / 800 / 1200) and three
  sensitivity rows — ℓ(FE only), ℓ(0), ℓ(β̂), β̂, bootstrap CI, −c/b; plus the
  torus null and the outage placebo in `calib/summary.json`.
- Table 2 `output/cv/table2.csv`: out-of-fold log-likelihood per visit, Somers' D,
  paired gain over the raw field, folds better, in-fold β range.
- Fig 1 profile curves + bootstrap; Fig 2 E[rating | log λ̂] raw vs corrected;
  Fig 3 the three maps: 311 as filed (A), corrected (B), survey on its own (C);
  Fig 5 the conclusion map, B alone with the planning districts named and the
  colorbar in multiples of the citywide rate and in absolute rates; Fig 5b the
  correction delta B − A in absolute rates, same layout;
  appendix A1 the sigmoid snapshot, A2 the two-year window, correction factor, w_R.
- Table 3 `output/districts/districts.csv` and Fig 4 `fig4_central.png`, `fig4_south.png`:
  the maps and the calibration inside Center City and South Philadelphia (planning
  districts), one table row per district for all 18.

## History

The first, larger pipeline (2026-09-30 → 10-01; nine scripts, 4,300 lines,
2.5 h) is preserved unchanged in `archive/` with its notes and outputs
(`output/archive/`), reproducible at git tag `v0-full-pipeline`. The live
pipeline does not read or cite it. See `archive/README.md`.
