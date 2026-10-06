# Project status

Point-level correction of 311 illegal-dumping reports for reporting propensity,
with street-light-outage reports per pole as the yardstick, calibrated to the
Litter Index field survey. **Read `notes/METHOD.md` first** — one page, the
whole method and the results. `notes/GUIDE_zh.md` is the zero-context
walkthrough in Chinese: every output file, every step, every number.

## Pipeline (run in order; every step re-runs from the raw symlinks in `data/`)

| script | one sentence | runtime | writes to `output/` |
|---|---|---|---|
| `src/01_data.py` | raw files → point tables, poles, block groups, support grid, survey visits, 40 spatial blocks | 20 s | `points_*.parquet`, `poles_xy`, `cbg*`, `support_grid`, `visits`, `cbg_blocks` |
| `src/02_reporting.py` | w_R = log(outage reports / poles) within h_w, centred; h_w chosen by a reliability rule | 40 s | `wR_events`, `wR_visits`, `wR_raster.npz`, `choices.json` |
| `src/03_field.py` | h chosen by a Spearman rule, τ checked by sweep + placebo, then every kernel sum the statistics need | 5 min | `fields/h*.npz`, `field_sweep.csv`, `placebo.csv`, `choices.json` |
| `src/04_calibrate.py` | profile likelihood for β, block-bootstrap CI, −c/b check, torus-shift null, outage placebo, sensitivity rows | 10 min first run (8 worker processes), 2 min with caches | `calib/table1.csv`, `calib/summary.json`, `figures/fig1_*`, `fig2_*` |
| `src/05_validate.py` | buffered spatially blocked 10-fold CV: raw vs corrected (β in-fold) | 13 min | `cv/table2.csv`, `cv/summary.json`, `figures/05_cv.png` |
| `src/06_surface.py` | the 50 m corrected litter-index GeoTIFF for two dates | 30 s | `surface/*.tif`, `surface_cbg.csv`, `figures/fig3_maps.png` |

Headline (2026-10-05): β̂ = 0.86, 95% CI [0.63, 1.08]; the correction adds Δℓ = +610 in sample and
+0.0137 ± 0.0042 log-likelihood per visit out of sample (9/10 spatial folds); torus-shift null p = 0.
Details in `notes/METHOD.md`.

Shared modules: `src/config.py` (every constant), `src/kernels.py` (quartic
kernel sums), `src/ordinal.py` (proportional-odds logit), `src/design.py`
(spline + fixed-effect design matrix). The data-driven choices (h_w, h, β̂, CI)
live in `output/choices.json`. Logs of each run: `output/logs/run_0X.log`.

```bash
cd /Users/yuxin/Dropbox/DDDI/311reporting-correction && for s in 01_data 02_reporting 03_field 04_calibrate 05_validate 06_surface; do python -u src/$s.py 2>&1 | tee output/logs/run_${s%%_*}.log; done
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
  Fig 3 maps of w_R, the corrected index, and the correction.

## History

The first, larger pipeline (2026-09-30 → 10-01; nine scripts, 4,300 lines,
2.5 h) is preserved unchanged in `archive/` with its notes and outputs
(`output/archive/`), reproducible at git tag `v0-full-pipeline`. The live
pipeline does not read or cite it. See `archive/README.md`.
