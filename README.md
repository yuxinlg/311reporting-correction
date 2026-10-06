# 311 reporting correction

Point-level correction of Philadelphia's 311 illegal-dumping reports for how
readily residents report, calibrated against the city's Litter Index field
survey. The output is a map of where dumping happens, not of where people
call 311.

## The idea

311 dumping reports are cheap, citywide and continuous, but places differ in
how readily residents call 311. Street-light-outage reports per street pole are
used as a yardstick for that reporting propensity: a broken light is equally
likely everywhere, so spatial variation in outage reports per pole is reporting
behaviour, `w_R`. Each dumping report is then reweighted by `exp(−β · w_R)` at
its location, the reweighted reports are kernel-smoothed into a continuous
field per unit of SafeGraph foot traffic, and the field is mapped to the 1–4
Litter Index through a proportional-odds model. The strength of the correction,
`β`, is the value at which the corrected field best predicts the survey
ratings (profile likelihood, spatial block bootstrap for the interval).

Headline (run of 2026-10-05): `β̂ = 0.86`, 95% CI `[0.63, 1.08]`. The correction
adds `Δℓ = +610` in sample with no extra parameters, `+0.0137 ± 0.0042`
log-likelihood per visit out of sample in 9 of 10 spatially blocked folds, and
a torus-shift null on the `w_R` map gives `p = 0`. Full method and results:
[`notes/METHOD.md`](notes/METHOD.md) (one page).

## Repository layout

```
src/        the pipeline: 01_data … 07_districts, plus config, kernels, ordinal, design
notes/      METHOD.md (method + results), STATUS.md (what each script writes, runtimes)
data/       symlinks to the raw data (untracked); data/README.md documents provenance
output/     everything the pipeline writes (untracked): tables, rasters, figures, logs
archive/    the first, larger pipeline, kept unchanged; reproducible at tag v0-full-pipeline
```

| script | one sentence | runtime |
|---|---|---|
| `01_data.py` | raw files → point tables, poles, block groups, support grid, survey visits, spatial blocks | 20 s |
| `02_reporting.py` | `w_R` = log(outage reports / poles) within `h_w`, centred; `h_w` chosen by a reliability rule | 40 s |
| `03_field.py` | kernel radius `h` chosen by a Spearman rule, `τ` checked by sweep and placebo, every kernel sum the statistics need | 5 min |
| `04_calibrate.py` | profile likelihood for `β`, block-bootstrap CI, first-order check, torus-shift null, outage placebo | 10 min (2 min with caches) |
| `05_validate.py` | buffered, spatially blocked 10-fold cross-validation: raw vs corrected field | 13 min |
| `06_surface.py` | the maps: 311 as filed (A), corrected (B), the survey on its own (C); the conclusion map and the correction delta | 30 s |
| `07_districts.py` | the same maps and calibration inside one planning district (Center City, South Philadelphia); a table row for all 18 | 1 min |

Every constant lives in `src/config.py`. The data-driven choices (`h_w`, `h`,
`β̂`, its CI) are recorded by the pipeline in `output/choices.json`.

## Running it

Requirements: Python 3.12 with numpy, pandas, geopandas, scipy, patsy,
rasterio, matplotlib (see `requirements.txt`). No JAX, NumPyro or statsmodels.

1. Create the symlinks in `data/` as described in [`data/README.md`](data/README.md).
   The raw inputs are Philadelphia 311 service requests (2021–2025), the
   street pole inventory, TIGER 2023 block groups, SafeGraph monthly stops per
   block group, the Litter Index survey CSV and the City Planning Commission's
   planning districts.
2. Run the scripts in order:

```bash
for s in 01_data 02_reporting 03_field 04_calibrate 05_validate 06_surface 07_districts; do python -u src/$s.py 2>&1 | tee output/logs/run_${s%%_*}.log; done
```

Each script re-runs from the raw symlinks and the outputs of the scripts
before it; nothing is read from the archive. The bootstrap and the torus null
in `04` use 8 worker processes and are cached and resumable.

## What comes out

- `output/calib/table1.csv`: calibration, one row per kernel radius and sensitivity.
- `output/cv/table2.csv`: out-of-sample log-likelihood, Somers' D, paired gain over the raw field.
- `output/surface/dumping_{window}.tif`: 50 m GeoTIFFs with the rate as filed, the corrected rate, their logs relative to the citywide rate, the correction factor and the correction delta, for 2021–2025 and 2022–2023; `litter_survey.tif` is the survey on its own.
- `output/districts/districts.csv`: one row per planning district.
- `output/figures/`: `fig1` profile curves, `fig2` the rating sigmoid, `fig3` the three maps, `fig4_*` district zooms, `fig5` the conclusion map, `fig5b` the correction delta, `A1`/`A2` appendix.

Coordinates are metres in EPSG:26918; map units are reports per year per
1,000 SafeGraph monthly stops.

## Data

Nothing under `data/` or `output/` is tracked: the raw data stay in their
source tree and are reached through symlinks, so the repository holds code and
notes only. See [`data/README.md`](data/README.md) for every input, its source
and the commands that recreate the links.
