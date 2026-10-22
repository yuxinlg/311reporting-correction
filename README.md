# 311 reporting correction

Point-level correction of Philadelphia's 311 illegal-dumping reports for how
readily residents report, with the city's Litter Index field survey as the
test. The output is a map of where dumping happens, not of where people
call 311.

## The idea

311 dumping reports are cheap, citywide and continuous, but places differ in
how readily residents call 311. Street-light-outage reports per street pole are
used as a yardstick for that reporting propensity: a broken light is equally
likely everywhere, so spatial variation in outage reports per pole is reporting
behaviour, `w_R`. The strength of the correction, `β`, is estimated from 311
alone: a Poisson regression of each block group's dumping reports on its `w_R`,
with SafeGraph foot traffic as exposure, under the assumption that true dumping
per unit of exposure is unrelated to reporting propensity. Each dumping report
is then reweighted by `exp(−β · w_R)` at its location and the reweighted reports
are kernel-smoothed into a continuous field. The survey never enters `β`; it
tests the result through a proportional-odds model of the 1–4 ratings on the
field, in sample and in spatially blocked cross-validation.

Headline (run of 2026-10-13): `β̂ = 0.62`, block-bootstrap 95% CI `[0.28, 0.95]`;
five other readings of the same slope span `−0.02` to `1.06`, so the value rests
on the identifying assumption. On the survey the correction at `β̂` adds
`Δℓ = +559` in sample and `+0.0123 ± 0.0029` log-likelihood per visit out of
sample in 9 of 10 spatially blocked folds; the survey's own preferred `β` is
`0.86 [0.63, 1.09]`, just above `β̂`. Torus-shift null `p = 0`. Full method and
results: [`notes/METHOD.md`](notes/METHOD.md) (one page).

## Repository layout

```
src/        the pipeline: 01_data … 08_validate, plus config, kernels, ordinal, design
notes/      METHOD.md (method + results), STATUS.md (what each script writes, runtimes)
data/       symlinks to the raw data (untracked); data/README.md documents provenance
output/     everything the pipeline writes (untracked): tables, rasters, figures, logs
archive/    the first, larger pipeline, kept unchanged; reproducible at tag v0-full-pipeline
```

| script | one sentence | runtime |
|---|---|---|
| `01_data.py` | raw files → point tables, poles, block groups, support grid, survey visits, spatial blocks | 20 s |
| `02_reporting.py` | `w_R` = log(outage reports / poles) within `h_w`, centred; `h_w` chosen by a reliability rule | 1 min |
| `03_field.py` | kernel radius `h` chosen by a Spearman rule, `τ` checked by sweep and placebo, the kernel sums the survey test needs | 4 min |
| `04_beta.py` | `β` from the block groups without the survey: Poisson regression of reports on `w_R` with exposure as offset, block-bootstrap CI, five other readings of the slope, torus-shift null | 10 s |
| `05_survey.py` | the survey as the test: its own profile of `β` with a bootstrap CI, the gain at `β̂`, outage placebo, buffered spatially blocked 10-fold cross-validation at `β̂` | 15 min |
| `06_surface.py` | the maps: 311 as filed (A), corrected (B), the survey on its own (C); the conclusion map and the correction delta | 30 s |
| `07_districts.py` | the same maps, `β` and survey check inside one planning district (Center City, South Philadelphia); a table row for all 18 | 20 s |
| `08_validate.py` | the plain comparison: block groups ranked by their own survey visits vs their 311 report rate as filed and corrected (quintiles, rank agreement, disagreement maps, chatty vs quiet) | 3 s |
| `09_validate_divisions.py` | the same comparison with the city's published Litter Index 2023–25 per division (`data/Litter_Index_Neighborhoods.geojson`) and 311 reports of those years; rows per year and for the kernel map | 15 s |

Every constant lives in `src/config.py`. The data-driven choices (`h_w`, `h`,
`β̂`, its CI) are recorded by the pipeline in `output/choices.json`.

## Running it

Requirements: Python 3.12 with numpy, pandas, geopandas, scipy, patsy,
rasterio, matplotlib (see `requirements.txt`). No JAX, NumPyro or statsmodels.

1. Create the symlinks in `data/` as described in [`data/README.md`](data/README.md).
   The raw inputs are Philadelphia 311 service requests (2021–2025), the
   street pole inventory, TIGER 2023 block groups, SafeGraph monthly stops per
   block group, the Litter Index survey CSV, the City Planning Commission's
   planning districts and the city's published Litter Index division scores
   (a downloaded file, see `data/README.md`).
2. Run the scripts in order:

```bash
for s in 01_data 02_reporting 03_field 04_beta 05_survey 06_surface 07_districts 08_validate 09_validate_divisions; do python -u src/$s.py 2>&1 | tee output/logs/run_${s%%_*}.log; done
```

Each script re-runs from the raw symlinks and the outputs of the scripts
before it; nothing is read from the archive. The survey bootstrap in `05` uses
8 worker processes and is cached and resumable.

## What comes out

- `output/beta/table1.csv`: `β̂` with its CI and the five other readings of the slope.
- `output/survey/table2.csv`: out-of-sample log-likelihood, Somers' D, paired gain of the corrected field over the raw one; `survey/summary.json` has the survey's own `β`, its CI and the gain at `β̂`.
- `output/surface/dumping_{window}.tif`: 50 m GeoTIFFs with the rate as filed, the corrected rate, their logs relative to the citywide rate, the correction factor and the correction delta, for 2021–2025 and 2022–2023; `litter_survey.tif` is the survey on its own.
- `output/districts/districts.csv`: one row per planning district.
- `output/validate/table3.csv`: the plain comparison of block groups' 311 rates, as filed and corrected, with their own survey visits.
- `output/validate_divisions/table3d.csv`: the same comparison with the city's published Litter Index 2023–25 per division, pooled and year by year.
- `output/figures/`: `fig1` the block-group regression, `fig2` the survey's profile and the rating sigmoid, `fig2b` the cross-validation, `fig3` the three maps, `fig4_*` district zooms, `fig5` the conclusion map, `fig5b` the correction delta, `fig6` the plain comparison with the survey, `fig6b` the before / after maps of the rate and of the fit to the survey, `A1`/`A2` appendix.

Coordinates are metres in EPSG:26918; map units are reports per year per
1,000 SafeGraph monthly stops.

## Data

Nothing under `data/` or `output/` is tracked: the raw data stay in their
source tree and are reached through symlinks, so the repository holds code and
notes only. See [`data/README.md`](data/README.md) for every input, its source
and the commands that recreate the links.
