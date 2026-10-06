# Data provenance

Nothing in this directory is tracked except this file. Every entry is a
**symlink** into the Dropbox source tree, so raw data stays single-sourced and
this repo stays cloneable. The source tree is **read-only for this project** —
never write through a symlink.

Source root: `/Users/yuxin/Dropbox/DDDI/Illegal-Dumping` (referred to as `$SRC`).

| link | target | what it is |
| --- | --- | --- |
| `311/` | `$SRC/replication/data/311/` | Philadelphia 311 service requests, one dir per year, `{year}/public_cases_fc.shp`. Fields used: `service_na`, geometry. |
| `Street_Poles.geojson` | `$SRC/replication/data/Street_Poles.geojson` | Street pole inventory, 203,036 poles, EPSG:4326. The **NT exposure**. Fields used: geometry, `bulb_type`, `light_date`, `owner`. |
| `tiger/` | `$SRC/replication/data/tiger/` | TIGER 2023 block groups; we use `tl_2023_42_bg/tl_2023_42_bg.shp` filtered to `COUNTYFP == "101"` (Philadelphia), 1,338 CBGs. |
| `cov_cbg.geojson` | `$SRC/replication/output/cov_cbg.geojson` | Per-CBG covariates built by the predecessor repo's `02_covariates.py`. Supplies the dumping exposure `alloc_avg_s_cnt` (SafeGraph avg monthly stops) and ~30 covariates. |
| `litter_surveys_raw_20250730.csv` | `$SRC/data/CSVFiles/litter_surveys_raw_20250730.csv` | Philadelphia Litter Index field surveys, 74,148 rows. Extracted 2025-07-30. |
| `Planning_Districts.geojson` | `$SRC/replication/data/Planning_Districts.geojson` | City Planning Commission planning districts, 18 polygons, EPSG:4326, field `dist_name` (Center City = "Central", South Philadelphia = "South", stadiums and Navy Yard = "Lower South"). Used by `07_districts.py` for the district zooms. |
| `ref_05a_output/` | `$SRC/replication/output/corrected_hawkes_streetlight/` | **Reference only.** The predecessor `05a` citywide decomposition, used as the regression-test target (β = 0.579). Not an input to this pipeline. |

## Recreating the links

```bash
SRC=/Users/yuxin/Dropbox/DDDI/Illegal-Dumping
ln -sfn $SRC/replication/data/311                              data/311
ln -sfn $SRC/replication/data/Street_Poles.geojson             data/Street_Poles.geojson
ln -sfn $SRC/replication/data/tiger                            data/tiger
ln -sfn $SRC/replication/output/cov_cbg.geojson                data/cov_cbg.geojson
ln -sfn $SRC/data/CSVFiles/litter_surveys_raw_20250730.csv     data/litter_surveys_raw_20250730.csv
ln -sfn $SRC/replication/data/Planning_Districts.geojson       data/Planning_Districts.geojson
ln -sfn $SRC/replication/output/corrected_hawkes_streetlight   data/ref_05a_output
```

## Notes on the litter survey

Characterised 2026-09-22 before any modelling:

- **74,148 rows, but effectively 2022–2023**: 27,901 (2022) + 45,879 (2023).
  2020 has 288 rows and 2024 has 79 — both dropped.
- **Streets Department, not PPR**: `department` is `streets` for 72,215 of
  73,780 in-window rows; `ppr` is 1,767.
- **A fixed-location panel, not a random sample.** 24,816 distinct
  `location_id`s, median 3 visits each; 88.5% surveyed in *both* 2022 and 2023.
- **Coverage is near-census**: 1,335 of 1,338 CBGs, median 24 distinct locations
  per CBG. Same 1,335 CBGs the decomposition fits.
- **`Litter Rating`** is ordinal 1–4, heavily skewed: 55% / 34% / 10% / 0.6%.
- **`Litter Types`** is multi-select. Bulk/dumping tokens (`trash_bags`,
  `construction`, `oversized`, `furniture`, `tires`, `vehicular`) are the
  dumping-relevant signal; the rest (`takeout_beverage`, `bottles_cans`,
  `paper_product`, `food_wrappers`) are casual litter.
- **Known confound:** survey timing is seasonal and unevenly spread across
  space. Mean rating is 1.39–1.40 in Aug/Oct but 2.01 in Nov and 1.94 in Mar,
  and CBG mean-survey-month correlates −0.405 with CBG mean rating. Month and
  `Creator` (56 surveyors) effects must be absorbed.
