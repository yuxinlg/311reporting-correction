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
| `ref_05a_output/` | `$SRC/replication/output/corrected_hawkes_streetlight/` | **Archived pipeline only.** The predecessor `05a` citywide decomposition (β = 0.579), read by the archived `00_triage.py` / `02_reporting.py` at tag `v0-full-pipeline`. The live pipeline (`src/`) never reads it; the link can be omitted if the archive is not re-run. |
| `Litter_Index_Blocks.geojson` | downloaded file, OpenDataPhilly "Litter Index" (2026-10-20) | Published hundred-block scores, **2017 and 2018 only**, 79,603 street segments, EPSG:4326. Read only by the exploratory `src/eda_litter_index.py`. |
| `Litter_Index_Neighborhoods.geojson` | downloaded file, OpenDataPhilly "Litter Index" (2026-10-20) | Published division scores, 2017, 2018, 2023, 2024, 2025, about 1,700 divisions per year. Read by `src/eda_litter_index.py` and `src/09_validate_divisions.py`. |
| `Litter_Index_Surveys.csv` | pulled by `src/download_litter_surveys.py` from the city's `Litter_Index_Surveys` feature service (2026-10-20) | The city's public visit-level survey table: 154,382 rows, years 2017, 2018, 2023, 2024 (no 2022, no 2025). OpenDataPhilly's "surveys" download link returns the neighborhood layer, hence the script. See the notes below. |

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
- **`survey_start` is UTC**, not local time (checked 2026-10-20 against the
  city's public layer: identical to the minute once that layer's epoch dates
  are read as UTC). Surveys run in daytime, so year-months are unaffected.

## Notes on the city's public visit table (`Litter_Index_Surveys.csv`)

Compared with the raw CSV on 2026-10-20 (`survey_start` to the minute + address):

- **Same survey, different years.** 99.9% of the raw 2023 Streets visits match a
  row in the public table; ratings agree for 99.5% of matches, litter types for
  99.3%. The public table adds 2017–18 (the old method: 49,465 and 36,073 rows,
  incl. CLIP vacant lots, SEPTA, PWD, PPR, SDP) and a full 2024 Streets season
  (22,898 visits, April–June, 31 surveyors) that the raw CSV lacks (79 PWD rows
  only). It has no 2022 (the raw CSV's 27,901) and no 2025.
- **The city omitted 9,145 of the 2023 Streets visits** (`omit_flag = 1`, 21%;
  `litter_rating` blank, the rating kept in `omitted_litter_rating`): every visit
  of three surveyors, 97.5% of whose ratings were 1. The raw CSV, and therefore
  `visits.parquet`, keeps them (12.7% of the pipeline's visits).
- **Surveyors are anonymised** (`surveyor_id` such as `surveyor_2023_30`), one
  code per surveyor per year.
- **Two coordinate pairs.** `raw_x_coord`/`raw_y_coord` equal the raw CSV's
  `x`/`y`; `x_coord`/`y_coord` are the city's corrected positions (`geom` codes
  the method: `AIS` address geocode, `CMID` centerline midpoint, …), a median
  48 m away. `good_segid` (31% of rows) is the street segment, matching
  `SEG_ID` in `Litter_Index_Blocks.geojson`.
- The published `blocks_averaged` per division (49,581 in 2023, 44,211 in 2024)
  exceeds the public table's visit counts (36,801 kept in 2023, 22,898 in 2024),
  so the published index is not simply the mean of these rows.
