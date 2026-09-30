# Phase 4 — survey table and the 311 intensity field

Scripts: `src/03_litter_panel.py` (survey rows → model table, CV blocks),
`src/04_field.py` (Stage 2 field at every visit), shared machinery in
`src/kernels.py` (`accumulate_st`, `build_W`, `shrunk_log`).

## Survey table (`output/litter_panel.parquet`)

72,084 visits (Streets dept, 2022–23, rating 1–4, with a `location_id` and a
CBG), 24,056 locations (median 3 visits, 91% surveyed in both years), 46
surveyors, 17 year-months, 1,335 CBGs. Rating 1/2/3/4 = 39,912 / 24,442 /
7,334 / 396; mean 1.559, sd 0.695. Bulk flag 6.4%, casual flag 50.9%.
Attached: production posterior `w_R_cbg` (± sd), 8 kernel `wR_*` columns at the
survey point, exposure and 31 CBG covariates. 40 spatial blocks by weighted
k-d bisection of CBG centroids: 1,644–1,954 visits per block, ~31 CBGs each.

## Field machinery — unit tests pass

Sparse accumulator vs dense recomputation on 300 visits: 1.5e-15 relative
error; `build_W` row sums vs dense: 2.2e-11; kernel mass on an unmasked 50 m
grid integrates to 1.00000 of the event count. Runtime: 202 s for the whole
sweep (6 bandwidths × 4 τ × 7 weight vectors, two denominators each, extras).

Built support: 105,979 cells of 50 m inside a CBG and within 75 m of a pole
(265 km²); rivers, parks, airport and rail yards drop out. Exposure spread
evenly over each CBG's support cells; 0.3% lost to 3 CBGs with no cell.

## The sweep (Spearman of rating with the Gamma-shrunk log field, α = 1)

| form | τ | h=150 | 300 | 500 | 800 | 1200 | 2000 |
|---|---|---|---|---|---|---|---|
| absolute | 120 | .279 | .301 | **.306** | .302 | .296 | .277 |
| edge-corrected absolute (÷ support fraction) | 120 | .276 | .304 | **.310** | .307 | .300 | .276 |
| **ratio (÷ kernel-smoothed exposure)** | 120 | .267 | .331 | .361 | .375 | **.380** | .372 |
| ratio | 240 | .292 | .346 | .367 | .376 | .378 | .372 |

Zero numerators at τ = 120: 11.2% / 2.8% / 0.74% / 0.35% / 0.22% / 0.00%.
τ is flat (120 vs 240 within ±0.006 everywhere), as the placebo predicted.

### Three findings that change the defaults

**1. The ratio form is much better, and for the right reason.** Dividing by
kernel-smoothed SafeGraph exposure lifts the correlation from .306 to .380.
Exposure *alone* has Spearman +0.014 (h=500) / +0.040 (h=1200) with the
rating, so this is a genuine relative-risk normalisation, not the rating's own
dependence on exposure. Alternative denominators do not do it: poles .303/.312,
population .300/.304, street length .325/.331 (vs .306 with none). Reports per
unit of human activity is the quantity the litter survey tracks; this is also
05a's exposure, so β stays on the conditional-on-exposure footing.

**2. The bandwidth optimum depends on the form.** Absolute density peaks at
h = 500 (sd 190 m). The ratio form peaks at h = 1200 (sd 454 m ≈ the median
CBG diameter of 424 m) with a plateau over 800–2000. Decoupling numerator and
denominator bandwidths (`field_decoupled.csv`) shows the numerator drives it:
at any denominator bandwidth the best numerator h is 1200; at any numerator h
the denominator is flat over 800–2000. Same-h is as good as any combination,
so the same-h discipline stands.

| h_num \ h_den | own CBG | 500 | 800 | 1200 | 2000 | none |
|---|---|---|---|---|---|---|
| 500 | .340 | .361 | .364 | .364 | .362 | .306 |
| 800 | .346 | .370 | .375 | .376 | .375 | .302 |
| 1200 | .344 | .371 | .377 | **.380** | .381 | .296 |
| 2000 | .328 | .357 | .365 | .369 | .372 | .277 |

**3. The continuous kernel beats polygon aggregation, including smoothed
polygons.** Same causal τ, evaluated at the visit date, own-CBG exposure:

| CBG-level benchmark | ratio form | raw count |
|---|---|---|
| own CBG | .317 | .256 |
| own + 1st-order queen neighbours | **.346** | .264 |
| own + 2nd-order neighbours | .330 | .222 |
| best point kernel (ratio, h = 1200) | **.380** | |

Pooling a CBG with its neighbours is the best a polygon can do (.346); the
kernel is +0.034 above that and +0.063 above the bare CBG. The gain is not
"any smoothing helps" — it is the distance weighting and evaluation at the
exact coordinates.

### What this means for the framing

The optimal ratio-form kernel has sd ≈ CBG diameter. The contribution is not
sub-CBG resolution; it is a **continuous relative-risk surface, evaluated at
exact survey coordinates and dates, that outperforms the best polygon-based
alternative by about 10% of the correlation**, plus the deliverable raster.
State it that way.

## Provisional defaults for Phase 6

- Primary field: **ratio form** (÷ kernel-smoothed exposure at the same h).
- h: the blocked CV with the 1-SE rule decides among {500, 800, 1200};
  **provisional default 800** (on the plateau, .375 vs .380, and it keeps more
  reporting contrast for the β extraction: ρ(h) = 0.71 at 800 vs 0.56 at
  1200, 0.85 at 500).
- τ = 120 d fixed; 240 reported as equivalent.
- Gamma shrinkage α = 1; zeros are 0.35% at h = 800, a non-issue.
- Corrected numerators carry β₀ = 0.554 with the kernel ŵ_R (primary) and the
  CBG posterior (comparison); `w̄` for both.
- Extras at (500, 120): deduplicated-pile field (75,941 piles, 35% repeats)
  and the Street Light Outage negative-control field.

Note: `field_summary.json` from this run labels the CBG-benchmark raw-count
column `abs_per_km2`; the code now says `count`. Cosmetic, fixed on next run.

## Outputs

`output/litter_panel.parquet`, `cbg_blocks.parquet`, `support_grid.parquet`,
`visit_fields.parquet` (140 columns), `field_sweep.csv`, `field_decoupled.csv`,
`field_alt_denominators.csv`, `field_summary.json`, `figures/04_field_sweep.png`.
