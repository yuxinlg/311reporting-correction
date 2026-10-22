# Litter Index: what changed with the method (exploratory)

Figures and tables in this folder; script `src/eda_litter_index.py`. All scores are put on the survey's 1-4 rating scale
with r = 1.5 x published - 1 for the old years (exact inverse of the 2024-25 formula; see the script header).

## What the files hold

| file | years | unit | n per year | score field |
|---|---|---|---|---|
| Litter_Index_Blocks | 2017, 2018 | street segment (hundred block) | 39,754, 39,849 | HUNDRED_BLOCK_SCORE, lattice 0 / 1.33 / 2 / 2.67 / 3.33 / 4 |
| Litter_Index_Neighborhoods | 2017, 2018, 2023, 2024, 2025 | division | 1,686, 1,686, 1,700, 1,693, 1,703 | division_score (all years); streets_division_score, blocks_averaged (2023-25) |
| litter_surveys_raw (streets) | 2022, 2023 | visit | 27,901, 44,314 | Litter Rating 1-4 |

The block file stops at 2018 although the portal advertises blocks 2017-2025; the new method has no public block layer here.

## Scoring

- Old method: one integer rating per block, published as (2/3)(1 + r). 98% of 2018 blocks sit exactly on that lattice (95% in 2017, so 2017 averaged more than one rating per block more often). 5.0% of blocks are grey (score 0, not scored).
- New method: the division score is the mean visit rating over `blocks_averaged` visits; in 2024 and 2025 division_score = (2/3)(1 + streets_division_score) for every division. In 2023, 54% of divisions sit above that line (an extra component, partly the parks survey), so 2023's division_score is not comparable with 2024-25 without using streets_division_score.
- Level: the citywide division mean on the 1-4 scale is 1.79 (2017), 1.87 (2018), 1.57 (2023), 1.32 (2024), 1.15 (2025); the spread (sd across divisions, same scale) falls from 0.58 in 2018 to 0.23 in 2025, and 64% of divisions score below 1.1 in 2025 (vs 6% in 2018). Under the new method the scale collapses toward 1 (fig 1C, fig 5).
- Whether that is cleaner streets or a drifting use of rating 1 the index cannot say, but the rating itself moves: 44% of 2022 visits rated 1 vs 62% in 2023 (fig 1B), and the mean rating swings by half a point across months within a year (fig 1D).

## Ranks across years

- Division ranks: Spearman 0.78 between 2017 and 2018, 0.68 between 2018 and 2023 (across the change), but only 0.43 between 2023 and 2024 and 0.36 between 2024 and 2025 (fig 2). The new method is less stable year to year than the change itself.
- Within the old method, segment-level scores agree weakly between 2017 and 2018 (Spearman 0.47).

## Sample locations

- Same hundred blocks: 99.0% of 2022-23 survey visits carry an address that is a 2018 hundred block, and 95% (2022) / 91% (2023) of the 2018 hundred blocks were visited. The street-class mix of visits matches the old block census (fig 3D). Locations did not change; the unit did (one score per block vs repeated visits averaged per division).
- The 2018 blocks not revisited in 2023 were cleaner in 2018 (local streets: mean 1.42 vs 1.77; coverage.csv), a small selection toward dirtier blocks.
- Sample size per division: median 21 blocks in 2018 (all streets), 28 / 25 / 43 blocks averaged in 2023 / 2024 / 2025 (fig 3A-C). 2025 nearly doubled the sample.
- Division polygons: 9% of divisions have a different polygon in 2023-25 than in 2017-18; 17 divisions are new in 2023 and 10 old ones are missing in 2024.

## Fields vs score

- Street class (fig 4A) and block length (fig 4B) relate to the score the same way under both methods: class gaps stay within 0.2, with collectors and minor arterials above local streets in every year, and the length profile keeps its shape (dirtiest around 150-200 m, the typical rowhouse block). The change shifts the level, not these relations.
- Blocks averaged per division vs the division score (fig 4C): Spearman 0.01 (2023), 0.20 (2024), 0.04 (2025); sampling intensity is not driving the scores.

## Caveats

- The 1-4 conversion of the old scores assumes the old block score is (2/3)(1 + rating); the lattice supports it, but 1-2% of old blocks score 4.0, which that formula cannot produce from a 1-4 rating.
- The published 2023 streets_division_score correlates 0.87 with the mean of the 2023 survey CSV rows by division, not 1.0: the published index uses a different window or extra records.
