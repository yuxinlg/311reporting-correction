# Phase 3 — Stage 1 reporting propensity (`src/02_reporting.py`)

## 1a. CBG refit — regression test PASS

Per-CBG D and R counts are identical to 05a's (max |diff| 0). With 05a's exact
MCMC settings (1 chain, 1000/1000, seed 0) every number reproduces:

| | β | 94% interval | a_NT | a_T | σ_R | σ_T | div | n |
|---|---|---|---|---|---|---|---|---|
| 05a | 0.5791 | [0.4698, 0.6860] | −1.077 | −5.264 | 0.620 | 1.488 | 0 | 1335 |
| refit, 05a settings | 0.5791 | [0.4698, 0.6860] | −1.077 | −5.264 | 0.620 | 1.488 | 0 | 1335 |
| **production**, 4 × 4000/8000, block-dense mass on globals | **0.554** | **[0.428, 0.687]** | −1.079 | −5.262 | 0.617 | 1.491 | 0 | 1335 |

**05a's single chain mixes poorly on the T-equation globals**: a_T n_eff 7
(R̂ 1.155), β n_eff 18 (R̂ 1.05), 55% of w_T with R̂ > 1.01. The T equation is
saturated, so a_T trades off against mean(w_T) and β against the projection of
w_T on w_R; only the shrinkage prior pins them. w_R is unaffected (n_eff ≥ 475
even in that run; production vs regression w_R means correlate 0.99996).

Production health: β n_eff 638 / R̂ 1.014, a_T n_eff 553 / R̂ 1.016, all other
globals R̂ ≤ 1.003, w_R R̂ 1.000 (n_eff ≥ 23,039), w_T max R̂ 1.014. The strict
R̂ < 1.01 target is missed by β and a_T; the Monte-Carlo se of β̄ is 0.003.
05a's 0.579 sits 1.5 of its own MC-se above the well-mixed mean. **Downstream
β₀ = 0.554 [0.428, 0.687].** 2,000 thinned joint draws of (β, w_R) are in
`output/reporting_draws.npz`.

corr(w_R, log R/P) = 0.998 on the 1,296 CBGs with R ≥ 5 (0.979 including the
39 with R < 5, where the +0.5 guard makes the raw log rate extreme).

## 1b. Point-level kernel surface

Quartic kernel, h_w = 600 m (sd 227 m), Gamma-shrunk toward the citywide rate
with κ = 1 report, centred on the pole-weighted mean.

| variant | window | denominator | sd at survey pts | Poisson sd | reliability | corr with CBG posterior (centroid / event-avg) |
|---|---|---|---|---|---|---|
| **pre24_naive** (primary) | 2021–23 | all poles × 1,095 d | 0.482 | 0.084 | 0.970 | **0.800 / 0.786** |
| pre24_atrisk (the plan's form) | 2021–23 | non-LED pole-days | 0.452 | 0.084 | 0.965 | 0.794 / 0.780 |
| full_atrisk | 2021–25 | non-LED pole-days | 0.442 | 0.073 | 0.972 | 0.823 / 0.797 |
| naive (05a-equivalent) | 2021–25 | all poles × 1,826 d | 0.462 | 0.073 | 0.975 | 0.836 / 0.812 |

Bandwidth: reliability 0.952 / 0.970 / 0.978 / 0.982 / 0.988 for h_w = 400 /
600 / 800 / 1000 / 1500; agreement with the CBG posterior 0.83 / 0.79 / 0.74 /
0.70 / 0.62 (event-averaged). Noise is not the constraint; contrast is. 600 m
is the default, 400 m the sensitivity variant.

### Why the primary is not the pole-time-at-risk form

Outage reports 2024–25 relative to 2021–23, annualised, by LED-share quartile
of the 2025 inventory:

| LED-share quartile | mean LED share | reports 2021–23 | reports 2024–25 | ratio (annualised) |
|---|---|---|---|---|
| Q1 low | 0.45 | 11,130 | 3,869 | 0.52 |
| Q2 | 0.59 | 15,209 | 5,042 | 0.50 |
| Q3 | 0.66 | 15,589 | 5,239 | 0.50 |
| Q4 high | 0.75 | 14,515 | 5,363 | 0.55 |

Reports halved everywhere; the most-converted quartile fell the *least*. LED
conversion did not reduce outage reports, so "non-LED pole-days at risk" is a
mis-specified denominator: it inflates the rate wherever 2023 conversions
removed pole-days (pre24_atrisk vs pre24_naive correlate 0.963). The pre-2024
window is kept because it coincides with the survey calibration window and
sidesteps 2024–25 anyway.

**The PSIP "contamination" 05a flagged is not contamination.** corr(w_R,
led_share) is +0.284 for the CBG posterior and +0.246 for the pre-2024 kernel
surface: the association predates the conversion. The rollout went to
high-complaint areas (or shares their geography); the proxy is not distorted
by it. The citywide halving of outage reports in 2024–25, and the 2022→2023
drop (21.7k → 12.9k at 10% converted), are something else, absorbed by the
centring.

### Kernel vs polygon

Agreement with the CBG posterior is 0.79–0.83, not the 0.998 the plan quoted
(that figure is the CBG posterior vs the CBG raw rate, which reproduces). The
kernel has less spread (sd 0.46 vs 0.59) because a 600 m kernel averages across
CBG boundaries; agreement rises monotonically as h_w shrinks toward the CBG
scale. Both are estimates of the same latent field at different resolutions
and both enter Phase 6.

## Outputs

`output/reporting_cbg.parquet` (per CBG: counts, production posterior, kernel
ŵ_R at the centroid for every variant), `reporting_draws.npz`,
`reporting_summary.json`, `wR_events.parquet` (116,772 reports),
`wR_survey_xy.parquet` (73,350 distinct survey coordinates),
`wR_raster.npz` (100 m, 36,931 cells), `figures/02_wR_surface.png`,
`figures/02_wR_shrinkage.png`. Shared kernel code in `src/kernels.py`.
