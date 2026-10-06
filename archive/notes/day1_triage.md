# Day-1 triage — results and gate decisions

Script: `src/00_triage.py` (runs in ~2 min). Outputs: `output/triage/`.
Sample: 72,084 Streets-department survey visits, 2022–23, with a 1–4 rating,
a `location_id` and a CBG (42 rows without `location_id` and 89 outside every
CBG dropped). 116,772 geocoded Illegal Dumping reports 2021–25. `w_R` for the
preview is 05a's reference posterior mean per CBG (Phase 3 refits it).

Kernel: compact quartic `K(u) = (1-u²)²` with radius `h`; **sd = h/√7 ≈ 0.38 h**.
Causal exponential in time, `τ` in days. Unit test: sparse kernel sums match a
dense recomputation on 300 random visits to 3e-16 relative error.

---

## T1 Placebo — past vs future 90-day count. **Ties.**

Joint OLS of rating on `log1p(past-90d)` and `log1p(future-90d)`, CBG-clustered.

| field | FE | past | future | past − future (t) |
|---|---|---|---|---|
| kernel h=200 | none | 0.116 | **0.154** | −0.038 (−2.2) |
| kernel h=200 | month | 0.103 | 0.095 | +0.009 (0.6) |
| kernel h=200 | month + location | 0.012 | 0.026 | −0.014 (−0.9) |
| kernel h=200 | month + surveyor + location | 0.029 | 0.023 | +0.005 (0.5) |
| count h=300 | none | 0.085 | **0.102** | −0.017 (−1.1) |
| count h=300 | month | 0.087 | 0.057 | +0.029 (2.1) |
| count h=300 | month + location | 0.011 | 0.005 | +0.007 (0.5) |
| count h=300 | month + surveyor + location | 0.029 | 0.006 | +0.023 (2.5) |

Without FE the *future* count predicts the rating better than the past count.
With location FE the within-location coefficients shrink by ~85% and past and
future are indistinguishable in 6 of 8 specs (two count-based specs reach
t ≈ 2–2.5, not robust). Within-location Pearson of rating with log λ̂ ranges
−0.003 … +0.024 … −0.033 across τ = 15 … 480 d (plan's Test 1 reproduced).

**Decision:** the paper is spatial. τ is a pooling window, fixed at 120 d
(sweep 60–480 as robustness, expect flat). Report the null prominently.

## T2 Spline / top-decile turnover

Decile means of rating by local dumping intensity:

| field | d1 | … | d7 | d8 | d9 | d10 |
|---|---|---|---|---|---|---|
| plan's construction: log1p past-90d count, 300 m | 1.254 | | 1.765 | 1.763 | 1.701 | (9 bins) |
| all-time count within 300 m | 1.153 | | 1.788 | 1.802 | 1.783 | **1.691** |
| kernel h=500 τ=120, raw | 1.170 | | 1.793 | 1.782 | 1.748 | 1.743 |
| kernel h=500 τ=120, corrected β₀=0.579 | 1.157 | | 1.761 | 1.772 | 1.751 | **1.812** |
| kernel h=200 τ=120, raw | 1.174 | | 1.704 | 1.755 | 1.770 | 1.776 |
| kernel h=200 τ=120, corrected | 1.169 | | 1.705 | 1.763 | 1.764 | **1.827** |

The turnover is real for hard counts and for the h=500 kernel, milder for the
h=200 exponential kernel (plateau). The 05a correction lifts the top decile in
both kernels and removes the negative slope on the top quintile of the spline
(3% → 0% of grid points). `g` must be a spline (4–5 df); the linearization test
(plan replacement #1) has preliminary support.

## T3 Variance budget and identification

| FE set | R² of log λ̂ (h200) | R² of rating |
|---|---|---|
| month-of-sample (17) | 0.144 | 0.100 |
| surveyor (46) | 0.137 | **0.274** |
| month + surveyor | 0.266 | 0.328 |
| location (24,056) | 0.854 | 0.477 |
| CBG (1,335) | 0.703 | 0.270 |

Surveyor FE explain 27% of rating variance — as much as CBG identity and more
than month. The surveyor–location bipartite graph is **a single connected
component** (all 72,084 visits); 40 spatial blocks each see 8–27 surveyors
(median 14), the dominant surveyor holds a median 25% (max 45%) of a block.
Surveyor and location effects are separately identified; surveyor FE are
mandatory and safe to include.

## T4 Geocoding atoms and dedup

| years | n | distinct coords | share on shared coords | max at one coord |
|---|---|---|---|---|
| all | 116,772 | 67,006 | 59.2% | 167 |
| 2022 | 24,024 | 15,440 | 51.7% | 77 |

Nearest-other-atom spacing: p10 3 m, p50 11 m, p90 42 m — the atoms are
addresses, densely spaced. What matters is how many atoms a kernel sees at a
survey point:

| quartic radius h | median atoms within h | share of visits with < 5 atoms | share with λ̂ = 0 |
|---|---|---|---|
| 100 | 9 | 32.5% | 19.3% |
| 150 | 20 | 19.9% | 11.2% |
| 200 | 38 | 13.1% | 6.7% |
| **300** | 85 | 5.6% | 2.8% |
| **500** | 235 | 1.3% | 0.7% |
| 800 | 616 | 0.4% | 0.4% |

Space-time dedup (50 m / 14 d): 116,772 events → 75,923 piles; **35% of
events are repeats**, so the dedup test (plan replacement #3) is meaningful.

**Decision:** minimum defensible quartic radius ≈ 300 m (sd ≈ 110 m). Below
that the kernel measures the geocoder and zeros exceed 5%.

## T5 Endogenous survey routing — **null**

311 dumping reports within 200 m per visit-week: 0.351 in the 4 weeks before
the visit, 0.353 in the 4 weeks after, 0.335 at |lag| ≥ 8 weeks. 30-day
past/future ratio 1.000 locally vs 1.005 citywide; city-adjusted per-visit
difference +0.010 (cluster SE 0.024, t = 0.4). Crews were not dispatched in
response to 311 complaints; the validation design stands.

## T6 Bandwidth, the CBG benchmark, and β̂

### Bandwidth sweep (Spearman of rating with log λ̂, τ = 120)

| quartic radius h | 100 | 150 | 200 | 300 | **500** | 800 | 1200 | 2000 |
|---|---|---|---|---|---|---|---|---|
| ≈ Gaussian sd | 38 | 57 | 76 | 113 | **189** | 302 | 454 | 756 |
| absolute density | .264 | .279 | .288 | .301 | **.306** | .302 | .296 | .277 |
| ratio form (÷ CBG exposure density) | | | .308 | .326 | .338 | **.340** | .336 | .319 |
| CBG-pooled benchmark, same τ | per km² **.289** | | | | per exposure **.320** | | | |

Two corrections to the plan:

1. **The plan's "h = 200 m optimum" is in Gaussian-sd units.** With the
   quartic kernel the plan itself prescribes, the interior optimum is at
   radius ≈ 500 m (sd ≈ 190 m), and the +0.307 → +0.276 decline the plan
   reports appears here between radius 500 and 2000 m. Default `h = 500 m`,
   sweep {300, 500, 800, 1200}. Never below 300 m (T4).
2. **The gain over CBG aggregation is real but modest**: +0.017 (absolute,
   .306 vs .289) and +0.020 (ratio, .340 vs .320) in Spearman — about 6% of the
   correlation. The optimal kernel sd (~190 m) is about the median CBG
   *radius* (212 m), not "well below the CBG diameter". The honest pitch is a
   continuous surface evaluated at exact coordinates with no boundary
   artefacts, not a dramatically finer resolution. The ratio form beats the
   absolute form at every bandwidth (+0.03), confirming the plan's preference.

Level reconciliation: CBG-mean rating vs CBG-mean log λ̂ (h=500): Spearman
+0.465 absolute, **+0.602 ratio form** (old plan: +0.58).

### Specification curve for β̂ = −c/b (OLS on the raw rating, 60 specs)

| FE | regressor | h=150 | 200 | 300 | 500 | 800 |
|---|---|---|---|---|---|---|
| month | w_R(s) | 1.59 | 1.41 | 1.21 | 1.05 | 0.95 |
| month | **w̄** | 1.75 | 1.62 | 1.51 | 1.60 | 1.81 |
| month + surveyor | w_R(s) | 0.99 | 0.86 | 0.72 | 0.59 | 0.50 |
| month + surveyor | **w̄** | 1.12 | 1.02 | 0.93 | 0.98 | 1.06 |

ρ(h) = slope of w̄ on w_R(s): 0.81, 0.78, 0.70, 0.58, 0.46 for h = 150 … 800.
The w_R(s) rows fall with h exactly as attenuation β·ρ(h) predicts; the w̄ rows
are flat in h, as they should be. Exposure denominator moves β̂ by ±0.1.

Median β̂ 1.23, p10–p90 [0.84, 1.62]; 5% of specs inside 05a's [0.470, 0.686];
median delta-method SE 0.13. **In the preferred specs (w̄, month + surveyor FE)
β̂ ≈ 1.0–1.3, about twice 05a's 0.579.** Specification spread (0.8) is six
times the sampling SE, as the plan predicted.

### Bulk-item outcome (casual-litter OVB check)

| FE | regressor | h | β̂ rating | β̂ bulk |
|---|---|---|---|---|
| month + surveyor | w̄ | 200 | 1.02 | 1.11 |
| month + surveyor | w̄ | 500 | 0.98 | 1.19 |
| month + surveyor | w_R(s) | 500 | 0.59 | 0.66 |

The dumping-specific bulk flag gives the **same or larger** β̂. The excess of
β̂ over 05a is *not* casual-litter omitted-variable bias. Remaining candidates:
(i) the cleanup/response channel — no longer diagnosable since case closure is
out of scope by decision; (ii) `w_R` proxying civic engagement that lowers
litter through channels other than reporting (same sign as (i)); (iii) 05a's β
being pulled by its LogNormal(0, 0.5) prior on a saturated equation. Treat 0.58
and ~1.1 as a bracket; the profile likelihood (Phase 6) with the ordinal link
and spline `g` is the headline, and the spec curve goes in the paper.

Dumping-311 predicts the rating more strongly in standardized units (b_std
0.25 vs 0.11) but bulk more strongly in relative terms (+42% vs +11% per log
unit; bulk has a 6.4% base rate). The differential-outcome test (plan
replacement #4) needs a proper binary link before it says anything.

---

## What changes in the plan

| plan item | change |
|---|---|
| default `h = 200 m`, sweep {150 … 800} | **`h = 500 m` quartic radius** (sd ≈ 190 m), sweep {300, 500, 800, 1200}; floor 300 m |
| "h ≈ 200 m genuinely resolves within CBG" | soften: optimal sd ≈ CBG radius; gain over CBG pooling ≈ +0.02 Spearman (6%) in both forms |
| time kernel | pooling window, τ = 120 d fixed; sweep reported as flat; placebo null reported |
| open-case stock, closure-time controls | dropped (closure out of scope, 2026-09-30 decision) |
| zeros / Gamma shrinkage | at h = 500 only 0.7% zeros; shrinkage becomes a robustness item, not a necessity |
| β̂ expectation | ~1.0–1.3 vs 05a 0.58; bulk check rules out casual-litter OVB; report as a bracket |
| endogenous routing (Risk 3) | tested, null; no restriction to first visits needed |
| surveyor FE | mandatory (27% of rating variance), identified (connected graph) |
