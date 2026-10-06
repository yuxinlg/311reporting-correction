# Point-level 311 correction, calibrated to the Litter Index

**Supersedes** Parts 2–3 of `go-through-replication-file-05-05a-corr-twinkling-candy.md`
(the CBG × month NB-GLM, BYM2, and the three-equation joint model). Part 1
(what 05/05a implement), Part 4 (new repo) and the data provenance stand.

---

## Context

The earlier plan analysed everything at **CBG level**. The PI wants a
**point-level** deliverable instead:

1. Fit 311 dumping as a **continuous space-time intensity field** with an
   explicit **spatial kernel and time kernel**, bandwidths set by hand.
2. Push it through a **sigmoid** onto the **Litter Rating 1–4 scale**.
3. **Validate against the litter survey.**
4. Allow **CBG-level spatial covariates** in.

### Your open question, answered

> *311 has its own spatial/time kernel, different from the litter survey's — how
> can the litter survey validate it?*

The kernel is **not a model of how the survey sampled**. It is only the operator
that turns the 311 point pattern into a **field** λ̂(s,t) defined everywhere.
Once it is a field you read it off at any coordinates you like — including the
exact (location, date) of each of the **73,780 survey visits**, every one of
which carries `x`, `y` and a `survey_start` timestamp. Nothing has to match.
Your PI is right, and that is the whole answer to the *spatial* half.

The *temporal* half was supposed to work the same way — 311 is a **flow**, litter
is a **stock**, and the time kernel is the flow→stock converter:

$$\text{stock}(s,t)=\int_{-\infty}^{t}\text{flow}(s,u)\,e^{-(t-u)/\tau}\,du$$

**I tested this before writing the plan, and it does not hold in this data.**
See the next section. It is the most important thing in this document.

---

## ⚠ Pre-tested: the spatial kernel works, the time kernel does not

Two independent tests, both run against the real files.

**Test 1 — does 311 timing predict litter timing?** Causal exponential kernel,
`h = 200 m`, Streets-department surveys, λ̂ evaluated at each visit's own
(location, date):

| τ | cross-sectional Spearman | **within-location** Pearson |
|---|---|---|
| 15 d | +0.253 | −0.004 (p = 0.24) |
| 30 d | +0.271 | +0.001 (p = 0.77) |
| 60 d | +0.291 | +0.014 |
| **120 d** | **+0.307** | **+0.031** |
| 240 d | +0.309 | +0.014 |
| 480 d | +0.297 | −0.047 |

**Test 2 — the placebo.** At `h = 300 m`, correlation of the rating with the
**past** 90-day dumping count is **+0.263**; with the **future** 90-day count it
is **+0.269**. A future report cannot cause present litter.

**These two agree, and the conclusion is unavoidable: essentially all of the
measured association is time-stationary spatial structure.** τ is not identified.
The stock-from-flow motivation has no empirical support here. A within-location
design finds nothing; a future-window placebo does as well as the causal one.

Now the good news — the **spatial** sweep, at τ = 120 d:

| `h` | 100 m | 150 m | **200 m** | 300 m | 500 m | 800 m |
|---|---|---|---|---|---|---|
| Spearman vs rating | +0.300 | +0.306 | **+0.307** | +0.304 | +0.294 | +0.276 |

**This is the empirical case for the redesign.** A clean interior optimum at
**h ≈ 150–200 m**, *well below* the 424 m median CBG diameter, degrading
monotonically as you coarsen — ~11% of the correlation lost by h = 800 m
(≈ 2 CBGs). Aggregating to CBG demonstrably throws away signal. You can now show
that rather than assert it.

### What this means for the paper

This is a **spatial-resolution** paper. The contribution is sub-CBG correction
and a continuous surface; the time kernel is a *pooling-window* parameter with an
optimum around 120–240 days, not a dynamics model, and τ must **not** be
interpreted as a litter half-life.

**This needs to reach the PI before implementation starts**, because the "spatial
*and* time kernel" framing of the instruction does not survive. The work is still
very much worth doing — it just has one axis, not two.

**Day-1 gate.** Before building any pipeline, rerun the placebo properly, with
month-of-sample and location effects absorbed (~10 lines). Then:

- **If the placebo still ties** → fix τ at a single defensible pooling window
  (or use the observed open-case stock, below), report the null prominently, and
  present the paper as spatial. A reviewer who thinks of this test and finds it
  missing will assume you hid it.
- **If a properly-specified test separates past from future** → the temporal
  story is live after all and τ becomes a real result.

---

## What the data supports (verified against the files)

| fact | value | why it matters |
|---|---|---|
| Dumping points, 2021–2025, geocoded | **116,772** | matches 05a's `T = 116,751` |
| …usable under a causal kernel (surveys end 2023-11) | **71,970** | **38% of events never enter the calibration**; 2024–25 surface is extrapolation |
| Survey visits with coords **and** timestamps | **73,780** (2022-03 → 2023-11) | point-level in space *and* time |
| Survey → nearest dumping report | median **37 m** | 311 far denser than the survey grid |
| Median CBG equivalent diameter | **424 m** (p25 334, p75 580) | h ≈ 200 m genuinely resolves *within* CBG |
| **Distinct coordinates among 24,024 2022 points** | **15,440**; 51.7% shared; **max 77 at one coordinate** | the process has **atoms** — address/intersection snapping. Hard floor of `h ≈ 150–200 m` |
| Repeat visits per location | median **3**; 23,020 locations ≥ 2 | 34.9% of repeat locations show **zero** rating variation |
| Between-location share of Var(rating) | **48.4%** | random-row CV leaks outright |
| Case closure times (2021–23) | median **15 d**, p90 **97**, p95 **148** | **heavy-tailed** — exponential predicts p90 = 50 vs 97 observed |
| `closed_dat` populated | **99.6%** | lets you build the *observed* stock instead of assuming τ |
| Litter Rating distribution | 1:40,682 2:25,159 3:7,519 **4:420** | top cutpoint rests on 420 obs |
| Surveyors / effective year-months | **56** (top 5 = 48%) / **17** (Jan-23 and Mar-23 empty) | nuisance effects; a winter gap the kernel never crosses |
| Apparent seasonality | 0.620 raw → **0.280** after within-location demeaning | **~55% of "seasonality" is compositional** — where they surveyed, not when |
| Outage counts by year | 21,908 → 8,281 | PSIP LED rollout, spatially uneven |
| **Effective independent units** | **~1,335**, not 73,780 | **every naive t-stat inflated ~7.4×** |

### The effect size — lead with this

From the 05a posterior: `Var(βw_R) / Var(βw_R + w_T)` = **5.4%**. The correction
explains 5% of cross-CBG variance in log dumping rate; `w_T` explains 95%. In
outcome units, the correction moves the predicted rating by **~0.05 points per sd
of reporting propensity**, on a scale whose total sd is 0.70 — about **0.07 sd**.

Frame the contribution as "small, measurable, and directionally validated against
ground truth," not "a correction that redraws the map." Say it before the referee
does.

---

## Decisions taken (from your answers)

| question | choice |
|---|---|
| Where the correction enters | **Both** — per-event weights *and* a covariate — and compare |
| Target scale for the sigmoid | **Litter Rating 1–4** |
| ŵ_R resolution | **Both** — CBG posterior refit *and* a point-level kernel surface |
| Old CBG-level plan | **Retired.** Only the streetlight equation survives, as the β anchor. |

---

## The model

### Stage 1 — reporting propensity ŵ_R

**(1a) CBG posterior.** Refit 05a's two-equation Poisson on pooled counts
(`$SRC/replication/05a_streetlight_decomposition.py:407-437`). **Regression test:
β = 0.5791 [0.4698, 0.6860]**, `a_NT = −1.077`, `a_T = −5.264`, `σ_R = 0.620`,
`σ_T = 1.488`, 0 divergences, n = 1,335. Nothing downstream is meaningful until
this passes. Refit rather than read `data/ref_05a_output/` — the README declares
it reference-only.

> **Know what this β is.** The `T` equation has one free `w_T,i` per observation
> `T_i` — it is **saturated**, so β is identified *only* through the shrinkage
> prior `w_T ~ N(0, σ_T)`. It is, to within Poisson weighting, the cross-sectional
> regression slope of log dumping rate on log outages-per-pole: the plain OLS
> slope is **0.4975**, which divided by the EIV reliability 0.924 gives **0.538**
> against the posterior **0.5791**. It therefore absorbs any genuine correlation
> between true dumping and civic engagement. Also note `β ~ LogNormal(0, 0.5)`
> cannot express β ≤ 0. State the identifying assumption plainly — *true breakage
> per pole is uniform citywide, so cross-CBG variation in outages-per-pole is
> reporting propensity* — and defend it.

**(1b) Point-level kernel surface**, with the PSIP confound fixed structurally.
Do **not** use a naive pooled outages/poles ratio: outages fell 62% over the
window under a spatially uneven rollout, and 05a's own validation already flagged
`corr(w_R, led_share) = 0.285` as *"conversion geography contaminates proxy."*

`Street_Poles.geojson` carries `light_date` and `bulb_type`, so build a genuine
**incidence rate with pole-time at risk**:

$$\hat w_R(s)=\log\frac{\sum_{j\in\text{outages, pre-relight}}K_{h_w}(\lVert s-s_j\rVert)}
{\sum_{p\in\text{poles}}K_{h_w}(\lVert s-s_p\rVert)\cdot \text{days}_p^{\text{non-LED}}}
\;-\;\overline{(\cdot)}_{\text{pole-weighted}}$$

This removes the PSIP confound rather than hoping pooling averages it away, and
it answers the objection 05a raised against itself.

**Use a separate, larger bandwidth `h_w ≈ 600–1,000 m`.** At `h = 300 m` a kernel
holds only ~30–60 outages, so Poisson noise alone gives `sd(log ratio) ≈ 0.13–0.18`
against a signal sd of 0.595 — ~5–10% attenuation. The reporting field is a
slow-varying neighbourhood trait and can afford the extra smoothing; conflating
`h` and `h_w` is a design error.

*Cross-check:* kernel ŵ_R at CBG centroids vs `w_R_mean_sl` should correlate
highly (05a reports `corr(w_R, log outages-per-pole) = 0.998`).

### Stage 2 — the 311 intensity field

$$\hat\lambda(s,t)=\frac{\sum_{j}m_j\,K_h(\lVert s-s_j\rVert)\,K_\tau(t-t_j)}
{\sum_{k}K_h(\lVert s-s_k\rVert)}$$

- **Use the ratio (relative-risk) form** with a kernel-smoothed exposure
  denominator (SafeGraph stops, street length, or population) at the **same** `h`.
  Edge effects cancel to first order because numerator and denominator truncate
  identically (Kelsall & Diggle 1995) — and it keeps β on the same
  *conditional-on-exposure* footing as 05a's, which the absolute-density version
  silently breaks (`corr(w_R, log E) = −0.242`). Report edge-corrected absolute
  density as the robustness variant.
  - If you do use absolute density: the city polygon is the **wrong support** —
    it includes the two rivers, Fairmount Park, the airport and rail yards, where
    dumping reports are structurally absent, so it *under*-corrects near all of
    them. Use the built/street support instead. Compute `c_edge` by
    FFT-convolving a 10 m raster of the support with `K_h` **once per h**, then
    interpolating; do not integrate polygons per point.
- **Spatial kernel**: **compact-support quartic/biweight** `K(u) ∝ (1−u²)²`,
  not Gaussian. Support is exactly `h`, so the search radius is `h` not `3h`
  (**9× fewer pairs**) and there is *no truncation error*. Default **`h = 200 m`**
  (the measured optimum, and at the geocoding-atom floor), swept over
  {150, 200, 300, 500, 800}. **Do not go below 150 m** — 51.7% of points share a
  coordinate and one holds 77 reports, so a finer kernel measures the geocoder.
- **Temporal kernel**: given the placebo result, do not fit τ as a physical
  parameter. Two defensible options, in preference order:
  1. **Observed open-case stock** — no free τ at all:
     $\hat\lambda_{\text{open}}(s,t)=\sum_j K_h\,\mathbf 1[\text{requested}_j\le t<\text{closed}_j]$.
     This is the literal stock, *measured*. Closure times are heavy-tailed
     (median 15 d, p90 97, p95 148; a matched-median exponential predicts p90 = 50,
     off by 1.95×), so the exponential is the wrong shape anyway.
  2. **Causal exponential** as the PI specified, swept over
     {15, 30, 60, 120, 240, 480} d, reported as a pooling window.
  The causal form matters regardless: a symmetric Gaussian lets future dumping
  predict present litter, and a reviewer will catch that.

**Computation — measured, and much cheaper than it looks.** `cKDTree` on 73,780
queries against 71,970 events: 3.9 M pairs at 200 m (~1 s), 32 M at 600 m (~4 s).
Three multipliers make the sweep trivial:

- compact support (radius `h`, not `3h`) → 9×;
- reduce the spatial search to the **24,816 unique locations**, then expand over
  dates → ~3×;
- cache one **CSR matrix `W_h` per bandwidth**, discretize query dates to days
  (~1,000 distinct), and every evaluation is a sparse matvec
  `λ̂[·,d] = W_h @ (v_{τ,d} ⊙ m)`.

Full 6 × 6 sweep: **under 15 minutes, under 2 GB**, and the β profile below is
nearly free because only `m` changes. No FFT needed for the fit — save it for the
final publication raster, where a first-order IIR recursion
`A_t = e^{−1/τ}A_{t−1} + n_t` over daily count rasters streams the whole window in
O(1) memory. **Unit test:** verify against a brute-force dense computation on 500
random query points, to machine precision.

### Stage 3 — the sigmoid, calibrated on the 1–4 rating

At each survey visit ℓ, at its own location and date:

$$\eta_\ell=a+g\big(\log\hat\lambda(s_\ell,t_\ell)\big)+c\,\bar w_R(s_\ell,t_\ell)
+\xi_{\text{month-of-sample}(\ell)}+\zeta_{\text{surveyor}(\ell)}$$

**Primary link — proportional-odds ordinal logit.** This *is* the PI's sigmoid:
three sigmoids sharing one linear predictor, `P(rating ≥ k) = σ(η − κ_k)`, with a
fitted value on the 1–4 scale as `E[rating] = 1 + Σ_k σ(η − κ_k)`. (Cutpoints
written κ to keep `c` free.) Proper likelihood for an integer outcome, and it
survives the 0.6% top category.

**Also report the literal `ŷ = 1 + 3σ(η)`** the PI described, by NLS, initialized
from the ordinal fit. Be honest about it: it assumes equally-spaced ratings (the
cutpoints will show they are not), is heteroskedastic by construction, and with
420 category-4 observations the predictions will compress into roughly [1, 2.5].
Report the fraction of the 1–4 range actually spanned.

Two link checks that matter: fit a **partial-PO** model and report `c_k/b_k` per
cutpoint (disagreement means the single-sigmoid framing is wrong), and a
**location-scale** ordinal model with scale ~ surveyor (heteroskedastic latent
scale biases the ratio below).

**`g` must be a spline, not a slope.** Mean rating by decile of local count:

```
1.180  1.246  1.352  1.481  1.613  1.716  1.781  1.793  1.743  1.727
                                                          ↑ turns over
```

It rises monotonically through decile 8 then **falls** — plausibly Center City
commercial areas with intensive municipal cleaning, or the highest reporting
propensity, which is exactly what you are correcting. Fit a natural cubic spline
(4–5 df) and plot it.

**Month effects: use month-of-sample FE (17 levels), not calendar-month dummies.**
Over half the apparent seasonality is compositional, and Jan-2023 / Mar-2023 have
zero surveys. State the consequence: this absorbs all citywide temporal variation
in λ̂, so `b` is identified from spatial variation alone — which, given the
placebo, is all there is anyway.

**No location random effect in the β specs.** A location *fixed* effect makes
time-invariant ŵ_R exactly unidentified; a *random* effect is correlated with
ŵ_R by the project's own premise, so it absorbs `c` and shrinks it toward zero
mechanically. Use a **Mundlak / correlated-random-effects** form — location means
of the time-varying regressors alongside within-location deviations — so between
and within contrasts are read separately, and block-bootstrap the SEs.

**CBG covariates** enter by point-in-polygon, piecewise constant. Rename
`cov_cbg`'s land-use column `T` on ingest (collides with 05a's `T_count`).

---

## Estimating β from ground truth

### The correction I have to make to my own first draft

The naive `β̂ = −c/b` with `ŵ_R(s)` as regressor **does not estimate β — it
estimates β·ρ(h)**, attenuated by the smoothing. λ̂ is a kernel functional, and
`log(K * λ_true e^{βw_R}) ≠ log(K * λ_true) + β w_R(s)`. Exactly:

$$\log\hat\lambda_{\text{corr}}(\beta)-\log\hat\lambda_{\text{raw}}
=\log \mathbb E_K\!\left[e^{-\beta w_R}\right]
=-\beta\bar w+\tfrac{\beta^2}{2}\mathrm{Var}_K(w_R)+O(\beta^3)$$

where `w̄` is the **event-kernel-weighted local mean** of `w_R`. So the regressor
must be `w̄`, not `w_R(s)`. With `w̄ = ρ·w_R + v` and ρ < 1, using `w_R` gives
`plim(−c/b) = β·ρ(h)`. At h ≈ 600 m against a 424 m median CBG you could lose
40–60% of β. **Fix: use `w̄(s,t)` as the regressor, and report ρ(h)** — the OLS
slope of `w̄` on `ŵ_R` — at each bandwidth, as a diagnostic for how much
reporting contrast the smoother destroyed.

With that fixed, thinning gives `log λ̂_raw = c₀ + βw̄ + log Λ`; the survey
observes Λ, not reports, so `η = a + b log Λ`, hence `b = θ`, `c = −θβ`, and
`β̂ = −c/b`. **The ratio cancels θ and the link's scale exactly** — under
proportional odds `b` and `c` share the same index, so rescaling η rescales both
identically. That part of the design is sound.

### Headline estimator: profile likelihood, not the ratio

```
for β in linspace(-0.2, 2.0, 111):
    m_j(β) = exp(-β · ŵ_R(s_j))
    λ̂_corr = W_h @ (v_{τ,d} ⊙ m(β))          # one sparse matvec per distinct date
    ℓ(β)   = max over (a, g, month, surveyor) of the ordinal log-likelihood
β̂ = argmax ℓ;  CI = {β : ℓ(β) ≥ ℓ(β̂) − 1.92}, critical value recalibrated by
                spatial block bootstrap
```

This uses the exact nonlinear dependence (the weights sit *inside* the kernel
sum) rather than a first-order linearization, has no ratio pathology, works with
a nonlinear `g`, and costs milliseconds per β. Keep `−c/b` **with w̄** as the
closed-form check; they must agree when `g` is linear. Report a **Fieller**
interval for the ratio alongside the delta-method one.

### Expect specification, not sampling, to dominate

A crude preview (log1p count within 300 m, CBG-level `w_R`, OLS on raw rating):

| spec | b | c | β̂ = −c/b |
|---|---|---|---|
| no FE | +0.141 | −0.170 | **1.212** |
| + month-of-sample FE | +0.118 | −0.158 | **1.340** |
| + month + surveyor FE | +0.145 | −0.113 | **0.778** |
| *05a posterior* | | | *0.579 [0.470, 0.686]* |

The **sign is right**, which genuinely supports the mechanism. But all three
exceed 05a's upper bound, and surveyor FE alone moves β̂ by 42%. Inflated
`se(c) ≈ 0.03` is an order of magnitude below the 0.56 spread across specs.
**Plan to report a specification curve, not a point estimate.** Note also that
attenuation (F1 above) predicts β̂ should come in *below* β — it comes in
*above*, so the confounds below dominate.

### Two threats to β̂, both real

**Cleanup/removal — the deepest one.** A 311 report triggers a service response
(median 15 d). High-`w_R` areas generate more reports *and* plausibly get faster
service; both make them look **cleaner** at survey time, loading onto `c` with the
**same sign** as reporting inflation. This biases β̂ **up** — exactly what the
table shows. Mitigations, all with data in hand: split events by `closed_dat`
into still-open vs already-closed at survey time and estimate separate
coefficients (under pure reporting inflation they should be similar; under
removal, closed events predict *less* litter); control for kernel-smoothed local
median closure time; report the open-stock specification alongside.

**Casual-litter OVB.** The 1–4 rating is mostly casual litter, and civic-minded
high-`w_R` neighbourhoods plausibly have less of it, inflating `|c|`. Check on
the bulk-item flag (`trash_bags`, `construction`, `oversized`, `furniture`,
`tires`, `vehicular`; 6.97% of visits), which is dumping-specific. The 1–4 rating
stays the deliverable scale; bulk is the check.

**And soften one claim:** β̂ is independent of 05a's *likelihood*, not of its
*data* — β₀ = 0.579 was estimated using the same dumping counts, and both are
cross-sectional projections onto the same `w_R`. Agreement is informative;
**disagreement does not cleanly falsify either**, because they rest on different
assumptions.

---

## Spec ladder — revised

| | spec | role |
|---|---|---|
| **S0** | `a + g(log λ̂_raw) + month + surveyor` | baseline |
| **S1′** | `a + b·log λ̂_raw + c·w̄_R + month + surveyor` | closed-form β̂ = −c/b |
| **S2** | `a + g(log λ̂_corr(β₀))` | plug-in 05a β; must beat S0 out of sample |
| **SP** | profile likelihood over β | **headline** |
| ~~S3~~ | — | **dropped, see below** |
| **S4** | `+ γ′X_CBG` | robustness on `b` only — **never with β** |

**Why S3 (`c ≈ 0` on the corrected field) is dropped.** To first order
`log λ̂_corr = log λ̂_raw − β₀w̄`, so S1 and S3 span the same 2-D column space:
identical log-likelihood, identical df, identical fitted values, with
`c₃ = b·ρ·(β₀ − β̂)`. It is a reparameterization of S1, not independent evidence —
reporting both is double-counting, and any AIC/LRT comparison between them is
degenerate. It also cannot detect a *misspecified correction form*, is
underpowered (you would need `se(c₃) < 0.013` to reject a 0.3 discrepancy; honest
SEs give ≈ 0.029), and with a location RE it is shrunk to zero mechanically. My
first draft had this as a headline test; it is not one.

**Four genuinely informative replacements:**

1. **Linearization test.** Does correcting *straighten* the top-decile turnover
   above? Plot the spline against `log λ̂_raw` and `log λ̂_corr`. Visual,
   non-circular, persuasive.
2. **Overidentification.** Re-add a reporting proxy **not used** to build `w_R` —
   potholes per km of street, abandoned vehicles per parking space, or ACS
   tenure. Residual explanatory power is a real completeness failure.
3. **Dedup test.** λ̂ built on space-time deduplicated "piles" (~50 m / 14 d)
   should need a *smaller* `|c|`, because repeat calling at one address **is**
   reporting propensity. Independent, and a partial correction in its own right.
4. **Differential outcome.** Dumping-311 should predict the **bulk** outcome
   better than the **casual-litter** outcome, and the correction should improve
   bulk more. If not, you are measuring generic neighbourhood disorder.

**Negative controls — mandatory given this much spatial autocorrelation.** A bare
ΔAIC is uninterpretable without them:

- **Placebo λ̂** from a 311 category with no litter content (potholes; street-light
  outages themselves). `b` should be ≈ 0.
- **Spatially permuted `ŵ_R` with matched autocorrelation** (torus shift or
  Fourier phase randomization) to build a null for the improvement the correction
  buys. Without this you cannot distinguish "the correction helped" from "any
  smooth spatial covariate helps."

---

## Validation design

**Buffered, spatially blocked, grouped, nested CV.**

- **Blocks:** cluster the 1,335 CBGs into ~40 contiguous blocks of roughly equal
  survey count (~1,850 each).
- **Outer:** 10 folds, 4 blocks each — location grouping is then automatic.
- **Buffer:** drop from training anything within **2·h_max** of the test blocks
  (1.2 km at h_max = 600 m). Expect to lose 20–30% of training data; report it.
  Buffered blocked CV is what a spatial-stats reviewer expects.
- **Inner:** tune (h, τ, spec) on the 9 training folds only. Report outer-fold
  performance only — otherwise the headline ΔAIC *and* β̂ are selection-biased.
- **Surveyors as random effects**, so unseen surveyors in a test fold predict at
  the population mean.
- **Metrics:** out-of-fold ordinal log-likelihood per observation, Somers' D, and
  a calibration curve **aggregated at block level** — row-level calibration looks
  excellent under spatially correlated residuals and is misleading.
- **Second axis:** temporal holdout, train 2022 (27,901) / test 2023 (45,879).
- **Third axis:** leave-one-district-out, with block-level calibration. If the
  model cannot rank unseen districts, the surface is not usable for policy.
- Report the (h, τ) CV surface as a **heatmap with uncertainty** (paired fold
  differences, block bootstrap) and apply the 1-SE rule. Given the placebo, expect
  the τ direction to be flat — say so rather than reporting a point-optimal τ.

**Every SE in the paper must be spatial block bootstrap or cluster-robust at a
scale ≥ h.** With ~1,335 independent units behind 73,780 rows, naive t-stats are
inflated ~7.4× and everything comes out p < 0.001, which will get the whole paper
discounted.

**Propagate two-stage uncertainty.** `ŵ_R` has posterior sd (mean 0.152, max
0.587; reliability 0.924 → ~8% attenuation in `c`, worse in low-pole CBGs) and β₀
has posterior [0.470, 0.686]. Plugging in posterior means and then computing CIs
treats a fitted quantity as data. Refit the calibration over ~200 joint posterior
draws of `(w_R, β)` and pool by Rubin's rules — cheap given the profile
architecture. SIMEX with the per-CBG `w_R_sd_sl` is the lighter alternative.

**Zeros.** With the *causal exponential* kernel (infinite backward support) only
**0.58%** of surveys hit λ̂ = 0 at h = 200 m — a non-issue. But with a **hard
90-day window** it is **28.3%** at h = 200 m and **58.5%** at h = 100 m. So the
choice of temporal kernel shape, not spatial sparsity, drives this. If you use a
hard window or the open-case stock, do **not** use a bare `log(λ̂ + ε)`; use a
Gamma-shrunk posterior mean
`λ̂ = (Σ m_j K K + α)/(denominator + α/λ̄)` with α ≈ 1–5 "prior events", report
sensitivity over α ∈ {0.5, 1, 2, 5}, and carry an explicit `1[λ̂ = 0]` indicator.

---

## Deliverable

A **citywide predicted-litter-index raster** — a 1–4 index at every point in
Philadelphia at ~200 m resolution, from free 311 data. The pitch: cheap continuous
311 reproducing what an expensive field audit would have found, at finer spatial
grain than the CBG aggregation everyone else uses.

Sell it on **spatial resolution and coverage**, not temporal tracking. Two
honesty constraints: the date argument moves the surface very little (a measured
property, not a shortcut), and the 2024–25 surface is **extrapolation** — 38% of
the events and all of those two years sit outside the calibration window.

Be explicit that `λ_corr = λ_raw·e^{−βw̄}` is *"reports that would have been filed
at citywide-average reporting propensity"* — still on the **report** scale, not
physical tonnage. The sigmoid then maps it to 1–4. These are two separable claims
— (1) does the correction improve agreement/ranking against ground truth, and
(2) is the 1–4 mapping calibrated — and each needs its own validation. Conflating
them invites the objection that the sigmoid absorbs whatever the correction got
wrong.

---

## Risks, in the order they will bite

1. **The temporal model already fails a placebo** (measured). Reframe as spatial;
   report the null. Tell the PI first.
2. **Cleanup/removal confound** biases β̂ up and is currently unaddressed.
3. **Endogenous survey routing** — if crews were dispatched in response to 311
   complaints, every correlation here is inflated and the validation collapses.
   Test whether a location's survey *timing* responds to recent nearby 311
   reports; if so, restrict to first visits or the regular route. **Run early.**
4. **Effect size is 5.4% of variance.** Frame it before the referee does.
5. **Geocoding atoms** — 51.7% of points share a coordinate, max 77 at one. Never
   go below h ≈ 150 m.
6. **Bad controls in S4** — `vac_area`, `landcare_area`, `ndvi_mean_4yr` and the
   land-use shares are plausibly *mediators* of dumping. And any X predicting
   reporting propensity competes with `w̄_R` and destroys `−c/b`, so **S4 must
   never be combined with the β extraction**.
7. **Construct validity** — validating an *illegal-dumping* surface against a
   general litter score. The bulk-item outcome is the answer; it is the check
   throughout even though the 1–4 rating is the deliverable scale.
8. Euclidean rather than network distance: a stated limitation.

---

## Code layout

```
src/
  01_build_panel.py     EXTEND — also emit point tables (below)
  02_reporting.py       Stage 1: CBG refit (β anchor) + pole-time-at-risk kernel ŵ_R
  03_field.py           Stage 2: CSR kernel machinery, ratio estimator, sweeps
  04_litter_panel.py    survey-level rows -> model table
  05_calibrate.py       Stage 3: ladder, profile likelihood, blocked CV
  06_surface.py         the citywide raster
  07_figures.py
```

### Reuse, not rewrite

| need | source |
|---|---|
| CBG frame, EPSG:26918 | `src/01_build_panel.py:74` `load_cbgs()` |
| poles → CBG, LED/PECO shares | `src/01_build_panel.py:159` `attach_poles()` |
| covariates, `alloc_avg_s_cnt` → `exposure` | `src/01_build_panel.py:198` `attach_covariates()` |
| point loader, dumping filter, → 26918 | `$SRC/replication/01_data_cleaning.py:144-178` |
| multi-year concat + date ceiling | `$SRC/replication/01_data_cleaning.py:181-227` |
| X/Y metres + T days build | `$SRC/replication/01_data_cleaning.py:509-573` |
| two-equation NumPyro model | `$SRC/replication/05a_streetlight_decomposition.py:407-437` |
| NUTS driver + posterior summary | `$SRC/replication/05a_streetlight_decomposition.py:440-480` |

(`$SRC` = `/Users/yuxin/Dropbox/DDDI/Illegal-Dumping`, **read-only**, port by copy.)

### The one change to `01_build_panel.py`

`build_panel()` (line 89) already reads every shapefile, filters to the two
categories, reprojects and parses `requested_` — then discards geometry at the
`groupby` on line 134. Two small edits: add `"closed_dat"` to the `columns=[...]`
list on **line 105**, and after **line 129** write the point tables from the same
`ev` frame, so the 528k-row-per-year files are read once. Likewise emit pole
coordinates (plus `light_date`, `bulb_type`) from `attach_poles()`, which already
holds the frame at line 166.

- `output/points_dumping.parquet` — `x, y, ts, t_days, closed_ts, GEOID`
- `output/points_outage.parquet` — `x, y, ts, GEOID`
- `output/poles_xy.parquet` — `x, y, light_date, bulb_type`

Keep the CBG × month panel; Stage 1's refit consumes it.

---

## Order of work

**Day 1 — triage before any pipeline.** These decide what the paper is:

1. **Forward/backward placebo** with month and location effects absorbed.
2. **Spline of rating vs log λ̂** — confirm the top-decile turnover; check whether
   correcting straightens it.
3. **Variance budget** — R² of log λ̂ on month FE, surveyor FE, both; the
   surveyor × block crosstab and surveyor–location overlap connectivity (if that
   graph is disconnected, surveyor and location effects are not separately
   identified and including both is fatal).
4. **Atom/dedup audit** across all years; fix the minimum defensible `h`.
5. **Endogenous-routing test** (Risk 3).
6. **Specification curve for β̂** across FE sets, bandwidths, `w̄` vs `w_R`, with
   and without the exposure denominator.

**Then build**, in this order:

7. Extend `01_build_panel.py`; verify 116,772 dumping points, 1,338 CBGs.
8. `02_reporting.py` — CBG refit must hit **β = 0.579** before anything proceeds;
   then the pole-time-at-risk kernel surface; cross-check the two.
9. `03_field.py` — CSR kernel machinery; unit-test against brute force.
10. `04_litter_panel.py` — drop 2020 (288 rows) and 2024 (79); restrict to
    `department == "streets"` (72,215 of 73,780 — PPR's 1,479 rows survey parks,
    where both the dumping process and the reporting channel differ); build the
    bulk flag; month-of-sample, `Creator`, `location_id`, CBG join.
11. `05_calibrate.py` — S0 → S1′ → S2 → SP at fixed (h, τ) first, then the sweep
    inside CV folds.
12. `06_surface.py`.

---

## Verification

- **Blocking regression test:** Stage 1a reproduces **β = 0.5791 [0.4698,
  0.6860]**, `a_NT = −1.077`, `a_T = −5.264`, `σ_R = 0.620`, `σ_T = 1.488`,
  0 divergences, n = 1,335. Also `corr(w_R, log outages-per-pole) ≈ 0.998`.
- **Panel totals:** D ≈ 116,751, R ≈ 75,956 over 1,335 CBGs (already printed by
  `01_build_panel.py:main()`).
- **KDE correctness:** sparse path vs brute-force dense on 500 random query
  points, to machine precision. Plus: flat weights and large `h` must integrate
  to the total event count within edge-correction tolerance.
- **Bandwidth optimum reproduces:** an interior maximum near 150–200 m
  (+0.307 at 200 m falling to +0.276 at 800 m on the raw field). If this vanishes
  under the full model, the point-level framing loses its main support.
- **Placebo separates** past from future — or is reported as failing.
- **Negative controls:** pothole-λ̂ gives `b ≈ 0`; the correction's improvement
  exceeds the permuted-`w_R` null.
- **The headline:** S2 beats S0 on held-out spatial blocks. If not, the correction
  does not survive at point level — report that; it is a finding either way.
- **β̂:** profile-likelihood estimate with block-bootstrap-calibrated CI, plus the
  `−c/b` (with `w̄`) closed form and a Fieller interval, plus `ρ(h)` at each
  bandwidth, plus the specification curve. Expect specification spread to swamp
  sampling error.
- **Level reconciliation:** aggregate visit-level predictions to CBG and correlate
  with CBG-mean rating — should land near the old plan's +0.58 raw / +0.64
  corrected.
- **Covariate attenuation:** γ at `h = 200` vs `h = 800`; shrinkage at the larger
  bandwidth is expected and confirms the resolution argument.
- **MCMC health** (Stage 1a): 0 divergences, R̂ < 1.01.
- **Predecessor repo untouched:** `git status` in
  `/Users/yuxin/Dropbox/DDDI/Illegal-Dumping` shows nothing from this work.
