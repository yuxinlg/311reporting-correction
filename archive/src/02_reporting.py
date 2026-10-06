"""
02_reporting.py
===============
Stage 1: reporting propensity w_R.

(1a) CBG posterior -- the beta anchor. Refits the predecessor's citywide
     two-equation Poisson (05a_streetlight_decomposition.py, section 2) on the
     pooled 2021-25 counts from output/panel_cbg_month.parquet:

        R_i ~ Poisson( P_i * exp(a_NT + w_R,i) )                 P = poles
        D_i ~ Poisson( E_i * exp(a_T + beta*w_R,i + w_T,i) )     E = SafeGraph stops

     BLOCKING regression test: beta = 0.5791 [0.4698, 0.6860], a_NT = -1.077,
     a_T = -5.264, sigma_R = 0.620, sigma_T = 1.488, 0 divergences, n = 1,335.
     Also corr(w_R, log R/P) ~ 0.998. Posterior draws of (beta, w_R) are
     saved for the two-stage uncertainty propagation in 05.

(1b) Point-level kernel incidence surface:

        w_R(s) = log[ sum_j K_hw(s - s_j)            (outage reports in window)
                    / sum_p K_hw(s - s_p) * days_p ]  (pole-days)
                 - citywide centre

     Variants
       pre24_naive   reports 2021-23 / all poles x 1,095 d                  PRIMARY
       pre24_atrisk  reports 2021-23 / non-LED pole-days 2021-23
       full_atrisk   reports 2021-25 / non-LED pole-days 2021-25
       naive         reports 2021-25 / all poles x 1,826 d  (05a-equivalent)
     Bandwidth h_w in {400, 600, 800, 1000, 1500} m quartic radius; default 600.

     Why the primary is NOT the pole-time-at-risk form the plan specified:
     LED conversion did not reduce outage reports. 2024-25 reports relative to
     2021-23 are 0.52 / 0.50 / 0.50 / 0.55 across LED-share quartiles (low to
     high), so poles keep generating outage reports after conversion and
     "non-LED pole-days at risk" is a mis-specified denominator (it inflates
     rates wherever 2023 conversions removed pole-days; pre24_atrisk vs
     pre24_naive correlate only 0.963). The pre-2024 window is kept because it
     coincides with the survey calibration window and sidesteps 2024-25 in
     any case. The +0.27 correlation of w_R with led_share is present already
     in the pre-2024 surface: the rollout went to high-complaint areas; the
     proxy is not contaminated by the conversion.

     LED poles without a light_date (8,126) take the date of the nearest dated
     LED pole; UNKNOWN and HPS poles are non-LED throughout (at-risk variants).

Outputs
-------
output/reporting_cbg.parquet      per CBG: counts, refit posterior, kernel w_R
                                  at the centroid per variant/bandwidth
output/reporting_draws.npz        beta (S,), w_R (S, n), GEOID order
output/reporting_summary.json     scalars, regression test, cross-checks
output/wR_events.parquet          kernel w_R at every dumping report
output/wR_survey_xy.parquet       kernel w_R at every distinct survey coordinate
output/wR_raster.npz              100 m raster of the primary surface
output/figures/02_*.png
"""

import os
import json
import time
import warnings

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as np_summary

from kernels import kernel_sum, SD_FACTOR

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

CRS = 26918

# --- 1a ----------------------------------------------------------------------
# Regression fit: identical settings to 05a, so the numbers must reproduce.
REG = dict(warmup=1000, samples=1000, chains=1, seed=0, target_accept=0.9, dense=False)
REF = {"beta": 0.5791, "beta_lo": 0.4698, "beta_hi": 0.6860, "a_NT": -1.077,
       "a_T": -5.264, "sigma_R": 0.620, "sigma_T": 1.488, "n": 1335}
TOL_BETA = 0.015
# Production fit: 05a's single chain mixes poorly on the T-equation globals
# (a_T n_eff ~ 7, beta n_eff ~ 18, w_T R-hat up to 1.11) because that equation
# is saturated and beta is identified only through the shrinkage prior. The
# production posterior uses 4 chains, a block-dense mass matrix over the five
# globals and a higher target acceptance; its draws feed 05's uncertainty
# propagation and its means are the pipeline's w_R and beta_0.
PROD = dict(warmup=4000, samples=8000, chains=4, seed=1, target_accept=0.95,
            dense=[("beta", "a_T", "sigma_T", "a_NT", "sigma_R")])
GLOBALS = ["beta", "a_NT", "a_T", "sigma_R", "sigma_T"]
N_DRAWS_SAVED = 2000      # thinned joint draws of (beta, w_R) for 05

# --- 1b ----------------------------------------------------------------------
HW_SWEEP = [400, 600, 800, 1000, 1500]
HW0 = 600                 # reliability 0.97 (Poisson noise vs signal); 800+
                          # only loses contrast (see reporting_summary.json)
PRIMARY = "pre24_naive"
KAPPA = 1.0               # Gamma-shrinkage prior mass, in "reports"
WIN = {
    "pre24": (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
    "full":  (pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01")),
}
RASTER_RES = 100          # m

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


# =============================================================================
# 1a. CBG refit
# =============================================================================

def two_eq_model_city(log_P, log_E, nt_obs=None, t_obs=None):
    """Verbatim port of 05a's citywide two-equation model."""
    n = log_P.shape[0]
    a_nt = numpyro.sample("a_NT", dist.Normal(0.0, 5.0))
    a_t = numpyro.sample("a_T", dist.Normal(0.0, 5.0))
    beta = numpyro.sample("beta", dist.LogNormal(0.0, 0.5))
    sigma_r = numpyro.sample("sigma_R", dist.HalfNormal(1.0))
    sigma_t = numpyro.sample("sigma_T", dist.HalfNormal(1.0))
    with numpyro.plate("cbg", n):
        z_r = numpyro.sample("z_R", dist.Normal(0.0, 1.0))
        z_t = numpyro.sample("z_T", dist.Normal(0.0, 1.0))
    w_r = numpyro.deterministic("w_R", z_r * sigma_r)
    w_t = numpyro.deterministic("w_T", z_t * sigma_t)
    with numpyro.plate("obs", n):
        numpyro.sample("NT", dist.Poisson(jnp.exp(log_P + a_nt + w_r)), obs=nt_obs)
        numpyro.sample("T", dist.Poisson(jnp.exp(log_E + a_t + beta * w_r + w_t)), obs=t_obs)


def load_cbg_table() -> pd.DataFrame:
    panel = pd.read_parquet(os.path.join(OUT, "panel_cbg_month.parquet"))
    static = pd.read_parquet(os.path.join(OUT, "cbg_static.parquet"))
    pooled = panel.groupby("GEOID")[["D", "R"]].sum().reset_index()
    cbg = static.merge(pooled, on="GEOID", how="left")
    cbg[["D", "R"]] = cbg[["D", "R"]].fillna(0).astype(int)

    # per-CBG agreement with 05a's counts, not just the totals
    ref = gpd.read_file(os.path.join(DATA, "ref_05a_output", "reporting_decomp_cbg_sl.geojson"))
    ref["GEOID"] = ref["GEOID"].astype(str)
    m = cbg.merge(ref[["GEOID", "T_count", "NT_curated"]], on="GEOID", how="left")
    dD, dR = (m["D"] - m["T_count"]).abs(), (m["R"] - m["NT_curated"]).abs()
    print(f"  per-CBG count check vs 05a: D max |diff| {dD.max():.0f} "
          f"({(dD > 0).sum()} CBGs differ), R max |diff| {dR.max():.0f} "
          f"({(dR > 0).sum()} CBGs differ)")
    return cbg


def _run_nuts(sub: pd.DataFrame, cfg: dict, label: str) -> dict:
    log_P = jnp.asarray(np.log(sub["n_poles"].to_numpy(float)))
    log_E = jnp.asarray(np.log(sub["exposure"].to_numpy(float)))
    nt = jnp.asarray(sub["R"].to_numpy(int))
    t = jnp.asarray(sub["D"].to_numpy(int))
    t0 = time.time()
    kernel = NUTS(two_eq_model_city, target_accept_prob=cfg["target_accept"],
                  dense_mass=cfg["dense"])
    mcmc = MCMC(kernel, num_warmup=cfg["warmup"], num_samples=cfg["samples"],
                num_chains=cfg["chains"], chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(cfg["seed"]), log_P=log_P, log_E=log_E,
             nt_obs=nt, t_obs=t, extra_fields=("diverging",))
    s = mcmc.get_samples()
    n_div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    by_chain = mcmc.get_samples(group_by_chain=True)
    diag = np_summary({k: by_chain[k] for k in GLOBALS + ["w_R", "w_T"]}, prob=0.94)

    beta = np.asarray(s["beta"])
    sc = {"beta": float(beta.mean()), "beta_lo": float(np.quantile(beta, 0.03)),
          "beta_hi": float(np.quantile(beta, 0.97)), "beta_sd": float(beta.std()),
          "a_NT": float(np.asarray(s["a_NT"]).mean()), "a_T": float(np.asarray(s["a_T"]).mean()),
          "sigma_R": float(np.asarray(s["sigma_R"]).mean()),
          "sigma_T": float(np.asarray(s["sigma_T"]).mean()),
          "n": int(len(sub)), "n_divergences": n_div, "seconds": round(time.time() - t0),
          "health": {k: {"r_hat": float(np.max(diag[k]["r_hat"])),
                         "n_eff_min": float(np.min(diag[k]["n_eff"]))} for k in diag}}
    h = sc["health"]
    print(f"  [{label}] {cfg['chains']}x{cfg['warmup']}/{cfg['samples']}, {sc['seconds']} s, "
          f"{n_div} divergences")
    print(f"     beta = {sc['beta']:.4f} [{sc['beta_lo']:.4f}, {sc['beta_hi']:.4f}] sd {sc['beta_sd']:.4f}  "
          f"a_NT = {sc['a_NT']:.3f}  a_T = {sc['a_T']:.3f}  sigma_R = {sc['sigma_R']:.3f}  "
          f"sigma_T = {sc['sigma_T']:.3f}  n = {sc['n']}")
    print("     R-hat / n_eff:  " + "  ".join(
        f"{k} {h[k]['r_hat']:.3f}/{h[k]['n_eff_min']:.0f}" for k in GLOBALS) +
        f"  |  w_R max {h['w_R']['r_hat']:.3f}/min {h['w_R']['n_eff_min']:.0f}"
        f"  w_T max {h['w_T']['r_hat']:.3f}/min {h['w_T']['n_eff_min']:.0f}")
    sc["samples"] = s
    return sc


def fit_cbg(cbg: pd.DataFrame, summary: dict) -> pd.DataFrame:
    print("\n[1a] Two-equation Poisson decomposition ...")
    sub = cbg[cbg["in_fit_sample"]].reset_index(drop=True)

    # --- regression fit: must reproduce 05a ---------------------------------
    reg = _run_nuts(sub, REG, "regression, 05a settings")
    print(f"     05a:  beta = {REF['beta']:.4f} [{REF['beta_lo']:.4f}, {REF['beta_hi']:.4f}]  "
          f"a_NT = {REF['a_NT']:.3f}  a_T = {REF['a_T']:.3f}  sigma_R = {REF['sigma_R']:.3f}  "
          f"sigma_T = {REF['sigma_T']:.3f}  n = {REF['n']}")
    checks = {
        "beta_within_tol": abs(reg["beta"] - REF["beta"]) < TOL_BETA,
        "interval_close": abs(reg["beta_lo"] - REF["beta_lo"]) < 0.02 and abs(reg["beta_hi"] - REF["beta_hi"]) < 0.02,
        "a_NT_close": abs(reg["a_NT"] - REF["a_NT"]) < 0.01,
        "a_T_close": abs(reg["a_T"] - REF["a_T"]) < 0.02,
        "sigmas_close": abs(reg["sigma_R"] - REF["sigma_R"]) < 0.01 and abs(reg["sigma_T"] - REF["sigma_T"]) < 0.02,
        "no_divergences": reg["n_divergences"] == 0, "n_matches": reg["n"] == REF["n"],
    }
    passed = all(checks.values())
    print(f"     REGRESSION TEST: {'PASS' if passed else 'FAIL'}  " +
          "  ".join(f"{k}={'ok' if v else 'X'}" for k, v in checks.items()))

    # --- production fit: well-mixed posterior for downstream -----------------
    prod = _run_nuts(sub, PROD, "production")
    hp = prod["health"]
    healthy = (max(hp[k]["r_hat"] for k in GLOBALS) < 1.01 and min(hp[k]["n_eff_min"] for k in GLOBALS) > 400
               and prod["n_divergences"] == 0)
    print(f"     PRODUCTION HEALTH (globals R-hat < 1.01, n_eff > 400, no divergences): "
          f"{'OK' if healthy else 'NOT MET'}")
    w_reg = np.asarray(reg["samples"]["w_R"]).mean(0)
    w_r = np.asarray(prod["samples"]["w_R"]); w_t = np.asarray(prod["samples"]["w_T"])
    print(f"     w_R means, regression vs production: corr {np.corrcoef(w_reg, w_r.mean(0))[0, 1]:.5f}, "
          f"max |diff| {np.abs(w_reg - w_r.mean(0)).max():.4f}")

    sub = sub.assign(w_R_mean=w_r.mean(0), w_R_sd=w_r.std(0),
                     w_T_mean=w_t.mean(0), w_T_sd=w_t.std(0),
                     log_rate_R=np.log((sub["R"] + 0.5) / sub["n_poles"]))   # +0.5 guards R = 0
    ok5 = sub["R"] >= 5
    c_all = np.corrcoef(sub["w_R_mean"], sub["log_rate_R"])[0, 1]
    c_5 = np.corrcoef(sub.loc[ok5, "w_R_mean"], sub.loc[ok5, "log_rate_R"])[0, 1]
    print(f"     corr(w_R, log R/P): {c_all:.4f} all CBGs, {c_5:.4f} on R >= 5 "
          f"(n = {int(ok5.sum())}; 05a reports 0.998)")

    summary["stage1a"] = {
        "regression": {k: v for k, v in reg.items() if k != "samples"},
        "regression_checks": checks, "regression_pass": passed,
        "production": {k: v for k, v in prod.items() if k != "samples"},
        "production_healthy": bool(healthy),
        "corr_wR_lograte_all": float(c_all), "corr_wR_lograte_R5": float(c_5),
    }
    s = prod["samples"]
    thin = max(1, len(np.asarray(s["beta"])) // N_DRAWS_SAVED)
    keep = slice(None, None, thin)
    np.savez_compressed(os.path.join(OUT, "reporting_draws.npz"),
                        beta=np.asarray(s["beta"])[keep], w_R=w_r[keep],
                        GEOID=sub["GEOID"].to_numpy().astype(str),
                        a_NT=np.asarray(s["a_NT"])[keep], a_T=np.asarray(s["a_T"])[keep],
                        sigma_R=np.asarray(s["sigma_R"])[keep], sigma_T=np.asarray(s["sigma_T"])[keep],
                        thin=thin)
    print(f"     saved {len(np.asarray(s['beta'])[keep]):,} thinned joint draws (every {thin}th).")
    return cbg.merge(sub[["GEOID", "w_R_mean", "w_R_sd", "w_T_mean", "w_T_sd", "log_rate_R"]],
                     on="GEOID", how="left")


# =============================================================================
# 1b. Pole-time-at-risk kernel surface
# =============================================================================

def load_poles() -> pd.DataFrame:
    p = pd.read_parquet(os.path.join(OUT, "poles_xy.parquet"))
    p["light_date"] = pd.to_datetime(p["light_date"], utc=True).dt.tz_localize(None)
    is_led = p["bulb_type"].eq("LED")
    dated = is_led & p["light_date"].notna()
    undated = is_led & p["light_date"].isna()
    tree = cKDTree(p.loc[dated, ["x", "y"]].to_numpy())
    dist_nn, idx = tree.query(p.loc[undated, ["x", "y"]].to_numpy())
    p.loc[undated, "light_date"] = p.loc[dated, "light_date"].to_numpy()[idx]
    p["conv"] = p["light_date"].where(is_led, pd.NaT)         # LED conversion date
    print(f"  {len(p):,} poles; {is_led.sum():,} LED ({undated.sum():,} took a nearest-neighbour "
          f"date, median distance {np.median(dist_nn):.0f} m); "
          f"{(~is_led).sum():,} UNKNOWN/HPS treated as non-LED throughout.")
    return p


def days_at_risk(p: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Non-LED days per pole inside [start, end)."""
    conv = p["conv"].fillna(end).clip(lower=start, upper=end)
    return (conv - start).dt.days.to_numpy(float)


def w_surface(qxy, pole_xy, pole_days, out_xy, h, pole_tree, out_tree, rbar=None):
    """
    Kernel incidence rate at query points, Gamma-shrunk toward the citywide
    rate rbar, returned as log(rate / rbar). Also returns the Kish n_eff of
    the numerator (Poisson noise floor) and a zero-numerator flag.
    """
    num, neff = kernel_sum(qxy, out_xy, h, source_tree=out_tree, return_neff=True)
    den = kernel_sum(qxy, pole_xy, h, weights=pole_days, source_tree=pole_tree)
    if rbar is None:
        rbar = num.sum() / den.sum()
    rate = (num + KAPPA) / (den + KAPPA / rbar)
    return np.log(rate / rbar), neff, num == 0


def build_surfaces(cbg: pd.DataFrame, summary: dict):
    print("\n[1b] Pole-time-at-risk kernel surface ...")
    p = load_poles()
    pole_xy = p[["x", "y"]].to_numpy()
    pole_tree = cKDTree(pole_xy)
    r = pd.read_parquet(os.path.join(OUT, "points_outage.parquet"))

    # query sets
    ev = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    ev_xy = ev[["x", "y"]].to_numpy()
    sv = pd.read_csv(os.path.join(DATA, "litter_surveys_raw_20250730.csv"), low_memory=False)
    sv.columns = [c.lstrip("﻿") for c in sv.columns]
    sxy = sv[["x", "y"]].drop_duplicates().reset_index(drop=True)
    sgeo = gpd.GeoSeries(gpd.points_from_xy(sxy["x"], sxy["y"]), crs=4326).to_crs(CRS)
    sv_xy = np.c_[sgeo.x.to_numpy(), sgeo.y.to_numpy()]
    geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson"))
    cen = geom.geometry.centroid
    cen_xy = np.c_[cen.x.to_numpy(), cen.y.to_numpy()]
    geom_ids = geom["GEOID"].astype(str).to_numpy()
    xmin, ymin, xmax, ymax = geom.total_bounds
    gx = np.arange(xmin + RASTER_RES / 2, xmax, RASTER_RES)
    gy = np.arange(ymin + RASTER_RES / 2, ymax, RASTER_RES)
    GX, GY = np.meshgrid(gx, gy)
    grid_xy = np.c_[GX.ravel(), GY.ravel()]
    gpts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid_xy[:, 0], grid_xy[:, 1]), crs=CRS)
    hit = gpts.sjoin(geom[["geometry"]], how="inner", predicate="within")
    inside = np.zeros(len(grid_xy), bool)
    inside[np.unique(hit.index.to_numpy())] = True
    print(f"  query sets: {len(ev_xy):,} dumping reports, {len(sv_xy):,} distinct survey "
          f"coordinates, {len(cen_xy):,} CBG centroids, {inside.sum():,}/{len(grid_xy):,} "
          f"raster cells inside the city ({RASTER_RES} m).")

    Q = {"events": ev_xy, "survey": sv_xy, "centroid": cen_xy, "grid": grid_xy[inside]}
    res = {k: pd.DataFrame(index=range(len(v))) for k, v in Q.items()}
    res["events"]["GEOID"] = ev["GEOID"].to_numpy()
    res["survey"]["lon"], res["survey"]["lat"] = sxy["x"].to_numpy(), sxy["y"].to_numpy()
    res["survey"]["x"], res["survey"]["y"] = sv_xy[:, 0], sv_xy[:, 1]
    res["centroid"]["GEOID"] = geom_ids
    res["grid"]["x"], res["grid"]["y"] = grid_xy[inside, 0], grid_xy[inside, 1]

    variants = {
        "pre24_naive": ("pre24", False),    # PRIMARY (see module docstring)
        "pre24_atrisk": ("pre24", True),    # the plan's pole-time-at-risk form
        "full_atrisk": ("full", True),
        "naive": ("full", False),           # 05a-equivalent
    }
    diag_rows = []
    for vname, (win, atrisk) in variants.items():
        start, end = WIN[win]
        m = (r["ts"] >= start) & (r["ts"] < end)
        out_xy = r.loc[m, ["x", "y"]].to_numpy()
        out_tree = cKDTree(out_xy)
        days = days_at_risk(p, start, end) if atrisk else np.full(len(p), (end - start).days, float)
        print(f"  variant {vname:>13}: {m.sum():,} outage reports, "
              f"{days.sum() / 1e6:,.2f}M pole-days at risk "
              f"({(days == 0).mean():.1%} of poles with zero at-risk days)")
        hws = HW_SWEEP if vname == PRIMARY else [HW0]
        for h in hws:
            t0 = time.time()
            # citywide rate from the pole set itself (numerator/denominator totals)
            rbar = m.sum() / days.sum()
            col = f"wR_{vname}_h{h}"
            for qname, qxy in Q.items():
                w, neff, zero = w_surface(qxy, pole_xy, days, out_xy, h, pole_tree, out_tree, rbar)
                res[qname][col] = w
                if qname == "survey":
                    res[qname][f"neff_{vname}_h{h}"] = neff
                    noise_sd = np.median(1.0 / np.sqrt(np.maximum(neff, 1)))
                    sig_sd = np.std(w)
                    diag_rows.append({"variant": vname, "h_w": h, "sd_wR_survey": sig_sd,
                                      "median_poisson_sd": noise_sd,
                                      "reliability": 1 - noise_sd ** 2 / max(sig_sd ** 2, 1e-12),
                                      "zero_num_share": float(zero.mean())})
            # centre on the pole-weighted mean (approximated over CBG centroids)
            cw = cbg.set_index("GEOID").reindex(geom_ids)["n_poles"].fillna(0).to_numpy(float)
            centre = np.average(res["centroid"][col].to_numpy(), weights=np.maximum(cw, 1e-9))
            for qname in Q:
                res[qname][col] -= centre
            print(f"    h_w={h:>4} (sd {h * SD_FACTOR:.0f} m): {time.time() - t0:4.1f} s   "
                  f"sd over survey pts {diag_rows[-1]['sd_wR_survey']:.3f}, median Poisson sd "
                  f"{diag_rows[-1]['median_poisson_sd']:.3f}, reliability {diag_rows[-1]['reliability']:.3f}, "
                  f"zero numerators {diag_rows[-1]['zero_num_share']:.2%}")
    diag = pd.DataFrame(diag_rows)
    summary["stage1b_noise"] = diag.round(4).to_dict(orient="records")

    # cross-checks against the CBG posterior
    cc = res["centroid"].merge(cbg[["GEOID", "w_R_mean", "log_rate_R", "led_share", "n_poles", "in_fit_sample"]],
                               on="GEOID", how="left")
    cc = cc[cc["in_fit_sample"].fillna(False)]
    xrows = []
    for col in [c for c in cc.columns if c.startswith("wR_")]:
        ok = np.isfinite(cc[col]) & np.isfinite(cc["w_R_mean"]) & np.isfinite(cc["led_share"])
        d = cc.loc[ok, col] - cc.loc[ok, "w_R_mean"]
        xrows.append({"surface": col,
                      "corr_with_cbg_wR": np.corrcoef(cc.loc[ok, col], cc.loc[ok, "w_R_mean"])[0, 1],
                      "corr_with_log_rate": np.corrcoef(cc.loc[ok, col], cc.loc[ok, "log_rate_R"])[0, 1],
                      "corr_diff_led_share": np.corrcoef(d, cc.loc[ok, "led_share"])[0, 1],
                      "corr_with_led_share": np.corrcoef(cc.loc[ok, col], cc.loc[ok, "led_share"])[0, 1],
                      "sd": float(cc.loc[ok, col].std())})
    x = pd.DataFrame(xrows)
    ok = np.isfinite(cc["w_R_mean"]) & np.isfinite(cc["led_share"])
    c05 = np.corrcoef(cc.loc[ok, "w_R_mean"], cc.loc[ok, "led_share"])[0, 1]
    print("\n  cross-check at CBG centroids (fit sample):")
    print(f"    CBG posterior w_R: corr with led_share {c05:+.3f}  (05a flagged +0.285)")
    print(x.round(3).to_string(index=False))

    # the kernel averaged over the CBG's own dumping reports is the fairer
    # comparison with a polygon posterior than a single centroid value
    wcols = [c for c in res["events"].columns if c.startswith("wR_")]
    ge = res["events"].groupby("GEOID")[wcols].mean().reset_index().merge(
        cbg[["GEOID", "w_R_mean", "in_fit_sample"]], on="GEOID")
    ge = ge[ge["in_fit_sample"].fillna(False)]
    ev_rows = []
    for c in wcols:
        okc = np.isfinite(ge[c]) & np.isfinite(ge["w_R_mean"])
        ev_rows.append({"surface": c,
                        "corr_event_avg_vs_cbg_wR": np.corrcoef(ge.loc[okc, c], ge.loc[okc, "w_R_mean"])[0, 1],
                        "sd_event_avg": float(ge.loc[okc, c].std()), "sd_cbg_wR": float(ge.loc[okc, "w_R_mean"].std())})
    xe = pd.DataFrame(ev_rows)
    print("  kernel averaged over each CBG's dumping reports vs CBG posterior:")
    print(xe.round(3).to_string(index=False))
    vc = res["events"][wcols].corr()
    print("  correlations among variants at the dumping-report locations:")
    print(vc.round(3).to_string())

    # is the PSIP denominator correction doing anything? outage reports 2024-25
    # vs 2021-23 by LED-share quartile (annualised ratio)
    panel = pd.read_parquet(os.path.join(OUT, "panel_cbg_month.parquet"))
    pre = panel[panel["year"] <= 2023].groupby("GEOID")["R"].sum()
    post = panel[panel["year"] >= 2024].groupby("GEOID")["R"].sum()
    q = cbg[cbg["in_fit_sample"]].set_index("GEOID").assign(pre=pre, post=post)
    q["led_q"] = pd.qcut(q["led_share"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    psip = q.groupby("led_q", observed=True).agg(n=("pre", "size"), led_share=("led_share", "mean"),
                                                 pre=("pre", "sum"), post=("post", "sum"))
    psip["post_over_pre_annualised"] = psip["post"] / psip["pre"] * (36 / 24)
    print("  outage reports 2024-25 vs 2021-23 by LED-share quartile (mechanical PSIP effect):")
    print(psip.round(3).to_string())

    summary["stage1b_crosscheck"] = {
        "corr_cbg_wR_led_share": float(c05),
        "centroid": x.round(4).to_dict(orient="records"),
        "event_avg": xe.round(4).to_dict(orient="records"),
        "variant_corr": vc.round(4).to_dict(),
        "psip_by_led_quartile": psip.round(4).reset_index().astype({"led_q": str}).to_dict(orient="records"),
    }

    # save
    res["events"].to_parquet(os.path.join(OUT, "wR_events.parquet"), index=False)
    res["survey"].to_parquet(os.path.join(OUT, "wR_survey_xy.parquet"), index=False)
    cbg_out = cbg.merge(res["centroid"], on="GEOID", how="left")
    cbg_out.to_parquet(os.path.join(OUT, "reporting_cbg.parquet"), index=False)
    prim = f"wR_{PRIMARY}_h{HW0}"
    Z = np.full(GX.shape, np.nan); Z.ravel()[inside] = res["grid"][prim].to_numpy()
    np.savez_compressed(os.path.join(OUT, "wR_raster.npz"), x=gx, y=gy, wR=Z, variant=prim, res=RASTER_RES)
    print(f"\n  Saved → output/wR_events.parquet, wR_survey_xy.parquet, reporting_cbg.parquet, wR_raster.npz")

    # figures
    _figures(cbg_out, cc, Z, gx, gy, geom, prim, x)


def _figures(cbg_out, cc, Z, gx, gy, geom, prim, x):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    im = ax.imshow(Z, origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]], cmap="Blues",
                   vmin=np.nanquantile(Z, 0.02), vmax=np.nanquantile(Z, 0.98))
    geom.boundary.plot(ax=ax, color=GRID, linewidth=0.15)
    ax.set_title(f"Kernel ŵ_R, {prim}: log outage reports per pole-day, centred", fontsize=9, loc="left")
    ax.set_axis_off(); plt.colorbar(im, ax=ax, shrink=0.7)
    ax = axes[1]
    ok = np.isfinite(cc[prim]) & np.isfinite(cc["w_R_mean"])
    ax.scatter(cc.loc[ok, "w_R_mean"], cc.loc[ok, prim], s=6, color=C1, alpha=0.5)
    lo, hi = np.nanmin(cc.loc[ok, "w_R_mean"]), np.nanmax(cc.loc[ok, "w_R_mean"])
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    r_ = x.loc[x["surface"] == prim, "corr_with_cbg_wR"].iloc[0]
    ax.set_xlabel("05a-refit CBG posterior mean w_R", color=MUTED, fontsize=9)
    ax.set_ylabel("kernel ŵ_R at CBG centroid", color=MUTED, fontsize=9)
    ax.set_title(f"CBG posterior vs kernel surface, r = {r_:.3f}", fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "02_wR_surface.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ok = np.isfinite(cbg_out["w_R_mean"])
    ax.errorbar(cbg_out.loc[ok, "log_rate_R"], cbg_out.loc[ok, "w_R_mean"],
                yerr=cbg_out.loc[ok, "w_R_sd"], fmt=".", color=C1, ms=3, alpha=0.4, lw=0.4)
    ax.set_xlabel("log(R / poles)", color=MUTED, fontsize=9); ax.set_ylabel("posterior w_R ± sd", color=MUTED, fontsize=9)
    ax.set_title("Stage 1a: shrinkage of w_R toward the raw log rate", fontsize=10, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "02_wR_shrinkage.png"), dpi=160); plt.close(fig)


# =============================================================================

def main():
    print("=" * 68); print("02_reporting.py — Stage 1: reporting propensity"); print("=" * 68)
    summary = {}
    cbg = load_cbg_table()
    cbg = fit_cbg(cbg, summary)
    build_surfaces(cbg, summary)
    with open(os.path.join(OUT, "reporting_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
    print("\nDone.")


if __name__ == "__main__":
    main()
