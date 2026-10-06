"""
06_surface.py
=============
The deliverable maps, on the built support (50 m cells inside a block group,
within 75 m of a pole). Three things, kept separate:

A. 311 illegal dumping as reported — long-term average report rate,
       λ_raw(s) = Σ_{j in window} K_H0(|s − s_j|) / Σ_c K_H0(|s − c|) E_c   per year
   in reports per year per RATE_PER SafeGraph monthly stops, for each window in
   LONG_WINDOWS (all five years; the two survey years).
B. The same after the reporting correction — every report weighted by
   exp(−β̂ w_R(s_j)), with β̂ from 04. "Reports that would have been filed at the
   citywide-average reporting propensity": the corrected picture of dumping.
   B / A is the correction factor.
C. The Litter Index on its own — survey ratings with month-of-sample and
   surveyor effects removed (one OLS on the survey alone), kernel-smoothed at
   the visit locations with the same K_H0, blank where the kernel sees fewer
   than SURVEY_MIN_NEFF visits. Nothing from 311 enters it.

Comparison: block-group means of A and B against block-group means of C (and
against the observed mean rating). The corrected map should agree with the
survey better than the raw one.

Appendix: the snapshot version — the 120-day field at DATE_IN and DATE_OUT
pushed through the rating model (E[rating] = how a surveyor would score the
cell). It answers "how well does 311 reproduce the survey", not "where is the
dumping".

Colours (config.py): what is on the ground — litter (CMAP_LITTER) and dumping
report rates (CMAP_DUMPING) — on the same diverging red–blue scale centred on
the citywide mean / rate; how people report — reporting propensity and the
correction factor — on the CMAP_PROPENSITY / CMAP_FACTOR pair (pink = chatty
places, green = quiet places, in both).

Writes  output/surface/dumping_{window}.tif      bands rate_raw, rate_corr (per year per 1,000 stops),
                                                  log_rel_raw, log_rel_corr (log of rate / citywide rate),
                                                  correction_factor (corrected / raw)
        output/surface/litter_survey.tif         bands rating_adj, rating_mean, n_eff
        output/surface/litter_index_{date}.tif   appendix snapshot: E_rating_corr, E_rating_raw, log_lam_corr, log_lam_raw
        output/surface/surface_cbg.csv, summary.json
        output/figures/fig3_maps.png (A, B, C), fig5_dumping_corrected.png (B alone, planning districts named),
        fig3b_zoom.png, A1_rating_snapshot.png, A2_windows_and_correction.png
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects

from config import (DATA, OUT, FIG, FIELDS, SURF, CRS, EPOCH, GRID_RES, TAU, ALPHA, BETA_GRID, DATE_IN, DATE_OUT,
                    LONG_WINDOWS, RATE_PER, SURVEY_MIN_NEFF, ZOOM, DISTRICT_FILE, DISTRICT_COL, INK, MUTED,
                    CMAP_LITTER, CMAP_DUMPING, CMAP_PROPENSITY, CMAP_FACTOR, map_style, load_choices, ensure_dirs)
from kernels import kernel_sum, accumulate_st, shrunk_log
from ordinal import OrderedLogit
from design import FE, design, spline_basis

warnings.filterwarnings("ignore")
OL = OrderedLogit(4)
I0 = int(np.argmin(np.abs(BETA_GRID)))


# =============================================================================
# raster helpers
# =============================================================================

class Raster:
    def __init__(self, geom, sxy):
        xmin, ymin, xmax, ymax = geom.total_bounds
        self.gx = np.arange(xmin + GRID_RES / 2, xmax, GRID_RES)
        self.gy = np.arange(ymin + GRID_RES / 2, ymax, GRID_RES)
        self.ix = np.round((sxy[:, 0] - self.gx[0]) / GRID_RES).astype(int)
        self.iy = np.round((sxy[:, 1] - self.gy[0]) / GRID_RES).astype(int)
        self.transform = from_origin(self.gx[0] - GRID_RES / 2, self.gy[-1] + GRID_RES / 2, GRID_RES, GRID_RES)
        self.extent = [self.gx[0] - GRID_RES / 2, self.gx[-1] + GRID_RES / 2, self.gy[0] - GRID_RES / 2, self.gy[-1] + GRID_RES / 2]

    def grid(self, values):
        """support-cell values -> 2-D array, north up, NaN off the support"""
        Z = np.full((len(self.gy), len(self.gx)), np.nan, dtype="float32")
        Z[self.iy, self.ix] = values
        return Z[::-1, :]

    def write(self, path, bands: dict, tags: dict):
        names = list(bands)
        arr = np.stack([self.grid(bands[n]) for n in names])
        with rasterio.open(path, "w", driver="GTiff", height=len(self.gy), width=len(self.gx), count=len(names),
                           dtype="float32", crs=f"EPSG:{CRS}", transform=self.transform, nodata=np.nan,
                           compress="deflate") as dst:
            dst.write(arr)
            for b, n in enumerate(names):
                dst.set_band_description(b + 1, n)
            dst.update_tags(**{k: str(v) for k, v in tags.items()})


def show(ax, Z, R, geom, title, label, vmin, vmax, cmap, zoom=None, ticks=None, ticklabels=None):
    """One map panel in the house style: km axes, block-group outlines, horizontal colorbar below."""
    im = ax.imshow(Z, origin="upper", extent=R.extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    map_style(ax, geom, title, xlim=zoom["x"] if zoom else None, ylim=zoom["y"] if zoom else None)
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.75, pad=0.08, aspect=35)
    if ticks is not None:
        cb.set_ticks(ticks); cb.set_ticklabels(ticklabels)
    cb.set_label(label, fontsize=8, color=MUTED); cb.ax.tick_params(labelsize=8, colors=MUTED)
    return im


def sym_lim(*arrays, q=0.90):
    """Symmetric colour limit for a centred field: the q-quantile of |values|."""
    return float(np.nanquantile(np.abs(np.concatenate([np.asarray(a).ravel() for a in arrays])), q))


def rate_scale(*arrays):
    """Symmetric colour range for a log report rate centred on the citywide rate, with ticks at round
    multiples (×0.1 … ×10) of that rate."""
    lim = sym_lim(*arrays)
    mult = np.array([0.03, 0.1, 0.3, 1, 3, 10, 30])
    t = np.log(mult); keep = np.abs(t) <= lim
    return -lim, lim, t[keep], [f"×{m:g}" for m in mult[keep]]


# =============================================================================
# A and B: long-term report rates, raw and corrected
# =============================================================================

def long_term(ev_xy, ev_ts, ev_t, m_hat, sxy, den, H0, t0, t1):
    m = ((ev_ts >= t0) & (ev_ts < t1)).to_numpy()
    years = (t1 - t0).days / 365.25
    o = accumulate_st(sxy, np.zeros(len(sxy)), ev_xy[m], ev_t[m], H0, [np.inf],
                      {"one": np.ones(int(m.sum())), "m": m_hat[m]}, causal=False)
    out = {"n_reports": int(m.sum()), "years": years}
    for which, key in (("raw", "one"), ("corr", "m")):
        num = o[(np.inf, key)]
        lam_bar = num.sum() / den.sum()                       # citywide rate over the window, support-grid based
        ll = shrunk_log(num, den, ALPHA, lam_bar)[0]
        out[f"rate_{which}"] = np.exp(ll) / years * RATE_PER
        out[f"log_rel_{which}"] = ll - np.log(lam_bar)
        out[f"citywide_rate_{which}"] = float(lam_bar / years * RATE_PER)
        out[f"zero_cells_{which}"] = float((num == 0).mean())
    out["correction_factor"] = out["rate_corr"] / out["rate_raw"]
    return out


# =============================================================================
# C: the survey on its own
# =============================================================================

def survey_map(v, sxy, H0):
    """Ratings with month + surveyor effects removed (OLS on the survey alone), kernel-smoothed."""
    y = v["rating"].to_numpy(float)
    Xfe, _ = FE(v).build(v)
    X1 = np.c_[np.ones(len(y)), Xfe]
    b = np.linalg.lstsq(X1, y, rcond=None)[0]
    fitted = X1 @ b
    adj = y - fitted + fitted.mean()                           # same mean, nuisance variation removed
    vxy = v[["x", "y"]].to_numpy()
    tree = cKDTree(vxy)
    den, neff = kernel_sum(sxy, vxy, H0, source_tree=tree, return_neff=True)
    ok = neff >= SURVEY_MIN_NEFF
    with np.errstate(invalid="ignore", divide="ignore"):
        rating_adj = np.where(ok, kernel_sum(sxy, vxy, H0, weights=adj, source_tree=tree) / den, np.nan)
        rating_mean = np.where(ok, kernel_sum(sxy, vxy, H0, weights=y, source_tree=tree) / den, np.nan)
    r2_fe = 1 - np.var(y - fitted) / np.var(y)
    return {"rating_adj": rating_adj, "rating_mean": rating_mean, "n_eff": neff, "coverage": float(ok.mean()),
            "fe_r2": float(r2_fe), "adj_sd_visits": float(adj.std()), "raw_sd_visits": float(y.std())}


# =============================================================================
# appendix: the snapshot pushed through the rating model
# =============================================================================

def fit_visit_models(v, F, ev_xy, ev_t, tree, m_hat, H0):
    y = v["rating"].to_numpy() - 1
    fe = FE(v).build(v)
    lam_raw = float(F["lam_bar_raw"])
    o = accumulate_st(v[["x", "y"]].to_numpy(), v["t_days"].to_numpy(), ev_xy, ev_t, H0, [TAU], {"m": m_hat}, src_tree=tree)
    x = {"corr": shrunk_log(o[(TAU, "m")], F["den"], ALPHA, lam_raw * m_hat.mean())[0],
         "raw": shrunk_log(F["num"][:, I0], F["den"], ALPHA, lam_raw)[0]}
    models = {}
    for which in ("corr", "raw"):
        X, pen, _, info = design(x[which], fe)
        th, ll, _ = OL.fit(X, y, penalty=pen)
        beta, cuts = OL.unpack(th, X.shape[1]); ng = X.shape[1] - fe[0].shape[1]
        models[which] = {"info": info, "g": beta[:ng], "cuts": cuts, "fe_mean": float((fe[0] @ beta[ng:]).mean()), "ll": ll}
    return models, lam_raw


def predict(model, x):
    B, _ = spline_basis(x, model["info"])
    eta = B @ model["g"] + model["fe_mean"]
    return 1.0 + expit(eta[:, None] - model["cuts"][None, :]).sum(1)


# =============================================================================

def main():
    t_start = time.time()
    print("=" * 68); print("06_surface.py — the maps"); print("=" * 68)
    ensure_dirs()
    ch = load_choices(); H0, HW0, beta_hat = ch["H0"], ch["HW0"], float(ch["beta_hat"])
    print(f"  β̂ = {beta_hat:.3f} (95% CI {ch['ci_95'][0]:.2f}–{ch['ci_95'][1]:.2f}), h = {H0}, α = {ALPHA}")
    v = pd.read_parquet(os.path.join(OUT, "visits.parquet"))
    ev = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    wr = pd.read_parquet(os.path.join(OUT, "wR_events.parquet"))[f"wR_h{HW0}"].to_numpy()
    m_hat = np.exp(-beta_hat * np.where(np.isfinite(wr), wr, 0.0))
    ev_xy, ev_t = ev[["x", "y"]].to_numpy(), ev["t_days"].to_numpy(); tree = cKDTree(ev_xy)
    sup = pd.read_parquet(os.path.join(OUT, "support_grid.parquet"))
    sxy, sE = sup[["x", "y"]].to_numpy(), sup["E_cell"].to_numpy()
    geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson")); geom["GEOID"] = geom["GEOID"].astype(str)
    R = Raster(geom, sxy)
    den = kernel_sum(sxy, sxy, H0, weights=sE, source_tree=cKDTree(sxy))
    obs = v.groupby("GEOID")["rating"].mean()
    summary = {"beta_hat": beta_hat, "ci_95": ch["ci_95"], "h": H0, "alpha": ALPHA, "n_cells": int(len(sup)),
               "rate_unit": f"reports per year per {RATE_PER:g} SafeGraph monthly stops", "long_term": {}, "comparison": {}}
    cbg = pd.DataFrame({"GEOID": sup["GEOID"]})

    # ---- A, B: long-term raw and corrected --------------------------------------------------------
    print("\n[A, B] long-term report rates on the support grid")
    LT = {}
    for name, (t0, t1) in LONG_WINDOWS.items():
        t = time.time()
        L = long_term(ev_xy, ev["ts"], ev_t, m_hat, sxy, den, H0, t0, t1)
        LT[name] = L
        R.write(os.path.join(SURF, f"dumping_{name}.tif"),
                {k: L[k] for k in ["rate_raw", "rate_corr", "log_rel_raw", "log_rel_corr", "correction_factor"]},
                {"window": name, "beta_hat": beta_hat, "h_m": H0, "unit": summary["rate_unit"]})
        for k in ["rate_raw", "rate_corr", "log_rel_raw", "log_rel_corr", "correction_factor"]:
            cbg[f"{k}_{name}"] = L[k]
        summary["long_term"][name] = {
            "n_reports": L["n_reports"], "years": round(L["years"], 2),
            "citywide_rate_raw": L["citywide_rate_raw"], "citywide_rate_corr": L["citywide_rate_corr"],
            "rate_raw_p1_p50_p99": [float(q) for q in np.quantile(L["rate_raw"], [.01, .5, .99])],
            "rate_corr_p1_p50_p99": [float(q) for q in np.quantile(L["rate_corr"], [.01, .5, .99])],
            "correction_factor_p1_p50_p99": [float(q) for q in np.quantile(L["correction_factor"], [.01, .5, .99])],
            "zero_cells": L["zero_cells_raw"]}
        s = summary["long_term"][name]
        print(f"  {name}: {L['n_reports']:,} reports over {L['years']:.1f} y; citywide {s['citywide_rate_raw']:.2f} raw / "
              f"{s['citywide_rate_corr']:.2f} corrected per year per {RATE_PER:g} stops; cell rate p1/p50/p99 raw "
              f"{s['rate_raw_p1_p50_p99'][0]:.2f}/{s['rate_raw_p1_p50_p99'][1]:.2f}/{s['rate_raw_p1_p50_p99'][2]:.2f}, "
              f"corrected {s['rate_corr_p1_p50_p99'][0]:.2f}/{s['rate_corr_p1_p50_p99'][1]:.2f}/{s['rate_corr_p1_p50_p99'][2]:.2f}; "
              f"correction factor p1/p50/p99 {s['correction_factor_p1_p50_p99'][0]:.2f}/{s['correction_factor_p1_p50_p99'][1]:.2f}/"
              f"{s['correction_factor_p1_p50_p99'][2]:.2f}; zero cells {L['zero_cells_raw']:.2%}   ({time.time() - t:.0f} s)")

    # ---- C: the survey on its own ------------------------------------------------------------------
    print("\n[C] the Litter Index on its own")
    t = time.time()
    S = survey_map(v, sxy, H0)
    R.write(os.path.join(SURF, "litter_survey.tif"), {k: S[k] for k in ["rating_adj", "rating_mean", "n_eff"]},
            {"h_m": H0, "min_n_eff": SURVEY_MIN_NEFF, "adjusted_for": "month-of-sample + surveyor (OLS on the survey)"})
    cbg["survey_rating_adj"] = S["rating_adj"]; cbg["survey_rating_mean"] = S["rating_mean"]
    summary["survey_map"] = {"coverage_share_of_support": S["coverage"], "fe_r2": S["fe_r2"],
                             "rating_adj_p1_p50_p99": [float(q) for q in np.nanquantile(S["rating_adj"], [.01, .5, .99])]}
    print(f"  month + surveyor effects explain {S['fe_r2']:.1%} of rating variance; smoothed adjusted rating covers "
          f"{S['coverage']:.1%} of the support (n_eff ≥ {SURVEY_MIN_NEFF}); p1/p50/p99 "
          + "/".join(f"{q:.2f}" for q in summary["survey_map"]["rating_adj_p1_p50_p99"]) + f"   ({time.time() - t:.0f} s)")

    # ---- comparison: do the 311 maps agree with the survey map? -------------------------------------
    print("\n[compare] block-group means: 311 maps vs the survey map (Spearman)")
    cb = cbg.groupby("GEOID").mean()
    cb["obs_mean_rating_2022_23"] = obs.reindex(cb.index)
    cb["n_cells"] = cbg.groupby("GEOID").size()
    ok_s = cb["survey_rating_adj"].notna(); ok_o = cb["obs_mean_rating_2022_23"].notna()
    cell_ok = np.isfinite(S["rating_adj"])
    for name in LONG_WINDOWS:
        row = {}
        for which in ("raw", "corr"):
            col = f"log_rel_{which}_{name}"
            row[f"cbg_vs_survey_map_{which}"] = float(spearmanr(cb.loc[ok_s, col], cb.loc[ok_s, "survey_rating_adj"]).correlation)
            row[f"cbg_vs_observed_{which}"] = float(spearmanr(cb.loc[ok_o, col], cb.loc[ok_o, "obs_mean_rating_2022_23"]).correlation)
            row[f"cell_vs_survey_map_{which}"] = float(spearmanr(LT[name][f"log_rel_{which}"][cell_ok], S["rating_adj"][cell_ok]).correlation)
        summary["comparison"][name] = row
        print(f"  {name}: block groups vs survey map  raw {row['cbg_vs_survey_map_raw']:.3f} → corrected {row['cbg_vs_survey_map_corr']:.3f};  "
              f"vs observed mean rating  raw {row['cbg_vs_observed_raw']:.3f} → corrected {row['cbg_vs_observed_corr']:.3f};  "
              f"cells vs survey map  raw {row['cell_vs_survey_map_raw']:.3f} → corrected {row['cell_vs_survey_map_corr']:.3f}")

    # ---- appendix: snapshot through the rating model ----------------------------------------------
    print("\n[appendix] 120-day snapshot pushed through the rating model")
    F = dict(np.load(os.path.join(FIELDS, f"h{H0}.npz")))
    models, lam_raw = fit_visit_models(v, F, ev_xy, ev_t, tree, m_hat, H0)
    summary["snapshot"] = {}
    snap = {}
    for label, date in [("in", DATE_IN), ("out", DATE_OUT)]:
        t_eval = float((date - EPOCH).days)
        o = accumulate_st(sxy, np.full(len(sxy), t_eval), ev_xy, ev_t, H0, [TAU], {"m": m_hat, "one": np.ones(len(ev_t))}, src_tree=tree)
        x_corr = shrunk_log(o[(TAU, "m")], den, ALPHA, lam_raw * m_hat.mean())[0]
        x_raw = shrunk_log(o[(TAU, "one")], den, ALPHA, lam_raw)[0]
        bands = {"E_rating_corr": predict(models["corr"], x_corr), "E_rating_raw": predict(models["raw"], x_raw),
                 "log_lam_corr": x_corr, "log_lam_raw": x_raw}
        R.write(os.path.join(SURF, f"litter_index_{date.date().isoformat()}.tif"), bands,
                {"beta_hat": beta_hat, "h_m": H0, "tau_days": TAU, "date": date.date().isoformat(), "extrapolation": label == "out"})
        cbg[f"E_rating_corr_{label}"] = bands["E_rating_corr"]; cbg[f"E_rating_raw_{label}"] = bands["E_rating_raw"]
        cbm = cbg.groupby("GEOID")[[f"E_rating_corr_{label}", f"E_rating_raw_{label}"]].mean().reindex(cb.index)
        rs = {w: float(spearmanr(cbm.loc[ok_o, f"E_rating_{w}_{label}"], cb.loc[ok_o, "obs_mean_rating_2022_23"]).correlation) for w in ("corr", "raw")}
        delta = bands["E_rating_corr"] - bands["E_rating_raw"]
        summary["snapshot"][date.date().isoformat()] = {
            "extrapolation": label == "out",
            "E_rating_corr_p1_p50_p99": [float(q) for q in np.quantile(bands["E_rating_corr"], [.01, .5, .99])],
            "correction_delta_sd": float(delta.std()), "cbg_spearman_obs_vs_corr": rs["corr"], "cbg_spearman_obs_vs_raw": rs["raw"]}
        snap[label] = bands
        print(f"  {date.date()} ({'survey window' if label == 'in' else 'EXTRAPOLATION'}): E[rating] p50 "
              f"{np.median(bands['E_rating_corr']):.2f}; block-group Spearman with observed: corrected {rs['corr']:.3f}, raw {rs['raw']:.3f}")

    cb = cbg.groupby("GEOID").mean(); cb["obs_mean_rating_2022_23"] = obs.reindex(cb.index)
    cb.reset_index().to_csv(os.path.join(SURF, "surface_cbg.csv"), index=False)
    with open(os.path.join(SURF, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _figures(R, geom, LT, S, snap, beta_hat, HW0, ev, v)
    print(f"\nDone in {time.time() - t_start:.0f} s. Saved → output/surface/dumping_*.tif, litter_survey.tif, litter_index_*.tif, "
          f"surface_cbg.csv, summary.json; figures fig3_maps.png, fig5_dumping_corrected.png, fig3b_zoom.png, "
          f"A1_rating_snapshot.png, A2_windows_and_correction.png")


def _figures(R, geom, LT, S, snap, beta_hat, HW0, ev, v):
    """House style after the project's reference map: km axes in EPSG:26918, block-group outlines,
    horizontal colorbars, titles top-left. One colour family per kind of data (config.py)."""
    main = "2021-2025"; other = [k for k in LT if k != main][0]
    L, Lo = LT[main], LT[other]
    # dumping report rates: one diverging scale centred on the citywide rate, shared by A and B (and the
    # two-year window), ticks in multiples of that rate
    r_lo, r_hi, r_ticks, r_labels = rate_scale(L["log_rel_raw"], L["log_rel_corr"])
    lab_rate = "dumping report rate as a multiple of the citywide rate   (blue = fewer, red = more)"
    # litter: diverging, centred on the citywide mean
    c_mean = float(np.nanmean(S["rating_adj"]))
    C = S["rating_adj"] - c_mean
    c_lim = sym_lim(C)
    lab_c = f"adjusted rating − citywide mean ({c_mean:.2f}), rating points   (blue = cleaner, red = dirtier)"
    panels = [(f"(A) 311 dumping reports as filed, {main} average", L["log_rel_raw"], lab_rate, r_lo, r_hi, CMAP_DUMPING, r_ticks, r_labels),
              (f"(B) corrected for reporting propensity, β̂ = {beta_hat:.2f}", L["log_rel_corr"], lab_rate, r_lo, r_hi, CMAP_DUMPING, r_ticks, r_labels),
              ("(C) Litter Index survey on its own, 2022–23 (month + surveyor adjusted)", C, lab_c, -c_lim, c_lim, CMAP_LITTER, None, None)]

    # fig3: citywide
    fig, axes = plt.subplots(1, 3, figsize=(19, 8))
    for ax, (title, val, lab, lo, hi, cmap, tk, tl) in zip(axes, panels):
        show(ax, R.grid(val), R, geom, title, lab, lo, hi, cmap, ticks=tk, ticklabels=tl)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_maps.png"), dpi=170); plt.close(fig)

    # fig5: the conclusion — B on its own, large, with the planning districts outlined and named; the colorbar
    # carries both the multiple of the citywide corrected rate and the absolute rate
    dist = gpd.read_file(os.path.join(DATA, DISTRICT_FILE)).to_crs(CRS)
    rate = L["citywide_rate_corr"]
    fig, ax = plt.subplots(figsize=(10.5, 11.5))
    show(ax, R.grid(L["log_rel_corr"]), R, geom,
         f"Estimated illegal dumping, {main}: 311 reports corrected for reporting propensity (β̂ = {beta_hat:.2f})",
         f"multiple of the citywide corrected rate (above) and reports per year per {RATE_PER:,.0f} SafeGraph monthly stops "
         f"(below; citywide {rate:.2f})\nblue = less dumping than the city average, red = more",
         r_lo, r_hi, CMAP_DUMPING, ticks=r_ticks, ticklabels=[f"{l}\n{np.exp(t) * rate:.2f}" for l, t in zip(r_labels, r_ticks)])
    dist.boundary.plot(ax=ax, color=INK, linewidth=0.9)
    for _, d in dist.iterrows():
        p = d.geometry.representative_point()
        ax.text(p.x, p.y, d[DISTRICT_COL], fontsize=7.5, ha="center", va="center", color=INK,
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_dumping_corrected.png"), dpi=170); plt.close(fig)

    # fig3b: the reference window, with the underlying points overlaid
    inz = lambda x, y: (x >= ZOOM["x"][0]) & (x <= ZOOM["x"][1]) & (y >= ZOOM["y"][0]) & (y <= ZOOM["y"][1])
    t0, t1 = LONG_WINDOWS[main]
    e = ev[inz(ev["x"], ev["y"]) & (ev["ts"] >= t0) & (ev["ts"] < t1)]
    vz = v[inz(v["x"], v["y"])]
    fig, axes = plt.subplots(1, 3, figsize=(19, 9))
    for ax, (title, val, lab, lo, hi, cmap, tk, tl) in zip(axes, panels):
        show(ax, R.grid(val), R, geom, title, lab, lo, hi, cmap, zoom=ZOOM, ticks=tk, ticklabels=tl)
    fig.suptitle(f"{ZOOM['name']} and surroundings — the same three maps at neighbourhood scale", x=0.01, ha="left",
                 fontsize=11, y=0.995)
    axes[0].scatter(e["x"], e["y"], s=4, marker="x", c="black", linewidths=0.4, alpha=0.45,
                    label=f"311 dumping reports, {main} ({len(e):,})")
    axes[2].scatter(vz["x"], vz["y"], s=3, marker="o", facecolors="#ffd400", edgecolors="black", linewidths=0.25,
                    alpha=0.8, label=f"survey visits, 2022–23 ({len(vz):,})")
    for ax in (axes[0], axes[2]):
        ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9, edgecolor=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(FIG, "fig3b_zoom.png"), dpi=170); plt.close(fig)

    # A2: the two-year window, the correction factor, w_R — pink = chatty places, green = quiet, in both
    cf = np.log(L["correction_factor"]); cf_lim = sym_lim(cf, q=0.99)
    z = np.load(os.path.join(OUT, "wR_raster.npz")); W = z["wR"]; w_lim = sym_lim(W)
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    show(axes[0, 0], R.grid(Lo["log_rel_raw"]), R, geom, f"311 as filed, {other} average", lab_rate, r_lo, r_hi, CMAP_DUMPING, ticks=r_ticks, ticklabels=r_labels)
    show(axes[0, 1], R.grid(Lo["log_rel_corr"]), R, geom, f"corrected, {other} average", lab_rate, r_lo, r_hi, CMAP_DUMPING, ticks=r_ticks, ticklabels=r_labels)
    show(axes[1, 0], R.grid(cf), R, geom, f"correction factor, {main}: log(corrected / raw)",
         "pink = reports discounted (chatty neighbourhoods); green = scaled up (quiet ones)", -cf_lim, cf_lim, CMAP_FACTOR)
    im = axes[1, 1].imshow(W, origin="lower", extent=[z["x"][0], z["x"][-1], z["y"][0], z["y"][-1]], cmap=CMAP_PROPENSITY,
                           vmin=-w_lim, vmax=w_lim, interpolation="nearest")
    map_style(axes[1, 1], geom, f"reporting propensity w_R from street poles (h_w = {HW0} m)")
    cb = plt.colorbar(im, ax=axes[1, 1], orientation="horizontal", shrink=0.75, pad=0.08, aspect=35)
    cb.set_label("log outage reports per pole, centred   (pink = chatty, green = quiet)", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "A2_windows_and_correction.png"), dpi=150); plt.close(fig)

    # A1: the sigmoid snapshot (appendix) — litter colours
    E = snap["in"]["E_rating_corr"]; e_mean = float(np.nanmean(E)); e_lim = sym_lim(E - e_mean)
    d = snap["in"]["E_rating_corr"] - snap["in"]["E_rating_raw"]; d_lim = sym_lim(d, q=0.99)
    fig, axes = plt.subplots(1, 3, figsize=(19, 8))
    show(axes[0], R.grid(E - e_mean), R, geom, f"311 → E[rating] through the calibrated sigmoid, {DATE_IN.date()}, corrected",
         f"E[rating] − citywide mean ({e_mean:.2f}), rating points", -e_lim, e_lim, CMAP_LITTER)
    show(axes[1], R.grid(d), R, geom, "effect of the correction on E[rating]: corrected − raw", "rating points", -d_lim, d_lim, CMAP_LITTER)
    show(axes[2], R.grid(snap["out"]["E_rating_corr"] - e_mean), R, geom, f"E[rating], {DATE_OUT.date()} — EXTRAPOLATION",
         f"E[rating] − {e_mean:.2f}, rating points", -e_lim, e_lim, CMAP_LITTER)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "A1_rating_snapshot.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
