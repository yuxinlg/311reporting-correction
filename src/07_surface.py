"""
07_surface.py
=============
The deliverable: a citywide predicted-litter-index surface from 311 data.

For every 50 m cell of the built support (inside a CBG, within 75 m of a
street pole; parks, rivers, airport and rail yards are NaN):

    lam_corr(s, t) = sum_j exp(-beta_hat w_R(s_j)) K_h(|s - s_j|) e^{-(t-t_j)/tau} 1[t >= t_j]
                     / sum_c K_h(|s - c|) E_c
    eta(s)         = g(log lam_corr) + mean month/surveyor effect
    E[rating](s)   = 1 + sum_k sigmoid(eta - kappa_k)        (the 1-4 index)
    P(rating >= 3) = sigmoid(eta - kappa_2)

g, the cutpoints and the FE come from the S2 ordinal model refit on the survey
visits at beta_hat (the profile-likelihood estimate from 05), with h, tau,
alpha as in 05. The raw-field (S0) surface is produced alongside so the
correction's effect, E_corr - E_raw, can be mapped.

Two evaluation dates: DATE_IN (inside the 2022-23 calibration window) and
DATE_OUT (end of 2025, an EXTRAPOLATION: 38% of the reports and both of
those years lie outside the calibration window; the surface there is a
statement about report rates, not a validated litter measurement).

Outputs
  output/surface/litter_index_<date>.tif      GeoTIFF, EPSG:26918, 50 m, bands:
      1 E_rating_corr  2 E_rating_raw  3 P_ge3_corr  4 log_lam_corr  5 log_lam_raw
  output/surface/surface_cbg.csv              CBG means of the bands + observed mean rating
  output/surface/summary.json
  output/figures/07_surface_*.png
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
from patsy import dmatrix, build_design_matrices
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import kernel_sum, accumulate_st, shrunk_log
from ordinal import OrderedLogit

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
SUR = os.path.join(OUT, "surface")
FIG = os.path.join(OUT, "figures")
os.makedirs(SUR, exist_ok=True)

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--h", type=int, default=800, help="quartic radius (m); 800 = beta headline, 1200 = CV choice")
_ap.add_argument("--tau", type=int, default=120)
_ap.add_argument("--alpha", type=float, default=1.0)
_ARGS, _ = _ap.parse_known_args()

CRS = 26918
EPOCH = pd.Timestamp("2021-01-01")
GRID_RES = 50
H, TAU, ALPHA = _ARGS.h, _ARGS.tau, _ARGS.alpha
TAG = f"h{H}_t{TAU}_a{ALPHA:g}"
SPLINE_DF = 5
WR_PRIMARY = "wR_pre24_naive_h600"
DATE_IN, DATE_OUT = pd.Timestamp("2023-06-30"), pd.Timestamp("2025-12-31")
FE_MAIN = ("month", "surv")

C1, C2 = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
OL = OrderedLogit(4)


def spline_basis(x, info=None):
    if info is None:
        B = dmatrix(f"cr(x, df={SPLINE_DF}) - 1", {"x": x}, return_type="dataframe")
        info = {"di": B.design_info, "mean": B.to_numpy().mean(0)}
        Bm = B.to_numpy()
    else:
        Bm = np.asarray(build_design_matrices([info["di"]], {"x": x})[0])
    return (Bm - info["mean"])[:, 1:], info


def fe_dummies(v):
    cols = [pd.get_dummies(v[f], drop_first=True).to_numpy(float) for f in FE_MAIN]
    return np.hstack(cols)


def fit_visit_model(v, F, ev, beta_hat):
    """S2 (spline g + FE) at beta_hat and S0, both on the survey visits; returns
    what the grid needs: theta, spline info, the mean FE contribution."""
    y = v["rating"].to_numpy() - 1
    den = F[f"den_expo_h{H}"].to_numpy()
    o = accumulate_st(v[["x", "y"]].to_numpy(), v["t_days"].to_numpy(), ev["xy"], ev["t"], H, [TAU],
                      {"m": np.exp(-beta_hat * ev["wk"]), "one": np.ones(len(ev["t"]))}, src_tree=ev["tree"])
    lam_bar = {}
    models = {}
    Xfe = fe_dummies(v)
    for label, key in [("corr", "m"), ("raw", "one")]:
        x, lb = shrunk_log(o[(TAU, key)], den, ALPHA)
        lam_bar[label] = lb
        B, info = spline_basis(x)
        X = np.hstack([B, Xfe])
        th, ll, ok = OL.fit(X, y)
        p = X.shape[1]; beta, cuts = OL.unpack(th, p)
        ng = B.shape[1]
        models[label] = {"theta": th, "info": info, "g_coef": beta[:ng], "cuts": cuts,
                         "fe_mean": float((Xfe @ beta[ng:]).mean()), "ll": ll, "x_range": (x.min(), x.max()),
                         "x_q": np.quantile(x, [0.005, 0.995])}
        print(f"  visit model {label}: ll {ll:.1f}, cutpoints {np.round(cuts, 3)}, mean FE {models[label]['fe_mean']:+.3f}")
    return models, lam_bar


def predict(model, x):
    B, _ = spline_basis(x, model["info"])
    eta = B @ model["g_coef"] + model["fe_mean"]
    cuts = model["cuts"]
    Er = 1.0 + expit(eta[:, None] - cuts[None, :]).sum(1)
    p_ge3 = expit(eta - cuts[1])
    return Er, p_ge3, eta


def main():
    t_start = time.time()
    print("=" * 68); print("07_surface.py — the citywide litter-index surface"); print("=" * 68)
    with open(os.path.join(OUT, "calib", "summary.json")) as f:
        cal = json.load(f)
    prof = [r for r in cal["profile"] if r["h"] == H and r["w_R"] == "k" and r["g"] == "spline"
            and r["fe"] == "+".join(FE_MAIN) and float(r.get("alpha", ALPHA)) == ALPHA]
    if not prof:
        raise SystemExit(f"no profile estimate for h={H}, alpha={ALPHA} in calib/summary.json (05 ran h in {sorted({r['h'] for r in cal['profile']})})")
    beta_hat = float(prof[0]["beta_hat"])
    ci = cal["bootstrap"]["profile_ci_recalibrated"] if H == cal["h0"] else [np.nan, np.nan]
    print(f"  beta_hat = {beta_hat:.3f}" + (f" (bootstrap-recalibrated 95% CI [{ci[0]:.2f}, {ci[1]:.2f}])" if H == cal["h0"]
          else " (bootstrap CI only available at h = %d)" % cal["h0"]) + f", h = {H}, tau = {TAU}, alpha = {ALPHA}")

    v = pd.read_parquet(os.path.join(OUT, "litter_panel.parquet"))
    F = pd.read_parquet(os.path.join(OUT, "visit_fields.parquet"))
    d = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    w = pd.read_parquet(os.path.join(OUT, "wR_events.parquet"))
    wk = w[WR_PRIMARY].to_numpy(float)
    ev = {"xy": d[["x", "y"]].to_numpy(float), "t": d["t_days"].to_numpy(float),
          "wk": np.where(np.isfinite(wk), wk, 0.0)}
    ev["tree"] = cKDTree(ev["xy"])
    sup = pd.read_parquet(os.path.join(OUT, "support_grid.parquet"))
    sxy, sE = sup[["x", "y"]].to_numpy(), sup["E_cell"].to_numpy()
    stree = cKDTree(sxy)
    print(f"  {len(sup):,} support cells at {GRID_RES} m; {len(ev['t']):,} reports")

    print("\n[model] refitting S2 and S0 on the visits at beta_hat ...")
    models, lam_bar = fit_visit_model(v, F, ev, beta_hat)

    print("\n[grid] denominators and fields ...")
    t0 = time.time()
    den = kernel_sum(sxy, sxy, H, weights=sE, source_tree=stree)
    summary = {"beta_hat": beta_hat, "ci": ci, "h": H, "tau": TAU, "alpha": ALPHA,
               "n_cells": int(len(sup)), "dates": {}}
    geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson")); geom["GEOID"] = geom["GEOID"].astype(str)
    xmin, ymin, xmax, ymax = geom.total_bounds
    gx = np.arange(xmin + GRID_RES / 2, xmax, GRID_RES); gy = np.arange(ymin + GRID_RES / 2, ymax, GRID_RES)
    ix = np.round((sxy[:, 0] - gx[0]) / GRID_RES).astype(int); iy = np.round((sxy[:, 1] - gy[0]) / GRID_RES).astype(int)
    transform = from_origin(gx[0] - GRID_RES / 2, gy[-1] + GRID_RES / 2, GRID_RES, GRID_RES)
    obs_cbg = v.groupby("GEOID")["rating"].mean()
    cbg_rows = []
    maps = {}
    for label, date in [("in", DATE_IN), ("out", DATE_OUT)]:
        t_eval = (date - EPOCH).days
        o = accumulate_st(sxy, np.full(len(sxy), float(t_eval)), ev["xy"], ev["t"], H, [TAU],
                          {"m": np.exp(-beta_hat * ev["wk"]), "one": np.ones(len(ev["t"]))}, src_tree=ev["tree"])
        bands = {}
        for which, key in [("corr", "m"), ("raw", "one")]:
            x, _ = shrunk_log(o[(TAU, key)], den, ALPHA, lam_bar=lam_bar[which])
            Er, p3, eta = predict(models[which], x)
            bands[f"E_rating_{which}"] = Er; bands[f"log_lam_{which}"] = x
            if which == "corr":
                bands["P_ge3_corr"] = p3
            lo, hi = models[which]["x_range"]
            summary["dates"].setdefault(date.date().isoformat(), {})[which] = {
                "E_rating_p1_p50_p99": [float(q) for q in np.quantile(Er, [.01, .5, .99])],
                "share_cells_outside_visit_x_range": float(((x < lo) | (x > hi)).mean()),
                "zero_numerators": float((o[(TAU, key)] == 0).mean())}
        delta = bands["E_rating_corr"] - bands["E_rating_raw"]
        summary["dates"][date.date().isoformat()]["correction_delta_p1_p50_p99"] = [float(q) for q in np.quantile(delta, [.01, .5, .99])]
        summary["dates"][date.date().isoformat()]["correction_delta_sd"] = float(delta.std())
        # GeoTIFF
        order = ["E_rating_corr", "E_rating_raw", "P_ge3_corr", "log_lam_corr", "log_lam_raw"]
        arr = np.full((len(order), len(gy), len(gx)), np.nan, dtype="float32")
        for b, name in enumerate(order):
            arr[b, iy, ix] = bands[name]
        arr = arr[:, ::-1, :]                                         # north up
        path = os.path.join(SUR, f"litter_index_{TAG}_{date.date().isoformat()}.tif")
        with rasterio.open(path, "w", driver="GTiff", height=len(gy), width=len(gx), count=len(order),
                           dtype="float32", crs=f"EPSG:{CRS}", transform=transform, nodata=np.nan,
                           compress="deflate") as dst:
            dst.write(arr)
            for b, name in enumerate(order):
                dst.set_band_description(b + 1, name)
            dst.update_tags(beta_hat=beta_hat, h_m=H, tau_days=TAU, alpha=ALPHA, date=date.date().isoformat(),
                            extrapolation=str(label == "out"))
        maps[label] = {"E": arr[0], "delta": arr[0] - arr[1]}
        # CBG means
        cb = pd.DataFrame({"GEOID": sup["GEOID"], **{k: bands[k] for k in order}}).groupby("GEOID").mean()
        cb["n_cells"] = sup.groupby("GEOID").size()
        cb["date"] = date.date().isoformat(); cb["obs_mean_rating_2022_23"] = obs_cbg.reindex(cb.index)
        cbg_rows.append(cb.reset_index())
        okc = cb["obs_mean_rating_2022_23"].notna()
        rs = spearmanr(cb.loc[okc, "obs_mean_rating_2022_23"], cb.loc[okc, "E_rating_corr"]).correlation
        rs0 = spearmanr(cb.loc[okc, "obs_mean_rating_2022_23"], cb.loc[okc, "E_rating_raw"]).correlation
        summary["dates"][date.date().isoformat()]["cbg_spearman_obs_vs_E_corr"] = float(rs)
        summary["dates"][date.date().isoformat()]["cbg_spearman_obs_vs_E_raw"] = float(rs0)
        s = summary["dates"][date.date().isoformat()]
        print(f"  {date.date()} ({'calibration window' if label == 'in' else 'EXTRAPOLATION'}): E[rating] corr "
              f"p1/p50/p99 {s['corr']['E_rating_p1_p50_p99'][0]:.2f}/{s['corr']['E_rating_p1_p50_p99'][1]:.2f}/{s['corr']['E_rating_p1_p50_p99'][2]:.2f}; "
              f"correction Δ sd {s['correction_delta_sd']:.3f}, p1/p99 {s['correction_delta_p1_p50_p99'][0]:+.2f}/{s['correction_delta_p1_p50_p99'][2]:+.2f}; "
              f"cells outside visit range {s['corr']['share_cells_outside_visit_x_range']:.1%}; "
              f"CBG Spearman with observed mean rating: corr {rs:.3f}, raw {rs0:.3f}   ({time.time() - t0:.0f} s)")
    pd.concat(cbg_rows).to_csv(os.path.join(SUR, f"surface_cbg_{TAG}.csv"), index=False)
    with open(os.path.join(SUR, f"summary_{TAG}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if H == cal["h0"] and TAU == cal["tau0"] and ALPHA == cal["alpha"]:      # the headline configuration keeps the plain names too
        pd.concat(cbg_rows).to_csv(os.path.join(SUR, "surface_cbg.csv"), index=False)
        with open(os.path.join(SUR, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    _figures(maps, gx, gy, geom, beta_hat)
    print(f"\n  Saved → output/surface/litter_index_{TAG}_*.tif, surface_cbg_{TAG}.csv, summary_{TAG}.json; figures 07_surface_*_{TAG}.png")
    print(f"Done in {time.time() - t_start:.0f} s.")


def _figures(maps, gx, gy, geom, beta_hat):
    ext = [gx[0] - GRID_RES / 2, gx[-1] + GRID_RES / 2, gy[0] - GRID_RES / 2, gy[-1] + GRID_RES / 2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    E = maps["in"]["E"]
    im = axes[0].imshow(E, origin="upper", extent=ext, cmap="Blues", vmin=np.nanquantile(E, .01), vmax=np.nanquantile(E, .99))
    geom.boundary.plot(ax=axes[0], color=GRID, linewidth=0.15)
    axes[0].set_title(f"Predicted litter index E[rating], {DATE_IN.date()}, corrected (β̂ = {beta_hat:.2f})", fontsize=10, loc="left")
    plt.colorbar(im, ax=axes[0], shrink=0.6)
    D = maps["in"]["delta"]; lim = np.nanquantile(np.abs(D), .99)
    im = axes[1].imshow(D, origin="upper", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
    geom.boundary.plot(ax=axes[1], color=GRID, linewidth=0.15)
    axes[1].set_title("Effect of the reporting correction: E[rating] corrected − raw", fontsize=10, loc="left")
    plt.colorbar(im, ax=axes[1], shrink=0.6)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"07_surface_map_{TAG}.png"), dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    E2 = maps["out"]["E"]
    im = axes[0].imshow(E2, origin="upper", extent=ext, cmap="Blues", vmin=np.nanquantile(E, .01), vmax=np.nanquantile(E, .99))
    geom.boundary.plot(ax=axes[0], color=GRID, linewidth=0.15)
    axes[0].set_title(f"E[rating], {DATE_OUT.date()} — EXTRAPOLATION beyond the 2022–23 calibration", fontsize=10, loc="left")
    plt.colorbar(im, ax=axes[0], shrink=0.6)
    dd = E2 - E; lim = np.nanquantile(np.abs(dd), .99)
    im = axes[1].imshow(dd, origin="upper", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
    geom.boundary.plot(ax=axes[1], color=GRID, linewidth=0.15)
    axes[1].set_title(f"Change {DATE_IN.date()} → {DATE_OUT.date()} (report-rate change through the sigmoid)", fontsize=10, loc="left")
    plt.colorbar(im, ax=axes[1], shrink=0.6)
    for ax in axes:
        ax.set_axis_off()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"07_surface_extrapolation_{TAG}.png"), dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
