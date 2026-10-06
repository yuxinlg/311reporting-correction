"""
06_surface.py
=============
The deliverable: the corrected litter-index surface on the built support.

For every 50 m cell of the support grid (inside a block group, within 75 m of a
pole):
    λ̂_β̂(s, t) = Σ_j exp(−β̂ w_R(s_j)) K_H0(|s − s_j|) e^{−(t − t_j)/τ} 1[t ≥ t_j]  /  Σ_c K_H0(|s − c|) E_c
    η(s)       = g(log λ̂_β̂) + mean month/surveyor effect
    E[rating]  = 1 + Σ_k σ(η − κ_k)                                    the 1–4 index
g, κ and the fixed effects come from the rating model refit on the survey
visits at β̂ (04). The raw-field (β = 0) surface is produced alongside so the
effect of the correction can be mapped. Two dates: DATE_IN (inside the survey
window) and DATE_OUT (an extrapolation: the model is calibrated on 2022–23).

Writes  output/surface/litter_index_{date}.tif   GeoTIFF, EPSG:26918, 50 m; bands
            1 E_rating_corr  2 E_rating_raw  3 log_lam_corr  4 log_lam_raw
        output/surface/surface_cbg.csv            block-group means + observed 2022–23 mean rating
        output/surface/summary.json
        output/figures/fig3_maps.png
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

from config import (OUT, FIG, FIELDS, SURF, CRS, EPOCH, GRID_RES, TAU, ALPHA, BETA_GRID, DATE_IN, DATE_OUT,
                    GRID, load_choices, ensure_dirs)
from kernels import kernel_sum, accumulate_st, shrunk_log
from ordinal import OrderedLogit
from design import FE, design, spline_basis

warnings.filterwarnings("ignore")
OL = OrderedLogit(4)
I0 = int(np.argmin(np.abs(BETA_GRID)))
BANDS = ["E_rating_corr", "E_rating_raw", "log_lam_corr", "log_lam_raw"]


def fit_visit_models(v, F, ev_xy, ev_t, tree, m_hat, H0):
    """Refit the rating model on the visits for the corrected (β̂) and raw (β = 0) fields."""
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
        models[which] = {"info": info, "g": beta[:ng], "cuts": cuts, "fe_mean": float((fe[0] @ beta[ng:]).mean()),
                         "ll": ll, "x_range": (float(x[which].min()), float(x[which].max()))}
        print(f"  visit model {which}: ℓ {ll:.1f}, cutpoints {np.round(cuts, 2)}, mean FE {models[which]['fe_mean']:+.3f}")
    return models, lam_raw


def predict(model, x):
    B, _ = spline_basis(x, model["info"])
    eta = B @ model["g"] + model["fe_mean"]
    return 1.0 + expit(eta[:, None] - model["cuts"][None, :]).sum(1)


def main():
    t_start = time.time()
    print("=" * 68); print("06_surface.py — the citywide litter-index surface"); print("=" * 68)
    ensure_dirs()
    ch = load_choices(); H0, HW0, beta_hat = ch["H0"], ch["HW0"], float(ch["beta_hat"])
    print(f"  β̂ = {beta_hat:.3f} (95% CI {ch['ci_95'][0]:.2f}–{ch['ci_95'][1]:.2f}), h = {H0}, τ = {TAU}, α = {ALPHA}")
    v = pd.read_parquet(os.path.join(OUT, "visits.parquet"))
    F = dict(np.load(os.path.join(FIELDS, f"h{H0}.npz")))
    ev = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    wr = pd.read_parquet(os.path.join(OUT, "wR_events.parquet"))[f"wR_h{HW0}"].to_numpy()
    m_hat = np.exp(-beta_hat * np.where(np.isfinite(wr), wr, 0.0))
    ev_xy, ev_t = ev[["x", "y"]].to_numpy(), ev["t_days"].to_numpy(); tree = cKDTree(ev_xy)
    sup = pd.read_parquet(os.path.join(OUT, "support_grid.parquet"))
    sxy, sE = sup[["x", "y"]].to_numpy(), sup["E_cell"].to_numpy()

    print("\n[model] refitting the rating model on the visits")
    models, lam_raw = fit_visit_models(v, F, ev_xy, ev_t, tree, m_hat, H0)

    print("\n[grid] fields on the support cells")
    den = kernel_sum(sxy, sxy, H0, weights=sE, source_tree=cKDTree(sxy))
    geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson")); geom["GEOID"] = geom["GEOID"].astype(str)
    xmin, ymin, xmax, ymax = geom.total_bounds
    gx = np.arange(xmin + GRID_RES / 2, xmax, GRID_RES); gy = np.arange(ymin + GRID_RES / 2, ymax, GRID_RES)
    ix = np.round((sxy[:, 0] - gx[0]) / GRID_RES).astype(int); iy = np.round((sxy[:, 1] - gy[0]) / GRID_RES).astype(int)
    transform = from_origin(gx[0] - GRID_RES / 2, gy[-1] + GRID_RES / 2, GRID_RES, GRID_RES)
    obs = v.groupby("GEOID")["rating"].mean()
    summary = {"beta_hat": beta_hat, "ci_95": ch["ci_95"], "h": H0, "tau": TAU, "alpha": ALPHA, "n_cells": int(len(sup)), "dates": {}}
    cbg_rows, maps = [], {}
    for label, date in [("in", DATE_IN), ("out", DATE_OUT)]:
        t0 = time.time()
        t_eval = float((date - EPOCH).days)
        o = accumulate_st(sxy, np.full(len(sxy), t_eval), ev_xy, ev_t, H0, [TAU], {"m": m_hat, "one": np.ones(len(ev_t))}, src_tree=tree)
        x_corr = shrunk_log(o[(TAU, "m")], den, ALPHA, lam_raw * m_hat.mean())[0]
        x_raw = shrunk_log(o[(TAU, "one")], den, ALPHA, lam_raw)[0]
        bands = {"E_rating_corr": predict(models["corr"], x_corr), "E_rating_raw": predict(models["raw"], x_raw),
                 "log_lam_corr": x_corr, "log_lam_raw": x_raw}
        arr = np.full((len(BANDS), len(gy), len(gx)), np.nan, dtype="float32")
        for b, name in enumerate(BANDS):
            arr[b, iy, ix] = bands[name]
        arr = arr[:, ::-1, :]
        path = os.path.join(SURF, f"litter_index_{date.date().isoformat()}.tif")
        with rasterio.open(path, "w", driver="GTiff", height=len(gy), width=len(gx), count=len(BANDS), dtype="float32",
                           crs=f"EPSG:{CRS}", transform=transform, nodata=np.nan, compress="deflate") as dst:
            dst.write(arr)
            for b, name in enumerate(BANDS):
                dst.set_band_description(b + 1, name)
            dst.update_tags(beta_hat=beta_hat, h_m=H0, tau_days=TAU, alpha=ALPHA, date=date.date().isoformat(),
                            extrapolation=str(label == "out"))
        maps[label] = {"E": arr[0], "delta": arr[0] - arr[1]}
        cb = pd.DataFrame({"GEOID": sup["GEOID"], **bands}).groupby("GEOID").mean()
        cb["date"] = date.date().isoformat(); cb["obs_mean_rating_2022_23"] = obs.reindex(cb.index)
        cbg_rows.append(cb.reset_index())
        okc = cb["obs_mean_rating_2022_23"].notna()
        rs = {w: float(spearmanr(cb.loc[okc, "obs_mean_rating_2022_23"], cb.loc[okc, f"E_rating_{w}"]).correlation) for w in ("corr", "raw")}
        delta = bands["E_rating_corr"] - bands["E_rating_raw"]
        lo, hi = models["corr"]["x_range"]
        summary["dates"][date.date().isoformat()] = {
            "extrapolation": label == "out",
            "E_rating_corr_p1_p50_p99": [float(q) for q in np.quantile(bands["E_rating_corr"], [.01, .5, .99])],
            "correction_delta_sd": float(delta.std()), "correction_delta_p1_p99": [float(q) for q in np.quantile(delta, [.01, .99])],
            "share_cells_outside_visit_range": float(((x_corr < lo) | (x_corr > hi)).mean()),
            "cbg_spearman_obs_vs_corr": rs["corr"], "cbg_spearman_obs_vs_raw": rs["raw"]}
        s = summary["dates"][date.date().isoformat()]
        print(f"  {date.date()} ({'survey window' if label == 'in' else 'EXTRAPOLATION'}): E[rating] p1/p50/p99 "
              f"{s['E_rating_corr_p1_p50_p99'][0]:.2f}/{s['E_rating_corr_p1_p50_p99'][1]:.2f}/{s['E_rating_corr_p1_p50_p99'][2]:.2f}; "
              f"correction Δ sd {s['correction_delta_sd']:.3f} (p1/p99 {s['correction_delta_p1_p99'][0]:+.2f}/{s['correction_delta_p1_p99'][1]:+.2f}); "
              f"block-group Spearman with observed mean rating: corrected {rs['corr']:.3f}, raw {rs['raw']:.3f}   ({time.time() - t0:.0f} s)")
    pd.concat(cbg_rows).to_csv(os.path.join(SURF, "surface_cbg.csv"), index=False)
    with open(os.path.join(SURF, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # figure: w_R, corrected index, effect of the correction
    z = np.load(os.path.join(OUT, "wR_raster.npz"))
    ext = [gx[0] - GRID_RES / 2, gx[-1] + GRID_RES / 2, gy[0] - GRID_RES / 2, gy[-1] + GRID_RES / 2]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    W = z["wR"]
    im = axes[0].imshow(W, origin="lower", extent=[z["x"][0], z["x"][-1], z["y"][0], z["y"][-1]], cmap="Blues",
                        vmin=np.nanquantile(W, .02), vmax=np.nanquantile(W, .98))
    axes[0].set_title(f"(a) reporting propensity w_R from poles (h_w = {HW0} m)", fontsize=10, loc="left")
    plt.colorbar(im, ax=axes[0], shrink=0.6)
    E = maps["in"]["E"]
    im = axes[1].imshow(E, origin="upper", extent=ext, cmap="Blues", vmin=np.nanquantile(E, .01), vmax=np.nanquantile(E, .99))
    axes[1].set_title(f"(b) corrected litter index E[rating], {DATE_IN.date()} (β̂ = {beta_hat:.2f})", fontsize=10, loc="left")
    plt.colorbar(im, ax=axes[1], shrink=0.6)
    D = maps["in"]["delta"]; lim = np.nanquantile(np.abs(D), .99)
    im = axes[2].imshow(D, origin="upper", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[2].set_title("(c) effect of the correction: corrected − raw", fontsize=10, loc="left")
    plt.colorbar(im, ax=axes[2], shrink=0.6)
    for ax in axes:
        geom.boundary.plot(ax=ax, color=GRID, linewidth=0.15); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_maps.png"), dpi=170); plt.close(fig)
    print(f"\nDone in {time.time() - t_start:.0f} s. Saved → output/surface/litter_index_*.tif, surface_cbg.csv, summary.json; figures/fig3_maps.png")


if __name__ == "__main__":
    main()
