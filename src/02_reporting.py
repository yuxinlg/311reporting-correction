"""
02_reporting.py
===============
Stage 1: the reporting-propensity surface w_R, built from street poles.

    w_R(s) = log[ Σ_outages K_hw(|s - s_j|) / Σ_poles K_hw(|s - s_p|) ]  −  mean over poles

Street-light-outage reports (OUTAGE_WINDOW) per pole, kernel-smoothed with a
quartic of radius h_w, Gamma-shrunk toward the citywide rate (KAPPA prior
reports), on the log scale, centred. Identifying assumption: true breakage per
pole is uniform citywide, so spatial variation in outage reports per pole is
reporting behaviour, not broken lights.

h_w is chosen by a stated rule, not by hand: the smallest radius in HW_SWEEP
whose Poisson-noise reliability at the survey visits is >= HW_RELIABILITY,
    reliability = 1 − noise_sd² / signal_sd²,
    noise_sd    = median over visits of 1 / sqrt(n_eff),  n_eff = Kish effective
                  number of outage reports inside the kernel (the Poisson floor
                  of a log count),
    signal_sd   = sd of w_R over the visits.
HW_ALT is the next smaller radius (sensitivity row in 04).

Writes
  output/wR_events.parquet   w_R at every dumping report   (wR_h{HW0}, wR_h{HW_ALT})
  output/wR_visits.parquet   w_R at every survey visit     (same columns)
  output/wR_raster.npz       100 m raster of w_R at HW0, NaN outside the city
  output/choices.json        HW0, HW_ALT, the reliability table
  output/figures/02_wR_map.png
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (OUT, FIG, CRS, OUTAGE_WINDOW, HW_SWEEP, HW_RELIABILITY, KAPPA, WR_RASTER_RES,
                    C1, MUTED, CMAP_PROPENSITY, style, map_style, update_choices, ensure_dirs)
from kernels import kernel_sum, SD_FACTOR

warnings.filterwarnings("ignore")


def w_surface(qxy, out_xy, out_tree, pole_xy, pole_tree, h, rbar):
    """log(shrunk outage rate per pole / citywide rate) at the query points, plus n_eff."""
    num, neff = kernel_sum(qxy, out_xy, h, source_tree=out_tree, return_neff=True)
    den = kernel_sum(qxy, pole_xy, h, source_tree=pole_tree)
    rate = (num + KAPPA) / (den + KAPPA / rbar)
    return np.log(rate / rbar), neff


def main():
    t0 = time.time()
    print("=" * 68); print("02_reporting.py — Stage 1: reporting propensity from poles"); print("=" * 68)
    ensure_dirs()
    poles = pd.read_parquet(os.path.join(OUT, "poles_xy.parquet"))
    pole_xy = poles[["x", "y"]].to_numpy(); pole_tree = cKDTree(pole_xy)
    r = pd.read_parquet(os.path.join(OUT, "points_outage.parquet"))
    r = r[(r["ts"] >= OUTAGE_WINDOW[0]) & (r["ts"] < OUTAGE_WINDOW[1])]
    out_xy = r[["x", "y"]].to_numpy(); out_tree = cKDTree(out_xy)
    rbar = len(r) / len(poles)                       # citywide outage reports per pole
    ev = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    v = pd.read_parquet(os.path.join(OUT, "visits.parquet"))
    ev_xy, v_xy = ev[["x", "y"]].to_numpy(), v[["x", "y"]].to_numpy()
    print(f"  {len(r):,} outage reports {OUTAGE_WINDOW[0].date()} → {OUTAGE_WINDOW[1].date()} over {len(poles):,} poles "
          f"({rbar:.3f} per pole); queries: {len(ev):,} dumping reports, {len(v):,} visits")

    # --- choose h_w: reliability at the visits -------------------------------
    print("\n[h_w] Poisson-noise reliability at the survey visits")
    rows, w_vis = [], {}
    for h in HW_SWEEP:
        w, neff = w_surface(v_xy, out_xy, out_tree, pole_xy, pole_tree, h, rbar)
        noise = float(np.median(1.0 / np.sqrt(np.maximum(neff, 1.0))))
        signal = float(w.std())
        rows.append({"h_w": h, "sd_signal": signal, "sd_noise": noise, "reliability": 1 - noise ** 2 / signal ** 2,
                     "median_n_eff": float(np.median(neff))})
        w_vis[h] = w
        print(f"  h_w = {h:>4} m (sd {h * SD_FACTOR:.0f} m): signal sd {signal:.3f}, noise sd {noise:.3f}, "
              f"reliability {rows[-1]['reliability']:.3f}, median n_eff {np.median(neff):.0f}")
    tab = pd.DataFrame(rows)
    ok = tab[tab["reliability"] >= HW_RELIABILITY]
    HW0 = int(ok["h_w"].min()) if len(ok) else int(tab["h_w"].max())
    i0 = HW_SWEEP.index(HW0)
    HW_ALT = HW_SWEEP[i0 - 1] if i0 > 0 else HW_SWEEP[i0 + 1]
    print(f"  rule: smallest h_w with reliability >= {HW_RELIABILITY} → HW0 = {HW0} m; sensitivity HW_ALT = {HW_ALT} m")

    # --- the surface at every query set, centred on the mean over poles -------
    print("\n[w_R] surfaces")
    centre = {}
    for h in (HW0, HW_ALT):
        wp, _ = w_surface(pole_xy, out_xy, out_tree, pole_xy, pole_tree, h, rbar)
        centre[h] = float(wp.mean())
    w_ev = {h: w_surface(ev_xy, out_xy, out_tree, pole_xy, pole_tree, h, rbar)[0] - centre[h] for h in (HW0, HW_ALT)}
    w_v = {h: w_vis[h] - centre[h] for h in (HW0, HW_ALT)}
    pd.DataFrame({f"wR_h{h}": w_ev[h] for h in (HW0, HW_ALT)}).to_parquet(os.path.join(OUT, "wR_events.parquet"), index=False)
    pd.DataFrame({f"wR_h{h}": w_v[h] for h in (HW0, HW_ALT)}).to_parquet(os.path.join(OUT, "wR_visits.parquet"), index=False)
    print(f"  at dumping reports: sd {w_ev[HW0].std():.3f} (h_w {HW0}), {w_ev[HW_ALT].std():.3f} (h_w {HW_ALT}); "
          f"correlation between the two {np.corrcoef(w_ev[HW0], w_ev[HW_ALT])[0, 1]:.3f}")

    # raster for the torus-shift null and the map
    geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson")); geom["GEOID"] = geom["GEOID"].astype(str)
    xmin, ymin, xmax, ymax = geom.total_bounds
    gx = np.arange(xmin + WR_RASTER_RES / 2, xmax, WR_RASTER_RES)
    gy = np.arange(ymin + WR_RASTER_RES / 2, ymax, WR_RASTER_RES)
    GX, GY = np.meshgrid(gx, gy)
    cells = gpd.GeoDataFrame(geometry=gpd.points_from_xy(GX.ravel(), GY.ravel()), crs=CRS)
    hit = cells.sjoin(geom[["geometry"]], how="inner", predicate="within")
    inside = np.zeros(len(cells), bool); inside[np.unique(hit.index.to_numpy())] = True
    Z = np.full(GX.size, np.nan)
    Z[inside] = w_surface(np.c_[GX.ravel()[inside], GY.ravel()[inside]], out_xy, out_tree, pole_xy, pole_tree, HW0, rbar)[0] - centre[HW0]
    Z = Z.reshape(GX.shape)
    np.savez_compressed(os.path.join(OUT, "wR_raster.npz"), x=gx, y=gy, wR=Z, res=WR_RASTER_RES, h_w=HW0)

    # internal cross-check: kernel w_R at block-group centroids vs the block group's own outage rate
    cbg = pd.read_parquet(os.path.join(OUT, "cbg.parquet"))
    cen = geom.geometry.centroid
    wc, _ = w_surface(np.c_[cen.x.to_numpy(), cen.y.to_numpy()], out_xy, out_tree, pole_xy, pole_tree, HW0, rbar)
    cc = pd.DataFrame({"GEOID": geom["GEOID"], "wR_centroid": wc - centre[HW0]}).merge(cbg, on="GEOID")
    cc["R"] = cc["GEOID"].map(r.groupby("GEOID").size()).fillna(0)
    cc = cc[(cc["n_poles"] >= 10) & (cc["R"] >= 5)]
    cc["log_rate"] = np.log(cc["R"] / cc["n_poles"])
    rho = spearmanr(cc["wR_centroid"], cc["log_rate"]).correlation
    print(f"  cross-check: Spearman of kernel w_R at centroids with the block group's own log(outages/poles) = {rho:.3f} "
          f"({len(cc):,} block groups with >= 10 poles and >= 5 reports)")

    update_choices(HW0=HW0, HW_ALT=HW_ALT, hw_reliability=tab.round(4).to_dict(orient="records"),
                   wR_centre=centre, wR_centroid_spearman=float(rho))

    # figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    lim = float(np.nanquantile(np.abs(Z), 0.98))
    im = ax.imshow(Z, origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]], cmap=CMAP_PROPENSITY, vmin=-lim, vmax=lim,
                   interpolation="nearest")
    map_style(ax, geom, f"reporting propensity w_R from street poles, h_w = {HW0} m")
    cb = plt.colorbar(im, ax=ax, orientation="horizontal", shrink=0.75, pad=0.08, aspect=35)
    cb.set_label("log outage reports per pole, centred   (orange = chatty, purple = quiet)", fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=8, colors=MUTED)
    ax = axes[1]
    ax.scatter(cc["log_rate"] - cc["log_rate"].mean(), cc["wR_centroid"], s=6, color=C1, alpha=0.5)
    style(ax, f"Block-group centroids: kernel w_R vs own log rate (Spearman {rho:.2f})",
          "log(outages / poles), centred", "kernel w_R at centroid")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "02_wR_map.png"), dpi=160); plt.close(fig)
    print(f"\nDone in {time.time() - t0:.0f} s. Saved → output/wR_events.parquet, wR_visits.parquet, wR_raster.npz, choices.json")


if __name__ == "__main__":
    main()
