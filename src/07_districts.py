"""
07_districts.py
===============
The same maps and the same calibration check, one planning district at a time.
Center City is the City Planning Commission's "Central" district and South
Philadelphia its "South" district (the stadiums and the Navy Yard are a separate
"Lower South"). Block groups are assigned to a district by their representative
point; support cells, reports and visits follow their block group, so every
number here is a subset of the citywide one.

For every district (table for all 18, figures for DISTRICTS):
  size         block groups, support cells, dumping reports in the main window, survey visits
  propensity   mean w_R at the district's reports (0 = citywide average, > 0 = chatty)
  rates        reports per year per RATE_PER stops, as filed and corrected, from the points
               and the district's exposure (no kernel), as multiples of the citywide rate;
               corrected / raw is the district's correction factor; the cell-level factor
               p10 / p50 / p90 comes from 06's raster
  agreement    Spearman of 06's maps (A as filed, B corrected) with the survey map C over
               the district's block groups and cells: does the correction help here?
  calibration  two views of β inside the district, on the cached kernel sums of 03:
               (i)  the citywide model's gain on the district's visits, with the citywide
                    coefficients: ℓ_d(β̂) − ℓ_d(0) at the grid point nearest β̂; over the 18
                    districts these sum to the citywide Δℓ on the grid
               (ii) the district's own β̂: the profile likelihood refitted on the district's
                    visits alone, with its own month and surveyor effects; the loss
                    ℓ_d(β̂_d) − ℓ_d(β̂) says whether the district is consistent with the
                    citywide value (a loss under ~2 is noise)
  figure       (A) as filed, (B) corrected, (C) survey on its own — 06's rasters on the
               citywide colour scales, zoomed to the district with its outline, reports and
               visits overlaid; (D) w_R; (E) correction factor (ratio); (F) the correction in
               absolute terms, corrected − as filed. The district's own profile curve is kept
               in profile_curves.csv and its numbers in the table, not drawn.

Reads   data/Planning_Districts.geojson; output/{cbg_geom.geojson, support_grid, points_dumping, wR_events,
        visits}.parquet, wR_raster.npz; fields/h{H0}.npz; calib/profile_h{H0}_month+surv_a{α}_hw{HW0}.npz;
        surface/dumping_{window}.tif, litter_survey.tif, surface_cbg.csv
Writes  output/districts/districts.csv (one row per district + citywide), profile_curves.csv,
        cbg_district.csv (GEOID -> district), summary.json
        output/figures/fig4_{district}.png
"""

import os
import json
import time
import warnings
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

from config import (DATA, OUT, FIG, FIELDS, CALIB, SURF, DIST, CRS, ALPHA, BETA_GRID, LONG_WINDOWS, RATE_PER,
                    DISTRICT_FILE, DISTRICT_COL, DISTRICTS, ZOOM_PAD, INK, MUTED, CMAP_LITTER, CMAP_DUMPING,
                    CMAP_PROPENSITY, CMAP_FACTOR, CMAP_DELTA, load_choices, ensure_dirs)
from ordinal import OrderedLogit
from design import FE, design

calib = importlib.import_module("04_calibrate")        # field_log, profile, argmax_parabola
surf = importlib.import_module("06_surface")           # show, sym_lim, rate_scale

warnings.filterwarnings("ignore")
OL = OrderedLogit(4)
I0 = int(np.argmin(np.abs(BETA_GRID)))
MAIN = "2021-2025"


def read_tif(path):
    """Bands by name (north up), the imshow extent and the affine transform."""
    with rasterio.open(path) as src:
        bands = {src.descriptions[i]: src.read(i + 1) for i in range(src.count)}
        b = src.bounds
        return bands, [b.left, b.right, b.bottom, b.top], src.transform


def sample(Z, tr, xy):
    r, c = rowcol(tr, xy[:, 0], xy[:, 1])
    return Z[np.asarray(r), np.asarray(c)]


def rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).correlation) if ok.sum() >= 10 else np.nan


# =============================================================================

def main():
    t_start = time.time()
    print("=" * 68); print("07_districts.py — the maps and the calibration inside one planning district"); print("=" * 68)
    ensure_dirs()
    ch = load_choices(); H0, HW0, beta_hat, ci = ch["H0"], ch["HW0"], float(ch["beta_hat"]), ch["ci_95"]
    i_hat = int(np.argmin(np.abs(BETA_GRID - beta_hat)))
    t0, t1 = LONG_WINDOWS[MAIN]; years = (t1 - t0).days / 365.25

    # ---- districts and the block-group assignment ---------------------------------------------------
    dist = gpd.read_file(os.path.join(DATA, DISTRICT_FILE)).to_crs(CRS)[[DISTRICT_COL, "geometry"]] \
              .rename(columns={DISTRICT_COL: "district"})
    geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson")); geom["GEOID"] = geom["GEOID"].astype(str)
    rep = geom.copy(); rep["geometry"] = geom.representative_point()
    cbg_d = rep.sjoin(dist, how="left", predicate="within")[["GEOID", "district"]]
    cbg_d = cbg_d[~cbg_d.index.duplicated()].reset_index(drop=True)
    cbg_d.to_csv(os.path.join(DIST, "cbg_district.csv"), index=False)
    to_d = cbg_d.set_index("GEOID")["district"]
    names = list(dist["district"])
    print(f"  {len(dist)} planning districts; {cbg_d['district'].notna().sum()} of {len(geom)} block groups assigned by "
          f"representative point; β̂ = {beta_hat:.3f} (grid point {BETA_GRID[i_hat]:g}), h = {H0}, h_w = {HW0}")

    # ---- inputs, each with a district column ---------------------------------------------------------
    sup = pd.read_parquet(os.path.join(OUT, "support_grid.parquet")); sup["district"] = sup["GEOID"].map(to_d)
    ev = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    wr = pd.read_parquet(os.path.join(OUT, "wR_events.parquet"))[f"wR_h{HW0}"].to_numpy()
    ev["wR"] = wr; ev["m"] = np.exp(-beta_hat * np.where(np.isfinite(wr), wr, 0.0)); ev["district"] = ev["GEOID"].map(to_d)
    ev = ev[(ev["ts"] >= t0) & (ev["ts"] < t1)].reset_index(drop=True)
    v = pd.read_parquet(os.path.join(OUT, "visits.parquet")); v["district"] = v["GEOID"].map(to_d)
    y = v["rating"].to_numpy() - 1
    cb = pd.read_csv(os.path.join(SURF, "surface_cbg.csv"), dtype={"GEOID": str}); cb["district"] = cb["GEOID"].map(to_d)

    # 06's rasters, sampled at the support cells
    D, extent, tr = read_tif(os.path.join(SURF, f"dumping_{MAIN}.tif"))
    S, _, _ = read_tif(os.path.join(SURF, "litter_survey.tif"))
    sxy = sup[["x", "y"]].to_numpy()
    cell = pd.DataFrame({"district": sup["district"], "log_rel_raw": sample(D["log_rel_raw"], tr, sxy),
                         "log_rel_corr": sample(D["log_rel_corr"], tr, sxy), "cf": sample(D["correction_factor"], tr, sxy),
                         "rating_adj": sample(S["rating_adj"], tr, sxy)})
    E_city = sup["E_cell"].sum()
    city_raw = len(ev) / E_city / years * RATE_PER
    city_corr = ev["m"].sum() / E_city / years * RATE_PER

    # ---- calibration pieces: the cached kernel sums and the citywide fit ----------------------------
    F = dict(np.load(os.path.join(FIELDS, f"h{H0}.npz"))); F["lam_bar_raw"] = float(F["lam_bar_raw"])
    z = np.load(os.path.join(CALIB, f"profile_h{H0}_month+surv_a{ALPHA:g}_hw{HW0}.npz"))
    assert np.allclose(z["grid"], BETA_GRID)
    fe_city = FE(v).build(v)
    lli = {}
    for i in (I0, i_hat):
        x = calib.field_log(F["num"][:, i], F["den"], F["lam_bar_raw"] * F["mean_m"][i], ALPHA)
        X, _, _, _ = design(x, fe_city)
        lli[i] = OL.loglik_i(z["thetas"][i], X, y)
    gain_i = lli[i_hat] - lli[I0]
    dll_city_grid = float(z["ll"][i_hat] - z["ll"][I0])
    print(f"  citywide model: ℓ({BETA_GRID[i_hat]:g}) − ℓ(0) = {gain_i.sum():+.1f} per-visit sum vs {dll_city_grid:+.1f} "
          f"from the cached profile (closure check)")

    # ---- one row per district ----------------------------------------------------------------------
    print(f"\n[districts] β profile refitted on each district's visits alone ({len(BETA_GRID)} fits each)")
    rows, curves = [], {}
    for d in names + ["citywide"]:
        t = time.time()
        md = np.ones(len(v), bool) if d == "citywide" else (v["district"] == d).to_numpy()
        me = np.ones(len(ev), bool) if d == "citywide" else (ev["district"] == d).to_numpy()
        mc = np.ones(len(cell), bool) if d == "citywide" else (cell["district"] == d).to_numpy()
        mb = np.ones(len(cb), bool) if d == "citywide" else (cb["district"] == d).to_numpy()
        E_d = sup.loc[mc, "E_cell"].sum()
        row = {"district": d, "n_cbg": int(mb.sum()), "n_cells": int(mc.sum()), "n_reports": int(me.sum()),
               "share_reports": float(me.sum() / len(ev)), "n_visits": int(md.sum()),
               "wR_mean_reports": float(np.nanmean(ev.loc[me, "wR"])),
               "rate_raw": float(me.sum() / E_d / years * RATE_PER), "rate_corr": float(ev.loc[me, "m"].sum() / E_d / years * RATE_PER)}
        row["rel_raw"], row["rel_corr"] = row["rate_raw"] / city_raw, row["rate_corr"] / city_corr
        row["factor"] = row["rate_corr"] / row["rate_raw"]
        row["cf_p10"], row["cf_p50"], row["cf_p90"] = [float(q) for q in np.nanquantile(cell.loc[mc, "cf"], [.1, .5, .9])]
        row["obs_mean_rating"] = float(v.loc[md, "rating"].mean())
        row["survey_map_mean"] = float(np.nanmean(cell.loc[mc, "rating_adj"]))
        for which in ("raw", "corr"):
            row[f"rho_cbg_{which}"] = rho(cb.loc[mb, f"log_rel_{which}_{MAIN}"].to_numpy(), cb.loc[mb, "survey_rating_adj"].to_numpy())
            row[f"rho_cell_{which}"] = rho(cell.loc[mc, f"log_rel_{which}"].to_numpy(), cell.loc[mc, "rating_adj"].to_numpy())
        # (i) the citywide model's gain on this district's visits
        row["dll_citywide_model"] = float(gain_i[md].sum()); row["share_of_citywide_dll"] = row["dll_citywide_model"] / dll_city_grid
        # (ii) the district's own profile
        vd = v[md].reset_index(drop=True)
        Fd = {"num": F["num"][md], "den": F["den"][md], "lam_bar_raw": F["lam_bar_raw"], "mean_m": F["mean_m"]}
        lls, _ = calib.profile(Fd, y[md], FE(vd).build(vd), range(len(BETA_GRID)))
        ll = np.array([lls[i] for i in range(len(BETA_GRID))]); curves[d] = ll - ll[I0]
        bh, llm, edge = calib.argmax_parabola(BETA_GRID, ll)
        row.update({"ll0_own": float(ll[I0]), "beta_own": bh, "at_edge": edge, "dll_own": float(llm - ll[I0]),
                    "dll_own_at_citywide_beta": float(ll[i_hat] - ll[I0]), "loss_at_citywide_beta": float(llm - ll[i_hat]),
                    "n_surveyors": int(vd["surv"].nunique()), "n_months": int(vd["month"].nunique())})
        rows.append(row)
        print(f"  {d:>21}: {row['n_cbg']:>4} bg, {row['n_reports']:>6,} reports, {row['n_visits']:>6,} visits | w̄_R {row['wR_mean_reports']:+.2f} | "
              f"rate ×{row['rel_raw']:.2f} → ×{row['rel_corr']:.2f} (factor {row['factor']:.2f}) | ρ_bg {row['rho_cbg_raw']:.2f} → "
              f"{row['rho_cbg_corr']:.2f} | Δℓ city model {row['dll_citywide_model']:+6.1f} | own β̂ {bh:.2f}{'*' if edge else ' '} "
              f"Δℓ {row['dll_own']:+6.1f}, loss at β̂ {row['loss_at_citywide_beta']:.1f}   ({time.time() - t:.0f} s)")
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(DIST, "districts.csv"), index=False)
    pd.DataFrame({"beta": BETA_GRID, **curves}).to_csv(os.path.join(DIST, "profile_curves.csv"), index=False)
    inside = tab[tab["district"] != "citywide"]
    print(f"  closure: Σ_districts Δℓ(citywide model) = {inside['dll_citywide_model'].sum():+.1f} + unassigned "
          f"{gain_i[v['district'].isna().to_numpy()].sum():+.1f} = {dll_city_grid:+.1f}")

    summary = {"beta_hat": beta_hat, "ci_95": ci, "beta_grid_point": float(BETA_GRID[i_hat]), "h": H0, "window": MAIN,
               "citywide_rate_raw": city_raw, "citywide_rate_corr": city_corr, "dll_citywide_grid": dll_city_grid,
               "assignment": "block group representative point within the planning district",
               "districts": {r["district"]: {k: (float(x) if isinstance(x, (np.floating, float)) else x) for k, x in r.items() if k != "district"}
                             for r in rows}}
    with open(os.path.join(DIST, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ---- figures for the requested districts ---------------------------------------------------------
    scales = {"rate": surf.rate_scale(D["log_rel_raw"], D["log_rel_corr"]),
              "c_mean": float(np.nanmean(S["rating_adj"])), "cf_lim": surf.sym_lim(np.log(D["correction_factor"]), q=0.99),
              "d_lim": surf.sym_lim(D["rate_delta"], q=0.95)}            # the same limits 06 uses citywide
    scales["c_lim"] = surf.sym_lim(S["rating_adj"] - scales["c_mean"])
    z = np.load(os.path.join(OUT, "wR_raster.npz")); res = float(z["res"])
    W = {"Z": z["wR"][::-1], "extent": [z["x"][0] - res / 2, z["x"][-1] + res / 2, z["y"][0] - res / 2, z["y"][-1] + res / 2],
         "lim": surf.sym_lim(z["wR"])}
    for d, label in DISTRICTS.items():
        row = tab[tab["district"] == d].iloc[0]
        poly = dist.loc[dist["district"] == d, "geometry"].iloc[0]
        _figure(d, label, poly, row, D, S, extent, W, scales, geom, ev[ev["district"] == d], v[v["district"] == d],
                beta_hat, HW0, os.path.join(FIG, f"fig4_{d.lower().replace(' ', '_')}.png"))
    print(f"\nDone in {time.time() - t_start:.0f} s. Saved → output/districts/districts.csv, profile_curves.csv, cbg_district.csv, "
          f"summary.json; figures " + ", ".join(f"fig4_{d.lower()}.png" for d in DISTRICTS))


def _figure(d, label, poly, row, D, S, extent, W, sc, geom, e, vz, beta_hat, HW0, path):
    """2 × 3: the three 06 maps, w_R, the correction factor and the correction delta, zoomed to the district."""
    x0, y0, x1, y1 = poly.bounds
    zoom = {"x": (x0 - ZOOM_PAD, x1 + ZOOM_PAD), "y": (y0 - ZOOM_PAD, y1 + ZOOM_PAD)}
    R = SimpleNamespace(extent=extent)
    outline = gpd.GeoSeries([poly], crs=CRS).boundary
    r_lo, r_hi, r_ticks, r_labels = sc["rate"]
    lab_rate = "dumping report rate as a multiple of the citywide rate   (blue = fewer, red = more)"
    C = S["rating_adj"] - sc["c_mean"]
    aspect = (zoom["y"][1] - zoom["y"][0]) / (zoom["x"][1] - zoom["x"][0])
    fig, axes = plt.subplots(2, 3, figsize=(19, 2 * (5.5 * aspect + 1.7) + 0.6))
    panels = [(axes[0, 0], D["log_rel_raw"], R, f"(A) 311 dumping reports as filed, {MAIN} average", lab_rate, r_lo, r_hi, CMAP_DUMPING, r_ticks, r_labels),
              (axes[0, 1], D["log_rel_corr"], R, f"(B) corrected for reporting propensity, β̂ = {beta_hat:.2f}", lab_rate, r_lo, r_hi, CMAP_DUMPING, r_ticks, r_labels),
              (axes[0, 2], C, R, "(C) Litter Index survey on its own, 2022–23 (month + surveyor adjusted)",
               f"adjusted rating − citywide mean ({sc['c_mean']:.2f}), rating points   (blue = cleaner, red = dirtier)", -sc["c_lim"], sc["c_lim"], CMAP_LITTER, None, None),
              (axes[1, 0], W["Z"], SimpleNamespace(extent=W["extent"]), f"(D) reporting propensity w_R from street poles (h_w = {HW0} m)",
               "log outage reports per pole, centred   (pink = chatty, green = quiet)", -W["lim"], W["lim"], CMAP_PROPENSITY, None, None),
              (axes[1, 1], np.log(D["correction_factor"]), R, f"(E) correction factor, {MAIN}: log(corrected / as filed)",
               "pink = reports discounted (chatty); green = scaled up (quiet)", -sc["cf_lim"], sc["cf_lim"], CMAP_FACTOR, None, None),
              (axes[1, 2], D["rate_delta"], R, f"(F) effect of the correction, {MAIN}: corrected − as filed",
               f"reports per year per {RATE_PER:,.0f} stops   (brown = reports removed, teal = reports added)",
               -sc["d_lim"], sc["d_lim"], CMAP_DELTA, None, None)]
    for ax, Z, Rz, title, lab, lo, hi, cmap, tk, tl in panels:
        surf.show(ax, Z, Rz, geom, title, lab, lo, hi, cmap, zoom=zoom, ticks=tk, ticklabels=tl)
        outline.plot(ax=ax, color=INK, linewidth=1.4)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MultipleLocator(1000)); axis.set_major_formatter(FuncFormatter(lambda val, p: f"{val / 1000:.0f}"))
    axes[0, 0].scatter(e["x"], e["y"], s=3, marker="x", c="black", linewidths=0.3, alpha=0.35,
                       label=f"311 dumping reports, {MAIN} ({len(e):,})")
    axes[0, 2].scatter(vz["x"], vz["y"], s=3, marker="o", facecolors="#ffd400", edgecolors="black", linewidths=0.25, alpha=0.8,
                       label=f"survey visits, 2022–23 ({len(vz):,})")
    for ax in (axes[0, 0], axes[0, 2]):
        ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9, edgecolor=MUTED)
    fig.suptitle(f"{label} ({d} planning district) — {row['n_cbg']} block groups, {row['n_reports']:,} dumping reports {MAIN}, "
                 f"{row['n_visits']:,} survey visits; report rate ×{row['rel_raw']:.2f} of citywide as filed, ×{row['rel_corr']:.2f} corrected "
                 f"(mean w_R at reports {row['wR_mean_reports']:+.2f})\n"
                 f"block groups, Spearman with the survey: as filed {row['rho_cbg_raw']:.2f} → corrected {row['rho_cbg_corr']:.2f};  "
                 f"citywide model's gain on these visits Δℓ = {row['dll_citywide_model']:+.0f};  "
                 f"β̂ from these visits alone {row['beta_own']:.2f}{' (grid edge)' if row['at_edge'] else ''} (Δℓ {row['dll_own']:+.0f})",
                 x=0.01, ha="left", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
