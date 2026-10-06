"""
04_field.py
===========
Stage 2: the 311 dumping intensity field, evaluated at every survey visit.

    lambda_hat(s, t) = sum_j m_j K_h(|s - s_j|) exp(-(t - t_j)/tau) 1[t >= t_j]
                       / D_h(s)

Spatial kernel: compact quartic, radius h (sd = h/sqrt(7)). Default h = 500 m
(the triage optimum; sd ~ 190 m), swept over {150, 300, 500, 800, 1200}.
Temporal kernel: causal exponential, tau = 120 d, swept over {60, 120, 240,
480}; the placebo showed tau is a pooling window, so the sweep should be flat.

Denominators D_h(s), all at the SAME h (edge effects cancel to first order):
    expo     kernel-smoothed SafeGraph exposure on the built support
             -> the ratio (relative-risk) form, primary; keeps beta on 05a's
                conditional-on-exposure footing
    support  kernel mass on the built support, as a fraction of pi h^2 / 3
             -> edge-corrected absolute density, robustness
    (none)   the raw kernel sum
The built support is a GRID_RES m grid of cells inside a CBG and within
SUPPORT_DIST m of a street pole: rivers, parks, the airport and rail yards drop
out. Each CBG's exposure is spread evenly over its support cells so totals are
preserved.

Corrected numerators use m_j = exp(-beta0 * w_R(s_j)) with beta0 the Stage 1a
production posterior mean and w_R either the kernel surface at the report
(primary, WR_PRIMARY) or the CBG posterior mean of the report's CBG.
w_bar(s, t) = event-kernel-weighted mean of w_R, the regressor S1' needs.

Also at the default (h, tau): the field built on deduplicated "piles"
(50 m / 14 d) and the negative-control field built from Street Light Outage
reports.

Unit tests
  - sparse accumulator vs dense recomputation on 300 random visits
  - kernel normalisation: flat weights on an unmasked grid integrate to the
    event count (pi h^2 / 3 per event) within grid discretisation error

Outputs
  output/support_grid.parquet   x, y, GEOID, E_cell
  output/visit_fields.parquet   one row per litter_panel visit; per (h, tau):
      num, num_k, num_c, wbar_k, wbar_c; per h: den_expo, den_support, n_within;
      num_dedup, num_outage at the default
  output/field_sweep.csv        Spearman(rating, log field) per (h, tau, form)
  output/figures/04_*.png
"""

import os
import json
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

from kernels import kernel_sum, accumulate_st, build_W, shrunk_log, quartic, SD_FACTOR

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

CRS = 26918
H_SWEEP = [150, 300, 500, 800, 1200, 2000]
TAU_SWEEP = [60, 120, 240, 480]
H0, TAU0 = 500, 120
GRID_RES = 50           # m
SUPPORT_DIST = 75       # m from a pole
ALPHA = 1.0             # Gamma-shrinkage prior events for the reported logs
WR_PRIMARY = "wR_pre24_naive_h600"
DEDUP_M, DEDUP_D = 50.0, 14.0

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


# =============================================================================
# data
# =============================================================================

def load_events(beta0: float) -> dict:
    d = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    w = pd.read_parquet(os.path.join(OUT, "wR_events.parquet"))
    assert len(d) == len(w) and (d["GEOID"].fillna("").to_numpy() == w["GEOID"].fillna("").to_numpy()).all(), \
        "wR_events.parquet is not row-aligned with points_dumping.parquet"
    rep = pd.read_parquet(os.path.join(OUT, "reporting_cbg.parquet")).set_index("GEOID")["w_R_mean"]
    wk = w[WR_PRIMARY].to_numpy(float)
    wc = d["GEOID"].map(rep).to_numpy(float)
    ok_k, ok_c = np.isfinite(wk), np.isfinite(wc)
    wk, wc = np.where(ok_k, wk, 0.0), np.where(ok_c, wc, 0.0)
    xy = d[["x", "y"]].to_numpy(float)
    ev = {"xy": xy, "t": d["t_days"].to_numpy(float), "ts": d["ts"], "tree": cKDTree(xy),
          "wk": wk, "wc": wc, "ok_k": ok_k.astype(float), "ok_c": ok_c.astype(float),
          "m_k": np.exp(-beta0 * wk), "m_c": np.exp(-beta0 * wc)}
    print(f"  {len(d):,} dumping reports; kernel w_R for {ok_k.mean():.2%}, CBG w_R for {ok_c.mean():.2%}; "
          f"beta0 = {beta0:.4f}; sd(w_R kernel) {wk[ok_k].std():.3f}, sd(w_R cbg) {wc[ok_c].std():.3f}")
    return ev


def pile_flag(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """1 for the first report of a space-time pile (DEDUP_M m / DEDUP_D d), 0 for repeats."""
    order = np.argsort(t, kind="stable")
    ts, pxy = t[order], xy[order]
    tree = cKDTree(pxy)
    first = np.ones(len(t), bool)
    for k, lst in enumerate(tree.query_ball_point(pxy, DEDUP_M)):
        if len(lst) > 1:
            arr = np.asarray(lst)
            prior = arr[arr < k]
            if len(prior) and (ts[k] - ts[prior].max()) <= DEDUP_D:
                first[k] = False
    out = np.empty(len(t), bool); out[order] = first
    return out


def build_support_grid() -> pd.DataFrame:
    print("\n[grid] Built support and exposure cells ...")
    geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson"))
    geom["GEOID"] = geom["GEOID"].astype(str)
    static = pd.read_parquet(os.path.join(OUT, "cbg_static.parquet")).set_index("GEOID")["exposure"]
    poles = pd.read_parquet(os.path.join(OUT, "poles_xy.parquet"))[["x", "y"]].to_numpy()
    xmin, ymin, xmax, ymax = geom.total_bounds
    gx = np.arange(xmin + GRID_RES / 2, xmax, GRID_RES)
    gy = np.arange(ymin + GRID_RES / 2, ymax, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    cells = pd.DataFrame({"x": GX.ravel(), "y": GY.ravel()})
    pts = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy(cells["x"], cells["y"]), crs=CRS)
    hit = pts.sjoin(geom[["GEOID", "geometry"]], how="inner", predicate="within")
    hit = hit[~hit.index.duplicated(keep="first")]
    n_in = len(hit)
    dist, _ = cKDTree(poles).query(hit[["x", "y"]].to_numpy())
    sup = pd.DataFrame(hit.drop(columns="geometry"))[dist <= SUPPORT_DIST].copy()
    n_cells = sup.groupby("GEOID").size()
    sup["E_cell"] = sup["GEOID"].map(static) / sup["GEOID"].map(n_cells)
    lost = static.reindex(geom["GEOID"]).drop(n_cells.index, errors="ignore")
    print(f"  {len(cells):,} cells in the bbox, {n_in:,} inside a CBG, {len(sup):,} on the built support "
          f"({GRID_RES} m cells within {SUPPORT_DIST} m of a pole; {len(sup) * GRID_RES ** 2 / 1e6:.0f} km²).")
    print(f"  exposure preserved on {n_cells.size} CBGs; {len(lost)} CBGs have no support cell "
          f"(exposure lost {lost.sum() / static.sum():.3%}).")
    sup = sup[["x", "y", "GEOID", "E_cell"]].reset_index(drop=True)
    sup.to_parquet(os.path.join(OUT, "support_grid.parquet"), index=False)
    return sup


# =============================================================================
# unit tests
# =============================================================================

def brute_check(v, ev, num_default, n_check=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(v), n_check, replace=False))
    vxy, vt = v[["x", "y"]].to_numpy()[idx], v["t_days"].to_numpy()[idx]
    lam = np.zeros(n_check)
    for s in range(0, n_check, 100):
        D = np.sqrt(((vxy[s:s + 100, None, :] - ev["xy"][None, :, :]) ** 2).sum(-1))
        dt = vt[s:s + 100, None] - ev["t"][None, :]
        lam[s:s + 100] = (quartic(D, H0) * np.where(dt >= 0, np.exp(-dt / TAU0), 0.0)).sum(1)
    err = np.abs(lam - num_default[idx]).max() / max(np.abs(lam).max(), 1e-12)
    # and the sparse-matrix route
    W = build_W(vxy, vt, ev["xy"], ev["t"], H0, TAU0, src_tree=ev["tree"])
    errW = np.abs(np.asarray(W.sum(1)).ravel() - lam).max() / max(np.abs(lam).max(), 1e-12)
    print(f"  [unit test] accumulator vs dense on {n_check} visits: rel err {err:.1e}; "
          f"build_W row sums vs dense: rel err {errW:.1e} — {'OK' if max(err, errW) < 1e-10 else 'MISMATCH'}")
    return bool(max(err, errW) < 1e-10)


def integration_check(ev, h=H0):
    """Flat weights on an unmasked grid must integrate to N * pi h^2 / 3."""
    xy = ev["xy"]
    pad = h + GRID_RES
    gx = np.arange(xy[:, 0].min() - pad, xy[:, 0].max() + pad, GRID_RES)
    gy = np.arange(xy[:, 1].min() - pad, xy[:, 1].max() + pad, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.c_[GX.ravel(), GY.ravel()]
    S = kernel_sum(grid, xy, h, source_tree=ev["tree"])
    total = S.sum() * GRID_RES ** 2 / (np.pi * h * h / 3.0)
    ratio = total / len(xy)
    print(f"  [unit test] kernel mass on an unmasked {GRID_RES} m grid ({len(grid):,} cells): "
          f"{ratio:.5f} of the event count (expect 1 ± ~{(GRID_RES / h) ** 2:.3f}) — "
          f"{'OK' if abs(ratio - 1) < 3 * (GRID_RES / h) ** 2 + 1e-3 else 'MISMATCH'}")
    return bool(abs(ratio - 1) < 3 * (GRID_RES / h) ** 2 + 1e-3)


# =============================================================================
# main
# =============================================================================

def main():
    t_start = time.time()
    print("=" * 68); print("04_field.py — Stage 2: the 311 intensity field"); print("=" * 68)
    with open(os.path.join(OUT, "reporting_summary.json")) as f:
        beta0 = float(json.load(f)["stage1a"]["production"]["beta"])
    v = pd.read_parquet(os.path.join(OUT, "litter_panel.parquet"))
    print(f"  {len(v):,} survey visits from litter_panel.parquet")
    ev = load_events(beta0)
    sup = build_support_grid()
    sup_xy, sup_E = sup[["x", "y"]].to_numpy(), sup["E_cell"].to_numpy()
    sup_tree = cKDTree(sup_xy)
    vxy, vt = v[["x", "y"]].to_numpy(), v["t_days"].to_numpy()

    weights = {k: ev[k] for k in ["m_k", "m_c", "wk", "wc", "ok_k", "ok_c"]}
    weights["one"] = np.ones(len(ev["t"]))

    print("\n[field] sweep over h and tau ...")
    F = pd.DataFrame(index=v.index)
    summary = {"beta0": beta0, "h_sweep": H_SWEEP, "tau_sweep": TAU_SWEEP, "h0": H0, "tau0": TAU0}
    for h in H_SWEEP:
        t0 = time.time()
        o = accumulate_st(vxy, vt, ev["xy"], ev["t"], h, TAU_SWEEP, weights, src_tree=ev["tree"])
        F[f"n_within_h{h}"] = o["n_within"]
        for tau in TAU_SWEEP:
            F[f"num_h{h}_t{tau}"] = o[(tau, "one")]
            F[f"num_k_h{h}_t{tau}"] = o[(tau, "m_k")]
            F[f"num_c_h{h}_t{tau}"] = o[(tau, "m_c")]
            with np.errstate(invalid="ignore", divide="ignore"):
                F[f"wbar_k_h{h}_t{tau}"] = np.where(o[(tau, "ok_k")] > 0, o[(tau, "wk")] / o[(tau, "ok_k")], np.nan)
                F[f"wbar_c_h{h}_t{tau}"] = np.where(o[(tau, "ok_c")] > 0, o[(tau, "wc")] / o[(tau, "ok_c")], np.nan)
        F[f"den_expo_h{h}"] = kernel_sum(vxy, sup_xy, h, weights=sup_E, source_tree=sup_tree)
        F[f"den_support_h{h}"] = kernel_sum(vxy, sup_xy, h, source_tree=sup_tree) * GRID_RES ** 2 / (np.pi * h * h / 3.0)
        z = (F[f"num_h{h}_t{TAU0}"] == 0).mean()
        print(f"  h={h:>4} (sd {h * SD_FACTOR:>3.0f} m): {time.time() - t0:5.1f} s   "
              f"events within h: mean {o['n_within'].mean():7.1f}   zero numerators (tau={TAU0}): {z:.2%}   "
              f"support fraction: median {np.median(F[f'den_support_h{h}']):.3f}, p5 {np.quantile(F[f'den_support_h{h}'], .05):.3f}")

    # default-bandwidth extras: dedup piles and the outage negative control
    t0 = time.time()
    first = pile_flag(ev["xy"], ev["t"])
    o = accumulate_st(vxy, vt, ev["xy"], ev["t"], H0, [TAU0], {"pile": first.astype(float)}, src_tree=ev["tree"])
    F[f"num_dedup_h{H0}_t{TAU0}"] = o[(TAU0, "pile")]
    r = pd.read_parquet(os.path.join(OUT, "points_outage.parquet"))
    rxy, rt = r[["x", "y"]].to_numpy(float), r["t_days"].to_numpy(float)
    o = accumulate_st(vxy, vt, rxy, rt, H0, [TAU0], None)
    F[f"num_outage_h{H0}_t{TAU0}"] = o[(TAU0, "one")]
    print(f"  dedup: {first.sum():,} piles from {len(first):,} reports ({1 - first.mean():.1%} repeats); "
          f"outage control from {len(r):,} reports   ({time.time() - t0:.1f} s)")
    summary["dedup"] = {"piles": int(first.sum()), "repeat_share": float(1 - first.mean())}

    # unit tests
    print("\n[tests]")
    ok1 = brute_check(v, ev, F[f"num_h{H0}_t{TAU0}"].to_numpy())
    ok2 = integration_check(ev)
    summary["unit_tests"] = {"sparse_vs_dense": ok1, "kernel_normalisation": ok2}

    # the sweep table: Spearman of rating with each form's shrunk log field
    print("\n[sweep] Spearman(rating, log field), Gamma-shrunk with alpha = 1")
    rows = []
    y = v["rating"].to_numpy()
    for h in H_SWEEP:
        for tau in TAU_SWEEP:
            num = F[f"num_h{h}_t{tau}"].to_numpy()
            for form, den in [("abs", np.ones(len(num))), ("support", F[f"den_support_h{h}"].to_numpy()),
                              ("expo", F[f"den_expo_h{h}"].to_numpy())]:
                ll, _ = shrunk_log(num, den, ALPHA)
                numk = F[f"num_k_h{h}_t{tau}"].to_numpy()
                llk, _ = shrunk_log(numk, den, ALPHA)
                rows.append({"h": h, "tau": tau, "form": form,
                             "spearman_raw": spearmanr(y, ll).correlation,
                             "spearman_corr_k": spearmanr(y, llk).correlation,
                             "zero_share": float((num == 0).mean())})
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(OUT, "field_sweep.csv"), index=False)
    piv = sw.pivot_table(index=["form", "tau"], columns="h", values="spearman_raw")
    print(piv.round(4).to_string())
    best = sw.loc[sw["spearman_raw"].idxmax()]
    print(f"  best raw: form={best['form']} h={int(best['h'])} tau={int(best['tau'])} rho={best['spearman_raw']:.4f}")
    summary["sweep_best"] = {"form": best["form"], "h": int(best["h"]), "tau": int(best["tau"]),
                             "spearman": float(best["spearman_raw"])}
    # rho(h): how much reporting contrast the smoother destroys (w_bar on w_R at the point)
    rhos = {}
    for h in H_SWEEP:
        wb = F[f"wbar_k_h{h}_t{TAU0}"].to_numpy(); wp = v[WR_PRIMARY].to_numpy()
        ok = np.isfinite(wb) & np.isfinite(wp)
        rhos[h] = float(np.polyfit(wp[ok], wb[ok], 1)[0])
    print("  rho(h) = slope of w_bar on w_R(s) [kernel]: " + "  ".join(f"h={h}: {r:.3f}" for h, r in rhos.items()))
    summary["rho_h"] = rhos

    diagnostics(v, F, ev, sup, sup_tree, y, summary)

    F.to_parquet(os.path.join(OUT, "visit_fields.parquet"), index=False)
    with open(os.path.join(OUT, "field_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    _figure(sw)
    print(f"\n  Saved → output/visit_fields.parquet ({F.shape[1]} columns), field_sweep.csv, field_summary.json")
    print(f"Done in {time.time() - t_start:.0f} s.")


def diagnostics(v, F, ev, sup, sup_tree, y, summary):
    """Where does the ratio form's gain come from? Decoupled bandwidths,
    alternative denominators, exposure alone, and CBG-level benchmarks."""
    vxy, vt = v[["x", "y"]].to_numpy(), v["t_days"].to_numpy()
    sup_xy, sup_E = sup[["x", "y"]].to_numpy(), sup["E_cell"].to_numpy()

    print("\n[diag 1] decoupled bandwidths: Spearman(rating, log num_hn / den_hd), tau = TAU0; "
          "rows = numerator h, cols = exposure-denominator h (0 = own CBG density, inf = none)")
    den = {h: F[f"den_expo_h{h}"].to_numpy() for h in H_SWEEP}
    den[0] = (v["exposure"] / v["area_km2"]).to_numpy()
    cols = [0] + H_SWEEP + ["inf"]
    rows = []
    for hn in H_SWEEP:
        num = F[f"num_h{hn}_t{TAU0}"].to_numpy()
        r = {"h_num": hn}
        for hd in cols:
            d = np.ones(len(y)) if hd == "inf" else den[hd]
            r[str(hd)] = spearmanr(y, shrunk_log(num, d, ALPHA)[0]).correlation
        rows.append(r)
    dec = pd.DataFrame(rows).set_index("h_num")
    print(dec.round(4).to_string())
    dec.to_csv(os.path.join(OUT, "field_decoupled.csv"))
    summary["decoupled_best"] = {"h_num": int(dec.max(axis=1).idxmax()),
                                 "h_den": dec.loc[dec.max(axis=1).idxmax()].idxmax(),
                                 "spearman": float(dec.to_numpy().max())}

    print("\n[diag 2] alternative denominators (same-h numerator), tau = TAU0")
    static = pd.read_parquet(os.path.join(OUT, "cbg_static.parquet")).set_index("GEOID")
    geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson")); geom["GEOID"] = geom["GEOID"].astype(str)
    area = geom.set_index("GEOID").geometry.area / 1e6
    ncell = sup.groupby("GEOID").size()
    pop_cbg = static["pop_density"].reindex(ncell.index) * area.reindex(ncell.index)
    sup_pop = (sup["GEOID"].map(pop_cbg) / sup["GEOID"].map(ncell)).fillna(0).to_numpy()
    sup_len = (sup["GEOID"].map(static["total_length_w"]) / sup["GEOID"].map(ncell)).fillna(0).to_numpy()
    poles = pd.read_parquet(os.path.join(OUT, "poles_xy.parquet"))[["x", "y"]].to_numpy()
    ptree = cKDTree(poles)
    alt_rows = []
    for h in [500, 1200]:
        num = F[f"num_h{h}_t{TAU0}"].to_numpy()
        alts = {"none": np.ones(len(y)), "expo (SafeGraph)": den[h],
                "poles": kernel_sum(vxy, poles, h, source_tree=ptree),
                "population": kernel_sum(vxy, sup_xy, h, weights=sup_pop, source_tree=sup_tree),
                "street length": kernel_sum(vxy, sup_xy, h, weights=sup_len, source_tree=sup_tree)}
        for name, d in alts.items():
            alt_rows.append({"h": h, "denominator": name,
                             "spearman": spearmanr(y, shrunk_log(num, d, ALPHA)[0]).correlation,
                             "spearman_den_alone": spearmanr(y, np.log(np.maximum(d, 1e-9))).correlation
                             if name != "none" else np.nan})
    alt = pd.DataFrame(alt_rows)
    print(alt.pivot_table(index="denominator", columns="h", values=["spearman", "spearman_den_alone"]).round(4).to_string())
    alt.to_csv(os.path.join(OUT, "field_alt_denominators.csv"), index=False)

    print("\n[diag 3] CBG-level benchmarks (same causal tau at the visit date, own-CBG exposure):")
    # per-CBG causal sums at the visit date via the cumsum trick
    ev_df = pd.DataFrame({"g": pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))["GEOID"],
                          "t": ev["t"]}).dropna()
    per_cbg = {g: (np.sort(s["t"].to_numpy()), None) for g, s in ev_df.groupby("g")}
    per_cbg = {g: (t, np.cumsum(np.exp(t / TAU0))) for g, (t, _) in per_cbg.items()}

    def cbg_num(gid, tv):
        if gid not in per_cbg:
            return np.zeros(len(tv))
        t, cs = per_cbg[gid]
        k = np.searchsorted(t, tv, side="right")
        return np.where(k > 0, np.exp(-tv / TAU0) * cs[np.maximum(k - 1, 0)], 0.0)

    touch = geom.sjoin(geom, how="inner", predicate="touches")[["GEOID_left", "GEOID_right"]]
    nb1 = touch.groupby("GEOID_left")["GEOID_right"].apply(set).to_dict()
    nb2 = {g: set().union(*[nb1.get(k, set()) for k in nb1.get(g, set()) | {g}]) | nb1.get(g, set())
           for g in geom["GEOID"]}
    expo = static["exposure"]
    out = {"own CBG": np.zeros(len(v)), "own + 1st-order neighbours": np.zeros(len(v)),
           "own + 2nd-order neighbours": np.zeros(len(v))}
    den_b = {k: np.zeros(len(v)) for k in out}
    for gid, idx in v.groupby("GEOID").indices.items():
        tv = vt[idx]
        for label, members in [("own CBG", {gid}), ("own + 1st-order neighbours", nb1.get(gid, set()) | {gid}),
                               ("own + 2nd-order neighbours", nb2.get(gid, set()) | {gid})]:
            num = np.zeros(len(idx)); E = 0.0
            for m in members:
                num += cbg_num(m, tv); E += float(expo.get(m, 0.0))
            out[label][idx] = num; den_b[label][idx] = E
    bench = {}
    for label in out:
        bench[label] = {"ratio": spearmanr(y, shrunk_log(out[label], den_b[label], ALPHA)[0]).correlation,
                        "count": spearmanr(y, shrunk_log(out[label], np.ones(len(y)), ALPHA)[0]).correlation}
        print(f"   {label:>28}: ratio (÷ exposure) {bench[label]['ratio']:+.4f}   raw count {bench[label]['count']:+.4f}")
    best_pt = summary["sweep_best"]
    print(f"   best point kernel (ratio form): h={best_pt['h']} tau={best_pt['tau']} {best_pt['spearman']:+.4f}")
    summary["cbg_benchmarks"] = bench


def _figure(sw):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for form, col in [("abs", C1), ("support", C3), ("expo", C2)]:
        s = sw[(sw["form"] == form) & (sw["tau"] == TAU0)]
        ax.plot(s["h"], s["spearman_raw"], "-o", color=col, lw=2, ms=5, label=form)
    ax.set_xscale("log"); ax.set_xticks(H_SWEEP); ax.set_xticklabels([str(h) for h in H_SWEEP])
    _style(ax, f"Spearman(rating, log λ̂) by h, τ = {TAU0} d", "quartic radius h (m)", "Spearman ρ")
    ax.legend(frameon=False, fontsize=8, title="denominator", title_fontsize=8)
    ax = axes[1]
    for form, col in [("abs", C1), ("support", C3), ("expo", C2)]:
        s = sw[(sw["form"] == form) & (sw["h"] == H0)]
        ax.plot(s["tau"], s["spearman_raw"], "-o", color=col, lw=2, ms=5, label=form)
    ax.set_xscale("log"); ax.set_xticks(TAU_SWEEP); ax.set_xticklabels([str(t) for t in TAU_SWEEP])
    _style(ax, f"Spearman(rating, log λ̂) by τ, h = {H0} m", "τ (days)", "Spearman ρ")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "04_field_sweep.png"), dpi=160); plt.close(fig)


def _style(ax, title, xlabel, ylabel):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9); ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


if __name__ == "__main__":
    main()
