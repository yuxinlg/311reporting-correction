"""
00_triage.py
============
Day-1 gate. Six diagnostics that decide what the paper is, run BEFORE the
modelling pipeline (02-06) is built. Everything here is a preview: OLS on the
raw 1-4 rating, CBG-level w_R read from the predecessor's reference output,
crude exposure by point-in-polygon. The pipeline does each of these properly.

  T1  Forward/backward placebo with month-of-sample and location effects
      absorbed. Does the PAST 90-day 311 count predict the rating better than
      the FUTURE 90-day count? If not, the time kernel is a pooling window,
      not a dynamics model.
  T2  Spline of rating vs log lambda-hat: is there a top-decile turnover, and
      does the 05a correction (beta0 = 0.579) straighten it?
  T3  Variance budget of log lambda-hat on month / surveyor / location FE, the
      surveyor x block crosstab, and surveyor-location connectivity (if that
      bipartite graph is disconnected, surveyor and location FE are not
      separately identified).
  T4  Geocoding-atom / dedup audit across all years -> minimum defensible h.
  T5  Endogenous-routing test: do 311 reports spike just BEFORE a survey visit
      at that location (crews dispatched in response to complaints)?
  T6  Specification curve for beta-hat = -c/b across FE sets, bandwidths,
      w-bar vs w_R(s), with/without an exposure denominator.

Inputs   output/points_dumping.parquet, output/cbg_geom.geojson,
         output/cbg_static.parquet, data/litter_surveys_raw_20250730.csv,
         data/ref_05a_output/reporting_decomp_cbg_sl.geojson (w_R, triage only)
Outputs  output/triage/*.csv, output/triage/visit_fields.parquet,
         output/triage/figures/*.png
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
import statsmodels.api as sm
from patsy import dmatrix, build_design_matrices
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
TRI = os.path.join(OUT, "triage")
FIG = os.path.join(TRI, "figures")
os.makedirs(FIG, exist_ok=True)

CRS = 26918
EPOCH = pd.Timestamp("2021-01-01")

# --- kernel settings ---------------------------------------------------------
H_SWEEP = [100, 150, 200, 300, 500, 800, 1200, 2000]   # m, quartic radius
# quartic K(u) = (1-u^2)^2 on |u|<1 has sd = h/sqrt(7) ~ 0.378 h, so a quartic
# radius of 500 m smooths like a Gaussian with sd ~ 190 m.
TAUS = [15, 30, 60, 120, 240, 480]            # d, causal exponential
H0, TAU0 = 200, 120                            # the measured optimum
BETA0 = 0.579                                  # 05a posterior mean
PLACEBO_WIN = 90                               # d, hard past/future windows
ROUTING_WIN = 30                               # d, pre/post window for T5
ES_WEEKS = 12                                  # event-study half-width
N_BLOCKS = 40                                  # spatial blocks for T3

BULK_TOKENS = {"trash_bags", "construction", "oversized", "furniture",
               "tires", "vehicular"}

# palette (dataviz reference instance, categorical slots 1-3 + neutrals)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


# =============================================================================
# helpers
# =============================================================================

def codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s)[0].astype(np.int64)


def demean(Y, groups, tol=1e-10, max_iter=500):
    """Iterative within-transformation for one or several FE dimensions."""
    Y = np.asarray(Y, float)
    one_d = Y.ndim == 1
    Y = Y.reshape(len(Y), -1).copy()
    sizes = [np.bincount(g).astype(float) for g in groups]
    for _ in range(max_iter):
        prev = Y.copy()
        for g, sz in zip(groups, sizes):
            for c in range(Y.shape[1]):
                m = np.bincount(g, weights=Y[:, c]) / sz
                Y[:, c] -= m[g]
        if np.abs(Y - prev).max() < tol:
            break
    return Y[:, 0] if one_d else Y


def ols_cluster(y, X, cluster, names):
    X = pd.DataFrame(np.asarray(X, float), columns=names)
    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(np.asarray(y, float), X).fit(
        cov_type="cluster", cov_kwds={"groups": np.asarray(cluster)})


def r2_of_fe(y, groups) -> float:
    resid = demean(y, groups)
    return 1.0 - resid.var() / y.var()


def safe_log(lam: np.ndarray):
    """log(lam + eps), eps = 5% of the median positive value. Returns (log, eps)."""
    pos = lam[lam > 0]
    eps = 0.05 * np.median(pos) if len(pos) else 1.0
    return np.log(lam + eps), eps


def style(ax, title=None, xlabel=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


# =============================================================================
# A. data
# =============================================================================

def load_visits() -> pd.DataFrame:
    print("\n[A] Survey visits ...")
    df = pd.read_csv(os.path.join(DATA, "litter_surveys_raw_20250730.csv"),
                     low_memory=False)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    n0 = len(df)
    df = df[df["department"].eq("streets")]
    df["ts"] = pd.to_datetime(df["survey_start"], format="%m/%d/%Y %H:%M",
                              errors="coerce")
    bad = df["ts"].isna()
    if bad.any():
        df.loc[bad, "ts"] = pd.to_datetime(df.loc[bad, "survey_start"],
                                           errors="coerce")
    df = df[df["ts"].notna() & df["ts"].dt.year.isin([2022, 2023])]
    df = df[df["Litter Rating"].between(1, 4)]
    n_noloc = int(df["location_id"].isna().sum())
    df = df[df["location_id"].notna()]
    print(f"  {n0:,} raw rows -> {len(df):,} streets-dept visits in 2022-23 "
          f"with a 1-4 rating ({n_noloc} rows without a location_id dropped).")

    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]),
                         crs=4326).to_crs(CRS)
    cbg = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson"))
    cbg["area_km2"] = cbg.geometry.area / 1e6
    j = g.sjoin(cbg[["GEOID", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    n_out = int(j["GEOID"].isna().sum())
    j = j[j["GEOID"].notna()].copy()
    j["px"], j["py"] = j.geometry.x, j.geometry.y
    j = pd.DataFrame(j.drop(columns="geometry"))
    print(f"  {n_out} visits outside every CBG dropped; {len(j):,} remain.")

    static = pd.read_parquet(os.path.join(OUT, "cbg_static.parquet"))
    ref = gpd.read_file(os.path.join(
        DATA, "ref_05a_output", "reporting_decomp_cbg_sl.geojson"))
    ref["GEOID"] = ref["GEOID"].astype(str)
    j = j.merge(static[["GEOID", "exposure"]], on="GEOID", how="left")
    j = j.merge(cbg[["GEOID", "area_km2"]], on="GEOID", how="left")
    j = j.merge(ref[["GEOID", "w_R_mean_sl"]], on="GEOID", how="left")

    v = pd.DataFrame({
        "x": j["px"].to_numpy(), "y": j["py"].to_numpy(),
        "ts": j["ts"].to_numpy(),
        "rating": j["Litter Rating"].astype(int).to_numpy(),
        "location_id": j["location_id"].to_numpy(),
        "surveyor": j["Creator"].to_numpy(),
        "GEOID": j["GEOID"].to_numpy(),
        "exposure": j["exposure"].to_numpy(),
        "area_km2": j["area_km2"].to_numpy(),
        "w_cbg": j["w_R_mean_sl"].to_numpy(),
        "types": j["Litter Types"].fillna("").to_numpy(),
    })
    v["t_days"] = ((v["ts"] - EPOCH) / pd.Timedelta(days=1)).astype(float)
    v["ym"] = v["ts"].dt.year * 12 + v["ts"].dt.month
    v["bulk"] = v["types"].apply(
        lambda s: int(bool(set(s.split(",")) & BULK_TOKENS)))
    v["loc"] = codes(v["location_id"])
    v["surv"] = codes(v["surveyor"])
    v["month"] = codes(v["ym"])
    v["cbg"] = codes(v["GEOID"])
    v["log_expo_dens"] = np.log(v["exposure"] / v["area_km2"])
    print(f"  {v['loc'].max() + 1:,} locations, {v['surv'].max() + 1} surveyors, "
          f"{v['month'].max() + 1} year-months, {v['cbg'].max() + 1:,} CBGs; "
          f"bulk flag on {v['bulk'].mean():.2%} of visits; "
          f"w_R missing for {int(v['w_cbg'].isna().sum())} visits.")
    return v


def load_events(v: pd.DataFrame) -> dict:
    print("\n[A] Dumping events ...")
    d = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    ref = gpd.read_file(os.path.join(
        DATA, "ref_05a_output", "reporting_decomp_cbg_sl.geojson"))
    ref["GEOID"] = ref["GEOID"].astype(str)
    wmap = ref.set_index("GEOID")["w_R_mean_sl"]
    wr = d["GEOID"].map(wmap).to_numpy(float)
    ok = np.isfinite(wr)
    print(f"  {len(d):,} events; w_R available for {ok.mean():.2%}.")
    return {
        "xy": d[["x", "y"]].to_numpy(float),
        "t": d["t_days"].to_numpy(float),
        "ts": d["ts"],
        "geoid": d["GEOID"].to_numpy(),
        "wr": np.where(ok, wr, 0.0), "wr_ok": ok.astype(float),
        "mcorr": np.where(ok, np.exp(-BETA0 * np.where(ok, wr, 0.0)), 1.0),
        "tree": cKDTree(d[["x", "y"]].to_numpy(float)),
    }


# =============================================================================
# B. one kernel pass per bandwidth
# =============================================================================

def kernel_pass(h: float, v: pd.DataFrame, ev: dict, want_es: bool,
                chunk: int = None) -> dict:
    n = len(v)
    if chunk is None:                       # keep ~5M pairs per chunk
        chunk = 4000 if h <= 800 else max(200, int(4000 * (800.0 / h) ** 2))
    vx, vy, vt = v["x"].to_numpy(), v["y"].to_numpy(), v["t_days"].to_numpy()
    out = {f"lam_t{tau}": np.zeros(n) for tau in TAUS}
    for k in ["pastK", "futK", "pastN", "futN", "past30N", "fut30N",
              "wbar_num", "wbar_den", "lam_corr", "n_within"]:
        out[k] = np.zeros(n)
    es = np.zeros(2 * ES_WEEKS)
    n_pairs = 0
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        qt = cKDTree(np.c_[vx[s:e], vy[s:e]])
        rec = qt.sparse_distance_matrix(ev["tree"], h, output_type="ndarray")
        i = rec["i"].astype(np.int64) + s
        j = rec["j"].astype(np.int64)
        d = rec["v"]
        n_pairs += len(i)
        K = (1.0 - (d / h) ** 2) ** 2
        dt = vt[i] - ev["t"][j]                 # > 0: event BEFORE the visit
        causal = dt >= 0
        bc = lambda w: np.bincount(i, weights=w, minlength=n)

        for tau in TAUS:
            out[f"lam_t{tau}"] += bc(K * np.where(causal, np.exp(-dt / tau), 0.0))
        w0 = K * np.where(causal, np.exp(-dt / TAU0), 0.0)
        out["wbar_num"] += bc(w0 * ev["wr_ok"][j] * ev["wr"][j])
        out["wbar_den"] += bc(w0 * ev["wr_ok"][j])
        out["lam_corr"] += bc(w0 * ev["mcorr"][j])

        past = (dt >= 0) & (dt < PLACEBO_WIN)
        fut = (dt < 0) & (dt >= -PLACEBO_WIN)
        out["pastK"] += bc(K * past)
        out["futK"] += bc(K * fut)
        out["pastN"] += bc(past.astype(float))
        out["futN"] += bc(fut.astype(float))
        out["past30N"] += bc(((dt >= 0) & (dt < ROUTING_WIN)).astype(float))
        out["fut30N"] += bc(((dt < 0) & (dt >= -ROUTING_WIN)).astype(float))
        out["n_within"] += bc(np.ones(len(i)))

        if want_es:
            wk = np.floor(dt / 7.0)
            m = (wk >= -ES_WEEKS) & (wk < ES_WEEKS)
            es += np.bincount((wk[m] + ES_WEEKS).astype(int),
                              minlength=2 * ES_WEEKS)
    out["es"] = es
    out["n_pairs"] = n_pairs
    return out


def brute_check(v, ev, F, n_check=300, seed=0):
    """Dense recomputation of lambda-hat(h0, tau0) on random visits."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(v), n_check, replace=False))
    vxy = v[["x", "y"]].to_numpy()[idx]
    vt = v["t_days"].to_numpy()[idx]
    lam = np.zeros(n_check)
    for s in range(0, n_check, 100):
        D = np.sqrt(((vxy[s:s + 100, None, :] - ev["xy"][None, :, :]) ** 2).sum(-1))
        K = np.where(D <= H0, (1.0 - (D / H0) ** 2) ** 2, 0.0)
        dt = vt[s:s + 100, None] - ev["t"][None, :]
        lam[s:s + 100] = (K * np.where(dt >= 0, np.exp(-dt / TAU0), 0.0)).sum(1)
    sparse_lam = F[f"lam_h{H0}_t{TAU0}"].to_numpy()[idx]
    err = np.abs(lam - sparse_lam).max()
    rel = err / max(np.abs(lam).max(), 1e-12)
    print(f"  [unit test] sparse vs dense lambda-hat on {n_check} visits: "
          f"max abs err {err:.2e} (rel {rel:.1e}) — "
          f"{'OK' if rel < 1e-10 else 'MISMATCH'}")
    return rel < 1e-10


# =============================================================================
# C. the six tests
# =============================================================================

def t1_placebo(v, F, summary):
    print("\n[T1] Forward/backward placebo")
    rows = []
    specs = {
        "none": [],
        "month": ["month"],
        "month+location": ["month", "loc"],
        "month+surveyor+location": ["month", "surv", "loc"],
    }
    for label, (h, pk, fk) in {
        f"count h={300}": (300, f"pastN_h300", f"futN_h300"),
        f"kernel h={H0}": (H0, f"pastK_h{H0}", f"futK_h{H0}"),
    }.items():
        xp, xf = np.log1p(F[pk].to_numpy()), np.log1p(F[fk].to_numpy())
        y = v["rating"].to_numpy(float)
        for sname, fes in specs.items():
            groups = [v[g].to_numpy() for g in fes]
            X = np.c_[xp, xf]
            yy, XX = (y, X) if not groups else (demean(y, groups), demean(X, groups))
            for var, cols, names in [("past", [0], ["past"]), ("future", [1], ["future"]),
                                     ("joint", [0, 1], ["past", "future"])]:
                res = ols_cluster(yy, XX[:, cols], v["cbg"], names)
                for nm in names:
                    rows.append({"field": label, "spec": sname, "model": var,
                                 "regressor": nm, "coef": res.params[nm],
                                 "se": res.bse[nm], "t": res.tvalues[nm]})
                if var == "joint":
                    L = np.zeros(len(res.params)); L[1] = 1; L[2] = -1
                    ct = res.t_test(L)
                    rows.append({"field": label, "spec": sname, "model": var,
                                 "regressor": "past - future",
                                 "coef": float(ct.effect), "se": float(ct.sd),
                                 "t": float(ct.tvalue)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TRI, "t1_placebo.csv"), index=False)
    show = tab[tab["model"] == "joint"].pivot_table(
        index=["field", "spec"], columns="regressor", values=["coef", "t"])
    print(show.round(4).to_string())

    # within-location Pearson vs tau (the plan's Test 1 table) at h = H0
    print("\n  Test-1 replication (h = 200): cross-sectional Spearman / "
          "within-location Pearson of rating with log lambda-hat")
    ylc = demean(v["rating"].to_numpy(float), [v["loc"].to_numpy()])
    trows = []
    for tau in TAUS:
        ll, _ = safe_log(F[f"lam_h{H0}_t{tau}"].to_numpy())
        rho = spearmanr(v["rating"], ll).correlation
        wl = demean(ll, [v["loc"].to_numpy()])
        pw = np.corrcoef(ylc, wl)[0, 1]
        trows.append({"tau": tau, "spearman_xs": rho, "pearson_within_loc": pw})
        print(f"    tau={tau:>3}: Spearman {rho:+.3f}   within-loc Pearson {pw:+.4f}")
    pd.DataFrame(trows).to_csv(os.path.join(TRI, "t1_tau_sweep.csv"), index=False)

    # the headline number for the gate
    j = tab[(tab["field"] == f"kernel h={H0}") & (tab["model"] == "joint")]
    for sname in specs:
        r = j[j["spec"] == sname].set_index("regressor")
        summary[f"placebo_{sname}"] = {
            "past": round(r.loc["past", "coef"], 4), "future": round(r.loc["future", "coef"], 4),
            "diff": round(r.loc["past - future", "coef"], 4),
            "diff_t": round(r.loc["past - future", "t"], 2)}

    # figure: past vs future coefficient by spec
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, label in zip(axes, [f"count h=300", f"kernel h={H0}"]):
        sub = tab[(tab["field"] == label) & (tab["model"] == "joint")
                  & tab["regressor"].isin(["past", "future"])]
        ypos = np.arange(len(specs))
        for k, (reg, col) in enumerate([("past", C1), ("future", C2)]):
            r = sub[sub["regressor"] == reg].set_index("spec").loc[list(specs)]
            ax.errorbar(r["coef"], ypos + (k - 0.5) * 0.25, xerr=1.96 * r["se"],
                        fmt="o", color=col, ms=5, capsize=2, lw=1.2, label=reg)
        ax.axvline(0, color=GRID, lw=1)
        ax.set_yticks(ypos); ax.set_yticklabels(list(specs), fontsize=8)
        style(ax, f"Placebo: {label}, joint model", "coef on log1p(90-day count)")
        ax.grid(axis="y", visible=False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "t1_placebo.png"), dpi=160)
    plt.close(fig)


def t2_spline(v, F, summary):
    print("\n[T2] Spline of rating vs log lambda-hat (h=200, tau=120)")
    y = v["rating"].to_numpy(float)
    # the plan's own construction: log1p(count within 300 m, past 90 d)
    for label, col in [("plan: log1p past-90d count, 300 m", "pastN_h300"),
                       ("all-time count within 300 m", "n_within_h300"),
                       ("kernel h=500, tau=120", "lam_h500_t120")]:
        x = np.log1p(F[col].to_numpy()) if "count" in label else safe_log(F[col].to_numpy())[0]
        dec = pd.qcut(x, 10, labels=False, duplicates="drop")
        dm = pd.Series(y).groupby(dec).mean()
        print(f"  {label:>36}: decile means  " + "  ".join(f"{m:.3f}" for m in dm))
        summary[f"decile_means[{label}]"] = [round(m, 3) for m in dm]
    groups = [v["month"].to_numpy(), v["surv"].to_numpy()]
    curves = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (label, col) in zip(axes, [("raw", "lam_h200_t120"), ("corr", "lam_corr_h200")]):
        ll, eps = safe_log(F[col].to_numpy())
        dec = pd.qcut(ll, 10, labels=False, duplicates="drop")
        dm = pd.Series(y).groupby(dec).mean()
        print(f"  {label:>4}: decile means  " + "  ".join(f"{m:.3f}" for m in dm))
        summary[f"decile_means_{label}"] = [round(m, 3) for m in dm]

        B = dmatrix("cr(x, df=5) - 1", {"x": ll}, return_type="dataframe")
        Bd, yd = demean(B.to_numpy(), groups), demean(y, groups)
        res = sm.OLS(yd, Bd).fit()
        grid = np.linspace(np.quantile(ll, 0.005), np.quantile(ll, 0.995), 200)
        Bg = build_design_matrices([B.design_info], {"x": grid})[0]
        curve = np.asarray(Bg) @ res.params - (np.asarray(B) @ res.params).mean() + y.mean()
        curves[label] = (grid, curve)
        slope = np.gradient(curve, grid)
        top = grid >= np.quantile(ll, 0.8)
        turnover = float(curve.max() - curve[top][-1])
        share_neg = float((slope[top] < 0).mean())
        summary[f"spline_{label}"] = {"drop_from_peak_to_p99.5": round(turnover, 3),
                                      "share_negative_slope_top_quintile": round(share_neg, 2)}
        print(f"        spline: peak {curve.max():.3f}, value at p99.5 {curve[top][-1]:.3f} "
              f"(drop {turnover:.3f}); slope negative on {share_neg:.0%} of top quintile")

        ax.plot(grid, curve, color=C1 if label == "raw" else C2, lw=2)
        qs = np.quantile(ll, np.linspace(0.05, 0.95, 10))
        ax.scatter(qs, dm.to_numpy()[:len(qs)], color=INK, s=14, zorder=3,
                   label="decile mean (no FE)")
        style(ax, f"E[rating | log λ̂_{label}]  (month + surveyor FE)",
              f"log λ̂_{label}", "rating")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "t2_spline.png"), dpi=160)
    plt.close(fig)


def t3_variance_budget(v, F, summary):
    print("\n[T3] Variance budget and identification")
    ll, _ = safe_log(F[f"lam_h{H0}_t{TAU0}"].to_numpy())
    y = v["rating"].to_numpy(float)
    out = {}
    for name, fes in [("month", ["month"]), ("surveyor", ["surv"]),
                      ("month+surveyor", ["month", "surv"]), ("location", ["loc"]),
                      ("cbg", ["cbg"])]:
        g = [v[k].to_numpy() for k in fes]
        out[name] = {"R2_loglam": r2_of_fe(ll, g), "R2_rating": r2_of_fe(y, g)}
        print(f"  R² on {name:>15} FE:  log λ̂ {out[name]['R2_loglam']:.3f}   "
              f"rating {out[name]['R2_rating']:.3f}")
    pd.DataFrame(out).T.to_csv(os.path.join(TRI, "t3_variance_budget.csv"))
    summary["variance_budget"] = {k: {kk: round(vv, 3) for kk, vv in d.items()}
                                  for k, d in out.items()}

    # surveyor-location bipartite connectivity
    S, L = v["surv"].max() + 1, v["loc"].max() + 1
    pairs = v[["surv", "loc"]].drop_duplicates()
    A = sparse.coo_matrix((np.ones(len(pairs)), (pairs["surv"], S + pairs["loc"])),
                          shape=(S + L, S + L))
    ncomp, lab = connected_components(A, directed=False)
    comp_size = np.bincount(lab[v["loc"].to_numpy() + S])
    big = comp_size.max() / len(v)
    print(f"  surveyor-location graph: {ncomp} components; largest holds "
          f"{big:.1%} of visits; {S} surveyors, {L:,} locations")
    summary["connectivity"] = {"components": int(ncomp), "largest_share": round(big, 4)}

    # surveyor x block crosstab
    locs = v.groupby("loc")[["x", "y"]].first()
    km = KMeans(n_clusters=N_BLOCKS, random_state=0, n_init=10).fit(locs.to_numpy())
    v["block"] = km.labels_[v["loc"].to_numpy()]
    ct = pd.crosstab(v["surveyor"], v["block"])
    ct.to_csv(os.path.join(TRI, "t3_surveyor_by_block.csv"))
    surv_per_block = (ct > 0).sum(axis=0)
    blocks_per_surv = (ct > 0).sum(axis=1)
    top_share = (ct.max(axis=0) / ct.sum(axis=0))
    print(f"  {N_BLOCKS} blocks: surveyors per block median {surv_per_block.median():.0f} "
          f"(min {surv_per_block.min()}, max {surv_per_block.max()}); "
          f"blocks per surveyor median {blocks_per_surv.median():.0f}; "
          f"dominant surveyor's share of a block median {top_share.median():.2f} "
          f"(max {top_share.max():.2f})")
    summary["surveyor_block"] = {"surv_per_block_median": float(surv_per_block.median()),
                                 "blocks_per_surv_median": float(blocks_per_surv.median()),
                                 "top_share_median": round(float(top_share.median()), 3),
                                 "top_share_max": round(float(top_share.max()), 3)}


def t4_atoms(ev, v, F, summary):
    print("\n[T4] Geocoding atoms and dedup audit")
    xy = ev["xy"]; yr = ev["ts"].dt.year.to_numpy()
    rows = []
    for label, m in [("all", np.ones(len(xy), bool))] + [
            (str(y), yr == y) for y in sorted(np.unique(yr))]:
        sub = pd.DataFrame(np.round(xy[m], 1), columns=["x", "y"])
        vc = sub.value_counts()
        rows.append({"years": label, "n": int(m.sum()), "distinct": len(vc),
                     "shared_share": float(vc[vc > 1].sum() / m.sum()),
                     "max_multiplicity": int(vc.max()),
                     "top5": ",".join(str(x) for x in vc.head(5).tolist())})
    at = pd.DataFrame(rows)
    print(at.to_string(index=False))
    at.to_csv(os.path.join(TRI, "t4_atoms.csv"), index=False)

    # spacing between distinct coordinates (the geocoding grid)
    ux = np.unique(np.round(xy, 1), axis=0)
    tree = cKDTree(ux)
    dnn, _ = tree.query(ux, k=2)
    q = np.quantile(dnn[:, 1], [0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"  {len(ux):,} distinct coordinates; nearest-other-atom distance "
          f"p10 {q[0]:.0f} m, p25 {q[1]:.0f}, p50 {q[2]:.0f}, p75 {q[3]:.0f}, p90 {q[4]:.0f}")
    summary["atom_spacing_m"] = {"p10": round(q[0]), "p25": round(q[1]), "p50": round(q[2]),
                                 "p75": round(q[3]), "p90": round(q[4])}

    # how many distinct atoms does a kernel at a survey point see?
    vxy = v[["x", "y"]].to_numpy()
    seen = {}
    for h in H_SWEEP:
        cnt = np.array([len(s) for s in tree.query_ball_point(vxy[::10], h)])
        seen[h] = {"median_atoms": float(np.median(cnt)), "p10_atoms": float(np.quantile(cnt, .1)),
                   "share_lt5": float((cnt < 5).mean())}
        print(f"  h={h:>3}: distinct atoms within h of a survey point — median "
              f"{seen[h]['median_atoms']:.0f}, p10 {seen[h]['p10_atoms']:.0f}, "
              f"share with <5 atoms {seen[h]['share_lt5']:.1%}")
    summary["atoms_within_h"] = seen

    # dedup into "piles": an event with a prior event within 50 m and 14 d
    order = np.argsort(ev["t"]); t = ev["t"][order]; pxy = xy[order]
    ptree = cKDTree(pxy)
    nb = ptree.query_ball_point(pxy, 50.0)
    repeat = np.zeros(len(t), bool)
    for k, lst in enumerate(nb):
        if len(lst) > 1:
            arr = np.asarray(lst)
            prior = arr[(arr < k)]
            if len(prior) and (t[k] - t[prior].max()) <= 14:
                repeat[k] = True
    piles = int((~repeat).sum())
    print(f"  space-time dedup (50 m / 14 d): {len(t):,} events -> {piles:,} piles "
          f"({repeat.mean():.1%} of events are repeats)")
    summary["dedup"] = {"events": int(len(t)), "piles": piles, "repeat_share": round(float(repeat.mean()), 4)}


def t5_routing(v, F, es, ev_t, summary):
    print("\n[T5] Endogenous survey routing")
    n = len(v)
    weeks = np.arange(-ES_WEEKS, ES_WEEKS)            # floor((t_visit - t_event)/7)
    per_visit = es / n
    # week k >= 0 : events k weeks BEFORE the visit; k < 0 : after
    pre = per_visit[(weeks >= 0) & (weeks < 4)].mean()
    post = per_visit[(weeks >= -4) & (weeks < 0)].mean()
    far = per_visit[np.abs(weeks + 0.5) >= 8].mean()
    print(f"  events within {H0} m per visit-week: 0-4 wk before {pre:.3f}, "
          f"0-4 wk after {post:.3f}, |lag| >= 8 wk {far:.3f}  "
          f"(before/after ratio {pre / post:.3f})")

    # per-visit paired test with a citywide pre/post normalisation
    ev_days = np.floor(np.asarray(ev_t)).astype(int)
    daily = np.bincount(ev_days - ev_days.min(), minlength=ev_days.max() - ev_days.min() + 1)
    cum = np.r_[0, np.cumsum(daily)]
    d0 = ev_days.min()
    tv = np.floor(v["t_days"].to_numpy()).astype(int) - d0
    def city(a, b):
        a = np.clip(a, 0, len(cum) - 1); b = np.clip(b, 0, len(cum) - 1)
        return cum[b] - cum[a]
    city_past = city(tv - ROUTING_WIN, tv)
    city_fut = city(tv, tv + ROUTING_WIN)
    past, fut = F[f"past30N_h{H0}"].to_numpy(), F[f"fut30N_h{H0}"].to_numpy()
    ratio_local = past.sum() / fut.sum()
    ratio_city = city_past.sum() / city_fut.sum()
    adj_diff = past - fut * (city_past / np.maximum(city_fut, 1))
    mean_adj = adj_diff.mean()
    g = v["cbg"].to_numpy()
    se_adj = float(np.sqrt((np.bincount(g, weights=adj_diff - mean_adj) ** 2).sum()) / n)
    print(f"  30-day window: local past/future ratio {ratio_local:.3f} vs citywide "
          f"{ratio_city:.3f}; excess ratio {ratio_local / ratio_city:.3f}")
    print(f"  city-adjusted per-visit (past - future) mean {mean_adj:+.4f} "
          f"(cluster SE {se_adj:.4f}, t = {mean_adj / se_adj:+.2f}); "
          f"relative to mean past count {past.mean():.3f} → {mean_adj / past.mean():+.1%}")
    summary["routing"] = {"pre4wk": round(pre, 4), "post4wk": round(post, 4),
                          "ratio_local_30d": round(ratio_local, 4), "ratio_city_30d": round(ratio_city, 4),
                          "adj_diff_mean": round(mean_adj, 4), "adj_diff_t": round(mean_adj / se_adj, 2),
                          "adj_diff_rel": round(mean_adj / past.mean(), 4)}

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    xs = -weeks - 0.5                                   # event time relative to visit, weeks
    cols = np.where(weeks >= 0, C1, C2)
    ax.bar(xs, per_visit, width=0.9, color=cols, edgecolor="white", linewidth=1)
    ax.axvline(0, color=INK, lw=1)
    style(ax, f"311 dumping reports within {H0} m of a survey point, by week relative to the visit",
          "weeks (negative = before the visit)", "reports per visit-week")
    ax.text(-ES_WEEKS + 0.2, per_visit.max() * 0.95, "before", color=C1, fontsize=9)
    ax.text(ES_WEEKS - 2.2, per_visit.max() * 0.95, "after", color=C2, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "t5_event_study.png"), dpi=160)
    plt.close(fig)


def t6_spec_curve(v, F, summary):
    print("\n[T6] Specification curve for beta-hat = -c/b (OLS on the raw rating)")
    ok = np.isfinite(v["w_cbg"].to_numpy()) & np.isfinite(v["log_expo_dens"].to_numpy())
    vv = v[ok].reset_index(drop=True); FF = F[ok].reset_index(drop=True)
    y = vv["rating"].to_numpy(float)
    specs = {"none": [], "month": ["month"], "month+surveyor": ["month", "surv"]}
    rows, rhos = [], []
    for h in [150, 200, 300, 500, 800]:
        ll, _ = safe_log(FF[f"lam_h{h}_t{TAU0}"].to_numpy())
        den = FF[f"wbar_den_h{h}"].to_numpy()
        wbar = np.where(den > 0, FF[f"wbar_num_h{h}"].to_numpy() / np.maximum(den, 1e-300),
                        vv["w_cbg"].to_numpy())
        wcbg = vv["w_cbg"].to_numpy()
        rho = np.polyfit(wcbg, wbar, 1)[0]
        rhos.append({"h": h, "rho": rho, "corr": np.corrcoef(wcbg, wbar)[0, 1]})
        for wname, w in [("w_R(s)", wcbg), ("w_bar", wbar)]:
            for expo in [False, True]:
                x1 = ll - (vv["log_expo_dens"].to_numpy() if expo else 0.0)
                for sname, fes in specs.items():
                    g = [vv[k].to_numpy() for k in fes]
                    X = np.c_[x1, w]
                    yy, XX = (y, X) if not g else (demean(y, g), demean(X, g))
                    res = ols_cluster(yy, XX, vv["cbg"], ["loglam", "w"])
                    b, c = res.params["loglam"], res.params["w"]
                    V = res.cov_params().loc[["loglam", "w"], ["loglam", "w"]].to_numpy()
                    grad = np.array([c / b ** 2, -1.0 / b])
                    beta = -c / b
                    se = float(np.sqrt(grad @ V @ grad))
                    rows.append({"h": h, "w": wname, "exposure": expo, "fe": sname,
                                 "b": b, "se_b": res.bse["loglam"], "c": c, "se_c": res.bse["w"],
                                 "beta_hat": beta, "se_beta": se, "n": len(yy)})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TRI, "t6_spec_curve.csv"), index=False)
    rho_tab = pd.DataFrame(rhos)
    rho_tab.to_csv(os.path.join(TRI, "t6_rho_h.csv"), index=False)
    print("  rho(h) = slope of w_bar on w_R(s):  " +
          "  ".join(f"h={r.h}: {r.rho:.3f}" for r in rho_tab.itertuples()))
    piv = tab.pivot_table(index=["fe", "w", "exposure"], columns="h", values="beta_hat")
    print("  beta-hat by spec (columns = h):")
    print(piv.round(3).to_string())
    q = tab["beta_hat"].quantile([0.1, 0.5, 0.9])
    inside = tab["beta_hat"].between(0.470, 0.686).mean()
    print(f"  {len(tab)} specs: beta-hat median {q[0.5]:.3f}, p10-p90 [{q[0.1]:.3f}, {q[0.9]:.3f}]; "
          f"{inside:.0%} inside 05a's [0.470, 0.686]; median delta-method SE {tab['se_beta'].median():.3f}")
    summary["spec_curve"] = {"n_specs": len(tab), "median": round(q[0.5], 3),
                             "p10": round(q[0.1], 3), "p90": round(q[0.9], 3),
                             "share_in_05a_band": round(inside, 3),
                             "median_se": round(float(tab["se_beta"].median()), 3),
                             "rho_h": {int(r.h): round(r.rho, 3) for r in rho_tab.itertuples()}}

    fig, ax = plt.subplots(figsize=(10, 4))
    t = tab.sort_values("beta_hat").reset_index(drop=True)
    ax.axhspan(0.470, 0.686, color=GRID, alpha=0.6, lw=0, label="05a 94% interval")
    ax.axhline(BETA0, color=MUTED, lw=1, ls="--")
    colmap = {"none": C1, "month": C2, "month+surveyor": C3}
    for fe, col in colmap.items():
        s = t[t["fe"] == fe]
        for wname, mk in [("w_R(s)", "o"), ("w_bar", "s")]:
            ss = s[s["w"] == wname]
            ax.errorbar(ss.index, ss["beta_hat"], yerr=1.96 * ss["se_beta"], fmt=mk,
                        color=col, ms=4, lw=0.8, capsize=0, label=f"{fe}, {wname}")
    style(ax, "β̂ = −c/b across specifications (sorted); circles w_R(s), squares w̄; "
              "bars = 95% delta-method CI, CBG-clustered",
          "specification rank", "β̂")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "t6_spec_curve.png"), dpi=160)
    plt.close(fig)

    # bandwidth sweep of the raw association (the empirical case for point level)
    sw = []
    for h in H_SWEEP:
        ll, _ = safe_log(F[f"lam_h{h}_t{TAU0}"].to_numpy())
        sw.append({"h": h, "spearman": spearmanr(v["rating"], ll).correlation,
                   "zero_share": float((F[f"lam_h{h}_t{TAU0}"] == 0).mean())})
    sw = pd.DataFrame(sw)
    sw.to_csv(os.path.join(TRI, "t6_bandwidth_sweep.csv"), index=False)
    print("  bandwidth sweep (tau=120) Spearman(rating, log λ̂): " +
          "  ".join(f"h={r.h}: {r.spearman:+.3f}" for r in sw.itertuples()))
    print("  share of visits with λ̂ = 0:                      " +
          "  ".join(f"h={r.h}: {r.zero_share:.2%}" for r in sw.itertuples()))
    summary["bandwidth_sweep"] = {int(r.h): round(r.spearman, 4) for r in sw.itertuples()}
    # CBG-level benchmark: same causal-exponential pooling, but the "kernel" is
    # the CBG polygon. This is what aggregation-to-CBG would deliver.
    bench = cbg_benchmark(v, F, summary)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(sw["h"], sw["spearman"], "-o", color=C1, lw=2, ms=5, label="point kernel, quartic radius h")
    ax.axhline(bench["per_area"], color=C2, lw=1.5, ls="--", label="CBG-pooled rate per km²")
    ax.axhline(bench["per_exposure"], color=C3, lw=1.5, ls=":", label="CBG-pooled rate per exposure")
    ax.set_xscale("log"); ax.set_xticks(H_SWEEP); ax.set_xticklabels([str(h) for h in H_SWEEP])
    for r in sw.itertuples():
        ax.annotate(f"{r.spearman:+.3f}", (r.h, r.spearman), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color=MUTED)
    style(ax, "Spearman(rating, log λ̂) by spatial bandwidth (τ = 120 d)", "h (m)", "Spearman ρ")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "t6_bandwidth_sweep.png"), dpi=160)
    plt.close(fig)

    # bulk-item outcome: the dumping-specific check on casual-litter OVB
    print("\n  Bulk-item outcome (linear probability), same specs, no exposure denominator:")
    yb = vv["bulk"].to_numpy(float)
    brow = []
    for h in [200, 500]:
        ll, _ = safe_log(FF[f"lam_h{h}_t{TAU0}"].to_numpy())
        den = FF[f"wbar_den_h{h}"].to_numpy()
        wbar = np.where(den > 0, FF[f"wbar_num_h{h}"].to_numpy() / np.maximum(den, 1e-300),
                        vv["w_cbg"].to_numpy())
        for wname, w in [("w_R(s)", vv["w_cbg"].to_numpy()), ("w_bar", wbar)]:
            for sname in ["month", "month+surveyor"]:
                g = [vv[k].to_numpy() for k in specs[sname]]
                X = np.c_[ll, w]
                for yname, yy0 in [("rating", y), ("bulk", yb)]:
                    yy, XX = demean(yy0, g), demean(X, g)
                    res = ols_cluster(yy, XX, vv["cbg"], ["loglam", "w"])
                    b, c = res.params["loglam"], res.params["w"]
                    V = res.cov_params().loc[["loglam", "w"], ["loglam", "w"]].to_numpy()
                    grad = np.array([c / b ** 2, -1.0 / b])
                    brow.append({"h": h, "w": wname, "fe": sname, "outcome": yname,
                                 "b": b, "t_b": res.tvalues["loglam"], "c": c,
                                 "t_c": res.tvalues["w"], "b_std": b / yy0.std(),
                                 "beta_hat": -c / b, "se_beta": float(np.sqrt(grad @ V @ grad))})
    bt = pd.DataFrame(brow)
    bt.to_csv(os.path.join(TRI, "t6_bulk_vs_rating.csv"), index=False)
    piv = bt.pivot_table(index=["fe", "w", "h"], columns="outcome",
                         values=["beta_hat", "b_std", "t_c"])
    print(piv.round(3).to_string())
    summary["bulk_vs_rating"] = {
        "beta_hat_bulk_median": round(float(bt[bt.outcome == "bulk"]["beta_hat"].median()), 3),
        "beta_hat_rating_median": round(float(bt[bt.outcome == "rating"]["beta_hat"].median()), 3),
        "b_std_bulk_median": round(float(bt[bt.outcome == "bulk"]["b_std"].median()), 3),
        "b_std_rating_median": round(float(bt[bt.outcome == "rating"]["b_std"].median()), 3)}


def cbg_benchmark(v, F, summary, tau=TAU0):
    """Causal-exponential pooled dumping intensity of the visit's own CBG,
    evaluated at the visit date; per km² and per SafeGraph exposure."""
    d = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    d = d[d["GEOID"].notna()]
    lam = np.zeros(len(v))
    tv_all = v["t_days"].to_numpy()
    vis_by_g = v.groupby("GEOID").indices
    for g, sub in d.groupby("GEOID"):
        idx = vis_by_g.get(g)
        if idx is None:
            continue
        t = np.sort(sub["t_days"].to_numpy())
        cs = np.cumsum(np.exp(t / tau))
        tv = tv_all[idx]
        k = np.searchsorted(t, tv, side="right")
        lam[idx] = np.where(k > 0, np.exp(-tv / tau) * cs[np.maximum(k - 1, 0)], 0.0)
    per_area = safe_log(lam / v["area_km2"].to_numpy())[0]
    per_expo = safe_log(lam / v["exposure"].to_numpy())[0]
    r_area = spearmanr(v["rating"], per_area).correlation
    r_expo = spearmanr(v["rating"], per_expo).correlation
    r_best = max(summary["bandwidth_sweep"].values())
    print(f"  CBG-level benchmark (same τ, polygon instead of kernel): Spearman "
          f"per km² {r_area:+.3f}, per exposure {r_expo:+.3f}; best point kernel {r_best:+.3f}; "
          f"zero share {float((lam == 0).mean()):.2%}")
    summary["cbg_benchmark"] = {"spearman_per_area": round(r_area, 4),
                                "spearman_per_exposure": round(r_expo, 4),
                                "best_point_kernel": round(r_best, 4)}
    return {"per_area": r_area, "per_exposure": r_expo}


# =============================================================================
# main
# =============================================================================

def main():
    t_start = time.time()
    print("=" * 68)
    print("00_triage.py — Day-1 gate")
    print("=" * 68)
    v = load_visits()
    ev = load_events(v)

    print("\n[B] Kernel passes ...")
    F = pd.DataFrame(index=v.index)
    es_h0 = None
    for h in H_SWEEP:
        t0 = time.time()
        o = kernel_pass(h, v, ev, want_es=(h == H0))
        for tau in TAUS:
            F[f"lam_h{h}_t{tau}"] = o[f"lam_t{tau}"]
        for k in ["pastK", "futK", "pastN", "futN", "past30N", "fut30N",
                  "wbar_num", "wbar_den", "lam_corr", "n_within"]:
            F[f"{k}_h{h}"] = o[k]
        if h == H0:
            es_h0 = o["es"]
        print(f"  h={h:>3} m: {o['n_pairs']:>12,} visit-event pairs   "
              f"({time.time() - t0:5.1f} s)   mean events within h: {o['n_within'].mean():7.1f}")
    ok = brute_check(v, ev, F)
    F.to_parquet(os.path.join(TRI, "visit_fields.parquet"), index=False)
    v.drop(columns=["types"]).to_parquet(os.path.join(TRI, "visits.parquet"), index=False)

    summary = {"n_visits": int(len(v)), "n_events": int(len(ev["t"])),
               "kernel_unit_test_ok": bool(ok)}
    t1_placebo(v, F, summary)
    t2_spline(v, F, summary)
    t3_variance_budget(v, F, summary)
    t4_atoms(ev, v, F, summary)
    t5_routing(v, F, es_h0, ev["t"], summary)
    t6_spec_curve(v, F, summary)

    with open(os.path.join(TRI, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nDone in {time.time() - t_start:.0f} s. Outputs in {os.path.relpath(TRI, ROOT)}/")


if __name__ == "__main__":
    main()
