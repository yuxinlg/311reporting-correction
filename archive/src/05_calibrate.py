"""
05_calibrate.py
===============
Stage 3, step 1: the calibration ladder at fixed (h, tau), the profile
likelihood for beta, spatial block bootstrap, negative controls and the link
checks. Step 2 (06_cv.py) does the buffered blocked CV.

Outcome: Litter Rating 1-4 -> proportional-odds logit (three sigmoids sharing
one linear predictor). Field: the ratio form (num / kernel-smoothed exposure),
Gamma-shrunk with ALPHA prior events, tau = 120 d, h in H_LIST (default H0).

Ladder (all with month-of-sample + surveyor FE unless stated)
  S0     g(log lam_raw)                         g = natural cubic spline, 5 df
  S0lin  b log lam_raw                          the slope version
  S1'    b log lam_raw + c w_bar                closed-form beta = -c/b
                                                (delta-method + Fieller CI,
                                                 block-clustered)
  S1'M   Mundlak: within/between split of log lam_raw + c w_bar; beta = -c/b_between
  S2     g(log lam_corr(beta0))                 plug-in Stage-1a beta0
  SP     profile likelihood over beta           HEADLINE
         lam_corr(beta) = sum_j exp(-beta w_R(s_j)) K v / D, refit g + FE per beta
w_R: kernel surface at the report (primary) or CBG posterior mean.

Uncertainty: cluster-robust by CV block for every coefficient; spatial block
bootstrap (N_BOOT resamples of the 40 blocks) for beta from S1' and SP, and a
bootstrap-recalibrated critical value for the profile CI.

Controls and checks
  - outage (Street Light Outage) field as a placebo lambda: b should be ~ 0
  - torus-shifted w_R raster (N_PERM shifts, autocorrelation preserved):
    null distribution for the S2 - S0 improvement
  - dedup: field on 50 m / 14 d piles should need a smaller |c|
  - differential outcome: bulk vs casual binary logits, raw vs corrected
  - partial proportional odds: -c_k / b_k per cutpoint
  - the literal 1 + 3 sigmoid(eta) by NLS; range spanned
  - alpha sensitivity {0.5, 1, 2, 5}
  - level reconciliation: CBG means of fitted E[rating] vs CBG mean rating
  - linearisation figure: g on raw vs corrected fields

Outputs  output/calib/*.csv, *.json, output/figures/05_*.png
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, norm
from scipy.optimize import least_squares
from scipy.special import expit
from patsy import dmatrix, build_design_matrices
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import accumulate_st, shrunk_log
from ordinal import OrderedLogit

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
CAL = os.path.join(OUT, "calib")
FIG = os.path.join(OUT, "figures")
os.makedirs(CAL, exist_ok=True); os.makedirs(FIG, exist_ok=True)

H_LIST = [500, 800, 1200]
H0, TAU0 = 800, 120
ALPHA = 1.0
SPLINE_DF = 5
BETA_GRID = np.round(np.arange(-0.2, 2.0001, 0.05), 3)      # 45 points
BOOT_GRID = np.round(np.arange(-0.2, 2.0001, 0.20), 3)      # 12 points, parabola-interpolated
N_BOOT = 100
BOOT_GTOL = 1e-3          # looser optimiser tolerance for the resamples (ll error << LR scale)
REUSE_BOOTSTRAP = True    # load calib/bootstrap.csv if it already holds N_BOOT resamples (58 min to redo)
ALPHA_LIST = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
ALPHA_ALT = 0.25          # second alpha at which the profile is also run
N_PERM = 50
CHI2_95_HALF = 1.92
WR_PRIMARY = "wR_pre24_naive_h600"
FE_MAIN = ("month", "surv")
SEED = 0

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

OL = OrderedLogit(4)


# =============================================================================
# data and fields
# =============================================================================

def load():
    v = pd.read_parquet(os.path.join(OUT, "litter_panel.parquet"))
    F = pd.read_parquet(os.path.join(OUT, "visit_fields.parquet"))
    with open(os.path.join(OUT, "field_summary.json")) as f:
        beta0 = float(json.load(f)["beta0"])
    d = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    w = pd.read_parquet(os.path.join(OUT, "wR_events.parquet"))
    rep = pd.read_parquet(os.path.join(OUT, "reporting_cbg.parquet")).set_index("GEOID")["w_R_mean"]
    wk = w[WR_PRIMARY].to_numpy(float)
    wc = d["GEOID"].map(rep).to_numpy(float)
    ev = {"xy": d[["x", "y"]].to_numpy(float), "t": d["t_days"].to_numpy(float),
          "wk": np.where(np.isfinite(wk), wk, 0.0), "wc": np.where(np.isfinite(wc), wc, 0.0)}
    ev["tree"] = cKDTree(ev["xy"])
    # point-level w_R at the visit, for filling w_bar where the kernel sum is empty
    v["wR_point_k"] = v[WR_PRIMARY].to_numpy(float)
    v["wR_point_c"] = v["w_R_cbg"].to_numpy(float)
    print(f"  {len(v):,} visits, {len(ev['t']):,} reports, beta0 = {beta0:.4f}")
    return v, F, beta0, ev


def fields_at(F, v, h, tau=TAU0, alpha=ALPHA):
    """log fields (ratio form) at bandwidth h, plus w_bar with point fallback."""
    den = F[f"den_expo_h{h}"].to_numpy()
    out = {"den": den}
    out["raw"], out["lam_bar"] = shrunk_log(F[f"num_h{h}_t{tau}"].to_numpy(), den, alpha)
    out["corr_k"], _ = shrunk_log(F[f"num_k_h{h}_t{tau}"].to_numpy(), den, alpha)
    out["corr_c"], _ = shrunk_log(F[f"num_c_h{h}_t{tau}"].to_numpy(), den, alpha)
    for k in ["k", "c"]:
        wb = F[f"wbar_{k}_h{h}_t{tau}"].to_numpy()
        pt = v[f"wR_point_{k}"].to_numpy()
        wb = np.where(np.isfinite(wb), wb, pt)
        out[f"wbar_{k}"] = np.where(np.isfinite(wb), wb, 0.0)
    out["zero"] = (F[f"num_h{h}_t{tau}"].to_numpy() == 0).astype(float)
    return out


def profile_numerators(v, ev, h, grid, tau=TAU0):
    """num(beta) = sum_j exp(-beta w_R,j) K v for every beta in grid, both w_R versions.
    One streaming pass with 2 * len(grid) weight vectors; cached."""
    path = os.path.join(CAL, f"profile_num_h{h}_t{tau}.npz")
    if os.path.exists(path):
        z = np.load(path)
        if np.allclose(z["grid"], grid):
            return z["num_k"], z["num_c"]
    t0 = time.time()
    weights = {}
    for i, b in enumerate(grid):
        weights[f"k{i}"] = np.exp(-b * ev["wk"])
        weights[f"c{i}"] = np.exp(-b * ev["wc"])
    o = accumulate_st(v[["x", "y"]].to_numpy(), v["t_days"].to_numpy(), ev["xy"], ev["t"],
                      h, [tau], weights, src_tree=ev["tree"])
    num_k = np.column_stack([o[(tau, f"k{i}")] for i in range(len(grid))])
    num_c = np.column_stack([o[(tau, f"c{i}")] for i in range(len(grid))])
    np.savez_compressed(path, grid=grid, num_k=num_k, num_c=num_c)
    print(f"    profile numerators h={h}: {len(grid)} betas x 2, {time.time() - t0:.0f} s")
    return num_k, num_c


# =============================================================================
# design matrices and fitting
# =============================================================================

def spline_basis(x, info=None):
    """Natural cubic spline, SPLINE_DF df, centred and with one column dropped
    so it does not span the constant (the cutpoints carry the intercept)."""
    if info is None:
        B = dmatrix(f"cr(x, df={SPLINE_DF}) - 1", {"x": x}, return_type="dataframe")
        info = {"di": B.design_info, "mean": B.to_numpy().mean(0)}
        Bm = B.to_numpy()
    else:
        Bm = np.asarray(build_design_matrices([info["di"]], {"x": x})[0])
    return (Bm - info["mean"])[:, 1:], info


def fe_dummies(v, fe):
    cols, names = [], []
    for f in fe:
        D = pd.get_dummies(v[f], drop_first=True)
        cols.append(D.to_numpy(float)); names += [f"{f}_{c}" for c in D.columns]
    return (np.hstack(cols), names) if cols else (np.zeros((len(v), 0)), [])


def design(v, x, fe=FE_MAIN, spline=True, extra=None, mundlak=False, info=None):
    cols, names = [], []
    if spline:
        B, info = spline_basis(x, info)
        cols.append(B); names += [f"g{k}" for k in range(B.shape[1])]
    elif mundlak:
        xm = pd.Series(x).groupby(v["loc"].to_numpy()).transform("mean").to_numpy()
        cols.append(np.c_[x - xm, xm - xm.mean()]); names += ["loglam_within", "loglam_between"]
    else:
        cols.append((x - x.mean())[:, None]); names += ["loglam"]
    for name, arr in (extra or {}).items():
        cols.append((arr - arr.mean())[:, None]); names.append(name)
    D, dn = fe_dummies(v, fe)
    cols.append(D); names += dn
    return np.hstack(cols), names, info


def fit(X, y, w=None, theta0=None, groups=None, gtol=1e-5):
    th, ll, ok = OL.fit(X, y, w, theta0, gtol=gtol)
    out = {"theta": th, "ll": ll, "ok": ok}
    if groups is not None:
        out["V"] = OL.cov_cluster(th, X, y, groups, w)
        out["se"] = np.sqrt(np.diag(out["V"]))
    return out


def beta_ratio(res, names, b_name="loglam", c_name="wbar"):
    """beta = -c/b with delta-method SE and Fieller 95% interval."""
    ib, ic = names.index(b_name), names.index(c_name)
    b, c = res["theta"][ib], res["theta"][ic]
    V = res["V"][np.ix_([ib, ic], [ib, ic])]
    beta = -c / b
    grad = np.array([c / b ** 2, -1.0 / b])
    se = float(np.sqrt(grad @ V @ grad))
    z2 = norm.ppf(0.975) ** 2
    # Fieller: (c + beta b)^2 <= z2 (Vcc + 2 beta Vbc + beta^2 Vbb)
    A = b ** 2 - z2 * V[0, 0]; Bq = 2 * (b * c - z2 * V[0, 1]); Cq = c ** 2 - z2 * V[1, 1]
    disc = Bq ** 2 - 4 * A * Cq
    if A > 0 and disc >= 0:
        r = np.sort((-Bq + np.array([-1, 1]) * np.sqrt(disc)) / (2 * A))
        fieller = (float(r[0]), float(r[1]))
    else:
        fieller = (np.nan, np.nan)
    return {"b": b, "se_b": float(np.sqrt(V[0, 0])), "c": c, "se_c": float(np.sqrt(V[1, 1])),
            "beta_hat": beta, "se_beta_delta": se, "fieller_lo": fieller[0], "fieller_hi": fieller[1]}


# =============================================================================
# profile likelihood
# =============================================================================

def profile(v, y, den, num_grid, grid, fe=FE_MAIN, spline=True, w=None, theta_path=None,
            alpha=ALPHA, gtol=1e-5):
    """ell(beta) maximised over g, FE for each beta; warm-started along the grid.
    Returns ell array and the per-beta theta for reuse as warm starts."""
    ll = np.empty(len(grid)); thetas = []
    th = None
    for i in range(len(grid)):
        x, _ = shrunk_log(num_grid[:, i], den, alpha)
        X, names, _ = design(v, x, fe=fe, spline=spline)
        t0 = th if theta_path is None else theta_path[i]
        r = fit(X, y, w=w, theta0=t0, gtol=gtol)
        ll[i] = r["ll"]; th = r["theta"]; thetas.append(th)
    return ll, thetas


def summarise_profile(grid, ll, crit=CHI2_95_HALF):
    """
    argmax by local quadratic interpolation. CI = {beta: ell >= max - crit}:
    from the same parabola when the threshold falls within one grid step of
    the maximum (the naive chi-square interval is that narrow here), else by
    linear interpolation of the grid values.
    """
    i = int(np.argmax(ll))
    a = None
    if 0 < i < len(grid) - 1:
        x0, x1, x2 = grid[i - 1:i + 2]; y0, y1, y2 = ll[i - 1:i + 2]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
        b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / denom
        if a < 0:
            beta_hat = float(np.clip(-b / (2 * a), x0, x2))
            ll_max = a * beta_hat ** 2 + b * beta_hat + (y0 - a * x0 ** 2 - b * x0)
        else:
            beta_hat, ll_max, a = grid[i], ll[i], None
    else:
        beta_hat, ll_max = grid[i], ll[i]
    step = float(np.min(np.diff(grid)))
    if a is not None and np.sqrt(crit / -a) < step:
        half = float(np.sqrt(crit / -a))
        return {"beta_hat": float(beta_hat), "ll_max": float(ll_max), "ci_lo": beta_hat - half,
                "ci_hi": beta_hat + half, "at_edge": False, "ci_method": "parabola"}
    thr = ll[i] - crit
    above = ll >= thr
    lo = grid[0] if above[0] else np.interp(thr, [ll[np.argmax(above) - 1], ll[np.argmax(above)]],
                                            [grid[np.argmax(above) - 1], grid[np.argmax(above)]])
    j = len(grid) - 1 - np.argmax(above[::-1])
    hi = grid[-1] if above[-1] else np.interp(thr, [ll[j + 1], ll[j]], [grid[j + 1], grid[j]])
    return {"beta_hat": float(beta_hat), "ll_max": float(ll_max), "ci_lo": float(lo), "ci_hi": float(hi),
            "at_edge": bool(above[0] or above[-1]), "ci_method": "grid"}


# =============================================================================
# main
# =============================================================================

def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    print("=" * 68); print("05_calibrate.py — Stage 3, step 1"); print("=" * 68)
    v, F, beta0, ev = load()
    y = v["rating"].to_numpy() - 1
    blocks = v["block"].to_numpy()
    summary = {"beta0": beta0, "h_list": H_LIST, "h0": H0, "tau0": TAU0, "alpha": ALPHA}

    # ---------------------------------------------------------------- ladder
    print("\n[ladder] month + surveyor FE, block-clustered SEs")
    ladder_rows, prof_rows, prof_curves = [], [], {}
    fits_h0 = {}
    for h in H_LIST:
        t0 = time.time()
        f = fields_at(F, v, h)
        Xfe, nfe = fe_dummies(v, FE_MAIN)
        r_fe = fit(Xfe, y)
        X0, n0, info0 = design(v, f["raw"]); r0 = fit(X0, y)
        X0l, n0l, _ = design(v, f["raw"], spline=False); r0l = fit(X0l, y, groups=blocks)
        X2k, _, _ = design(v, f["corr_k"]); r2k = fit(X2k, y)
        X2c, _, _ = design(v, f["corr_c"]); r2c = fit(X2c, y)
        row = {"h": h, "ll_FE_only": r_fe["ll"], "ll_S0": r0["ll"], "ll_S0lin": r0l["ll"],
               "b_S0lin": r0l["theta"][0], "se_b_S0lin": r0l["se"][0],
               "ll_S2_kernel": r2k["ll"], "ll_S2_cbg": r2c["ll"],
               "dll_S2_S0_kernel": r2k["ll"] - r0["ll"], "dll_S2_S0_cbg": r2c["ll"] - r0["ll"],
               "zero_share": float(f["zero"].mean())}
        for wr in ["k", "c"]:
            X1, n1, _ = design(v, f["raw"], spline=False, extra={"wbar": f[f"wbar_{wr}"]})
            r1 = fit(X1, y, groups=blocks)
            br = beta_ratio(r1, n1)
            row.update({f"S1_{wr}_{k}": val for k, val in br.items()})
            row[f"ll_S1_{wr}"] = r1["ll"]
            X1m, n1m, _ = design(v, f["raw"], spline=False, mundlak=True, extra={"wbar": f[f"wbar_{wr}"]})
            r1m = fit(X1m, y, groups=blocks)
            brm = beta_ratio(r1m, n1m, b_name="loglam_between")
            row[f"S1M_{wr}_beta_hat"] = brm["beta_hat"]; row[f"S1M_{wr}_se_beta"] = brm["se_beta_delta"]
            row[f"S1M_{wr}_b_within"] = r1m["theta"][n1m.index("loglam_within")]
            row[f"S1M_{wr}_b_between"] = r1m["theta"][n1m.index("loglam_between")]
            X1s, n1s, _ = design(v, f["raw"], spline=True, extra={"wbar": f[f"wbar_{wr}"]})
            r1s = fit(X1s, y, groups=blocks)
            row[f"S1spline_{wr}_c"] = r1s["theta"][n1s.index("wbar")]
            row[f"S1spline_{wr}_se_c"] = r1s["se"][n1s.index("wbar")]
            row[f"ll_S1spline_{wr}"] = r1s["ll"]
            if h == H0 and wr == "k":
                fits_h0.update({"fe": r_fe, "S0": (r0, X0, info0), "S0lin": (r0l, n0l), "S1": (r1, n1),
                                "S2": (r2k, X2k), "fields": f})
        ladder_rows.append(row)
        print(f"  h={h}: ll FE-only {row['ll_FE_only']:.1f}  S0 {row['ll_S0']:.1f}  S2(kernel) {row['ll_S2_kernel']:.1f} "
              f"(Δ {row['dll_S2_S0_kernel']:+.1f})  S2(cbg) Δ {row['dll_S2_S0_cbg']:+.1f}   "
              f"S1' kernel: b {row['S1_k_b']:.3f} c {row['S1_k_c']:+.3f} → β̂ {row['S1_k_beta_hat']:.3f} "
              f"(δ-SE {row['S1_k_se_beta_delta']:.3f}, Fieller [{row['S1_k_fieller_lo']:.2f}, {row['S1_k_fieller_hi']:.2f}])   "
              f"S1' cbg β̂ {row['S1_c_beta_hat']:.3f}   Mundlak β̂ (kernel) {row['S1M_k_beta_hat']:.3f}   "
              f"({time.time() - t0:.0f} s)")

        # ------------------------------------------------------------ profile
        num_k, num_c = profile_numerators(v, ev, h, BETA_GRID)
        for wr, num in [("k", num_k), ("c", num_c)]:
            for spline in [True, False]:
                if not spline and wr == "c":
                    continue
                t0 = time.time()
                ll, thetas = profile(v, y, f["den"], num, BETA_GRID, spline=spline)
                s = summarise_profile(BETA_GRID, ll)
                label = f"h{h}_{wr}_{'spline' if spline else 'linear'}"
                prof_curves[label] = ll
                prof_rows.append({"h": h, "w_R": wr, "g": "spline" if spline else "linear", "fe": "+".join(FE_MAIN),
                                  "alpha": ALPHA, **s})
                print(f"    profile {label:>18}: β̂ {s['beta_hat']:.3f}  95% [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
                      f"{'  (edge)' if s['at_edge'] else ''}   ll max {s['ll_max']:.1f} vs S0 {row['ll_S0']:.1f}   ({time.time() - t0:.0f} s)")
                if h == H0 and wr == "k" and spline:
                    fits_h0["profile_thetas"] = thetas; fits_h0["profile_ll"] = ll
                    fits_h0["num_k"] = num_k
        # variants at the default bandwidth: month-only FE, and a smaller alpha
        if h == H0:
            ll_m, _ = profile(v, y, f["den"], num_k, BETA_GRID, fe=("month",), spline=True)
            s = summarise_profile(BETA_GRID, ll_m)
            prof_rows.append({"h": h, "w_R": "k", "g": "spline", "fe": "month", "alpha": ALPHA, **s})
            prof_curves[f"h{h}_k_spline_monthFE"] = ll_m
            print(f"    profile {'h800_k_spline, month FE only':>18}: β̂ {s['beta_hat']:.3f}  95% [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")
            ll_a, _ = profile(v, y, f["den"], num_k, BETA_GRID, spline=True, alpha=ALPHA_ALT)
            s = summarise_profile(BETA_GRID, ll_a)
            prof_rows.append({"h": h, "w_R": "k", "g": "spline", "fe": "+".join(FE_MAIN), "alpha": ALPHA_ALT, **s})
            prof_curves[f"h{h}_k_spline_alpha{ALPHA_ALT}"] = ll_a
            print(f"    profile {'h800_k_spline, alpha=' + str(ALPHA_ALT):>18}: β̂ {s['beta_hat']:.3f}  95% [{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]")

    ladder = pd.DataFrame(ladder_rows); ladder.to_csv(os.path.join(CAL, "ladder.csv"), index=False)
    prof = pd.DataFrame(prof_rows); prof.to_csv(os.path.join(CAL, "profile_summary.csv"), index=False)
    pd.DataFrame({"beta": BETA_GRID, **prof_curves}).to_csv(os.path.join(CAL, "profile_curves.csv"), index=False)
    summary["ladder"] = ladder.round(4).to_dict(orient="records")
    summary["profile"] = prof.round(4).to_dict(orient="records")

    # ------------------------------------------------------------- bootstrap
    f = fits_h0["fields"]; den = f["den"]; num_k = fits_h0["num_k"]
    full_prof = summarise_profile(BETA_GRID, fits_h0["profile_ll"])
    boot_path = os.path.join(CAL, "bootstrap.csv")
    boot = pd.read_csv(boot_path) if (REUSE_BOOTSTRAP and os.path.exists(boot_path)) else None
    if boot is not None and len(boot) == N_BOOT:
        print(f"\n[bootstrap] reusing {N_BOOT} saved spatial block resamples from {os.path.relpath(boot_path, ROOT)}")
    else:
        print(f"\n[bootstrap] {N_BOOT} spatial block resamples at h={H0} (kernel w_R): S1' and the profile")
        t0 = time.time()
        X1, n1, _ = design(v, f["raw"], spline=False, extra={"wbar": f["wbar_k"]})
        th1 = fits_h0["S1"][0]["theta"]
        boot_idx = [i for i, b in enumerate(BETA_GRID) if any(np.isclose(b, BOOT_GRID))]
        theta_path = [fits_h0["profile_thetas"][i] for i in boot_idx]
        # corrected numerator at the full-sample beta-hat, for an exact LR in each resample
        o = accumulate_st(v[["x", "y"]].to_numpy(), v["t_days"].to_numpy(), ev["xy"], ev["t"], H0, [TAU0],
                          {"m": np.exp(-full_prof["beta_hat"] * ev["wk"])}, src_tree=ev["tree"])
        x_full, _ = shrunk_log(o[(TAU0, "m")], den, ALPHA)
        X_full, _, _ = design(v, x_full)
        th_full = fit(X_full, y)["theta"]
        nb = blocks.max() + 1
        brows = []
        for r in range(N_BOOT):
            draw = rng.integers(0, nb, nb)
            w = np.bincount(draw, minlength=nb)[blocks].astype(float)
            r1 = fit(X1, y, w=w, theta0=th1, gtol=BOOT_GTOL)
            ib, ic = n1.index("loglam"), n1.index("wbar")
            beta_s1 = -r1["theta"][ic] / r1["theta"][ib]
            ll_b, _ = profile(v, y, den, num_k[:, boot_idx], BOOT_GRID, w=w, theta_path=theta_path, gtol=BOOT_GTOL)
            s = summarise_profile(BOOT_GRID, ll_b)
            ll_at_full = fit(X_full, y, w=w, theta0=th_full, gtol=BOOT_GTOL)["ll"]
            brows.append({"rep": r, "beta_S1": beta_s1, "beta_prof": s["beta_hat"],
                          "lr_at_full": max(2 * (s["ll_max"] - ll_at_full), 0.0)})
            if (r + 1) % 20 == 0:
                print(f"    {r + 1}/{N_BOOT} done ({time.time() - t0:.0f} s)")
        boot = pd.DataFrame(brows); boot.to_csv(boot_path, index=False)
    crit = float(np.quantile(boot["lr_at_full"], 0.95))
    recal = summarise_profile(BETA_GRID, fits_h0["profile_ll"], crit=crit / 2)
    summary["bootstrap"] = {
        "n": N_BOOT,
        "S1_beta_sd": float(boot["beta_S1"].std()), "S1_beta_q": [float(q) for q in boot["beta_S1"].quantile([.025, .5, .975])],
        "prof_beta_sd": float(boot["beta_prof"].std()), "prof_beta_q": [float(q) for q in boot["beta_prof"].quantile([.025, .5, .975])],
        "lr_crit_95_bootstrap": crit, "lr_crit_95_chi2": 3.84,
        "profile_ci_chi2": [full_prof["ci_lo"], full_prof["ci_hi"]],
        "profile_ci_recalibrated": [recal["ci_lo"], recal["ci_hi"]]}
    print(f"  S1' β̂ bootstrap sd {summary['bootstrap']['S1_beta_sd']:.3f}, 95% pct "
          f"[{summary['bootstrap']['S1_beta_q'][0]:.3f}, {summary['bootstrap']['S1_beta_q'][2]:.3f}]")
    print(f"  profile β̂ bootstrap sd {summary['bootstrap']['prof_beta_sd']:.3f}, 95% pct "
          f"[{summary['bootstrap']['prof_beta_q'][0]:.3f}, {summary['bootstrap']['prof_beta_q'][2]:.3f}]")
    print(f"  LR critical value: χ² 3.84 vs bootstrap {crit:.2f}; profile CI χ² "
          f"[{full_prof['ci_lo']:.3f}, {full_prof['ci_hi']:.3f}] → recalibrated [{recal['ci_lo']:.3f}, {recal['ci_hi']:.3f}]")

    # ------------------------------------------------------- negative controls
    print("\n[controls]")
    ctrl = {}
    # (a) outage placebo field at h=500 (only bandwidth it was built at), vs dumping at 500
    f5 = fields_at(F, v, 500)
    den5 = F["den_expo_h500"].to_numpy()
    lo_out, _ = shrunk_log(F["num_outage_h500_t120"].to_numpy(), den5, ALPHA)
    Xo, no_, _ = design(v, lo_out, spline=False); ro = fit(Xo, y, groups=blocks)
    Xd, nd_, _ = design(v, f5["raw"], spline=False); rd = fit(Xd, y, groups=blocks)
    r_fe = fits_h0["fe"]
    ctrl["outage_placebo"] = {"b_outage": ro["theta"][0], "se": ro["se"][0], "dll_vs_FE": ro["ll"] - r_fe["ll"],
                              "b_dumping_h500": rd["theta"][0], "se_dumping": rd["se"][0], "dll_dumping_vs_FE": rd["ll"] - r_fe["ll"],
                              "spearman_outage": spearmanr(y, lo_out).correlation}
    print(f"  outage placebo (h=500): b {ro['theta'][0]:+.4f} (se {ro['se'][0]:.4f}), Δll vs FE {ro['ll'] - r_fe['ll']:+.1f}   "
          f"| dumping: b {rd['theta'][0]:+.4f} (se {rd['se'][0]:.4f}), Δll {rd['ll'] - r_fe['ll']:+.1f}")

    # (b) torus-shifted w_R raster: null for the S2 - S0 improvement at H0
    t0 = time.time()
    z = np.load(os.path.join(OUT, "wR_raster.npz"), allow_pickle=True)
    gx, gy, Z, res = z["x"], z["y"], z["wR"], float(z["res"])
    Zc = np.where(np.isfinite(Z), Z, 0.0)
    ix = np.clip(np.round((ev["xy"][:, 0] - gx[0]) / res).astype(int), 0, len(gx) - 1)
    iy = np.clip(np.round((ev["xy"][:, 1] - gy[0]) / res).astype(int), 0, len(gy) - 1)
    weights = {}
    for p in range(N_PERM):
        sx, sy = rng.integers(0, len(gx)), rng.integers(0, len(gy))
        wshift = Zc[(iy + sy) % len(gy), (ix + sx) % len(gx)]
        weights[f"p{p}"] = np.exp(-beta0 * wshift)
    o = accumulate_st(v[["x", "y"]].to_numpy(), v["t_days"].to_numpy(), ev["xy"], ev["t"], H0, [TAU0], weights,
                      src_tree=ev["tree"])
    r0 = fits_h0["S0"][0]
    dll_null = []
    for p in range(N_PERM):
        x, _ = shrunk_log(o[(TAU0, f"p{p}")], den, ALPHA)
        Xp, _, _ = design(v, x); rp = fit(Xp, y, theta0=r0["theta"])
        dll_null.append(rp["ll"] - r0["ll"])
    dll_null = np.array(dll_null)
    dll_actual = fits_h0["S2"][0]["ll"] - r0["ll"]
    ctrl["permutation_null"] = {"n_perm": N_PERM, "dll_actual_S2_S0": dll_actual,
                                "dll_null_mean": float(dll_null.mean()), "dll_null_sd": float(dll_null.std()),
                                "dll_null_q95": float(np.quantile(dll_null, .95)), "dll_null_max": float(dll_null.max()),
                                "p_value": float((dll_null >= dll_actual).mean())}
    pd.DataFrame({"dll_null": dll_null}).to_csv(os.path.join(CAL, "permutation_null.csv"), index=False)
    print(f"  torus-shift null ({N_PERM} shifts, {time.time() - t0:.0f} s): actual Δll(S2−S0) {dll_actual:+.2f}; "
          f"null mean {dll_null.mean():+.2f} sd {dll_null.std():.2f} q95 {np.quantile(dll_null, .95):+.2f} max {dll_null.max():+.2f}; "
          f"p = {ctrl['permutation_null']['p_value']:.2f}")

    # (c) dedup at h=500: S1' on piles vs all reports
    lo_dd, _ = shrunk_log(F["num_dedup_h500_t120"].to_numpy(), den5, ALPHA)
    dd = {}
    for label, x in [("all reports", f5["raw"]), ("piles (50 m / 14 d)", lo_dd)]:
        X1, n1_, _ = design(v, x, spline=False, extra={"wbar": f5["wbar_k"]}); r1 = fit(X1, y, groups=blocks)
        dd[label] = beta_ratio(r1, n1_)
        print(f"  dedup h=500 {label:>20}: b {dd[label]['b']:.3f}  c {dd[label]['c']:+.3f} (se {dd[label]['se_c']:.3f})  β̂ {dd[label]['beta_hat']:.3f}")
    ctrl["dedup"] = dd

    # (d) differential outcome: bulk vs casual, raw vs corrected (binary logit, block-clustered)
    diff = {}
    Xfe, _ = fe_dummies(v, FE_MAIN)
    for outcome in ["bulk", "casual"]:
        yo = v[outcome].to_numpy()
        res_o = {}
        for label, x in [("raw", f["raw"]), ("corr_k", f["corr_k"])]:
            Xb = sm.add_constant(np.c_[x - x.mean(), Xfe])
            m = sm.Logit(yo, Xb).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": blocks})
            res_o[label] = {"b": float(m.params[1]), "se": float(m.bse[1]), "ll": float(m.llf)}
        m0 = sm.Logit(yo, sm.add_constant(Xfe)).fit(disp=0, maxiter=200)
        res_o["ll_FE_only"] = float(m0.llf)
        res_o["dll_raw"] = res_o["raw"]["ll"] - res_o["ll_FE_only"]
        res_o["dll_corr_minus_raw"] = res_o["corr_k"]["ll"] - res_o["raw"]["ll"]
        diff[outcome] = res_o
        print(f"  {outcome:>7}: b_raw {res_o['raw']['b']:+.3f} (se {res_o['raw']['se']:.3f})  Δll raw vs FE {res_o['dll_raw']:+.1f}  "
              f"Δll corr − raw {res_o['dll_corr_minus_raw']:+.2f}")
    ctrl["differential_outcome"] = diff

    # (e) partial proportional odds: -c_k / b_k per cutpoint
    ppo = []
    for k in [2, 3, 4]:
        yk = (v["rating"].to_numpy() >= k).astype(int)
        Xb = sm.add_constant(np.c_[f["raw"] - f["raw"].mean(), f["wbar_k"] - f["wbar_k"].mean(), Xfe])
        m = sm.Logit(yk, Xb).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": blocks})
        b, c = m.params[1], m.params[2]
        ppo.append({"cut": f">={k}", "n_pos": int(yk.sum()), "b_k": b, "se_b": m.bse[1], "c_k": c, "se_c": m.bse[2], "beta_k": -c / b})
        print(f"  partial-PO rating>={k} (n={yk.sum():,}): b {b:.3f} (se {m.bse[1]:.3f})  c {c:+.3f} (se {m.bse[2]:.3f})  −c/b {-c / b:.3f}")
    ctrl["partial_po"] = ppo

    # (f) the literal 1 + 3 sigmoid(eta) by NLS, initialised from S0lin
    r0l, n0l = fits_h0["S0lin"]
    X0l, _, _ = design(v, f["raw"], spline=False)
    Xa = np.c_[np.ones(len(y)), X0l]
    th_init = np.r_[0.0, r0l["theta"][:X0l.shape[1]]]
    def resid(t): return 1.0 + 3.0 * expit(Xa @ t) - (y + 1)
    nls = least_squares(resid, th_init, method="lm", max_nfev=2000)
    yhat_nls = 1.0 + 3.0 * expit(Xa @ nls.x)
    yhat_ord = OL.expected_rating(r0l["theta"], X0l)
    ctrl["nls_sigmoid"] = {"rmse_nls": float(np.sqrt(np.mean((yhat_nls - y - 1) ** 2))),
                           "rmse_ordinal_Erating": float(np.sqrt(np.mean((yhat_ord - y - 1) ** 2))),
                           "nls_range_p1_p99": [float(np.quantile(yhat_nls, .01)), float(np.quantile(yhat_nls, .99))],
                           "nls_min_max": [float(yhat_nls.min()), float(yhat_nls.max())],
                           "share_of_1_4_range_spanned_p1_p99": float((np.quantile(yhat_nls, .99) - np.quantile(yhat_nls, .01)) / 3),
                           "cutpoints_S0lin": [float(c) for c in OL.unpack(r0l["theta"], X0l.shape[1])[1]]}
    print(f"  NLS 1+3σ(η): RMSE {ctrl['nls_sigmoid']['rmse_nls']:.4f} vs ordinal E[rating] {ctrl['nls_sigmoid']['rmse_ordinal_Erating']:.4f}; "
          f"predictions p1–p99 [{ctrl['nls_sigmoid']['nls_range_p1_p99'][0]:.2f}, {ctrl['nls_sigmoid']['nls_range_p1_p99'][1]:.2f}] "
          f"= {ctrl['nls_sigmoid']['share_of_1_4_range_spanned_p1_p99']:.0%} of the 1–4 range; "
          f"ordinal cutpoints {np.round(ctrl['nls_sigmoid']['cutpoints_S0lin'], 2)} (equal spacing would be the PI's assumption)")

    # (g) alpha sensitivity at H0 (S0 spline ll and S1' beta)
    al = []
    for a in ALPHA_LIST:
        fa = fields_at(F, v, H0, alpha=a)
        Xs, _, _ = design(v, fa["raw"]); rs = fit(Xs, y)
        X1, n1_, _ = design(v, fa["raw"], spline=False, extra={"wbar": fa["wbar_k"]}); r1 = fit(X1, y, groups=blocks)
        br = beta_ratio(r1, n1_)
        al.append({"alpha": a, "ll_S0": rs["ll"], "b": br["b"], "c": br["c"], "beta_hat": br["beta_hat"]})
    ctrl["alpha_sensitivity"] = al
    print("  alpha sensitivity (h=800): " + "  ".join(f"α={r['alpha']}: ll_S0 {r['ll_S0']:.1f}, β̂ {r['beta_hat']:.3f}" for r in al))

    # (h) level reconciliation
    r0, X0, info0 = fits_h0["S0"]
    Er0 = OL.expected_rating(r0["theta"], X0)
    r2, X2 = fits_h0["S2"]
    Er2 = OL.expected_rating(r2["theta"], X2)
    # field-only versions: g(log lam) at the mean FE effect, comparable to the old CBG-level plan
    def field_only(r, X):
        p = X.shape[1]; beta, cuts = OL.unpack(r["theta"], p); ng = X.shape[1] - Xfe.shape[1]
        eta = X[:, :ng] @ beta[:ng] + (X[:, ng:] @ beta[ng:]).mean()
        return 1.0 + expit(eta[:, None] - cuts[None, :]).sum(1)
    g = pd.DataFrame({"g": v["GEOID"], "y": y + 1, "e0": Er0, "e2": Er2,
                      "f0": field_only(r0, X0), "f2": field_only(r2, X2)}).groupby("g").mean()
    ctrl["level_reconciliation"] = {
        "spearman_S0": spearmanr(g["y"], g["e0"]).correlation, "pearson_S0": g["y"].corr(g["e0"]),
        "spearman_S2": spearmanr(g["y"], g["e2"]).correlation, "pearson_S2": g["y"].corr(g["e2"]),
        "spearman_S0_field_only": spearmanr(g["y"], g["f0"]).correlation,
        "spearman_S2_field_only": spearmanr(g["y"], g["f2"]).correlation, "n_cbg": int(len(g))}
    lr = ctrl["level_reconciliation"]
    print(f"  CBG-level reconciliation, Spearman of CBG-mean fitted E[rating] with CBG-mean rating: "
          f"with FE S0 {lr['spearman_S0']:.3f} / S2 {lr['spearman_S2']:.3f}; field only S0 {lr['spearman_S0_field_only']:.3f} "
          f"/ S2 {lr['spearman_S2_field_only']:.3f} (old CBG plan: +0.58 raw / +0.64 corrected)")
    summary["controls"] = ctrl

    # ---------------------------------------------------------------- figures
    _figures(v, y, f, fits_h0, prof_curves, boot, dll_null, dll_actual, beta0)

    with open(os.path.join(CAL, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Outputs in output/calib/")


def _style(ax, title, xlabel, ylabel):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9); ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def _figures(v, y, f, fits_h0, prof_curves, boot, dll_null, dll_actual, beta0):
    # profile curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, col, ls in [(f"h{H0}_k_spline", C1, "-"), (f"h{H0}_c_spline", C2, "-"), (f"h{H0}_k_linear", C1, "--"),
                           (f"h{H0}_k_spline_monthFE", C3, "-"), (f"h{H0}_k_spline_alpha{ALPHA_ALT}", C3, "--"),
                           ("h500_k_spline", MUTED, ":"), ("h1200_k_spline", MUTED, "-.")]:
        if label in prof_curves:
            ll = prof_curves[label]; ax.plot(BETA_GRID, ll - ll.max(), color=col, ls=ls, lw=1.8, label=label)
    ax.axhline(-CHI2_95_HALF, color=GRID, lw=1); ax.axvline(beta0, color=MUTED, lw=1, ls="--")
    ax.text(beta0, ax.get_ylim()[0] * 0.95, " β₀ (05a)", color=MUTED, fontsize=8)
    ax.set_ylim(-15, 1)
    _style(ax, "Profile log-likelihood for β (relative to its maximum)", "β", "ℓ(β) − max ℓ")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "05_profile.png"), dpi=160); plt.close(fig)

    # linearisation: g on raw vs corrected fields, as E[rating] at the mean FE
    r0, X0, info0 = fits_h0["S0"]; r2, X2 = fits_h0["S2"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, x, r, X, col in [("raw", f["raw"], r0, X0, C1), (f"corrected (β₀={beta0:.2f})", f["corr_k"], r2, X2, C2)]:
        grid = np.linspace(np.quantile(x, .005), np.quantile(x, .995), 200)
        B, info = spline_basis(x); Bg, _ = spline_basis(grid, info)
        p = X.shape[1]; beta, cuts = OL.unpack(r["theta"], p)
        ng = Bg.shape[1]
        eta_fe = (X[:, ng:] @ beta[ng:]).mean()
        eta = Bg @ beta[:ng] + eta_fe
        Er = 1.0 + expit(eta[:, None] - cuts[None, :]).sum(1)
        ax.plot(grid, Er, color=col, lw=2, label=label)
        dec = pd.qcut(x, 10, labels=False, duplicates="drop"); dm = pd.Series(y + 1.0).groupby(dec).mean()
        qs = pd.Series(x).groupby(dec).median()
        ax.scatter(qs, dm, color=col, s=14, zorder=3)
    _style(ax, "E[rating | log λ̂] at the mean month/surveyor effect; dots = decile means", "log λ̂ (ratio form)", "rating")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "05_linearisation.png"), dpi=160); plt.close(fig)

    # bootstrap and permutation
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    ax.hist(boot["beta_prof"], bins=20, color=C1, alpha=0.7, label="profile β̂")
    ax.hist(boot["beta_S1"], bins=20, color=C2, alpha=0.5, label="S1' −c/b")
    ax.axvline(beta0, color=MUTED, ls="--", lw=1)
    _style(ax, f"Spatial block bootstrap of β̂ ({len(boot)} resamples of 40 blocks)", "β̂", "count"); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    ax.hist(dll_null, bins=20, color=GRID, edgecolor=MUTED, label="torus-shifted ŵ_R")
    ax.axvline(dll_actual, color=C2, lw=2, label="actual S2 − S0")
    _style(ax, "Δ log-lik from the correction vs the permutation null", "Δℓ (S2 − S0)", "count"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "05_bootstrap_permutation.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
