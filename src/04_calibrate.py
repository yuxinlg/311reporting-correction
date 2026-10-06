"""
04_calibrate.py
===============
Stage 3: calibrate the field to the 1-4 rating and estimate β. Statistics only;
every kernel sum comes from 03.

Model — proportional-odds logit, three sigmoids sharing one index:
    P(rating ≥ k) = σ( g(log λ̂_β) + month-of-sample + surveyor − κ_k ),  g a 5-df natural spline.

Profile likelihood — for each β on BETA_GRID, log λ̂_β = Gamma-shrunk log of
num[β] / den; refit g, the fixed effects and the cutpoints; ℓ(β) is the maximised
log-likelihood. β̂ is the maximum (parabola through the best three grid points,
then one exact kernel pass at β̂). ℓ(0) is the raw-field baseline.

Table 1 — rows for h in H_LIST, plus sensitivities at H0: month-only FE,
α = ALPHA_ALT, h_w = HW_ALT. Each row: ℓ_FE-only, ℓ(0), ℓ(β̂), β̂, −c/b.

CI — spatial block bootstrap at H0: N_BOOT resamples of the N_BLOCKS blocks
(frequency weights); each resample refits the profile on the BETA_GRID points
within ±BOOT_HALF steps of β̂ (extended if its maximum sits on an edge) and the
linear −c/b. Percentile 95% intervals.

Checks at H0
  −c/b     one linear fit  b·log λ̂_0 + c·w̄ + FE  →  β = −c/b, the first-order
           version of the estimator; must land near β̂
  torus    N_PERM shifted copies of the w_R map, each allowed its best β over
           {0} ∪ NULL_BETAS: Δ_p = max_β ℓ_p(β) − ℓ(0), vs the real map on the
           same coarse grid; p = share of Δ_p ≥ Δ_real
  placebo  the field built from street-light-outage reports instead of dumping
           reports: linear slope (block-clustered SE) and Δℓ vs FE-only, next
           to the dumping field's

Writes  output/calib/{table1.csv, profile_curves.csv, bootstrap.csv, torus_null.csv, summary.json}
        output/figures/fig1_profile.png, fig2_linearisation.png
"""

import os
import json
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import expit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (OUT, FIG, FIELDS, CALIB, TAU, ALPHA, ALPHA_ALT, BETA_GRID, NULL_BETAS, N_PERM, N_BOOT,
                    BOOT_MAX_MOVES, BOOT_GTOL, GTOL, N_JOBS, SEED, C1, C2, C3, MUTED, GRID, style, load_choices,
                    update_choices, ensure_dirs)
from kernels import accumulate_st, shrunk_log
from ordinal import OrderedLogit
from design import FE, design, spline_basis

warnings.filterwarnings("ignore")
OL = OrderedLogit(4)
I0 = int(np.argmin(np.abs(BETA_GRID)))            # index of β = 0 on the grid


# =============================================================================
# pieces
# =============================================================================

def field_log(num, den, lam_bar, alpha):
    return shrunk_log(num, den, alpha, lam_bar)[0]


def fit(X, y, pen, theta0=None, w=None, gtol=GTOL):
    th, ll, _ = OL.fit(X, y, w, theta0, penalty=pen, gtol=gtol)
    return th, ll


def profile(F, y, fe, idx, alpha=ALPHA, which="num", w=None, theta_path=None, gtol=GTOL):
    """ℓ(β) for the grid indices `idx`, warm-started; returns ll dict and theta dict."""
    num, mean_m = F[which], F["mean_m" + which[3:]]
    lls, ths = {}, {}
    th = None
    for i in idx:
        x = field_log(num[:, i], F["den"], F["lam_bar_raw"] * mean_m[i], alpha)
        X, pen, _, _ = design(x, fe)
        t0 = theta_path[i] if theta_path is not None else th
        th, ll = fit(X, y, pen, t0, w, gtol)
        lls[i], ths[i] = ll, th
    return lls, ths


def argmax_parabola(grid, ll):
    """Maximum of ℓ over the grid, refined by a parabola through the best three points."""
    i = int(np.argmax(ll))
    if 0 < i < len(grid) - 1:
        x0, x1, x2 = grid[i - 1:i + 2]; y0, y1, y2 = ll[i - 1:i + 2]
        den = (x0 - x1) * (x0 - x2) * (x1 - x2)
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
        b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / den
        if a < 0:
            bh = float(np.clip(-b / (2 * a), x0, x2))
            return bh, float(a * bh ** 2 + b * bh + (y0 - a * x0 ** 2 - b * x0)), False
    return float(grid[i]), float(ll[i]), True


def ratio_fit(x, wbar, y, fe, w=None, theta0=None, gtol=GTOL):
    """Linear S1: b·x + c·w̄ + FE; returns (theta, ll, −c/b, X, pen)."""
    X, pen, _, _ = design(x, fe, spline=False, extra={"wbar": wbar})
    th, ll = fit(X, y, pen, theta0, w, gtol)
    return th, ll, float(-th[1] / th[0]), X, pen


# =============================================================================
# worker processes for the bootstrap and the torus null
# (spawned; each loads its inputs from disk, so nothing large is pickled)
# =============================================================================

def _worker_state():
    ch = load_choices(); H0, HW0 = ch["H0"], ch["HW0"]
    v = pd.read_parquet(os.path.join(OUT, "visits.parquet"))
    y = v["rating"].to_numpy() - 1
    F = dict(np.load(os.path.join(FIELDS, f"h{H0}.npz"))); F["lam_bar_raw"] = float(F["lam_bar_raw"])
    fe = FE(v).build(v)
    z = np.load(os.path.join(CALIB, f"profile_h{H0}_month+surv_a{ALPHA:g}_hw{HW0}.npz"))
    ths = {i: z["thetas"][i] for i in range(len(BETA_GRID))}
    c = dict(np.load(os.path.join(CALIB, "boot_context.npz")))
    return v, y, F, fe, ths, c


def _boot_chunk(reps):
    """Resamples `reps`: walk the β grid uphill from β̂ until the maximum is bracketed, then a parabola."""
    v, y, F, fe, ths, c = _worker_state()
    blocks = v["block"].to_numpy(); nb = int(blocks.max()) + 1
    i_hat = int(c["i_hat"])
    rows = []
    for r in reps:
        rr = np.random.default_rng(SEED + 1000 + int(r))
        w = np.bincount(rr.integers(0, nb, nb), minlength=nb)[blocks].astype(float)
        cen, lls, moves = i_hat, {}, 0
        while True:
            ks = [i for i in (cen - 1, cen, cen + 1) if 0 <= i < len(BETA_GRID)]
            lls.update(profile(F, y, fe, [i for i in ks if i not in lls], w=w, theta_path=ths, gtol=BOOT_GTOL)[0])
            arr = np.array([lls[k] for k in ks]); j = int(np.argmax(arr))
            on_edge = (j == 0 and ks[0] > 0) or (j == len(ks) - 1 and ks[-1] < len(BETA_GRID) - 1)
            if on_edge and moves < BOOT_MAX_MOVES:
                cen = ks[j]; moves += 1; continue
            break
        bh, _, edge = argmax_parabola(BETA_GRID[ks], arr)
        _, _, ratio, _, _ = ratio_fit(c["x0"], F["wbar"], y, fe, w, c["th1"], BOOT_GTOL)
        rows.append({"rep": int(r), "beta_prof": bh, "beta_ratio": ratio, "moves": moves, "n_fits": len(lls), "at_edge": edge})
    return rows


def _null_chunk(shifts):
    """Torus-shifted maps `shifts`: best Δℓ over β in {0} ∪ NULL_BETAS for each."""
    v, y, F, fe, ths, c = _worker_state()
    ll0 = float(c["ll0"])
    coarse = [int(np.argmin(np.abs(BETA_GRID - b))) for b in NULL_BETAS]
    out = []
    for p in shifts:
        best = ll0
        for k, b in enumerate(NULL_BETAS):
            x = field_log(F["num_null"][:, p, k], F["den"], F["lam_bar_raw"] * F["mean_m_null"][p, k], ALPHA)
            X, pen, _, _ = design(x, fe)
            _, ll = fit(X, y, pen, ths[coarse[k]], gtol=BOOT_GTOL)
            best = max(best, ll)
        out.append({"shift": int(p), "dll_best": best - ll0})
    return out


def _parallel(func, chunks, label, t0):
    results = []
    with ProcessPoolExecutor(max_workers=N_JOBS, mp_context=mp.get_context("spawn")) as ex:
        futures = [ex.submit(func, ch) for ch in chunks]
        for n_done, fut in enumerate(as_completed(futures), 1):
            results += fut.result()
            yield n_done, len(chunks), results


# =============================================================================
# main
# =============================================================================

def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    print("=" * 68); print("04_calibrate.py — Stage 3: profile likelihood for β"); print("=" * 68)
    ensure_dirs()
    ch = load_choices(); H0, H_LIST, HW0, HW_ALT = ch["H0"], ch["H_LIST"], ch["HW0"], ch["HW_ALT"]
    v = pd.read_parquet(os.path.join(OUT, "visits.parquet"))
    y = v["rating"].to_numpy() - 1
    blocks = v["block"].to_numpy(); nb = int(blocks.max()) + 1
    F = {h: dict(np.load(os.path.join(FIELDS, f"h{h}.npz"))) for h in H_LIST}
    for h in H_LIST:
        F[h]["lam_bar_raw"] = float(F[h]["lam_bar_raw"])
    fe_main = FE(v).build(v)
    fe_month = FE(v, surveyor=False).build(v)
    th_fe, ll_fe = fit(fe_main[0], y, fe_main[1])
    print(f"  {len(v):,} visits, H0 = {H0}, H_LIST = {H_LIST}, h_w = {HW0} (alt {HW_ALT}); ℓ(FE only) = {ll_fe:.1f}")

    # ------------------------------------------------------------- Table 1
    print("\n[profile] ℓ(β) over the grid; β̂ by parabola; −c/b from one linear fit")
    rows, curves, keep = [], {}, {}
    specs = [(h, "month+surv", ALPHA, HW0, "num") for h in H_LIST] + \
            [(H0, "month", ALPHA, HW0, "num"), (H0, "month+surv", ALPHA_ALT, HW0, "num"),
             (H0, "month+surv", ALPHA, HW_ALT, "num_alt")]
    for h, fe_name, alpha, hw, which in specs:
        t0 = time.time()
        fe = fe_main if fe_name == "month+surv" else fe_month
        label = f"h{h}_{fe_name}_a{alpha:g}_hw{hw}"
        cache = os.path.join(CALIB, f"profile_{label}.npz")          # full-sample fits per grid point, reused on rerun
        if os.path.exists(cache) and np.allclose(np.load(cache)["grid"], BETA_GRID):
            z = np.load(cache); ll = z["ll"]; ths = {i: z["thetas"][i] for i in range(len(BETA_GRID))}
        else:
            lls, ths = profile(F[h], y, fe, range(len(BETA_GRID)), alpha, which)
            ll = np.array([lls[i] for i in range(len(BETA_GRID))])
            np.savez(cache, grid=BETA_GRID, ll=ll, thetas=np.stack([ths[i] for i in range(len(BETA_GRID))]))
        bh, llm, edge = argmax_parabola(BETA_GRID, ll)
        x0 = field_log(F[h][which][:, I0], F[h]["den"], F[h]["lam_bar_raw"], alpha)
        th1, ll1, ratio, X1, pen1 = ratio_fit(x0, F[h]["wbar"], y, fe)
        curves[label] = ll
        rows.append({"h": h, "fe": fe_name, "alpha": alpha, "h_w": hw, "ll_FE": ll_fe, "ll_0": ll[I0],
                     "ll_hat": llm, "dll_hat": llm - ll[I0], "beta_hat": bh, "at_edge": edge,
                     "beta_ratio": ratio, "b": th1[0], "c": th1[1]})
        if h == H0 and fe_name == "month+surv" and alpha == ALPHA and hw == HW0:
            keep.update({"ll": ll, "ths": ths, "th1": th1, "X1": X1, "pen1": pen1, "x0": x0, "beta_hat": bh})
        print(f"  h={h:>4} FE={fe_name:>10} α={alpha:<4g} h_w={hw}: ℓ(0) {ll[I0]:.1f}  ℓ(β̂) {llm:.1f}  Δ {llm - ll[I0]:+.1f}  "
              f"β̂ {bh:.3f}{' (edge)' if edge else ''}   −c/b {ratio:.3f} (b {th1[0]:.3f}, c {th1[1]:+.3f})   ({time.time() - t0:.0f} s)")
    beta_hat = keep["beta_hat"]
    pd.DataFrame(rows).to_csv(os.path.join(CALIB, "table1.csv"), index=False)              # partial; rewritten at the end
    pd.DataFrame({"beta": BETA_GRID, **curves}).to_csv(os.path.join(CALIB, "profile_curves.csv"), index=False)

    # exact refit at β̂ (one kernel pass) — the model the figure and the surface use
    ev = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    wr = pd.read_parquet(os.path.join(OUT, "wR_events.parquet"))[f"wR_h{HW0}"].to_numpy()
    wr = np.where(np.isfinite(wr), wr, 0.0)
    ev_xy, ev_t = ev[["x", "y"]].to_numpy(), ev["t_days"].to_numpy()
    o = accumulate_st(v[["x", "y"]].to_numpy(), v["t_days"].to_numpy(), ev_xy, ev_t, H0, [TAU],
                      {"m": np.exp(-beta_hat * wr)}, src_tree=cKDTree(ev_xy))
    x_hat = field_log(o[(TAU, "m")], F[H0]["den"], F[H0]["lam_bar_raw"] * np.exp(-beta_hat * wr).mean(), ALPHA)
    X_hat, pen_hat, _, info_hat = design(x_hat, fe_main)
    th_hat, ll_hat = fit(X_hat, y, pen_hat, keep["ths"][int(np.argmin(np.abs(BETA_GRID - beta_hat)))])
    X_0, pen_0, _, info_0 = design(keep["x0"], fe_main)
    th_0 = keep["ths"][I0]
    print(f"  exact refit at β̂ = {beta_hat:.3f}: ℓ = {ll_hat:.1f} (parabola said {rows[H_LIST.index(H0)]['ll_hat']:.1f})")
    rows[H_LIST.index(H0)]["ll_hat_exact"] = ll_hat

    # ------------------------------------------------------------- bootstrap
    # resamples are seeded individually (SEED + 1000 + rep), run in N_JOBS worker processes in chunks of 10,
    # and appended to bootstrap.csv as chunks finish, so an interrupted run resumes and the result does not
    # depend on the interruption or on the number of workers
    i_hat = int(np.argmin(np.abs(BETA_GRID - beta_hat)))
    np.savez(os.path.join(CALIB, "boot_context.npz"), beta_hat=beta_hat, i_hat=i_hat, x0=keep["x0"],
             th1=keep["th1"], ll0=keep["ll"][I0])
    boot_path = os.path.join(CALIB, "bootstrap.csv")
    brows = pd.read_csv(boot_path).to_dict(orient="records") if os.path.exists(boot_path) else []
    done = {int(r["rep"]) for r in brows}
    chunks = [[r for r in range(a, min(a + 10, N_BOOT)) if r not in done] for a in range(0, N_BOOT, 10)]
    chunks = [c for c in chunks if c]
    if not chunks:
        print(f"\n[bootstrap] reusing {len(brows)} saved resamples ({os.path.relpath(boot_path, OUT)})")
    else:
        print(f"\n[bootstrap] {N_BOOT} spatial block resamples at h = {H0}, {N_JOBS} workers"
              + (f", {len(done)} already saved" if done else ""))
        t0 = time.time()
        for n_done, n_all, new in _parallel(_boot_chunk, chunks, "bootstrap", t0):
            pd.DataFrame(brows + new).to_csv(boot_path, index=False)
            print(f"    {len(done) + len(new)}/{N_BOOT} ({time.time() - t0:.0f} s)", flush=True)
        brows += new
    boot = pd.DataFrame(brows).sort_values("rep").reset_index(drop=True)
    ci_prof = [float(q) for q in boot["beta_prof"].quantile([.025, .975])]
    ci_ratio = [float(q) for q in boot["beta_ratio"].quantile([.025, .975])]
    rows[H_LIST.index(H0)].update({"ci_lo": ci_prof[0], "ci_hi": ci_prof[1], "boot_sd": float(boot["beta_prof"].std()),
                                   "ratio_ci_lo": ci_ratio[0], "ratio_ci_hi": ci_ratio[1]})
    print(f"  profile β̂ {beta_hat:.3f}, bootstrap sd {boot['beta_prof'].std():.3f}, 95% [{ci_prof[0]:.3f}, {ci_prof[1]:.3f}]; "
          f"−c/b {rows[H_LIST.index(H0)]['beta_ratio']:.3f}, 95% [{ci_ratio[0]:.3f}, {ci_ratio[1]:.3f}]; "
          f"{int(boot['moves'].gt(0).sum())} resamples walked away from β̂ (mean {boot['n_fits'].mean():.1f} fits each), "
          f"{int(boot['at_edge'].sum())} ended unbracketed")

    # ------------------------------------------------------------- torus null
    print(f"\n[torus] {N_PERM} shifted w_R maps, each at its best β over {{0}} ∪ {NULL_BETAS}, {N_JOBS} workers")
    t0 = time.time()
    ll0 = keep["ll"][I0]
    coarse = [int(np.argmin(np.abs(BETA_GRID - b))) for b in NULL_BETAS]
    d_real_coarse = float(max(keep["ll"][i] for i in [I0] + coarse) - ll0)
    d_real = ll_hat - ll0
    null_path = os.path.join(CALIB, "torus_null.csv")
    if os.path.exists(null_path) and len(pd.read_csv(null_path)) == N_PERM:
        tn = pd.read_csv(null_path)
    else:
        tn = []
        for _, _, tn in _parallel(_null_chunk, [list(range(a, min(a + 5, N_PERM))) for a in range(0, N_PERM, 5)], "torus", t0):
            pass
        tn = pd.DataFrame(tn).sort_values("shift"); tn.to_csv(null_path, index=False)
    dnull = tn["dll_best"].to_numpy()
    p_val = float((dnull >= d_real_coarse).mean())
    print(f"  real map: Δℓ = ℓ(β̂) − ℓ(0) = {d_real:+.1f} (coarse grid {d_real_coarse:+.1f}); shifted maps: mean {dnull.mean():+.1f}, "
          f"q95 {np.quantile(dnull, .95):+.1f}, max {dnull.max():+.1f}; p = {p_val:.2f}   ({time.time() - t0:.0f} s)")

    # ------------------------------------------------------------- outage placebo
    xo = shrunk_log(F[H0]["num_outage"], F[H0]["den"], ALPHA)[0]
    res = {}
    for name, x in [("outage", xo), ("dumping", keep["x0"])]:
        X, pen, _, _ = design(x, fe_main, spline=False)
        th, ll = fit(X, y, pen)
        se = float(np.sqrt(OL.cov_cluster(th, X, y, blocks)[0, 0]))
        res[name] = {"slope": float(th[0]), "se": se, "dll_vs_FE": float(ll - ll_fe)}
    print(f"\n[placebo] linear slope on log field: outage {res['outage']['slope']:+.3f} (se {res['outage']['se']:.3f}, "
          f"Δℓ vs FE {res['outage']['dll_vs_FE']:+.1f})  vs  dumping {res['dumping']['slope']:+.3f} (se {res['dumping']['se']:.3f}, "
          f"Δℓ {res['dumping']['dll_vs_FE']:+.1f})")

    # ------------------------------------------------------------- save
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(CALIB, "table1.csv"), index=False)
    pd.DataFrame({"beta": BETA_GRID, **curves}).to_csv(os.path.join(CALIB, "profile_curves.csv"), index=False)
    summary = {"H0": H0, "H_LIST": H_LIST, "HW0": HW0, "beta_hat": beta_hat, "ci_95": ci_prof,
               "boot_sd": float(boot["beta_prof"].std()), "beta_ratio": rows[H_LIST.index(H0)]["beta_ratio"],
               "ratio_ci_95": ci_ratio, "ll_FE": ll_fe, "ll_0": float(ll0), "ll_hat": float(ll_hat),
               "torus": {"n": N_PERM, "d_real": float(d_real), "d_real_coarse": d_real_coarse,
                         "null_mean": float(dnull.mean()), "null_q95": float(np.quantile(dnull, .95)),
                         "null_max": float(dnull.max()), "p": p_val},
               "placebo": res, "table1": t1.round(4).to_dict(orient="records")}
    with open(os.path.join(CALIB, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    update_choices(beta_hat=beta_hat, ci_95=ci_prof)
    _figures(keep, curves, H_LIST, H0, beta_hat, ci_prof, boot, x_hat, th_hat, X_hat, info_hat, th_0, X_0, info_0, y, fe_main)
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Outputs in output/calib/, figures fig1_profile.png, fig2_linearisation.png")


def _figures(keep, curves, H_LIST, H0, beta_hat, ci, boot, x_hat, th_hat, X_hat, info_hat, th_0, X_0, info_0, y, fe):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for h, col in zip(H_LIST, [MUTED, C1, C3]):
        ll = curves[f"h{h}_month+surv_a{ALPHA:g}_hw{load_choices()['HW0']}"]
        ax.plot(BETA_GRID, ll - ll[I0], color=col, lw=2.2 if h == H0 else 1.3, label=f"h = {h} m")
    ax.axvspan(ci[0], ci[1], color=C1, alpha=0.10, lw=0)
    ax.axvline(beta_hat, color=C1, lw=1, ls="--"); ax.axvline(0, color=GRID, lw=1)
    ax.text(beta_hat, 5, f" β̂ = {beta_hat:.2f}", color=C1, fontsize=8)
    style(ax, "Profile log-likelihood relative to the raw field, ℓ(β) − ℓ(0)", "β", "Δℓ")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1]
    ax.hist(boot["beta_prof"], bins=25, color=C1, alpha=0.75, label="profile β̂")
    ax.hist(boot["beta_ratio"], bins=25, color=C2, alpha=0.45, label="−c/b")
    style(ax, f"Spatial block bootstrap ({len(boot)} resamples of 40 blocks)", "β̂", "count")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_profile.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    n_fe = fe[0].shape[1]
    for label, x, th, X, info, col in [("raw field (β = 0)", keep["x0"], th_0, X_0, info_0, C1),
                                        (f"corrected (β̂ = {beta_hat:.2f})", x_hat, th_hat, X_hat, info_hat, C2)]:
        grid = np.linspace(np.quantile(x, .005), np.quantile(x, .995), 200)
        Bg, _ = spline_basis(grid, info)
        p = X.shape[1]; beta, cuts = OL.unpack(th, p); ng = p - n_fe
        eta = Bg @ beta[:ng] + (X[:, ng:] @ beta[ng:]).mean()
        ax.plot(grid, 1.0 + expit(eta[:, None] - cuts[None, :]).sum(1), color=col, lw=2, label=label)
        dec = pd.qcut(x, 10, labels=False, duplicates="drop")
        ax.scatter(pd.Series(x).groupby(dec).median(), pd.Series(y + 1.0).groupby(dec).mean(), color=col, s=14, zorder=3)
    style(ax, "E[rating | log λ̂] at the mean month/surveyor effect; dots = decile means", "log λ̂ (reports per unit exposure)", "rating")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_linearisation.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
