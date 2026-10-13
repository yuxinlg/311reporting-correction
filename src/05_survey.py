"""
05_survey.py
============
Stage 4: the Litter Index survey as the test of the correction. β̂ comes from
04 without the survey; here the 72,084 survey ratings say whether the
correction at β̂ helps, and which β the survey itself would pick.

Rating model (design.py, ordinal.py) — proportional-odds logit, three sigmoids
sharing one index:
    P(rating ≥ k) = σ( g(log λ̂_β) + month-of-sample + surveyor − κ_k ),  g a 5-df natural spline

A. In sample — the survey's profile of β. For each β on BETA_GRID (kernel sums
   from 03), refit the rating model; ℓ(β) is the maximised log-likelihood.
   β_survey = argmax (parabola through the best three grid points, then one
   exact kernel pass). One exact pass at β̂ gives Δℓ(β̂) = ℓ(β̂) − ℓ(0), the gain
   of the correction on the survey, and the loss ℓ(β_survey) − ℓ(β̂). CI of
   β_survey: spatial block bootstrap, N_BOOT resamples of the N_BLOCKS blocks
   (frequency weights; each resample walks the grid from β_survey until its
   maximum is bracketed), percentile. The test: Δℓ(β̂) > 0 and β̂ inside the
   survey's interval.
   Outage placebo: the field built from street-light-outage reports instead of
   dumping reports should predict ratings far less than the dumping field
   (linear slope with block-clustered SE, Δℓ over the fixed effects alone).

B. Out of sample — buffered, spatially blocked N_FOLDS-fold CV. The test fold
   is whole blocks; training drops every visit within BUFFER m of a test visit.
   Specs, all with month FE + ridged surveyor effects and spline knots from the
   training fold:
     FE   month + surveyor only                   the floor
     S0   g(log λ̂_0)                              raw field, every h in H_LIST
     S2   g(log λ̂_β̂)                              corrected field at the 04 value, H0;
                                                  nothing is chosen in-fold
   Reported: out-of-fold log-likelihood per visit, Somers' D, paired gain of S2
   over S0 (mean ± fold SE, folds better); block-level calibration of the
   held-out predictions. Surveyors appear in training through their other
   blocks, so absolute levels are optimistic; paired differences are not.

Writes  output/survey/{profile_curves.csv, profile_h{h}.npz, bootstrap.csv, fit_at_beta_hat.npz,
                       folds.csv, table2.csv, block_calibration*.csv, summary.json}
        output/figures/fig2_survey.png (profile, bootstrap, the rating sigmoid), fig2b_cv.png
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

from config import (OUT, FIG, FIELDS, SURVEY, TAU, ALPHA, BETA_GRID, N_BOOT, BOOT_MAX_MOVES, BOOT_GTOL, GTOL, N_JOBS,
                    N_FOLDS, BUFFER, SEED, C1, C2, C3, MUTED, GRID, style, load_choices, ensure_dirs)
from kernels import accumulate_st, shrunk_log
from ordinal import OrderedLogit, somers_d
from design import FE, design, spline_basis

warnings.filterwarnings("ignore")
OL = OrderedLogit(4)
I0 = int(np.argmin(np.abs(BETA_GRID)))            # index of β = 0 on the grid


# =============================================================================
# pieces
# =============================================================================

def field_log(num, den, lam_bar, alpha=ALPHA):
    return shrunk_log(num, den, alpha, lam_bar)[0]


def fit(X, y, pen, theta0=None, w=None, gtol=GTOL):
    th, ll, _ = OL.fit(X, y, w, theta0, penalty=pen, gtol=gtol)
    return th, ll


def profile(F, y, fe, idx, w=None, theta_path=None, gtol=GTOL):
    """ℓ(β) for the grid indices `idx`, warm-started; returns ll dict and theta dict."""
    lls, ths, th = {}, {}, None
    for i in idx:
        x = field_log(F["num"][:, i], F["den"], F["lam_bar_raw"] * F["mean_m"][i])
        X, pen, _, _ = design(x, fe)
        th, ll = fit(X, y, pen, theta_path[i] if theta_path is not None else th, w, gtol)
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


# =============================================================================
# worker processes for the bootstrap (spawned; inputs re-read from disk)
# =============================================================================

def _worker_state():
    ch = load_choices(); H0 = ch["H0"]
    v = pd.read_parquet(os.path.join(OUT, "visits.parquet"))
    y = v["rating"].to_numpy() - 1
    F = dict(np.load(os.path.join(FIELDS, f"h{H0}.npz"))); F["lam_bar_raw"] = float(F["lam_bar_raw"])
    fe = FE(v).build(v)
    z = np.load(os.path.join(SURVEY, f"profile_h{H0}.npz"))
    ths = {i: z["thetas"][i] for i in range(len(BETA_GRID))}
    c = dict(np.load(os.path.join(SURVEY, "boot_context.npz")))
    return v, y, F, fe, ths, c


def _boot_chunk(reps):
    """Resamples `reps`: walk the β grid uphill from the survey's maximum until bracketed, then a parabola."""
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
        rows.append({"rep": int(r), "beta_survey": bh, "moves": moves, "n_fits": len(lls), "at_edge": edge})
    return rows


# =============================================================================
# cross-validation pieces
# =============================================================================

def buffered_train(xy, test, buffer):
    """Training rows: not in the test fold and not within `buffer` m of any test visit."""
    tr = ~test
    if buffer > 0:
        d, _ = cKDTree(xy[test]).query(xy[tr], distance_upper_bound=buffer)
        idx = np.flatnonzero(tr)
        tr = np.zeros(len(xy), bool); tr[idx[~np.isfinite(d)]] = True
    return tr


def fit_score(spec, x, fe, y, tr, te, theta0=None):
    """Fit one spec on tr, score on te. x: field values (n,). Returns metrics, E[rating] on te, theta."""
    Xfe, pen_fe = fe
    if spec == "FE":
        X_tr, X_te, pen = Xfe[tr], Xfe[te], pen_fe
    else:
        X_tr, pen, _, info = design(x[tr], (Xfe[tr], pen_fe))
        X_te, _, _, _ = design(x[te], (Xfe[te], pen_fe), info=info)
    th, ll_tr, _ = OL.fit(X_tr, y[tr], theta0=theta0, penalty=pen, gtol=GTOL)
    Er = OL.expected_rating(th, X_te)
    return {"ll": float(OL.loglik_i(th, X_te, y[te]).sum()), "n": int(te.sum()), "ll_train": float(ll_tr),
            "somers_d": somers_d(y[te], Er), "rmse": float(np.sqrt(np.mean((Er - y[te] - 1) ** 2)))}, Er, th


# =============================================================================
# main
# =============================================================================

def main():
    t_start = time.time()
    print("=" * 68); print("05_survey.py — Stage 4: the survey as the test"); print("=" * 68)
    ensure_dirs()
    ch = load_choices(); H0, H_LIST, HW0, beta_hat, ci_hat = ch["H0"], ch["H_LIST"], ch["HW0"], float(ch["beta_hat"]), ch["ci_95"]
    v = pd.read_parquet(os.path.join(OUT, "visits.parquet"))
    y = v["rating"].to_numpy() - 1
    xy = v[["x", "y"]].to_numpy()
    blocks = v["block"].to_numpy(); nb = int(blocks.max()) + 1
    F = {h: dict(np.load(os.path.join(FIELDS, f"h{h}.npz"))) for h in H_LIST}
    for h in H_LIST:
        F[h]["lam_bar_raw"] = float(F[h]["lam_bar_raw"])
    fe = FE(v).build(v)
    th_fe, ll_fe = fit(fe[0], y, fe[1])
    print(f"  {len(v):,} visits, {nb} blocks; β̂ from 04 = {beta_hat:.3f} [{ci_hat[0]:.2f}, {ci_hat[1]:.2f}]; H0 = {H0}, h in {H_LIST}; "
          f"ℓ(FE only) = {ll_fe:.1f}")

    # ------------------------------------------------------------- A1. the survey's profile
    print("\n[A] in sample: the survey's own profile ℓ(β) on the grid")
    curves, ths = {}, {}
    for h in H_LIST:
        t0 = time.time()
        cache = os.path.join(SURVEY, f"profile_h{h}.npz")
        if os.path.exists(cache) and np.allclose(np.load(cache)["grid"], BETA_GRID):
            z = np.load(cache); ll = z["ll"]; ths[h] = {i: z["thetas"][i] for i in range(len(BETA_GRID))}
        else:
            lls, th = profile(F[h], y, fe, range(len(BETA_GRID)))
            ll = np.array([lls[i] for i in range(len(BETA_GRID))]); ths[h] = th
            np.savez(cache, grid=BETA_GRID, ll=ll, thetas=np.stack([th[i] for i in range(len(BETA_GRID))]))
        curves[h] = ll
        bs, llm, edge = argmax_parabola(BETA_GRID, ll)
        print(f"  h = {h:>4}: ℓ(0) {ll[I0]:.1f}, max {llm:.1f} at β = {bs:.2f}{' (edge)' if edge else ''}, "
              f"Δℓ(max) {llm - ll[I0]:+.1f}; at the grid point nearest β̂ ({BETA_GRID[np.argmin(np.abs(BETA_GRID - beta_hat))]:g}): "
              f"{ll[np.argmin(np.abs(BETA_GRID - beta_hat))] - ll[I0]:+.1f}   ({time.time() - t0:.0f} s)")
    pd.DataFrame({"beta": BETA_GRID, **{f"h{h}": curves[h] for h in H_LIST}}).to_csv(os.path.join(SURVEY, "profile_curves.csv"), index=False)
    ll = curves[H0]; ll0 = float(ll[I0])
    beta_survey, _, edge = argmax_parabola(BETA_GRID, ll)

    # exact refits at β̂ (04) and at β_survey: one kernel pass with two weight vectors
    ev = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    wr = pd.read_parquet(os.path.join(OUT, "wR_events.parquet"))[f"wR_h{HW0}"].to_numpy()
    wr = np.where(np.isfinite(wr), wr, 0.0)
    ev_xy, ev_t = ev[["x", "y"]].to_numpy(), ev["t_days"].to_numpy()
    m = {"hat": np.exp(-beta_hat * wr), "surv": np.exp(-beta_survey * wr)}
    o = accumulate_st(xy, v["t_days"].to_numpy(), ev_xy, ev_t, H0, [TAU], m, src_tree=cKDTree(ev_xy))
    x0 = field_log(F[H0]["num"][:, I0], F[H0]["den"], F[H0]["lam_bar_raw"])
    exact = {}
    for key, b in (("hat", beta_hat), ("surv", beta_survey)):
        x = field_log(o[(TAU, key)], F[H0]["den"], F[H0]["lam_bar_raw"] * m[key].mean())
        X, pen, _, info = design(x, fe)
        th, llx = fit(X, y, pen, ths[H0][int(np.argmin(np.abs(BETA_GRID - b)))])
        exact[key] = {"x": x, "theta": th, "ll": llx, "X": X, "info": info}
    X_0, _, _, info_0 = design(x0, fe)
    dll_hat, dll_surv = exact["hat"]["ll"] - ll0, exact["surv"]["ll"] - ll0
    loss = exact["surv"]["ll"] - exact["hat"]["ll"]
    print(f"  exact refits: Δℓ(β̂ = {beta_hat:.2f}) = {dll_hat:+.1f}; Δℓ(β_survey = {beta_survey:.2f}) = {dll_surv:+.1f}; "
          f"loss of the survey's fit at β̂: {loss:.1f}")
    np.savez(os.path.join(SURVEY, "fit_at_beta_hat.npz"), beta_hat=beta_hat, beta_survey=beta_survey, x0=x0,
             x_hat=exact["hat"]["x"], theta_hat=exact["hat"]["theta"], theta_0=ths[H0][I0], ll_hat=exact["hat"]["ll"], ll_0=ll0,
             x_surv=exact["surv"]["x"], theta_surv=exact["surv"]["theta"], ll_surv=exact["surv"]["ll"])

    # ------------------------------------------------------------- A2. bootstrap of β_survey
    i_hat = int(np.argmin(np.abs(BETA_GRID - beta_survey)))
    np.savez(os.path.join(SURVEY, "boot_context.npz"), beta_survey=beta_survey, i_hat=i_hat)
    boot_path = os.path.join(SURVEY, "bootstrap.csv")
    brows = pd.read_csv(boot_path).to_dict(orient="records") if os.path.exists(boot_path) else []
    done = {int(r["rep"]) for r in brows}
    chunks = [[r for r in range(a, min(a + 10, N_BOOT)) if r not in done] for a in range(0, N_BOOT, 10)]
    chunks = [c for c in chunks if c]
    if not chunks:
        print(f"\n[bootstrap] reusing {len(brows)} saved resamples")
    else:
        print(f"\n[bootstrap] {N_BOOT} spatial block resamples of the survey's β at h = {H0}, {N_JOBS} workers"
              + (f", {len(done)} already saved" if done else ""))
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=N_JOBS, mp_context=mp.get_context("spawn")) as ex:
            futures = [ex.submit(_boot_chunk, c) for c in chunks]
            new = []
            for fut in as_completed(futures):
                new += fut.result()
                pd.DataFrame(brows + new).to_csv(boot_path, index=False)
                print(f"    {len(done) + len(new)}/{N_BOOT} ({time.time() - t0:.0f} s)", flush=True)
        brows += new
    boot = pd.DataFrame(brows).sort_values("rep").reset_index(drop=True)
    ci_surv = [float(q) for q in boot["beta_survey"].quantile([.025, .975])]
    inside = bool(ci_surv[0] <= beta_hat <= ci_surv[1])
    print(f"  β_survey = {beta_survey:.3f}, bootstrap sd {boot['beta_survey'].std():.3f}, 95% [{ci_surv[0]:.2f}, {ci_surv[1]:.2f}]; "
          f"β̂ = {beta_hat:.2f} is {'inside' if inside else 'OUTSIDE'} the survey's interval; "
          f"{int(boot['at_edge'].sum())} resamples ended unbracketed")

    # ------------------------------------------------------------- A3. outage placebo
    xo = shrunk_log(F[H0]["num_outage"], F[H0]["den"], ALPHA)[0]
    placebo = {}
    for name, x in (("outage", xo), ("dumping", x0)):
        X, pen, _, _ = design(x, fe, spline=False)
        th, llx = fit(X, y, pen)
        se = float(np.sqrt(OL.cov_cluster(th, X, y, blocks)[0, 0]))
        placebo[name] = {"slope": float(th[0]), "se": se, "dll_vs_FE": float(llx - ll_fe)}
    print(f"  placebo: linear slope on the log field — outage {placebo['outage']['slope']:+.3f} (se {placebo['outage']['se']:.3f}, "
          f"Δℓ vs FE {placebo['outage']['dll_vs_FE']:+.1f}) vs dumping {placebo['dumping']['slope']:+.3f} "
          f"(se {placebo['dumping']['se']:.3f}, Δℓ {placebo['dumping']['dll_vs_FE']:+.1f})")

    # ------------------------------------------------------------- B. blocked CV
    print(f"\n[B] out of sample: {N_FOLDS} spatially blocked folds, buffer {BUFFER} m; S2 at β̂ = {beta_hat:.2f} fixed")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(nb)
    fold_of_block = np.empty(nb, int)
    for k in range(N_FOLDS):
        fold_of_block[perm[k::N_FOLDS]] = k
    fold = fold_of_block[blocks]
    xs = {h: field_log(F[h]["num"][:, I0], F[h]["den"], F[h]["lam_bar_raw"]) for h in H_LIST}
    rows, warm = [], {}
    oof = {"S0": np.full(len(v), np.nan), "S2": np.full(len(v), np.nan)}
    for k in range(N_FOLDS):
        t0 = time.time()
        te = fold == k
        tr = buffered_train(xy, te, BUFFER)
        lost = int((~te).sum() - tr.sum())
        jobs = [("FE", H0, None)] + [("S0", h, xs[h]) for h in H_LIST] + [("S2", H0, exact["hat"]["x"])]
        for spec, h, x in jobs:
            sc, Er, th = fit_score(spec, x, fe, y, tr, te, warm.get((spec, h)))
            warm[(spec, h)] = th
            rows.append({"fold": k, "spec": spec, "h": h, "n_train": int(tr.sum()), "buffer_lost": lost, **sc})
            if h == H0 and spec in oof:
                oof[spec][te] = Er
        s0 = [r for r in rows if r["fold"] == k and r["spec"] == "S0" and r["h"] == H0][0]
        s2 = [r for r in rows if r["fold"] == k and r["spec"] == "S2"][0]
        print(f"  fold {k}: test {te.sum():,}, train {tr.sum():,} ({lost / (~te).sum():.0%} dropped by the buffer); "
              f"h = {H0}: S0 {s0['ll'] / s0['n']:.4f}, S2 {s2['ll'] / s2['n']:.4f}   ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(SURVEY, "folds.csv"), index=False)
    res = pd.DataFrame(rows)
    t2 = []
    for (spec, h), g in res.groupby(["spec", "h"]):
        g = g.sort_values("fold"); base = res[(res["spec"] == "S0") & (res["h"] == h)].sort_values("fold")
        dd = (g["ll"].to_numpy() - base["ll"].to_numpy()) / g["n"].to_numpy()
        t2.append({"spec": spec, "h": h, "ll_per_visit": g["ll"].sum() / g["n"].sum(), "somers_d": g["somers_d"].mean(),
                   "rmse": g["rmse"].mean(), "gain_vs_S0": dd.mean(), "gain_se": dd.std(ddof=1) / np.sqrt(len(dd)),
                   "folds_better": int((dd > 0).sum())})
    t2 = pd.DataFrame(t2).sort_values(["h", "spec"]); t2.to_csv(os.path.join(SURVEY, "table2.csv"), index=False)
    print("  out-of-fold log-likelihood per visit:")
    print(t2.pivot_table(index="spec", columns="h", values="ll_per_visit").round(4).to_string())
    g2 = t2[t2["spec"] == "S2"].iloc[0]
    print(f"  S2 vs S0 at h = {H0}: {g2['gain_vs_S0']:+.4f} ± {g2['gain_se']:.4f} per visit, {g2['folds_better']}/{N_FOLDS} folds better; "
          f"Somers' D {t2[(t2['spec'] == 'S0') & (t2['h'] == H0)]['somers_d'].iloc[0]:.3f} → {g2['somers_d']:.3f}")
    cal = []
    for spec in ("S0", "S2"):
        bc = pd.DataFrame({"block": blocks, "obs": y + 1.0, "pred": oof[spec]}).groupby("block").mean()
        slope, intercept = np.polyfit(bc["pred"], bc["obs"], 1)
        cal.append({"spec": spec, "slope": slope, "intercept": intercept, "corr": bc["obs"].corr(bc["pred"]),
                    "rmse_block": float(np.sqrt(np.mean((bc["obs"] - bc["pred"]) ** 2)))})
        bc.to_csv(os.path.join(SURVEY, f"block_calibration_{spec}.csv"))
        print(f"  block-level calibration {spec}: slope {slope:.3f}, corr {cal[-1]['corr']:.3f}, block RMSE {cal[-1]['rmse_block']:.3f}")
    pd.DataFrame(cal).to_csv(os.path.join(SURVEY, "block_calibration.csv"), index=False)

    # ------------------------------------------------------------- save
    summary = {"beta_hat": beta_hat, "ci_95_beta_hat": ci_hat, "beta_survey": beta_survey, "ci_95_survey": ci_surv,
               "beta_hat_inside_survey_ci": inside, "H0": H0, "ll_FE": ll_fe, "ll_0": ll0, "ll_hat": exact["hat"]["ll"],
               "ll_survey": exact["surv"]["ll"], "dll_hat": dll_hat, "dll_survey": dll_surv, "loss_at_beta_hat": loss,
               "profile_max_by_h": {str(h): argmax_parabola(BETA_GRID, curves[h])[0] for h in H_LIST},
               "placebo": placebo, "cv": {"folds": N_FOLDS, "buffer": BUFFER, "table2": t2.round(5).to_dict(orient="records"),
                                          "block_calibration": cal}}
    with open(os.path.join(SURVEY, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    _figures(curves, H_LIST, H0, beta_hat, beta_survey, ci_surv, boot, exact, x0, X_0, info_0, ths[H0][I0], y, fe, t2, nb)
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Outputs in output/survey/, figures fig2_survey.png, fig2b_cv.png")


def _figures(curves, H_LIST, H0, beta_hat, beta_survey, ci, boot, exact, x0, X_0, info_0, th_0, y, fe, t2, nb):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.1))
    ax = axes[0]
    for h, col in zip(H_LIST, [MUTED, C1, C3]):
        ax.plot(BETA_GRID, curves[h] - curves[h][I0], color=col, lw=2.2 if h == H0 else 1.3, label=f"h = {h} m")
    ax.axvspan(ci[0], ci[1], color=C1, alpha=0.10, lw=0)
    ax.axvline(beta_survey, color=C1, lw=1, ls="--"); ax.axvline(beta_hat, color=C2, lw=1.8); ax.axvline(0, color=GRID, lw=1)
    ax.text(beta_hat, -150, f" β̂ (04) = {beta_hat:.2f}", color=C2, fontsize=8, ha="right")
    ax.text(beta_survey, -300, f" survey {beta_survey:.2f}", color=C1, fontsize=8)
    style(ax, "The survey's profile: ℓ(β) − ℓ(0), month + surveyor effects", "β", "Δℓ")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1]
    ax.hist(boot["beta_survey"], bins=25, color=C1, alpha=0.75, label="survey β, block bootstrap")
    ax.axvline(beta_hat, color=C2, lw=1.8, label=f"β̂ from the block groups ({beta_hat:.2f})")
    style(ax, f"Does the survey agree with β̂? ({len(boot)} resamples of {nb} blocks)", "β", "count")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    n_fe = fe[0].shape[1]
    for label, x, th, X, info, col in [("raw field (β = 0)", x0, th_0, X_0, info_0, C1),
                                        (f"corrected (β̂ = {beta_hat:.2f})", exact["hat"]["x"], exact["hat"]["theta"], exact["hat"]["X"], exact["hat"]["info"], C2)]:
        grid = np.linspace(np.quantile(x, .005), np.quantile(x, .995), 200)
        Bg, _ = spline_basis(grid, info)
        p = X.shape[1]; beta, cuts = OL.unpack(th, p); ng = p - n_fe
        eta = Bg @ beta[:ng] + (X[:, ng:] @ beta[ng:]).mean()
        ax.plot(grid, 1.0 + expit(eta[:, None] - cuts[None, :]).sum(1), color=col, lw=2, label=label)
        dec = pd.qcut(x, 10, labels=False, duplicates="drop")
        ax.scatter(pd.Series(x).groupby(dec).median(), pd.Series(y + 1.0).groupby(dec).mean(), color=col, s=14, zorder=3)
    style(ax, "E[rating | log λ̂], mean month/surveyor effect; dots = decile means", "log λ̂ (reports per unit exposure)", "rating")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_survey.png"), dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for spec, col, mk in [("FE", MUTED, "x"), ("S0", C1, "o"), ("S2", C2, "s")]:
        s = t2[t2["spec"] == spec].sort_values("h")
        ax.plot(s["h"], s["ll_per_visit"], "-" + mk, color=col, lw=1.8, ms=6, label=spec + (f" (β̂ = {beta_hat:.2f})" if spec == "S2" else ""))
    ax.set_xscale("log"); ax.set_xticks(H_LIST); ax.set_xticklabels([str(h) for h in H_LIST]); ax.minorticks_off()
    style(ax, "Out-of-fold log-likelihood per visit", "h (m)", "ℓ / visit"); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for spec, col, mk in [("S0", C1, "o"), ("S2", C2, "s")]:
        bc = pd.read_csv(os.path.join(SURVEY, f"block_calibration_{spec}.csv"))
        ax.scatter(bc["pred"], bc["obs"], color=col, s=18, marker=mk, alpha=0.8, label=spec)
    lo, hi = ax.get_xlim(); ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    style(ax, f"Block-level calibration ({nb} held-out blocks, h = {H0})", "predicted mean rating", "observed mean rating")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2b_cv.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
