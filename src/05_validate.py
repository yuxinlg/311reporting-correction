"""
05_validate.py
==============
Out-of-sample validation: buffered, spatially blocked 10-fold cross-validation.

Folds: the N_BLOCKS spatial blocks (01) in N_FOLDS folds. The test fold is
whole blocks; training drops every visit within BUFFER m of a test visit, so
the neighbours of a test point never teach the model that predicts it.

Specs — all with month FE + ridged surveyor effects (a surveyor unseen in
training predicts at the population mean), spline knots from the training fold:
  FE   month + surveyor only                      the floor
  S0   g(log λ̂_0)                                 raw field
  S1   b·log λ̂_0 + c·w̄                           linear reduced form of the correction
  S2   g(log λ̂_β), β chosen on the TRAINING fold  the correction (argmax of the training
                                                 log-likelihood over CV_BETA_GRID)
S2 at H0 only; FE, S0, S1 at every h in H_LIST.

Reported per spec × h: out-of-fold log-likelihood per visit, Somers' D, paired
gain over S0 (mean ± fold SE, folds better), the range of β chosen in-fold;
block-level calibration at H0 (observed vs predicted block means).

Note: surveyors appear in training through their other blocks, so the surveyor
effects are a leakage channel common to every spec. Paired differences between
specs are immune to it; absolute levels are optimistic.

Writes  output/cv/{folds.csv, table2.csv, block_calibration.csv, summary.json}
        output/figures/05_cv.png
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (OUT, FIG, FIELDS, CVDIR, ALPHA, BETA_GRID, CV_BETA_GRID, N_FOLDS, BUFFER, GTOL, SEED,
                    C1, C2, C3, MUTED, style, load_choices, ensure_dirs)
from kernels import shrunk_log
from ordinal import OrderedLogit, somers_d
from design import FE, design

warnings.filterwarnings("ignore")
OL = OrderedLogit(4)
I0 = int(np.argmin(np.abs(BETA_GRID)))
SPECS = ["FE", "S0", "S1", "S2"]


def buffered_train(xy, test, buffer):
    """Training rows: not in the test fold and not within `buffer` m of any test visit."""
    tr = ~test
    if buffer > 0:
        d, _ = cKDTree(xy[test]).query(xy[tr], distance_upper_bound=buffer)
        idx = np.flatnonzero(tr)
        tr = np.zeros(len(xy), bool); tr[idx[~np.isfinite(d)]] = True
    return tr


def fit_score(spec, x, wbar, fe, y, tr, te, theta0=None):
    """Fit one spec on tr, score on te. x: field values (n,). Returns metrics and E[rating] on te."""
    Xfe, pen_fe = fe
    if spec == "FE":
        X_tr, X_te, pen = Xfe[tr], Xfe[te], pen_fe
    elif spec == "S1":
        X_tr, pen, _, _ = design(x[tr], (Xfe[tr], pen_fe), spline=False, extra={"wbar": wbar[tr]})
        X_te, _, _, _ = design(x[te], (Xfe[te], pen_fe), spline=False, extra={"wbar": wbar[te]})
    else:
        X_tr, pen, _, info = design(x[tr], (Xfe[tr], pen_fe))
        X_te, _, _, _ = design(x[te], (Xfe[te], pen_fe), info=info)
    th, ll_tr, _ = OL.fit(X_tr, y[tr], theta0=theta0, penalty=pen, gtol=GTOL)
    Er = OL.expected_rating(th, X_te)
    return {"ll": float(OL.loglik_i(th, X_te, y[te]).sum()), "n": int(te.sum()), "ll_train": float(ll_tr),
            "somers_d": somers_d(y[te], Er), "rmse": float(np.sqrt(np.mean((Er - y[te] - 1) ** 2)))}, Er, th


def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    print("=" * 68); print("05_validate.py — buffered spatially blocked CV"); print("=" * 68)
    ensure_dirs()
    ch = load_choices(); H0, H_LIST = ch["H0"], ch["H_LIST"]
    v = pd.read_parquet(os.path.join(OUT, "visits.parquet"))
    y = v["rating"].to_numpy() - 1
    xy = v[["x", "y"]].to_numpy()
    blocks = v["block"].to_numpy(); nb = int(blocks.max()) + 1
    fe = FE(v).build(v)
    fields = {}
    for h in H_LIST:
        F = np.load(os.path.join(FIELDS, f"h{h}.npz"))
        lam = float(F["lam_bar_raw"])
        fields[h] = {"x0": shrunk_log(F["num"][:, I0], F["den"], ALPHA, lam)[0], "wbar": F["wbar"],
                     "xb": {b: shrunk_log(F["num"][:, i], F["den"], ALPHA, lam * F["mean_m"][i])[0]
                            for b in CV_BETA_GRID for i in [int(np.argmin(np.abs(BETA_GRID - b)))]}}
    perm = rng.permutation(nb)
    fold_of_block = np.empty(nb, int)
    for k in range(N_FOLDS):
        fold_of_block[perm[k::N_FOLDS]] = k
    fold = fold_of_block[blocks]
    print(f"  {len(v):,} visits, {nb} blocks, {N_FOLDS} folds, buffer {BUFFER} m; H0 = {H0}, h in {H_LIST}; "
          f"in-fold β grid {CV_BETA_GRID[0]} … {CV_BETA_GRID[-1]} step {CV_BETA_GRID[1] - CV_BETA_GRID[0]:g}")

    rows, warm = [], {}
    oof = {("S0", H0): np.full(len(v), np.nan), ("S2", H0): np.full(len(v), np.nan)}
    for k in range(N_FOLDS):
        t0 = time.time()
        te = fold == k
        tr = buffered_train(xy, te, BUFFER)
        lost = int((~te).sum() - tr.sum())
        for h in H_LIST:
            f = fields[h]
            for spec in SPECS:
                if spec == "S2" and h != H0:
                    continue
                if spec == "S2":
                    best = None
                    for b in CV_BETA_GRID:
                        sc, Er, th = fit_score("S0", f["xb"][b], f["wbar"], fe, y, tr, te, warm.get(("S2", h, b)))
                        warm[("S2", h, b)] = th
                        if best is None or sc["ll_train"] > best[0]["ll_train"]:
                            best = (sc, Er, b)
                    sc, Er, beta_fit = best
                else:
                    sc, Er, th = fit_score(spec, f["x0"], f["wbar"], fe, y, tr, te, warm.get((spec, h)))
                    warm[(spec, h)] = th; beta_fit = np.nan
                rows.append({"fold": k, "spec": spec, "h": h, "n_train": int(tr.sum()), "buffer_lost": lost,
                             "beta_fit": beta_fit, **sc})
                if (spec, h) in oof:
                    oof[(spec, h)][te] = Er
        s2 = [r for r in rows if r["fold"] == k and r["spec"] == "S2"][0]
        s0 = [r for r in rows if r["fold"] == k and r["spec"] == "S0" and r["h"] == H0][0]
        print(f"  fold {k}: test {te.sum():,}, train {tr.sum():,} ({lost / (~te).sum():.0%} dropped by the buffer); "
              f"h = {H0}: S0 {s0['ll'] / s0['n']:.4f}, S2 {s2['ll'] / s2['n']:.4f} (β in-fold {s2['beta_fit']:.1f})   ({time.time() - t0:.0f} s)",
              flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(CVDIR, "folds.csv"), index=False)       # saved after every fold
    res = pd.DataFrame(rows)

    # Table 2: per spec × h
    t2 = []
    for (spec, h), g in res.groupby(["spec", "h"]):
        g = g.sort_values("fold"); base = res[(res["spec"] == "S0") & (res["h"] == h)].sort_values("fold")
        d = (g["ll"].to_numpy() - base["ll"].to_numpy()) / g["n"].to_numpy()
        t2.append({"spec": spec, "h": h, "ll_per_visit": g["ll"].sum() / g["n"].sum(), "somers_d": g["somers_d"].mean(),
                   "rmse": g["rmse"].mean(), "gain_vs_S0": d.mean(), "gain_se": d.std(ddof=1) / np.sqrt(len(d)),
                   "folds_better": int((d > 0).sum()), "beta_fit_min": g["beta_fit"].min(), "beta_fit_max": g["beta_fit"].max()})
    t2 = pd.DataFrame(t2).sort_values(["h", "spec"]); t2.to_csv(os.path.join(CVDIR, "table2.csv"), index=False)
    print("\n  out-of-fold log-likelihood per visit:")
    print(t2.pivot_table(index="spec", columns="h", values="ll_per_visit").round(4).to_string())
    print("  paired gain over S0 per visit (mean ± fold SE, folds better of %d):" % N_FOLDS)
    for r in t2[t2["spec"].isin(["S1", "S2"])].itertuples():
        print(f"    {r.spec} h={int(r.h):>4}: {r.gain_vs_S0:+.4f} ± {r.gain_se:.4f} ({r.folds_better}/{N_FOLDS})"
              + (f"   β in-fold {r.beta_fit_min:.1f}–{r.beta_fit_max:.1f}" if r.spec == "S2" else ""))
    print("  Somers' D:"); print(t2.pivot_table(index="spec", columns="h", values="somers_d").round(3).to_string())

    # block-level calibration at H0
    cal = []
    for spec in ["S0", "S2"]:
        bc = pd.DataFrame({"block": blocks, "obs": y + 1.0, "pred": oof[(spec, H0)]}).groupby("block").mean()
        slope, intercept = np.polyfit(bc["pred"], bc["obs"], 1)
        cal.append({"spec": spec, "slope": slope, "intercept": intercept, "corr": bc["obs"].corr(bc["pred"]),
                    "rmse_block": float(np.sqrt(np.mean((bc["obs"] - bc["pred"]) ** 2)))})
        bc.to_csv(os.path.join(CVDIR, f"block_calibration_{spec}.csv"))
        print(f"  block-level calibration {spec}: slope {slope:.3f}, intercept {intercept:+.3f}, corr {cal[-1]['corr']:.3f}, "
              f"block RMSE {cal[-1]['rmse_block']:.3f}")
    pd.DataFrame(cal).to_csv(os.path.join(CVDIR, "block_calibration.csv"), index=False)

    summary = {"H0": H0, "folds": N_FOLDS, "buffer": BUFFER, "table2": t2.round(5).to_dict(orient="records"),
               "block_calibration": cal}
    with open(os.path.join(CVDIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    ax = axes[0]
    for spec, col, mk in [("FE", MUTED, "x"), ("S0", C1, "o"), ("S1", C3, "d"), ("S2", C2, "s")]:
        s = t2[t2["spec"] == spec].sort_values("h")
        ax.plot(s["h"], s["ll_per_visit"], "-" + mk, color=col, lw=1.8, ms=6, label=spec + (" (β in-fold)" if spec == "S2" else ""))
    ax.set_xscale("log"); ax.set_xticks(H_LIST); ax.set_xticklabels([str(h) for h in H_LIST]); ax.minorticks_off()
    style(ax, "Out-of-fold log-likelihood per visit", "h (m)", "ℓ / visit"); ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for spec, col, mk in [("S0", C1, "o"), ("S2", C2, "s")]:
        bc = pd.read_csv(os.path.join(CVDIR, f"block_calibration_{spec}.csv"))
        ax.scatter(bc["pred"], bc["obs"], color=col, s=18, marker=mk, alpha=0.8, label=spec)
    lo, hi = ax.get_xlim(); ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    style(ax, f"Block-level calibration ({nb} held-out blocks, h = {H0})", "predicted mean rating", "observed mean rating")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "05_cv.png"), dpi=160); plt.close(fig)
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Outputs in output/cv/")


if __name__ == "__main__":
    main()
