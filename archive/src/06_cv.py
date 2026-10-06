"""
06_cv.py
========
Stage 3, step 2: buffered, spatially blocked, grouped, nested cross-validation.

Outer: the 40 CV blocks (03_litter_panel) in 10 folds of 4 blocks. Training
drops every visit within BUFFER m of a test visit. Location grouping is
automatic (blocks are unions of CBGs). Surveyor effects use deviation coding
with a small ridge, so a surveyor unseen in training predicts at the
population mean.

Specs (all with month FE + ridged surveyor effects; g = 5-df natural spline
with knots from the training fold):
  FE       month + surveyor only                        baseline
  S0       g(log lam_raw)
  S0fine   g(log lam_raw,h) + b_f log lam_raw,150       two-scale check
  S1       b log lam_raw + c w_bar_k
  S2k      g(log lam_corr(beta0)), kernel w_R           the plug-in correction
  S2c      g(log lam_corr(beta0)), CBG posterior w_R
  S2kfit   g(log lam_corr(beta)), beta chosen on the TRAINING fold from a
           coarse grid by training log-likelihood     (out-of-sample SP)
  S4       g(log lam_raw) + CBG covariates              prediction robustness
over h in H_LIST and tau in TAU_LIST (ratio form, alpha = 1).

Reported
  - out-of-fold log-likelihood per visit, Somers' D, Spearman, RMSE of
    E[rating], for every (spec, h, tau); paired fold differences vs S0 with
    a fold-based SE; the (h, tau) surface for S2k with the 1-SE rule
  - block-level calibration (40 test blocks: observed vs predicted mean)
  - nested selection: inner 5-fold on the training blocks picks (spec, h)
    among {S0, S2k, S2c} x H_LIST at tau = 120; refit; outer score
  - temporal holdout: train 2022 -> test 2023 and the reverse
  - buffer sensitivity: 0 and 2400 m for the default configuration

Outputs  output/cv/*.csv, summary.json, output/figures/06_*.png
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from patsy import dmatrix, build_design_matrices
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernels import shrunk_log
from ordinal import OrderedLogit, somers_d

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
CAL = os.path.join(OUT, "calib")
CV = os.path.join(OUT, "cv")
FIG = os.path.join(OUT, "figures")
os.makedirs(CV, exist_ok=True)

H_LIST = [500, 800, 1200]
TAU_LIST = [120, 240]
H0, TAU0 = 800, 120
H_FINE = 150
ALPHA = 1.0
SPLINE_DF = 5
N_FOLDS = 10
N_INNER = 5
BUFFER = 1200                 # m, outer folds (the plan's 2 x h_max); with 10 folds this drops
                              # roughly a quarter of the training visits
BUFFER_INNER = 600            # m, inner selection folds: the full buffer around inner + outer
                              # test blocks would leave the inner training folds too thin
BUFFER_SENS = [0, 2400]
RIDGE_SURV = 1.0
GTOL = 1e-4
ALPHA_LIST = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
SEED = 0
BETA_FIT_GRID = [0.3, 0.55, 0.8, 1.05, 1.3, 1.55]     # S2kfit candidates (must lie on 05's grid)
WR_PRIMARY = "wR_pre24_naive_h600"
COVARS = ["pop_density", "med_inc_avg", "edu_hs_avg", "vac_area", "ndvi_mean_4yr",
          "use_res", "use_com", "total_length_w"]
SPECS = ["FE", "S0", "S0fine", "S1", "S2k", "S2c", "S2kfit", "S4"]
INNER_SPECS = ["S0", "S2k", "S2c"]

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
OL = OrderedLogit(4)


# =============================================================================
# data
# =============================================================================

def load():
    v = pd.read_parquet(os.path.join(OUT, "litter_panel.parquet"))
    F = pd.read_parquet(os.path.join(OUT, "visit_fields.parquet"))
    with open(os.path.join(OUT, "field_summary.json")) as f:
        beta0 = float(json.load(f)["beta0"])
    prof = {}
    for h in H_LIST:
        z = np.load(os.path.join(CAL, f"profile_num_h{h}_t{TAU0}.npz"))
        prof[h] = {"grid": z["grid"], "num_k": z["num_k"]}
    v["wR_point_k"] = v[WR_PRIMARY].to_numpy(float)
    v["cmonth"] = pd.to_datetime(v["ts"]).dt.month
    for c in COVARS:
        x = v[c].to_numpy(float)
        x = np.log1p(np.maximum(x, 0)) if c in ("pop_density", "med_inc_avg", "vac_area", "total_length_w") else x
        x = np.where(np.isfinite(x), x, np.nanmean(x))
        v[f"cov_{c}"] = (x - x.mean()) / (x.std() + 1e-12)
    return v, F, beta0, prof


def fields(F, v, h, tau, alpha=ALPHA):
    den = F[f"den_expo_h{h}"].to_numpy()
    raw, _ = shrunk_log(F[f"num_h{h}_t{tau}"].to_numpy(), den, alpha)
    ck, _ = shrunk_log(F[f"num_k_h{h}_t{tau}"].to_numpy(), den, alpha)
    cc, _ = shrunk_log(F[f"num_c_h{h}_t{tau}"].to_numpy(), den, alpha)
    wb = F[f"wbar_k_h{h}_t{tau}"].to_numpy()
    wb = np.where(np.isfinite(wb), wb, v["wR_point_k"].to_numpy())
    return {"raw": raw, "corr_k": ck, "corr_c": cc, "wbar_k": np.where(np.isfinite(wb), wb, 0.0), "den": den}


# =============================================================================
# design
# =============================================================================

def spline_basis(x, info=None):
    if info is None:
        B = dmatrix(f"cr(x, df={SPLINE_DF}) - 1", {"x": x}, return_type="dataframe")
        info = {"di": B.design_info, "mean": B.to_numpy().mean(0)}
        Bm = B.to_numpy()
    else:
        Bm = np.asarray(build_design_matrices([info["di"]], {"x": x})[0])
    return (Bm - info["mean"])[:, 1:], info


class FE:
    """Month dummies (drop first) + deviation-coded surveyors, consistent
    columns across folds; penalty vector marks the surveyor columns.
    month_col = "month" (year-month, 17 levels) for the spatial folds;
    "cmonth" (calendar month) for the temporal holdout, where 2023's
    year-months are unseen when training on 2022."""

    def __init__(self, v, month_col="month"):
        self.month_col = month_col
        self.months = np.sort(v[month_col].unique())
        surv_counts = v["surv"].value_counts()
        self.ref = int(surv_counts.index[0])
        self.survs = [int(s) for s in np.sort(v["surv"].unique()) if s != self.ref]

    def build(self, vs):
        m = vs[self.month_col].to_numpy(); s = vs["surv"].to_numpy()
        M = np.column_stack([(m == k).astype(float) for k in self.months[1:]])
        S = np.column_stack([np.where(s == k, 1.0, np.where(s == self.ref, -1.0, 0.0)) for k in self.survs])
        X = np.hstack([M, S])
        pen = np.r_[np.zeros(M.shape[1]), np.full(S.shape[1], RIDGE_SURV)]
        return X, pen


def make_X(spec, tr, te, f, fe, fine=None, cov=None, x_corr_fit=None):
    """Design matrices for train/test, plus the penalty vector."""
    cols_tr, cols_te = [], []
    def add_spline(x):
        B_tr, info = spline_basis(x[tr]); B_te, _ = spline_basis(x[te], info)
        cols_tr.append(B_tr); cols_te.append(B_te)
    def add_linear(x):
        mu = x[tr].mean(); cols_tr.append((x[tr] - mu)[:, None]); cols_te.append((x[te] - mu)[:, None])
    if spec == "S0" or spec == "S4":
        add_spline(f["raw"])
    elif spec == "S0fine":
        add_spline(f["raw"]); add_linear(fine)
    elif spec == "S1":
        add_linear(f["raw"]); add_linear(f["wbar_k"])
    elif spec == "S2k":
        add_spline(f["corr_k"])
    elif spec == "S2c":
        add_spline(f["corr_c"])
    elif spec == "S2kfit":
        add_spline(x_corr_fit)
    if spec == "S4":
        mu = cov[tr].mean(0); cols_tr.append(cov[tr] - mu); cols_te.append(cov[te] - mu)
    n_main = sum(c.shape[1] for c in cols_tr)
    Xf_tr, pen_fe = fe[0][tr], fe[1]
    Xf_te = fe[0][te]
    X_tr = np.hstack(cols_tr + [Xf_tr]) if cols_tr else Xf_tr
    X_te = np.hstack(cols_te + [Xf_te]) if cols_te else Xf_te
    pen = np.r_[np.zeros(n_main), pen_fe]
    return X_tr, X_te, pen


def score(theta, X_te, y_te):
    ll_i = OL.loglik_i(theta, X_te, y_te)
    Er = OL.expected_rating(theta, X_te)
    return {"ll": float(ll_i.sum()), "n": int(len(y_te)), "somers_d": somers_d(y_te, Er),
            "spearman": float(spearmanr(y_te, Er).correlation), "rmse": float(np.sqrt(np.mean((Er - y_te - 1) ** 2)))}, Er


def fit_eval(spec, tr, te, f, fe, y, fine=None, cov=None, prof=None, h=None):
    """Fit on tr, score on te. For S2kfit choose beta on the training fold."""
    x_fit, beta_fit = None, np.nan
    if spec == "S2kfit":
        best = (-np.inf, None, None)
        for b in BETA_FIT_GRID:
            i = int(np.argmin(np.abs(prof[h]["grid"] - b)))
            x, _ = shrunk_log(prof[h]["num_k"][:, i], f["den"], ALPHA)
            X_tr, _, pen = make_X("S2kfit", tr, te, f, fe, x_corr_fit=x)
            th, ll, _ = OL.fit(X_tr, y[tr], penalty=pen, gtol=GTOL)
            if ll > best[0]:
                best = (ll, b, x)
        beta_fit, x_fit = best[1], best[2]
    X_tr, X_te, pen = make_X(spec, tr, te, f, fe, fine=fine, cov=cov, x_corr_fit=x_fit)
    th, ll_tr, ok = OL.fit(X_tr, y[tr], penalty=pen, gtol=GTOL)
    sc, Er = score(th, X_te, y[te])
    sc.update({"ll_train": float(ll_tr), "converged": bool(ok), "beta_fit": beta_fit})
    return sc, Er


# =============================================================================
# folds and buffers
# =============================================================================

def buffered_train(v, test_mask, buffer):
    """Training rows: not in test, not within `buffer` m of any test visit."""
    tr = ~test_mask
    if buffer > 0:
        tree = cKDTree(v.loc[test_mask, ["x", "y"]].to_numpy())
        d, _ = tree.query(v.loc[tr, ["x", "y"]].to_numpy(), distance_upper_bound=buffer)
        keep = ~np.isfinite(d)
        idx = np.flatnonzero(tr)
        tr = np.zeros(len(v), bool); tr[idx[keep]] = True
    return tr


def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    print("=" * 68); print("06_cv.py — buffered blocked CV"); print("=" * 68)
    v, F, beta0, prof = load()
    y = v["rating"].to_numpy() - 1
    blocks = v["block"].to_numpy()
    nb = blocks.max() + 1
    fe_obj = FE(v); fe = fe_obj.build(v)
    cov = v[[f"cov_{c}" for c in COVARS]].to_numpy()
    fine_all = {tau: shrunk_log(F[f"num_h{H_FINE}_t{tau}"].to_numpy(), F[f"den_expo_h{H_FINE}"].to_numpy(), ALPHA)[0]
                for tau in TAU_LIST}
    fld = {(h, tau): fields(F, v, h, tau) for h in H_LIST for tau in TAU_LIST}
    print(f"  {len(v):,} visits, {nb} blocks, {N_FOLDS} outer folds, buffer {BUFFER} m, beta0 {beta0:.3f}")

    perm = rng.permutation(nb)
    fold_of_block = np.empty(nb, int)
    for k in range(N_FOLDS):
        fold_of_block[perm[k::N_FOLDS]] = k
    fold = fold_of_block[blocks]
    summary = {"beta0": beta0, "buffer": BUFFER, "folds": N_FOLDS, "blocks": int(nb)}

    # ------------------------------------------------------------ outer CV
    print("\n[outer] every (spec, h, tau) on every fold")
    rows, block_cal = [], []
    oof = {}                                                  # (spec,h,tau) -> E[rating] per visit
    for k in range(N_FOLDS):
        t0 = time.time()
        te = fold == k
        tr = buffered_train(v, te, BUFFER)
        n_lost = int((~te).sum() - tr.sum())
        for h in H_LIST:
            for tau in TAU_LIST:
                f = fld[(h, tau)]
                for spec in SPECS:
                    if spec == "FE" and (h != H0 or tau != TAU0):
                        continue
                    if spec == "S2kfit" and tau != TAU0:
                        continue
                    sc, Er = fit_eval(spec, tr, te, f, fe, y, fine=fine_all[tau], cov=cov, prof=prof, h=h)
                    rows.append({"fold": k, "spec": spec, "h": h, "tau": tau, "n_train": int(tr.sum()),
                                 "buffer_lost": n_lost, **sc})
                    key = (spec, h, tau)
                    oof.setdefault(key, np.full(len(v), np.nan))[te] = Er
        print(f"  fold {k}: test {te.sum():,} visits, train {tr.sum():,} ({n_lost:,} dropped by the buffer, "
              f"{n_lost / (~te).sum():.1%})   {time.time() - t0:.0f} s")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(CV, "outer_folds.csv"), index=False)

    # per-config aggregate: pooled ll per visit, mean Somers' D, paired diff vs S0 with fold SE
    agg_rows = []
    for (spec, h, tau), g in res.groupby(["spec", "h", "tau"]):
        g = g.sort_values("fold")
        base = res[(res["spec"] == "S0") & (res["h"] == h) & (res["tau"] == tau)].sort_values("fold")
        ll_per = g["ll"].sum() / g["n"].sum()
        d = (g["ll"].to_numpy() - base["ll"].to_numpy()) / g["n"].to_numpy() if len(base) == len(g) else np.full(len(g), np.nan)
        agg_rows.append({"spec": spec, "h": h, "tau": tau, "ll_per_visit": ll_per,
                         "somers_d": g["somers_d"].mean(), "spearman": g["spearman"].mean(), "rmse": g["rmse"].mean(),
                         "dll_vs_S0_per_visit": float(np.nanmean(d)), "dll_vs_S0_se": float(np.nanstd(d, ddof=1) / np.sqrt(len(d))),
                         "folds_better_than_S0": int(np.nansum(d > 0)), "beta_fit_mean": g["beta_fit"].mean()})
    agg = pd.DataFrame(agg_rows).sort_values(["tau", "h", "spec"])
    agg.to_csv(os.path.join(CV, "outer_summary.csv"), index=False)
    print("\n  out-of-fold log-likelihood per visit (tau = 120):")
    piv = agg[agg["tau"] == TAU0].pivot_table(index="spec", columns="h", values="ll_per_visit")
    print(piv.round(5).to_string())
    print("  paired Δll vs S0 per visit (mean ± fold SE), tau = 120:")
    for spec in ["S1", "S2k", "S2c", "S2kfit", "S0fine", "S4"]:
        s = agg[(agg["spec"] == spec) & (agg["tau"] == TAU0)]
        print(f"    {spec:>7}: " + "  ".join(f"h={int(r.h)}: {r.dll_vs_S0_per_visit:+.5f} ± {r.dll_vs_S0_se:.5f} ({r.folds_better_than_S0}/10 folds)"
                                          for r in s.itertuples()))
    print("  Somers' D (tau = 120):")
    print(agg[agg["tau"] == TAU0].pivot_table(index="spec", columns="h", values="somers_d").round(4).to_string())

    # (h, tau) surface for S2k and S0 with the 1-SE rule
    surf = agg[agg["spec"].isin(["S0", "S2k"])].copy()
    surf.to_csv(os.path.join(CV, "h_tau_surface.csv"), index=False)
    best = surf[surf["spec"] == "S2k"].sort_values("ll_per_visit", ascending=False).iloc[0]
    # 1-SE rule: fold-paired SE of the difference between each config and the best
    base_best = res[(res["spec"] == "S2k") & (res["h"] == best["h"]) & (res["tau"] == best["tau"])].sort_values("fold")
    one_se = []
    for (h, tau), g in res[res["spec"] == "S2k"].groupby(["h", "tau"]):
        g = g.sort_values("fold")
        d = (g["ll"].to_numpy() - base_best["ll"].to_numpy()) / g["n"].to_numpy()
        one_se.append({"h": h, "tau": tau, "diff_to_best": d.mean(), "se": d.std(ddof=1) / np.sqrt(len(d)),
                       "within_1se": bool(d.mean() + d.std(ddof=1) / np.sqrt(len(d)) >= 0)})
    one_se = pd.DataFrame(one_se)
    smallest = one_se[one_se["within_1se"]].sort_values(["h", "tau"]).iloc[0]
    print(f"\n  S2k (h, tau) surface: best h={int(best['h'])} tau={int(best['tau'])} "
          f"({best['ll_per_visit']:.5f}/visit); smallest h within 1 SE: h={int(smallest['h'])} tau={int(smallest['tau'])}")
    print(one_se.round(5).to_string(index=False))
    summary["surface"] = {"best": {"h": int(best["h"]), "tau": int(best["tau"]), "ll_per_visit": float(best["ll_per_visit"])},
                          "one_se_choice": {"h": int(smallest["h"]), "tau": int(smallest["tau"])},
                          "table": one_se.round(6).to_dict(orient="records")}

    # block-level calibration for the default configuration
    for spec in ["S0", "S2k"]:
        Er = oof[(spec, H0, TAU0)]
        bc = pd.DataFrame({"block": blocks, "obs": y + 1.0, "pred": Er}).groupby("block").mean()
        slope, intercept = np.polyfit(bc["pred"], bc["obs"], 1)
        block_cal.append({"spec": spec, "slope": slope, "intercept": intercept,
                          "corr": bc["obs"].corr(bc["pred"]), "rmse_block": float(np.sqrt(np.mean((bc["obs"] - bc["pred"]) ** 2)))})
        bc.to_csv(os.path.join(CV, f"block_calibration_{spec}.csv"))
        print(f"  block-level calibration {spec} (h={H0}): slope {slope:.3f}, intercept {intercept:+.3f}, "
              f"corr {bc['obs'].corr(bc['pred']):.3f}, block RMSE {block_cal[-1]['rmse_block']:.3f}")
    summary["block_calibration"] = block_cal

    # ------------------------------------------------------ nested selection
    print(f"\n[nested] inner {N_INNER}-fold on the training blocks (buffer {BUFFER_INNER} m) picks (spec, h) "
          f"among {INNER_SPECS} x {H_LIST}, tau={TAU0}")
    nrows = []
    for k in range(N_FOLDS):
        t0 = time.time()
        te = fold == k
        tr_blocks = np.flatnonzero(fold_of_block != k)
        inner_perm = rng.permutation(tr_blocks)
        inner_fold_of_block = np.full(nb, -1)
        for j in range(N_INNER):
            inner_fold_of_block[inner_perm[j::N_INNER]] = j
        inner = inner_fold_of_block[blocks]
        scores, shares = {}, []
        for j in range(N_INNER):
            ite = inner == j
            itr = buffered_train(v, ite | te, BUFFER_INNER) & (inner >= 0)
            shares.append(itr.sum() / max((inner >= 0).sum() - ite.sum(), 1))
            if itr.sum() < 2000:          # buffer ate the inner training fold; skip it
                continue
            for h in H_LIST:
                f = fld[(h, TAU0)]
                for spec in INNER_SPECS:
                    sc, _ = fit_eval(spec, itr, ite, f, fe, y)
                    scores.setdefault((spec, h), []).append(sc["ll"] / sc["n"])
        pick = max(scores, key=lambda key: np.mean(scores[key]))
        tr = buffered_train(v, te, BUFFER)
        sc, _ = fit_eval(pick[0], tr, te, fld[(pick[1], TAU0)], fe, y)
        nrows.append({"fold": k, "pick_spec": pick[0], "pick_h": pick[1], "inner_train_share": float(np.mean(shares)), **sc})
        print(f"  fold {k}: picked {pick[0]} h={pick[1]}  →  outer ll/visit {sc['ll'] / sc['n']:.5f}, Somers' D {sc['somers_d']:.4f}"
              f"   (inner training kept {np.mean(shares):.0%} after the buffer; {time.time() - t0:.0f} s)")
    nested = pd.DataFrame(nrows); nested.to_csv(os.path.join(CV, "nested.csv"), index=False)
    ll_nested = nested["ll"].sum() / nested["n"].sum()
    fixed = res[(res["spec"] == "S2k") & (res["h"] == H0) & (res["tau"] == TAU0)]
    ll_fixed = fixed["ll"].sum() / fixed["n"].sum()
    hind = agg.sort_values("ll_per_visit", ascending=False).iloc[0]
    print(f"  nested OOF ll/visit {ll_nested:.5f}   |  fixed default S2k h={H0} {ll_fixed:.5f}   |  best-in-hindsight "
          f"{hind['spec']} h={int(hind['h'])} tau={int(hind['tau'])} {hind['ll_per_visit']:.5f} (selection-biased)")
    print(f"  picks: {nested['pick_spec'].value_counts().to_dict()}  h: {nested['pick_h'].value_counts().to_dict()}")
    summary["nested"] = {"ll_per_visit": ll_nested, "somers_d": float(nested["somers_d"].mean()),
                         "picks_spec": nested["pick_spec"].value_counts().to_dict(),
                         "picks_h": {int(k): int(c) for k, c in nested["pick_h"].value_counts().items()},
                         "fixed_default_ll_per_visit": ll_fixed,
                         "hindsight": {"spec": hind["spec"], "h": int(hind["h"]), "tau": int(hind["tau"]), "ll_per_visit": float(hind["ll_per_visit"])}}

    # ------------------------------------------------------- temporal holdout
    print("\n[temporal] train one year, test the other (no buffer; same locations; calendar-month FE)")
    fe_cal = FE(v, month_col="cmonth").build(v)
    trows = []
    for yr_tr, yr_te in [(2022, 2023), (2023, 2022)]:
        tr = (v["year"] == yr_tr).to_numpy(); te = (v["year"] == yr_te).to_numpy()
        for spec in ["FE", "S0", "S2k", "S2c", "S1"]:
            sc, _ = fit_eval(spec, tr, te, fld[(H0, TAU0)], fe_cal, y)
            trows.append({"train": yr_tr, "test": yr_te, "spec": spec, **sc})
        s = pd.DataFrame(trows); s = s[(s["train"] == yr_tr)]
        print(f"  {yr_tr} → {yr_te}: " + "  ".join(f"{r.spec} {r.ll / r.n:.5f}" for r in s.itertuples()) +
              f"   (Somers' D S0 {s[s.spec == 'S0'].somers_d.iloc[0]:.4f}, S2k {s[s.spec == 'S2k'].somers_d.iloc[0]:.4f})")
    temporal = pd.DataFrame(trows); temporal.to_csv(os.path.join(CV, "temporal.csv"), index=False)
    summary["temporal"] = temporal.round(6).to_dict(orient="records")

    # ------------------------------------------------------ buffer sensitivity
    print("\n[buffer] default configuration under other buffers")
    brows = []
    for buf in BUFFER_SENS + [BUFFER]:
        for k in range(N_FOLDS):
            te = fold == k; tr = buffered_train(v, te, buf)
            for spec in ["S0", "S2k"]:
                sc, _ = fit_eval(spec, tr, te, fld[(H0, TAU0)], fe, y)
                brows.append({"buffer": buf, "fold": k, "spec": spec, "n_train": int(tr.sum()), **sc})
    bs = pd.DataFrame(brows); bs.to_csv(os.path.join(CV, "buffer_sensitivity.csv"), index=False)
    for buf, g in bs.groupby("buffer"):
        s0 = g[g.spec == "S0"]; s2 = g[g.spec == "S2k"]
        print(f"  buffer {buf:>4} m: train share {g['n_train'].mean() / (len(v) * (1 - 1 / N_FOLDS)):.1%}   "
              f"S0 {s0.ll.sum() / s0.n.sum():.5f}   S2k {s2.ll.sum() / s2.n.sum():.5f}   Δ {(s2.ll.sum() - s0.ll.sum()) / s0.n.sum():+.5f}")
    summary["buffer_sensitivity"] = bs.groupby(["buffer", "spec"]).apply(lambda g: g.ll.sum() / g.n.sum()).unstack().round(6).to_dict()

    # -------------------------------------------------------- alpha out of fold
    print(f"\n[alpha] Gamma-shrinkage alpha chosen out of fold at h={H0}, tau={TAU0}")
    arows = []
    for a in ALPHA_LIST:
        fa = fields(F, v, H0, TAU0, alpha=a)
        for k in range(N_FOLDS):
            te = fold == k; tr = buffered_train(v, te, BUFFER)
            for spec in ["S0", "S2k"]:
                sc, _ = fit_eval(spec, tr, te, fa, fe, y)
                arows.append({"alpha": a, "fold": k, "spec": spec, **sc})
    al = pd.DataFrame(arows); al.to_csv(os.path.join(CV, "alpha_sensitivity.csv"), index=False)
    atab = al.groupby(["spec", "alpha"]).apply(lambda g: g.ll.sum() / g.n.sum()).unstack()
    print(atab.round(5).to_string())
    best_alpha = {spec: float(atab.loc[spec].idxmax()) for spec in atab.index}
    print(f"  best alpha out of fold: {best_alpha}")
    summary["alpha"] = {"table": atab.round(6).to_dict(), "best": best_alpha}

    _figures(agg, one_se, oof, y, blocks)
    with open(os.path.join(CV, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Outputs in output/cv/")


def _style(ax, title, xlabel, ylabel):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9); ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def _figures(agg, one_se, oof, y, blocks):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    for spec, col, mk in [("S0", C1, "o"), ("S2k", C2, "s"), ("S2c", C3, "^"), ("S1", MUTED, "d")]:
        s = agg[(agg["spec"] == spec) & (agg["tau"] == TAU0)].sort_values("h")
        ax.plot(s["h"], s["ll_per_visit"], "-" + mk, color=col, lw=1.8, ms=5, label=spec)
    ax.set_xscale("log"); ax.set_xticks(H_LIST); ax.set_xticklabels([str(h) for h in H_LIST])
    _style(ax, f"Out-of-fold log-likelihood per visit, τ = {TAU0}", "h (m)", "ll / visit")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    Er0, Er2 = oof[("S0", H0, TAU0)], oof[("S2k", H0, TAU0)]
    bc = pd.DataFrame({"b": blocks, "obs": y + 1.0, "p0": Er0, "p2": Er2}).groupby("b").mean()
    ax.scatter(bc["p0"], bc["obs"], color=C1, s=18, label="S0", alpha=0.8)
    ax.scatter(bc["p2"], bc["obs"], color=C2, s=18, marker="s", label="S2k", alpha=0.8)
    lo, hi = min(bc[["p0", "p2", "obs"]].min()), max(bc[["p0", "p2", "obs"]].max())
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    _style(ax, f"Block-level calibration (40 test blocks, h = {H0})", "predicted mean rating", "observed mean rating")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "06_cv.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
