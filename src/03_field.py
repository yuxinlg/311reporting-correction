"""
03_field.py
===========
Stage 2: the 311 dumping field at every survey visit. Geometry only, no regression.

    λ̂_β(s,t) = Σ_j exp(−β·w_R(s_j)) K_h(|s−s_j|) exp(−(t−t_j)/τ) 1[t ≥ t_j]
               / Σ_c K_h(|s−c|) E_c

K_h is a quartic kernel of radius h (sd = h/√7); the time kernel is a causal
exponential with scale τ; E_c is the SafeGraph exposure spread over the built
support (01). β = 0 is the raw field; the correction weights each report by
exp(−β w_R) so reports from high-propensity places count less.

Pass A — choose h and check τ, on the raw field:
  Spearman(rating, log λ̂_0) over H_SWEEP × TAU_SWEEP. Rule: at τ = TAU, H0 is the
  smallest h within H_RULE_TOL of the maximum; H_LIST = H0 and its two neighbours.
  The τ row is expected to be flat. Placebo: kernel-weighted reports within H0
  in the PLACEBO_WIN days before vs after each visit, rating regressed on both
  with month and location effects absorbed. If past and future tie, the time
  kernel carries no dynamics and τ is a pooling window.

Pass B — every kernel sum the statistics need, for h in H_LIST:
  num[β] for β in BETA_GRID; the pieces of w̄ (event-kernel-weighted mean of
  w_R, for the −c/b check); the exposure denominator; the shrinkage target λ̄
  computed on the support grid, not on the survey. At H0 only: num_alt[β] with
  the HW_ALT surface, num_null[p, β] for N_PERM torus-shifted w_R maps at
  NULL_BETAS, and the outage-placebo numerator.

Unit test: sparse accumulator vs a dense recomputation on 300 visits.

Writes
  output/fields/h{h}.npz   grid, den, num (n×|grid|), wbar, lam_bar_raw, mean_m
                           [+ num_alt, mean_m_alt, num_null (n×N_PERM×|NULL_BETAS|),
                              mean_m_null, num_outage at H0]
  output/field_sweep.csv, output/placebo.csv, output/choices.json (+ H0, H_LIST)
  output/figures/03_sweep.png
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (OUT, FIG, FIELDS, EPOCH, H_SWEEP, H_RULE_TOL, TAU_SWEEP, TAU, PLACEBO_WIN, ALPHA,
                    BETA_GRID, NULL_BETAS, N_PERM, DATE_IN, SEED, C1, C2, C3, MUTED, style,
                    load_choices, update_choices, ensure_dirs)
from kernels import kernel_sum, accumulate_st, pairs_within, quartic, shrunk_log, SD_FACTOR

warnings.filterwarnings("ignore")


# =============================================================================
# small helpers
# =============================================================================

def demean(Y, groups, tol=1e-9, max_iter=200):
    """Iterative within-transformation for one or several fixed-effect dimensions."""
    one_d = np.ndim(Y) == 1
    Y = np.array(Y, float).reshape(len(Y), -1)
    sizes = [np.bincount(g).astype(float) for g in groups]
    for _ in range(max_iter):
        prev = Y.copy()
        for g, sz in zip(groups, sizes):
            for c in range(Y.shape[1]):
                Y[:, c] -= (np.bincount(g, weights=Y[:, c]) / sz)[g]
        if np.abs(Y - prev).max() < tol:
            break
    return Y[:, 0] if one_d else Y


def ols_cluster(y, X, cluster):
    """OLS with intercept; cluster-robust covariance. Returns coef, se, cov (slopes only)."""
    X1 = np.c_[np.ones(len(y)), X]
    A = np.linalg.inv(X1.T @ X1)
    b = A @ X1.T @ y
    e = y - X1 @ b
    _, inv = np.unique(cluster, return_inverse=True)
    S = np.zeros((inv.max() + 1, X1.shape[1])); np.add.at(S, inv, X1 * e[:, None])
    G, (n, k) = S.shape[0], X1.shape
    V = A @ (S.T @ S * G / (G - 1) * (n - 1) / (n - k)) @ A
    return b[1:], np.sqrt(np.diag(V))[1:], V[1:, 1:]


def placebo_sums(v_xy, v_t, ev_xy, ev_t, tree, h, win):
    """Kernel-weighted reports within h in the `win` days before / after each visit."""
    n = len(v_xy); past = np.zeros(n); fut = np.zeros(n)
    for i, j, d in pairs_within(v_xy, tree, h):
        K = quartic(d, h); dt = v_t[i] - ev_t[j]
        past += np.bincount(i, weights=K * ((dt >= 0) & (dt < win)), minlength=n)
        fut += np.bincount(i, weights=K * ((dt < 0) & (dt >= -win)), minlength=n)
    return past, fut


def brute_check(v_xy, v_t, ev_xy, ev_t, h, num0, n_check=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(v_xy), n_check, replace=False))
    lam = np.zeros(n_check)
    for s in range(0, n_check, 50):
        D = np.sqrt(((v_xy[idx[s:s + 50], None, :] - ev_xy[None, :, :]) ** 2).sum(-1))
        dt = v_t[idx[s:s + 50], None] - ev_t[None, :]
        lam[s:s + 50] = (quartic(D, h) * np.where(dt >= 0, np.exp(-dt / TAU), 0.0)).sum(1)
    err = np.abs(lam - num0[idx]).max() / max(np.abs(lam).max(), 1e-12)
    print(f"  [unit test] sparse accumulator vs dense recomputation on {n_check} visits at h = {h}: "
          f"relative error {err:.1e} — {'OK' if err < 1e-10 else 'MISMATCH'}")
    return bool(err < 1e-10)


# =============================================================================
# main
# =============================================================================

def main():
    t_start = time.time()
    print("=" * 68); print("03_field.py — Stage 2: the 311 field at every visit"); print("=" * 68)
    ensure_dirs()
    ch = load_choices(); HW0, HW_ALT = ch["HW0"], ch["HW_ALT"]
    v = pd.read_parquet(os.path.join(OUT, "visits.parquet"))
    ev = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    wr = pd.read_parquet(os.path.join(OUT, "wR_events.parquet"))
    wv = pd.read_parquet(os.path.join(OUT, "wR_visits.parquet"))
    sup = pd.read_parquet(os.path.join(OUT, "support_grid.parquet"))
    v_xy, v_t, y = v[["x", "y"]].to_numpy(), v["t_days"].to_numpy(), v["rating"].to_numpy()
    ev_xy, ev_t = ev[["x", "y"]].to_numpy(), ev["t_days"].to_numpy()
    tree = cKDTree(ev_xy)
    sup_xy, sup_E = sup[["x", "y"]].to_numpy(), sup["E_cell"].to_numpy()
    sup_tree = cKDTree(sup_xy)
    w0, w_alt = wr[f"wR_h{HW0}"].to_numpy(), wr[f"wR_h{HW_ALT}"].to_numpy()
    print(f"  {len(v):,} visits, {len(ev):,} dumping reports, {len(sup):,} support cells; w_R at h_w = {HW0} (alt {HW_ALT})")

    # ------------------------------------------------------------- Pass A
    print(f"\n[A] choosing h: Spearman(rating, log λ̂_0) over h × τ  (α = {ALPHA})")
    rows, den = [], {}
    for h in H_SWEEP:
        t0 = time.time()
        o = accumulate_st(v_xy, v_t, ev_xy, ev_t, h, TAU_SWEEP, None, src_tree=tree)
        den[h] = kernel_sum(v_xy, sup_xy, h, weights=sup_E, source_tree=sup_tree)
        for tau in TAU_SWEEP:
            num = o[(tau, "one")]
            rows.append({"h": h, "tau": tau, "spearman": spearmanr(y, shrunk_log(num, den[h], ALPHA)[0]).correlation,
                         "zero_share": float((num == 0).mean())})
        r = [x for x in rows if x["h"] == h and x["tau"] == TAU][0]
        print(f"  h = {h:>4} m (sd {h * SD_FACTOR:>3.0f} m): Spearman at τ = {TAU}: {r['spearman']:.4f}, "
              f"zero numerators {r['zero_share']:.2%}   ({time.time() - t0:.0f} s)")
    sw = pd.DataFrame(rows); sw.to_csv(os.path.join(OUT, "field_sweep.csv"), index=False)
    piv = sw.pivot_table(index="tau", columns="h", values="spearman")
    print("  full table (rows τ, columns h):"); print(piv.round(4).to_string())
    at_tau = sw[sw["tau"] == TAU].set_index("h")["spearman"]
    H0 = int(at_tau[at_tau >= at_tau.max() - H_RULE_TOL].index.min())
    i0 = H_SWEEP.index(H0)
    H_LIST = [H_SWEEP[k] for k in (i0 - 1, i0, i0 + 1) if 0 <= k < len(H_SWEEP)]
    print(f"  rule: smallest h within {H_RULE_TOL} of the maximum ({at_tau.max():.4f} at h = {int(at_tau.idxmax())}) → "
          f"H0 = {H0} m; H_LIST = {H_LIST}. τ row at H0: " +
          ", ".join(f"{int(t)}: {s:.4f}" for t, s in piv[H0].items()) + " (flat → τ fixed at %d)" % TAU)

    # placebo: past vs future window at H0
    past, fut = placebo_sums(v_xy, v_t, ev_xy, ev_t, tree, H0, PLACEBO_WIN)
    X = np.c_[np.log1p(past), np.log1p(fut)]
    prow = []
    for label, groups in [("none", []), ("month + location", [v["month"].to_numpy(), v["loc"].to_numpy()])]:
        yy, XX = (y.astype(float), X) if not groups else (demean(y.astype(float), groups), demean(X, groups))
        b, se, V = ols_cluster(yy, XX, v["block"].to_numpy())
        d = b[0] - b[1]; sd = np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])
        prow.append({"fe": label, "past": b[0], "se_past": se[0], "future": b[1], "se_future": se[1],
                     "past_minus_future": d, "se_diff": sd, "t_diff": d / sd})
        print(f"  placebo ({PLACEBO_WIN} d windows, h = {H0}, FE: {label:>16}): past {b[0]:+.4f} (se {se[0]:.4f})  "
              f"future {b[1]:+.4f} (se {se[1]:.4f})  past − future {d:+.4f} (t = {d / sd:+.2f})")
    pd.DataFrame(prow).to_csv(os.path.join(OUT, "placebo.csv"), index=False)

    # ------------------------------------------------------------- Pass B
    print(f"\n[B] kernel sums for the statistics, h in {H_LIST}, β grid of {len(BETA_GRID)} ({BETA_GRID[0]} … {BETA_GRID[-1]})")
    rng = np.random.default_rng(SEED)
    z = np.load(os.path.join(OUT, "wR_raster.npz"))
    gx, gy, Z, res = z["x"], z["y"], np.where(np.isfinite(z["wR"]), z["wR"], 0.0), float(z["res"])
    ix = np.clip(np.round((ev_xy[:, 0] - gx[0]) / res).astype(int), 0, len(gx) - 1)
    iy = np.clip(np.round((ev_xy[:, 1] - gy[0]) / res).astype(int), 0, len(gy) - 1)
    shifts = [(int(rng.integers(0, len(gx))), int(rng.integers(0, len(gy)))) for _ in range(N_PERM)]
    ok0 = np.isfinite(w0).astype(float); w0f = np.where(np.isfinite(w0), w0, 0.0)
    w_vis = wv[f"wR_h{HW0}"].to_numpy()
    lam_bar, unit_ok = {}, None
    for h in H_LIST:
        t0 = time.time()
        weights = {f"b{i}": np.exp(-b * w0f) for i, b in enumerate(BETA_GRID)}
        weights["w"] = w0f * ok0; weights["ok"] = ok0
        if h == H0:
            for i, b in enumerate(BETA_GRID):
                weights[f"a{i}"] = np.exp(-b * np.where(np.isfinite(w_alt), w_alt, 0.0))
            for p, (sx, sy) in enumerate(shifts):
                ws = Z[(iy + sy) % len(gy), (ix + sx) % len(gx)]
                for k, b in enumerate(NULL_BETAS):
                    weights[f"n{p}_{k}"] = np.exp(-b * ws)
        o = accumulate_st(v_xy, v_t, ev_xy, ev_t, h, [TAU], weights, src_tree=tree)
        num = np.column_stack([o[(TAU, f"b{i}")] for i in range(len(BETA_GRID))])
        wbar = np.where(o[(TAU, "ok")] > 0, o[(TAU, "w")] / np.maximum(o[(TAU, "ok")], 1e-12), w_vis)
        wbar = np.where(np.isfinite(wbar), wbar, 0.0)
        # shrinkage target on the support grid (survey-independent), raw field at DATE_IN
        t_in = float((DATE_IN - EPOCH).days)
        os_ = accumulate_st(sup_xy, np.full(len(sup_xy), t_in), ev_xy, ev_t, h, [TAU], None, src_tree=tree)
        den_sup = kernel_sum(sup_xy, sup_xy, h, weights=sup_E, source_tree=sup_tree)
        lam_bar[h] = float(os_[(TAU, "one")].sum() / den_sup.sum())
        out = {"grid": BETA_GRID, "den": den[h], "num": num, "wbar": wbar, "lam_bar_raw": lam_bar[h],
               "mean_m": np.array([np.exp(-b * w0f).mean() for b in BETA_GRID])}
        if h == H0:
            out["num_alt"] = np.column_stack([o[(TAU, f"a{i}")] for i in range(len(BETA_GRID))])
            out["mean_m_alt"] = np.array([np.exp(-b * np.where(np.isfinite(w_alt), w_alt, 0.0)).mean() for b in BETA_GRID])
            out["num_null"] = np.stack([np.column_stack([o[(TAU, f"n{p}_{k}")] for k in range(len(NULL_BETAS))])
                                        for p in range(N_PERM)], axis=1)                      # n × N_PERM × |NULL_BETAS|
            out["mean_m_null"] = np.array([[np.exp(-b * Z[(iy + sy) % len(gy), (ix + sx) % len(gx)]).mean()
                                            for b in NULL_BETAS] for (sx, sy) in shifts])
            out["null_betas"] = np.array(NULL_BETAS)
            r = pd.read_parquet(os.path.join(OUT, "points_outage.parquet"))
            oo = accumulate_st(v_xy, v_t, r[["x", "y"]].to_numpy(), r["t_days"].to_numpy(), h, [TAU], None)
            out["num_outage"] = oo[(TAU, "one")]
            unit_ok = brute_check(v_xy, v_t, ev_xy, ev_t, h, num[:, int(np.argmin(np.abs(BETA_GRID)))])
        np.savez_compressed(os.path.join(FIELDS, f"h{h}.npz"), **out)
        i_raw = int(np.argmin(np.abs(BETA_GRID)))
        print(f"  h = {h:>4}: {len(weights)} weight vectors, zero numerators {(num[:, i_raw] == 0).mean():.2%}, "
              f"λ̄_raw (support grid) {lam_bar[h]:.4g}, Spearman raw {spearmanr(y, shrunk_log(num[:, i_raw], den[h], ALPHA, lam_bar[h])[0]).correlation:.4f}"
              f"   ({time.time() - t0:.0f} s)")

    update_choices(H0=H0, H_LIST=H_LIST, TAU=TAU,
                   sweep_at_tau={str(int(h)): round(float(s), 4) for h, s in at_tau.items()},
                   placebo=[{k: (round(float(x), 4) if not isinstance(x, str) else x) for k, x in r.items()} for r in prow],
                   lam_bar_raw={str(h): lam_bar[h] for h in H_LIST}, unit_test_pass=unit_ok)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for tau, col in zip(TAU_SWEEP, [MUTED, C1, C2, C3]):
        s = sw[sw["tau"] == tau].sort_values("h")
        ax.plot(s["h"], s["spearman"], "-o", color=col, lw=2 if tau == TAU else 1.2, ms=4, label=f"τ = {tau} d")
    ax.axvline(H0, color=MUTED, lw=1, ls="--"); ax.text(H0, ax.get_ylim()[0], f" H0 = {H0}", color=MUTED, fontsize=8)
    ax.set_xscale("log"); ax.set_xticks(H_SWEEP); ax.set_xticklabels([str(h) for h in H_SWEEP]); ax.minorticks_off()
    style(ax, "Spearman(rating, log λ̂_raw) by kernel radius", "quartic radius h (m)", "Spearman ρ")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "03_sweep.png"), dpi=160); plt.close(fig)
    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Saved → output/fields/h*.npz, field_sweep.csv, placebo.csv, choices.json")


if __name__ == "__main__":
    main()
