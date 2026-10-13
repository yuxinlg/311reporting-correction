"""
04_beta.py
==========
Stage 3: the strength of the correction, β, from the block groups. No survey.

Model — one Poisson regression across block groups i:

    T_i ~ Poisson( E_i · exp(a + β · w̄_R,i) )

T_i is the number of dumping reports in BETA_WINDOW, E_i the SafeGraph exposure
(offset), w̄_R,i the mean kernel w_R (02) over the block group's support cells.
β is the elasticity of the report rate to reporting propensity: a block group
one unit of w_R chattier files e^β times the reports at equal true dumping.

Identifying assumption: true dumping per unit of exposure is unrelated to
reporting propensity across block groups (the residual w_T is independent of
w_R — the same assumption as the two-equation decomposition of the archived
pipeline). Nothing from the survey enters; 05 then asks whether the survey
agrees with the value found here.

Fit: IRLS on the Poisson likelihood (the point process's own likelihood with
the counts aggregated); robust sandwich SE; 95% CI from a spatial block
bootstrap of the N_BLOCKS blocks (frequency weights), percentile.

Table 1 — the main row and five other readings of the same slope:
  window     reports from BETA_WINDOW_ALT (the survey years) instead of all five
  h_w alt    w_R at HW_ALT (a smoother propensity surface; β is in its units)
  log rate   least squares of log((T + ½) / E) on w̄_R: every block group weighted
             equally instead of by its expected count
  elasticity log E as a covariate instead of an offset: γ is the elasticity of
             reports to exposure, β the slope at equal exposure
  within     N_BLOCKS block fixed effects: the slope from neighbouring block
             groups only, broad citywide contrasts removed
The spread of these rows is the honest uncertainty of a survey-free β: they
agree only if true dumping really is unrelated to w_R however the comparison
is made.

Torus-shift null — N_PERM shifted copies of the w_R raster, each regressed the
same way; the deviance gain of the real map against theirs.

Writes  output/beta/{cbg.csv, table1.csv, bootstrap.csv, torus_null.csv, summary.json}
        output/choices.json (+ beta_hat, ci_95)
        output/figures/fig1_beta.png
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (OUT, FIG, BETA, LONG_WINDOWS, BETA_WINDOW, BETA_WINDOW_ALT, N_BOOT, N_PERM, N_BLOCKS, SEED,
                    C1, C2, C3, MUTED, GRID, style, load_choices, update_choices, ensure_dirs)

warnings.filterwarnings("ignore")


# =============================================================================
# the two regressions
# =============================================================================

def poisson_fit(X, y, offset, w=None, maxiter=200, tol=1e-10):
    """Poisson regression by IRLS. X without a constant (added here), offset = log exposure, w = frequency
    weights. Returns coef (constant first), robust sandwich covariance, deviance."""
    X1 = np.c_[np.ones(len(y)), np.asarray(X, float)]
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    b = np.zeros(X1.shape[1]); b[0] = np.log(np.sum(w * y) / np.sum(w * np.exp(offset)))
    for _ in range(maxiter):
        eta = X1 @ b + offset; mu = np.exp(eta)
        z = (eta - offset) + (y - mu) / mu
        A = X1.T @ ((w * mu)[:, None] * X1)
        b_new = np.linalg.pinv(A) @ (X1.T @ (w * mu * z))
        done = np.max(np.abs(b_new - b)) < tol
        b = b_new
        if done:
            break
    mu = np.exp(X1 @ b + offset)
    A_inv = np.linalg.pinv(X1.T @ ((w * mu)[:, None] * X1))      # pinv: a block absent from a resample has no FE
    S = X1 * (w * (y - mu))[:, None]
    V = A_inv @ (S.T @ S) @ A_inv
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2.0 * np.sum(w * (np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu)))
    return b, V, float(dev)


def ols_fit(X, y, w=None):
    """Least squares with a constant and frequency weights; robust covariance. Returns coef, cov."""
    X1 = np.c_[np.ones(len(y)), np.asarray(X, float)]
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    A_inv = np.linalg.inv(X1.T @ (w[:, None] * X1))
    b = A_inv @ (X1.T @ (w * y))
    e = y - X1 @ b
    S = X1 * (w * e)[:, None]
    return b, A_inv @ (S.T @ S) @ A_inv


SPECS = ["main", "window", "hw_alt", "log_rate", "elasticity", "within"]


def fit_spec(spec, d, w=None):
    """One reading of the slope on the block-group table d. Returns dict(beta, se, extra...)."""
    y, logE = d["T"].to_numpy(float), np.log(d["exposure"].to_numpy(float))
    wbar = d["wbar"].to_numpy()
    if spec == "main":
        b, V, dev = poisson_fit(wbar[:, None], y, logE, w)
        return {"beta": b[1], "se": np.sqrt(V[1, 1]), "a": b[0], "deviance": dev}
    if spec == "window":
        b, V, dev = poisson_fit(wbar[:, None], d["T_alt"].to_numpy(float), logE, w)
        return {"beta": b[1], "se": np.sqrt(V[1, 1]), "deviance": dev}
    if spec == "hw_alt":
        b, V, dev = poisson_fit(d["wbar_alt"].to_numpy()[:, None], y, logE, w)
        return {"beta": b[1], "se": np.sqrt(V[1, 1]), "deviance": dev}
    if spec == "log_rate":
        b, V = ols_fit(wbar[:, None], np.log((y + 0.5) / d["exposure"].to_numpy(float)), w)
        return {"beta": b[1], "se": np.sqrt(V[1, 1])}
    if spec == "elasticity":
        b, V, dev = poisson_fit(np.c_[wbar, logE], y, np.zeros(len(y)), w)
        return {"beta": b[1], "se": np.sqrt(V[1, 1]), "gamma": b[2], "deviance": dev}
    if spec == "within":
        blocks = d["block"].to_numpy()
        D = np.column_stack([(blocks == k).astype(float) for k in range(1, N_BLOCKS)])
        b, V, dev = poisson_fit(np.c_[wbar, D], y, logE, w)
        return {"beta": b[1], "se": np.sqrt(V[1, 1]), "deviance": dev}
    raise ValueError(spec)


# =============================================================================
# main
# =============================================================================

def main():
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    print("=" * 68); print("04_beta.py — Stage 3: β from the block groups (no survey)"); print("=" * 68)
    ensure_dirs()
    ch = load_choices(); HW0, HW_ALT = ch["HW0"], ch["HW_ALT"]

    # ------------------------------------------------------------- the block-group table
    cbg = pd.read_parquet(os.path.join(OUT, "cbg.parquet"))
    ev = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    sup = pd.read_parquet(os.path.join(OUT, "support_grid.parquet"))
    ws = pd.read_parquet(os.path.join(OUT, "wR_support.parquet"))
    blocks = pd.read_parquet(os.path.join(OUT, "cbg_blocks.parquet"))
    counts = {}
    for key, col in ((BETA_WINDOW, "T"), (BETA_WINDOW_ALT, "T_alt")):
        t0, t1 = LONG_WINDOWS[key]
        counts[col] = ev[(ev["ts"] >= t0) & (ev["ts"] < t1)].groupby("GEOID").size().rename(col)
    sup = sup.assign(wbar=ws[f"wR_h{HW0}"].to_numpy(), wbar_alt=ws[f"wR_h{HW_ALT}"].to_numpy())
    wb = sup.groupby("GEOID")[["wbar", "wbar_alt"]].mean()            # cells share the block group's exposure equally
    d = (cbg.merge(counts["T"], on="GEOID", how="left").merge(counts["T_alt"], on="GEOID", how="left")
            .merge(wb, on="GEOID", how="left").merge(blocks, on="GEOID", how="left"))
    d[["T", "T_alt"]] = d[["T", "T_alt"]].fillna(0).astype(int)
    ok = (d["exposure"] > 0) & (d["n_poles"] > 0) & d["wbar"].notna()
    d = d[ok].reset_index(drop=True)
    d["log_rate"] = np.log((d["T"] + 0.5) / d["exposure"])
    sd_w = float(d["wbar"].std())
    print(f"  {len(d):,} block groups with exposure, poles and support cells ({int((~ok).sum())} dropped); "
          f"{d['T'].sum():,} dumping reports {BETA_WINDOW} ({(d['T'] == 0).sum()} block groups with none); "
          f"w̄_R: sd {sd_w:.3f}, correlation with log exposure {d['wbar'].corr(np.log(d['exposure'])):+.2f}")

    # ------------------------------------------------------------- Table 1 with block-bootstrap CIs
    print(f"\n[table 1] the slope under six readings; CI from {N_BOOT} resamples of the {N_BLOCKS} blocks")
    nb = N_BLOCKS; bl = d["block"].to_numpy()
    boot_w = [np.bincount(rng.integers(0, nb, nb), minlength=nb)[bl].astype(float) for _ in range(N_BOOT)]
    rows, boot = [], {}
    for spec in SPECS:
        t0 = time.time()
        r = fit_spec(spec, d)
        boot[spec] = np.array([fit_spec(spec, d, w)["beta"] for w in boot_w])
        lo, hi = np.quantile(boot[spec], [.025, .975])
        rows.append({"spec": spec, "beta": r["beta"], "se_robust": r["se"], "ci_lo": lo, "ci_hi": hi,
                     "boot_sd": boot[spec].std(), "beta_x_sd": r["beta"] * (sd_w if spec != "hw_alt" else d["wbar_alt"].std()),
                     "gamma": r.get("gamma", np.nan), "deviance": r.get("deviance", np.nan), "a": r.get("a", np.nan)})
        print(f"  {spec:>10}: β {r['beta']:+.3f} (robust se {r['se']:.3f}; bootstrap [{lo:.2f}, {hi:.2f}])"
              + (f", exposure elasticity γ {r['gamma']:.2f}" if "gamma" in r else "") + f"   ({time.time() - t0:.0f} s)")
    t1 = pd.DataFrame(rows); t1.to_csv(os.path.join(BETA, "table1.csv"), index=False)
    pd.DataFrame({"rep": np.arange(N_BOOT), **{f"beta_{s}": boot[s] for s in SPECS}}).to_csv(os.path.join(BETA, "bootstrap.csv"), index=False)
    main_row = t1.iloc[0]
    beta_hat, ci = float(main_row["beta"]), [float(main_row["ci_lo"]), float(main_row["ci_hi"])]
    print(f"  β̂ = {beta_hat:.3f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}]; β̂ · sd(w̄_R) = {beta_hat * sd_w:.2f}: a block group one sd "
          f"chattier files e^{beta_hat * sd_w:.2f} = {np.exp(beta_hat * sd_w):.2f}× the reports at equal true dumping")
    print(f"  spread across the six readings: {t1['beta'].min():+.2f} … {t1['beta'].max():+.2f}")

    # ------------------------------------------------------------- torus-shift null
    print(f"\n[torus] {N_PERM} shifted w_R rasters, each given its own slope")
    z = np.load(os.path.join(OUT, "wR_raster.npz"))
    gx, gy, Z, res = z["x"], z["y"], np.where(np.isfinite(z["wR"]), z["wR"], 0.0), float(z["res"])
    ix = np.clip(np.round((sup["x"].to_numpy() - gx[0]) / res).astype(int), 0, len(gx) - 1)
    iy = np.clip(np.round((sup["y"].to_numpy() - gy[0]) / res).astype(int), 0, len(gy) - 1)
    y, logE = d["T"].to_numpy(float), np.log(d["exposure"].to_numpy(float))
    _, _, dev0 = poisson_fit(np.zeros((len(y), 0)), y, logE)
    d_real = dev0 - float(main_row["deviance"])
    null = []
    for p in range(N_PERM):
        sx, sy = int(rng.integers(0, len(gx))), int(rng.integers(0, len(gy)))
        wp = pd.Series(Z[(iy + sy) % len(gy), (ix + sx) % len(gx)]).groupby(sup["GEOID"].to_numpy()).mean()
        wp = d["GEOID"].map(wp).fillna(0.0).to_numpy()
        b, _, dev = poisson_fit(wp[:, None], y, logE)
        null.append({"shift": p, "beta": b[1], "deviance_gain": dev0 - dev})
    tn = pd.DataFrame(null); tn.to_csv(os.path.join(BETA, "torus_null.csv"), index=False)
    p_val = float((tn["deviance_gain"] >= d_real).mean())
    print(f"  real map: deviance gain {d_real:,.0f}; shifted maps: mean {tn['deviance_gain'].mean():,.0f}, "
          f"q95 {tn['deviance_gain'].quantile(.95):,.0f}, max {tn['deviance_gain'].max():,.0f}; slopes {tn['beta'].min():+.2f} … "
          f"{tn['beta'].max():+.2f}; p = {p_val:.2f}")

    # ------------------------------------------------------------- save
    d.to_csv(os.path.join(BETA, "cbg.csv"), index=False)
    summary = {"beta_hat": beta_hat, "ci_95": ci, "se_robust": float(main_row["se_robust"]), "sd_wbar": sd_w,
               "beta_x_sd": beta_hat * sd_w, "n_cbg": int(len(d)), "n_reports": int(d["T"].sum()), "window": BETA_WINDOW,
               "spread_min": float(t1["beta"].min()), "spread_max": float(t1["beta"].max()),
               "torus": {"n": N_PERM, "deviance_gain_real": d_real, "null_mean": float(tn["deviance_gain"].mean()),
                         "null_q95": float(tn["deviance_gain"].quantile(.95)), "null_max": float(tn["deviance_gain"].max()),
                         "p": p_val},
               "table1": t1.round(4).to_dict(orient="records")}
    with open(os.path.join(BETA, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    update_choices(beta_hat=beta_hat, ci_95=ci)
    _figure(d, t1, boot["main"], tn, d_real, beta_hat, ci, sd_w)
    print(f"\nDone in {time.time() - t_start:.0f} s. Outputs in output/beta/, figure fig1_beta.png")


def _figure(d, t1, boot_main, tn, d_real, beta_hat, ci, sd_w):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    ax.scatter(d["wbar"], d["log_rate"], s=5, color=C1, alpha=0.3)
    dec = pd.qcut(d["wbar"], 10, labels=False, duplicates="drop")
    ax.scatter(d["wbar"].groupby(dec).median(), d["log_rate"].groupby(dec).mean(), s=28, color=C2, zorder=3, label="decile means")
    grid = np.linspace(d["wbar"].quantile(.005), d["wbar"].quantile(.995), 50)
    a_hat = float(t1.loc[t1["spec"] == "main", "a"].iloc[0])
    ax.plot(grid, a_hat + beta_hat * grid, color=C2, lw=2, label=f"Poisson fit, β̂ = {beta_hat:.2f}")
    style(ax, f"Block groups ({len(d):,}): log report rate vs reporting propensity", "w̄_R (mean kernel w_R over the block group)",
          "log(reports / exposure)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax = axes[1]
    ax.hist(boot_main, bins=25, color=C1, alpha=0.7)
    ax.axvspan(ci[0], ci[1], color=C1, alpha=0.12, lw=0)
    for r in t1.itertuples():
        ax.axvline(r.beta, color=C2 if r.spec == "main" else MUTED, lw=1.6 if r.spec == "main" else 0.9,
                   ls="-" if r.spec == "main" else "--")
        ax.text(r.beta, ax.get_ylim()[1] * (0.95 - 0.07 * r.Index), f" {r.spec}", fontsize=7, color=C2 if r.spec == "main" else MUTED)
    ax.axvline(0, color=GRID, lw=1)
    style(ax, f"Block bootstrap of β̂ ({len(boot_main)} resamples) and the six readings", "β", "count")
    ax = axes[2]
    ax.hist(tn["deviance_gain"], bins=20, color=MUTED, alpha=0.7, label="torus-shifted w_R maps")
    ax.axvline(d_real, color=C2, lw=2, label=f"real map ({d_real:,.0f})")
    style(ax, "Deviance gain of the slope: real map vs shifted maps", "deviance gain", "count")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_beta.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
