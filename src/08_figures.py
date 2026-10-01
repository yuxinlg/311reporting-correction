"""
08_figures.py
=============
Publication figure set and the headline table, assembled from the saved
outputs of 00-07 (no model fitting here except none). Skips any figure whose
inputs are not present yet.

  fig1_bandwidth.png       Spearman vs h for the three field forms + CBG benchmarks
  fig2_reporting.png       kernel w_R surface and its agreement with the CBG posterior
  fig3_profile.png         profile likelihood curves, bootstrap distribution, 05a band
  fig4_spec_curve.png      beta-hat across every specification (triage OLS, ordinal S1',
                           Mundlak, profile variants), 05a band
  fig5_cv.png              out-of-fold ll by spec and h; block-level calibration
  tables.md                headline numbers in one place

Outputs  output/figures/fig*.png, output/figures/tables.md
"""

import os
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
H0, TAU0 = 800, 120


def style(ax, title=None, xlabel=None, ylabel=None, grid_axis="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def have(*paths):
    return all(os.path.exists(os.path.join(OUT, p)) for p in paths)


def fig1_bandwidth():
    if not have("field_sweep.csv", "field_summary.json"):
        return
    sw = pd.read_csv(os.path.join(OUT, "field_sweep.csv"))
    with open(os.path.join(OUT, "field_summary.json")) as f:
        fs = json.load(f)
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for form, col, label in [("abs", C1, "absolute density"), ("support", C3, "edge-corrected absolute"),
                             ("expo", C2, "ratio: ÷ kernel-smoothed exposure")]:
        s = sw[(sw["form"] == form) & (sw["tau"] == TAU0)].sort_values("h")
        ax.plot(s["h"], s["spearman_raw"], "-o", color=col, lw=2, ms=5, label=label)
    bench = fs.get("cbg_benchmarks", {})
    for label, ls in [("own CBG", ":"), ("own + 1st-order neighbours", "--")]:
        if label in bench:
            ax.axhline(bench[label]["ratio"], color=MUTED, lw=1.2, ls=ls, label=f"CBG polygon, {label} (ratio)")
    hs = sorted(sw["h"].unique())
    ax.set_xscale("log"); ax.set_xticks(hs); ax.set_xticklabels([str(int(h)) for h in hs])
    sec = ax.secondary_xaxis("top", functions=(lambda h: h / np.sqrt(7), lambda s: s * np.sqrt(7)))
    sec.set_xlabel("kernel sd (m)", color=MUTED, fontsize=9); sec.tick_params(colors=MUTED, labelsize=8)
    style(ax, f"Agreement with the Litter Index by spatial bandwidth (Spearman, τ = {TAU0} d)",
          "quartic kernel radius h (m)", "Spearman ρ with the 1–4 rating", grid_axis="y")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_bandwidth.png"), dpi=200); plt.close(fig)


def fig2_reporting():
    if not have("wR_raster.npz", "reporting_cbg.parquet", "cbg_geom.geojson"):
        return
    z = np.load(os.path.join(OUT, "wR_raster.npz"), allow_pickle=True)
    Z, gx, gy, prim = z["wR"], z["x"], z["y"], str(z["variant"])
    cbg = pd.read_parquet(os.path.join(OUT, "reporting_cbg.parquet"))
    geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    im = ax.imshow(Z, origin="lower", extent=[gx[0], gx[-1], gy[0], gy[-1]], cmap="Blues",
                   vmin=np.nanquantile(Z, .02), vmax=np.nanquantile(Z, .98))
    geom.boundary.plot(ax=ax, color=GRID, linewidth=0.15)
    ax.set_axis_off()
    ax.set_title("Reporting propensity ŵ_R: log outage reports per pole-day (2021–23), kernel 600 m, centred",
                 fontsize=9.5, loc="left", color=INK)
    plt.colorbar(im, ax=ax, shrink=0.65)
    ax = axes[1]
    c = cbg[cbg["in_fit_sample"]]
    ok = np.isfinite(c[prim]) & np.isfinite(c["w_R_mean"])
    ax.scatter(c.loc[ok, "w_R_mean"], c.loc[ok, prim], s=7, color=C1, alpha=0.5)
    lo, hi = c.loc[ok, "w_R_mean"].min(), c.loc[ok, "w_R_mean"].max()
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    r = np.corrcoef(c.loc[ok, "w_R_mean"], c.loc[ok, prim])[0, 1]
    style(ax, f"CBG posterior (two-equation model) vs kernel surface at the centroid, r = {r:.2f}",
          "CBG posterior mean w_R", "kernel ŵ_R")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_reporting.png"), dpi=200); plt.close(fig)


def fig3_profile():
    if not have("calib/profile_curves.csv", "calib/bootstrap.csv", "calib/summary.json", "reporting_summary.json"):
        return
    pc = pd.read_csv(os.path.join(OUT, "calib", "profile_curves.csv"))
    boot = pd.read_csv(os.path.join(OUT, "calib", "bootstrap.csv"))
    with open(os.path.join(OUT, "calib", "summary.json")) as f:
        cal = json.load(f)
    with open(os.path.join(OUT, "reporting_summary.json")) as f:
        rep = json.load(f)["stage1a"]["production"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    for col, color, ls, label in [(f"h{H0}_k_spline", C2, "-", f"h = {H0}, kernel ŵ_R, spline g (headline)"),
                                  (f"h{H0}_c_spline", C1, "-", f"h = {H0}, CBG posterior w_R"),
                                  (f"h{H0}_k_linear", C2, "--", f"h = {H0}, linear g"),
                                  ("h500_k_spline", MUTED, ":", "h = 500"), ("h1200_k_spline", MUTED, "-.", "h = 1200")]:
        if col in pc:
            ax.plot(pc["beta"], pc[col] - pc[col].max(), color=color, ls=ls, lw=1.8, label=label)
    ax.axvspan(rep["beta_lo"], rep["beta_hi"], color=GRID, alpha=0.7, lw=0, label="05a two-equation β₀, 94% interval")
    ax.axvline(rep["beta"], color=MUTED, lw=1, ls="--")
    ci = cal["bootstrap"]["profile_ci_recalibrated"]
    ax.axvspan(ci[0], ci[1], color=C2, alpha=0.12, lw=0, label="profile 95% CI, block-bootstrap calibrated")
    ax.set_ylim(-60, 2)
    style(ax, "Profile log-likelihood for the reporting elasticity β", "β", "ℓ(β) − max ℓ", grid_axis="y")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax = axes[1]
    ax.hist(boot["beta_prof"], bins=24, color=C2, alpha=0.75, label="profile β̂")
    ax.hist(boot["beta_S1"], bins=24, color=C1, alpha=0.45, label="closed-form −c/b")
    ax.axvspan(rep["beta_lo"], rep["beta_hi"], color=GRID, alpha=0.7, lw=0)
    ax.axvline(rep["beta"], color=MUTED, lw=1, ls="--")
    style(ax, f"Spatial block bootstrap ({len(boot)} resamples of 40 blocks)", "β̂", "resamples", grid_axis="y")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_profile.png"), dpi=200); plt.close(fig)


def fig4_spec_curve():
    if not have("calib/ladder.csv", "calib/profile_summary.csv", "reporting_summary.json"):
        return
    rows = []
    lad = pd.read_csv(os.path.join(OUT, "calib", "ladder.csv"))
    for r in lad.itertuples():
        rows.append({"family": "ordinal S1′ −c/b", "label": f"S1′ kernel w̄, h={r.h}", "beta": r.S1_k_beta_hat,
                     "lo": r.S1_k_fieller_lo, "hi": r.S1_k_fieller_hi})
        rows.append({"family": "ordinal S1′ −c/b", "label": f"S1′ CBG w̄, h={r.h}", "beta": r.S1_c_beta_hat,
                     "lo": r.S1_c_fieller_lo, "hi": r.S1_c_fieller_hi})
        rows.append({"family": "ordinal S1′ −c/b", "label": f"Mundlak, h={r.h}", "beta": r.S1M_k_beta_hat,
                     "lo": r.S1M_k_beta_hat - 1.96 * r.S1M_k_se_beta, "hi": r.S1M_k_beta_hat + 1.96 * r.S1M_k_se_beta})
    prof = pd.read_csv(os.path.join(OUT, "calib", "profile_summary.csv"))
    with open(os.path.join(OUT, "calib", "summary.json")) as f:
        cal = json.load(f)
    ci = cal["bootstrap"]["profile_ci_recalibrated"]
    for r in prof.itertuples():
        head = (r.h == H0 and r.w_R == "k" and r.g == "spline" and r.fe == "month+surv" and float(r.alpha) == 1.0)
        rows.append({"family": "profile likelihood", "label": f"profile h={r.h} {r.w_R} {r.g} {r.fe} α={r.alpha}",
                     "beta": r.beta_hat, "lo": ci[0] if head else np.nan, "hi": ci[1] if head else np.nan, "head": head})
    if have("triage/t6_spec_curve.csv"):
        tri = pd.read_csv(os.path.join(OUT, "triage", "t6_spec_curve.csv"))
        for r in tri.itertuples():
            rows.append({"family": "OLS on the raw rating (triage)", "label": f"OLS {r.fe} {r.w} h={r.h}",
                         "beta": r.beta_hat, "lo": r.beta_hat - 1.96 * r.se_beta, "hi": r.beta_hat + 1.96 * r.se_beta})
    df = pd.DataFrame(rows).sort_values("beta").reset_index(drop=True)
    df.to_csv(os.path.join(FIG, "fig4_spec_curve_data.csv"), index=False)
    with open(os.path.join(OUT, "reporting_summary.json")) as f:
        rep = json.load(f)["stage1a"]["production"]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.axhspan(rep["beta_lo"], rep["beta_hi"], color=GRID, alpha=0.8, lw=0, label="05a two-equation β₀ (94%)")
    ax.axhline(rep["beta"], color=MUTED, lw=1, ls="--")
    fam_col = {"OLS on the raw rating (triage)": C1, "ordinal S1′ −c/b": C3, "profile likelihood": C2}
    for fam, col in fam_col.items():
        s = df[df["family"] == fam]
        ok = np.isfinite(s["lo"])
        ax.errorbar(s.index[ok], s.loc[ok, "beta"], yerr=[s.loc[ok, "beta"] - s.loc[ok, "lo"], s.loc[ok, "hi"] - s.loc[ok, "beta"]],
                    fmt="o", color=col, ms=3.5, lw=0.8, capsize=0, alpha=0.9, label=fam)
        ax.plot(s.index[~ok], s.loc[~ok, "beta"], "o", color=col, ms=3.5, alpha=0.9)
    if "head" in df:
        hd = df[df.get("head", False) == True]
        ax.plot(hd.index, hd["beta"], "s", color=INK, ms=7, mfc="none", mew=1.5, label="headline (profile, h=800, kernel, spline)")
    style(ax, "β̂ across specifications, sorted (bars: Fieller / delta / bootstrap-calibrated 95% intervals)",
          "specification rank", "β̂", grid_axis="y")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_spec_curve.png"), dpi=200); plt.close(fig)


def fig5_cv():
    if not have("cv/outer_summary.csv", "cv/block_calibration_S0.csv", "cv/block_calibration_S2k.csv"):
        return
    agg = pd.read_csv(os.path.join(OUT, "cv", "outer_summary.csv"))
    b0 = pd.read_csv(os.path.join(OUT, "cv", "block_calibration_S0.csv"))
    b2 = pd.read_csv(os.path.join(OUT, "cv", "block_calibration_S2k.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    for spec, col, mk, label in [("S0", C1, "o", "raw field S0"), ("S2k", C2, "s", "corrected S2 (kernel ŵ_R)"),
                                 ("S2c", C3, "^", "corrected S2 (CBG w_R)"), ("S2kfit", C4, "d", "corrected, β fit in fold")]:
        s = agg[(agg["spec"] == spec) & (agg["tau"] == TAU0)].sort_values("h")
        if len(s):
            ax.plot(s["h"], s["ll_per_visit"], "-" + mk, color=col, lw=1.8, ms=5, label=label)
    fe = agg[agg["spec"] == "FE"]
    if len(fe):
        ax.axhline(fe["ll_per_visit"].iloc[0], color=MUTED, lw=1, ls=":", label="month + surveyor only")
    hs = sorted(agg["h"].unique()); ax.set_xscale("log"); ax.set_xticks(hs); ax.set_xticklabels([str(int(h)) for h in hs])
    style(ax, "Out-of-fold ordinal log-likelihood per visit", "h (m)", "ℓ / visit", grid_axis="y")
    ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    for spec, col, mk in [("S2k", C2, "s"), ("S2c", C3, "^"), ("S1", MUTED, "d"), ("S2kfit", C4, "o")]:
        s = agg[(agg["spec"] == spec) & (agg["tau"] == TAU0)].sort_values("h")
        if len(s):
            ax.errorbar(s["h"], s["dll_vs_S0_per_visit"], yerr=1.96 * s["dll_vs_S0_se"], fmt="-" + mk, color=col, ms=5, lw=1.5, capsize=2, label=spec)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xscale("log"); ax.set_xticks(hs); ax.set_xticklabels([str(int(h)) for h in hs])
    style(ax, "Gain over the raw field, paired by fold (± 1.96 fold-SE)", "h (m)", "Δℓ / visit vs S0", grid_axis="y")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    ax.scatter(b0["pred"], b0["obs"], color=C1, s=20, alpha=0.8, label="S0")
    ax.scatter(b2["pred"], b2["obs"], color=C2, s=20, marker="s", alpha=0.8, label="S2 (kernel)")
    lo = min(b0["pred"].min(), b2["pred"].min(), b0["obs"].min()); hi = max(b0["pred"].max(), b2["pred"].max(), b0["obs"].max())
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
    style(ax, f"Block-level calibration, 40 held-out blocks (h = {H0})", "predicted mean rating", "observed mean rating")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_cv.png"), dpi=200); plt.close(fig)


def tables():
    lines = ["# Headline numbers (assembled by 08_figures.py)\n"]
    if have("reporting_summary.json"):
        with open(os.path.join(OUT, "reporting_summary.json")) as f:
            r = json.load(f)["stage1a"]
        p, g = r["production"], r["regression"]
        lines += ["## Stage 1a", f"- regression test vs 05a: {'PASS' if r['regression_pass'] else 'FAIL'} "
                  f"(β {g['beta']:.4f} [{g['beta_lo']:.4f}, {g['beta_hi']:.4f}])",
                  f"- production β₀ = {p['beta']:.3f} [{p['beta_lo']:.3f}, {p['beta_hi']:.3f}], n_eff {p['health']['beta']['n_eff_min']:.0f}, R̂ {p['health']['beta']['r_hat']:.3f}\n"]
    if have("field_summary.json"):
        with open(os.path.join(OUT, "field_summary.json")) as f:
            fs = json.load(f)
        lines += ["## Stage 2", f"- best field: {fs['sweep_best']}", f"- CBG benchmarks: {json.dumps(fs.get('cbg_benchmarks', {}))}",
                  f"- ρ(h): {fs['rho_h']}\n"]
    if have("calib/summary.json"):
        with open(os.path.join(OUT, "calib", "summary.json")) as f:
            c = json.load(f)
        b = c["bootstrap"]
        head = [r for r in c["profile"] if r["h"] == H0 and r["w_R"] == "k" and r["g"] == "spline" and r["fe"] == "month+surv" and float(r.get("alpha", 1)) == 1.0][0]
        lines += ["## Stage 3", f"- headline profile β̂ = {head['beta_hat']:.3f}; naive χ² CI [{head['ci_lo']:.3f}, {head['ci_hi']:.3f}]; "
                  f"block-bootstrap calibrated CI [{b['profile_ci_recalibrated'][0]:.3f}, {b['profile_ci_recalibrated'][1]:.3f}] "
                  f"(LR critical value {b['lr_crit_95_bootstrap']:.1f} vs χ² 3.84)",
                  f"- bootstrap sd: profile {b['prof_beta_sd']:.3f}, S1′ {b['S1_beta_sd']:.3f}"]
        for r in c["ladder"]:
            lines.append(f"- h={r['h']}: Δℓ(S2−S0) kernel {r['dll_S2_S0_kernel']:+.0f} / CBG {r['dll_S2_S0_cbg']:+.0f}; "
                         f"S1′ β̂ {r['S1_k_beta_hat']:.3f} Fieller [{r['S1_k_fieller_lo']:.2f}, {r['S1_k_fieller_hi']:.2f}]")
        ctrl = c.get("controls", {})
        if ctrl:
            pn = ctrl["permutation_null"]
            lines += [f"- permutation null: actual Δℓ {pn['dll_actual_S2_S0']:+.0f} vs torus-shift null mean {pn['dll_null_mean']:+.0f} "
                      f"(q95 {pn['dll_null_q95']:+.0f}), p = {pn['p_value']:.2f}",
                      f"- outage placebo b = {ctrl['outage_placebo']['b_outage']:+.3f} (se {ctrl['outage_placebo']['se']:.3f}) vs dumping {ctrl['outage_placebo']['b_dumping_h500']:+.3f}",
                      f"- NLS 1+3σ(η) spans {ctrl['nls_sigmoid']['share_of_1_4_range_spanned_p1_p99']:.0%} of 1–4; cutpoints {np.round(ctrl['nls_sigmoid']['cutpoints_S0lin'], 2).tolist()}",
                      f"- α sensitivity (S1′ β̂): " + ", ".join(f"α={a['alpha']}: {a['beta_hat']:.2f}" for a in ctrl["alpha_sensitivity"]) + "\n"]
    if have("cv/summary.json"):
        with open(os.path.join(OUT, "cv", "summary.json")) as f:
            cv = json.load(f)
        lines += ["## Validation", f"- nested OOF ℓ/visit {cv['nested']['ll_per_visit']:.5f} (picks {cv['nested']['picks_spec']}, h {cv['nested']['picks_h']}); "
                  f"fixed default {cv['nested']['fixed_default_ll_per_visit']:.5f}",
                  f"- S2k (h, τ) surface best {cv['surface']['best']}, 1-SE choice {cv['surface']['one_se_choice']}",
                  f"- block calibration: {cv['block_calibration']}", f"- α out of fold: {cv.get('alpha', {}).get('best')}\n"]
    if have("surface/summary.json"):
        with open(os.path.join(OUT, "surface", "summary.json")) as f:
            s = json.load(f)
        lines += ["## Surface", f"- β̂ {s['beta_hat']:.3f}, h {s['h']}, cells {s['n_cells']:,}"]
        for d, r in s["dates"].items():
            lines.append(f"- {d}: E[rating] p1/p50/p99 {np.round(r['corr']['E_rating_p1_p50_p99'], 2).tolist()}, "
                         f"correction Δ sd {r['correction_delta_sd']:.3f}, CBG Spearman obs vs corr {r['cbg_spearman_obs_vs_E_corr']:.3f} (raw {r['cbg_spearman_obs_vs_E_raw']:.3f})")
    with open(os.path.join(FIG, "tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    for fn in [fig1_bandwidth, fig2_reporting, fig3_profile, fig4_spec_curve, fig5_cv, tables]:
        try:
            fn(); print(f"  {fn.__name__}: ok")
        except Exception as e:                      # keep going; report what failed
            print(f"  {fn.__name__}: FAILED ({type(e).__name__}: {e})")
    print(f"Figures in {os.path.relpath(FIG, ROOT)}/")


if __name__ == "__main__":
    main()
