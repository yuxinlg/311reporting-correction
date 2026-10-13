"""
08_validate.py
==============
The plain comparison: do the corrected 311 reports agree with what the Litter
Index survey saw on the ground, and do they agree better than the reports as
filed? No rating model, no kernel: block groups, counts and ranks.

Ground truth per block group: the mean survey rating of its own visits, after
month-of-sample and surveyor effects are removed by one OLS on the survey
(06's adjustment), for block groups with at least VALID_MIN_VISITS visits.
Higher = dirtier.

311 per block group, reports in VALID_WINDOW (the survey years; 2021–25 as a
second row) per unit of SafeGraph exposure, three versions:
  as filed     every report counts 1
  corrected    every report counts exp(−β̂ w_R) with β̂ from 04
  survey β     the same at the survey's own β from 05 (how far 0.62 is from
               what the survey would have wanted; table only)
Log rates are Gamma-shrunk toward the citywide rate as everywhere else.

Four readings, each for as filed vs corrected:
  1. Block groups in survey quintiles (cleanest → dirtiest): the mean 311 rate
     per quintile. Dirtier per the survey should mean more reports, and the
     corrected line should climb more steadily.
  2. Rank agreement: Spearman of the 311 log rate with the survey rating; the
     corrected − as filed difference with a 40-block bootstrap interval. Also
     the "dirtiest tenth" hit rate: of the block groups the survey puts in its
     top VALID_TOP_SHARE, the share that 311 also puts in its top tenth.
  3. Block group by block group: the percentile rank of a block group in 311
     against its percentile rank in the survey. The correction moved it
     closer to the survey or further away; the share moved closer and a map
     of where.
  4. Why it works: block groups in w_R tertiles (quiet / middle / chatty). As
     filed, the chatty tertile reports far more than the survey says it is
     dirtier; after the correction the 311 gap between chatty and quiet
     should match the survey's gap.
The same Spearman is also given for 06's kernel maps (the deliverable rasters
averaged per block group) so the map and the counts can be compared.

Second figure, the before / after view: the 311 rate per block group as filed,
corrected, and the difference (top row); how each version fits the survey —
311 read as a rating through its own straight line minus what the survey
found, in rating points — and the change in that misfit (bottom row).

Writes  output/validate/{cbg.csv, table3.csv, summary.json}
        output/figures/fig6_validation.png, fig6b_before_after.png
"""

import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (OUT, FIG, BETA, SURVEY, SURF, VALID, ALPHA, LONG_WINDOWS, RATE_PER, VALID_WINDOW, VALID_MIN_VISITS,
                    VALID_TOP_SHARE, N_BLOCKS, N_BOOT, SEED, C1, C2, C3, MUTED, GRID, INK, CMAP_DELTA, style, map_style,
                    load_choices, ensure_dirs)
from kernels import shrunk_log
from design import FE

warnings.filterwarnings("ignore")
VERSIONS = ["raw", "corr", "surv"]
LABEL = {"raw": "as filed", "corr": "corrected", "surv": "at the survey's β"}


def pct_rank(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def agreement(d, col, top=VALID_TOP_SHARE):
    """Spearman with the survey, dirtiest-tenth hit rate, and the mean absolute percentile-rank error."""
    rho = float(spearmanr(d[col], d["survey"]).correlation)
    n_top = max(1, int(round(top * len(d))))
    top_s = set(d.nlargest(n_top, "survey").index); top_r = set(d.nlargest(n_top, col).index)
    hit = len(top_s & top_r) / n_top
    err = np.abs(pct_rank(d[col]) - pct_rank(d["survey"]))
    return rho, hit, float(err.mean()), err


def main():
    t_start = time.time()
    print("=" * 68); print("08_validate.py — do the corrected reports agree with the survey?"); print("=" * 68)
    ensure_dirs()
    ch = load_choices(); HW0, beta_hat = ch["HW0"], float(ch["beta_hat"])
    beta_surv = float(json.load(open(os.path.join(SURVEY, "summary.json")))["beta_survey"])
    rng = np.random.default_rng(SEED)

    # ---- ground truth: the block group's own visits, month + surveyor effects removed ----------------
    v = pd.read_parquet(os.path.join(OUT, "visits.parquet"))
    y = v["rating"].to_numpy(float)
    Xfe, _ = FE(v).build(v)
    X1 = np.c_[np.ones(len(y)), Xfe]
    fitted = X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
    v["adj"] = y - fitted + fitted.mean()
    g = v.groupby("GEOID").agg(survey=("adj", "mean"), rating_mean=("rating", "mean"), n_visits=("adj", "size"))
    g = g[g["n_visits"] >= VALID_MIN_VISITS]
    print(f"  ground truth: {len(g):,} block groups with ≥ {VALID_MIN_VISITS} visits (median {int(g['n_visits'].median())}); "
          f"adjusted mean rating p10 / p50 / p90 = {g['survey'].quantile(.1):.2f} / {g['survey'].median():.2f} / {g['survey'].quantile(.9):.2f}")

    # ---- 311 per block group: counts as filed and reweighted ---------------------------------------
    cbg = pd.read_parquet(os.path.join(OUT, "cbg.parquet")).set_index("GEOID")
    bg = pd.read_csv(os.path.join(BETA, "cbg.csv"), dtype={"GEOID": str}).set_index("GEOID")     # wbar, block
    ev = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    wr = pd.read_parquet(os.path.join(OUT, "wR_events.parquet"))[f"wR_h{HW0}"].to_numpy()
    wr = np.where(np.isfinite(wr), wr, 0.0)
    ev["m_corr"] = np.exp(-beta_hat * wr); ev["m_surv"] = np.exp(-beta_surv * wr); ev["m_raw"] = 1.0
    d = g.join(cbg[["exposure"]]).join(bg[["wbar", "block"]], how="inner")
    d = d[d["exposure"] > 0].copy()
    tables = {}
    for win in [VALID_WINDOW] + [w for w in LONG_WINDOWS if w != VALID_WINDOW]:
        t0, t1 = LONG_WINDOWS[win]; years = (t1 - t0).days / 365.25
        e = ev[(ev["ts"] >= t0) & (ev["ts"] < t1)]
        s = e.groupby("GEOID")[["m_raw", "m_corr", "m_surv"]].sum().reindex(d.index).fillna(0.0)
        for ver in VERSIONS:
            T = s[f"m_{ver}"].to_numpy(); E = d["exposure"].to_numpy()
            lam = T.sum() / E.sum()
            d[f"log_{ver}_{win}"] = shrunk_log(T, E, ALPHA, lam)[0] - np.log(lam)          # log rate relative to citywide
            d[f"rate_{ver}_{win}"] = np.exp(d[f"log_{ver}_{win}"])                           # multiple of the citywide rate
            d[f"abs_{ver}_{win}"] = np.exp(shrunk_log(T, E, ALPHA, lam)[0]) / years * RATE_PER   # reports per year per RATE_PER stops
            d[f"n_{ver}_{win}"] = T
    # 06's kernel maps averaged per block group, for the comparison row
    sc = pd.read_csv(os.path.join(SURF, "surface_cbg.csv"), dtype={"GEOID": str}).set_index("GEOID")
    for win in LONG_WINDOWS:
        d[f"map_raw_{win}"] = sc[f"log_rel_raw_{win}"].reindex(d.index); d[f"map_corr_{win}"] = sc[f"log_rel_corr_{win}"].reindex(d.index)
    print(f"  311: {len(d):,} block groups; β̂ = {beta_hat:.2f} (04), survey's own β = {beta_surv:.2f} (05)")

    # ---- the readings --------------------------------------------------------------------------------
    rows, boot_rows = [], []
    blocks = d["block"].to_numpy().astype(int)
    boot_w = [np.bincount(rng.integers(0, N_BLOCKS, N_BLOCKS), minlength=N_BLOCKS)[blocks] for _ in range(N_BOOT)]
    err = {}
    for win in LONG_WINDOWS:
        for ver in VERSIONS:
            col = f"log_{ver}_{win}"
            rho, hit, mae, err[(ver, win)] = agreement(d, col)
            rows.append({"window": win, "version": ver, "source": "counts", "spearman": rho, "top_tenth_hit": hit,
                         "mean_abs_rank_error": mae})
        for ver in ("raw", "corr"):
            col = f"map_{ver}_{win}"
            rho, hit, mae, _ = agreement(d.dropna(subset=[col]), col)
            rows.append({"window": win, "version": ver, "source": "kernel map (06)", "spearman": rho, "top_tenth_hit": hit,
                         "mean_abs_rank_error": mae})
        # bootstrap of corrected − as filed (Spearman and share closer), blocks resampled
        diffs, closer = [], []
        for w in boot_w:
            idx = np.repeat(np.arange(len(d)), w)
            dd = d.iloc[idx]
            diffs.append(spearmanr(dd[f"log_corr_{win}"], dd["survey"]).correlation - spearmanr(dd[f"log_raw_{win}"], dd["survey"]).correlation)
            e_r = np.abs(pct_rank(dd[f"log_raw_{win}"]) - pct_rank(dd["survey"])); e_c = np.abs(pct_rank(dd[f"log_corr_{win}"]) - pct_rank(dd["survey"]))
            closer.append(float((e_c < e_r).mean()))
        moved = err[("raw", win)] - err[("corr", win)]
        rho_of = {r["version"]: r["spearman"] for r in rows if r["window"] == win and r["source"] == "counts"}
        boot_rows.append({"window": win, "spearman_gain": rho_of["corr"] - rho_of["raw"],
                          "spearman_gain_lo": float(np.quantile(diffs, .025)), "spearman_gain_hi": float(np.quantile(diffs, .975)),
                          "share_closer": float((moved > 0).mean()), "share_closer_lo": float(np.quantile(closer, .025)),
                          "share_closer_hi": float(np.quantile(closer, .975)), "share_further": float((moved < 0).mean())})
        d[f"closer_{win}"] = moved
    t3 = pd.DataFrame(rows); b3 = pd.DataFrame(boot_rows)
    t3.to_csv(os.path.join(VALID, "table3.csv"), index=False)
    print("\n[agreement with the survey, block groups]")
    for win in LONG_WINDOWS:
        tw = t3[(t3["window"] == win) & (t3["source"] == "counts")].set_index("version"); bw = b3[b3["window"] == win].iloc[0]
        print(f"  {win}: Spearman as filed {tw.loc['raw', 'spearman']:.3f} → corrected {tw.loc['corr', 'spearman']:.3f} "
              f"(gain {bw['spearman_gain']:+.3f}, 95% [{bw['spearman_gain_lo']:+.3f}, {bw['spearman_gain_hi']:+.3f}]); at the survey's β {tw.loc['surv', 'spearman']:.3f}")
        print(f"            dirtiest tenth caught by 311's top tenth: {tw.loc['raw', 'top_tenth_hit']:.0%} → {tw.loc['corr', 'top_tenth_hit']:.0%} "
              f"({tw.loc['surv', 'top_tenth_hit']:.0%} at the survey's β); mean |rank error| {tw.loc['raw', 'mean_abs_rank_error']:.3f} → {tw.loc['corr', 'mean_abs_rank_error']:.3f}")
        print(f"            block groups moved closer to the survey's rank: {bw['share_closer']:.0%} [{bw['share_closer_lo']:.0%}, {bw['share_closer_hi']:.0%}], "
              f"further: {bw['share_further']:.0%}")
        tm = t3[(t3["window"] == win) & (t3["source"] != "counts")].set_index("version")
        print(f"            06's kernel map instead of counts: {tm.loc['raw', 'spearman']:.3f} → {tm.loc['corr', 'spearman']:.3f}")

    # survey quintiles and w_R tertiles, main window
    win = VALID_WINDOW
    d["survey_q"] = pd.qcut(d["survey"], 5, labels=False)
    d["wR_t"] = pd.qcut(d["wbar"], 3, labels=["quiet", "middle", "chatty"])
    q = d.groupby("survey_q").agg(survey=("survey", "mean"), raw=(f"rate_raw_{win}", "mean"), corr=(f"rate_corr_{win}", "mean"),
                                  surv=(f"rate_surv_{win}", "mean"), n=("survey", "size"))
    tert = d.groupby("wR_t").agg(wbar=("wbar", "mean"), survey=("survey", "mean"), raw=(f"log_raw_{win}", "mean"),
                                 corr=(f"log_corr_{win}", "mean"), surv=(f"log_surv_{win}", "mean"), n=("survey", "size"))
    print(f"\n[survey quintiles, cleanest → dirtiest; mean 311 rate as a multiple of citywide, {win}]")
    for k, r in q.iterrows():
        print(f"  Q{k + 1} (rating {r['survey']:.2f}, {int(r['n'])} bg): as filed ×{r['raw']:.2f}, corrected ×{r['corr']:.2f}, at the survey's β ×{r['surv']:.2f}")
    print(f"  dirtiest / cleanest quintile ratio: as filed {q['raw'].iloc[-1] / q['raw'].iloc[0]:.2f}, corrected {q['corr'].iloc[-1] / q['corr'].iloc[0]:.2f}")
    print(f"\n[w_R tertiles, {win}: mean survey rating; mean log 311 rate relative to citywide]")
    for k, r in tert.iterrows():
        print(f"  {k:>6} (w̄_R {r['wbar']:+.2f}, {int(r['n'])} bg): survey {r['survey']:.2f}; 311 as filed {r['raw']:+.2f}, corrected {r['corr']:+.2f}, at the survey's β {r['surv']:+.2f}")
    # each version translated into rating points by its own straight line through the block groups
    # (rating ~ a + b · log rate), so the three can be read on the survey's scale
    for ver in VERSIONS:
        b, a = np.polyfit(d[f"log_{ver}_{win}"], d["survey"], 1)
        d[f"pred_{ver}"] = a + b * d[f"log_{ver}_{win}"]
        tert[f"pred_{ver}"] = d.groupby("wR_t")[f"pred_{ver}"].mean()
    gap = {"survey": tert.loc["chatty", "survey"] - tert.loc["quiet", "survey"],
           **{v_: tert.loc["chatty", f"pred_{v_}"] - tert.loc["quiet", f"pred_{v_}"] for v_ in VERSIONS}}
    print(f"  chatty − quiet gap in rating points: survey {gap['survey']:+.3f}; 311 as filed {gap['raw']:+.3f}, corrected {gap['corr']:+.3f}, "
          f"at the survey's β {gap['surv']:+.3f}   (311 translated through its own line rating ~ a + b·log rate)")

    # misfit per block group in rating points: 311 read as a rating minus what the survey found
    for ver in ("raw", "corr"):
        d[f"misfit_{ver}"] = d[f"pred_{ver}"] - d["survey"]
    d["misfit_change"] = d["misfit_corr"].abs() - d["misfit_raw"].abs()
    mis = {"mean_abs_raw": float(d["misfit_raw"].abs().mean()), "mean_abs_corr": float(d["misfit_corr"].abs().mean()),
           "share_closer": float((d["misfit_change"] < 0).mean()), "share_further": float((d["misfit_change"] > 0).mean()),
           "slope_rating_per_log_raw": float(np.polyfit(d[f"log_raw_{win}"], d["survey"], 1)[0]),
           "slope_rating_per_log_corr": float(np.polyfit(d[f"log_corr_{win}"], d["survey"], 1)[0])}
    print(f"\n[misfit in rating points, {win}] mean |311 read as a rating − survey|: as filed {mis['mean_abs_raw']:.3f} → corrected "
          f"{mis['mean_abs_corr']:.3f}; block groups closer {mis['share_closer']:.0%}, further {mis['share_further']:.0%}; "
          f"one log unit of 311 rate = {mis['slope_rating_per_log_raw']:.3f} (as filed) / {mis['slope_rating_per_log_corr']:.3f} (corrected) rating points")

    # ---- save ---------------------------------------------------------------------------------------
    d.reset_index().to_csv(os.path.join(VALID, "cbg.csv"), index=False)
    summary = {"beta_hat": beta_hat, "beta_survey": beta_surv, "window": win, "n_cbg": int(len(d)), "min_visits": VALID_MIN_VISITS,
               "agreement": t3.round(4).to_dict(orient="records"), "bootstrap": b3.round(4).to_dict(orient="records"),
               "survey_quintiles": q.round(4).reset_index().to_dict(orient="records"),
               "wR_tertiles": tert.round(4).reset_index().to_dict(orient="records"), "chatty_minus_quiet": gap, "misfit": mis}
    with open(os.path.join(VALID, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _figure(d, q, tert, t3, b3, win, beta_hat)
    _figure_before_after(d, win, beta_hat, mis)
    print(f"\nDone in {time.time() - t_start:.0f} s. Outputs in output/validate/, figures fig6_validation.png, fig6b_before_after.png")


def _figure_before_after(d, win, beta_hat, mis):
    """Two rows of block-group maps. Top: the 311 report rate as filed, corrected, and the difference. Bottom: how
    each version fits the survey — 311 read as a rating minus what the survey found — and the change in the misfit."""
    import importlib
    surf = importlib.import_module("06_surface")                  # rate_scale, sym_lim: the same colour rules as the maps
    geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson")); geom["GEOID"] = geom["GEOID"].astype(str)
    cols = [f"log_raw_{win}", f"log_corr_{win}", f"abs_raw_{win}", f"abs_corr_{win}", "misfit_raw", "misfit_corr", "misfit_change"]
    m = geom.merge(d[cols].reset_index(), on="GEOID", how="left")
    m["rate_delta"] = m[f"abs_corr_{win}"] - m[f"abs_raw_{win}"]
    r_lo, r_hi, r_ticks, r_labels = surf.rate_scale(d[f"log_raw_{win}"].to_numpy(), d[f"log_corr_{win}"].to_numpy())
    d_lim = surf.sym_lim(m["rate_delta"].dropna().to_numpy(), q=0.95)
    f_lim = surf.sym_lim(np.r_[m["misfit_raw"].dropna(), m["misfit_corr"].dropna()], q=0.95)
    c_lim = surf.sym_lim(m["misfit_change"].dropna().to_numpy(), q=0.95)
    lab_rate = "report rate as a multiple of the citywide rate   (blue = fewer, red = more)"
    lab_fit = "rating points   (red = 311 reads dirtier than the survey found, blue = cleaner)"
    panels = [(f"log_raw_{win}", f"(A) 311 dumping reports as filed, {win}, per block group", lab_rate, "RdBu_r", r_lo, r_hi, r_ticks, r_labels),
              (f"log_corr_{win}", f"(B) corrected for reporting propensity, β̂ = {beta_hat:.2f}", lab_rate, "RdBu_r", r_lo, r_hi, r_ticks, r_labels),
              ("rate_delta", "(C) the correction: corrected − as filed",
               f"reports per year per {RATE_PER:,.0f} SafeGraph stops   (brown = reports removed, teal = added)", CMAP_DELTA, -d_lim, d_lim, None, None),
              ("misfit_raw", f"(D) fit to the survey, as filed   (mean |misfit| {mis['mean_abs_raw']:.3f})", lab_fit, "RdBu_r", -f_lim, f_lim, None, None),
              ("misfit_corr", f"(E) fit to the survey, corrected   (mean |misfit| {mis['mean_abs_corr']:.3f})", lab_fit, "RdBu_r", -f_lim, f_lim, None, None),
              ("misfit_change", f"(F) change in the misfit: |E| − |D|   ({mis['share_closer']:.0%} of block groups closer, {mis['share_further']:.0%} further)",
               "rating points   (teal = closer to the survey after the correction, brown = further)", CMAP_DELTA, -c_lim, c_lim, None, None)]
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    for ax, (col, title, lab, cmap, lo, hi, ticks, labels) in zip(axes.ravel(), panels):
        geom.plot(ax=ax, color="#f2f1ee", edgecolor="none")
        m.dropna(subset=[col]).plot(column=col, ax=ax, cmap=cmap, vmin=lo, vmax=hi, edgecolor="none")
        map_style(ax, geom, title)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
        cb = plt.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.75, pad=0.08, aspect=35)
        if ticks is not None:
            cb.set_ticks(ticks); cb.set_ticklabels(labels)
        cb.set_label(lab, fontsize=8, color=MUTED); cb.ax.tick_params(labelsize=8, colors=MUTED)
    fig.suptitle(f"Block groups with ≥ {VALID_MIN_VISITS} survey visits ({len(d):,}); 311 read as a rating through its own straight line "
                 f"(one log unit of report rate = {mis['slope_rating_per_log_corr']:.2f} rating points); grey = too few visits",
                 x=0.01, ha="left", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(os.path.join(FIG, "fig6b_before_after.png"), dpi=150); plt.close(fig)


def _figure(d, q, tert, t3, b3, win, beta_hat):
    tw = t3[(t3["window"] == win) & (t3["source"] == "counts")].set_index("version"); bw = b3[b3["window"] == win].iloc[0]
    geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson")); geom["GEOID"] = geom["GEOID"].astype(str)
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))

    # (A) survey quintiles → 311 rate
    ax = axes[0, 0]
    x = np.arange(5)
    ax.plot(x, q["raw"], "-o", color=C1, lw=2, ms=7, label="311 as filed")
    ax.plot(x, q["corr"], "-s", color=C2, lw=2, ms=7, label=f"311 corrected (β̂ = {beta_hat:.2f})")
    ax.axhline(1, color=GRID, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"Q{k + 1}\ncleanest" if k == 0 else (f"Q{k + 1}\ndirtiest" if k == 4 else f"Q{k + 1}") for k in x])
    style(ax, f"(A) Block groups in survey quintiles: how many 311 reports do they file? ({win})",
          "survey rating quintile (month + surveyor adjusted)", "311 report rate, multiple of the citywide rate")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # (B) survey deciles → mean 311 percentile rank, against the diagonal
    ax = axes[0, 1]
    dec = pd.qcut(d["survey"], 10, labels=False)
    xs = pct_rank(d["survey"])
    for ver, col, mk in (("raw", C1, "o"), ("corr", C2, "s")):
        r = pct_rank(d[f"log_{ver}_{win}"])
        gm = pd.Series(r).groupby(dec.to_numpy()).agg(["mean", "sem"]); gx = pd.Series(xs).groupby(dec.to_numpy()).mean()
        ax.errorbar(gx, gm["mean"], yerr=1.96 * gm["sem"], fmt="-" + mk, color=col, lw=2, ms=7, capsize=3,
                    label=f"{LABEL[ver]}: Spearman {tw.loc[ver, 'spearman']:.2f}, dirtiest tenth caught {tw.loc[ver, 'top_tenth_hit']:.0%}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect agreement")
    style(ax, f"(B) Survey deciles: where does 311 rank them on average? ({len(d):,} block groups)",
          "survey percentile rank (0 = cleanest, 1 = dirtiest)", "mean 311 percentile rank, ± 95% of the mean")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # (C) w_R tertiles: survey vs 311, all in rating points
    ax = axes[0, 2]
    x = np.arange(3); wdt = 0.26
    ax.bar(x - wdt, tert["survey"] - tert["survey"].mean(), wdt, color=C3, label="survey rating")
    ax.bar(x, tert["pred_raw"] - tert["pred_raw"].mean(), wdt, color=C1, label="311 as filed, read as a rating")
    ax.bar(x + wdt, tert["pred_corr"] - tert["pred_corr"].mean(), wdt, color=C2, label="311 corrected, read as a rating")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{k}\n(w̄_R {tert.loc[k, 'wbar']:+.2f})" for k in tert.index])
    style(ax, "(C) Quiet vs chatty block groups: the survey's rating and 311 read as a rating",
          "w_R tertile: quiet → chatty neighbourhoods", "rating points, deviation from the mean of the three tertiles")
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    # (D, E) where 311 disagrees with the survey, as filed and corrected; (F) the change
    cols = {"D": (f"dis_raw", "(D) 311 as filed − survey, percentile ranks"),
            "E": (f"dis_corr", "(E) 311 corrected − survey, percentile ranks"),
            "F": (f"closer_{win}", "(F) the correction's effect on the disagreement")}
    ps = pct_rank(d["survey"])
    d = d.assign(dis_raw=pct_rank(d[f"log_raw_{win}"]) - ps, dis_corr=pct_rank(d[f"log_corr_{win}"]) - ps)
    m = geom.merge(d[["dis_raw", "dis_corr", f"closer_{win}"]].reset_index(), on="GEOID", how="left")
    lim_d = float(np.nanquantile(np.abs(np.r_[m["dis_raw"], m["dis_corr"]]), 0.95)); lim_f = float(np.nanquantile(np.abs(m[f"closer_{win}"]), 0.95))
    for ax, key, cmap, lim, lab in ((axes[1, 0], "D", "RdBu_r", lim_d, "red = 311 ranks it dirtier than the survey does, blue = cleaner"),
                                    (axes[1, 1], "E", "RdBu_r", lim_d, "red = 311 ranks it dirtier than the survey does, blue = cleaner"),
                                    (axes[1, 2], "F", CMAP_DELTA, lim_f,
                                     f"|disagreement| as filed − corrected: teal = closer to the survey ({bw['share_closer']:.0%}), brown = further ({bw['share_further']:.0%})")):
        col, title = cols[key]
        geom.plot(ax=ax, color="#f2f1ee", edgecolor="none")
        m.dropna(subset=[col]).plot(column=col, ax=ax, cmap=cmap, vmin=-lim, vmax=lim, edgecolor="none")
        mae = float(np.abs(m[col]).mean()) if key != "F" else None
        map_style(ax, geom, title + (f"   (mean |disagreement| {mae:.3f})" if mae is not None else ""))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim, lim))
        cb = plt.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.75, pad=0.08, aspect=35)
        cb.set_label(lab, fontsize=8, color=MUTED); cb.ax.tick_params(labelsize=8, colors=MUTED)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_validation.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
