"""
09_validate_divisions.py
========================
08's plain comparison once more, with the city's published Litter Index for
2023–2025 as the test instead of the 2022–23 survey visits, and 311 reports
from the same three years. Same readings, same figures; the unit is the city's
electoral division (about 1,700 polygons, the unit the index is published on).

Ground truth per division: the published streets score pooled over DIV_YEARS
(each year's DIV_SCORE weighted by DIV_N, i.e. the mean rating over all the
visits behind it), for divisions with at least VALID_MIN_VISITS visits in
total. No month or surveyor adjustment is possible — the file holds means, not
visits — so this is the city's number as published. Higher = dirtier.

311 per division: reports in the window inside the division per unit of
exposure, three versions as in 08 (as filed; corrected, e^{−β̂ w_R}; at the
survey's β), Gamma-shrunk log rates. Exposure = the SafeGraph stops of the
support cells inside the division (a block group's exposure spread evenly over
its cells, as everywhere else), w̄_R = the mean kernel w_R over those cells,
the bootstrap block = that of the block group under the division's
representative point.

Rows: the three years pooled (main; figures) and each year on its own (that
year's reports against that year's published score). 06's kernel map is rebuilt
for the pooled window and averaged per division for the comparison row.

The city's public visit table (SURVEY_PUBLIC; 2023 and 2024 only) allows 08's
own ground truth for those two years: the visits' ratings with month-of-sample
and surveyor effects removed by one OLS, averaged per division, the visits the
city omitted from its index (omit_flag = 1) dropped. Two more rows test the
2023–24 reports against that (adjusted, and the plain mean), next to the
published 2023–24 score on the same reports.

Reads   data/DIV_FILE, data/SURVEY_PUBLIC; output/{points_dumping, wR_events, support_grid, wR_support,
        cbg_blocks}.parquet, cbg_geom.geojson, choices.json, survey/summary.json
Writes  output/validate_divisions/{divisions.csv, table3d.csv, summary.json}
        output/figures/fig7_divisions.png, fig7b_divisions_before_after.png
"""

import os
import json
import time
import warnings
import importlib

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (DATA, OUT, FIG, SURVEY, VALID_DIV, CRS, ALPHA, RATE_PER, VALID_MIN_VISITS, VALID_TOP_SHARE,
                    N_BLOCKS, N_BOOT, SEED, DIV_FILE, DIV_YEARS, DIV_SCORE, DIV_N, SURVEY_PUBLIC, SURVEY_PUBLIC_YEARS,
                    SURVEY_PUBLIC_DEPT, C1, C2, C3, MUTED, GRID, INK, CMAP_DELTA, style, map_style, load_choices, ensure_dirs)
from kernels import shrunk_log, kernel_sum
from design import FE

warnings.filterwarnings("ignore")
VERSIONS = ["raw", "corr", "surv"]
LABEL = {"raw": "as filed", "corr": "corrected", "surv": "at the survey's β"}
MAIN = f"{DIV_YEARS[0]}-{DIV_YEARS[-1]}"
PUB = f"{SURVEY_PUBLIC_YEARS[0]}-{SURVEY_PUBLIC_YEARS[-1]}"
WINDOWS = {MAIN: (pd.Timestamp(f"{DIV_YEARS[0]}-01-01"), pd.Timestamp(f"{DIV_YEARS[-1] + 1}-01-01")),
           **{str(y): (pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in DIV_YEARS},
           PUB: (pd.Timestamp(f"{SURVEY_PUBLIC_YEARS[0]}-01-01"), pd.Timestamp(f"{SURVEY_PUBLIC_YEARS[-1] + 1}-01-01"))}
# the tests: (name, 311 window, ground-truth column) — the published score pooled (main; figures) and per year,
# then the visit table's years: the published score, the adjusted visit mean, the plain visit mean
MAIN_TEST = f"published {MAIN}"
TESTS = ([(MAIN_TEST, MAIN, "survey")] + [(f"published {y}", str(y), f"survey_{y}") for y in DIV_YEARS] +
         [(f"published {PUB}", PUB, "survey_pub"), (f"visits {PUB}, month + surveyor adjusted", PUB, "visits_adj"),
          (f"visits {PUB}, plain mean", PUB, "visits_mean")])


def pct_rank(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def agreement(d, col, scol, top=VALID_TOP_SHARE):
    """Spearman with the published score, dirtiest-tenth hit rate, mean absolute percentile-rank error."""
    rho = float(spearmanr(d[col], d[scol]).correlation)
    n_top = max(1, int(round(top * len(d))))
    top_s = set(d.nlargest(n_top, scol).index); top_r = set(d.nlargest(n_top, col).index)
    hit = len(top_s & top_r) / n_top
    err = np.abs(pct_rank(d[col]) - pct_rank(d[scol]))
    return rho, hit, float(err.mean()), err


def within(points: gpd.GeoDataFrame, polys: gpd.GeoDataFrame) -> pd.Series:
    """division_num of the polygon containing each point (NaN outside), one match per point, index kept."""
    j = gpd.sjoin(points, polys[["division_num", "geometry"]], predicate="within", how="left")
    return j[~j.index.duplicated(keep="first")]["division_num"]


def visit_ground_truth(geom):
    """08's ground truth from the city's public visit table: Streets visits of SURVEY_PUBLIC_YEARS, the city's
    omitted visits dropped, ratings with month-of-sample + surveyor effects removed by one OLS, averaged per
    division (NaN under VALID_MIN_VISITS visits). Positions are the city's corrected coordinates."""
    s = pd.read_csv(os.path.join(DATA, SURVEY_PUBLIC), low_memory=False)
    s = s[s["year"].isin(SURVEY_PUBLIC_YEARS) & s["department"].eq(SURVEY_PUBLIC_DEPT) & s["omit_flag"].ne(1)
          & s["litter_rating"].between(1, 4)].copy()
    s["ts"] = pd.to_datetime(s["survey_start"])                        # local time (download_litter_surveys.py)
    x = s["x_coord"].fillna(s["raw_x_coord"]); y = s["y_coord"].fillna(s["raw_y_coord"])
    pts = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(x, y), crs=4326).to_crs(CRS)
    s["division_num"] = within(pts, geom).reindex(s.index).to_numpy()
    s = s[s["division_num"].notna()].copy()
    s["month"] = pd.factorize(s["ts"].dt.year * 12 + s["ts"].dt.month, sort=True)[0]
    s["surv"] = pd.factorize(s["surveyor_id"])[0]
    r = s["litter_rating"].to_numpy(float)
    Xfe, _ = FE(s).build(s)
    X1 = np.c_[np.ones(len(r)), Xfe]
    fitted = X1 @ np.linalg.lstsq(X1, r, rcond=None)[0]
    s["adj"] = r - fitted + fitted.mean()
    out = s.groupby("division_num").agg(visits_adj=("adj", "mean"), visits_mean=("litter_rating", "mean"), n_visits=("adj", "size"))
    out.loc[out["n_visits"] < VALID_MIN_VISITS, ["visits_adj", "visits_mean"]] = np.nan
    info = {"n_visits": int(len(s)), "by_year": {int(k): int(v) for k, v in s["year"].value_counts().sort_index().items()},
            "n_surveyors": int(s["surv"].nunique()), "n_months": int(s["month"].nunique()),
            "fe_r2": float(1 - np.var(r - fitted) / np.var(r)), "mean_rating": float(r.mean())}
    return out, info


def main():
    t_start = time.time()
    print("=" * 68); print("09_validate_divisions.py — the published Litter Index 2023–25 as the test"); print("=" * 68)
    ensure_dirs()
    ch = load_choices(); HW0, H0, beta_hat = ch["HW0"], ch["H0"], float(ch["beta_hat"])
    beta_surv = float(json.load(open(os.path.join(SURVEY, "summary.json")))["beta_survey"])
    rng = np.random.default_rng(SEED)

    # ---- the divisions and their published scores ---------------------------------------------------
    nb = gpd.read_file(os.path.join(DATA, DIV_FILE)).to_crs(CRS)
    nb = nb[nb["year"].isin(DIV_YEARS)].copy()
    geom = nb.sort_values("year").drop_duplicates("division_num", keep="last")[["division_num", "geometry"]].reset_index(drop=True)
    ps = nb.pivot_table(index="division_num", columns="year", values=DIV_SCORE)
    pn = nb.pivot_table(index="division_num", columns="year", values=DIV_N)
    g = pd.DataFrame({"survey": (ps * pn).sum(1) / pn.sum(1), "n_blocks": pn.sum(1)})
    for y in DIV_YEARS:
        g[f"survey_{y}"] = ps[y].where(pn[y] >= VALID_MIN_VISITS)
        g[f"n_blocks_{y}"] = pn[y]
    g = g[g["n_blocks"] >= VALID_MIN_VISITS]
    ys = SURVEY_PUBLIC_YEARS
    g["survey_pub"] = ((ps[ys] * pn[ys]).sum(1) / pn[ys].sum(1)).where(pn[ys].sum(1) >= VALID_MIN_VISITS)
    print(f"  published index: {len(geom):,} divisions, {', '.join(f'{y}: {int((nb.year == y).sum()):,}' for y in DIV_YEARS)}; "
          f"{len(g):,} with ≥ {VALID_MIN_VISITS} visits over {MAIN} (median {int(g['n_blocks'].median())}); pooled {DIV_SCORE} "
          f"p10 / p50 / p90 = {g['survey'].quantile(.1):.2f} / {g['survey'].median():.2f} / {g['survey'].quantile(.9):.2f}; "
          f"per year " + ", ".join(f"{y}: {g[f'survey_{y}'].mean():.2f}" for y in DIV_YEARS) + " (means)")
    vt, vinfo = visit_ground_truth(geom)
    g = g.join(vt, how="left")
    print(f"  public visit table: {vinfo['n_visits']:,} {SURVEY_PUBLIC_DEPT} visits {PUB} after dropping the city's omitted ones "
          f"({', '.join(f'{k}: {v:,}' for k, v in vinfo['by_year'].items())}); {vinfo['n_surveyors']} surveyors, {vinfo['n_months']} "
          f"year-months, month + surveyor effects explain {vinfo['fe_r2']:.1%} of rating variance; {int(g['visits_adj'].notna().sum()):,} "
          f"divisions with ≥ {VALID_MIN_VISITS} visits (median {int(g['n_visits'].median())}); division Spearman of the adjusted "
          f"visit mean with the published {PUB} score {g['visits_adj'].corr(g['survey_pub'], method='spearman'):.3f} "
          f"(plain mean {g['visits_mean'].corr(g['survey_pub'], method='spearman'):.3f})")

    # ---- exposure, w̄_R and the bootstrap block per division ----------------------------------------
    sup = pd.read_parquet(os.path.join(OUT, "support_grid.parquet"))
    sup["wR"] = pd.read_parquet(os.path.join(OUT, "wR_support.parquet"))[f"wR_h{HW0}"].to_numpy()
    cells = gpd.GeoDataFrame(sup, geometry=gpd.points_from_xy(sup["x"], sup["y"]), crs=CRS)
    sup["division_num"] = within(cells, geom).reindex(sup.index).to_numpy()
    per = sup.dropna(subset=["division_num"]).groupby("division_num").agg(exposure=("E_cell", "sum"), wbar=("wR", "mean"), n_cells=("wR", "size"))
    cbg_geom = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson")); cbg_geom["GEOID"] = cbg_geom["GEOID"].astype(str)
    blocks = pd.read_parquet(os.path.join(OUT, "cbg_blocks.parquet")).set_index("GEOID")["block"]
    rep = geom.copy(); rep["geometry"] = geom.representative_point()
    rep = gpd.sjoin_nearest(rep, cbg_geom[["GEOID", "geometry"]], how="left")
    rep = rep[~rep.index.duplicated(keep="first")]
    block = rep.set_index("division_num")["GEOID"].map(blocks).rename("block")
    d = g.join(per, how="inner").join(block, how="inner")
    d = d[d["exposure"] > 0].copy()
    print(f"  support: {sup['division_num'].notna().mean():.1%} of cells inside a division; {len(d):,} divisions with exposure "
          f"(median {int(d['n_cells'].median())} cells); w̄_R sd {d['wbar'].std():.2f}")

    # ---- 311 per division: counts as filed and reweighted ------------------------------------------
    ev = pd.read_parquet(os.path.join(OUT, "points_dumping.parquet"))
    wr = pd.read_parquet(os.path.join(OUT, "wR_events.parquet"))[f"wR_h{HW0}"].to_numpy()
    wr = np.where(np.isfinite(wr), wr, 0.0)
    ev["m_corr"] = np.exp(-beta_hat * wr); ev["m_surv"] = np.exp(-beta_surv * wr); ev["m_raw"] = 1.0
    ev["division_num"] = within(gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev["x"], ev["y"]), crs=CRS), geom).reindex(ev.index).to_numpy()
    n_win = {}
    for win, (t0, t1) in WINDOWS.items():
        years = (t1 - t0).days / 365.25
        e = ev[(ev["ts"] >= t0) & (ev["ts"] < t1)]
        n_win[win] = int(len(e))
        s = e.groupby("division_num")[["m_raw", "m_corr", "m_surv"]].sum().reindex(d.index).fillna(0.0)
        for ver in VERSIONS:
            T = s[f"m_{ver}"].to_numpy(); E = d["exposure"].to_numpy()
            lam = T.sum() / E.sum()
            d[f"log_{ver}_{win}"] = shrunk_log(T, E, ALPHA, lam)[0] - np.log(lam)          # log rate relative to citywide
            d[f"rate_{ver}_{win}"] = np.exp(d[f"log_{ver}_{win}"])                           # multiple of the citywide rate
            d[f"abs_{ver}_{win}"] = np.exp(shrunk_log(T, E, ALPHA, lam)[0]) / years * RATE_PER   # reports per year per RATE_PER stops
            d[f"n_{ver}_{win}"] = T
    print(f"  311: {n_win[MAIN]:,} dumping reports {MAIN} ({ev.loc[(ev['ts'] >= WINDOWS[MAIN][0]), 'division_num'].notna().mean():.1%} inside a division); "
          f"β̂ = {beta_hat:.2f} (04), survey's own β = {beta_surv:.2f} (05); last report {ev['ts'].max().date()}")

    # 06's kernel map rebuilt for the pooled window, averaged per division (comparison row)
    surf = importlib.import_module("06_surface")
    sxy = sup[["x", "y"]].to_numpy()
    den = kernel_sum(sxy, sxy, H0, weights=sup["E_cell"].to_numpy(), source_tree=cKDTree(sxy))
    L = surf.long_term(ev[["x", "y"]].to_numpy(), ev["ts"], ev["t_days"].to_numpy(), np.exp(-beta_hat * wr), sxy, den, H0, *WINDOWS[MAIN])
    for ver in ("raw", "corr"):
        sup[f"map_{ver}"] = L[f"log_rel_{ver}"]
        d[f"map_{ver}_{MAIN}"] = sup.dropna(subset=["division_num"]).groupby("division_num")[f"map_{ver}"].mean().reindex(d.index)

    # ---- the readings --------------------------------------------------------------------------------
    rows, boot_rows = [], []
    blocks_arr = d["block"].to_numpy().astype(int)
    boot_w = [np.bincount(rng.integers(0, N_BLOCKS, N_BLOCKS), minlength=N_BLOCKS)[blocks_arr] for _ in range(N_BOOT)]
    err = {}
    for name, win, scol in TESTS:
        dw = d.dropna(subset=[scol])
        for ver in VERSIONS:
            col = f"log_{ver}_{win}"
            rho, hit, mae, err[(ver, name)] = agreement(dw, col, scol)
            rows.append({"test": name, "window": win, "ground_truth": scol, "version": ver, "source": "counts", "n_divisions": len(dw),
                         "n_reports": n_win[win], "spearman": rho, "top_tenth_hit": hit, "mean_abs_rank_error": mae})
        if name == MAIN_TEST:
            for ver in ("raw", "corr"):
                col = f"map_{ver}_{win}"
                rho, hit, mae, _ = agreement(dw.dropna(subset=[col]), col, scol)
                rows.append({"test": name, "window": win, "ground_truth": scol, "version": ver, "source": "kernel map (06)",
                             "n_divisions": int(dw[col].notna().sum()), "n_reports": n_win[win], "spearman": rho, "top_tenth_hit": hit,
                             "mean_abs_rank_error": mae})
        keep = d.index.get_indexer(dw.index)
        diffs, closer = [], []
        for w in boot_w:
            idx = np.repeat(np.arange(len(dw)), w[keep])
            dd = dw.iloc[idx]
            diffs.append(spearmanr(dd[f"log_corr_{win}"], dd[scol]).correlation - spearmanr(dd[f"log_raw_{win}"], dd[scol]).correlation)
            e_r = np.abs(pct_rank(dd[f"log_raw_{win}"]) - pct_rank(dd[scol])); e_c = np.abs(pct_rank(dd[f"log_corr_{win}"]) - pct_rank(dd[scol]))
            closer.append(float((e_c < e_r).mean()))
        moved = err[("raw", name)] - err[("corr", name)]
        rho_of = {r["version"]: r["spearman"] for r in rows if r["test"] == name and r["source"] == "counts"}
        boot_rows.append({"test": name, "window": win, "ground_truth": scol, "spearman_gain": rho_of["corr"] - rho_of["raw"],
                          "spearman_gain_lo": float(np.quantile(diffs, .025)), "spearman_gain_hi": float(np.quantile(diffs, .975)),
                          "share_closer": float((moved > 0).mean()), "share_closer_lo": float(np.quantile(closer, .025)),
                          "share_closer_hi": float(np.quantile(closer, .975)), "share_further": float((moved < 0).mean())})
        if name == MAIN_TEST:
            d.loc[dw.index, f"closer_{win}"] = moved
    t3 = pd.DataFrame(rows); b3 = pd.DataFrame(boot_rows)
    t3.to_csv(os.path.join(VALID_DIV, "table3d.csv"), index=False)
    print("\n[agreement with the published index, divisions]")
    for name, win, scol in TESTS:
        tw = t3[(t3["test"] == name) & (t3["source"] == "counts")].set_index("version"); bw = b3[b3["test"] == name].iloc[0]
        print(f"  {name} ({int(tw.loc['raw', 'n_divisions']):,} divisions, {n_win[win]:,} reports {win}): Spearman as filed "
              f"{tw.loc['raw', 'spearman']:.3f} → corrected {tw.loc['corr', 'spearman']:.3f} (gain {bw['spearman_gain']:+.3f}, 95% "
              f"[{bw['spearman_gain_lo']:+.3f}, {bw['spearman_gain_hi']:+.3f}]); at the survey's β {tw.loc['surv', 'spearman']:.3f}")
        print(f"            dirtiest tenth caught by 311's top tenth: {tw.loc['raw', 'top_tenth_hit']:.0%} → {tw.loc['corr', 'top_tenth_hit']:.0%} "
              f"({tw.loc['surv', 'top_tenth_hit']:.0%} at the survey's β); mean |rank error| {tw.loc['raw', 'mean_abs_rank_error']:.3f} → "
              f"{tw.loc['corr', 'mean_abs_rank_error']:.3f}; divisions closer {bw['share_closer']:.0%} [{bw['share_closer_lo']:.0%}, "
              f"{bw['share_closer_hi']:.0%}], further {bw['share_further']:.0%}")
        if name == MAIN_TEST:
            tm = t3[(t3["test"] == name) & (t3["source"] != "counts")].set_index("version")
            print(f"            06's kernel map instead of counts: {tm.loc['raw', 'spearman']:.3f} → {tm.loc['corr', 'spearman']:.3f}")

    # survey quintiles and w_R tertiles, pooled window
    win = MAIN
    d["survey_q"] = pd.qcut(d["survey"], 5, labels=False)
    d["wR_t"] = pd.qcut(d["wbar"], 3, labels=["quiet", "middle", "chatty"])
    q = d.groupby("survey_q").agg(survey=("survey", "mean"), raw=(f"rate_raw_{win}", "mean"), corr=(f"rate_corr_{win}", "mean"),
                                  surv=(f"rate_surv_{win}", "mean"), n=("survey", "size"))
    tert = d.groupby("wR_t").agg(wbar=("wbar", "mean"), survey=("survey", "mean"), raw=(f"log_raw_{win}", "mean"),
                                 corr=(f"log_corr_{win}", "mean"), surv=(f"log_surv_{win}", "mean"), n=("survey", "size"))
    print(f"\n[published-score quintiles, cleanest → dirtiest; mean 311 rate as a multiple of citywide, {win}]")
    for k, r in q.iterrows():
        print(f"  Q{k + 1} (score {r['survey']:.2f}, {int(r['n'])} divisions): as filed ×{r['raw']:.2f}, corrected ×{r['corr']:.2f}, at the survey's β ×{r['surv']:.2f}")
    print(f"  dirtiest / cleanest quintile ratio: as filed {q['raw'].iloc[-1] / q['raw'].iloc[0]:.2f}, corrected {q['corr'].iloc[-1] / q['corr'].iloc[0]:.2f}")
    print(f"\n[w_R tertiles, {win}: mean published score; mean log 311 rate relative to citywide]")
    for k, r in tert.iterrows():
        print(f"  {k:>6} (w̄_R {r['wbar']:+.2f}, {int(r['n'])} divisions): score {r['survey']:.2f}; 311 as filed {r['raw']:+.2f}, corrected {r['corr']:+.2f}, at the survey's β {r['surv']:+.2f}")
    for ver in VERSIONS:
        b, a = np.polyfit(d[f"log_{ver}_{win}"], d["survey"], 1)
        d[f"pred_{ver}"] = a + b * d[f"log_{ver}_{win}"]
        tert[f"pred_{ver}"] = d.groupby("wR_t")[f"pred_{ver}"].mean()
    gap = {"survey": tert.loc["chatty", "survey"] - tert.loc["quiet", "survey"],
           **{v_: tert.loc["chatty", f"pred_{v_}"] - tert.loc["quiet", f"pred_{v_}"] for v_ in VERSIONS}}
    print(f"  chatty − quiet gap in rating points: published {gap['survey']:+.3f}; 311 as filed {gap['raw']:+.3f}, corrected {gap['corr']:+.3f}, "
          f"at the survey's β {gap['surv']:+.3f}   (311 translated through its own line score ~ a + b·log rate)")

    for ver in ("raw", "corr"):
        d[f"misfit_{ver}"] = d[f"pred_{ver}"] - d["survey"]
    d["misfit_change"] = d["misfit_corr"].abs() - d["misfit_raw"].abs()
    mis = {"mean_abs_raw": float(d["misfit_raw"].abs().mean()), "mean_abs_corr": float(d["misfit_corr"].abs().mean()),
           "share_closer": float((d["misfit_change"] < 0).mean()), "share_further": float((d["misfit_change"] > 0).mean()),
           "slope_rating_per_log_raw": float(np.polyfit(d[f"log_raw_{win}"], d["survey"], 1)[0]),
           "slope_rating_per_log_corr": float(np.polyfit(d[f"log_corr_{win}"], d["survey"], 1)[0])}
    print(f"\n[misfit in rating points, {win}] mean |311 read as a score − published|: as filed {mis['mean_abs_raw']:.3f} → corrected "
          f"{mis['mean_abs_corr']:.3f}; divisions closer {mis['share_closer']:.0%}, further {mis['share_further']:.0%}; "
          f"one log unit of 311 rate = {mis['slope_rating_per_log_raw']:.3f} (as filed) / {mis['slope_rating_per_log_corr']:.3f} (corrected) rating points")

    # ---- save ---------------------------------------------------------------------------------------
    d.reset_index().to_csv(os.path.join(VALID_DIV, "divisions.csv"), index=False)
    summary = {"beta_hat": beta_hat, "beta_survey": beta_surv, "window": win, "years": DIV_YEARS, "score": DIV_SCORE, "n_divisions": int(len(d)),
               "n_reports": n_win, "min_visits": VALID_MIN_VISITS, "public_visits": vinfo,
               "visits_adj_vs_published_spearman": float(g["visits_adj"].corr(g["survey_pub"], method="spearman")),
               "agreement": t3.round(4).to_dict(orient="records"),
               "bootstrap": b3.round(4).to_dict(orient="records"), "survey_quintiles": q.round(4).reset_index().to_dict(orient="records"),
               "wR_tertiles": tert.round(4).reset_index().to_dict(orient="records"), "chatty_minus_quiet": gap, "misfit": mis}
    with open(os.path.join(VALID_DIV, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _figure(d, geom, q, tert, t3, b3, win, beta_hat)
    _figure_before_after(d, geom, surf, win, beta_hat, mis)
    print(f"\nDone in {time.time() - t_start:.0f} s. Outputs in output/validate_divisions/, figures fig7_divisions.png, fig7b_divisions_before_after.png")


def _figure(d, geom, q, tert, t3, b3, win, beta_hat):
    tw = t3[(t3["test"] == MAIN_TEST) & (t3["source"] == "counts")].set_index("version"); bw = b3[b3["test"] == MAIN_TEST].iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))

    # (A) published-score quintiles → 311 rate
    ax = axes[0, 0]
    x = np.arange(5)
    ax.plot(x, q["raw"], "-o", color=C1, lw=2, ms=7, label="311 as filed")
    ax.plot(x, q["corr"], "-s", color=C2, lw=2, ms=7, label=f"311 corrected (β̂ = {beta_hat:.2f})")
    ax.axhline(1, color=GRID, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"Q{k + 1}\ncleanest" if k == 0 else (f"Q{k + 1}\ndirtiest" if k == 4 else f"Q{k + 1}") for k in x])
    style(ax, f"(A) Divisions in published-index quintiles: how many 311 reports do they file? ({win})",
          "published streets score quintile (visits pooled over the three years)", "311 report rate, multiple of the citywide rate")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # (B) deciles → mean 311 percentile rank
    ax = axes[0, 1]
    dec = pd.qcut(d["survey"], 10, labels=False)
    xs = pct_rank(d["survey"])
    for ver, col, mk in (("raw", C1, "o"), ("corr", C2, "s")):
        r = pct_rank(d[f"log_{ver}_{win}"])
        gm = pd.Series(r).groupby(dec.to_numpy()).agg(["mean", "sem"]); gx = pd.Series(xs).groupby(dec.to_numpy()).mean()
        ax.errorbar(gx, gm["mean"], yerr=1.96 * gm["sem"], fmt="-" + mk, color=col, lw=2, ms=7, capsize=3,
                    label=f"{LABEL[ver]}: Spearman {tw.loc[ver, 'spearman']:.2f}, dirtiest tenth caught {tw.loc[ver, 'top_tenth_hit']:.0%}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect agreement")
    style(ax, f"(B) Published-score deciles: where does 311 rank them on average? ({len(d):,} divisions)",
          "published score percentile rank (0 = cleanest, 1 = dirtiest)", "mean 311 percentile rank, ± 95% of the mean")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # (C) w_R tertiles
    ax = axes[0, 2]
    x = np.arange(3); wdt = 0.26
    ax.bar(x - wdt, tert["survey"] - tert["survey"].mean(), wdt, color=C3, label="published score")
    ax.bar(x, tert["pred_raw"] - tert["pred_raw"].mean(), wdt, color=C1, label="311 as filed, read as a score")
    ax.bar(x + wdt, tert["pred_corr"] - tert["pred_corr"].mean(), wdt, color=C2, label="311 corrected, read as a score")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{k}\n(w̄_R {tert.loc[k, 'wbar']:+.2f})" for k in tert.index])
    style(ax, "(C) Quiet vs chatty divisions: the published score and 311 read as a score",
          "w_R tertile: quiet → chatty neighbourhoods", "rating points, deviation from the mean of the three tertiles")
    ax.legend(frameon=False, fontsize=9, loc="lower left")

    # (D, E, F) disagreement maps
    cols = {"D": ("dis_raw", "(D) 311 as filed − published score, percentile ranks"),
            "E": ("dis_corr", "(E) 311 corrected − published score, percentile ranks"),
            "F": (f"closer_{win}", "(F) the correction's effect on the disagreement")}
    ps = pct_rank(d["survey"])
    d = d.assign(dis_raw=pct_rank(d[f"log_raw_{win}"]) - ps, dis_corr=pct_rank(d[f"log_corr_{win}"]) - ps)
    m = geom.merge(d[["dis_raw", "dis_corr", f"closer_{win}"]].reset_index(), on="division_num", how="left")
    lim_d = float(np.nanquantile(np.abs(np.r_[m["dis_raw"], m["dis_corr"]]), 0.95)); lim_f = float(np.nanquantile(np.abs(m[f"closer_{win}"]), 0.95))
    for ax, key, cmap, lim, lab in ((axes[1, 0], "D", "RdBu_r", lim_d, "red = 311 ranks it dirtier than the index does, blue = cleaner"),
                                    (axes[1, 1], "E", "RdBu_r", lim_d, "red = 311 ranks it dirtier than the index does, blue = cleaner"),
                                    (axes[1, 2], "F", CMAP_DELTA, lim_f,
                                     f"|disagreement| as filed − corrected: teal = closer to the index ({bw['share_closer']:.0%}), brown = further ({bw['share_further']:.0%})")):
        col, title = cols[key]
        geom.plot(ax=ax, color="#f2f1ee", edgecolor="none")
        m.dropna(subset=[col]).plot(column=col, ax=ax, cmap=cmap, vmin=-lim, vmax=lim, edgecolor="none")
        mae = float(np.abs(m[col]).mean()) if key != "F" else None
        map_style(ax, geom, title + (f"   (mean |disagreement| {mae:.3f})" if mae is not None else ""))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-lim, lim))
        cb = plt.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.75, pad=0.08, aspect=35)
        cb.set_label(lab, fontsize=8, color=MUTED); cb.ax.tick_params(labelsize=8, colors=MUTED)
    fig.suptitle(f"The published Litter Index {win} (streets score pooled over the three years, {len(d):,} divisions with ≥ {VALID_MIN_VISITS} visits) "
                 f"against 311 dumping reports of the same years, as filed and corrected", x=0.01, ha="left", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(os.path.join(FIG, "fig7_divisions.png"), dpi=150); plt.close(fig)


def _figure_before_after(d, geom, surf, win, beta_hat, mis):
    """Two rows of division maps: the 311 rate as filed, corrected, the difference; the fit to the published score of each
    (311 read as a score minus the published score) and the change in the misfit. Same layout as fig6b."""
    cols = [f"log_raw_{win}", f"log_corr_{win}", f"abs_raw_{win}", f"abs_corr_{win}", "misfit_raw", "misfit_corr", "misfit_change"]
    m = geom.merge(d[cols].reset_index(), on="division_num", how="left")
    m["rate_delta"] = m[f"abs_corr_{win}"] - m[f"abs_raw_{win}"]
    r_lo, r_hi, r_ticks, r_labels = surf.rate_scale(d[f"log_raw_{win}"].to_numpy(), d[f"log_corr_{win}"].to_numpy())
    d_lim = surf.sym_lim(m["rate_delta"].dropna().to_numpy(), q=0.95)
    f_lim = surf.sym_lim(np.r_[m["misfit_raw"].dropna(), m["misfit_corr"].dropna()], q=0.95)
    c_lim = surf.sym_lim(m["misfit_change"].dropna().to_numpy(), q=0.95)
    lab_rate = "report rate as a multiple of the citywide rate   (blue = fewer, red = more)"
    lab_fit = "rating points   (red = 311 reads dirtier than the published index, blue = cleaner)"
    panels = [(f"log_raw_{win}", f"(A) 311 dumping reports as filed, {win}, per division", lab_rate, "RdBu_r", r_lo, r_hi, r_ticks, r_labels),
              (f"log_corr_{win}", f"(B) corrected for reporting propensity, β̂ = {beta_hat:.2f}", lab_rate, "RdBu_r", r_lo, r_hi, r_ticks, r_labels),
              ("rate_delta", "(C) the correction: corrected − as filed",
               f"reports per year per {RATE_PER:,.0f} SafeGraph stops   (brown = reports removed, teal = added)", CMAP_DELTA, -d_lim, d_lim, None, None),
              ("misfit_raw", f"(D) fit to the published index, as filed   (mean |misfit| {mis['mean_abs_raw']:.3f})", lab_fit, "RdBu_r", -f_lim, f_lim, None, None),
              ("misfit_corr", f"(E) fit to the published index, corrected   (mean |misfit| {mis['mean_abs_corr']:.3f})", lab_fit, "RdBu_r", -f_lim, f_lim, None, None),
              ("misfit_change", f"(F) change in the misfit: |E| − |D|   ({mis['share_closer']:.0%} of divisions closer, {mis['share_further']:.0%} further)",
               "rating points   (teal = closer to the index after the correction, brown = further)", CMAP_DELTA, -c_lim, c_lim, None, None)]
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
    fig.suptitle(f"Divisions with ≥ {VALID_MIN_VISITS} visits behind the published {win} index ({len(d):,}); 311 read as a score through its own "
                 f"straight line (one log unit of report rate = {mis['slope_rating_per_log_corr']:.2f} rating points); grey = too few visits",
                 x=0.01, ha="left", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(os.path.join(FIG, "fig7b_divisions_before_after.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
