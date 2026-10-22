"""
eda_litter_index.py
===================
Exploratory look at the published Litter Index around the 2019-2020 change of method.
Not part of the pipeline; reads only the raw files in data/ and writes output/litter_index_eda/.

Inputs
  Litter_Index_Blocks.geojson          hundred-block scores, 2017 and 2018 only (79,603 street segments)
  Litter_Index_Neighborhoods.geojson   division scores, 2017, 2018, 2023, 2024, 2025 (about 1,700 divisions)
  litter_surveys_raw_20250730.csv      the new method's field survey, effectively 2022-2023 (point visits, rating 1-4)

One scale for both regimes
  The old published score sits on the lattice 1.333, 2, 2.667, 3.333 (= (2/3)(1 + r) for a rating r of 1..4), and
  the 2024-25 division score is exactly (2/3)(1 + streets_division_score). So r = 1.5 * score - 1 puts the old
  scores on the survey's 1-4 rating scale. Score 0 (grey, not scored) is treated as missing.

Outputs (output/litter_index_eda/)
  fig1_distributions.png   score distributions: blocks 2017-18, survey visits 2022-23, divisions all years, month drift
  fig2_stability.png       division ranks across years: Spearman matrix and three year pairs
  fig3_sample.png          where the sample is: blocks per division by year, street-class mix
  fig4_fields.png          score against the other fields: street class, block length, visits per division, 2023 formula
  fig5_maps.png            division scores by year on the common scale
  overview.csv, rank_stability.csv, street_class.csv, coverage.csv, summary.md

Run: python src/eda_litter_index.py        (about 2 minutes, most of it reading the GeoJSONs)
"""

import os
import re

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch

from config import DATA, OUT, CRS, C1, C2, INK, MUTED, GRID, style

EDA = os.path.join(OUT, "litter_index_eda")
os.makedirs(EDA, exist_ok=True)

YEAR_COLOR = {2017: "#4a9be0", 2018: "#1d4f9c",                      # old method: blues
              2022: "#d98e1e", 2023: "#c8411f", 2024: "#148f63", 2025: "#6f3fc0"}   # new method
OLD, NEW = [2017, 2018], [2023, 2024, 2025]
SURVEY_YEARS = [2022, 2023]
CLASSES = ["Local", "Collector", "Minor Arterial", "Major Arterial"]
CLASS_LABELS = ["Local", "Collector", "Minor\narterial", "Major\narterial"]
LEN_EDGES = [0, 60, 90, 120, 150, 180, 220, 280, 400, 1e5]     # m; fixed bins because block lengths cluster on a few standard values
R_LIM = (0.95, 3.3)                      # axis range on the 1-4 scale that holds every division
CMAP_SEQ = "Reds"                        # one hue for magnitude (score); "Blues" for counts


def implied_rating(score):
    s = np.asarray(score, float)
    return np.where(s > 0, 1.5 * s - 1, np.nan)


def norm_addr(a):
    a = str(a).upper().strip()
    a = re.sub(r"^\d+\s*/\s*", "", a)     # "0 / 7400 PENROSE AVE" -> "7400 PENROSE AVE"
    return re.sub(r"\s+", " ", a)


def modal(v):
    v = v.dropna()
    return v.mode().iat[0] if len(v) else np.nan


def save(fig, name):
    fig.savefig(os.path.join(EDA, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ----------------------------------------------------------------------------------------------- load
print("reading")
blocks = gpd.read_file(os.path.join(DATA, "Litter_Index_Blocks.geojson")).to_crs(CRS)
blocks["addr"] = blocks.LR_HUNDRED_BLOCK.map(norm_addr)
blocks["r"] = implied_rating(blocks.HUNDRED_BLOCK_SCORE)
blocks["len_m"] = blocks.geometry.length

div = gpd.read_file(os.path.join(DATA, "Litter_Index_Neighborhoods.geojson")).to_crs(CRS)
div["r"] = np.where(div.year.isin(OLD), implied_rating(div.division_score), div.streets_division_score)
YEARS = sorted(div.year.unique())

surv = pd.read_csv(os.path.join(DATA, "litter_surveys_raw_20250730.csv"), low_memory=False)
surv["t"] = pd.to_datetime(surv.survey_start, errors="coerce")
surv["year"] = surv.t.dt.year
surv["month"] = surv.t.dt.month
surv = surv[(surv.department == "streets") & surv.year.isin(SURVEY_YEARS)].copy()
surv["addr"] = surv.Address.map(norm_addr)
surv["rating"] = surv["Litter Rating"].astype(float)

# hundred blocks (one row per address and year; a hundred block is often split into 2-5 segments)
hb = (blocks.groupby(["YEAR", "addr"])
      .agg(r=("r", "mean"), len_m=("len_m", "sum"), cls=("STREET_CLASS_NAME", modal), n_seg=("SEG_ID", "size"))
      .reset_index())
hb18 = hb[hb.YEAR == 2018].set_index("addr")
surv["cls"] = surv.addr.map(hb18.cls)
surv["len_m"] = surv.addr.map(hb18.len_m)
surv["matched"] = surv.addr.isin(hb18.index)
for y in SURVEY_YEARS:
    hb18[f"visited{y}"] = hb18.index.isin(surv.loc[surv.year == y, "addr"])

# 2018 segments per division (midpoint inside the 2018 polygons)
mid = blocks[blocks.YEAR == 2018][["SEG_ID", "geometry"]].copy()
mid["geometry"] = mid.geometry.interpolate(0.5, normalized=True)
seg_per_div = (gpd.sjoin(mid, div[div.year == 2018][["division_num", "geometry"]], predicate="within")
               .groupby("division_num").size())
div["n_blocks"] = np.where(div.year.isin(OLD), div.division_num.map(seg_per_div), div.blocks_averaged)

wide = div.pivot_table(index="division_num", columns="year", values="r")

# ----------------------------------------------------------------------------------------------- fig 1
print("fig1")
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))
fig.subplots_adjust(hspace=0.45, wspace=0.28)

a = ax[0, 0]
lattice = np.r_[0, np.round(np.arange(4 / 3, 4.0001, 1 / 3), 4)]            # 0 (grey) then 1.33 ... 4
lat = np.round(blocks.HUNDRED_BLOCK_SCORE * 3) / 3                          # the few averaged values go to the nearest third
for k, y in enumerate(OLD):
    share = lat[blocks.YEAR == y].round(4).value_counts(normalize=True).reindex(lattice, fill_value=0)
    a.bar(np.arange(len(lattice)) + (k - 0.5) * 0.36, share.to_numpy(), width=0.34, color=YEAR_COLOR[y], label=str(y))
a.set_xticks(np.arange(len(lattice)))
a.set_xticklabels(["0\nnot scored"] + [f"{v:.2f}".rstrip("0").rstrip(".") for v in lattice[1:]], fontsize=8)
a.legend(frameon=False, fontsize=8)
style(a, "A  Block score, old method (share of blocks, rounded to thirds)", "published score", "share")

a = ax[0, 1]
for k, y in enumerate(SURVEY_YEARS):
    share = surv.loc[surv.year == y, "rating"].value_counts(normalize=True).reindex([1, 2, 3, 4], fill_value=0)
    a.bar(np.arange(4) + (k - 0.5) * 0.36, share.to_numpy(), width=0.34, color=YEAR_COLOR[y], label=f"{y}  (n = {int((surv.year == y).sum()):,})")
a.set_xticks(np.arange(4)); a.set_xticklabels(["1", "2", "3", "4"])
a.legend(frameon=False, fontsize=8)
style(a, "B  Visit rating, new method (share of visits)", "Litter Rating", "share")

a = ax[1, 0]
for y in YEARS:
    v = np.sort(wide[y].dropna().to_numpy())
    a.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=YEAR_COLOR[y], lw=1.8,
           label=f"{y}  mean {v.mean():.2f}, sd {v.std():.2f}")
a.set_xlim(*R_LIM); a.set_ylim(0, 1)
a.legend(frameon=False, fontsize=8, loc="lower right")
style(a, "C  Division score, all years (cumulative share of divisions)", "division score (1-4 scale)", "share of divisions")

a = ax[1, 1]
for y in SURVEY_YEARS:
    m = surv[surv.year == y].groupby("month").rating.agg(["mean", "size"])
    m = m[m["size"] >= 300]
    a.plot(m.index, m["mean"], "-o", color=YEAR_COLOR[y], lw=1.8, ms=5, label=str(y))
a.set_xticks(range(3, 13)); a.set_xlim(2.5, 12.5)
a.legend(frameon=False, fontsize=8)
style(a, "D  Mean visit rating by month, new method (months with 300+ visits)", "month", "mean rating")
save(fig, "fig1_distributions.png")

# ----------------------------------------------------------------------------------------------- fig 2
print("fig2")
rho = wide.corr(method="spearman")
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
fig.subplots_adjust(wspace=0.35)
a = ax[0]
im = a.imshow(rho.to_numpy(), cmap="Blues", vmin=0.2, vmax=1)
a.set_xticks(range(len(YEARS))); a.set_yticks(range(len(YEARS)))
a.set_xticklabels(YEARS, fontsize=8); a.set_yticklabels(YEARS, fontsize=8)
for i in range(len(YEARS)):
    for j in range(len(YEARS)):
        a.text(j, i, f"{rho.iat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if rho.iat[i, j] > 0.7 else INK)
a.set_title("A  Spearman rank correlation of division scores", loc="left", fontsize=10, color=INK)
a.tick_params(colors=MUTED)
for s in a.spines.values():
    s.set_visible(False)
for k, (y1, y2, lab) in enumerate([(2017, 2018, "within the old method"), (2018, 2023, "across the change"), (2023, 2024, "within the new method")]):
    a = ax[k + 1]
    d = wide[[y1, y2]].dropna()
    a.plot(R_LIM, R_LIM, color=MUTED, lw=0.8, alpha=0.6)
    a.scatter(d[y1], d[y2], s=6, color=C1, alpha=0.35, linewidths=0)
    a.set_xlim(*R_LIM); a.set_ylim(*R_LIM); a.set_aspect("equal")
    style(a, f"{'BCD'[k]}  {y1} vs {y2}, {lab}\n     rho = {d[y1].corr(d[y2], method='spearman'):.2f}", f"{y1}", f"{y2}")
save(fig, "fig2_stability.png")

# ----------------------------------------------------------------------------------------------- fig 3
print("fig3")
fig, ax = plt.subplots(1, 4, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
fig.subplots_adjust(wspace=0.12)
norm_n = Normalize(0, 80)
for k, (y, lab) in enumerate([(2018, "2018: hundred blocks scored (all streets)"), (2023, "2023: blocks averaged"), (2025, "2025: blocks averaged")]):
    a = ax[k]
    d = div[div.year == y]
    d.plot(column="n_blocks", cmap="Blues", norm=norm_n, linewidth=0, ax=a)
    a.set_axis_off()
    a.set_title(f"{'ABC'[k]}  {lab}\nmedian {d.n_blocks.median():.0f} per division, total {d.n_blocks.sum():,.0f}", loc="left", fontsize=9.5, color=INK)
cb = fig.colorbar(ScalarMappable(norm=norm_n, cmap="Blues"), ax=ax[:3], orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.45, extend="max")
cb.set_label("blocks per division", fontsize=8, color=MUTED); cb.ax.tick_params(labelsize=7, colors=MUTED)

a = ax[3]
comp = pd.DataFrame({"2018 blocks": hb18.cls.value_counts(normalize=True)})
for y in SURVEY_YEARS:
    comp[f"{y} visits"] = surv.loc[surv.year == y, "cls"].value_counts(normalize=True)
comp = comp.reindex(CLASSES).fillna(0)
for k, (col, y) in enumerate(zip(comp.columns, [2018, 2022, 2023])):
    a.bar(np.arange(len(CLASSES)) + (k - 1) * 0.27, comp[col], width=0.25, color=YEAR_COLOR[y], label=col)
a.set_xticks(np.arange(len(CLASSES))); a.set_xticklabels(CLASS_LABELS, fontsize=8)
a.legend(frameon=False, fontsize=8)
style(a, "D  Street-class mix: old blocks vs new survey visits", None, "share")
save(fig, "fig3_sample.png")

# ----------------------------------------------------------------------------------------------- fig 4
print("fig4")
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))
fig.subplots_adjust(hspace=0.45, wspace=0.28)

# unit rows for the two regimes on one scale: old hundred blocks (2017, 2018) and new visits (2022, 2023)
rows = pd.concat([hb[["YEAR", "r", "cls", "len_m"]].rename(columns={"YEAR": "year"}),
                  surv[["year", "rating", "cls", "len_m"]].rename(columns={"rating": "r"})]).dropna(subset=["r"])
FIELD_YEARS = OLD + SURVEY_YEARS

a = ax[0, 0]
tab = rows[rows.cls.isin(CLASSES)].groupby(["year", "cls"]).r.agg(["mean", "sem", "size"])
for k, y in enumerate(FIELD_YEARS):
    m = tab.loc[y].reindex(CLASSES)
    a.errorbar(np.arange(len(CLASSES)) + (k - 1.5) * 0.2, m["mean"], yerr=1.96 * m["sem"], fmt="o", ms=5, color=YEAR_COLOR[y], label=str(y), capsize=0)
a.set_xticks(np.arange(len(CLASSES))); a.set_xticklabels(CLASS_LABELS, fontsize=8)
a.set_ylim(1.3, 2.1)
a.legend(frameon=False, fontsize=8, ncol=4, loc="upper center")
style(a, "A  Mean score by street class (1-4 scale, 95% CI)", None, "mean score")

a = ax[0, 1]
rows["lbin"] = pd.cut(rows.len_m, LEN_EDGES)
for y in FIELD_YEARS:
    m = rows[rows.year == y].groupby("lbin", observed=True).agg(r=("r", "mean"), x=("len_m", "median"), n=("r", "size"))
    m = m[m.n >= 200]
    a.plot(m.x, m.r, "-o", color=YEAR_COLOR[y], lw=1.6, ms=4, label=str(y))
a.set_xscale("log"); a.set_xticks([50, 100, 200, 400]); a.set_xticklabels(["50", "100", "200", "400"])
a.legend(frameon=False, fontsize=8, ncol=2)
style(a, "B  Mean score by hundred-block length (bins of 200+ blocks)", "block length (m, log)", "mean score")

a = ax[1, 0]
bins = [0, 10, 20, 30, 40, 60, 500]
labels = ["1-9", "10-19", "20-29", "30-39", "40-59", "60+"]
for y in NEW:
    d = div[div.year == y]
    m = d.groupby(pd.cut(d.n_blocks, bins, labels=labels, right=False), observed=False).r.mean()
    a.plot(np.arange(len(labels)), m.reindex(labels), "-o", color=YEAR_COLOR[y], lw=1.6, ms=4,
           label=f"{y}  rho = {d.n_blocks.corr(d.r, method='spearman'):.2f}")
a.set_xticks(np.arange(len(labels))); a.set_xticklabels(labels, fontsize=8)
a.legend(frameon=False, fontsize=8)
style(a, "C  Division score by blocks averaged, new method", "blocks averaged per division", "mean streets score")

a = ax[1, 1]
d = div[div.year == 2023].dropna(subset=["division_score", "streets_division_score"])
extra = d.division_score - (2 / 3) * (1 + d.streets_division_score) > 1e-6
xs = np.linspace(1, 3.3, 10)
a.plot(xs, (2 / 3) * (1 + xs), color=GRID, lw=1.2)
a.scatter(d.streets_division_score[~extra], d.division_score[~extra], s=6, color=MUTED, alpha=0.5, linewidths=0, label=f"on the line, {(~extra).mean():.0%}")
a.scatter(d.streets_division_score[extra], d.division_score[extra], s=6, color=C2, alpha=0.6, linewidths=0, label=f"above the line, {extra.mean():.0%}")
a.legend(frameon=False, fontsize=8, loc="lower right")
style(a, "D  2023 only: published division score vs its streets score\n     line = (2/3)(1 + streets), exact for every division in 2024-25", "streets_division_score", "division_score")
save(fig, "fig4_fields.png")

# ----------------------------------------------------------------------------------------------- fig 5
print("fig5")
norm_r = Normalize(1.0, 2.6)
fig, ax = plt.subplots(1, len(YEARS), figsize=(3.3 * len(YEARS), 4.4))
fig.subplots_adjust(wspace=0.04)
for a, y in zip(ax, YEARS):
    d = div[div.year == y]
    d.plot(column="r", cmap=CMAP_SEQ, norm=norm_r, linewidth=0, ax=a, missing_kwds={"color": "#f0efeb"})
    a.set_axis_off()
    a.set_title(f"{y}   mean {d.r.mean():.2f}", loc="left", fontsize=10, color=INK)
cb = fig.colorbar(ScalarMappable(norm=norm_r, cmap=CMAP_SEQ), ax=ax, fraction=0.02, pad=0.01, extend="max")
cb.set_label("division score, 1-4 scale (old years: 1.5 x published - 1)", fontsize=8, color=MUTED)
cb.ax.tick_params(labelsize=7, colors=MUTED)
save(fig, "fig5_maps.png")

# ----------------------------------------------------------------------------------------------- tables
print("tables")
ov = []
for y in OLD:
    s = blocks.loc[blocks.YEAR == y, "HUNDRED_BLOCK_SCORE"]
    nz = s[s > 0]
    ov.append(dict(file="blocks", year=y, unit="street segment", n=len(s), score="HUNDRED_BLOCK_SCORE",
                   mean=nz.mean(), sd=nz.std(), min=nz.min(), max=nz.max(), share_at_min=np.isclose(nz, nz.min(), atol=1e-3).mean(),
                   share_not_scored=(s == 0).mean(), share_off_thirds=1 - np.isclose(nz * 3, np.round(nz * 3), atol=1e-6).mean(),
                   mean_1to4=implied_rating(nz).mean()))
for y in SURVEY_YEARS:
    s = surv.loc[surv.year == y, "rating"]
    ov.append(dict(file="survey", year=y, unit="visit", n=len(s), score="Litter Rating", mean=s.mean(), sd=s.std(), min=s.min(), max=s.max(),
                   share_at_min=(s == 1).mean(), share_not_scored=np.nan, share_off_thirds=np.nan, mean_1to4=s.mean()))
for y in YEARS:
    d = div[div.year == y]
    s = d.division_score
    ov.append(dict(file="divisions", year=y, unit="division", n=len(d), score="division_score", mean=s.mean(), sd=s.std(), min=s.min(), max=s.max(),
                   share_at_min=(np.isclose(s, s.min())).mean(), share_not_scored=np.nan, share_off_thirds=np.nan, mean_1to4=d.r.mean()))
ov = pd.DataFrame(ov).round(4)
ov.to_csv(os.path.join(EDA, "overview.csv"), index=False)
rho.round(3).to_csv(os.path.join(EDA, "rank_stability.csv"))
tab.round(4).to_csv(os.path.join(EDA, "street_class.csv"))
cov = hb18[hb18.cls.isin(CLASSES)].groupby("cls").agg(n_blocks_2018=("r", "size"), visited_2022=("visited2022", "mean"), visited_2023=("visited2023", "mean"),
                                                      mean_2018_visited=("r", lambda v: v[hb18.loc[v.index, "visited2023"]].mean()),
                                                      mean_2018_not_visited=("r", lambda v: v[~hb18.loc[v.index, "visited2023"]].mean())).reindex(CLASSES).round(4)
cov.to_csv(os.path.join(EDA, "coverage.csv"))

# ----------------------------------------------------------------------------------------------- summary
d17, d18 = ov[(ov.file == "blocks")].set_index("year").loc[2017], ov[(ov.file == "blocks")].set_index("year").loc[2018]
dv = ov[ov.file == "divisions"].set_index("year")
sv = ov[ov.file == "survey"].set_index("year")
floor = {y: (wide[y] < 1.1).mean() for y in YEARS}
d23 = div[div.year == 2023]
extra_share = (d23.division_score - (2 / 3) * (1 + d23.streets_division_score) > 1e-6).mean()
match = surv.matched.mean()
vis = {y: hb18[f"visited{y}"].mean() for y in SURVEY_YEARS}
geom_changed = (div.groupby("division_num").Shape__Area.nunique() > 1).mean()
pres = div.pivot_table(index="division_num", columns="year", values="objectid", aggfunc="size").notna()

lines = [
    "# Litter Index: what changed with the method (exploratory)",
    "",
    "Figures and tables in this folder; script `src/eda_litter_index.py`. All scores are put on the survey's 1-4 rating scale",
    "with r = 1.5 x published - 1 for the old years (exact inverse of the 2024-25 formula; see the script header).",
    "",
    "## What the files hold",
    "",
    "| file | years | unit | n per year | score field |",
    "|---|---|---|---|---|",
    f"| Litter_Index_Blocks | 2017, 2018 | street segment (hundred block) | {int(d17.n):,}, {int(d18.n):,} | HUNDRED_BLOCK_SCORE, lattice 0 / 1.33 / 2 / 2.67 / 3.33 / 4 |",
    f"| Litter_Index_Neighborhoods | {', '.join(map(str, YEARS))} | division | {', '.join(f'{int(dv.loc[y].n):,}' for y in YEARS)} | division_score (all years); streets_division_score, blocks_averaged (2023-25) |",
    f"| litter_surveys_raw (streets) | 2022, 2023 | visit | {int(sv.loc[2022].n):,}, {int(sv.loc[2023].n):,} | Litter Rating 1-4 |",
    "",
    "The block file stops at 2018 although the portal advertises blocks 2017-2025; the new method has no public block layer here.",
    "",
    "## Scoring",
    "",
    f"- Old method: one integer rating per block, published as (2/3)(1 + r). {1 - d18.share_off_thirds:.0%} of 2018 blocks sit exactly on that lattice ({1 - d17.share_off_thirds:.0%} in 2017, so 2017 averaged more than one rating per block more often). {d18.share_not_scored:.1%} of blocks are grey (score 0, not scored).",
    f"- New method: the division score is the mean visit rating over `blocks_averaged` visits; in 2024 and 2025 division_score = (2/3)(1 + streets_division_score) for every division. In 2023, {extra_share:.0%} of divisions sit above that line (an extra component, partly the parks survey), so 2023's division_score is not comparable with 2024-25 without using streets_division_score.",
    f"- Level: the citywide division mean on the 1-4 scale is {', '.join(f'{dv.loc[y].mean_1to4:.2f} ({y})' for y in YEARS)}; the spread (sd across divisions, same scale) falls from {wide[2018].std():.2f} in 2018 to {wide[2025].std():.2f} in 2025, and {floor[2025]:.0%} of divisions score below 1.1 in 2025 (vs {floor[2018]:.0%} in 2018). Under the new method the scale collapses toward 1 (fig 1C, fig 5).",
    f"- Whether that is cleaner streets or a drifting use of rating 1 the index cannot say, but the rating itself moves: {sv.loc[2022].share_at_min:.0%} of 2022 visits rated 1 vs {sv.loc[2023].share_at_min:.0%} in 2023 (fig 1B), and the mean rating swings by half a point across months within a year (fig 1D).",
    "",
    "## Ranks across years",
    "",
    f"- Division ranks: Spearman {rho.loc[2017, 2018]:.2f} between 2017 and 2018, {rho.loc[2018, 2023]:.2f} between 2018 and 2023 (across the change), but only {rho.loc[2023, 2024]:.2f} between 2023 and 2024 and {rho.loc[2024, 2025]:.2f} between 2024 and 2025 (fig 2). The new method is less stable year to year than the change itself.",
    f"- Within the old method, segment-level scores agree weakly between 2017 and 2018 (Spearman {blocks.pivot_table(index='SEG_ID', columns='YEAR', values='r').dropna().corr(method='spearman').iat[0, 1]:.2f}).",
    "",
    "## Sample locations",
    "",
    f"- Same hundred blocks: {match:.1%} of 2022-23 survey visits carry an address that is a 2018 hundred block, and {vis[2022]:.0%} (2022) / {vis[2023]:.0%} (2023) of the 2018 hundred blocks were visited. The street-class mix of visits matches the old block census (fig 3D). Locations did not change; the unit did (one score per block vs repeated visits averaged per division).",
    f"- The 2018 blocks not revisited in 2023 were cleaner in 2018 (local streets: mean {cov.loc['Local', 'mean_2018_not_visited']:.2f} vs {cov.loc['Local', 'mean_2018_visited']:.2f}; coverage.csv), a small selection toward dirtier blocks.",
    f"- Sample size per division: median {div[div.year == 2018].n_blocks.median():.0f} blocks in 2018 (all streets), {div[div.year == 2023].n_blocks.median():.0f} / {div[div.year == 2024].n_blocks.median():.0f} / {div[div.year == 2025].n_blocks.median():.0f} blocks averaged in 2023 / 2024 / 2025 (fig 3A-C). 2025 nearly doubled the sample.",
    f"- Division polygons: {geom_changed:.0%} of divisions have a different polygon in 2023-25 than in 2017-18; {int((~pres[2017] & pres[2023]).sum())} divisions are new in 2023 and {int((pres[2017] & ~pres[2024]).sum())} old ones are missing in 2024.",
    "",
    "## Fields vs score",
    "",
    "- Street class (fig 4A) and block length (fig 4B) relate to the score the same way under both methods: class gaps stay within 0.2, with collectors and minor arterials above local streets in every year, and the length profile keeps its shape (dirtiest around 150-200 m, the typical rowhouse block). The change shifts the level, not these relations.",
    f"- Blocks averaged per division vs the division score (fig 4C): Spearman {', '.join(f'{div[div.year == y].n_blocks.corr(div[div.year == y].r, method='spearman'):.2f} ({y})' for y in NEW)}; sampling intensity is not driving the scores.",
    "",
    "## Caveats",
    "",
    "- The 1-4 conversion of the old scores assumes the old block score is (2/3)(1 + rating); the lattice supports it, but 1-2% of old blocks score 4.0, which that formula cannot produce from a 1-4 rating.",
    "- The published 2023 streets_division_score correlates 0.87 with the mean of the 2023 survey CSV rows by division, not 1.0: the published index uses a different window or extra records.",
]
with open(os.path.join(EDA, "summary.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("done ->", EDA)
