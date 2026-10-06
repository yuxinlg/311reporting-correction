"""
03_litter_panel.py
==================
Survey-level rows -> the model table every calibration reads.

Filters (from the plan and the data README)
  - department == "streets" (PPR's 1,767 rows survey parks; both the dumping
    process and the reporting channel differ there; pwd* rows are hydrants)
  - survey year in {2022, 2023} (2020: 288 rows, 2024: 79 -- dropped)
  - Litter Rating in 1..4
  - a location_id (42 rows lack one) and a Philadelphia CBG (89 fall outside)

Columns
  x, y (m, EPSG:26918), lon, lat, ts, t_days, year, ym (year*12+month),
  rating (1-4), bulk (any dumping-type token), casual (any casual token),
  n_types, types (raw string), location_id, surveyor, GEOID,
  integer codes loc / surv / month / cbg / block,
  exposure, area_km2, log_expo_dens,
  w_R_cbg, w_R_cbg_sd            production posterior from 02 (Stage 1a)
  wR_<variant>_h<hw>             kernel surfaces from 02 at the survey point
  every CBG covariate from cov_cbg (for S4; never with the beta extraction)

Spatial blocks for blocked CV: the 1,335 fitted CBGs are cut into N_BLOCKS
contiguous k-d rectangles with roughly equal survey counts (weighted median
bisection on CBG centroids). Saved per CBG in output/cbg_blocks.parquet.

Outputs
  output/litter_panel.parquet, output/cbg_blocks.parquet
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")

CRS = 26918
EPOCH = pd.Timestamp("2021-01-01")
YEARS = [2022, 2023]
DEPT = "streets"
N_BLOCKS = 40

BULK_TOKENS = {"trash_bags", "construction", "oversized", "furniture", "tires", "vehicular"}
CASUAL_TOKENS = {"takeout_beverage", "bottles_cans", "paper_product", "food_wrappers",
                 "food", "plastic_bags", "textile", "cigarette_butts", "glass"}

COV_DROP = {"GEOID", "alloc_avg_s_cnt", "alloc_avg_d_cnt", "reporting_rate", "exposure",
            "n_poles", "n_led", "led_share", "n_peco", "peco_share", "n_pre24", "pre24_share",
            "in_fit_sample"}


def codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s)[0].astype(np.int64)


def kd_blocks(xy: np.ndarray, w: np.ndarray, n_blocks: int) -> np.ndarray:
    """Weighted k-d bisection: contiguous rectangles with ~equal total weight."""
    ids = np.zeros(len(xy), int)
    counter = [0]

    def rec(idx, nb):
        if nb == 1 or len(idx) <= 1:
            ids[idx] = counter[0]; counter[0] += 1
            return
        pts = xy[idx]
        ax = int(np.argmax(pts.max(0) - pts.min(0)))
        order = idx[np.argsort(pts[:, ax], kind="stable")]
        cw = np.cumsum(w[order])
        nb_left = nb // 2
        cut = int(np.searchsorted(cw, (nb_left / nb) * cw[-1]))
        cut = min(max(cut, 1), len(order) - 1)
        rec(order[:cut], nb_left)
        rec(order[cut:], nb - nb_left)

    rec(np.arange(len(xy)), n_blocks)
    return ids


def main():
    print("=" * 68); print("03_litter_panel.py — survey rows -> model table"); print("=" * 68)

    df = pd.read_csv(os.path.join(DATA, "litter_surveys_raw_20250730.csv"), low_memory=False)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    n0 = len(df)
    df["ts"] = pd.to_datetime(df["survey_start"], format="%m/%d/%Y %H:%M", errors="coerce")
    bad = df["ts"].isna()
    if bad.any():
        df.loc[bad, "ts"] = pd.to_datetime(df.loc[bad, "survey_start"], errors="coerce")

    steps = [("raw rows", np.ones(len(df), bool)),
             (f"department == {DEPT}", df["department"].eq(DEPT).to_numpy()),
             (f"year in {YEARS}", df["ts"].dt.year.isin(YEARS).to_numpy()),
             ("rating in 1..4", df["Litter Rating"].between(1, 4).to_numpy()),
             ("has location_id", df["location_id"].notna().to_numpy())]
    keep = np.ones(len(df), bool)
    for name, m in steps:
        keep &= m
        print(f"  {name:>24}: {keep.sum():>7,} rows")
    df = df[keep].copy()

    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=4326).to_crs(CRS)
    cbg = gpd.read_file(os.path.join(OUT, "cbg_geom.geojson"))
    cbg["GEOID"] = cbg["GEOID"].astype(str)
    cbg["area_km2"] = cbg.geometry.area / 1e6
    j = g.sjoin(cbg[["GEOID", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    n_out = int(j["GEOID"].isna().sum())
    j = j[j["GEOID"].notna()].copy()
    j["px"], j["py"] = j.geometry.x, j.geometry.y
    j = pd.DataFrame(j.drop(columns="geometry"))
    print(f"  {'inside a CBG':>24}: {len(j):>7,} rows  ({n_out} outside dropped)")

    # --- static CBG data ------------------------------------------------------
    static = pd.read_parquet(os.path.join(OUT, "cbg_static.parquet"))
    rep = pd.read_parquet(os.path.join(OUT, "reporting_cbg.parquet"))
    cov_cols = [c for c in static.columns if c not in COV_DROP]
    static = static[["GEOID", "exposure", "in_fit_sample"] + cov_cols].merge(
        cbg[["GEOID", "area_km2"]], on="GEOID")
    rep = rep[["GEOID", "w_R_mean", "w_R_sd"]].rename(
        columns={"w_R_mean": "w_R_cbg", "w_R_sd": "w_R_cbg_sd"})

    # --- kernel w_R at the survey coordinate (02 evaluated every distinct lon/lat)
    wr = pd.read_parquet(os.path.join(OUT, "wR_survey_xy.parquet"))
    wr_cols = [c for c in wr.columns if c.startswith("wR_")]
    wr = wr[["lon", "lat"] + wr_cols]        # keyed on the raw CSV lon/lat, exact floats

    types = j["Litter Types"].fillna("")
    tok = types.apply(lambda s: set(t for t in s.split(",") if t))
    v = pd.DataFrame({
        "x": j["px"].to_numpy(), "y": j["py"].to_numpy(),
        "lon": j["x"].to_numpy(), "lat": j["y"].to_numpy(),
        "ts": j["ts"].to_numpy(),
        "rating": j["Litter Rating"].astype(int).to_numpy(),
        "bulk": tok.apply(lambda s: int(bool(s & BULK_TOKENS))).to_numpy(),
        "casual": tok.apply(lambda s: int(bool(s & CASUAL_TOKENS))).to_numpy(),
        "n_types": tok.apply(len).to_numpy(),
        "types": types.to_numpy(),
        "location_id": j["location_id"].to_numpy(),
        "surveyor": j["Creator"].to_numpy(),
        "GEOID": j["GEOID"].to_numpy(),
        "immediate_action": j["Requires Immediate Action"].eq("Y").astype(int).to_numpy(),
    })
    v["t_days"] = ((v["ts"] - EPOCH) / pd.Timedelta(days=1)).astype(float)
    v["year"] = v["ts"].dt.year
    v["ym"] = v["ts"].dt.year * 12 + v["ts"].dt.month
    v = v.merge(static, on="GEOID", how="left").merge(rep, on="GEOID", how="left")
    v = v.merge(wr, on=["lon", "lat"], how="left")
    v["log_expo_dens"] = np.log(v["exposure"] / v["area_km2"])

    # --- integer codes ------------------------------------------------------
    v["loc"] = codes(v["location_id"]); v["surv"] = codes(v["surveyor"])
    v["month"] = codes(v["ym"]); v["cbg"] = codes(v["GEOID"])

    # --- spatial blocks (per CBG, weighted by survey count) -------------------
    cnt = v.groupby("GEOID").size().rename("n_visits")
    cen = cbg.set_index("GEOID").geometry.centroid
    bl = pd.DataFrame({"GEOID": cnt.index, "n_visits": cnt.to_numpy(),
                       "cx": cen.reindex(cnt.index).x.to_numpy(),
                       "cy": cen.reindex(cnt.index).y.to_numpy()})
    bl["block"] = kd_blocks(bl[["cx", "cy"]].to_numpy(), bl["n_visits"].to_numpy(float), N_BLOCKS)
    v = v.merge(bl[["GEOID", "block"]], on="GEOID", how="left")
    bsz = v.groupby("block").size()
    bl.to_parquet(os.path.join(OUT, "cbg_blocks.parquet"), index=False)

    # --- report -------------------------------------------------------------
    print(f"\n  panel: {len(v):,} visits, {v['loc'].nunique():,} locations, "
          f"{v['surv'].nunique()} surveyors, {v['month'].nunique()} year-months, "
          f"{v['cbg'].nunique():,} CBGs")
    print(f"  rating: " + "  ".join(f"{k}: {n:,}" for k, n in v['rating'].value_counts().sort_index().items())
          + f"   mean {v['rating'].mean():.3f}  sd {v['rating'].std():.3f}")
    print(f"  bulk flag {v['bulk'].mean():.2%}, casual flag {v['casual'].mean():.2%}, "
          f"immediate action {v['immediate_action'].mean():.2%}")
    rv = v.groupby("loc").size()
    print(f"  visits per location: median {rv.median():.0f}, "
          f"{(rv >= 2).sum():,} locations with >= 2; "
          f"{v.groupby('loc')['year'].nunique().eq(2).mean():.1%} surveyed in both years")
    print(f"  blocks: {N_BLOCKS}, visits per block min {bsz.min():,} median {bsz.median():.0f} "
          f"max {bsz.max():,}; CBGs per block median {bl.groupby('block').size().median():.0f}")
    miss = {c: int(v[c].isna().sum()) for c in ["w_R_cbg", "log_expo_dens", wr_cols[0]] if c in v}
    print(f"  missing: {miss}  (visits in the 3 CBGs outside the fit sample keep NaN w_R_cbg)")
    print(f"  {len(wr_cols)} kernel w_R columns, {len(cov_cols)} CBG covariates attached.")

    v.to_parquet(os.path.join(OUT, "litter_panel.parquet"), index=False)
    print(f"\n  Saved → output/litter_panel.parquet, output/cbg_blocks.parquet")


if __name__ == "__main__":
    main()
