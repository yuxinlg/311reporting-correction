"""
01_data.py
==========
Raw inputs -> the tables every later script reads. Nothing here is a model.

Reads (symlinks in data/, read-only)
  311/{year}/public_cases_fc.shp     Illegal Dumping and Street Light Outage reports
  Street_Poles.geojson               street pole inventory
  tiger/tl_2023_42_bg                census block groups (Philadelphia = county 101)
  cov_cbg.geojson                    SafeGraph average monthly stops per CBG (the exposure)
  litter_surveys_raw_20250730.csv    Litter Index field survey

Writes (output/)
  points_dumping.parquet   x, y, ts, t_days, GEOID       one row per dumping report
  points_outage.parquet    x, y, ts, t_days, GEOID       one row per outage report
  poles_xy.parquet         x, y, GEOID                   one row per pole
  cbg.parquet              GEOID, area_km2, exposure, n_poles
  cbg_geom.geojson         GEOID, geometry (EPSG:26918)
  support_grid.parquet     x, y, GEOID, E_cell           50 m cells on the built support
  visits.parquet           the survey model table (columns listed in build_visits)
  cbg_blocks.parquet       GEOID, block                  40 contiguous spatial blocks, every block group

Coordinates are metres in EPSG:26918; t_days counts days from EPOCH.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

from config import (DATA, OUT, CRS, EPOCH, STUDY_YEARS, SVC_DUMPING, SVC_OUTAGE, EXPOSURE_COL,
                    PHILLY_COUNTYFP, SURVEY_YEARS, SURVEY_DEPT, BULK_TOKENS, GRID_RES, SUPPORT_DIST,
                    N_BLOCKS, ensure_dirs)

warnings.filterwarnings("ignore")


# =============================================================================
# census block groups
# =============================================================================

def load_cbgs() -> gpd.GeoDataFrame:
    bg = gpd.read_file(os.path.join(DATA, "tiger", "tl_2023_42_bg", "tl_2023_42_bg.shp"))
    bg = bg[bg["COUNTYFP"] == PHILLY_COUNTYFP].to_crs(CRS)
    bg["GEOID"] = bg["GEOID"].astype(str)
    out = bg[["GEOID", "geometry"]].reset_index(drop=True)
    out["area_km2"] = out.geometry.area / 1e6
    print(f"[cbg] {len(out)} Philadelphia block groups; median area {out['area_km2'].median():.3f} km²")
    return out


def to_cbg(points: gpd.GeoDataFrame, cbg: gpd.GeoDataFrame) -> np.ndarray:
    """GEOID of the block group containing each point (NaN outside), one match per point."""
    j = points.reset_index(drop=True).sjoin(cbg[["GEOID", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    return j["GEOID"].to_numpy()


# =============================================================================
# 311 reports
# =============================================================================

def load_311(cbg: gpd.GeoDataFrame):
    parts = []
    for yr in STUDY_YEARS:
        shp = os.path.join(DATA, "311", str(yr), "public_cases_fc.shp")
        g = gpd.read_file(shp, columns=["service_na", "requested_"])
        g = g[g["service_na"].isin([SVC_DUMPING, SVC_OUTAGE]) & g["geometry"].notna()].to_crs(CRS)
        parts.append(g)
    ev = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS)
    ev["ts"] = pd.to_datetime(ev["requested_"], errors="coerce")
    ev = ev[ev["ts"].notna() & ev["ts"].dt.year.isin(STUDY_YEARS)].reset_index(drop=True)
    pts = pd.DataFrame({"x": ev.geometry.x.to_numpy(), "y": ev.geometry.y.to_numpy(),
                        "ts": ev["ts"].to_numpy(), "GEOID": to_cbg(ev, cbg),
                        "service": ev["service_na"].to_numpy()})
    pts["t_days"] = ((pts["ts"] - EPOCH) / pd.Timedelta(days=1)).astype(float)
    cols = ["x", "y", "ts", "t_days", "GEOID"]
    d = pts.loc[pts["service"].eq(SVC_DUMPING), cols].reset_index(drop=True)
    r = pts.loc[pts["service"].eq(SVC_OUTAGE), cols].reset_index(drop=True)
    d.to_parquet(os.path.join(OUT, "points_dumping.parquet"), index=False)
    r.to_parquet(os.path.join(OUT, "points_outage.parquet"), index=False)
    print(f"[311] {len(d):,} dumping reports and {len(r):,} outage reports, {STUDY_YEARS[0]}–{STUDY_YEARS[-1]} "
          f"({d['GEOID'].isna().sum():,} / {r['GEOID'].isna().sum():,} outside every block group)")
    by_year = pd.DataFrame({"dumping": d["ts"].dt.year.value_counts().sort_index(),
                            "outage": r["ts"].dt.year.value_counts().sort_index()})
    print(by_year.to_string())


# =============================================================================
# poles and exposure
# =============================================================================

def load_poles(cbg: gpd.GeoDataFrame) -> pd.Series:
    p = gpd.read_file(os.path.join(DATA, "Street_Poles.geojson")).to_crs(CRS)
    poles = pd.DataFrame({"x": p.geometry.x.to_numpy(), "y": p.geometry.y.to_numpy(), "GEOID": to_cbg(p, cbg)})
    poles.to_parquet(os.path.join(OUT, "poles_xy.parquet"), index=False)
    n = poles.groupby("GEOID").size().rename("n_poles")
    print(f"[poles] {len(poles):,} poles; per block group median {n.median():.0f} (p5 {n.quantile(.05):.0f}, p95 {n.quantile(.95):.0f})")
    return n


def load_exposure() -> pd.Series:
    cov = gpd.read_file(os.path.join(DATA, "cov_cbg.geojson"))
    cov["GEOID"] = cov["GEOID"].astype(str)
    e = cov.set_index("GEOID")[EXPOSURE_COL].astype(float).rename("exposure")
    print(f"[exposure] {EXPOSURE_COL} (SafeGraph avg monthly stops) for {e.notna().sum():,} block groups; "
          f"median {e.median():,.0f}, {(~(e > 0)).sum()} missing or zero")
    return e


# =============================================================================
# built support: 50 m cells inside a block group and within 75 m of a pole
# =============================================================================

def build_support_grid(cbg: gpd.GeoDataFrame, exposure: pd.Series) -> pd.DataFrame:
    poles = pd.read_parquet(os.path.join(OUT, "poles_xy.parquet"))[["x", "y"]].to_numpy()
    xmin, ymin, xmax, ymax = cbg.total_bounds
    gx = np.arange(xmin + GRID_RES / 2, xmax, GRID_RES)
    gy = np.arange(ymin + GRID_RES / 2, ymax, GRID_RES)
    GX, GY = np.meshgrid(gx, gy)
    cells = gpd.GeoDataFrame({"x": GX.ravel(), "y": GY.ravel()},
                             geometry=gpd.points_from_xy(GX.ravel(), GY.ravel()), crs=CRS)
    cells["GEOID"] = to_cbg(cells, cbg)
    cells = pd.DataFrame(cells.drop(columns="geometry"))
    cells = cells[cells["GEOID"].notna()]
    dist, _ = cKDTree(poles).query(cells[["x", "y"]].to_numpy())
    sup = cells[dist <= SUPPORT_DIST].copy()
    n_cells = sup.groupby("GEOID").size()
    sup["E_cell"] = (sup["GEOID"].map(exposure).fillna(0.0) / sup["GEOID"].map(n_cells)).to_numpy()
    sup = sup[["x", "y", "GEOID", "E_cell"]].reset_index(drop=True)
    sup.to_parquet(os.path.join(OUT, "support_grid.parquet"), index=False)
    print(f"[support] {len(sup):,} cells of {GRID_RES} m inside a block group and within {SUPPORT_DIST} m of a pole "
          f"({len(sup) * GRID_RES ** 2 / 1e6:.0f} km²); exposure spread evenly over each block group's cells")
    return sup


# =============================================================================
# survey visits and spatial blocks
# =============================================================================

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


def build_visits(cbg: gpd.GeoDataFrame, exposure: pd.Series) -> pd.DataFrame:
    """
    One row per Streets-department survey visit, 2022-23, with a 1-4 rating,
    a location_id and a block group. Columns:
      x, y, lon, lat, ts, t_days, year, rating (1-4), bulk (any dumping-type
      litter token), location_id, surveyor, GEOID, exposure (of the block group),
      integer codes loc / surv / month (year-month), block (0..N_BLOCKS-1)
    """
    df = pd.read_csv(os.path.join(DATA, "litter_surveys_raw_20250730.csv"), low_memory=False)
    df.columns = [c.lstrip("﻿") for c in df.columns]
    df["ts"] = pd.to_datetime(df["survey_start"], format="%m/%d/%Y %H:%M", errors="coerce")
    bad = df["ts"].isna()
    if bad.any():
        df.loc[bad, "ts"] = pd.to_datetime(df.loc[bad, "survey_start"], errors="coerce")
    n_raw = len(df)
    keep = (df["department"].eq(SURVEY_DEPT) & df["ts"].dt.year.isin(SURVEY_YEARS)
            & df["Litter Rating"].between(1, 4) & df["location_id"].notna())
    df = df[keep].copy()
    g = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=4326).to_crs(CRS)
    geoid = to_cbg(g, cbg)
    inside = pd.notna(geoid)
    g, geoid = g[inside], geoid[inside]
    tokens = g["Litter Types"].fillna("").apply(lambda s: set(t for t in s.split(",") if t))
    v = pd.DataFrame({
        "x": g.geometry.x.to_numpy(), "y": g.geometry.y.to_numpy(),
        "lon": g["x"].to_numpy(), "lat": g["y"].to_numpy(),
        "ts": g["ts"].to_numpy(),
        "rating": g["Litter Rating"].astype(int).to_numpy(),
        "bulk": tokens.apply(lambda s: int(bool(s & BULK_TOKENS))).to_numpy(),
        "location_id": g["location_id"].to_numpy(),
        "surveyor": g["Creator"].to_numpy(),
        "GEOID": geoid,
    })
    v["t_days"] = ((v["ts"] - EPOCH) / pd.Timedelta(days=1)).astype(float)
    v["year"] = v["ts"].dt.year
    v["exposure"] = v["GEOID"].map(exposure).to_numpy()
    v["loc"] = pd.factorize(v["location_id"])[0]
    v["surv"] = pd.factorize(v["surveyor"])[0]
    v["month"] = pd.factorize(v["ts"].dt.year * 12 + v["ts"].dt.month, sort=True)[0]

    # spatial blocks: block groups cut into N_BLOCKS contiguous rectangles of ~equal visit count; a block
    # group without visits joins the block of the nearest centroid that has them, so every block group has one
    cnt = v.groupby("GEOID").size()
    cen_all = cbg.set_index("GEOID").geometry.centroid
    cen = cen_all.reindex(cnt.index)
    bl = pd.DataFrame({"GEOID": cnt.index, "n_visits": cnt.to_numpy(), "cx": cen.x.to_numpy(), "cy": cen.y.to_numpy()})
    bl["block"] = kd_blocks(bl[["cx", "cy"]].to_numpy(), bl["n_visits"].to_numpy(float), N_BLOCKS)
    _, nearest = cKDTree(bl[["cx", "cy"]].to_numpy()).query(np.c_[cen_all.x.to_numpy(), cen_all.y.to_numpy()])
    blocks_all = pd.DataFrame({"GEOID": cen_all.index, "block": bl["block"].to_numpy()[nearest]})
    v = v.merge(blocks_all, on="GEOID", how="left")
    blocks_all.to_parquet(os.path.join(OUT, "cbg_blocks.parquet"), index=False)
    v.to_parquet(os.path.join(OUT, "visits.parquet"), index=False)

    per_loc = v.groupby("loc").size()
    bsz = v.groupby("block").size()
    print(f"[survey] {n_raw:,} raw rows → {len(v):,} visits ({SURVEY_DEPT} dept, {SURVEY_YEARS}, rating 1–4, "
          f"with location_id, inside a block group)")
    print(f"         {v['loc'].nunique():,} locations (median {per_loc.median():.0f} visits each), "
          f"{v['surv'].nunique()} surveyors, {v['month'].nunique()} year-months, {v['GEOID'].nunique():,} block groups")
    print("         rating " + "  ".join(f"{k}: {n:,}" for k, n in v["rating"].value_counts().sort_index().items())
          + f"   mean {v['rating'].mean():.3f}  sd {v['rating'].std():.3f};  bulk flag {v['bulk'].mean():.1%}")
    print(f"         {N_BLOCKS} spatial blocks: {bsz.min():,}–{bsz.max():,} visits each; "
          f"{len(blocks_all) - len(bl)} block groups without visits attached to the nearest block")
    return v


# =============================================================================

def main():
    t0 = time.time()
    print("=" * 68); print("01_data.py — raw inputs → tables"); print("=" * 68)
    ensure_dirs()
    cbg = load_cbgs()
    load_311(cbg)
    n_poles = load_poles(cbg)
    exposure = load_exposure()
    tab = cbg[["GEOID", "area_km2"]].merge(exposure.reset_index(), on="GEOID", how="left") \
                                     .merge(n_poles.reset_index(), on="GEOID", how="left")
    tab["n_poles"] = tab["n_poles"].fillna(0).astype(int)
    tab.to_parquet(os.path.join(OUT, "cbg.parquet"), index=False)
    cbg[["GEOID", "geometry"]].to_file(os.path.join(OUT, "cbg_geom.geojson"), driver="GeoJSON")
    build_support_grid(cbg, exposure)
    build_visits(cbg, exposure)
    print(f"\nDone in {time.time() - t0:.0f} s. Outputs in output/.")


if __name__ == "__main__":
    main()
