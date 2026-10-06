"""
01_build_panel.py
=================
Build the CBG x month panel that every downstream model reads.

Ported from the predecessor repo's `05a_streetlight_decomposition.py`
(sections 0 and 1), with one substantive change: **counts are resolved by
month instead of pooled over the whole window.** 05a fit a single pooled
cross-section; this project needs a temporal dimension so the outage equation
can carry a citywide time effect tau_m (which absorbs the PSIP LED conversion
structurally) and the dumping equation a time effect v_m.

Outputs
-------
output/panel_cbg_month.parquet   one row per CBG x month
    GEOID, year, month, month_idx, D (dumping reports), R (outage reports)

output/cbg_static.parquet        one row per CBG (1,338)
    GEOID, n_poles, n_led, led_share, n_peco, peco_share, n_pre24,
    pre24_share, exposure, plus every covariate column from cov_cbg.geojson

output/cbg_geom.geojson          GEOID + geometry (EPSG:26918), for adjacency
                                 and maps

Point tables (added for the point-level redesign; the same 528k-row-per-year
shapefiles are read once and both the panel and the points come out of the
same `ev` frame):

output/points_dumping.parquet    one row per geocoded Illegal Dumping report
    x, y (m, EPSG:26918), ts, t_days (days since EPOCH), GEOID
    (GEOID is NaN for the handful of points outside every Philadelphia CBG)

Design decision (2026-09-30): every 311 record is a *report-filing* event and
nothing more. Case closure (`closed_dat`) is deliberately not carried -- the
city's service response is a separate process that this project does not
model, for dumping or for outages.
output/points_outage.parquet     one row per geocoded Street Light Outage report
    x, y, ts, t_days, GEOID
output/poles_xy.parquet          one row per pole
    x, y, light_date, bulb_type, owner, GEOID

Nothing outside this repo is written to. The Dropbox source tree is read-only.
"""

import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
os.makedirs(OUT, exist_ok=True)

CRS = 26918

# --- study window -----------------------------------------------------------
STUDY_YEARS = [2021, 2022, 2023, 2024, 2025]
BASE_YEAR = min(STUDY_YEARS)
N_MONTHS = len(STUDY_YEARS) * 12
# t_days in the point tables counts days from this date (matches 05a's T
# axis, which starts at the first study year).
EPOCH = pd.Timestamp(f"{BASE_YEAR}-01-01")

# --- the two 311 streams ----------------------------------------------------
# T: the outcome. NT: the instrument -- a SINGLE category, pure streetlight.
# The predecessor's archived 04 used a 5-category basket; 05a narrowed it to
# this one and that is what the published beta = 0.579 rests on.
SVC_DUMPING = "Illegal Dumping"
SVC_OUTAGE = "Street Light Outage"

# --- paths ------------------------------------------------------------------
TIGER_BG = os.path.join(DATA, "tiger", "tl_2023_42_bg", "tl_2023_42_bg.shp")
POLES_PATH = os.path.join(DATA, "Street_Poles.geojson")
COV_CBG_PATH = os.path.join(DATA, "cov_cbg.geojson")

PHILLY_COUNTYFP = "101"
EXPOSURE_COL = "alloc_avg_s_cnt"        # SafeGraph avg monthly stop counts
PSIP_CUTOFF = pd.Timestamp("2024-01-01", tz="UTC")

PANEL_PATH = os.path.join(OUT, "panel_cbg_month.parquet")
STATIC_PATH = os.path.join(OUT, "cbg_static.parquet")
GEOM_PATH = os.path.join(OUT, "cbg_geom.geojson")
POINTS_D_PATH = os.path.join(OUT, "points_dumping.parquet")
POINTS_R_PATH = os.path.join(OUT, "points_outage.parquet")
POLES_XY_PATH = os.path.join(OUT, "poles_xy.parquet")


# =============================================================================
# CBG frame
# =============================================================================

def load_cbgs() -> gpd.GeoDataFrame:
    """Philadelphia 2023 block groups, EPSG:26918."""
    print("\n[cbg] Loading TIGER 2023 block groups ...")
    bg = gpd.read_file(TIGER_BG)
    bg = bg[bg["COUNTYFP"] == PHILLY_COUNTYFP].to_crs(CRS)
    bg["GEOID"] = bg["GEOID"].astype(str)
    out = bg[["GEOID", "geometry"]].reset_index(drop=True)
    print(f"  {len(out)} Philadelphia CBGs.")
    return out


# =============================================================================
# 311 counts, by CBG x month
# =============================================================================

def build_panel(cbg: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Count Illegal Dumping (D) and Street Light Outage (R) reports per
    CBG x month over STUDY_YEARS.

    Filtering to the two service categories happens BEFORE the spatial join --
    the raw files carry ~560k records per year and we need ~5% of them.
    """
    print(f"\n[311] Counting reports per CBG x month (years {STUDY_YEARS}) ...")

    parts = []
    for yr in STUDY_YEARS:
        shp = os.path.join(DATA, "311", str(yr), "public_cases_fc.shp")
        if not os.path.exists(shp):
            print(f"  [SKIP] {shp} not found.")
            continue
        gdf = gpd.read_file(shp, columns=["service_na", "requested_"])
        gdf = gdf[gdf["service_na"].isin([SVC_DUMPING, SVC_OUTAGE])]
        gdf = gdf[gdf["geometry"].notna()].to_crs(CRS)
        print(f"  {yr}: {len(gdf):>7,} geolocated records in the two categories")
        parts.append(gdf)

    if not parts:
        raise FileNotFoundError("No 311 shapefiles found under data/311/{year}/.")

    ev = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=CRS)

    # requested_ is the report date; drop rows whose date falls outside the
    # window (a year's file can contain stragglers from adjacent years).
    ev["ts"] = pd.to_datetime(ev["requested_"], errors="coerce")
    n_bad = int(ev["ts"].isna().sum())
    ev = ev[ev["ts"].notna()]
    ev = ev[ev["ts"].dt.year.isin(STUDY_YEARS)]
    if n_bad:
        print(f"  Dropped {n_bad:,} records with an unparseable date.")

    ev["year"] = ev["ts"].dt.year.astype(int)
    ev["month"] = ev["ts"].dt.month.astype(int)
    ev["month_idx"] = (ev["year"] - BASE_YEAR) * 12 + (ev["month"] - 1)

    # Left join so every geocoded point survives into the point tables; a
    # point on a shared CBG boundary can match twice under `within`, so keep
    # the first match per original row.
    ev = ev.reset_index(drop=True)
    joined = ev.sjoin(cbg[["GEOID", "geometry"]], how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    n_in = int(joined["GEOID"].notna().sum())
    print(f"  {n_in:,}/{len(ev):,} records fall inside a Philadelphia CBG.")

    _write_point_tables(joined)

    joined = joined[joined["GEOID"].notna()]
    joined["is_D"] = joined["service_na"].eq(SVC_DUMPING)
    joined["is_R"] = joined["service_na"].eq(SVC_OUTAGE)
    counts = (
        joined.groupby(["GEOID", "month_idx"])[["is_D", "is_R"]].sum().reset_index()
    )
    counts.columns = ["GEOID", "month_idx", "D", "R"]

    # complete the rectangle: every CBG x every month, zero-filled
    full = pd.MultiIndex.from_product(
        [cbg["GEOID"].tolist(), range(N_MONTHS)], names=["GEOID", "month_idx"]
    ).to_frame(index=False)
    panel = full.merge(counts, on=["GEOID", "month_idx"], how="left")
    panel[["D", "R"]] = panel[["D", "R"]].fillna(0).astype(int)
    panel["year"] = BASE_YEAR + panel["month_idx"] // 12
    panel["month"] = panel["month_idx"] % 12 + 1

    print(f"  Panel: {len(panel):,} rows "
          f"({panel['GEOID'].nunique()} CBGs x {N_MONTHS} months)")
    print(f"  Totals — D (dumping): {panel['D'].sum():,}   "
          f"R (outages): {panel['R'].sum():,}")
    return panel[["GEOID", "year", "month", "month_idx", "D", "R"]]


def _write_point_tables(joined: gpd.GeoDataFrame) -> None:
    """
    Emit the event-level point tables the kernel field reads. Called from
    build_panel() so the shapefiles are read exactly once.
    """
    pts = pd.DataFrame({
        "x": joined.geometry.x.to_numpy(),
        "y": joined.geometry.y.to_numpy(),
        "ts": joined["ts"].to_numpy(),
        "GEOID": joined["GEOID"].to_numpy(),
        "service_na": joined["service_na"].to_numpy(),
    })
    pts["t_days"] = ((pts["ts"] - EPOCH) / pd.Timedelta(days=1)).astype(float)

    cols = ["x", "y", "ts", "t_days", "GEOID"]
    d = pts.loc[pts["service_na"].eq(SVC_DUMPING), cols].reset_index(drop=True)
    r = pts.loc[pts["service_na"].eq(SVC_OUTAGE), cols].reset_index(drop=True)

    d.to_parquet(POINTS_D_PATH, index=False)
    r.to_parquet(POINTS_R_PATH, index=False)

    print(f"\n[points] Dumping: {len(d):,} geocoded points "
          f"({d['GEOID'].isna().sum():,} outside every CBG).")
    print(f"         Outage:  {len(r):,} geocoded points "
          f"({r['GEOID'].isna().sum():,} outside every CBG).")
    print(f"  Saved → {os.path.relpath(POINTS_D_PATH, ROOT)}")
    print(f"  Saved → {os.path.relpath(POINTS_R_PATH, ROOT)}")


# =============================================================================
# Poles (the NT exposure) and covariates
# =============================================================================

def attach_poles(cbg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Spatial join poles -> CBG. Denominator is ALL poles regardless of owner:
    residents report an outage without knowing whether Streets or PECO owns
    the pole, and there is no 311 routing data to split them.
    """
    print("\n[poles] Joining the pole inventory to CBGs ...")
    poles = gpd.read_file(POLES_PATH).to_crs(CRS)
    poles["light_dt"] = pd.to_datetime(poles["light_date"], errors="coerce", utc=True)
    print(f"  {len(poles):,} poles loaded.")

    j = poles.sjoin(cbg[["GEOID", "geometry"]], how="inner", predicate="within")
    print(f"  {len(j):,} fall inside a CBG ({len(poles) - len(j):,} outside the layer).")

    # pole coordinates for the point-level reporting surface (all poles, not
    # just those inside a CBG; GEOID is NaN for the ones outside the layer)
    pxy = pd.DataFrame({
        "x": poles.geometry.x.to_numpy(),
        "y": poles.geometry.y.to_numpy(),
        # tz-aware; keep as a Series so parquet stores a proper timestamp
        "light_date": poles["light_dt"].reset_index(drop=True),
        "bulb_type": poles["bulb_type"].to_numpy(),
        "owner": poles["owner"].to_numpy(),
    })
    pxy["GEOID"] = j["GEOID"].reindex(poles.index).to_numpy()
    pxy.to_parquet(POLES_XY_PATH, index=False)
    print(f"  Saved → {os.path.relpath(POLES_XY_PATH, ROOT)} "
          f"({len(pxy):,} poles; light_date populated for "
          f"{pxy['light_date'].notna().mean():.1%}, "
          f"bulb_type LED for {pxy['bulb_type'].eq('LED').mean():.1%})")

    j["is_led"] = j["bulb_type"].eq("LED")
    j["is_peco"] = j["owner"].eq("PECO")
    j["is_pre24"] = j["light_dt"] < PSIP_CUTOFF
    agg = j.groupby("GEOID").agg(
        n_poles=("is_led", "size"),
        n_led=("is_led", "sum"),
        n_peco=("is_peco", "sum"),
        n_pre24=("is_pre24", "sum"),
    ).reset_index()

    cbg = cbg.merge(agg, on="GEOID", how="left")
    for c in ["n_poles", "n_led", "n_peco", "n_pre24"]:
        cbg[c] = cbg[c].fillna(0).astype(int)
    with np.errstate(invalid="ignore"):
        for num, name in [("n_led", "led_share"), ("n_peco", "peco_share"),
                          ("n_pre24", "pre24_share")]:
            cbg[name] = np.where(cbg["n_poles"] > 0, cbg[num] / cbg["n_poles"], np.nan)

    q = cbg["n_poles"]
    print(f"  Poles per CBG: median {q.median():.0f} "
          f"(p5 {q.quantile(.05):.0f}, p95 {q.quantile(.95):.0f}); "
          f"{(q == 0).sum()} zero-pole CBGs, {(q < 10).sum()} with <10.")
    return cbg


def attach_covariates(cbg: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge cov_cbg.geojson, exposing alloc_avg_s_cnt as `exposure`."""
    print("\n[cov] Attaching covariates and the dumping exposure ...")
    cov = gpd.read_file(COV_CBG_PATH)
    cov["GEOID"] = cov["GEOID"].astype(str)
    cov = pd.DataFrame(cov.drop(columns="geometry"))
    # the land-use share `T` (transportation) collides with 05a's T_count
    if "T" in cov.columns:
        cov = cov.rename(columns={"T": "lu_T"})

    if EXPOSURE_COL not in cov.columns:
        raise KeyError(f"'{EXPOSURE_COL}' absent from cov_cbg.geojson.")
    cov["exposure"] = cov[EXPOSURE_COL]

    cbg = cbg.merge(cov, on="GEOID", how="left")
    n_bad = int((~(cbg["exposure"] > 0)).sum())
    print(f"  Exposure = {EXPOSURE_COL} (SafeGraph avg monthly stops).")
    print(f"  {len(cov.columns) - 1} covariate columns merged; "
          f"{n_bad} CBGs with missing/zero exposure (excluded downstream).")
    return cbg


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 68)
    print("01_build_panel.py — CBG x month panel for the reporting correction")
    print("=" * 68)

    cbg = load_cbgs()
    panel = build_panel(cbg)

    static = attach_poles(cbg)
    static = attach_covariates(static)

    # the fit sample: needs both denominators strictly positive
    fit_ok = (static["exposure"] > 0) & (static["n_poles"] > 0)
    static["in_fit_sample"] = fit_ok
    print(f"\n[sample] {int(fit_ok.sum())}/{len(static)} CBGs have "
          f"exposure > 0 and n_poles > 0 (the fit sample).")

    geom = static[["GEOID", "geometry"]].copy()
    static_df = pd.DataFrame(static.drop(columns="geometry"))

    panel.to_parquet(PANEL_PATH, index=False)
    static_df.to_parquet(STATIC_PATH, index=False)
    gpd.GeoDataFrame(geom, geometry="geometry", crs=CRS).to_file(
        GEOM_PATH, driver="GeoJSON")

    print(f"\n  Saved → {os.path.relpath(PANEL_PATH, ROOT)}")
    print(f"  Saved → {os.path.relpath(STATIC_PATH, ROOT)}")
    print(f"  Saved → {os.path.relpath(GEOM_PATH, ROOT)}")

    # --- regression-test scaffolding: pooled totals must match 05a -----------
    pooled = panel.groupby("GEOID")[["D", "R"]].sum().reset_index()
    chk = static_df[["GEOID", "in_fit_sample"]].merge(pooled, on="GEOID")
    chk = chk[chk["in_fit_sample"]]
    print(f"\n[check] Pooled totals over the fit sample "
          f"({len(chk)} CBGs): D = {chk['D'].sum():,}, R = {chk['R'].sum():,}")
    print("        05a reported T = 116,751 and NT = 75,956 over 1,335 CBGs.")
    print("        Differences are expected: 05a counted by shapefile-year with")
    print("        no date filter, this counts by the `requested_` date.")


if __name__ == "__main__":
    main()
