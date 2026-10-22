"""
download_litter_surveys.py
==========================
Pull the city's Litter Index survey points (one row per visit) from the public ArcGIS feature service and
save them as data/Litter_Index_Surveys.csv. OpenDataPhilly's "surveys" download link points at the
neighborhood layer, so this is the only way to get the visit-level table.

The service returns at most a few thousand rows per request; the script walks the table in pages ordered
by objectid until the server has nothing left. Dates come back as epoch milliseconds (UTC); they are written
as ISO timestamps in America/New_York alongside the raw value. No geometry is requested: the coordinates
are already attributes (x_coord, y_coord in WGS84; raw_* the uncorrected ones).

Run: python src/download_litter_surveys.py      (about 80 requests, a few minutes)
"""

import os
import json
import time
import urllib.request
import urllib.parse

import pandas as pd

from config import DATA

URL = "https://services.arcgis.com/fLeGjb7u4uXqeF9q/ArcGIS/rest/services/Litter_Index_Surveys/FeatureServer/0/query"
OUT_FILE = os.path.join(DATA, "Litter_Index_Surveys.csv")
PAGE = 2000
DATE_COLS = ["survey_start", "survey_end"]


def fetch(offset, tries=5):
    params = {"where": "1=1", "outFields": "*", "returnGeometry": "false", "orderByFields": "objectid",
              "resultOffset": offset, "resultRecordCount": PAGE, "f": "json"}
    url = URL + "?" + urllib.parse.urlencode(params)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.loads(r.read().decode("utf-8"))
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception as e:                       # transient server errors: wait and retry
            if k == tries - 1:
                raise
            print(f"    retry {k + 1} at offset {offset}: {e}")
            time.sleep(3 * (k + 1))


def main():
    t0 = time.time()
    rows, offset = [], 0
    while True:
        d = fetch(offset)
        feats = [f["attributes"] for f in d.get("features", [])]
        rows.extend(feats)
        print(f"  offset {offset:>7,}: {len(feats):,} rows  (total {len(rows):,})")
        if not feats or not d.get("exceededTransferLimit", False):
            break
        offset += len(feats)
        time.sleep(0.2)
    df = pd.DataFrame(rows)
    for c in DATE_COLS:
        ms = pd.to_numeric(df[c], errors="coerce")
        df[f"{c}_ms"] = ms
        df[c] = pd.to_datetime(ms, unit="ms", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values("objectid").reset_index(drop=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"\n{len(df):,} rows, {df.shape[1]} columns -> {OUT_FILE}  ({time.time() - t0:.0f} s)")
    print(df["year"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
