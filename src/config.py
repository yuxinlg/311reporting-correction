"""
config.py
=========
Every constant of the pipeline, in one place. Scripts 01-06 import from here
and never define their own tuning numbers.

Three values are NOT fixed here because the pipeline chooses them itself with a
stated rule and records them in output/choices.json:
    HW0, HW_ALT   kernel radius for the reporting surface w_R      (chosen in 02)
    H0, H_LIST    kernel radius for the 311 field, and its neighbours (chosen in 03)
    beta_hat      the strength of the correction, from the block groups  (estimated in 04)
"""

import os
import json

import numpy as np
import pandas as pd

# --- paths -------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
FIG = os.path.join(OUT, "figures")
FIELDS = os.path.join(OUT, "fields")
BETA = os.path.join(OUT, "beta")
SURVEY = os.path.join(OUT, "survey")
SURF = os.path.join(OUT, "surface")
DIST = os.path.join(OUT, "districts")
VALID = os.path.join(OUT, "validate")
CHOICES = os.path.join(OUT, "choices.json")

CRS = 26918                                  # UTM zone 18N, metres
EPOCH = pd.Timestamp("2021-01-01")           # t_days counts days from here

# --- raw data ----------------------------------------------------------------
STUDY_YEARS = [2021, 2022, 2023, 2024, 2025] # 311 shapefiles read
SVC_DUMPING = "Illegal Dumping"
SVC_OUTAGE = "Street Light Outage"
EXPOSURE_COL = "alloc_avg_s_cnt"             # SafeGraph average monthly stops per CBG
PHILLY_COUNTYFP = "101"

SURVEY_YEARS = [2022, 2023]
SURVEY_DEPT = "streets"
BULK_TOKENS = {"trash_bags", "construction", "oversized", "furniture", "tires", "vehicular"}

# --- support grid (where the city has streets: inside a CBG, near a pole) ----
GRID_RES = 50                                # m
SUPPORT_DIST = 75                            # m from a street pole

# --- spatial blocks for bootstrap and CV ---------------------------------------
N_BLOCKS = 40

# --- Stage 1: reporting propensity w_R ----------------------------------------
OUTAGE_WINDOW = (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"))
HW_SWEEP = [400, 600, 800, 1000]             # m, quartic radius
HW_RELIABILITY = 0.95                        # rule: smallest h_w with reliability >= this
KAPPA = 1.0                                  # Gamma shrinkage of the outage rate, in reports
WR_RASTER_RES = 100                          # m

# --- Stage 2: the 311 field ---------------------------------------------------
H_SWEEP = [300, 500, 800, 1200, 2000]        # m, quartic radius
H_RULE_TOL = 0.01                            # rule: smallest h within this of the max Spearman
TAU_SWEEP = [60, 120, 240, 480]              # days
TAU = 120                                    # days, fixed (pooling window)
PLACEBO_WIN = 90                             # days, past vs future window
ALPHA = 1.0                                  # Gamma shrinkage, prior reports at the citywide rate
BETA_GRID = np.round(np.arange(-0.2, 2.0001, 0.1), 2)     # 23 points for the survey profile (05); beta = 0 is the raw field
DATE_IN = pd.Timestamp("2023-06-30")         # snapshot inside the survey window (appendix)
DATE_OUT = pd.Timestamp("2025-12-31")        # snapshot extrapolation (appendix)

# --- the deliverable maps (06): long-term average report rates, raw and corrected ---
LONG_WINDOWS = {"2021-2025": (pd.Timestamp("2021-01-01"), pd.Timestamp("2026-01-01")),   # all reports
                "2022-2023": (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01"))}   # the survey years
RATE_PER = 1000.0                            # map unit: reports per year per 1,000 SafeGraph monthly stops

# --- Stage 3: beta from the block groups (04), no survey ------------------------
#     T_i ~ Poisson( E_i · exp(a + β · w̄_R,i) )   per block group, reports in BETA_WINDOW, SafeGraph exposure
#     as offset, w̄_R the mean kernel w_R over the block group's support cells. Identifying assumption: true
#     dumping per unit of exposure is unrelated to reporting propensity across block groups.
BETA_WINDOW = "2021-2025"                    # key of LONG_WINDOWS; the deliverable map's window
BETA_WINDOW_ALT = "2022-2023"                # sensitivity row
N_PERM = 50                                  # torus-shifted w_R maps for the null
SURVEY_MIN_NEFF = 10                         # survey-only map is blank where the kernel sees fewer visits
ZOOM = {"name": "West Fairmount Park", "x": (480500, 484500), "y": (4423800, 4429200)}   # m, for the zoomed figure

# --- district zooms (07): the same maps, β and survey check inside one planning district ---
DISTRICT_FILE = "Planning_Districts.geojson"  # City Planning Commission districts (18), symlink in data/
DISTRICT_COL = "dist_name"
DISTRICTS = {"Central": "Center City", "South": "South Philadelphia"}   # zoomed figures: district -> label
ZOOM_PAD = 300                               # m of margin around a district's bounding box

# --- the plain comparison with the survey (08): block groups ranked by the survey ----
VALID_WINDOW = "2022-2023"                   # reports compared with the survey: the survey years (2021–25 as a second row)
VALID_MIN_VISITS = 10                        # a block group needs this many visits for its survey mean to count
VALID_TOP_SHARE = 0.10                       # "dirtiest tenth" for the hit rate

# --- Stage 4: the survey as the test (05): rating model -------------------------
SPLINE_DF = 5
RIDGE_SURV = 1.0
GTOL = 1e-6                                  # per-observation gradient tolerance of the ordinal fits
N_JOBS = 8                                   # worker processes for the survey-profile bootstrap

# --- bootstrap (spatial blocks; 04 for beta, 05 for the survey's own beta) --------
N_BOOT = 200
BOOT_MAX_MOVES = 5                           # each resample walks the β grid uphill from the maximum one step at a
                                             # time until its maximum is bracketed, then fits a parabola (3 points)
BOOT_GTOL = 1e-4                             # looser tolerance for the resamples (ll error < 1, which moves a
                                             # parabola argmax by < 0.01 at this curvature)

# --- cross-validation ---------------------------------------------------------
N_FOLDS = 10
BUFFER = 1200                                # m dropped around the test blocks

SEED = 0

# --- plotting -----------------------------------------------------------------
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
# two colour families: what is on the ground (litter, dumping) vs how people report
CMAP_LITTER = "RdBu_r"       # survey ratings and E[rating]: blue = cleaner, red = dirtier, centred on the citywide mean
CMAP_DUMPING = "RdBu_r"      # 311 dumping report rates: blue = fewer than the citywide rate, red = more; log scale
CMAP_PROPENSITY = "PiYG_r"   # reporting propensity w_R: pink = chatty neighbourhoods, green = quiet ones
CMAP_FACTOR = "PiYG"         # correction factor log(corrected / raw): pink = discounted (chatty), green = scaled up (quiet)
CMAP_DELTA = "BrBG"          # the correction in absolute terms (corrected − raw): brown = reports removed, teal = reports added


def ensure_dirs():
    for d in [OUT, FIG, FIELDS, BETA, SURVEY, SURF, DIST, VALID]:
        os.makedirs(d, exist_ok=True)


def load_choices() -> dict:
    with open(CHOICES) as f:
        return json.load(f)


def update_choices(**kw):
    d = load_choices() if os.path.exists(CHOICES) else {}
    d.update(kw)
    with open(CHOICES, "w") as f:
        json.dump(d, f, indent=2)
    return d


def map_style(ax, geom, title, xlim=None, ylim=None, tick_km=None):
    """Map panel in the house style: block-group boundaries as thin black lines, axes in km
    (EPSG:26918), title top-left. Call after imshow; the caller adds the colorbar."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    geom.boundary.plot(ax=ax, color="black", linewidth=0.25, alpha=0.55)
    if xlim is not None:
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    if tick_km is None:
        span = (xlim[1] - xlim[0]) if xlim is not None else (geom.total_bounds[2] - geom.total_bounds[0])
        tick_km = 0.5 if span < 8000 else 2
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(tick_km * 1000))
        axis.set_major_formatter(FuncFormatter(lambda v, p: f"{v / 1000:.1f}" if tick_km < 1 else f"{v / 1000:.0f}"))
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_xlabel("easting (km, EPSG:26918)", fontsize=9, color=MUTED)
    ax.set_ylabel("northing (km)", fontsize=9, color=MUTED)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    for s in ax.spines.values():
        s.set_color(MUTED); s.set_linewidth(0.6)


def style(ax, title=None, xlabel=None, ylabel=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
