"""
config.py
=========
Every constant of the pipeline, in one place. Scripts 01-06 import from here
and never define their own tuning numbers.

Two values are NOT fixed here because the pipeline chooses them itself with a
stated rule and records them in output/choices.json:
    HW0, HW_ALT   kernel radius for the reporting surface w_R      (chosen in 02)
    H0, H_LIST    kernel radius for the 311 field, and its neighbours (chosen in 03)
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
CALIB = os.path.join(OUT, "calib")
CVDIR = os.path.join(OUT, "cv")
SURF = os.path.join(OUT, "surface")
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
ALPHA_ALT = 0.25
BETA_GRID = np.round(np.arange(-0.2, 2.0001, 0.1), 2)     # 23 points; beta = 0 is the raw field
NULL_BETAS = [0.5, 1.0, 1.5, 2.0]            # torus-shifted maps are evaluated at {0} + these
N_PERM = 50
DATE_IN = pd.Timestamp("2023-06-30")         # inside the survey window
DATE_OUT = pd.Timestamp("2025-12-31")        # extrapolation

# --- Stage 3: rating model ----------------------------------------------------
SPLINE_DF = 5
RIDGE_SURV = 1.0
GTOL = 1e-6                                  # per-observation gradient tolerance of the ordinal fits
N_JOBS = 8                                   # worker processes for the bootstrap and the torus null

# --- bootstrap ----------------------------------------------------------------
N_BOOT = 200
BOOT_MAX_MOVES = 5                           # each resample walks the β grid uphill from beta_hat one step at a
                                             # time until its maximum is bracketed, then fits a parabola (3 points)
BOOT_GTOL = 1e-4                             # looser tolerance for the resamples and the torus null (ll error < 1,
                                             # which moves a parabola argmax by < 0.01 at this curvature)

# --- cross-validation ---------------------------------------------------------
N_FOLDS = 10
BUFFER = 1200                                # m dropped around the test blocks
CV_BETA_GRID = np.round(np.arange(0.0, 2.0001, 0.2), 2)   # in-fold choice of beta; a subset of BETA_GRID

SEED = 0

# --- plotting -----------------------------------------------------------------
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def ensure_dirs():
    for d in [OUT, FIG, FIELDS, CALIB, CVDIR, SURF]:
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
