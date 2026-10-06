"""
design.py
=========
The one design matrix every regression in the pipeline uses:

    eta = g(x)  [or b*x, + c*extra]  +  month-of-sample effects  +  surveyor effects

- g: natural cubic spline with SPLINE_DF degrees of freedom (patsy `cr`),
  centred and with its first column dropped so it does not span the constant
  (the ordinal cutpoints carry the intercept).
- month-of-sample: dummies, first level dropped.
- surveyor: deviation coding (the most frequent surveyor is the reference,
  coded -1) with a small ridge, so a surveyor who does not appear in a
  training fold is held at the population mean instead of being unidentified.

The spline knots and column means are fitted on one sample (training) and
reused on another (test) through the `info` object returned by design().
"""

import numpy as np
import pandas as pd
from patsy import dmatrix, build_design_matrices

from config import SPLINE_DF, RIDGE_SURV


def spline_basis(x, info=None):
    if info is None:
        B = dmatrix(f"cr(x, df={SPLINE_DF}) - 1", {"x": x}, return_type="dataframe")
        info = {"di": B.design_info, "mean": B.to_numpy().mean(0)}
        Bm = B.to_numpy()
    else:
        Bm = np.asarray(build_design_matrices([info["di"]], {"x": x})[0])
    return (Bm - info["mean"])[:, 1:], info


class FE:
    """Month-of-sample dummies + deviation-coded surveyors, with consistent
    columns for any subset of the visits. Set surveyor=False for the
    month-only sensitivity."""

    def __init__(self, v: pd.DataFrame, surveyor: bool = True):
        self.months = np.sort(v["month"].unique())
        self.surveyor = surveyor
        counts = v["surv"].value_counts()
        self.ref = int(counts.index[0])
        self.survs = [int(s) for s in np.sort(v["surv"].unique()) if s != self.ref]

    def build(self, vs: pd.DataFrame):
        m = vs["month"].to_numpy()
        M = np.column_stack([(m == k).astype(float) for k in self.months[1:]])
        if not self.surveyor:
            return M, np.zeros(M.shape[1])
        s = vs["surv"].to_numpy()
        S = np.column_stack([np.where(s == k, 1.0, np.where(s == self.ref, -1.0, 0.0)) for k in self.survs])
        X = np.hstack([M, S])
        pen = np.r_[np.zeros(M.shape[1]), np.full(S.shape[1], RIDGE_SURV)]
        return X, pen


def design(x, fe, spline=True, extra=None, info=None):
    """
    x      : field values at the visits, shape (n,)
    fe     : (X_fe, pen_fe) for the same n rows, from FE.build()
    spline : g(x) as a spline (True) or a single linear column (False)
    extra  : dict name -> (n,) array of additional linear columns
    info   : spline info from a previous call (reuse training knots on test rows)
    Returns X (n, p), penalty (p,), column names, info.
    """
    cols, names = [], []
    if spline:
        B, info = spline_basis(x, info)
        cols.append(B); names += [f"g{k}" for k in range(B.shape[1])]
    else:
        cols.append(np.asarray(x, float)[:, None]); names.append("x")
    for name, arr in (extra or {}).items():
        cols.append(np.asarray(arr, float)[:, None]); names.append(name)
    n_main = sum(c.shape[1] for c in cols)
    X_fe, pen_fe = fe
    X = np.hstack(cols + [X_fe])
    pen = np.r_[np.zeros(n_main), pen_fe]
    return X, pen, names, info
