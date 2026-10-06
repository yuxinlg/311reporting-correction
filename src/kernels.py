"""
kernels.py
==========
Compact-support quartic (biweight) kernel sums on the plane, shared by the
reporting surface (02), the 311 field (03), the maps (06) and the district
zooms (07).

    K_h(d) = (1 - (d/h)^2)^2   for d < h, else 0
    sd(K_h) = h / sqrt(7)  ~ 0.378 h  (a quartic radius of 500 m smooths like a
                                       Gaussian with sd ~ 190 m)

All coordinates are metres in EPSG:26918. Pairs are enumerated with
cKDTree.sparse_distance_matrix in chunks of queries so memory stays flat.
"""

import numpy as np
from scipy.spatial import cKDTree

SD_FACTOR = 1.0 / np.sqrt(7.0)


def quartic(d: np.ndarray, h: float) -> np.ndarray:
    u2 = (d / h) ** 2
    return np.where(u2 < 1.0, (1.0 - u2) ** 2, 0.0)


def pairs_within(query_xy: np.ndarray, source_tree: cKDTree, h: float,
                 chunk: int = None):
    """
    Yield (i, j, d) arrays for every (query, source) pair with d <= h.
    i indexes query_xy, j indexes the tree's data. Chunk size scales with
    1/h^2 so each block holds roughly 5M pairs.
    """
    n = len(query_xy)
    if chunk is None:
        chunk = max(200, int(4000 * (800.0 / max(h, 1.0)) ** 2)) if h > 800 else 4000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        qt = cKDTree(query_xy[s:e])
        rec = qt.sparse_distance_matrix(source_tree, h, output_type="ndarray")
        yield (rec["i"].astype(np.int64) + s, rec["j"].astype(np.int64), rec["v"])


def kernel_sum(query_xy: np.ndarray, source_xy: np.ndarray, h: float,
               weights: np.ndarray = None, source_tree: cKDTree = None,
               return_neff: bool = False):
    """
    S(q) = sum_j w_j K_h(|q - s_j|). If return_neff, also return the Kish
    effective number of sources behind each sum, (sum w K)^2 / sum (w K)^2,
    which sets the Poisson noise floor of log S.
    """
    tree = source_tree if source_tree is not None else cKDTree(source_xy)
    n = len(query_xy)
    S = np.zeros(n)
    S2 = np.zeros(n) if return_neff else None
    for i, j, d in pairs_within(query_xy, tree, h):
        wk = quartic(d, h)
        if weights is not None:
            wk = wk * weights[j]
        S += np.bincount(i, weights=wk, minlength=n)
        if return_neff:
            S2 += np.bincount(i, weights=wk * wk, minlength=n)
    if return_neff:
        with np.errstate(divide="ignore", invalid="ignore"):
            neff = np.where(S2 > 0, S * S / S2, 0.0)
        return S, neff
    return S


# =============================================================================
# space-time: causal exponential in time on top of the quartic in space
# =============================================================================

def accumulate_st(query_xy, query_t, src_xy, src_t, h, taus, weights=None,
                  src_tree=None, causal=True):
    """
    For every query point q with time t_q and every tau:
        S[tau, name](q) = sum_j w_name,j * K_h(|q - s_j|) * exp(-(t_q - t_j)/tau) * 1[t_q >= t_j]
    `weights` is a dict name -> array over sources (default {"one": 1}).
    Returns dict with keys (tau, name) plus "n_within" (all-time count within h,
    ignoring time). Nothing is stored per pair, so memory stays flat.
    """
    tree = src_tree if src_tree is not None else cKDTree(src_xy)
    n = len(query_xy)
    weights = weights or {"one": np.ones(len(src_xy))}
    out = {(tau, k): np.zeros(n) for tau in taus for k in weights}
    out["n_within"] = np.zeros(n)
    for i, j, d in pairs_within(query_xy, tree, h):
        K = quartic(d, h)
        dt = query_t[i] - src_t[j]
        if causal:
            K = np.where(dt >= 0, K, 0.0)
        out["n_within"] += np.bincount(i, minlength=n)
        for tau in taus:
            v = K * np.exp(-np.maximum(dt, 0.0) / tau) if np.isfinite(tau) else K
            for k, w in weights.items():
                out[(tau, k)] += np.bincount(i, weights=v * w[j], minlength=n)
    return out


def shrunk_log(num, den, alpha=1.0, lam_bar=None):
    """
    Gamma-shrunk log rate: log((num + alpha) / (den + alpha / lam_bar)), with
    alpha "prior events" and lam_bar the citywide ratio. Handles num = 0.
    """
    if lam_bar is None:
        lam_bar = num.sum() / den.sum()
    return np.log((num + alpha) / (den + alpha / lam_bar)), lam_bar
