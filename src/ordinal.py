"""
ordinal.py
==========
Proportional-odds (cumulative) logit with analytic gradient, L-BFGS, optional
frequency weights, warm starts, and a cluster-robust sandwich covariance.
Written because the calibration refits the same model hundreds of times
(profile likelihood over beta, block bootstrap, blocked CV) and statsmodels'
OrderedModel is an order of magnitude too slow for that.

Model:  P(y <= k | x) = expit(kappa_k - x'beta),  k = 0..K-2,  y in {0..K-1}
        cutpoints kappa_0 < ... < kappa_{K-2} parameterised as
        kappa_0, kappa_0 + exp(d_1), kappa_0 + exp(d_1) + exp(d_2), ...
X must NOT contain a constant column (the cutpoints carry the intercept).

E[y + 1] (the 1..K "rating" scale) = 1 + sum_k P(y > k).
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

_TINY = 1e-300


class OrderedLogit:
    def __init__(self, n_cat: int):
        self.K = n_cat

    # --- parameterisation ---------------------------------------------------
    def unpack(self, theta, p):
        beta = theta[:p]
        cuts = np.empty(self.K - 1)
        cuts[0] = theta[p]
        if self.K > 2:
            cuts[1:] = theta[p] + np.cumsum(np.exp(theta[p + 1:]))
        return beta, cuts

    def pack(self, beta, cuts):
        d = np.log(np.diff(cuts)) if self.K > 2 else np.zeros(0)
        return np.concatenate([beta, [cuts[0]], d])

    def start(self, X, y):
        """beta = 0, cutpoints at the logits of the cumulative frequencies."""
        freq = np.bincount(y, minlength=self.K) / len(y)
        cum = np.clip(np.cumsum(freq)[:-1], 1e-6, 1 - 1e-6)
        return self.pack(np.zeros(X.shape[1]), np.log(cum / (1 - cum)))

    # --- likelihood pieces --------------------------------------------------
    def _parts(self, theta, X, y):
        p = X.shape[1]
        K = self.K
        beta, cuts = self.unpack(theta, p)
        eta = X @ beta
        cu = np.where(y < K - 1, cuts[np.minimum(y, K - 2)], np.inf) - eta
        cl = np.where(y > 0, cuts[np.maximum(y - 1, 0)], -np.inf) - eta
        Fu, Fl = expit(cu), expit(cl)
        P = np.maximum(Fu - Fl, _TINY)
        fu = np.where(np.isfinite(cu), Fu * (1.0 - Fu), 0.0)
        fl = np.where(np.isfinite(cl), Fl * (1.0 - Fl), 0.0)
        return beta, cuts, eta, P, fu, fl

    def _cut_scores(self, y, P, fu, fl):
        """Per-observation dl_i/dkappa_m, shape (n, K-1)."""
        n = len(y)
        G = np.zeros((n, self.K - 1))
        mu = y <= self.K - 2
        G[np.flatnonzero(mu), y[mu]] += fu[mu] / P[mu]
        ml = y >= 1
        G[np.flatnonzero(ml), y[ml] - 1] -= fl[ml] / P[ml]
        return G

    def _chain_cuts(self, theta, p, Gcut):
        """Map d/dkappa (n, K-1) to d/d(kappa_0, d_1, ...) (n, K-1)."""
        out = np.empty_like(Gcut)
        out[:, 0] = Gcut.sum(1)
        if self.K > 2:
            suffix = np.cumsum(Gcut[:, ::-1], axis=1)[:, ::-1]          # sum_{m >= j}
            out[:, 1:] = np.exp(theta[p + 1:]) * suffix[:, 1:]
        return out

    def scores(self, theta, X, y):
        """Per-observation score matrix (n, p + K - 1)."""
        p = X.shape[1]
        _, _, _, P, fu, fl = self._parts(theta, X, y)
        d_eta = (fl - fu) / P
        return np.hstack([X * d_eta[:, None], self._chain_cuts(theta, p, self._cut_scores(y, P, fu, fl))])

    def nll_grad(self, theta, X, y, w=None):
        p = X.shape[1]
        _, _, _, P, fu, fl = self._parts(theta, X, y)
        w = np.ones(len(y)) if w is None else w
        nll = -np.sum(w * np.log(P))
        d_eta = (fl - fu) / P
        g_beta = -(X.T @ (w * d_eta))
        Gcut = self._cut_scores(y, P, fu, fl)
        g_cut = -self._chain_cuts(theta, p, Gcut * w[:, None]).sum(0)
        return nll, np.concatenate([g_beta, g_cut])

    def loglik(self, theta, X, y, w=None):
        return -self.nll_grad(theta, X, y, w)[0]

    def loglik_i(self, theta, X, y):
        _, _, _, P, _, _ = self._parts(theta, X, y)
        return np.log(P)

    # --- fitting ------------------------------------------------------------
    def fit(self, X, y, w=None, theta0=None, maxiter=5000, gtol=1e-5, penalty=None):
        """
        L-BFGS on column-standardised X (unit-variance columns cut the cold
        start from ~45 s to ~3 s on 72k x 65); coefficients are returned in
        the original units. theta0 (original units) enables warm starts.
        `penalty`: optional array (p,) of ridge weights lambda_k on the
        ORIGINAL-unit coefficients, 0.5 * sum lambda_k beta_k^2 added to the
        negative log-likelihood. Used for surveyor effects in the CV so a
        surveyor absent from a training fold is held at the population mean.
        The returned log-likelihood excludes the penalty.
        """
        p = X.shape[1]
        sc = X.std(0)
        sc[sc == 0] = 1.0
        Xs = X / sc
        if theta0 is None:
            t0 = self.start(Xs, y)
        else:
            t0 = np.asarray(theta0, float).copy()
            t0[:p] = t0[:p] * sc
        pen = None if penalty is None else np.asarray(penalty, float)
        # The objective is the MEAN negative log-likelihood (per unit weight), so `gtol`
        # is a per-observation gradient tolerance that does not scale with n. With the
        # summed objective the gradient is O(n) and gtol is never reached, so every fit
        # runs to the ftol criterion (hundreds of evaluations) regardless of gtol.
        n_w = float(len(y) if w is None else np.sum(w))

        def obj(t):
            nll, g = self.nll_grad(t, Xs, y, w)
            if pen is not None:
                b = t[:p] / sc
                nll += 0.5 * np.sum(pen * b * b)
                g[:p] += pen * b / sc
            return nll / n_w, g / n_w

        res = minimize(obj, t0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "maxfun": maxiter * 2, "ftol": 1e-14, "gtol": gtol})
        theta = res.x.copy()
        theta[:p] = theta[:p] / sc
        ll = -self.nll_grad(res.x, Xs, y, w)[0]
        return theta, ll, bool(res.success)

    def hessian(self, theta, X, y, w=None, eps=1e-5):
        """Numerical Hessian of the NEGATIVE log-likelihood (central differences of the analytic gradient)."""
        d = len(theta)
        H = np.empty((d, d))
        for k in range(d):
            e = np.zeros(d); e[k] = eps
            gp = self.nll_grad(theta + e, X, y, w)[1]
            gm = self.nll_grad(theta - e, X, y, w)[1]
            H[:, k] = (gp - gm) / (2 * eps)
        return 0.5 * (H + H.T)

    def cov_cluster(self, theta, X, y, groups, w=None):
        """Sandwich H^-1 (sum_g s_g s_g') H^-1 with a G/(G-1) factor."""
        H = self.hessian(theta, X, y, w)
        S = self.scores(theta, X, y)
        if w is not None:
            S = S * w[:, None]
        g = np.asarray(groups)
        codes, inv = np.unique(g, return_inverse=True)
        Sg = np.zeros((len(codes), S.shape[1]))
        np.add.at(Sg, inv, S)
        meat = Sg.T @ Sg * len(codes) / (len(codes) - 1)
        Hinv = np.linalg.pinv(H)
        return Hinv @ meat @ Hinv

    # --- prediction ---------------------------------------------------------
    def probs(self, theta, X):
        p = X.shape[1]
        beta, cuts = self.unpack(theta, p)
        eta = X @ beta
        cdf = expit(cuts[None, :] - eta[:, None])               # P(y <= k), k = 0..K-2
        cdf = np.hstack([cdf, np.ones((len(eta), 1))])
        return np.diff(np.hstack([np.zeros((len(eta), 1)), cdf]), axis=1)

    def expected_rating(self, theta, X):
        """1 + sum_k P(y > k), on the 1..K scale."""
        p = X.shape[1]
        beta, cuts = self.unpack(theta, p)
        eta = X @ beta
        return 1.0 + expit(eta[:, None] - cuts[None, :]).sum(1)

    def linear_predictor(self, theta, X):
        return X @ theta[:X.shape[1]]


def somers_d(y, score):
    """
    Somers' D_yx of an ordinal outcome y vs a continuous score: among pairs
    with different y, (concordant - discordant) / all such pairs, ties in
    the score counting as neither. O(n log n) via Mann-Whitney U per
    category pair (the contingency-table route is O(n^2) for a continuous
    score).
    """
    from scipy.stats import rankdata
    y = np.asarray(y); score = np.asarray(score, float)
    cats = np.unique(y)
    cd, pairs = 0.0, 0.0
    for ia in range(len(cats)):
        for ib in range(ia + 1, len(cats)):
            a, b = score[y == cats[ia]], score[y == cats[ib]]
            na, nb = len(a), len(b)
            if na == 0 or nb == 0:
                continue
            r = rankdata(np.concatenate([a, b]))
            u_b = r[na:].sum() - nb * (nb + 1) / 2          # U for group b: pairs with s_b > s_a (+0.5 ties)
            cd += 2 * u_b - na * nb                          # concordant - discordant
            pairs += na * nb
    return float(cd / pairs) if pairs else np.nan
