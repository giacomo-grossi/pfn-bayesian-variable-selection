"""Reference method for the linear simulations: Bayesian spike-and-slab regression, the correctly
specified model.

Spike-and-slab
    y = alpha + X_g beta_g + eps, eps ~ N(0, sigma^2), with
        gamma_j | pi ~ Bernoulli(pi), pi ~ Beta(1, 1)    (beta-binomial prior on the model:
                                                          multiplicity correction, Scott and
                                                          Berger, 2010)
        beta_g | g, sigma^2 ~ N(0, g sigma^2 (X_g' X_g)^-1)   (Zellner g-prior)
        g ~ hyper-g/n with a = 3 (Liang et al., 2008), p(alpha, sigma^2) prop. to 1/sigma^2.
    alpha, beta and sigma^2 are integrated out analytically, so that, with R^2_g the coefficient
    of determination of the least-squares fit on the columns in gamma,
        p(y | gamma, g) prop. to (1 + g)^((n - 1 - k)/2) {1 + g (1 - R^2_g)}^(-(n - 1)/2),
    and g is integrated numerically on a fixed grid. The model space is explored by
    Metropolis-Hastings with add/delete and swap moves (Brown, Vannucci and Fearn, 1998),
    30000 iterations per chain of which 5000 burn-in; the inclusion probabilities are the
    inclusion frequencies after burn-in.
    Two chains, one from the empty model and one from the five variables most correlated with y;
    their PIPs are averaged and their largest disagreement is reported as a diagnostic.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betaln, logsumexp

# grid for g and log-weights of the hyper-g/n prior p(g) = (a-2)/(2n) (1 + g/n)^(-a/2), a = 3
_GGRID = np.exp(np.linspace(np.log(1e-2), np.log(1e7), 80))


def _log_g_weights(n, a=3.0):
    lg = np.log(_GGRID)
    dlg = np.gradient(lg)
    logp = np.log((a - 2) / (2 * n)) - (a / 2) * np.log1p(_GGRID / n)
    w = logp + lg + np.log(dlg)                     # integral over log g: p(g) g d(log g)
    return w - logsumexp(w)


def _logmarg_table(n, lw, kmax, m=3000):
    """log p(y | gamma) (g integrated out) on a grid of t = -log(1 - R^2), one row per size k."""
    t = np.linspace(0.0, 30.0, m)
    one_minus = np.exp(-t)
    k = np.arange(kmax + 1)[:, None, None]
    v = 0.5 * (n - 1 - k) * np.log1p(_GGRID)[None, None, :] \
        - 0.5 * (n - 1) * np.log1p(_GGRID[None, None, :] * one_minus[None, :, None])
    return t, logsumexp(v + lw[None, None, :], axis=2)


def spike_slab(X, y, iters=20000, burn=5000, seed=0, start=None):
    """Metropolis-Hastings on gamma with add/delete and swap moves (Brown, Vannucci and Fearn,
    1998). Returns the inclusion frequencies after burn-in and the mean model size."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    Xc = X - X.mean(0)
    Xc = Xc / np.maximum(Xc.std(0), 1e-12)
    yc = y - y.mean()
    yy = float(yc @ yc)
    kmax = n - 3
    tgrid, table = _logmarg_table(n, _log_g_weights(n), kmax)
    lprior = np.array([betaln(k + 1, p - k + 1) - betaln(1, 1) for k in range(kmax + 1)])
    cache = {}

    def logpost(idx):
        key = tuple(sorted(idx))
        v = cache.get(key)
        if v is None:
            k = len(key)
            if k:
                Q, _ = np.linalg.qr(Xc[:, key])
                r2 = float(np.sum((Q.T @ yc) ** 2)) / yy
            else:
                r2 = 0.0
            t = -np.log(max(1.0 - min(r2, 1.0), 1e-13))
            v = float(np.interp(min(t, tgrid[-1]), tgrid, table[k])) + lprior[k]
            if len(cache) < 200000:
                cache[key] = v
        return v

    gam = np.zeros(p, bool)
    if start is not None:
        gam[list(start)] = True
    lp = logpost(np.flatnonzero(gam))
    counts = np.zeros(p); sizes = 0.0; kept = 0
    for it in range(iters):
        inn = np.flatnonzero(gam); k = inn.size
        if rng.uniform() < 0.5 or k == 0 or k == p:          # add or delete one variable
            j = rng.integers(p)
            prop = gam.copy(); prop[j] = not prop[j]
            if prop.sum() > kmax:
                prop = None
        else:                                                 # swap an included and an excluded
            out = np.flatnonzero(~gam)
            prop = gam.copy(); prop[rng.choice(inn)] = False; prop[rng.choice(out)] = True
        if prop is not None:
            lq = logpost(np.flatnonzero(prop))
            if np.log(rng.uniform()) < lq - lp:               # both moves are symmetric
                gam, lp = prop, lq
        if it >= burn:
            counts += gam; sizes += gam.sum(); kept += 1
    return counts / kept, sizes / kept


def fit_spike_slab(X, y, seed=0, iters=30000, burn=5000):
    Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
    top = list(np.argsort(-np.abs(Xs.T @ (y - y.mean())))[:5])
    q1, s1 = spike_slab(X, y, iters, burn, seed=2 * seed)
    q2, s2 = spike_slab(X, y, iters, burn, seed=2 * seed + 1, start=top)
    q = (q1 + q2) / 2
    return {"score": q, "selected": q > 0.5,
            "diag": {"pip_chain_diff_max": float(np.abs(q1 - q2).max()),
                     "size_mean": (s1 + s2) / 2}}
