"""BART and DART, the two MCMC competitors (R package BART, function wbart).

    BART  Chipman, George and McCulloch (2010): sum of ntree trees, uniform splitting
          probabilities over the p variables.
    DART  Linero (2018): the same model with a Dirichlet prior on the splitting probabilities
          (sparse = TRUE, a = 0.5, b = 1, rho = p: the package defaults, which are Linero's).

Both are run with the same MCMC, the lengths of Linero (2018, Sec. 4: "no more than 5000 warmup
iterations and 5000 sampling iterations") in 4 independent chains, as he recommends several
chains when the predictors are correlated (the package default of one chain with 100 burn-in
iterations is far too short for variable selection):

    4 independent chains x (5000 burn-in + 5000 iterations, one draw kept every 5)

DART switches on the Dirichlet update after half of the burn-in (as in the package), so each
chain first runs 2500 iterations of plain BART, the warm start suggested by Linero. Convergence is checked on every data
set with split R-hat (sigma, number of variables used, f at each observed point) and with the
agreement of the inclusion probabilities across chains.

Each call returns
    score     variable ranking: DART pip (ties broken by the splitting probability),
              BART average share of splits (its pip is 1 for almost every variable)
    selected  median probability model, pip > 0.5
    pip, share, varprob
    F, sigma, used   B posterior draws of f at the observed points, of sigma, and of the
                     variables used in each draw
    diag      convergence diagnostics
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

R_SCRIPT = os.path.join(os.path.dirname(__file__), "r", "bart_dart.R")

MODELS = {
    "bart": dict(sparse=False, ntree=200),
    "dart": dict(sparse=True, ntree=200),
}
MCMC = dict(nchain=4, nskip=5000, ndpost=1000, keepevery=5)


def fit(model, X, y, seed=0, B=400, **overrides):
    """Run BART or DART on (X, y); overrides change the MCMC or the number of trees."""
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript not found: install R and the package BART (see the notebook)")
    cfg = {**MODELS[model], **MCMC, **overrides}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        np.savetxt(os.path.join(d, "x.csv"), X, delimiter=",", fmt="%.10g")
        np.savetxt(os.path.join(d, "y.txt"), y, fmt="%.10g")
        cmd = ["Rscript", R_SCRIPT, d, "TRUE" if cfg["sparse"] else "FALSE", str(int(seed)),
               str(cfg["nchain"]), str(cfg["nskip"]), str(cfg["ndpost"]), str(cfg["keepevery"]),
               str(cfg["ntree"]), str(int(B))]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"bart_dart.R failed: {r.stderr[-800:]}")
        st = np.genfromtxt(os.path.join(d, "stats.csv"), delimiter=",", names=True)
        dg = np.genfromtxt(os.path.join(d, "diag.csv"), delimiter=",", names=True)
        F = np.loadtxt(os.path.join(d, "F.csv"), delimiter=",", ndmin=2)
        sigma = np.loadtxt(os.path.join(d, "sigma.txt"), ndmin=1)
        used = np.loadtxt(os.path.join(d, "used.csv"), delimiter=",", ndmin=2).astype(bool)
    pip, share, vp = st["pip"], st["share"], st["varprob"]
    score = share if model == "bart" else pip + 1e-6 * vp
    return {"score": score, "selected": pip > 0.5, "pip": pip, "share": share, "varprob": vp,
            "F": F, "sigma": sigma, "used": used,
            "diag": {k: float(dg[k]) for k in dg.dtype.names}, "time": time.time() - t0}


def bart(X, y, seed=0, **kw):
    return fit("bart", X, y, seed, **kw)


def dart(X, y, seed=0, **kw):
    return fit("dart", X, y, seed, **kw)
