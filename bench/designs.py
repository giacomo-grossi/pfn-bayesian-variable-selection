"""Simulation designs of Chapter 6 of the thesis (none is drawn from the meta-prior).

    LIN_EQ / LIN_TOE  linear model, 5 active variables with coefficient 5, Gaussian design with
                      equicorrelated (Cov = rho) or Toeplitz (Cov = rho^|i-j|) covariance
    FRI_EQ / FRI_TOE  Friedman (1991) function on 5 active variables, the same Gaussian designs
                      transformed to uniform margins (Gaussian copula), as Friedman requires
    C                 y = x1 x2 + sin(x3 x4) + 1{x5 > 0} x6 (Luo, 2026), Toeplitz Gaussian design;
                      x1..x5 act only through interactions

The active variables are at random positions; y = f(x_S0) + eps with Gaussian noise and
SNR = Var f / sigma^2 (Var f estimated on the training and test rows of the data set), so the
scale of f does not matter. Every data set is seeded by a hash of (design, n, p, rho, SNR,
replicate), so all methods see identical data.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

N_TEST = 1000  # held-out rows (used only by the optional prediction of bench.run --predict)


def seed_of(*key) -> int:
    h = hashlib.sha256(repr(key).encode()).hexdigest()
    return int(h[:15], 16)


# --------------------------------------------------------------------------------------
# regression functions (inputs: the active columns, in the order of S0)
# --------------------------------------------------------------------------------------

def f_lin5(x):
    return 5.0 * x.sum(axis=1)


def f_friedman(x):  # Friedman (1991), uniform margins
    return (10 * np.sin(np.pi * x[:, 0] * x[:, 1]) + 20 * (x[:, 2] - 0.5) ** 2
            + 10 * x[:, 3] + 5 * x[:, 4])


def f_C(x):  # interactions and a threshold; x3, x4 act only through sin(x3 x4)
    return x[:, 0] * x[:, 1] + np.sin(x[:, 2] * x[:, 3]) + (x[:, 4] > 0) * x[:, 5]


DESIGNS = {"LIN_EQ": (f_lin5, 5), "LIN_TOE": (f_lin5, 5),
           "FRI_EQ": (f_friedman, 5), "FRI_TOE": (f_friedman, 5), "C": (f_C, 6)}
SIMPLE_DESIGNS = ("LIN_EQ", "LIN_TOE", "FRI_EQ", "FRI_TOE")
SIMPLE_N = {"LIN_EQ": 50, "LIN_TOE": 50, "FRI_EQ": 100, "FRI_TOE": 100}
UNIFORM_MARGINS = ("FRI_EQ", "FRI_TOE")


def equicorr_sample(rng, n, p, rho):
    return np.sqrt(rho) * rng.standard_normal((n, 1)) + np.sqrt(1 - rho) * rng.standard_normal((n, p))


def ar1_sample(rng, n, p, rho):
    X = np.empty((n, p))
    X[:, 0] = rng.standard_normal(n)
    s = np.sqrt(1 - rho ** 2)
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + s * rng.standard_normal(n)
    return X


@dataclass
class Dataset:
    key: tuple
    X: np.ndarray
    y: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    f_test: np.ndarray
    S0: np.ndarray
    sigma: float
    rho: float

    @property
    def id(self):
        return "|".join(map(str, self.key))

    @property
    def gamma(self):
        g = np.zeros(self.X.shape[1], dtype=int)
        g[self.S0] = 1
        return g


def make_dataset(design, n, p, rho, snr, rep, n_test=N_TEST) -> Dataset:
    key = (design, n, p, rho, snr, rep)
    rng = np.random.default_rng(seed_of(*key))
    f, s = DESIGNS[design]
    if design.endswith("_EQ"):
        Xall = equicorr_sample(rng, n + n_test, p, rho)
    else:
        Xall = ar1_sample(rng, n + n_test, p, rho)
    if design in UNIFORM_MARGINS:
        Xall = ndtr(Xall)
    S0 = np.sort(rng.choice(p, size=s, replace=False))
    fx = f(Xall[:, S0])
    sigma = np.sqrt(fx.var() / snr)
    yall = fx + sigma * rng.standard_normal(n + n_test)
    return Dataset(key=key, X=Xall[:n], y=yall[:n], X_test=Xall[n:], y_test=yall[n:],
                   f_test=fx[n:], S0=S0, sigma=float(sigma), rho=rho)


# --------------------------------------------------------------------------------------
# study grids
# --------------------------------------------------------------------------------------

def grid(study: str, reps=None):
    """Yield (design, n, p, rho, snr, rep, extra) for each data set of a study."""
    R = reps or 100
    if study == "simple":
        # Section 6.1: four designs, p = 100, rho in {0.3, 0.7}, SNR in {1, 5}
        for d in SIMPLE_DESIGNS:
            for rho, snr, r in itertools.product((0.3, 0.7), (1, 5), range(R)):
                yield d, SIMPLE_N[d], 100, rho, snr, r, {}
    elif study == "ablation":
        # Section 6.2: full against ablated network on design C, n grows so that the effect of
        # the prior can emerge (n <= 900 is inside the meta-prior)
        for n, rho, snr, r in itertools.product((100, 300, 800), (0.3, 0.7), (1, 5), range(R)):
            yield "C", n, 100, rho, snr, r, {}
    elif study == "c_simple":
        # the n = 100 slice of "ablation" (identical data sets), for the comparison with DART
        for rho, snr, r in itertools.product((0.3, 0.7), (1, 5), range(R)):
            yield "C", 100, 100, rho, snr, r, {}
    else:
        raise ValueError(study)
