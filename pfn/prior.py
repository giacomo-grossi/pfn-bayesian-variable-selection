"""Meta-prior of SelectPFN (Luo, 2026, Section 3.3 and Table 14 of Appendix A).

Every training task is generated hierarchically:
    (i)   task size N ~ LU(50, 1000), context fraction n/N ~ U(0.5, 0.9), p ~ LU(5, pmax);
    (ii)  inclusion indicators gamma from a Beta-Bernoulli prior;
    (iii) covariates from a Gaussian copula with one of four dependence structures and
          per-column marginal transforms, independent of (gamma, f);
    (iv)  a regression function f that uses the active columns only, drawn from a mixture
          of seven families;
    (v)   noise with a random signal-to-noise ratio (Gaussian, t3 or heteroscedastic).

Choices that the paper does not spell out are marked "our choice" and collected in
`PriorConfig`, so they can be changed in one place.

All randomness goes through a numpy Generator, so a task is reproducible from its seed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr  # standard normal cdf

# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------

FAMILIES = ("mlp", "additive", "pairwise", "tree", "gp", "linear", "interaction")

#: Table 14 family weights.
DEFAULT_FAMILY_WEIGHTS = {
    "mlp": 0.30,
    "additive": 0.12,
    "pairwise": 0.12,
    "tree": 0.10,
    "gp": 0.10,
    "linear": 0.14,
    "interaction": 0.12,
}

#: Ablation of Section 5.4.4: no interaction-only family, no weak-effect regime, and every
#: interacting variable of the pairwise family keeps its main effect.
ABLATION_FAMILY_WEIGHTS = {
    "mlp": 0.40,
    "additive": 0.15,
    "pairwise": 0.15,
    "tree": 0.10,
    "gp": 0.10,
    "linear": 0.10,
    "interaction": 0.0,
}


@dataclass
class PriorConfig:
    # (i) sizes
    N_range: tuple = (50, 1000)            # rows per task, log-uniform
    ctx_frac_range: tuple = (0.5, 0.9)     # fraction of context rows
    p_min: int = 5
    p_max: int = 50                        # curriculum: 50 -> 200 -> 500
    # (ii) sparsity
    mean_active_max: float = 20.0          # E|S| ~ LU(1, min(20, p/3))
    nu_range: tuple = (2.0, 50.0)
    max_active: int = 40
    redraw_empty_prob: float = 0.9
    # (iii) covariates (our choices for the parameters the paper does not give)
    dependence_weights: dict = field(default_factory=lambda: {
        "indep": 0.25, "ar1": 0.25, "block": 0.25, "factor": 0.25})
    rho_range: tuple = (0.0, 0.9)
    block_size_range: tuple = (2, 20)
    n_factors_range: tuple = (1, 5)
    marginal_weights: dict = field(default_factory=lambda: {
        "identity": 0.40, "uniform": 0.15, "lognormal": 0.15, "tukeyh": 0.15,
        "discrete": 0.15})
    tukey_h_range: tuple = (0.1, 0.4)
    lognormal_sigma_range: tuple = (0.3, 1.0)
    discrete_levels_range: tuple = (2, 10)
    n_reference: int = 512                 # reference rows for scaling and checks
    # (iv) functions
    family_weights: dict = field(default_factory=lambda: dict(DEFAULT_FAMILY_WEIGHTS))
    weak_regime_prob: float = 0.5          # effect sizes LU(0.1, 1) w.p. 0.5, else U(0.3, 1)
    pairwise_drop_main_prob: float = 0.3   # 0 in the ablation
    interaction_weak_main_prob: float = 0.3
    effect_check_rows: int = 256
    effect_threshold: float = 0.01
    effect_max_redraws: int = 4
    # (v) noise
    snr_range: tuple = (0.25, 20.0)
    noise_weights: dict = field(default_factory=lambda: {
        "gaussian": 0.6, "t3": 0.2, "hetero": 0.2})

    @staticmethod
    def ablation(**kw) -> "PriorConfig":
        cfg = PriorConfig(family_weights=dict(ABLATION_FAMILY_WEIGHTS),
                          weak_regime_prob=0.0, pairwise_drop_main_prob=0.0, **kw)
        return cfg

    def replace(self, **kw) -> "PriorConfig":
        return dataclasses.replace(self, **kw)


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def log_uniform(rng, a, b, size=None):
    return np.exp(rng.uniform(np.log(a), np.log(b), size=size))


def _choice(rng, weights: dict):
    keys = list(weights)
    w = np.array([weights[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=w / w.sum())]


def _standardize_cols(A, ref=None, eps=1e-8):
    """Centre and scale columns of A with the statistics of `ref` (default: A)."""
    ref = A if ref is None else ref
    mu = ref.mean(0)
    sd = ref.std(0)
    sd = np.where(sd < eps, 1.0, sd)
    return (A - mu) / sd


class _Component:
    """A function component centred and scaled to unit variance on the reference sample."""

    def __init__(self, fn, Xref_s):
        self.fn = fn
        v = fn(Xref_s)
        self.mu = float(v.mean())
        sd = float(v.std())
        self.sd = sd if sd > 1e-8 else 1.0

    def __call__(self, Xs):
        return (self.fn(Xs) - self.mu) / self.sd


# --------------------------------------------------------------------------------------
# (iii) covariate law
# --------------------------------------------------------------------------------------

@dataclass
class CovariateLaw:
    """Gaussian copula: latent Z ~ N(0, Sigma), then x_j = T_j(z_j)."""
    Sigma: np.ndarray
    chol: np.ndarray
    transforms: list            # list of (name, params) per column
    dependence: str

    def sample_latent(self, rng, n):
        E = rng.standard_normal((n, self.Sigma.shape[0]))
        return E @ self.chol.T

    def transform(self, Z):
        X = np.empty_like(Z)
        for j, (name, par) in enumerate(self.transforms):
            z = Z[:, j]
            if name == "identity":
                X[:, j] = z
            elif name == "uniform":
                X[:, j] = ndtr(z)
            elif name == "lognormal":
                X[:, j] = np.exp(par * z)
            elif name == "tukeyh":
                X[:, j] = z * np.exp(par * z * z / 2.0)
            elif name == "discrete":
                X[:, j] = np.floor(ndtr(z) * par)
            else:  # pragma: no cover
                raise ValueError(name)
        return X


def _latent_correlation(rng, p, dependence, cfg: PriorConfig):
    if dependence == "indep":
        return np.eye(p)
    rho = rng.uniform(*cfg.rho_range)
    if dependence == "ar1":
        idx = np.arange(p)
        return rho ** np.abs(idx[:, None] - idx[None, :])
    if dependence == "block":
        S = np.eye(p)
        start = 0
        while start < p:
            b = int(rng.integers(cfg.block_size_range[0], cfg.block_size_range[1] + 1))
            stop = min(p, start + b)
            r = rng.uniform(*cfg.rho_range)
            S[start:stop, start:stop] = r
            start = stop
        np.fill_diagonal(S, 1.0)
        return S
    if dependence == "factor":
        # our choice: K factors, each column loads with equal weight on one or two of them
        K = int(rng.integers(cfg.n_factors_range[0], cfg.n_factors_range[1] + 1))
        L = np.zeros((p, K))
        for j in range(p):
            k = rng.choice(K, size=min(K, int(rng.integers(1, 3))), replace=False)
            L[j, k] = 1.0
        L /= np.sqrt(L.sum(1, keepdims=True))
        S = rho * (L @ L.T) + (1.0 - rho) * np.eye(p)
        d = np.sqrt(np.diag(S))
        return S / d[:, None] / d[None, :]
    raise ValueError(dependence)


def sample_covariate_law(rng, p, cfg: PriorConfig) -> CovariateLaw:
    dependence = _choice(rng, cfg.dependence_weights)
    Sigma = _latent_correlation(rng, p, dependence, cfg)
    chol = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
    transforms = []
    for _ in range(p):
        name = _choice(rng, cfg.marginal_weights)
        if name == "tukeyh":
            par = rng.uniform(*cfg.tukey_h_range)
        elif name == "lognormal":
            par = rng.uniform(*cfg.lognormal_sigma_range)
        elif name == "discrete":
            par = int(rng.integers(cfg.discrete_levels_range[0], cfg.discrete_levels_range[1] + 1))
        else:
            par = None
        transforms.append((name, par))
    return CovariateLaw(Sigma=Sigma, chol=chol, transforms=transforms, dependence=dependence)


# --------------------------------------------------------------------------------------
# (iv) function families. Each builder takes the standardized reference rows of the
# active columns (R x s) and returns a callable on (rows x s).
# --------------------------------------------------------------------------------------

def _effect_sizes(rng, s, weak_regime):
    if weak_regime:
        return log_uniform(rng, 0.1, 1.0, size=s)
    return rng.uniform(0.3, 1.0, size=s)


def _g1(rng, Xref_col):
    a = rng.normal(0.0, np.sqrt(1.0 / 8.0), size=8)
    c = log_uniform(rng, 0.5, 3.0)
    xi = rng.normal(0.0, c, size=8)
    th = rng.uniform(0.0, 2 * np.pi, size=8)

    def fn(u):
        return np.cos(u[:, None] * xi[None, :] + th[None, :]) @ a

    return _Component(lambda X, fn=fn: fn(X[:, 0]), Xref_col[:, None])


def _g2(rng, Xref_pair):
    a = rng.normal(0.0, np.sqrt(1.0 / 12.0), size=12)
    c = log_uniform(rng, 0.5, 2.0)
    xi = rng.normal(0.0, c, size=(12, 2))
    th = rng.uniform(0.0, 2 * np.pi, size=12)

    def fn(X):
        return np.cos(X @ xi.T + th[None, :]) @ a

    return _Component(fn, Xref_pair)


class _Sum:
    """f(Xs) = sum_k w_k * component_k(Xs[:, cols_k])."""

    def __init__(self):
        self.terms = []  # (weight, cols, component)

    def add(self, w, cols, comp):
        self.terms.append((float(w), np.asarray(cols, dtype=int), comp))

    def __call__(self, Xs):
        out = np.zeros(Xs.shape[0])
        for w, cols, comp in self.terms:
            out += w * comp(Xs[:, cols])
        return out


def build_linear(rng, Xref, cfg, weak):
    s = Xref.shape[1]
    beta = _effect_sizes(rng, s, weak) * rng.choice([-1.0, 1.0], size=s)
    return lambda Xs: Xs @ beta


def build_additive(rng, Xref, cfg, weak):
    s = Xref.shape[1]
    beta = _effect_sizes(rng, s, weak)
    f = _Sum()
    for j in range(s):
        f.add(beta[j], [j], _g1(rng, Xref[:, j]))
    return f


def build_pairwise(rng, Xref, cfg, weak):
    s = Xref.shape[1]
    beta = _effect_sizes(rng, s, weak)
    if s == 1:
        return build_additive(rng, Xref, cfg, weak)
    main = rng.uniform(size=s) >= cfg.pairwise_drop_main_prob
    n_pairs = int(rng.integers(1, s + 1))
    pairs = set()
    for _ in range(n_pairs):
        a, b = rng.choice(s, size=2, replace=False)
        pairs.add((min(a, b), max(a, b)))
    # a variable without main effect gets at least one interaction partner
    in_pair = {v for pr in pairs for v in pr}
    for j in range(s):
        if not main[j] and j not in in_pair:
            k = int(rng.choice([v for v in range(s) if v != j]))
            pairs.add((min(j, k), max(j, k)))
            in_pair.update((j, k))
    f = _Sum()
    for j in range(s):
        if main[j]:
            f.add(beta[j], [j], _g1(rng, Xref[:, j]))
    for a, b in sorted(pairs):
        w = log_uniform(rng, 0.5, 1.5)
        if rng.uniform() < 0.5:
            comp = _g2(rng, Xref[:, [a, b]])
        else:
            comp = _Component(lambda X: X[:, 0] * X[:, 1], Xref[:, [a, b]])
        f.add(w, [a, b], comp)
    return f


def _odd_pair_fn(rng):
    kind = int(rng.integers(4))
    c = log_uniform(rng, 0.5, 2.0)
    if kind == 0:
        return lambda X: X[:, 0] * X[:, 1]
    if kind == 1:
        return lambda X: np.sin(c * X[:, 0] * X[:, 1])
    if kind == 2:
        return lambda X: np.tanh(c * X[:, 0]) * np.tanh(c * X[:, 1])
    return lambda X: np.sign(X[:, 0]) * np.sign(X[:, 1])


def build_interaction(rng, Xref, cfg, weak):
    s = Xref.shape[1]
    if s == 1:
        f = _Sum()
        f.add(1.0, [0], _g1(rng, Xref[:, 0]))
        return f
    order = list(rng.permutation(s))
    groups = []
    while order:
        r = len(order)
        if r >= 3 and (r == 3 or rng.uniform() < 0.2):     # our choice: triples w.p. 0.2
            groups.append([order.pop(), order.pop(), order.pop()])
        elif r >= 2:
            groups.append([order.pop(), order.pop()])
        else:
            # a leftover variable joins the last group without entering it; the effect
            # check then removes it from the active set (as described in Appendix A)
            groups[-1].append(None)
            groups[-1].append(order.pop())
    f = _Sum()
    for g in groups:
        w = log_uniform(rng, 0.3, 1.0)
        pf = _odd_pair_fn(rng)
        cols = [v for v in g if v is not None]
        if None in g:
            cols = cols[:-1]          # the leftover does not enter
        if len(cols) == 2:
            f.add(w, cols, _Component(pf, Xref[:, cols]))
        else:  # triple: odd pair function times tanh of the third variable
            comp = _Component(lambda X, pf=pf: pf(X[:, :2]) * np.tanh(X[:, 2]), Xref[:, cols])
            f.add(w, cols, comp)
    if rng.uniform() < cfg.interaction_weak_main_prob:
        for j in range(s):
            f.add(log_uniform(rng, 0.05, 0.3), [j], _g1(rng, Xref[:, j]))
    return f


_ACTS = {
    "tanh": np.tanh,
    "relu": lambda z: np.maximum(z, 0.0),
    "sin": np.sin,
    "gelu": lambda z: 0.5 * z * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3))),
    "softplus": lambda z: np.logaddexp(0.0, z),
}


def build_mlp(rng, Xref, cfg, weak):
    """Sparse-input network, eq. (3): first-layer rows N(0, 1/s) for active inputs."""
    s = Xref.shape[1]
    depth = int(rng.integers(1, 4))
    widths = [int(rng.integers(8, 65)) for _ in range(depth)]
    layers = []
    fan_in = s
    h = Xref
    for li, H in enumerate(widths):
        sd = np.sqrt(1.0 / s) if li == 0 else np.sqrt(1.0 / fan_in)
        W = rng.normal(0.0, sd, size=(fan_in, H))
        b = rng.normal(0.0, 0.5, size=H)              # N(0, 0.25)
        act = _ACTS[list(_ACTS)[int(rng.integers(len(_ACTS)))]]
        z = h @ W + b
        mu, sz = z.mean(0), z.std(0)
        sz = np.where(sz < 1e-8, 1.0, sz)
        layers.append((W, b, mu, sz, act))
        h = act((z - mu) / sz)
        fan_in = H
    w_out = rng.normal(0.0, np.sqrt(1.0 / fan_in), size=fan_in)

    def fn(Xs):
        h = Xs
        for W, b, mu, sz, act in layers:
            h = act((h @ W + b - mu) / sz)
        return h @ w_out

    return fn


def build_tree(rng, Xref, cfg, weak):
    s = Xref.shape[1]
    trees = []
    used = set()
    n_trees = int(rng.integers(1, 6))
    while len(trees) < n_trees or (len(used) < s and len(trees) < 64):
        r = int(rng.integers(1, 5))
        n_int = 2 ** r - 1
        var = rng.integers(0, s, size=n_int)
        unused = [v for v in range(s) if v not in used]
        if unused:                         # make sure every active variable is a split variable
            var[0] = rng.choice(unused)
        thr = np.array([Xref[rng.integers(Xref.shape[0]), v] for v in var])
        leaves = rng.standard_normal(2 ** r)
        used.update(int(v) for v in var)
        trees.append((r, var, thr, leaves))

    def fn(Xs):
        out = np.zeros(Xs.shape[0])
        for r, var, thr, leaves in trees:
            node = np.zeros(Xs.shape[0], dtype=int)
            for _ in range(r):
                go_right = Xs[np.arange(Xs.shape[0]), var[node]] > thr[node]
                node = 2 * node + 1 + go_right
            out += leaves[node - (2 ** r - 1)]
        return out

    return fn


def draw_gp(rng, Xs):
    """f at the rows of the task from a GP with unit variance and SE-ARD kernel."""
    s = Xs.shape[1]
    ell = log_uniform(rng, 0.3, 2.5, size=s)
    Z = Xs / ell
    sq = (Z ** 2).sum(1)
    K = np.exp(-0.5 * np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0))
    K[np.diag_indices_from(K)] += 1e-6
    L = np.linalg.cholesky(K)
    return L @ rng.standard_normal(Xs.shape[0])


_BUILDERS = {
    "linear": build_linear,
    "additive": build_additive,
    "pairwise": build_pairwise,
    "interaction": build_interaction,
    "mlp": build_mlp,
    "tree": build_tree,
}


def total_effects(f, Xref_s, rng):
    """Permutation estimate of each active variable's total effect,
    1/2 E[(f(x) - f(x with x_j permuted))^2] / Var f (Table 14)."""
    base = f(Xref_s)
    var = base.var()
    if var < 1e-12:
        return np.zeros(Xref_s.shape[1])
    eff = np.empty(Xref_s.shape[1])
    for j in range(Xref_s.shape[1]):
        Xp = Xref_s.copy()
        Xp[:, j] = Xp[rng.permutation(Xp.shape[0]), j]
        eff[j] = 0.5 * np.mean((base - f(Xp)) ** 2) / var
    return eff


# --------------------------------------------------------------------------------------
# a single task
# --------------------------------------------------------------------------------------

@dataclass
class Task:
    X: np.ndarray          # (N, p) standardized covariates
    y: np.ndarray          # (N,)   response, standardized with context statistics
    gamma: np.ndarray      # (p,)   inclusion indicators
    n: int                 # number of context (training) rows; the rest are query rows
    family: str
    dependence: str
    snr: float
    noise: str


def sample_sizes(rng, cfg: PriorConfig):
    """(N, n, p) shared by all tasks of a batch."""
    N = int(round(log_uniform(rng, *cfg.N_range)))
    n = int(round(N * rng.uniform(*cfg.ctx_frac_range)))
    n = min(max(n, 2), N - 1)
    p = int(round(log_uniform(rng, cfg.p_min, cfg.p_max)))
    return N, n, max(p, 1)


def sample_gamma(rng, p, cfg: PriorConfig):
    hi = max(1.0, min(cfg.mean_active_max, p / 3.0))
    while True:
        mean_active = log_uniform(rng, 1.0, hi) if hi > 1.0 else 1.0
        wbar = min(mean_active / p, 0.999)
        nu = log_uniform(rng, *cfg.nu_range)
        w = rng.beta(wbar * nu, (1.0 - wbar) * nu)
        gamma = (rng.uniform(size=p) < w).astype(np.int64)
        if gamma.sum() > cfg.max_active:
            on = np.flatnonzero(gamma)
            gamma[:] = 0
            gamma[rng.choice(on, size=cfg.max_active, replace=False)] = 1
        if gamma.sum() > 0 or rng.uniform() >= cfg.redraw_empty_prob:
            return gamma


def _add_noise(rng, f_task, Xs_task, cfg: PriorConfig):
    snr = float(log_uniform(rng, *cfg.snr_range))
    var_f = float(f_task.var())
    sigma = np.sqrt(var_f / snr) if var_f > 1e-12 else 1.0
    kind = _choice(rng, cfg.noise_weights)
    N = f_task.shape[0]
    if kind == "gaussian":
        eps = rng.standard_normal(N)
    elif kind == "t3":
        eps = rng.standard_t(3, size=N) / np.sqrt(3.0)
    else:  # heteroscedastic: log-sd linear in one covariate (our choice of form)
        if Xs_task.shape[1] > 0:
            k = int(rng.integers(Xs_task.shape[1]))
            a = rng.uniform(0.5, 1.5)
            sd = np.exp(0.5 * a * np.clip(Xs_task[:, k], -4, 4))
            sd /= np.sqrt(np.mean(sd ** 2))
        else:
            sd = np.ones(N)
        eps = sd * rng.standard_normal(N)
    return f_task + sigma * eps, snr, kind


def sample_task(rng, N, n, p, cfg: PriorConfig) -> Task:
    law = sample_covariate_law(rng, p, cfg)
    gamma = sample_gamma(rng, p, cfg)
    R = cfg.n_reference
    Z_task = law.sample_latent(rng, N)
    Z_ref = law.sample_latent(rng, R)
    X_task_raw = law.transform(Z_task)
    X_ref_raw = law.transform(Z_ref)

    both = np.vstack([X_task_raw, X_ref_raw])
    X_task = _standardize_cols(X_task_raw, both)
    X_ref = _standardize_cols(X_ref_raw, both)

    active = np.flatnonzero(gamma)
    family = _choice(rng, cfg.family_weights) if active.size else "null"
    weak = rng.uniform() < cfg.weak_regime_prob

    if active.size == 0:
        f_task = np.zeros(N)
    elif family == "gp":
        f_task = draw_gp(rng, X_task[:, active])         # GP draws skip the effect check
    else:
        Xr = X_ref[:, active]
        Xchk = Xr[: cfg.effect_check_rows]
        frozen = np.zeros(active.size, dtype=bool)
        f = None
        for attempt in range(cfg.effect_max_redraws + 1):
            f = _BUILDERS[family](rng, Xr, cfg, weak)
            eff = total_effects(f, Xchk, rng)
            if np.all(eff >= cfg.effect_threshold):
                break
            if attempt == cfg.effect_max_redraws:
                frozen = eff < cfg.effect_threshold        # freeze weak variables at 0
        fz = frozen

        def f_frozen(Xs, f=f, fz=fz):
            if fz.any():
                Xs = Xs.copy()
                Xs[:, fz] = 0.0
            return f(Xs)

        f_task = f_frozen(X_task[:, active])
        gamma = gamma.copy()
        gamma[active[frozen]] = 0                          # relabel frozen variables inactive

    y, snr, noise = _add_noise(rng, f_task, X_task[:, active], cfg)
    mu, sd = y[:n].mean(), y[:n].std()
    y = (y - mu) / (sd if sd > 1e-8 else 1.0)

    return Task(X=X_task.astype(np.float32), y=y.astype(np.float32), gamma=gamma, n=n,
                family=family, dependence=law.dependence, snr=snr, noise=noise)


def sample_batch(rng, cfg: PriorConfig, max_cells: float = 3.5e5, max_batch: int = 64,
                 sizes=None):
    """A batch of tasks sharing (N, n, p); batch size keeps about `max_cells` cells."""
    N, n, p = sample_sizes(rng, cfg) if sizes is None else sizes
    B = int(max(1, min(max_batch, max_cells // (N * (p + 1)))))
    tasks = [sample_task(rng, N, n, p, cfg) for _ in range(B)]
    return {
        "X": np.stack([t.X for t in tasks]),
        "y": np.stack([t.y for t in tasks]),
        "gamma": np.stack([t.gamma for t in tasks]).astype(np.float32),
        "n": n,
        "family": [t.family for t in tasks],
    }
