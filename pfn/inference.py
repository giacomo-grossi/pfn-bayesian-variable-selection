"""Inference with a trained SelectPFN (Luo, 2026, Algorithm 2, Sections 3.7-3.9).

    sp = SelectPFN.load("runs/small/final.pt")
    q = sp.pips(X, y)                       # posterior inclusion probabilities
    S = sp.mpm(q)                           # median probability model, q_j > 1/2
    pred = sp.predict(X, y, X_test, B=64)   # model-averaged prediction over sampled subsets
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .train import load_model


def _standardize(X, Xref):
    mu = Xref.mean(0)
    sd = Xref.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd


@dataclass
class Prediction:
    mean: np.ndarray             # BMA predictive mean on the original scale
    mean_mpm: np.ndarray         # prediction from the MPM subset alone
    subsets: list                # distinct sampled subsets
    weights: np.ndarray          # frequency of each distinct subset
    logits: np.ndarray           # (n_subsets, m, M) bar-distribution logits (standardized scale)
    y_mu: float
    y_sd: float


class SelectPFN:
    def __init__(self, model, ckpt, device="cpu"):
        self.model = model
        self.ckpt = ckpt
        self.device = device
        tc = ckpt["train_config"]
        self.p_max = int(max(tc.get("curriculum", (500,))))

    @classmethod
    def load(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        m, ck = load_model(path, device)
        return cls(m, ck, device)

    # ----------------------------------------------------------------------------------
    def _prep(self, X, y, X_test=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        Xs = _standardize(X, X)
        Xt = None if X_test is None else _standardize(np.asarray(X_test, dtype=np.float64), X)
        mu, sd = y.mean(), y.std()
        sd = sd if sd > 1e-8 else 1.0
        return Xs, (y - mu) / sd, Xt, mu, sd

    @torch.no_grad()
    def _q_block(self, Xs, ys):
        X = torch.as_tensor(Xs, dtype=torch.float32, device=self.device)[None]
        y = torch.as_tensor(ys, dtype=torch.float32, device=self.device)[None]
        return torch.sigmoid(self.model.select(X, y)).squeeze(0).cpu().numpy()

    def pips(self, X, y, R=8, refine_k=None, seed=0):
        """Posterior inclusion probabilities. For p > p_max, average over R random partitions
        into blocks of at most p_max variables, then refine with one joint pass on the top-k
        (Section 3.9; implemented, like the paper, but not validated beyond p = 500)."""
        Xs, ys, _, _, _ = self._prep(X, y)
        p = Xs.shape[1]
        if p <= self.p_max:
            return self._q_block(Xs, ys)
        rng = np.random.default_rng(seed)
        q = np.zeros(p)
        for _ in range(R):
            perm = rng.permutation(p)
            nb = int(np.ceil(p / self.p_max))
            for blk in np.array_split(perm, nb):
                q[blk] += self._q_block(Xs[:, blk], ys)
        q /= R
        k = refine_k or self.p_max
        top = np.argsort(-q)[:k]
        q[top] = self._q_block(Xs[:, top], ys)
        return q

    @staticmethod
    def mpm(q):
        return np.asarray(q) > 0.5

    @torch.no_grad()
    def predict(self, X, y, X_test, B=64, q=None, seed=0, chunk=16):
        """Model-averaged predictive (BMA) over B subsets drawn from prod Bern(q_j);
        each prediction pass sees only the columns of its subset."""
        Xs, ys, Xt, mu, sd = self._prep(X, y, X_test)
        if q is None:
            q = self.pips(X, y)
        rng = np.random.default_rng(seed)
        draws = rng.uniform(size=(B, q.size)) < q[None, :]
        uniq, counts = np.unique(draws, axis=0, return_counts=True)
        mpm = self.mpm(q)
        subsets = list(uniq) + [mpm]
        n, m = Xs.shape[0], Xt.shape[0]
        Xall = np.vstack([Xs, Xt]).astype(np.float32)
        yall = np.concatenate([ys, np.zeros(m)]).astype(np.float32)
        logits = []
        for i in range(0, len(subsets), chunk):
            S = torch.as_tensor(np.stack(subsets[i:i + chunk]), device=self.device)
            k = S.shape[0]
            Xb = torch.as_tensor(Xall, device=self.device)[None].expand(k, -1, -1)
            yb = torch.as_tensor(yall, device=self.device)[None].expand(k, -1)
            logits.append(self.model.predict(Xb, yb, n, S).float().cpu())
        logits = torch.cat(logits)
        means = self.model.bar.mean(logits).cpu().numpy() * sd + mu     # (K+1, m)
        w = counts / counts.sum()
        return Prediction(mean=(w[:, None] * means[:-1]).sum(0), mean_mpm=means[-1],
                          subsets=list(uniq), weights=w, logits=logits[:-1].numpy(),
                          y_mu=mu, y_sd=sd)
