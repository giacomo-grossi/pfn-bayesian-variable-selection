"""SelectPFN network (Luo, 2026, Sections 3.2 and 3.5, Appendix A).

A table transformer with alternating attention within rows (over the cells of a row) and
within columns (over rows; keys and values from context rows only). There is no embedding of
column identity, so the network is equivariant to permutations of the feature columns.

The same backbone runs two passes, told apart by a learned pass token:
  * selection pass: the context rows (with responses) of all p columns; a readout pools each
    feature column over the context rows and returns a logit s_j, q_j = sigmoid(s_j);
  * prediction pass: only the columns of a subset S (excluded columns are removed, not zeroed),
    context and query rows; the target cell of each query row is read by a bar-distribution
    head over the standardized response.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# --------------------------------------------------------------------------------------
# bar distribution with half-normal tails (Mueller et al., 2022)
# --------------------------------------------------------------------------------------

_HALFNORMAL_MEDIAN = 0.6744897501960817  # Phi^{-1}(0.75)


class BarDistribution(nn.Module):
    """Piecewise-constant density on fixed bins, with half-normal outer tails."""

    def __init__(self, borders: torch.Tensor):
        super().__init__()
        borders = torch.as_tensor(borders, dtype=torch.float32)
        self.register_buffer("borders", borders)
        w = borders[1:] - borders[:-1]
        self.register_buffer("widths", w)
        # tail scales: the half-normal has its median at the width of the outer bin
        self.register_buffer("tail_scale", torch.stack([w[0], w[-1]]) / _HALFNORMAL_MEDIAN)

    @property
    def n_bins(self):
        return self.widths.numel()

    def log_prob(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """logits (..., M), y (...) -> log density (...)."""
        logits = logits.to(self.borders.device).float()
        y = y.to(self.borders.device)
        logp = torch.log_softmax(logits, dim=-1)
        b = self.borders
        M = self.n_bins
        idx = torch.searchsorted(b.contiguous(), y.contiguous().float(), right=True) - 1
        idx = idx.clamp(0, M - 1)
        lp_bin = logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        out = lp_bin - torch.log(self.widths[idx])
        # tails: bin 0 covers (-inf, b1], bin M-1 covers [b_{M-1}, inf)
        sL, sR = self.tail_scale[0], self.tail_scale[1]
        left = idx == 0
        right = idx == M - 1
        zl = (b[1] - y).clamp_min(0.0) / sL
        zr = (y - b[M - 1]).clamp_min(0.0) / sR
        hn_l = math.log(2.0) - 0.5 * math.log(2 * math.pi) - torch.log(sL) - 0.5 * zl ** 2
        hn_r = math.log(2.0) - 0.5 * math.log(2 * math.pi) - torch.log(sR) - 0.5 * zr ** 2
        out = torch.where(left, lp_bin + hn_l, out)
        out = torch.where(right, lp_bin + hn_r, out)
        return out

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        p = torch.softmax(logits.to(self.borders.device).float(), dim=-1)
        b = self.borders
        centers = (b[1:] + b[:-1]) / 2
        c = centers.clone()
        c[0] = b[1] - self.tail_scale[0] * math.sqrt(2 / math.pi)
        c[-1] = b[-2] + self.tail_scale[1] * math.sqrt(2 / math.pi)
        return (p * c).sum(-1)

    def cdf(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """CDF at y (used for CRPS and quantiles)."""
        p = torch.softmax(logits.to(self.borders.device).float(), dim=-1)
        y = y.to(self.borders.device)
        b = self.borders
        M = self.n_bins
        cum = torch.cumsum(p, -1)
        idx = (torch.searchsorted(b.contiguous(), y.contiguous().float(), right=True) - 1).clamp(0, M - 1)
        below = torch.where(idx > 0, cum.gather(-1, (idx - 1).clamp_min(0).unsqueeze(-1)).squeeze(-1),
                            torch.zeros_like(y, dtype=torch.float32))
        pk = p.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        frac = ((y - b[idx]) / self.widths[idx]).clamp(0, 1)
        out = below + pk * frac
        # tails
        sL, sR = self.tail_scale
        zl = (b[1] - y).clamp_min(0) / sL
        zr = (y - b[M - 1]).clamp_min(0) / sR
        left_mass = pk * (1 - torch.erf(zl / math.sqrt(2)))   # mass of left tail below y
        out = torch.where(idx == 0, left_mass, out)
        out = torch.where(idx == M - 1, below + pk * torch.erf(zr / math.sqrt(2)), out)
        return out


# --------------------------------------------------------------------------------------
# backbone
# --------------------------------------------------------------------------------------

class _MHA(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, kv=None, mask=None):
        """x (L, Tq, d); kv (L, Tk, d) defaults to x; mask (L or 1, 1, 1, Tk) bool, True = keep."""
        kv = x if kv is None else kv
        L, Tq, d = x.shape
        Tk = kv.shape[1]
        h = self.h
        q = self.q(x).view(L, Tq, h, d // h).transpose(1, 2)
        k, v = self.kv(kv).view(L, Tk, 2, h, d // h).unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(L, Tq, d))


class TableLayer(nn.Module):
    """Pre-LN layer: feature attention within rows, then row attention within columns, MLP."""

    def __init__(self, d, heads, mlp_ratio=4):
        super().__init__()
        self.ln_f = nn.LayerNorm(d)
        self.att_f = _MHA(d, heads)
        self.ln_r = nn.LayerNorm(d)
        self.att_r = _MHA(d, heads)
        self.ln_m = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, h, n_ctx: int, col_mask=None):
        """h (B, N, C, d); col_mask (B, C) bool, True = real column (None = all real)."""
        B, N, C, d = h.shape
        # attention within each row, over its C cells
        x = self.ln_f(h).reshape(B * N, C, d)
        m = None
        if col_mask is not None:
            m = col_mask[:, None, :].expand(B, N, C).reshape(B * N, 1, 1, C)
        h = h + self.att_f(x, mask=m).view(B, N, C, d)
        # attention within each column, over rows; keys and values from context rows only
        x = self.ln_r(h).transpose(1, 2).reshape(B * C, N, d)
        h = h + self.att_r(x, kv=x[:, :n_ctx]).view(B, C, N, d).transpose(1, 2)
        h = h + self.mlp(self.ln_m(h))
        return h


class _AttnPool(nn.Module):
    """Learned-query attention pooling over rows."""

    def __init__(self, d):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)

    def forward(self, x):  # x (..., R, d) -> (..., d)
        s = (self.k(x) @ self.query) / math.sqrt(x.shape[-1])
        a = torch.softmax(s.float(), dim=-1).to(x.dtype)
        return (a.unsqueeze(-1) * self.v(x)).sum(-2)


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class SelectPFNModel(nn.Module):
    def __init__(self, d=128, n_layers=6, n_heads=4, borders=None, n_bins=256,
                 mlp_ratio=4, grad_checkpoint=False):
        super().__init__()
        self.d = d
        self.grad_checkpoint = grad_checkpoint
        self.enc_x = _mlp(1, d, d)
        self.enc_y = _mlp(1, d, d)
        self.miss_token = nn.Parameter(torch.zeros(d))
        self.target_col = nn.Parameter(torch.randn(d) * 0.02)
        self.pass_emb = nn.Embedding(2, d)        # 0 = selection, 1 = prediction
        self.layers = nn.ModuleList([TableLayer(d, n_heads, mlp_ratio) for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(d)
        # selection readout
        self.pool_x = _AttnPool(d)
        self.pool_y = _AttnPool(d)
        self.glob = _mlp(2, d, d)
        self.sel_head = _mlp(5 * d, d, 1)
        # prediction readout
        if borders is None:
            borders = torch.linspace(-4, 4, n_bins + 1)
        self.bar = BarDistribution(borders)
        self.pred_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, self.bar.n_bins))

    # ----------------------------------------------------------------------------------
    def _embed(self, X, y, n_ctx, pass_id):
        """X (B, N, C), y (B, N) (query entries ignored) -> cells (B, N, C+1, d)."""
        B, N, C = X.shape
        is_ctx = (torch.arange(N, device=X.device) < n_ctx).view(1, N, 1)
        y = torch.where(is_ctx[..., 0], y, torch.zeros_like(y))   # query responses never enter
        hx = self.enc_x(X.unsqueeze(-1))                         # (B, N, C, d)
        ey = self.enc_y(y.unsqueeze(-1))                         # (B, N, d)
        ey_ctx = ey * is_ctx                                     # response added to context rows
        hx = hx + ey_ctx.unsqueeze(2)
        tgt = torch.where(is_ctx, ey, self.miss_token.view(1, 1, -1).expand(B, N, -1))
        tgt = tgt + self.target_col
        h = torch.cat([hx, tgt.unsqueeze(2)], dim=2)
        return h + self.pass_emb.weight[pass_id]

    def _backbone(self, h, n_ctx, col_mask):
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                h = checkpoint(layer, h, n_ctx, col_mask, use_reentrant=False)
            else:
                h = layer(h, n_ctx, col_mask)
        return self.ln_out(h)

    # ----------------------------------------------------------------------------------
    def select(self, X, y, n_ctx=None):
        """Selection pass. X (B, n, p) context rows, y (B, n) -> logits (B, p)."""
        B, n, p = X.shape
        n_ctx = n if n_ctx is None else n_ctx
        X, y = X[:, :n_ctx], y[:, :n_ctx]
        h = self._backbone(self._embed(X, y, n_ctx, 0), n_ctx, None)
        hx, hy = h[:, :, :p], h[:, :, p]                       # (B, n, p, d), (B, n, d)
        hx = hx.transpose(1, 2)                                 # (B, p, n, d)
        mean_x = hx.mean(2)
        att_x = self.pool_x(hx)
        mean_y = hy.mean(1)
        att_y = self.pool_y(hy)
        g = self.glob(torch.tensor([math.log(n_ctx), math.log(p)], device=X.device,
                                   dtype=h.dtype)).expand(B, -1)
        feat = torch.cat([mean_x, att_x,
                          mean_y[:, None].expand(-1, p, -1), att_y[:, None].expand(-1, p, -1),
                          g[:, None].expand(-1, p, -1)], dim=-1)
        return self.sel_head(feat).squeeze(-1)

    def predict(self, X, y, n_ctx, subset):
        """Prediction pass on subsets.

        X (B, N, p), y (B, N) (query responses unused), subset (B, p) bool.
        Excluded columns are removed: each task's selected columns are gathered into a padded
        table and padded cells are masked out of the attention within rows, so the output does
        not depend on excluded columns (eq. 4). Returns logits for query rows (B, N - n_ctx, M).
        """
        B, N, p = X.shape
        k = subset.sum(1)
        kmax = int(k.max().item()) if B else 0
        order = torch.argsort((~subset).to(torch.int8), dim=1, stable=True)[:, :kmax]   # selected first
        Xs = torch.gather(X, 2, order.unsqueeze(1).expand(B, N, kmax)) if kmax else X[:, :, :0]
        real = torch.arange(kmax, device=X.device)[None, :] < k[:, None]                # (B, kmax)
        Xs = torch.where(real[:, None, :], Xs, torch.zeros_like(Xs))
        col_mask = torch.cat([real, torch.ones(B, 1, dtype=torch.bool, device=X.device)], 1)
        h = self._backbone(self._embed(Xs, y, n_ctx, 1), n_ctx, col_mask)
        return self.pred_head(h[:, n_ctx:, kmax])

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
