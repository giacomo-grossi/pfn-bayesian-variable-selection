"""Training of SelectPFN (Luo, 2026, Section 3.6, Algorithm 1, Appendix A).

Objective (eq. 5): L = L_sel + alpha * L_pred, with
  L_sel  = mean binary log loss of the selection head against gamma,
  L_pred = mean negative log density (bar distribution) of query responses, from a prediction
           pass on a subset drawn from the kernel
           K(. | gamma, q) = rho1 * delta_S(gamma) + rho2 * C(. | S(gamma)) + (1-rho1-rho2) * prod Bern(sg(q_j)).

Usage (one GPU):
    python -m pfn.train --preset small --out runs/small          # the network of the thesis
    python -m pfn.train --preset ablation --out runs/ablation    # the ablated network (Section 6.2)
    python -m pfn.train --preset tiny --out runs/tiny --device cpu     # smoke test
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .data import TaskStream, estimate_borders, make_loader, p_max_at
from .model import SelectPFNModel
from .prior import PriorConfig


@dataclass
class TrainConfig:
    # model
    d: int = 128
    n_layers: int = 6
    n_heads: int = 4
    n_bins: int = 256
    grad_checkpoint: bool = False
    # optimization (Appendix A)
    steps: int = 80_000
    lr: float = 3e-4
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.01
    warmup_frac: float = 0.02
    clip: float = 1.0
    bf16: bool = True                # mixed precision; on GPUs without bf16 (T4) fp16 + GradScaler
    # objective and kernel
    alpha: float = 1.0
    rho1: float = 0.4
    rho2: float = 0.3
    self_sample_start: float = 0.10
    subset_cap: int = 64
    corrupt_drop: float = 0.1
    corrupt_add_mean: float = 1.0
    # data
    curriculum: tuple = (50, 200, 500)
    max_cells: float = 3.5e5
    max_batch: int = 64
    workers: int = -1                # -1: one data worker per CPU core
    prior: dict = field(default_factory=dict)   # overrides of PriorConfig
    # bookkeeping
    seed: int = 0
    log_every: int = 50
    ckpt_every: int = 1000
    out: str = "runs/run"
    init_from: str = ""


PRESETS = {
    "small": dict(d=128, n_layers=6, n_heads=4, steps=80_000),
    # ablated meta-prior (thesis Section 6.2; Luo, 2026, Section 5.4.4)
    "ablation": dict(d=128, n_layers=6, n_heads=4, steps=80_000, prior={"ablation": True}),
    # CPU smoke test
    "tiny": dict(d=32, n_layers=2, n_heads=4, steps=60, n_bins=64, workers=0, max_cells=2e4,
                 max_batch=8, curriculum=(20, 30, 40), bf16=False, log_every=10, ckpt_every=30),
}


def build_prior(tc: TrainConfig) -> PriorConfig:
    over = dict(tc.prior)
    ablation = over.pop("ablation", False)
    cfg = PriorConfig.ablation() if ablation else PriorConfig()
    return cfg.replace(**over) if over else cfg


# --------------------------------------------------------------------------------------
# kernel K(. | gamma, q)
# --------------------------------------------------------------------------------------

def sample_subsets(gamma: torch.Tensor, q: torch.Tensor, step: int, tc: TrainConfig,
                   gen: torch.Generator) -> torch.Tensor:
    """gamma, q (B, p) -> boolean subsets (B, p). q must already be detached."""
    B, p = gamma.shape
    dev = gamma.device
    use_self = step >= tc.self_sample_start * tc.steps
    w = torch.tensor([tc.rho1, tc.rho2, (1 - tc.rho1 - tc.rho2) if use_self else 0.0])
    comp = torch.multinomial(w / w.sum(), B, replacement=True, generator=gen).to(dev)
    true_s = gamma > 0.5
    # corrupted: drop each true element w.p. 0.1, add Poisson(1) inactive variables
    u = torch.rand(B, p, generator=gen).to(dev)
    corrupt = true_s & (u >= tc.corrupt_drop)
    n_add = torch.poisson(torch.full((B,), tc.corrupt_add_mean), generator=gen).to(dev)
    score = torch.rand(B, p, generator=gen).to(dev).masked_fill(true_s, -1.0)  # inactive only
    rank = score.argsort(1, descending=True).argsort(1)
    corrupt = corrupt | ((rank < n_add[:, None]) & ~true_s)
    # self-sampled from the current selection head
    selfs = torch.rand(B, p, generator=gen).to(dev) < q
    S = torch.where((comp == 0)[:, None], true_s, torch.where((comp == 1)[:, None], corrupt, selfs))
    # cap at subset_cap columns (random subsample)
    k = S.sum(1)
    if (k > tc.subset_cap).any():
        r = torch.rand(B, p, generator=gen).to(dev).masked_fill(~S, -1.0)
        keep = r.argsort(1, descending=True).argsort(1) < tc.subset_cap
        S = S & keep
    return S


def lr_at(step, tc: TrainConfig):
    warm = max(1, int(tc.warmup_frac * tc.steps))
    if step < warm:
        return tc.lr * (step + 1) / warm
    t = (step - warm) / max(1, tc.steps - warm)
    return tc.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


# --------------------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------------------

def save_ckpt(path, model, opt, step, tc: TrainConfig, scaler=None):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict() if opt else None,
                "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
                "step": step, "train_config": dataclasses.asdict(tc),
                "borders": model.bar.borders.cpu()}, tmp)
    os.replace(tmp, path)


def load_model(path, device="cpu") -> tuple[SelectPFNModel, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    tc = ck["train_config"]
    m = SelectPFNModel(tc["d"], tc["n_layers"], tc["n_heads"], borders=ck["borders"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


# --------------------------------------------------------------------------------------
# training loop
# --------------------------------------------------------------------------------------

def train(tc: TrainConfig, device=None, resume=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(tc.out, exist_ok=True)
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    prior = build_prior(tc)
    with open(os.path.join(tc.out, "config.json"), "w") as f:
        json.dump({"train": dataclasses.asdict(tc), "prior": dataclasses.asdict(prior)}, f,
                  indent=2, default=str)

    last = os.path.join(tc.out, "last.pt")
    start = 0
    if resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        borders = ck["borders"]
    elif tc.init_from:
        ck = torch.load(tc.init_from, map_location="cpu", weights_only=False)
        borders = ck["borders"]
    else:
        ck = None
        print("estimating bar-distribution borders from the prior ...", flush=True)
        borders = estimate_borders(prior, n_bins=tc.n_bins, curriculum=tc.curriculum)

    model = SelectPFNModel(tc.d, tc.n_layers, tc.n_heads, borders=borders,
                           grad_checkpoint=tc.grad_checkpoint).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, betas=tc.betas,
                            weight_decay=tc.weight_decay)
    if ck is not None:
        model.load_state_dict(ck["model"])
        if resume and os.path.exists(last):
            opt.load_state_dict(ck["opt"])
            start = int(ck["step"])
            print(f"resumed at step {start}", flush=True)
        else:
            print(f"warm start from {tc.init_from}", flush=True)
    print(f"model: {model.n_params() / 1e6:.2f}M parameters, device {device}", flush=True)

    stream = TaskStream(prior, tc.steps, curriculum=tc.curriculum, max_cells=tc.max_cells,
                        max_batch=tc.max_batch, seed=tc.seed + 1000 * start)
    stream.step_counter.value = start
    workers = tc.workers if tc.workers >= 0 else max(1, (os.cpu_count() or 2))
    it = iter(make_loader(stream, workers))
    gen = torch.Generator().manual_seed(tc.seed + start)
    use_amp = tc.bf16 and device.startswith("cuda")
    amp_dtype = torch.bfloat16
    if use_amp and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16            # e.g. Tesla T4 on Colab
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    if ck is not None and resume and os.path.exists(last) and ck.get("scaler") and scaler.is_enabled():
        scaler.load_state_dict(ck["scaler"])
    print(f"data workers {workers}; precision "
          f"{'fp32' if not use_amp else str(amp_dtype).split('.')[-1]}", flush=True)

    log_path = os.path.join(tc.out, "log.csv")
    new_log = not os.path.exists(log_path) or start == 0
    logf = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(logf)
    if new_log:
        log.writerow(["step", "lr", "loss", "l_sel", "l_pred", "pmax", "B", "N", "p", "sec"])
    t0 = time.time()
    t_last, wait = t0, 0.0        # wait: time spent waiting for the data workers
    acc = np.zeros(3)
    nacc = 0
    model.train()
    for step in range(start, tc.steps):
        stream.step_counter.value = step
        tw = time.time()
        b = next(it)
        wait += time.time() - tw
        X = b["X"].to(device, non_blocking=True)
        y = b["y"].to(device, non_blocking=True)
        g = b["gamma"].to(device, non_blocking=True)
        n = b["n"]
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, tc)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model.select(X[:, :n], y[:, :n])
            l_sel = F.binary_cross_entropy_with_logits(logits.float(), g)
            if tc.alpha > 0:
                q = torch.sigmoid(logits.detach().float())
                S = sample_subsets(g, q, step, tc, gen)
                out = model.predict(X, y, n, S)
                l_pred = -model.bar.log_prob(out, y[:, n:]).mean()
            else:
                l_pred = torch.zeros((), device=device)
            loss = l_sel + tc.alpha * l_pred
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.clip)
        scaler.step(opt)
        scaler.update()

        acc += [loss.item(), l_sel.item(), l_pred.item()]
        nacc += 1
        if (step + 1) % tc.log_every == 0:
            a = acc / nacc
            pmax = p_max_at(step, tc.steps, tc.curriculum)
            row = [step + 1, f"{lr_at(step, tc):.2e}", f"{a[0]:.4f}", f"{a[1]:.4f}", f"{a[2]:.4f}",
                   pmax, X.shape[0], X.shape[1], X.shape[2], f"{time.time() - t0:.0f}"]
            log.writerow(row)
            logf.flush()
            now = time.time()
            print("step {} lr {} loss {} sel {} pred {} pmax {} B {} N {} p {} t {}s".format(*row)
                  + f" | {(now - t_last) / tc.log_every:.3f} s/step, data wait "
                  f"{100 * wait / max(now - t_last, 1e-9):.0f}%", flush=True)
            t_last, wait = now, 0.0
            acc[:] = 0
            nacc = 0
        if (step + 1) % tc.ckpt_every == 0 or step + 1 == tc.steps:
            save_ckpt(last, model, opt, step + 1, tc, scaler)
    save_ckpt(os.path.join(tc.out, "final.pt"), model, None, tc.steps, tc)
    logf.close()
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--preset", default="small", choices=sorted(PRESETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, help="weight of the prediction loss (default 1)")
    ap.add_argument("--init-from", default="")
    ap.add_argument("--device")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)
    kw = dict(PRESETS[args.preset])
    if args.steps:
        kw["steps"] = args.steps
    if args.workers is not None:
        kw["workers"] = args.workers
    if args.alpha is not None:
        kw["alpha"] = args.alpha
    kw.update(seed=args.seed, out=args.out, init_from=args.init_from)
    tc = TrainConfig(**kw)
    train(tc, device=args.device, resume=not args.no_resume)


if __name__ == "__main__":
    main()
