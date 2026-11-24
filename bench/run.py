"""Runs the methods of Chapter 6 of the thesis on the simulated data sets and writes one JSON line
per (data set, method), with the AUPRC of the ranking and the FDR and TPR of the median
probability model. Rows already in the output file are skipped (failed rows are retried);
shards split the data sets across jobs.

    python -m bench.run --study simple --reps 100 --methods selectpfn --sp small=runs/small/final.pt --out results/simple
    python -m bench.run --study simple --reps 100 --methods dart --designs FRI_EQ,FRI_TOE --out results/simple_mcmc --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback

import numpy as np

from .designs import grid, make_dataset
from .metrics import ranking_metrics, selection_metrics

_SP = {}


def _selectpfn(ckpt, device, ds, predict):
    from pfn.inference import SelectPFN
    if (ckpt, device) not in _SP:
        _SP[(ckpt, device)] = SelectPFN.load(ckpt, device)
    sp = _SP[(ckpt, device)]
    q = sp.pips(ds.X, ds.y)
    out = {"score": q, "selected": q > 0.5}
    if predict:
        pr = sp.predict(ds.X, ds.y, ds.X_test, B=64, q=q, seed=int(ds.key[5]))
        out["rmse"] = float(np.sqrt(np.mean((pr.mean - ds.y_test) ** 2)))
        out["rmse_mpm"] = float(np.sqrt(np.mean((pr.mean_mpm - ds.y_test) ** 2)))
    return out


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                if not r.get("error"):
                    keys.add((r["id"], r["method"]))
            except Exception:
                pass
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--study", required=True, choices=["simple", "ablation", "c_simple"],
                    help="simple: Section 6.1; ablation, c_simple: Section 6.2 (networks, DART)")
    ap.add_argument("--methods", default="selectpfn",
                    help="selectpfn, bart, dart, spikeslab (linear designs)")
    ap.add_argument("--designs", default="", help="comma-separated subset of the study's designs")
    ap.add_argument("--sp", action="append", default=[], help="name=path of a SelectPFN checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--predict", action="store_true", help="SelectPFN: also held-out RMSE")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    designs = set(args.designs.split(",")) if args.designs else None

    ckpts = dict(kv.split("=", 1) for kv in args.sp)
    methods = []
    for m in args.methods.split(","):
        methods += [f"selectpfn:{k}" for k in ckpts] if m == "selectpfn" else [m]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"shard{args.shard:03d}.jsonl")
    done = done_keys(path)
    with open(path, "a") as out:
        for i, (d, n, p, rho, snr, rep, extra) in enumerate(grid(args.study, args.reps)):
            if i % args.nshards != args.shard:
                continue
            if designs and d not in designs:
                continue
            ds = make_dataset(d, n, p, rho, snr, rep, **extra)
            for m in methods:
                if (ds.id, m) in done:
                    continue
                row = {"study": args.study, "id": ds.id, "design": d, "n": n, "p": p, "rho": rho,
                       "snr": snr, "rep": rep, "method": m}
                t0 = time.time()
                try:
                    if m.startswith("selectpfn"):
                        o = _selectpfn(ckpts[m.split(":")[1]], args.device, ds, args.predict)
                    elif m == "spikeslab":
                        from . import linear
                        o = linear.fit_spike_slab(ds.X, ds.y, seed=rep)
                        if "diag" in o:
                            row["diag"] = o["diag"]
                    else:
                        from . import bart
                        o = bart.fit(m, ds.X, ds.y, seed=rep, B=8)
                        row["diag"] = o["diag"]
                    row["time"] = time.time() - t0
                    row.update(ranking_metrics(o["score"], ds.gamma))
                    row["score"] = [round(float(v), 5) for v in o["score"]]
                    if o.get("selected") is not None:          # scores only: no selection
                        row.update(selection_metrics(o["selected"], ds.gamma))
                    for k in ("rmse", "rmse_mpm"):
                        if k in o:
                            row[k] = o[k]
                    row["sigma"] = ds.sigma
                except Exception as e:                    # keep going; record the failure
                    row["error"] = repr(e)[:500]
                    if args.verbose:
                        traceback.print_exc()
                out.write(json.dumps(row) + "\n")
                out.flush()
                if args.verbose:
                    print(m, ds.id, {k: row.get(k) for k in ("auprc", "fdp", "time", "error")},
                          flush=True)


if __name__ == "__main__":
    main()
