"""Tables of Section 6.1 of the thesis (study "simple"): linear designs (n = 50) and Friedman designs
(n = 100), p = 100, equicorrelated or Toeplitz covariance with rho in {0.3, 0.7}, SNR in {1, 5}.

Methods: rSelectPFN (all designs), spike-and-slab (linear designs), BART and DART (Friedman
designs).

Table 1, selection (per setting and method; standard errors over replicates in brackets)
    AUPRC        ranking of the variables by the method's score
    FDR, power   of the median probability model (score > 0.5). FDR = mean FDP, with
                 FDP = 0 when nothing is selected (Benjamini-Hochberg convention)
    size, empty  mean size of the selected set and share of empty selections
Table 2, paired AUPRC differences against rSelectPFN on the same data sets.

    python -m bench.simple_report --results results/simple_pfn,results/simple_ref [--save r.md]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .report import load

NAMES = {"selectpfn:small": "rSelectPFN", "bart": "BART", "dart": "DART",
         "spikeslab": "Spike-and-slab"}
ORDER = ["rSelectPFN", "Spike-and-slab", "DART", "BART"]
SIMPLE = ["LIN_EQ", "LIN_TOE", "FRI_EQ", "FRI_TOE"]


def rows(df):
    out = []
    for _, r in df.iterrows():
        rec = {"id": r["id"], "design": r["design"], "rho": r["rho"], "snr": r["snr"],
               "method": NAMES.get(r["method"], r["method"]), "AUPRC": r["auprc"]}
        k = r.get("n_selected", np.nan)
        if pd.notna(k):                       # methods that select
            k = int(k)
            rec.update({"FDR": r["fdp"], "power": r["tpr"], "size": k, "empty": float(k == 0)})
        out.append(rec)
    return pd.DataFrame(out)


def fmt(v):
    v = pd.Series(v, dtype=float).dropna()
    if not len(v):
        return ""
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return f"{v.mean():.3f} ({se:.3f})"


def table(t, by, cols):
    res = []
    for key, d in t.groupby(by + ["method"], sort=False):
        row = dict(zip(by + ["method"], key))
        row["datasets"] = len(d)
        for c in cols:
            row[c] = fmt(d[c]) if c in d else ""
        res.append(row)
    out = pd.DataFrame(res)
    out["_o"] = out["method"].map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return out.sort_values(by + ["_o"]).drop(columns="_o")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", required=True, help="one or more folders, comma-separated")
    ap.add_argument("--save")
    ap.add_argument("--all", action="store_true",
                    help="use every row, not only the data sets shared by all methods of a design")
    args = ap.parse_args(argv)
    df = load(args.results)
    df = df[df["design"].isin(SIMPLE)]
    if "error" in df:
        bad = df["error"].notna()
        if bad.any():
            print(f"{bad.sum()} rows with errors (skipped)")
        df = df[~bad]
    if not args.all:
        # compare the methods on the same data sets: within each design keep only the data sets
        # on which every method run on that design has a row (e.g. the first 100 replicates when
        # BART and DART were run on 100 and rSelectPFN on 1000)
        keep = []
        for d, dd in df.groupby("design"):
            nm = dd["method"].nunique()
            cnt = dd.groupby("id")["method"].nunique()
            keep += list(cnt.index[cnt == nm])
        df = df[df["id"].isin(set(keep))]
    t = rows(df)
    t["design"] = pd.Categorical(t["design"], SIMPLE, ordered=True)
    sel_cols = ["AUPRC", "FDR", "power", "size", "empty"]
    parts = [("1a. Selection, by design", table(t, ["design"], sel_cols)),
             ("1b. Selection, by setting", table(t, ["design", "rho", "snr"], sel_cols))]
    w = t.pivot_table(index=["design", "id"], columns="method", values="AUPRC", observed=True)
    if "rSelectPFN" in w:
        pr = []
        for m in [c for c in ORDER if c in w.columns and c != "rSelectPFN"]:
            for d, dd in w.groupby(level=0, observed=True):
                x = (dd["rSelectPFN"] - dd[m]).dropna()
                if len(x):
                    pr.append({"design": d, "vs": m, "pairs": len(x),
                               "AUPRC(rSelectPFN) - AUPRC(method)": fmt(x),
                               "share rSelectPFN better": f"{(x > 0).mean():.2f}"})
        parts.append(("2. Paired AUPRC differences", pd.DataFrame(pr)))
    text = "\n\n".join(f"### {h}\n\n{x.to_markdown(index=False)}" for h, x in parts)
    print(text)
    if args.save:
        open(args.save, "w").write(text + "\n")


if __name__ == "__main__":
    main()
