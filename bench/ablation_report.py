"""Report of the prior ablation on design C (studies "ablation" and "c_simple").

Two networks with the same architecture and training length differ only in the meta-prior:
    rSelectPFN               the full meta-prior (preset "small")
    no interaction-only      preset "ablation": no interaction-only family, no weak-effect regime,
                             every interacting variable of the pairwise family keeps its main
                             effect (Luo, 2026, Section 5.4.4)
Design C:  y = x1 x2 + sin(x3 x4) + 1{x5 > 0} x6 + noise, Gaussian AR(1) predictors, p = 100,
n in {100, 300, 800}, rho in {0.3, 0.7}, SNR in {1, 5}. DART is run at n = 100 only.

Tables (95% intervals in brackets: percentile bootstrap over data sets, 2000 resamples;
differences between the networks are paired, i.e. bootstrapped on per-data-set differences)
    1  AUPRC and MPM (FDR, power, size, Hamming = false positives + false negatives), by n and
       method
    2  share of data sets in which the MPM includes each of the six active variables, by n
    3  paired differences full - ablated: AUPRC and inclusion of x3, x4, by n and by setting
    4  (if the simple study was rerun with the ablated network) paired AUPRC differences on the
       four simple designs
Figure: inclusion of each group of active variables against n, both networks.

    python -m bench.ablation_report --results R/ablation_pfn,R/ablation_dart[,R/simple_ablated,
        R/simple_pfn] --save R/ablation_report.md --fig R/figures
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from .designs import make_dataset
from .report import load

FULL, ABL, DART = "rSelectPFN", "rSelectPFN, no interaction-only", "DART"
NAMES = {"selectpfn:full": FULL, "selectpfn:small": FULL, "selectpfn:ablated": ABL,
         "selectpfn:ablation": ABL, "dart": DART}
ORDER = [FULL, ABL, DART]
VARS = ["x1", "x2", "x3", "x4", "x5", "x6"]
GROUPS = {"x1, x2 (x1 x2)": ["x1", "x2"], "x3, x4 (sin x3 x4)": ["x3", "x4"],
          "x5 (threshold)": ["x5"], "x6": ["x6"]}
SIMPLE = ["LIN_EQ", "LIN_TOE", "FRI_EQ", "FRI_TOE"]
B = 2000


def ci(x, seed=0):
    x = np.asarray(pd.Series(x, dtype=float).dropna(), float)
    if not x.size:
        return ""
    if x.size == 1 or np.all(x == x[0]):
        return f"{x.mean():.3f} [{x.mean():.3f}, {x.mean():.3f}]"
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, size=(B, x.size))].mean(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return f"{x.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def ci_num(x, seed=0):
    x = np.asarray(pd.Series(x, dtype=float).dropna(), float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, x.size, size=(B, x.size))].mean(1)
    return x.mean(), *np.quantile(m, [0.025, 0.975])


def per_dataset(df):
    cache, out = {}, []
    for _, r in df.iterrows():
        if r["id"] not in cache:
            ds = make_dataset(r["design"], int(r["n"]), int(r["p"]), float(r["rho"]), r["snr"],
                              int(r["rep"]))
            cache[r["id"]] = (ds.gamma.astype(bool), np.asarray(ds.S0))
        g, S0 = cache[r["id"]]
        q = np.asarray(r["score"], float)
        mpm = q > 0.5
        k, tp = int(mpm.sum()), int((mpm & g).sum())
        rec = {"id": r["id"], "design": r["design"], "n": int(r["n"]), "rho": r["rho"],
               "snr": r["snr"], "method": NAMES.get(r["method"], r["method"]),
               "AUPRC": r["auprc"], "FDR": (k - tp) / max(1, k), "power": tp / g.sum(),
               "size": k, "Hamming": (k - tp) + (g.sum() - tp)}
        if r["design"] == "C":
            for v, j in zip(VARS, S0):         # f_C reads the active columns in the order of S0
                rec[f"in {v}"] = float(mpm[j])
                rec[f"q {v}"] = float(q[j])
            rec["in x3,x4"] = (rec["in x3"] + rec["in x4"]) / 2
        out.append(rec)
    return pd.DataFrame(out)


def table(t, by, cols):
    res = []
    for key, d in t.groupby(by + ["method"], sort=False):
        row = dict(zip(by + ["method"], key if isinstance(key, tuple) else (key,)))
        row["datasets"] = len(d)
        row.update({c: ci(d[c]) for c in cols if c in d})
        res.append(row)
    out = pd.DataFrame(res)
    out["_o"] = out["method"].map({m: i for i, m in enumerate(ORDER)}).fillna(99)
    return out.sort_values(by + ["_o"]).drop(columns="_o")


def paired(t, by, cols, a=FULL, b=ABL):
    res = []
    for key, d in t.groupby(by, sort=True):
        row = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        for c in cols:
            w = d.pivot_table(index="id", columns="method", values=c)
            if a in w and b in w:
                x = (w[a] - w[b]).dropna()
                row["pairs"] = len(x)
                row[f"{c}: full - ablated"] = ci(x)
        res.append(row)
    return pd.DataFrame(res)


def figure(t, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    INK, MUTED, GRID = "#222222", "#6b6a66", "#e6e6e3"
    COL = {FULL: "#1c5cab", ABL: "#6b6a66", DART: "#d0661f"}
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "font.size": 8.5, "axes.titlesize": 8.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02})
    titles = {"x1, x2 (x1 x2)": r"$x_1, x_2$: $x_1 x_2$",
              "x3, x4 (sin x3 x4)": r"$x_3, x_4$: $\sin(x_3 x_4)$",
              "x5 (threshold)": r"$x_5$: $\mathbb{1}\{x_5 > 0\}\, x_6$",
              "x6": r"$x_6$: $\mathbb{1}\{x_5 > 0\}\, x_6$"}
    c = t[t["design"] == "C"]
    ns = sorted(c["n"].unique())
    fig, axs = plt.subplots(1, 4, figsize=(5.4, 1.95), sharey=True)
    for ax, (gname, vs) in zip(axs, GROUPS.items()):
        for m in ORDER:
            d = c[c["method"] == m]
            if d.empty:
                continue
            pts = []
            for n in ns:
                dn = d[d["n"] == n]
                if len(dn):
                    pts.append((n, *ci_num(dn[[f"in {v}" for v in vs]].mean(1))))
            if not pts:
                continue
            x, mu, lo, hi = map(np.array, zip(*pts))
            ax.plot(x, mu, color=COL[m], lw=1.0, marker="o", ms=3.2, mec="white", mew=0.5)
            ax.fill_between(x, lo, hi, color=COL[m], alpha=0.15, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.minorticks_off()
        ax.set_xlim(ns[0] * 0.85, ns[-1] * 1.15)
        ax.set_ylim(0, 1.02)
        ax.set_title(titles[gname], loc="left", pad=3, fontsize=7.5)
        ax.set_xlabel("$n$")
    axs[0].set_ylabel("in the MPM")
    h = [Line2D([], [], color=COL[m], lw=1.0, marker="o", ms=3.2, label=m) for m in ORDER
         if m in set(c["method"])]
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.88))
    fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, 0.87),
               handletextpad=0.3, columnspacing=1.4)
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "ablation_detection.pdf"))
    fig.savefig(os.path.join(out, "ablation_detection.png"), dpi=220)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--results", required=True, help="one or more folders, comma-separated")
    ap.add_argument("--save")
    ap.add_argument("--fig", help="folder for the figure")
    args = ap.parse_args(argv)
    df = load(args.results)
    if "error" in df:
        bad = df["error"].notna()
        if bad.any():
            print(f"{bad.sum()} rows with errors (skipped)")
        df = df[~bad]
    df = df[df["design"].isin(["C"] + SIMPLE)]
    df = df[df["method"].map(NAMES).notna()].drop_duplicates(["id", "method"])
    # simple designs: only the data sets on which both networks were run (the full network has
    # 1000 replicates there, the ablated one 100)
    simple = df["design"].isin(SIMPLE)
    both = df[simple].groupby("id")["method"].nunique()
    df = df[~simple | df["id"].isin(set(both.index[both > 1]))]
    t = per_dataset(df)
    c = t[t["design"] == "C"]
    main_cols = ["AUPRC", "FDR", "power", "size", "Hamming"]
    parts = [("1. Design C, by n", table(c, ["n"], main_cols)),
             ("1b. Design C, by setting", table(c, ["n", "rho", "snr"], main_cols)),
             ("2. Design C: share of data sets whose MPM includes each active variable, by n",
              table(c, ["n"], [f"in {v}" for v in VARS])),
             ("2b. The same, by setting",
              table(c, ["n", "rho", "snr"], [f"in {v}" for v in VARS])),
             ("3. Design C: paired differences full - ablated, by n",
              paired(c, ["n"], ["AUPRC", "in x3,x4", "Hamming"])),
             ("3b. The same, by setting",
              paired(c, ["n", "rho", "snr"], ["AUPRC", "in x3,x4", "Hamming"]))]
    s = t[t["design"].isin(SIMPLE)]
    if not s.empty and s["method"].nunique() > 1:
        parts.append(("4. Simple designs: paired AUPRC differences full - ablated",
                      paired(s, ["design"], ["AUPRC"])))
    text = "\n\n".join(f"### {h}\n\n{x.to_markdown(index=False)}" for h, x in parts)
    print(text)
    if args.save:
        open(args.save, "w").write(text + "\n")
    if args.fig:
        figure(t, args.fig)


if __name__ == "__main__":
    main()
