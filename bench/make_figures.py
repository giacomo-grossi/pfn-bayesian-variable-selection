"""Figure 6.1 of the thesis (AUPRC by setting) from the report of bench.simple_report.

Usage: python make_figures.py simple_report_completo.md output_folder
Writes the figure as PDF (for LaTeX) and PNG (for a quick look).

Style: sober, for print. rSelectPFN in dark blue, the spike-and-slab model in neutral grey and
DART in muted orange; thin marks, a light horizontal grid, one legend, Computer Modern text at
8-9 pt, width equal to the text width of the thesis (5.4 in) so that LaTeX does not rescale the
fonts.
"""

import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
txt = open(src).read()

TEXTWIDTH = 5.4
COL = {"rSelectPFN": "#1c5cab", "Spike-and-slab": "#6b6a66", "DART": "#d0661f"}
INK, MUTED, GRID = "#222222", "#6b6a66", "#e6e6e3"
NAME = {"LIN_EQ": "Linear, equicorrelated", "LIN_TOE": "Linear, Toeplitz",
        "FRI_EQ": "Friedman, equicorrelated", "FRI_TOE": "Friedman, Toeplitz"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 8.5, "axes.titlesize": 8.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def table(title):
    blk = txt.split("### " + title)[1].split("###")[0]
    rows = [r for r in blk.strip().splitlines() if r.startswith("|")]
    head = [h.strip() for h in rows[0].strip("|").split("|")]
    return [dict(zip(head, [c.strip() for c in r.strip("|").split("|")])) for r in rows[2:]]


def mse(s):
    m = re.match(r"([-\d.]+) \(([\d.na]+)\)", s)
    return float(m.group(1)), (float(m.group(2)) if m.group(2) != "nan" else 0.0)


def save(fig, name):
    fig.savefig(f"{out}/{name}.pdf")
    fig.savefig(f"{out}/{name}.png", dpi=220)
    plt.close(fig)


def method_handles(methods):
    return [Line2D([], [], color=COL[m], marker="o", ms=4, lw=0, label=m) for m in methods]


# ---------------------------------------------------------------------------------------------
# Figure 6.1: AUPRC by setting, points with 95% intervals
# ---------------------------------------------------------------------------------------------
sel = table("1b. Selection, by setting")
sets = [("0.3", "1"), ("0.3", "5"), ("0.7", "1"), ("0.7", "5")]
fig, axs = plt.subplots(2, 2, figsize=(TEXTWIDTH, 3.6), sharey=True)
for ax, d in zip(axs.flat, ["LIN_EQ", "LIN_TOE", "FRI_EQ", "FRI_TOE"]):
    rows = [r for r in sel if r["design"] == d]
    meths = [m for m in COL if any(r["method"] == m for r in rows)]
    for k, m in enumerate(meths):
        v = [mse(next(r for r in rows if r["method"] == m and (r["rho"], r["snr"]) == s)["AUPRC"])
             for s in sets]
        x = np.arange(4) + (k - 0.5) * 0.22
        ax.errorbar(x, [a for a, _ in v], yerr=[1.96 * e for _, e in v], fmt="o", ms=3.6,
                    color=COL[m], elinewidth=0.9, capsize=0, mec="white", mew=0.5)
    ax.axhline(0.05, color=MUTED, lw=0.6, ls=(0, (1, 2)))
    ax.set_xticks(range(4), [f"$\\rho={a}$\nSNR {b}" for a, b in sets])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_title(NAME[d], loc="left", pad=3)
for ax in axs[:, 0]:
    ax.set_ylabel("AUPRC")
fig.tight_layout(h_pad=1.2, w_pad=1.0, rect=(0, 0, 1, 0.94))
fig.legend(handles=method_handles(COL), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.935),
           handletextpad=0.3, columnspacing=1.6)
save(fig, "auprc_by_setting")
print("ok")
