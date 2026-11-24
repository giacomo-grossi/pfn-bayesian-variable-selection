"""Loading of the JSONL results written by bench.run, shared by bench.simple_report and
bench.ablation_report."""

from __future__ import annotations

import glob
import json
import os

import pandas as pd


def load(results):
    """results: one folder or several, comma-separated; for a (data set, method) present in more
    than one folder the row of the first folder is kept."""
    rows = []
    for k, folder in enumerate(results.split(",")):
        for f in sorted(glob.glob(os.path.join(folder, "*.jsonl"))):
            with open(f) as h:
                rows += [{**json.loads(line), "_folder": k} for line in h if line.strip()]
    df = pd.DataFrame(rows)
    if len(df) and "error" in df:
        # a failed row that was later retried successfully is dropped
        keys = ["id", "method"]
        df = (df.assign(_err=df["error"].notna()).sort_values(["_err", "_folder"], kind="stable")
                .drop_duplicates(keys, keep="first").drop(columns="_err"))
        bad = df[df["error"].notna()]
        for m, g in bad.groupby("method"):
            print(f"[report] {m}: {len(g)} failed rows, e.g. {g['error'].iloc[0][:200]}")
    elif len(df):
        keys = ["id", "method"]
        df = df.sort_values("_folder", kind="stable").drop_duplicates(keys, keep="first")
    return df.drop(columns="_folder", errors="ignore")
