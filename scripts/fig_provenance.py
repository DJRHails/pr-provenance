"""One figure: detector-flagged share vs signed share vs the lead cluster, weekly.

    PYTHONPATH=. python scripts/fig_provenance.py   # writes figures/fig_provenance.png
"""

import os

import numpy as np
import pandas as pd
from graphs import ci_fill, finalize, footnotes, label_lines, save_chart, set_theme, subplots


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def main():
    set_theme()
    assign = pd.read_parquet("labels/assignments.parquet")
    agents = pd.read_parquet("labels/agents.parquet")
    scores = pd.read_parquet("labels/isogram_scores.parquet")

    kept = assign.merge(agents[["day", "line", "agent"]], on=["day", "line"], how="left")
    weekly = kept.groupby("week").agg(
        signed=("agent", lambda a: (a != "human_unsigned").mean()),
        lead=("lead", "mean"),
    )
    det = scores.groupby("week")["iso_ai_touched"].agg(["mean", "size", "sum"])
    weeks = pd.to_datetime(weekly.index)

    fig, ax = subplots("wide", height=4.4)
    lo, hi = wilson(det["sum"].to_numpy(), det["size"].to_numpy())
    line_det = ax.plot(weeks, det["mean"] * 100, label="Detector-flagged*")[0]
    ci_fill(ax, weeks, lo * 100, hi * 100, color=line_det.get_color())
    ax.plot(weeks, weekly["signed"] * 100, label="Agent-signed†")
    ax.plot(weeks, weekly["lead"] * 100, label="'Claude' cluster‡")
    label_lines(ax)
    ax.set_ylim(0, 100)

    finalize(
        ax,
        title="Agents sign a third of pull requests; a detector flags nine in ten",
        descriptor="Share of sampled GitHub pull-request descriptions, weekly, %",
    )
    footnotes(
        fig,
        "*isogram (EditLens-replication Qwen3.5-9B detector) verdict on 250 descriptions/week",
        "†explicit agent signature: Claude Code / Codex / Copilot / Cursor / Devin / Jules / …",
        "‡the arriving KL-k-means component of louisabraham.github.io/load-bearing",
        source="Source: DJRHails/pr-provenance (467,387 descriptions, 2025-01-05 to 2026-08-23); "
        "fig_provenance.py",
    )
    os.makedirs("figures", exist_ok=True)
    save_chart("figures/fig_provenance.py")


if __name__ == "__main__":
    main()
