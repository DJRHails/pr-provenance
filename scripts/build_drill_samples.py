"""Sample documents per (band × month) — the board's drill-down data.

For every composition band and every month it has mass in, up to three real documents: day,
repo, a truncated excerpt, the detector score, the named model where a trailer names one, and a
GitHub search URL that pins the PR by repo + creation timestamp. Bands with doc-level membership
use it directly; the predicted-family bands use the embedding-attribution sample (family
posterior > 0.8), which is exactly the population those bands are apportioned from.

    PYTHONPATH=. python scripts/build_drill_samples.py \
        --out ../isogram/web/public/github-samples.json
"""

import argparse
import json
import urllib.parse
from collections import defaultdict

import pandas as pd

PER_CELL = 3
EXCERPT = 260


def excerpt_of(body: str) -> str:
    text = " ".join((body or "").split())
    return text[:EXCERPT] + ("…" if len(text) > EXCERPT else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assign = pd.read_parquet("labels/assignments.parquet")
    agents = pd.read_parquet("labels/agents.parquet")
    full = pd.read_parquet("labels/isogram_full.parquet")[["day", "line", "iso_ai_touched", "iso_score"]]
    att = pd.read_parquet("labels/embedding_attribution.parquet")[
        ["day", "line", "family_pred", "family_conf"]
    ]
    m = (
        assign.merge(agents, on=["day", "line"], how="left")
        .merge(full, on=["day", "line"], how="left")
        .merge(att, on=["day", "line"], how="left")
    )

    def band(r):
        if r.agent == "claude-code":
            return "claude_code"
        if r.agent == "codex":
            return "codex"
        if r.agent == "jules":
            return "jules"
        if r.agent != "human_unsigned":
            return "other_agents"
        if not r.iso_ai_touched:
            return "human"
        if r.family_conf is not None and r.family_conf > 0.8:
            return {
                "claude-code": "uns_claude",
                "codex": "uns_codex",
                "jules": "uns_jules",
            }.get(r.family_pred, "uns_unattributed")
        return "uns_unattributed"

    m["band"] = [band(r) for r in m.itertuples()]
    m["month"] = m["week"].str[:7]

    picks = pd.concat(
        [
            g.sample(min(PER_CELL, len(g)), random_state=args.seed)
            for _, g in m.groupby(["band", "month"], sort=True)
        ]
    ).reset_index(drop=True)

    wanted = defaultdict(dict)
    for i, (day, line) in enumerate(zip(picks["day"], picks["line"])):
        wanted[day][int(line)] = i
    bodies = [""] * len(picks)
    stamps = [""] * len(picks)
    for day, lines in wanted.items():
        with open(f"data/days/{day}.jsonl", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh):
                if lineno in lines:
                    row = json.loads(raw)
                    bodies[lines[lineno]] = row["body"]
                    stamps[lines[lineno]] = row.get("ts") or ""

    out: dict[str, dict[str, list]] = defaultdict(dict)
    for i, r in enumerate(picks.itertuples()):
        q = urllib.parse.quote(f"repo:{r.repo} is:pr created:{stamps[i]}")
        out[r.band].setdefault(r.month, []).append(
            {
                "day": r.day,
                "repo": r.repo,
                "excerpt": excerpt_of(bodies[i]),
                "score": round(float(r.iso_score), 3) if pd.notna(r.iso_score) else None,
                "model": r.model if isinstance(r.model, str) and r.model else None,
                "url": f"https://github.com/search?type=pullrequests&q={q}",
            }
        )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {args.out}: {len(picks):,} samples across {picks['band'].nunique()} bands")


if __name__ == "__main__":
    main()
