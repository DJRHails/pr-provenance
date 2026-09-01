"""Early answers from a part-fetched history: per-week samples for detector scoring.

The coarse-to-fine backfill lands whole weeks of 2021-2024 at increasing resolution; this builds
a scoring sample from whatever is on disk RIGHT NOW, without waiting for the full corpus or the
k-means refit. It applies the upstream cleaning per week (bot-author filter, >= MIN_WORDS
distinct words, identical-word-set dedup, <= MAX_PER_AUTHOR docs per author), attaches agent
signatures, samples N per week, and writes the scoring input. Numbers from this path are
provisional (the vocabulary-floor recheck of the full pipeline is skipped — a ~0.1% effect);
the full refit replaces them when the sweep completes.

    PYTHONPATH=. python scripts/quick_history_scan.py --per-week 400 --before 2025-01-01
    # writes .data/history_sample.parquet — score it with isogram scripts/score_pr_corpus.py
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import date, timedelta

import pandas as pd

from scripts.attribute_agents import attribute
from vendor.analyze import BOT_LOGIN, BOT_SUFFIX, MAX_PER_AUTHOR, MIN_WORDS, tokens
from vendor.fetch_day import path as day_path

ANCHOR = date(2024, 12, 30)


def monday_of(day: date) -> date:
    return day - timedelta(days=(day - ANCHOR).days % 7)


def whole_weeks(before: date) -> list[date]:
    days = sorted(
        date.fromisoformat(os.path.basename(f).removesuffix(".jsonl"))
        for f in glob.glob("data/days/*.jsonl")
    )
    by_week = defaultdict(set)
    for d in days:
        if d < before:
            by_week[monday_of(d)].add(d)
    return sorted(w for w, got in by_week.items() if len(got) == 7)


def kept_rows(week: date) -> list[dict]:
    """The upstream per-week filters, applied to one week's seven files."""
    seen, by_author, out = set(), Counter(), []
    for i in range(7):
        day = week + timedelta(days=i)
        with open(day_path(day), encoding="utf-8") as fh:
            for lineno, line in enumerate(fh):
                row = json.loads(line)
                author = (row.get("author") or "").lower()
                if author.endswith(BOT_SUFFIX) or author in BOT_LOGIN:
                    continue
                words = tokens(row["body"])
                key = frozenset(words)
                if len(key) < MIN_WORDS or key in seen:
                    continue
                if by_author[author] >= MAX_PER_AUTHOR:
                    continue
                seen.add(key)
                by_author[author] += 1
                agent, _, model = attribute(row.get("author") or "", row["body"])
                out.append(
                    {
                        "day": str(day),
                        "line": lineno,
                        "week": str(week),
                        "agent": agent,
                        "model": model,
                        "text": row["body"],
                    }
                )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-week", type=int, default=400)
    ap.add_argument("--before", default="2025-01-01", help="only weeks starting before this")
    ap.add_argument("--out", default=".data/history_sample.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    parts = []
    for week in whole_weeks(date.fromisoformat(args.before)):
        rows = pd.DataFrame(kept_rows(week))
        parts.append(rows.sample(min(args.per_week, len(rows)), random_state=args.seed))
    if not parts:
        raise SystemExit("no whole pre-2025 weeks on disk yet")
    sample = pd.concat(parts).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sample.to_parquet(args.out, index=False)
    signed = (sample["agent"] != "human_unsigned").mean()
    print(
        f"wrote {args.out}: {len(sample):,} rows over {sample['week'].nunique()} weeks; "
        f"signed share {signed:.2%}"
    )
    print(sample.groupby(sample["week"].str[:4]).size().to_string())


if __name__ == "__main__":
    main()
