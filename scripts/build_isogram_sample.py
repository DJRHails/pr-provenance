"""Stratified sample of kept descriptions for isogram (AI-text detector) scoring.

Uniform per week — the same number of documents drawn from every whole week — so weekly
AI-share estimates carry equal precision across the corpus, seeded so the sample is
reproducible. Joins each sampled `(day, line)` back to its body text in `data/days/`.

    PYTHONPATH=. python scripts/build_isogram_sample.py --per-week 250
    # writes labels/isogram_sample.parquet (text included — the scoring input)
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd

ASSIGNMENTS = "labels/assignments.parquet"
OUT = "labels/isogram_sample.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-week", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(ASSIGNMENTS)
    parts = [
        g.sample(min(args.per_week, len(g)), random_state=args.seed)
        for _, g in df.groupby("week", sort=True)
    ]
    sample = pd.concat(parts).reset_index(drop=True)

    wanted = defaultdict(set)
    for day, line in zip(sample["day"], sample["line"]):
        wanted[day].add(int(line))
    texts = {}
    for day, lines in wanted.items():
        with open(os.path.join("data/days", f"{day}.jsonl"), encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh):
                if lineno in lines:
                    texts[(day, lineno)] = json.loads(raw)["body"]
    sample["text"] = [texts[(d, int(li))] for d, li in zip(sample["day"], sample["line"])]
    sample.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(sample):,} rows across {sample['week'].nunique()} weeks")


if __name__ == "__main__":
    main()
