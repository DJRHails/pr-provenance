"""Build the input for isogram embedding extraction (family/model identification).

Population, all with signature lines + URLs stripped (`attribute_stylometric.clean`) so the
embedding classifier reads prose, never footers:

- every signed kept row (the calibration labels),
- every raw row whose signature names a model version (the model-specific labels),
- a per-week sample of unsigned kept rows the detector flagged (the population to attribute),
- a per-week sample of unsigned unflagged rows (the human reference frame).

    PYTHONPATH=. python scripts/build_embed_input.py   # writes labels/embed_input.parquet
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd

from scripts.attribute_stylometric import clean

OUT = "labels/embed_input.parquet"


def attach_texts(df: pd.DataFrame) -> pd.DataFrame:
    wanted = defaultdict(dict)
    for i, (day, line) in enumerate(zip(df["day"], df["line"])):
        wanted[day][int(line)] = i
    texts = [""] * len(df)
    for day, lines in wanted.items():
        with open(os.path.join("data/days", f"{day}.jsonl"), encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh):
                if lineno in lines:
                    texts[lines[lineno]] = clean(json.loads(raw)["body"])
    return df.assign(text=texts)


def per_week_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return pd.concat(
        [g.sample(min(n, len(g)), random_state=seed) for _, g in df.groupby("week", sort=True)]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flagged-per-week", type=int, default=500)
    ap.add_argument("--human-per-week", type=int, default=150)
    args = ap.parse_args()

    agents = pd.read_parquet("labels/agents.parquet")
    kept = pd.read_parquet("labels/assignments.parquet")[
        ["day", "line", "week", "component", "lead", "author"]
    ]
    full = pd.read_parquet("labels/isogram_full.parquet")[["day", "line", "iso_ai_touched"]]
    m = kept.merge(agents, on=["day", "line"], how="left").merge(
        full, on=["day", "line"], how="left"
    )

    signed = m[m["agent"] != "human_unsigned"].assign(group="signed")
    uns = m[m["agent"] == "human_unsigned"]
    flagged = per_week_sample(uns[uns["iso_ai_touched"]], args.flagged_per_week, args.seed).assign(
        group="unsigned_flagged"
    )
    human = per_week_sample(uns[~uns["iso_ai_touched"]], args.human_per_week, args.seed).assign(
        group="unsigned_clean"
    )

    # versioned-model rows outside the kept corpus (raw-only), for the model-specific probe
    versioned = agents[
        agents["model"].str.startswith("claude-", na=False)
        & (agents["model"] != "claude-unversioned")
    ]
    extra = versioned.merge(kept[["day", "line"]], on=["day", "line"], how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"].drop(columns=["_merge"]).assign(
        group="versioned_raw", week="", component=-1, lead=False, author="", iso_ai_touched=None
    )

    cols = ["day", "line", "week", "component", "lead", "agent", "model", "group"]
    out = pd.concat([signed, flagged, human, extra])[cols].reset_index(drop=True)
    out = attach_texts(out)
    out = out[out["text"].str.split().str.len() >= 5].reset_index(drop=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} rows")
    print(out["group"].value_counts().to_string())


if __name__ == "__main__":
    main()
