"""Characteristic vocabulary per model family — supervised, from the signed documents.

The upstream board ranks words for an *unsupervised* cluster; here the partition is the thing we
actually know: which family SIGNED the document (Claude Code trailers → Claude, Codex footers →
GPT, Jules → Gemini). For each family the ranking is upstream's lift formula with the
contrast that matters: how often the word occurs in that family's SIGNED docs against how often
it occurs in the HUMAN baseline (unsigned docs the detector reads as human) — against
"everything else" the unsigned mass of the same family sits in the denominator and cancels the
very vocabulary being measured. Each word also carries its weekly corpus-wide appearance series
so the board can draw its history as a rate.

    PYTHONPATH=. python scripts/build_family_vocab.py    # writes labels/family_vocab.json
"""

import json
import re

import numpy as np
import pandas as pd

from scripts.assign_clusters import documents_with_identity
from vendor.analyze import MIN_AUTHORS

FAMILIES = {"claude": ["claude-code"], "gpt": ["codex"], "gemini": ["jules"]}
TOP_N = 120
OUT = "labels/family_vocab.json"

# The signature is not the style: footer URLs (collapsed to [x-url] tokens), agent and vendor
# names, and trailer machinery would top every list and say nothing about the prose.
_PROVENANCE = re.compile(
    r"""(?ix)
    \[            # any collapsed-URL token
    | claude | codex | chatgpt | openai | anthropic | jules | gemini
    | copilot | cursor | devin | co-authored | noreply | coauthored
    """
)


def main():
    X, week_of, weeks, vocab, meta = documents_with_identity()
    agents = pd.read_parquet("labels/agents.parquet")
    key = pd.DataFrame(
        {"day": [m[0] for m in meta], "line": np.asarray([m[1] for m in meta], np.int32)}
    )
    agent = key.merge(agents[["day", "line", "agent"]], on=["day", "line"], how="left")[
        "agent"
    ].to_numpy()
    iso = pd.read_parquet("labels/isogram_full.parquet")[["day", "line", "iso_ai_touched"]]
    flagged = (
        key.merge(iso, on=["day", "line"], how="left")["iso_ai_touched"]
        .fillna(False)
        .to_numpy()
    )
    human_mask = (agent == "human_unsigned") & ~flagged

    X = X.tocsr()
    corpus = np.asarray(X.sum(axis=0)).ravel()
    human = np.asarray(X[human_mask].sum(axis=0)).ravel()
    n_human = max(human.sum(), 1.0)
    n_weeks = len(weeks)
    out = {"weeks": weeks, "families": {}}
    for fam, signers in FAMILIES.items():
        mask = np.isin(agent, signers)
        inside = np.asarray(X[mask].sum(axis=0)).ravel()
        here = max(inside.sum(), 1.0)
        lift = (inside / here) / ((human + MIN_AUTHORS / 2) / n_human)
        ranked = np.lexsort((np.asarray(vocab), -lift))
        rank = np.array(
            [v for v in ranked if not _PROVENANCE.search(vocab[int(v)])][:TOP_N]
        )
        # weekly corpus-wide appearance series for each ranked word, in one sparse pass
        sub = X[:, rank].tocsc()
        series = np.zeros((TOP_N, n_weeks))
        for j in range(TOP_N):
            col = sub.getcol(j).tocoo()
            np.add.at(series[j], week_of[col.row], col.data)
        out["families"][fam] = [
            {
                "w": vocab[int(v)],
                "lift": round(float(lift[v]), 1),
                "total": int(corpus[v]),
                "series": [int(x) for x in series[j]],
            }
            for j, v in enumerate(rank)
        ]
        print(f"{fam}: {int(mask.sum()):,} signed docs; top: "
              + ", ".join(vocab[int(v)] for v in rank[:8]))

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
