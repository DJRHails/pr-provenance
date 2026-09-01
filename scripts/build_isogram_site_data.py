"""Build the data bundle for isogram.hails.info/github (the React board in isogram's web/).

One JSON, baked into the page at build time (the page is statically rendered): the weekly
provenance stack, the lead cluster's vocabulary with per-word weekly rate series (the
word-lines), the named-Claude-version timeline, stat tiles, and the embedding-attribution
shares. Word series come from the committed `analysis/analysis.js` (the exact published fit).

    PYTHONPATH=. python scripts/build_isogram_site_data.py \
        --out ../isogram/web/src/github/data.json --top-words 160
"""

import argparse
import json

import pandas as pd

from scripts.build_web_data import BANDS, version_counts, weekly_bands


def load_analysis(path="analysis/analysis.js"):
    text = open(path, encoding="utf-8").read()
    return json.loads(text[text.index("{") : text.rstrip().rstrip(";").rindex("}") + 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-words", type=int, default=160)
    args = ap.parse_args()

    analysis = load_analysis()
    lead = next(c for c in analysis["components"] if c["lead"])
    n = args.top_words
    words = [
        {
            "w": w,
            "lift": lift,
            "total": int(sum(series)),
            "series": [int(v) for v in series],
        }
        for w, lift, series in zip(
            lead["word_list"][:n], lead["word_lift"][:n], lead["series"][:n]
        )
    ]

    tbl = weekly_bands()
    versions = version_counts()
    attribution = pd.read_parquet("labels/embedding_attribution.parquet")
    fam = attribution["family_pred"].value_counts(normalize=True).round(3).to_dict()

    recent = tbl.tail(4).sum()
    total_recent = recent.sum()
    payload = {
        "updated": str(tbl.index[-1]),
        "weeks": [str(w) for w in tbl.index],
        "bands": [{"key": k, "label": label} for k, label, _ in BANDS],
        "stack": tbl.astype(int).to_numpy().tolist(),
        "words_per_week": analysis["words_per_week"],
        "lead_share": {
            "start": lead["start_share"],
            "end": lead["end_share"],
        },
        "words": words,
        "versions": [
            {"model": m, "week": str(w), "n": int(c)}
            for m, w, c in versions[["model", "week", "n"]].itertuples(index=False)
        ],
        "tiles": {
            "ai_recent": round(float((total_recent - recent["human"]) / total_recent), 3),
            "signed_recent": round(
                float(
                    (
                        recent["claude_code"]
                        + recent["codex"]
                        + recent["jules"]
                        + recent["other_agents"]
                    )
                    / total_recent
                ),
                3,
            ),
            "documents": int(tbl.to_numpy().sum()),
        },
        "family_attribution": fam,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"wrote {args.out}: {len(words)} words, {len(tbl)} weeks")


if __name__ == "__main__":
    main()
