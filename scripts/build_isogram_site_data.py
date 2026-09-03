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
    fam_vocab = json.load(open("labels/family_vocab.json", encoding="utf-8"))
    families = {f: ws[: args.top_words] for f, ws in fam_vocab["families"].items()}

    # full contiguous week grid from the fit — a week never collected stays as an EMPTY week
    # (all-zero row), so the charts show a gap instead of quietly closing time up
    tbl = weekly_bands().reindex(analysis["weeks"], fill_value=0)
    versions = version_counts()
    attribution = pd.read_parquet("labels/embedding_attribution.parquet")
    fam = attribution["family_pred"].value_counts(normalize=True).round(3).to_dict()

    # split the unsigned-flagged band by the calibrated embedding family posteriors: the mean
    # posterior mass over that week's attributed sample apportions the week's flagged count.
    # Weeks with no attributed sample (the 2021-2024 history; agent house styles do not exist
    # there anyway) keep their whole mass as "unattributed".
    post = attribution.groupby("week")[["p_claude_code", "p_codex", "p_jules"]].mean()
    split_cols = {"uns_claude": [], "uns_codex": [], "uns_jules": [], "uns_unattributed": []}
    for week, total in tbl["unsigned_ai"].items():
        if week in post.index and total > 0:
            pc, px, pj = post.loc[week]
            c, x, j = round(total * pc), round(total * px), round(total * pj)
            split_cols["uns_claude"].append(c)
            split_cols["uns_codex"].append(x)
            split_cols["uns_jules"].append(j)
            split_cols["uns_unattributed"].append(int(total) - c - x - j)
        else:
            for k in ("uns_claude", "uns_codex", "uns_jules"):
                split_cols[k].append(0)
            split_cols["uns_unattributed"].append(int(total))
    tbl = tbl.assign(**split_cols).drop(columns=["unsigned_ai"])

    # grouped by model family: each family's predicted (hatched) band sits directly under its
    # signed band in the same hue
    band_order = [
        ("human", "Human (unflagged)"),
        ("uns_unattributed", "Flagged, unattributed"),
        ("uns_jules", "Gemini (Jules) — predicted"),
        ("jules", "Gemini (Jules) — signed"),
        ("uns_codex", "GPT (Codex) — predicted"),
        ("codex", "GPT (Codex) — signed"),
        ("other_agents", "Other agents — signed"),
        ("uns_claude", "Claude — predicted"),
        ("claude_code", "Claude — signed"),
    ]
    tbl = tbl[[k for k, _ in band_order]]

    recent = tbl.tail(4).sum()
    total_recent = recent.sum()
    payload = {
        "updated": str(tbl.index[-1]),
        "weeks": [str(w) for w in tbl.index],
        "bands": [{"key": k, "label": label} for k, label in band_order],
        "stack": tbl.astype(int).to_numpy().tolist(),
        "words_per_week": analysis["words_per_week"],
        "lead_share": {
            "start": lead["start_share"],
            "end": lead["end_share"],
        },
        "families": families,
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
    print(f"wrote {args.out}: {sum(len(w) for w in families.values())} family words, "
          f"{len(tbl)} weeks")


if __name__ == "__main__":
    main()
