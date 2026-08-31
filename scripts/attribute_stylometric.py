"""Stylometric agent attribution: which agent wrote the descriptions that are NOT signed?

The explicit signatures (`attribute_agents.py`) are ground truth for ~21% of rows. This trains a
bag-of-words classifier on those signed rows — with every signature line and every URL stripped,
so the model has to read the prose, not the footer — validates it on held-out signed rows split
by AUTHOR (an account's docs never straddle train/val, so the classifier cannot lean on one
account's idiosyncrasies), and then attributes the unsigned rows.

Honest scope: this attributes a *house style*, not a checkpoint. Agents share underlying models
(Cursor can run Claude; Copilot runs several), and prompt scaffolding shapes the prose as much as
the model does. The per-class validation numbers say how far the styles are separable at all; the
unsigned attributions are hypotheses weighted by those numbers, not labels.

    PYTHONPATH=. python scripts/attribute_stylometric.py
    # writes labels/stylometric.parquet (unsigned rows: predicted agent + confidence)
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

AGENTS = ["claude-code", "codex", "copilot", "cursor", "devin", "jules"]
MIN_WORDS = 20
OUT = "labels/stylometric.parquet"

_STRIP = re.compile(
    r"""(?imx)
    ^.*co-authored-by:.*$                       # any co-author trailer
    | ^.*generated\ with.*$                     # Claude Code / generic footers
    | ^.*\[codex\ task\].*$                     # Codex task link line
    | ^.*link\ to\ devin\ session.*$            # Devin session line
    | ^.*cursor_agent_pr_body.*$                # Cursor body markers
    | ^.*(jules\.google|all-hands\.dev|app\.warp\.dev|ampcode\.com).*$
    | ^.*(?:🤖|🧑‍💻).*$                          # robot-emoji footer lines
    | https?://\S+                              # every URL — links are provenance, not prose
    | <!--.*?-->                                # HTML comments (agent body markers hide in them)
    """,
)


def clean(body: str) -> str:
    return _STRIP.sub(" ", body)


def load_texts(df: pd.DataFrame) -> list[str]:
    wanted = {}
    for i, (day, line) in enumerate(zip(df["day"], df["line"])):
        wanted.setdefault(day, {})[int(line)] = i
    out = [""] * len(df)
    for day, lines in wanted.items():
        with open(os.path.join("data/days", f"{day}.jsonl"), encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh):
                if lineno in lines:
                    out[lines[lineno]] = clean(json.loads(raw)["body"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-per-agent", type=int, default=20000)
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    agents = pd.read_parquet("labels/agents.parquet")
    kept = pd.read_parquet("labels/assignments.parquet")[
        ["day", "line", "week", "component", "author"]
    ]
    df = kept.merge(agents, on=["day", "line"], how="left")

    counts = df["agent"].value_counts()
    classes = [a for a in AGENTS if counts.get(a, 0) >= 1000]
    print(f"classes with >=1000 kept signed rows: {classes}")
    signed = df[df["agent"].isin(classes)].copy()
    signed = pd.concat(
        [
            g.sample(min(args.max_per_agent, len(g)), random_state=args.seed)
            for _, g in signed.groupby("agent", sort=True)
        ]
    ).reset_index(drop=True)
    print("training pool:", signed["agent"].value_counts().to_dict())

    texts = load_texts(signed)
    words = pd.Series(texts).str.split().str.len()
    ok = words >= MIN_WORDS
    signed, texts = signed[ok.values].reset_index(drop=True), [t for t, k in zip(texts, ok) if k]

    # split by author: no account contributes to both sides
    author_hash = pd.util.hash_array(signed["author"].to_numpy(dtype=object)) % 100
    val = author_hash < 20
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=200_000, sublinear_tf=True)
    Xtr = vec.fit_transform(pd.Series(texts)[~val])
    Xva = vec.transform(pd.Series(texts)[val])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, signed["agent"][~val])
    print(classification_report(signed["agent"][val], clf.predict(Xva), digits=3))

    unsigned = df[df["agent"] == "human_unsigned"].reset_index(drop=True)
    utexts = load_texts(unsigned)
    uwords = pd.Series(utexts).str.split().str.len()
    uok = (uwords >= MIN_WORDS).to_numpy()
    proba = clf.predict_proba(vec.transform(pd.Series(utexts)[uok]))
    pred = clf.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)

    out = unsigned.loc[uok, ["day", "line", "week", "component"]].reset_index(drop=True)
    out["predicted_agent"] = pred
    out["confidence"] = conf.astype(np.float32)
    os.makedirs("labels", exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(out):,} unsigned rows attributed")
    print(out["predicted_agent"].value_counts().to_string())
    print("high-confidence (>0.8):")
    print(out[out.confidence > 0.8]["predicted_agent"].value_counts().to_string())


if __name__ == "__main__":
    main()
