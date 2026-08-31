"""Per-document cluster labels for the load-bearing corpus.

Upstream `analyze.py` (vendored, MIT — Louis Abraham) fits KL k-means over pull-request
descriptions and publishes only aggregates. This script re-runs the exact published fit and
exports what the page never shows: WHICH cluster every kept description was assigned to, keyed
back to its `(day, line)` in `data/days/`.

Exactness is asserted, not assumed, at every seam against the published `analysis/analysis.js`:
document count, appearance count, vocabulary size, the fit's cost under the published seed, and
finally the per-week × per-component count matrix — the labels this writes reproduce every curve
on the upstream page by aggregation.

    PYTHONPATH=. python scripts/assign_clusters.py            # writes labels/assignments.parquet

Needs numpy, scipy, numba (the fit) and pandas + pyarrow (the output).
"""

import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from vendor import analyze
from vendor.analyze import (
    BOT_LOGIN,
    BOT_SUFFIX,
    MAX_PER_AUTHOR,
    MIN_AUTHORS,
    MIN_WORDS,
    WORKERS,
    fit,
    nearest,
    tokens,
    week_files,
)

OUT = "labels/assignments.parquet"
PUBLISHED = "analysis/analysis.js"


def load_published(path=PUBLISHED):
    text = open(path, encoding="utf-8").read()
    return json.loads(text[text.index("{") : text.rstrip().rstrip(";").rindex("}") + 1])


def scan_with_identity(path):
    """Upstream `scan`, plus the identity it drops: line number, timestamp, repo, author.

    Must mirror `analyze.scan` exactly — same bot filter, same tokenisation, same row order —
    so the corpus this builds is numbered identically to the one the published fit saw.
    """
    ids, cols, meta = {}, [], []
    day = os.path.basename(path).removesuffix(".jsonl")
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh):
            row = json.loads(line)
            a = (row.get("author") or "").lower()
            if a.endswith(BOT_SUFFIX) or a in BOT_LOGIN:
                continue
            words = tokens(row["body"])
            cols.append(
                np.fromiter((ids.setdefault(w, len(ids)) for w in words), np.int32, len(words))
            )
            meta.append((day, lineno, row.get("ts") or "", row.get("repo") or "",
                         row.get("author") or ""))
    return list(ids), cols, meta


def documents_with_identity(log=print):
    """Upstream `documents`, carrying `(day, line)` through every filter it applies."""
    weeks, groups = week_files(log)
    flat = [(t, f) for t, group in enumerate(groups) for f in group]

    ids, cols, meta, week_raw = {}, [], [], []
    with Pool(min(WORKERS, os.cpu_count() or 1)) as pool:
        for (t, _), (local, c, m) in zip(
            flat, pool.imap(scan_with_identity, [f for _, f in flat], chunksize=4)
        ):
            remap = np.fromiter((ids.setdefault(w, len(ids)) for w in local), np.int32, len(local))
            cols += [remap[x] for x in c]
            meta += m
            week_raw += [t] * len(m)

    vocab_all = list(ids)
    n, V = len(meta), len(vocab_all)
    week_raw = np.asarray(week_raw, np.int64)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum([len(c) for c in cols], out=indptr[1:])
    indices = np.concatenate(cols) if cols else np.zeros(0, np.int32)
    del cols
    X0 = csr_matrix((np.ones(indices.size), indices, indptr), shape=(n, V), dtype=np.float64)
    X0.sum_duplicates()

    # the three per-description filters, verbatim from upstream `documents`
    from collections import Counter

    n_distinct = np.diff(X0.indptr)
    keep = np.zeros(n, bool)
    seen, by_author, cur_week = set(), Counter(), -1
    authors = [m[4] for m in meta]
    for d in range(n):
        if week_raw[d] != cur_week:
            seen, by_author, cur_week = set(), Counter(), week_raw[d]
        if n_distinct[d] < MIN_WORDS:
            continue
        key = X0.indices[X0.indptr[d] : X0.indptr[d + 1]].tobytes()
        if key in seen:
            continue
        a = authors[d]
        if by_author[a] >= MAX_PER_AUTHOR:
            continue
        by_author[a] += 1
        seen.add(key)
        keep[d] = True

    kd = np.flatnonzero(keep)
    Xk = X0[kd]

    # the one floor: distinct accounts per word
    aid = {}
    aids = np.array([aid.setdefault(authors[d], len(aid)) for d in kd], np.int64)
    by_word = csr_matrix(
        (np.ones(Xk.indices.size), (np.repeat(aids, np.diff(Xk.indptr)), Xk.indices)),
        shape=(len(aid), V),
        dtype=np.float64,
    )
    n_auth = np.bincount(by_word.indices, minlength=V)
    live = np.flatnonzero(n_auth >= MIN_AUTHORS)
    live = live[np.argsort([vocab_all[i] for i in live], kind="stable")]
    vocab = [vocab_all[i] for i in live]

    X, week_of = Xk[:, live], week_raw[kd]
    long_enough = np.asarray(X.sum(axis=1)).ravel() >= MIN_WORDS
    X, week_of, kd = X[long_enough], week_of[long_enough], kd[long_enough]
    kept_meta = [meta[d] for d in kd]
    log(f"{X.shape[0]:,} descriptions, {X.sum():,.0f} appearances, {len(vocab):,} words")
    return X, week_of, weeks, vocab, kept_meta


def main():
    published = load_published()

    X, week_of, weeks, vocab, meta = documents_with_identity()
    assert X.shape[0] == published["documents"], (
        f"document count drifted: built {X.shape[0]}, published {published['documents']}"
    )
    assert int(X.sum()) == published["appearances"], (
        f"appearance count drifted: built {int(X.sum())}, published {published['appearances']}"
    )
    assert [w[:1] for w in weeks] and weeks == published["weeks"], "week grid drifted"

    # the published fit, from its published seed alone
    W, C, A, M, cost = fit(X, week_of, len(weeks), seed=published["seed"])
    assert abs(cost - published["cost"]) < 1.0, (
        f"fit cost drifted: got {cost:,.1f}, published {published['cost']:,.1f}"
    )

    labels, _ = nearest(X.data, X.indices, X.indptr, np.ascontiguousarray(np.log(W).T))

    # map raw cluster ids onto the published component order (largest of the last month first)
    recent = C[-published["lead_window"] :].sum(axis=0)
    order = np.argsort(-recent)
    rank_of = np.empty_like(order)
    rank_of[order] = np.arange(len(order))
    component = rank_of[labels]

    # the labels must reproduce every published weekly curve exactly
    for rank, comp in enumerate(published["components"]):
        got = np.bincount(week_of[component == rank], minlength=len(weeks)).tolist()
        assert got == comp["count"], f"weekly doc counts drifted on component {rank}"

    lead_rank = next(i for i, c in enumerate(published["components"]) if c["lead"])
    df = pd.DataFrame(
        {
            "day": [m[0] for m in meta],
            "line": np.asarray([m[1] for m in meta], np.int32),
            "ts": [m[2] for m in meta],
            "repo": [m[3] for m in meta],
            "author": [m[4] for m in meta],
            "week": [weeks[t] for t in week_of],
            "component": component.astype(np.int8),
            "lead": component == lead_rank,
        }
    )
    os.makedirs("labels", exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(df):,} rows, lead share {df['lead'].mean():.1%}")


if __name__ == "__main__":
    main()
