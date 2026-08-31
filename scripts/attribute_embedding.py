"""Family and model identification from the detector's own embeddings.

Uses isogram's pooled representation (extracted by isogram's `scripts/embed_pr_corpus.py` over
`labels/embed_input.parquet` — signature lines and URLs stripped) instead of surface n-grams:

1. **Family probe** — multinomial logistic regression over the embedding, trained on
   signature-labelled rows (claude-code / codex / jules / cursor), author-disjoint validation,
   probabilities calibrated with isotonic regression on a held-out calibration fold.
2. **Model probe** — the harder, model-SPECIFIC question: within Claude-signed rows whose
   trailer names a version (`claude-opus-4-8`, `claude-sonnet-5`, …), can the embedding
   separate versions? Grouped 5-fold CV by author; classes with >=30 rows.
3. **Application** — calibrated family posteriors for every unsigned detector-flagged row in
   the embed input, written to `labels/embedding_attribution.parquet`.

    PYTHONPATH=. python scripts/attribute_embedding.py \
        --embeddings .data/pr_embeddings.npy --keys .data/pr_embeddings.parquet
"""

import argparse

import numpy as np
import pandas as pd

FAMILIES = ["claude-code", "codex", "jules", "cursor"]
OUT = "labels/embedding_attribution.parquet"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--embeddings", default=".data/pr_embeddings.npy")
    ap.add_argument("--keys", default=".data/pr_embeddings.parquet")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler

    X = np.load(args.embeddings).astype(np.float32)
    keys = pd.read_parquet(args.keys)
    assert len(X) == len(keys), (len(X), len(keys))
    authors = pd.read_parquet("labels/assignments.parquet")[["day", "line", "author"]]
    keys = keys.merge(authors, on=["day", "line"], how="left")
    keys["author"] = keys["author"].fillna("")

    scaler = StandardScaler()

    # ---- family probe (author-disjoint train / calibration / validation) ----
    fam = keys[keys["agent"].isin(FAMILIES)].copy()
    h = pd.util.hash_array(fam["author"].to_numpy(dtype=object)) % 100
    tr, cal, va = (h >= 30), (h >= 15) & (h < 30), (h < 15)
    Xf = scaler.fit_transform(X[fam.index])
    from sklearn.frozen import FrozenEstimator

    base = LogisticRegression(max_iter=3000, C=1.0)
    base.fit(Xf[tr], fam["agent"][tr])
    clf = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
    clf.fit(Xf[cal], fam["agent"][cal])
    print("== family probe (embedding, author-disjoint validation) ==")
    print(classification_report(fam["agent"][va], clf.predict(Xf[va]), digits=3))
    conf = clf.predict_proba(Xf[va]).max(axis=1)
    correct = clf.predict(Xf[va]) == fam["agent"][va].to_numpy()
    for lo, hi in [(0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            print(
                f"  calibration: conf [{lo:.2f},{hi:.2f}) n={m.sum():5d} "
                f"mean_conf={conf[m].mean():.3f} accuracy={correct[m].mean():.3f}"
            )

    # ---- model probe: named Claude versions, grouped CV by author ----
    ver = keys[
        keys["model"].str.startswith("claude-", na=False)
        & (keys["model"] != "claude-unversioned")
    ].copy()
    counts = ver["model"].value_counts()
    ver = ver[ver["model"].isin(counts[counts >= 30].index)]
    print(f"\n== model probe: {len(ver)} rows, {ver['model'].nunique()} versions ==")
    Xv = scaler.transform(X[ver.index])
    groups = ver["author"].where(ver["author"] != "", ver["day"])
    pred = cross_val_predict(
        LogisticRegression(max_iter=3000, C=1.0),
        Xv,
        ver["model"],
        groups=groups,
        cv=GroupKFold(n_splits=5),
    )
    print(classification_report(ver["model"], pred, digits=3, zero_division=0))
    chance = ver["model"].value_counts(normalize=True).iloc[0]
    print(f"majority-class baseline accuracy: {chance:.3f}; "
          f"macro-F1: {f1_score(ver['model'], pred, average='macro', zero_division=0):.3f}")

    # ---- apply: calibrated family posteriors for the unsigned flagged population ----
    uns = keys[keys["group"] == "unsigned_flagged"]
    proba = clf.predict_proba(scaler.transform(X[uns.index]))
    out = uns[["day", "line", "week", "component"]].copy()
    out["family_pred"] = clf.classes_[proba.argmax(axis=1)]
    out["family_conf"] = proba.max(axis=1).astype(np.float32)
    for j, c in enumerate(clf.classes_):
        out[f"p_{c.replace('-', '_')}"] = proba[:, j].astype(np.float32)
    out.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}: {len(out):,} unsigned flagged rows")
    print(out["family_pred"].value_counts(normalize=True).round(3).to_string())
    hi = out[out["family_conf"] > 0.8]
    print(f"conf>0.8 ({len(hi):,} rows):")
    print(hi["family_pred"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
