# pr-provenance

**Who — or what — wrote this pull request?** A dataset of GitHub pull-request descriptions with
per-document provenance labels: the *way of writing* it clusters into, the *coding agent* that
signed it, the *model* where the signature names one, and an *AI-text-detector score* from
[isogram](https://github.com/DJRHails/isogram) (private repo; an EditLens replication).

Bootstrapped from **[louisabraham/load-bearing](https://github.com/louisabraham/load-bearing)**
(MIT) — the corpus behind *[The load-bearing vocabulary of
Claude](https://louisabraham.github.io/load-bearing/)* — whose methodology this repo first
replicates exactly and then extends from aggregates to per-document labels.

## Replication, verified

Upstream fits KL k-means (k=10) over word counts of sampled PR descriptions and shows one
cluster "arriving": 0.9% of the corpus in early 2025 → 37.4% by late August 2026, with
`load-bearing`, `plainly`, `genuinely`, `nowhere`, `premise` among its most characteristic
words — vocabulary distinctive of Claude.

- Re-running vendored `analyze.py` on the bootstrapped corpus (upstream commit `a446ffb`)
  reproduces the published `analysis.js` **byte-identically** (467,387 documents, 52,506,137
  word appearances, cost 129,872,592.4, seed 3, arrival confirmed).
- `scripts/assign_clusters.py` re-runs the published fit and exports what the page never shows —
  the cluster of every kept document — asserting exactness at every seam: document count,
  appearance count, fit cost under the published seed, and every component's weekly count series.
- `verification/refetch_2026-08-30.jsonl` is our **independent re-collection** of a day already
  in the corpus (the sampling windows are seeded on the date): 998/1000 rows overlap with
  upstream's file, 997/998 bodies byte-identical (the deltas are PRs edited or deleted since).

## Layout

| path | what it is |
|---|---|
| `data/days/YYYY-MM-DD.jsonl` | the corpus: ~1,000 PR descriptions/day (`ts`, `repo`, `author`, `body`), 2025-01-01 →, immutable files. Through 2026-08-30 bootstrapped from upstream; grown daily by `.github/workflows/daily.yml` after that. |
| `vendor/` | upstream `fetch_day.py` + `analyze.py`, unmodified (MIT, `LICENSE.upstream`, commit in `UPSTREAM_COMMIT`) |
| `analysis/analysis.js` | our re-run of the upstream fit (byte-identical to the published one) |
| `labels/assignments.parquet` | per kept document (`day`, `line` → `week`, `component` 0–9 in the published order, `lead`) |
| `labels/agents.parquet` | per raw document: signing `agent`, `model_raw`/`model` where named, `bot_author`, review-bot mentions |
| `labels/stylometric.parquet` | unsigned kept documents: predicted agent + confidence (bag-of-words over prose with signatures and URLs stripped) |
| `labels/isogram_sample.parquet` / `labels/isogram_scores.parquet` | stratified 250/week sample (21,500 docs) and its detector verdicts |
| `scripts/` | everything above is one script each; all run from a clean checkout (`PYTHONPATH=. python scripts/<x>.py`) |

Join key throughout: **(`day`, `line`)** — the day file and 0-based line number in it.

## Results

<!-- RESULTS -->

## Provenance & licensing

MIT. The corpus through 2026-08-30 and the vendored pipeline are Louis Abraham's (MIT,
attribution in `LICENSE`); every label, script and result in `labels/`, `scripts/` and this
README is ours. GitHub PR bodies are public data collected via GitHub's search API under its
rate limits; author logins are retained as published by GitHub.
