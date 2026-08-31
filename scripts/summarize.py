"""Assemble the README results tables from the label parquets.

Reads labels/{assignments,agents,stylometric,isogram_scores}.parquet and prints the markdown
that goes into README.md's Results section, so every published number is reproducible by
re-running one script over the committed labels.

    PYTHONPATH=. python scripts/summarize.py
"""

import pandas as pd

pd.set_option("display.width", 200)


def quarter(week: pd.Series) -> pd.Series:
    ts = pd.to_datetime(week)
    return ts.dt.year.astype(str) + "-Q" + ts.dt.quarter.astype(str)


def main():
    assign = pd.read_parquet("labels/assignments.parquet")
    agents = pd.read_parquet("labels/agents.parquet")
    stylo = pd.read_parquet("labels/stylometric.parquet")
    scores = pd.read_parquet("labels/isogram_scores.parquet")

    kept = assign.merge(agents, on=["day", "line"], how="left")
    kept["q"] = quarter(kept["week"])
    signed = kept["agent"] != "human_unsigned"

    print("### Signed-agent share of the kept corpus, by quarter\n")
    tbl = (
        kept.assign(signed=signed)
        .groupby("q")
        .agg(docs=("signed", "size"), signed_share=("signed", "mean"))
    )
    agent_q = (
        kept[signed]
        .groupby(["q", "agent"])
        .size()
        .unstack(fill_value=0)
        .reindex(tbl.index, fill_value=0)
    )
    top = agent_q.sum().sort_values(ascending=False).head(5).index
    out = pd.concat([tbl, (agent_q[top].T / tbl["docs"]).T.round(3)], axis=1)
    out["signed_share"] = out["signed_share"].round(3)
    print(out.to_markdown())

    print("\n### Named Claude models in signature trailers (all raw rows)\n")
    named = agents[
        agents["model"].str.startswith("claude-", na=False)
        & (agents["model"] != "claude-unversioned")
    ]
    print(named["model"].value_counts().to_markdown())

    sc = scores.merge(agents, on=["day", "line"], how="left")
    sc["q"] = quarter(sc["week"])
    sc["signed"] = sc["agent"] != "human_unsigned"

    print("\n### isogram verdicts on the stratified sample (250 docs/week)\n")
    print(f"threshold: bundle val-fit; sample n={len(sc):,}\n")
    tbl = sc.groupby("q").agg(
        n=("iso_ai_touched", "size"),
        ai_touched=("iso_ai_touched", "mean"),
        mean_score=("iso_score", "mean"),
        signed_share=("signed", "mean"),
    )
    print(tbl.round(3).to_markdown())

    print("\n### Detector vs the signatures (sample)\n")
    tbl = sc.groupby("signed").agg(
        n=("iso_ai_touched", "size"),
        ai_touched=("iso_ai_touched", "mean"),
        mean_score=("iso_score", "mean"),
    )
    tbl.index = tbl.index.map({True: "signed by an agent", False: "unsigned"})
    print(tbl.round(3).to_markdown())
    per_agent = (
        sc[sc["signed"]]
        .groupby("agent")
        .agg(n=("iso_ai_touched", "size"), ai_touched=("iso_ai_touched", "mean"))
        .sort_values("n", ascending=False)
    )
    print()
    print(per_agent.round(3).to_markdown())

    print("\n### Detector vs the clustering (sample)\n")
    tbl = sc.assign(lead=sc["lead"].map({True: "lead ('Claude') cluster", False: "other 9"})).groupby(
        "lead"
    ).agg(n=("iso_ai_touched", "size"), ai_touched=("iso_ai_touched", "mean"))
    print(tbl.round(3).to_markdown())

    print("\n### Unsigned rows the detector flags: closest agent style (sample)\n")
    uns = sc[~sc["signed"]].merge(
        stylo[["day", "line", "predicted_agent", "confidence"]], on=["day", "line"], how="left"
    )
    flagged = uns[uns["iso_ai_touched"] & uns["predicted_agent"].notna()]
    tbl = flagged["predicted_agent"].value_counts().to_frame("n")
    tbl["share"] = (tbl["n"] / len(flagged)).round(3)
    print(f"unsigned & flagged with a style prediction: n={len(flagged):,}\n")
    print(tbl.to_markdown())
    print("\nby period:")
    flagged = flagged.assign(q=quarter(flagged["week"]))
    print(
        flagged.groupby(["q", "predicted_agent"]).size().unstack(fill_value=0).to_markdown()
    )


if __name__ == "__main__":
    main()
