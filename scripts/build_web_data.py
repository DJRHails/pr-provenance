"""Build the docs/ board: inject freshly computed SVG charts + data into docs/index.html.

Server-side geometry so the page is correct with JavaScript disabled and CI can regenerate it
(weekly job); the page's own script only adds the hover layer from `docs/provenance.js`. Charts
carry concrete light-mode fills as SVG attributes plus band classes, so the page's dark-mode CSS
can override by class while a rasterizer sees a complete light rendering.

    PYTHONPATH=. python scripts/build_web_data.py   # rewrites docs/index.html between markers
"""

import json
import re

import pandas as pd

BANDS = [
    ("human", "Human (unflagged)", "#d6d5cd"),
    ("unsigned_ai", "Unsigned, detector-flagged", "#e87ba4"),
    ("other_agents", "Other agents", "#eda100"),
    ("jules", "Jules", "#1baf7a"),
    ("codex", "Codex", "#eb6834"),
    ("claude_code", "Claude Code", "#2a78d6"),
]
DIRECT = {  # short forms for the right-edge direct labels; the legend carries the full names
    "human": "Human",
    "unsigned_ai": "Unsigned, flagged",
}
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 960, 380, 44, 150, 12, 28
VW, ROW_H, VPAD_T, VPAD_B = 960, 26, 8, 30


def weekly_bands() -> pd.DataFrame:
    assign = pd.read_parquet("labels/assignments.parquet")[["day", "line", "week", "lead"]]
    agents = pd.read_parquet("labels/agents.parquet")[["day", "line", "agent"]]
    full = pd.read_parquet("labels/isogram_full.parquet")[["day", "line", "iso_ai_touched"]]
    m = assign.merge(agents, on=["day", "line"]).merge(full, on=["day", "line"], how="left")
    m["iso_ai_touched"] = m["iso_ai_touched"].fillna(False)

    def band(row_agent, flagged):
        if row_agent == "claude-code":
            return "claude_code"
        if row_agent == "codex":
            return "codex"
        if row_agent == "jules":
            return "jules"
        if row_agent != "human_unsigned":
            return "other_agents"
        return "unsigned_ai" if flagged else "human"

    m["band"] = [band(a, f) for a, f in zip(m["agent"], m["iso_ai_touched"])]
    tbl = m.groupby(["week", "band"]).size().unstack(fill_value=0)
    for key, _, _ in BANDS:
        if key not in tbl:
            tbl[key] = 0
    return tbl[[k for k, _, _ in BANDS]].sort_index()


def version_counts() -> pd.DataFrame:
    ag = pd.read_parquet("labels/agents.parquet")
    v = ag[ag["model"].str.startswith("claude-", na=False) & (ag["model"] != "claude-unversioned")]
    v = v.assign(week=pd.to_datetime(v["day"]).dt.to_period("W-SUN").dt.start_time.dt.date)
    return v.groupby(["model", "week"]).size().reset_index(name="n")


def x_of(i, n, x0=PAD_L, x1=W - PAD_R):
    return x0 + (x1 - x0) * i / max(n - 1, 1)


def stack_svg(tbl: pd.DataFrame) -> tuple[str, dict]:
    weeks = list(tbl.index)
    n = len(weeks)
    shares = tbl.div(tbl.sum(axis=1), axis=0)
    y0, y1 = H - PAD_B, PAD_T
    parts, cum = [], pd.Series(0.0, index=tbl.index)
    band_paths = []
    for key, label, color in BANDS:
        lo, hi = cum.copy(), cum + shares[key]
        top = " ".join(
            f"{x_of(i, n):.1f},{y0 + (y1 - y0) * hi.iloc[i]:.1f}" for i in range(n)
        )
        bot = " ".join(
            f"{x_of(i, n):.1f},{y0 + (y1 - y0) * lo.iloc[i]:.1f}" for i in range(n - 1, -1, -1)
        )
        band_paths.append(
            f'<polygon class="band band-{key}" fill="{color}" stroke="#fcfcfb" '
            f'stroke-width="1.5" points="{top} {bot}"><title>{label}</title></polygon>'
        )
        cum = hi
    # y gridlines + labels
    for frac in (0.25, 0.5, 0.75):
        y = y0 + (y1 - y0) * frac
        parts.append(
            f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}"/>'
            f'<text class="tick" x="{PAD_L - 6}" y="{y + 3.5:.1f}" text-anchor="end">'
            f"{int(frac * 100)}%</text>"
        )
    parts += band_paths
    # x ticks: first week of each quarter
    seen = set()
    for i, wk in enumerate(weeks):
        ts = pd.Timestamp(wk)
        q = (ts.year, ts.quarter)
        if q in seen:
            continue
        seen.add(q)
        x = x_of(i, n)
        parts.append(
            f'<line class="grid" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + 4}"/>'
            f'<text class="tick" x="{x:.1f}" y="{y0 + 16}" text-anchor="middle">'
            f"{ts.year} Q{ts.quarter}</text>"
        )
    # direct labels at the right edge, centred in each band's final extent
    cum2 = 0.0
    for key, label, color in BANDS:
        s = shares[key].iloc[-1]
        if s > 0.03:
            ymid = y0 + (y1 - y0) * (cum2 + s / 2)
            parts.append(
                f'<text class="direct direct-{key}" x="{W - PAD_R + 8}" y="{ymid + 3.5:.1f}">'
                f"{DIRECT.get(key, label)}</text>"
            )
        cum2 += s
    hover = {
        "weeks": [str(w) for w in weeks],
        "bands": [[key, label] for key, label, _ in BANDS],
        "counts": tbl.astype(int).to_numpy().tolist(),
        "x0": PAD_L,
        "x1": W - PAD_R,
        "y0": y0,
        "y1": y1,
    }
    svg = (
        f'<svg id="stack" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Weekly provenance composition of pull-request descriptions">'
        + "".join(parts)
        + f'<rect id="stack-hit" x="{PAD_L}" y="{y1}" width="{W - PAD_R - PAD_L}" '
        f'height="{y0 - y1}" fill="transparent"/></svg>'
    )
    return svg, hover


def versions_svg(v: pd.DataFrame) -> tuple[str, list]:
    first_seen = v.groupby("model")["week"].min().sort_values()
    rows = list(first_seen.index)
    weeks = sorted(v["week"].unique())
    n = len(weeks)
    height = VPAD_T + ROW_H * len(rows) + VPAD_B
    x_lab = 190
    parts, dots = [], []
    for r, model in enumerate(rows):
        y = VPAD_T + ROW_H * (r + 0.5)
        parts.append(
            f'<line class="grid" x1="{x_lab}" y1="{y:.1f}" x2="{VW - 20}" y2="{y:.1f}"/>'
            f'<text class="rowlab" x="{x_lab - 8}" y="{y + 3.5:.1f}" text-anchor="end">'
            f"{model}</text>"
        )
        for _, row in v[v["model"] == model].iterrows():
            i = weeks.index(row["week"])
            x = x_lab + (VW - 20 - x_lab) * i / max(n - 1, 1)
            rad = min(1.8 * (row["n"] ** 0.5) + 1.2, 11)
            parts.append(
                f'<circle class="vdot" cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" '
                f'fill="#2a78d6" fill-opacity="0.75" data-model="{model}" '
                f'data-week="{row["week"]}" data-n="{int(row["n"])}">'
                f'<title>{model} — week of {row["week"]}: {int(row["n"])}</title></circle>'
            )
            dots.append([model, str(row["week"]), int(row["n"])])
    seen = set()
    yb = height - VPAD_B
    for i, wk in enumerate(weeks):
        ts = pd.Timestamp(wk)
        q = (ts.year, ts.quarter)
        if q in seen:
            continue
        seen.add(q)
        x = x_lab + (VW - 20 - x_lab) * i / max(n - 1, 1)
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{yb + 18}" text-anchor="middle">'
            f"{ts.year} Q{ts.quarter}</text>"
        )
    svg = (
        f'<svg id="versions" viewBox="0 0 {VW} {height}" role="img" '
        f'aria-label="Named Claude model versions in signature trailers, weekly">'
        + "".join(parts)
        + "</svg>"
    )
    return svg, dots


def stat_tiles(tbl: pd.DataFrame) -> str:
    recent = tbl.tail(4).sum()
    total_recent = recent.sum()
    ai = (total_recent - recent["human"]) / total_recent
    signed = (recent["claude_code"] + recent["codex"] + recent["jules"] + recent["other_agents"])
    tiles = [
        (f"{ai:.0%}", "AI-touched or agent-signed, last 4 weeks"),
        (f"{signed / total_recent:.0%}", "carry an explicit agent signature"),
        (f"{int(tbl.to_numpy().sum()):,}", "descriptions clustered, labelled and scored"),
    ]
    return "".join(
        f'<div class="tile"><div class="tile-n">{num}</div><div class="tile-l">{lab}</div></div>'
        for num, lab in tiles
    )


def quarter_table(tbl: pd.DataFrame) -> str:
    q = pd.PeriodIndex(pd.to_datetime(tbl.index), freq="Q").astype(str)
    byq = tbl.groupby(q).sum()
    shares = byq.div(byq.sum(axis=1), axis=0)
    head = "".join(f"<th>{label}</th>" for _, label, _ in BANDS)
    rows = "".join(
        "<tr><td>{}</td>{}</tr>".format(
            idx, "".join(f"<td>{shares.loc[idx, k]:.1%}</td>" for k, _, _ in BANDS)
        )
        for idx in byq.index
    )
    return f"<table><thead><tr><th>quarter</th>{head}</tr></thead><tbody>{rows}</tbody></table>"


def inject(html: str, marker: str, payload: str) -> str:
    pattern = re.compile(
        rf"""(?sx)
        (<!--\ {marker}\ -->)   # opening marker
        .*?                     # current payload, replaced
        (<!--\ /{marker}\ -->)  # closing marker
        """
    )
    # function replacement: a payload starting with a digit (a date) after \1 would otherwise be
    # read as a group reference, and backslashes in payloads would be escapes
    out, n = pattern.subn(lambda m: m.group(1) + payload + m.group(2), html)
    if n != 1:
        raise SystemExit(f"marker {marker} not found exactly once in docs/index.html")
    return out


def main():
    tbl = weekly_bands()
    stack, hover = stack_svg(tbl)
    vsvg, dots = versions_svg(version_counts())

    html = open("docs/index.html", encoding="utf-8").read()
    html = inject(html, "CHART:STACK", stack)
    html = inject(html, "CHART:VERSIONS", vsvg)
    html = inject(html, "TILES", stat_tiles(tbl))
    html = inject(html, "TABLE", quarter_table(tbl))
    html = inject(html, "UPDATED", str(tbl.index[-1]))
    open("docs/index.html", "w", encoding="utf-8").write(html)

    with open("docs/provenance.js", "w", encoding="utf-8") as fh:
        fh.write("window.PROVENANCE = ")
        json.dump({"stack": hover, "dots": dots}, fh, separators=(",", ":"))
        fh.write(";\n")
    print(f"rebuilt docs/: {len(tbl)} weeks, last {tbl.index[-1]}")


if __name__ == "__main__":
    main()
