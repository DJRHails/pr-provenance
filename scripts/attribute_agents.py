"""Explicit agent/model attribution for every pull-request description in `data/days/`.

Coding agents sign their work: Claude Code appends a `Generated with [Claude Code]` footer and a
`Co-authored-by: Claude <model>` trailer, Codex appends a `[Codex Task](chatgpt.com/codex/…)`
link, Cursor's background agent wraps the body in `CURSOR_AGENT_PR_BODY` markers, Devin links its
session, Jules and Copilot author under their own logins. This script scans every raw row
(bot-authored rows included — the clustering excludes them, these labels do not) and records:

- `agent`  — the authoring agent, from the strongest explicit marker (`human_unsigned` when none)
- `model_raw` / `model` — the model string where the signature names one (Claude co-author
  trailers and aider trailers do; Copilot/Codex/Devin footers do not), plus a normalised id
- `bot_author` — the author-login bot test the upstream clustering uses to drop rows
- `review_mentions` — review-bot boilerplate (Bugbot, Gemini Code Assist, CodeRabbit, qodo,
  Devin's review badge) that indicates tooling on the repo but NOT authorship of the body

Attribution is conservative: a marker must be the agent's own signature, not a mention of the
tool in prose, so `agent` under-counts AI authorship (an unsigned Claude PR reads as
`human_unsigned`) and essentially never over-counts it. That direction is deliberate — the
downstream comparison asks how much MORE the detector finds than the signatures admit.

    PYTHONPATH=. python scripts/attribute_agents.py     # writes labels/agents.parquet
"""

import glob
import json
import os
import re

import pandas as pd

OUT = "labels/agents.parquet"

_CLAUDE_COAUTHOR = re.compile(
    r"""(?ix)                     # case-insensitive, verbose
    co-authored-by:\s*
    (?P<name>claude[^<\n(]{0,60})  # the co-author display name, e.g. "Claude Opus 4.8"
    """
)
_AIDER_COAUTHOR = re.compile(
    r"""(?ix)
    co-authored-by:\s*aider\s*
    \(  (?P<model>[^)\n]{1,80})  \)   # aider names the model in parens
    """
)
_CLAUDE_MODEL = re.compile(
    r"""(?ix)
    claude \s+
    (?P<family>opus|sonnet|haiku|fable)   # model family
    \s* (?P<major>\d+) (?:\.(?P<minor>\d+))?   # version, e.g. 4.8 or 5
    """
)

# (agent, compiled body regex) in priority order: the first authorship signature wins.
_BODY_SIGNATURES = [
    (
        "claude-code",
        re.compile(
            r"""(?ix)
            generated\ with\ \[?claude\ code   # the footer line
            | claude\.ai/code                  # old footer URL
            | claude\.com/claude-code          # new footer URL
            """
        ),
    ),
    (
        "codex",
        re.compile(
            r"""(?ix)
            chatgpt\.com/codex          # Codex task-link footer
            | \[codex\ task\]
            """
        ),
    ),
    (
        "cursor",
        re.compile(
            r"""(?ix)
            cursor_agent_pr_body        # background-agent body markers
            | cursor\.com/agents/
            | co-authored-by:\s*cursor
            """
        ),
    ),
    (
        "devin",
        re.compile(
            r"""(?ix)
            link\ to\ devin\ session
            | app\.devin\.ai/sessions
            """
        ),
    ),
    (
        "jules",
        re.compile(r"(?i)jules\.google\.com"),
    ),
    (
        "openhands",
        re.compile(
            r"""(?ix)
            ai\ agent\ \(openhands\)
            | app\.all-hands\.dev/conversations
            """
        ),
    ),
    (
        "copilot",
        re.compile(
            r"""(?ix)
            co-authored-by:\s*copilot
            | copilot-swe-agent
            """
        ),
    ),
    (
        "aider",
        re.compile(r"(?i)co-authored-by:\s*aider"),
    ),
    (
        "gemini",
        re.compile(
            r"""(?ix)
            generated-by:\s*gemini
            | co-authored-by:\s*gemini
            """
        ),
    ),
    (
        "warp",
        re.compile(
            r"""(?ix)
            app\.warp\.dev/conversation
            | co-authored-by:\s*warp
            """
        ),
    ),
    (
        "amp",
        re.compile(r"(?i)ampcode\.com/threads"),
    ),
    (
        "claude-coauthor-only",  # a Claude trailer without the Claude Code footer
        re.compile(r"(?i)co-authored-by:\s*claude"),
    ),
]

# author logins that ARE the agent
_AUTHOR_AGENTS = {
    "copilot": "copilot",
    "devin-ai-integration[bot]": "devin",
    "google-labs-jules[bot]": "jules",
    "cursor[bot]": "cursor",
    "openhands-agent": "openhands",
    "sweep-ai[bot]": "sweep",
    "codegen-sh[bot]": "codegen",
}

_REVIEW_MENTIONS = [
    ("bugbot", re.compile(r"(?i)cursor\.com/bugbot")),
    ("gemini-code-assist", re.compile(r"(?i)gemini\ code\ assist")),
    ("coderabbit", re.compile(r"(?i)coderabbit")),
    ("qodo", re.compile(r"(?i)qodo|pr-agent")),
    ("devin-review", re.compile(r"(?i)devin-review-badge")),
]

_BOT_SUFFIX = ("[bot]", "-bot")


def normalize_claude(name: str) -> str:
    """`Claude Opus 4.8 (1M context)` → `claude-opus-4-8`; bare `Claude` → `claude-unversioned`."""
    m = _CLAUDE_MODEL.search(name)
    if not m:
        return "claude-unversioned"
    version = m.group("major") + (f"-{m.group('minor')}" if m.group("minor") else "")
    return f"claude-{m.group('family').lower()}-{version}"


def attribute(author: str, body: str) -> tuple[str, str, str]:
    """(agent, model_raw, model) for one row."""
    login = (author or "").lower()
    agent = _AUTHOR_AGENTS.get(login)
    if agent is None:
        for name, pat in _BODY_SIGNATURES:
            if pat.search(body):
                agent = name
                break
    if agent is None:
        return "human_unsigned", "", ""

    if agent in ("claude-code", "claude-coauthor-only"):
        m = _CLAUDE_COAUTHOR.search(body)
        raw = m.group("name").strip() if m else ""
        return agent, raw, normalize_claude(raw) if raw else "claude-unversioned"
    if agent == "aider":
        m = _AIDER_COAUTHOR.search(body)
        raw = m.group("model").strip() if m else ""
        return agent, raw, raw.lower()
    return agent, "", ""


def main():
    rows = []
    for path in sorted(glob.glob("data/days/*.jsonl")):
        day = os.path.basename(path).removesuffix(".jsonl")
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh):
                r = json.loads(line)
                author, body = r.get("author") or "", r["body"]
                agent, model_raw, model = attribute(author, body)
                login = author.lower()
                rows.append(
                    {
                        "day": day,
                        "line": lineno,
                        "agent": agent,
                        "model_raw": model_raw,
                        "model": model,
                        "bot_author": login.endswith(_BOT_SUFFIX) or login == "copilot",
                        "review_mentions": ",".join(
                            n for n, p in _REVIEW_MENTIONS if p.search(body)
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    os.makedirs("labels", exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT}: {len(df):,} rows")
    print(df["agent"].value_counts().to_string())


if __name__ == "__main__":
    main()
