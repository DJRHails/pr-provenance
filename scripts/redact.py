"""Redact credential-shaped strings from fetched PR bodies.

PR descriptions occasionally contain leaked live-looking credentials (an AWS key in a public
2024 PR body blocked the first historical push via GitHub push protection). The corpus is
public data, but republishing credentials is poor hygiene and GitHub refuses the push — so
every fetched day is passed through these high-precision patterns before it is committed.
Matches become `[REDACTED:<kind>]`; everything else is byte-identical to what GitHub served.
This is the one deliberate deviation from mirroring bodies verbatim (bodies are also truncated
at 8,000 chars by the upstream fetcher).

    PYTHONPATH=. python scripts/redact.py data/days/2024-11-04.jsonl [...]   # in place
"""

import json
import re
import sys

_PATTERNS = [
    (
        "aws-access-key",
        re.compile(
            r"""(?x)
            \b
            (?: AKIA | ABIA | ACCA | ASIA )   # AWS access-key-id prefixes
            [0-9A-Z]{16}
            \b
            """
        ),
    ),
    (
        "github-token",
        re.compile(
            r"""(?x)
            \b
            (?: gh[pousr]_[A-Za-z0-9]{36,255}          # classic fine/oauth/user tokens
              | github_pat_[A-Za-z0-9_]{60,255}        # fine-grained PAT
            )
            \b
            """
        ),
    ),
    (
        "slack-token",
        re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,250}\b"),
    ),
    (
        "google-api-key",
        re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    ),
    (
        "openai-key",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,250}\b"),
    ),
    (
        "anthropic-key",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,250}\b"),
    ),
    (
        "stripe-key",
        re.compile(r"\b[rs]k_live_[0-9a-zA-Z]{20,250}\b"),
    ),
    (
        "private-key-block",
        re.compile(r"-----BEGIN [A-Z ]{0,20}PRIVATE KEY( BLOCK)?-----"),
    ),
    (
        "twilio-sid",
        # account/API-key string identifiers (AC/SK + 32 hex)
        re.compile(r"\b(?:AC|SK)[0-9a-fA-F]{32}\b"),
    ),
    (
        "postman-key",
        # PMAK = API key, PMAT = collection access key
        re.compile(r"\bPMA[KT]-[0-9A-Za-z-]{20,64}\b"),
    ),
    (
        "npm-token",
        re.compile(r"\bnpm_[A-Za-z0-9]{36}\b"),
    ),
    (
        "gitlab-pat",
        re.compile(r"\bglpat-[0-9A-Za-z_-]{20,50}\b"),
    ),
    (
        "sendgrid-key",
        re.compile(r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b"),
    ),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
]


def redact_text(text: str) -> tuple[str, int]:
    hits = 0
    for kind, pattern in _PATTERNS:
        text, n = pattern.subn(f"[REDACTED:{kind}]", text)
        hits += n
    return text, hits


def redact_file(path: str) -> int:
    """Redact one day file in place; returns the number of redactions."""
    out, hits = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            row["body"], n = redact_text(row.get("body") or "")
            hits += n
            out.append(json.dumps(row, ensure_ascii=False) + "\n")
    if hits:
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(out)
    return hits


if __name__ == "__main__":
    for p in sys.argv[1:]:
        n = redact_file(p)
        if n:
            print(f"{p}: {n} redactions")
