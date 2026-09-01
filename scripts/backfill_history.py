"""Extend the corpus backwards to 2021 — coarse to fine, so answers arrive early.

Upstream's `fetch_day.py` deliberately stops at 2025-01-01; this driver reuses its `fetch`
(identical date-seeded windows, identical query, one immutable file per day) over anchor-aligned
whole weeks of 2021-2024, fetched in resolution order rather than chronologically:

    pass 0  half-yearly — the first whole week of each January and July (plus 2024-12-30/31,
            which complete the corpus's first 2025 week), 8 weeks: a yearly-resolution curve
    pass 1  quarterly   — the first whole week of each April and October
    pass 2  monthly     — the first whole week of every remaining month
    pass 3  fill        — every remaining week, newest first

Each pass refines the same estimand, so the history can be read at yearly resolution within
half an hour and sharpens as the sweep runs. Whole weeks only (the analysis drops part-weeks);
one commit+push per fetched week makes any interruption resumable from the repository alone —
rerunning skips every day already on disk.

    PYTHONPATH=. python scripts/backfill_history.py            # the full coarse-to-fine sweep
    PYTHONPATH=. python scripts/backfill_history.py --passes 2 # stop after the monthly pass
"""

import argparse
import os
import subprocess
import sys
from datetime import date, timedelta

from vendor.fetch_day import fetch, path

FIRST_MONDAY = date(2021, 1, 4)  # anchor-aligned: (2024-12-30 − this) is a multiple of 7
LAST_MONDAY = date(2024, 12, 23)  # the last whole pre-corpus week (ends 2024-12-29)


def mondays() -> list[date]:
    out, d = [], FIRST_MONDAY
    while d <= LAST_MONDAY:
        out.append(d)
        d += timedelta(days=7)
    return out


def first_week_of_month(year: int, month: int, weeks: set[date]) -> date | None:
    d = date(year, month, 1)
    d += timedelta(days=(7 - d.weekday()) % 7)  # first Monday on/after the 1st
    return d if d in weeks else None


def planned_weeks() -> list[tuple[int, date]]:
    """(pass_rank, monday) for every 2021-2024 week, coarse passes first."""
    all_weeks = set(mondays())
    rank_of: dict[date, int] = {}
    for rank, months in ((0, (1, 7)), (1, (4, 10)), (2, tuple(range(1, 13)))):
        for year in range(2021, 2025):
            for month in months:
                wk = first_week_of_month(year, month, all_weeks)
                if wk is not None and wk not in rank_of:
                    rank_of[wk] = rank
    plan = [(rank_of.get(w, 3), w) for w in all_weeks]
    # coarse passes chronologically; the fill pass newest-first
    plan.sort(key=lambda rw: (rw[0], -rw[1].toordinal() if rw[0] == 3 else rw[1].toordinal()))
    return plan


def commit_push(message: str) -> None:
    subprocess.run(["git", "add", "data/days"], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        return
    subprocess.run(["git", "commit", "-q", "-m", message], check=True)
    subprocess.run(["git", "push", "-q", "origin", "main"], check=True)


def fetch_days(days: list[date], token: str) -> int:
    failed = 0
    for day in days:
        if os.path.exists(path(day)):
            continue
        n = fetch(day, token)
        if n < 0:
            failed += 1
            print(f"{day}  failed (continuing)", flush=True)
        else:
            print(f"{day}  {n} descriptions", flush=True)
    return failed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--passes", type=int, default=4, help="run passes 0..N-1 (default: all 4)")
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        sys.exit("set GITHUB_TOKEN")

    failed = fetch_days([date(2024, 12, 30), date(2024, 12, 31)], token)
    commit_push("corpus: backfill 2024-12-30 .. 2024-12-31 (completes the first 2025 week)")

    current_rank = -1
    for rank, monday in planned_weeks():
        if rank >= args.passes:
            break
        if rank != current_rank:
            print(f"--- pass {rank} ---", flush=True)
            current_rank = rank
        days = [monday + timedelta(days=i) for i in range(7)]
        if all(os.path.exists(path(d)) for d in days):
            continue
        failed += fetch_days(days, token)
        commit_push(f"corpus: backfill week of {monday} (pass {rank})")
        print(f"WEEK_DONE {monday} pass={rank}", flush=True)
    print(f"sweep complete through pass {args.passes - 1}: {failed} failed days", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
