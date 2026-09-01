"""Extend the corpus backwards — the same protocol, applied to history.

Upstream's `fetch_day.py` deliberately stops at 2025-01-01; this driver reuses its `fetch`
(identical date-seeded windows, identical query, one immutable file per day) to walk BACKWARDS
from a boundary, so at every moment the corpus is contiguous whole weeks from the earliest
fetched day to the present. Progress is committed and pushed every `--commit-every` days, which
makes the sweep resumable from nothing but the repository: rerunning skips every day already on
disk.

    PYTHONPATH=. python scripts/backfill_history.py --from 2021-01-04 --to 2024-12-31
    # ~25 s/day against the search rate limit; safe to interrupt and rerun

`--from` should be a Monday (whole weeks are what the analysis keeps).
"""

import argparse
import os
import subprocess
import sys
from datetime import date, timedelta

from vendor.fetch_day import fetch, path


def commit_push(lo: date, hi: date) -> None:
    subprocess.run(["git", "add", "data/days"], check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode == 0:
        return
    subprocess.run(
        ["git", "commit", "-q", "-m", f"corpus: backfill {lo} .. {hi}"], check=True
    )
    subprocess.run(["git", "push", "-q", "origin", "main"], check=True)
    print(f"pushed {lo} .. {hi}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from", dest="lo", required=True, help="earliest day (a Monday)")
    ap.add_argument("--to", dest="hi", required=True, help="latest day of the sweep")
    ap.add_argument("--commit-every", type=int, default=28)
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or ""
    if not token:
        sys.exit("set GITHUB_TOKEN")
    lo, hi = date.fromisoformat(args.lo), date.fromisoformat(args.hi)
    if lo.weekday() != 0:
        sys.exit(f"--from {lo} is not a Monday; whole weeks are what the analysis keeps")

    days = [hi - timedelta(days=i) for i in range((hi - lo).days + 1)]
    done_since_commit, newest_pending, failed = 0, None, 0
    for day in days:
        if os.path.exists(path(day)):
            continue
        n = fetch(day, token)
        if n < 0:
            failed += 1
            print(f"{day}  failed (continuing)", flush=True)
            continue
        print(f"{day}  {n} descriptions", flush=True)
        newest_pending = newest_pending or day
        done_since_commit += 1
        if done_since_commit >= args.commit_every:
            commit_push(day, newest_pending)
            done_since_commit, newest_pending = 0, None
    if newest_pending is not None:
        commit_push(days[-1], newest_pending)
    print(f"sweep complete: {failed} failed days (rerun to retry)", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
