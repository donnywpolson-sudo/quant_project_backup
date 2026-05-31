#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import time

BATCH_SIZE = 5
COMMIT_MESSAGE = "batch commit"
DELAY_SECONDS = 2.0


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def ensure_git_repo() -> None:
    p = run(["git", "rev-parse", "--is-inside-work-tree"], check=False)
    if p.returncode != 0 or p.stdout.strip() != "true":
        raise SystemExit("Not inside a git work tree.")


def ensure_upstream() -> None:
    p = run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        check=False,
    )
    if p.returncode != 0:
        branch = run(["git", "branch", "--show-current"]).stdout.strip()
        raise SystemExit(
            f"No upstream configured.\n"
            f"Run once manually:\n"
            f"  git push -u origin {branch}"
        )


def changed_paths() -> list[str]:
    p = run(["git", "status", "--porcelain=v1", "-z"])
    entries = p.stdout.split("\0")
    paths: list[str] = []

    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1

        if not entry:
            continue

        xy = entry[:2]
        path = entry[3:]

        if xy[0] in {"R", "C"}:
            paths.append(path)
            if i < len(entries):
                i += 1
            continue

        if xy != "  ":
            paths.append(path)

    seen: set[str] = set()
    return [p for p in paths if not (p in seen or seen.add(p))]


def has_staged_changes() -> bool:
    return run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 1


def unpushed_count() -> int:
    p = run(["git", "rev-list", "--count", "@{u}..HEAD"], check=False)
    if p.returncode != 0:
        return 0
    return int(p.stdout.strip() or "0")


def stage_commit_push(batch: list[str], batch_num: int) -> None:
    print(f"\nBatch {batch_num}: staging {len(batch)} path(s)")
    for path in batch:
        print(f"  {path}")

    run(["git", "add", "--", *batch])

    if has_staged_changes():
        run(["git", "commit", "-m", f"{COMMIT_MESSAGE} {batch_num}"])
        print("Committed.")
    else:
        print("Nothing staged; skipping commit.")
        return

    run(["git", "push"])
    print("Pushed.")


def main() -> int:
    ensure_git_repo()
    ensure_upstream()

    batch_num = 1

    while True:
        paths = changed_paths()

        if paths:
            batch = paths[:BATCH_SIZE]
            stage_commit_push(batch, batch_num)
            batch_num += 1
            time.sleep(DELAY_SECONDS)
            continue

        ahead = unpushed_count()
        if ahead > 0:
            print(f"\nNo file changes left. Pushing {ahead} unpushed commit(s).")
            run(["git", "push"])
            time.sleep(DELAY_SECONDS)
            continue

        print("\nDone: no unstaged, uncommitted, or unpushed changes left.")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed: {' '.join(e.cmd)}", file=sys.stderr)
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        raise SystemExit(e.returncode)