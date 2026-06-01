#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from datetime import datetime


COMMIT_PREFIX = "sync"


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


def has_staged_changes() -> bool:
    return run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 1


def ensure_synced_with_upstream() -> None:
    # Final check should observe state only; fetch is safe, pull is not.
    run(["git", "fetch"])

    status = run(["git", "status", "--porcelain=v1"]).stdout.strip()
    if status:
        raise SystemExit(f"Workspace not clean after sync:\n{status}")

    local = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    upstream = run(["git", "rev-parse", "@{u}"]).stdout.strip()

    if local != upstream:
        ahead = run(["git", "rev-list", "--count", "@{u}..HEAD"]).stdout.strip()
        behind = run(["git", "rev-list", "--count", "HEAD..@{u}"]).stdout.strip()
        raise SystemExit(
            f"HEAD does not match upstream after sync:\n"
            f"  local:    {local}\n"
            f"  upstream: {upstream}\n"
            f"  ahead:    {ahead}\n"
            f"  behind:   {behind}"
        )


def main() -> int:
    ensure_git_repo()
    ensure_upstream()

    # Bring in remote changes first while preserving local uncommitted work.
    run(["git", "pull", "--rebase", "--autostash"])

    # Stage all tracked/untracked modifications, deletions, and additions.
    run(["git", "add", "-A"])

    if has_staged_changes():
        msg = f"{COMMIT_PREFIX} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        run(["git", "commit", "-m", msg])
        print(f"Committed: {msg}")
    else:
        print("No local changes to commit.")

    # Rebase again in case remote changed while local commit was being created.
    run(["git", "pull", "--rebase"])

    run(["git", "push"])

    ensure_synced_with_upstream()

    print("Synced: workspace clean and HEAD matches upstream.")
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