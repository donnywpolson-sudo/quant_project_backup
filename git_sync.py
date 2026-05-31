#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from datetime import datetime


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def ensure_git_repo() -> None:
    p = run(["git", "rev-parse", "--is-inside-work-tree"], check=False)
    if p.returncode != 0 or p.stdout.strip() != "true":
        raise SystemExit("Not inside a git work tree.")


def has_changes() -> bool:
    return bool(run(["git", "status", "--porcelain"]).stdout.strip())


def has_staged_changes() -> bool:
    return run(["git", "diff", "--cached", "--quiet"], check=False).returncode == 1


def main() -> int:
    ensure_git_repo()

    # Pull first, preserving local uncommitted work.
    run(["git", "pull", "--rebase", "--autostash"])

    # Stage everything: modified, deleted, untracked.
    run(["git", "add", "-A"])

    if has_staged_changes():
        msg = "sync " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run(["git", "commit", "-m", msg])
        print(f"Committed: {msg}")
    else:
        print("No local changes to commit.")

    # Pull again in case remote changed while committing.
    run(["git", "pull", "--rebase"])

    run(["git", "push"])
    print("Synced.")
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