import subprocess
import sys
from datetime import datetime
from pathlib import Path
import glob


REPO_ROOT = Path(__file__).resolve().parent
INDEX_LOCK = REPO_ROOT / ".git" / "index.lock"
GIT_ADD_PATHS = [
    ".gitattributes",
    ".gitignore",
    "README.md",
    "git_sync.py",
    "run.py",
    "quant_flowchart.md",
    "audit",
    "configs",
    "core",
    "pipeline",
    "tests",
    "data/L0_ohlcv_1m/*.py",
    "data/L1_mbp1/*.py",
    "data/market_sessions.yaml",
    "_legacy",
    "tools",
]


def run_git(args: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        check=True,
        timeout=timeout,
    )


def has_changes() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return bool(result.stdout.strip())


def existing_git_add_paths() -> list[str]:
    """Return only existing add pathspecs so missing optional dirs do not abort git add."""
    paths: list[str] = []
    for raw in GIT_ADD_PATHS:
        if any(ch in raw for ch in "*?[]"):
            matches = sorted(glob.glob(str(REPO_ROOT / raw)))
            paths.extend(str(Path(m).relative_to(REPO_ROOT)).replace("\\", "/") for m in matches)
        elif (REPO_ROOT / raw).exists():
            paths.append(raw)
    return paths

def git_commit_and_push(commit_message=None):
    if INDEX_LOCK.exists():
        raise SystemExit(
            f"ERROR: Git lock exists: {INDEX_LOCK}\n"
            "Close/finish any Git operation first. If no Git process is running, delete this stale lock and rerun."
        )

    if not has_changes():
        print("No Git changes to commit.")
        return

    # Set default message if none is provided
    if not commit_message:
        commit_message = f"updates - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    try:
        # 1. Stage all changes
        add_paths = existing_git_add_paths()
        if not add_paths:
            print("No existing code paths to stage.")
            return
        run_git(["add", "-A", "--", *add_paths], timeout=600)

        if not has_changes():
            print("No staged changes to commit.")
            return
        
        # 2. Commit the changes
        run_git(["commit", "-m", commit_message], timeout=300)
        
        # 3. Push to the current branch
        run_git(["push"], timeout=600)
        
        print(f"Successfully committed: '{commit_message}' and pushed.")
    
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    except subprocess.TimeoutExpired as e:
        print(f"ERROR: Git command timed out: {e.cmd}")

if __name__ == "__main__":
    # If a message is passed, use it. Otherwise, use the default.
    msg = sys.argv[1] if len(sys.argv) > 1 else None
    git_commit_and_push(msg)
