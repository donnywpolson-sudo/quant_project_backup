import subprocess
import sys
from datetime import datetime
from pathlib import Path
import glob


REPO_ROOT = Path(__file__).resolve().parent
INDEX_LOCK = REPO_ROOT / '.git' / 'index.lock'
GIT_ADD_PATHS = [
    '.gitattributes',
    '.gitignore',
    'README.md',
    'git_sync.py',
    'requirements.txt',
    'run.py',
    'project_layout.md',
    'code_to_text.py',
    'codex_custom_instructions.md',
    'configs',
    'pipeline',
    'tests',
    'data/L0_ohlcv_1m/*.py',
    'data/L1_mbp1/*.py',
    'data/market_sessions.yaml',
]


def run_git(args: list[str], *, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['git', *args],
        cwd=REPO_ROOT,
        text=True,
        check=True,
        timeout=timeout,
    )


def has_working_changes() -> bool:
    """Return True if the working tree has staged, unstaged, or untracked changes."""
    result = subprocess.run(
        ['git', 'status', '--porcelain'],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return bool(result.stdout.strip())


def has_staged_changes() -> bool:
    """Return True if there are changes staged for commit."""
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only'],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return bool(result.stdout.strip())


def has_tracked_path(pathspec: str) -> bool:
    result = subprocess.run(
        ['git', 'ls-files', '--', pathspec],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return bool(result.stdout.strip())


def existing_git_add_paths() -> list[str]:
    """Return existing add pathspecs so missing optional dirs do not abort git add."""
    paths: list[str] = []
    for raw in GIT_ADD_PATHS:
        if any(ch in raw for ch in '*?[]'):
            matches = sorted(glob.glob(str(REPO_ROOT / raw)))
            paths.extend(str(Path(m).relative_to(REPO_ROOT)).replace('\\', '/') for m in matches)
        elif (REPO_ROOT / raw).exists() or has_tracked_path(raw):
            paths.append(raw)
    return paths


def git_commit_and_push(commit_message: str | None = None) -> None:
    if INDEX_LOCK.exists():
        raise SystemExit(
            f'ERROR: Git lock exists: {INDEX_LOCK}\n'
            'Close/finish any Git operation first. If no Git process is running, delete this stale lock and rerun.'
        )

    if not has_working_changes():
        print('No Git changes to commit.')
        return

    if not commit_message:
        commit_message = f"updates - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    try:
        # Stage whitelisted repo paths. Missing optional paths are filtered out first.
        add_paths = existing_git_add_paths()
        if add_paths:
            run_git(['add', '-A', '--', *add_paths], timeout=600)
        else:
            # Fallback avoids a false no-op when repo contents exist outside the whitelist.
            # .gitignore still applies.
            run_git(['add', '-A'], timeout=600)

        if not has_staged_changes():
            print('No staged changes to commit.')
            return

        run_git(['commit', '-m', commit_message], timeout=300)
        run_git(['push', '-u', 'origin', 'HEAD'], timeout=600)

        print(f"Successfully committed: '{commit_message}' and pushed.")

    except subprocess.CalledProcessError as e:
        print(f'ERROR: Git command failed with exit code {e.returncode}: {e.cmd}')
    except subprocess.TimeoutExpired as e:
        print(f'ERROR: Git command timed out: {e.cmd}')


if __name__ == '__main__':
    msg = sys.argv[1] if len(sys.argv) > 1 else None
    git_commit_and_push(msg)
