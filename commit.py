import subprocess
import os
import sys

def github_commit_all(commit_message="Update project files"):
    try:
        # 1. Check if we are in a git repo
        if not os.path.exists(".git"):
            print("Initializing Git repository...")
            subprocess.run(["git", "init"], check=True)

        # 2. Check if there is a remote 'origin'
        result = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
        if result.returncode != 0:
            print("ERROR: No remote 'origin' found. Please add it manually:")
            print("  git remote add origin https://github.com/your-username/your-repo.git")
            sys.exit(1)

        # 3. Check for changes to commit
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not status.stdout.strip():
            print("No changes to commit. Working tree clean.")
            return

        # 4. Add all files (respects .gitignore)
        subprocess.run(["git", "add", "."], check=True)

        # 5. Commit
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        # 6. Get current branch name
        branch_result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
        current_branch = branch_result.stdout.strip()
        if not current_branch:
            current_branch = "main"  # fallback

        # 7. Push
        subprocess.run(["git", "push", "-u", "origin", current_branch], check=True)

        print(f"Successfully committed and pushed to '{current_branch}'.")

    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # You can pass a custom message as a command-line argument
    msg = sys.argv[1] if len(sys.argv) > 1 else "My automated commit"
    github_commit_all(msg)