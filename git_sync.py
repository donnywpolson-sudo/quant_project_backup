import subprocess
import sys
from datetime import datetime

def git_commit_and_push(commit_message=None):
    # Set default message if none is provided
    if not commit_message:
        commit_message = f"updates - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    try:
        # 1. Stage all changes
        subprocess.run(["git", "add", "."], check=True)
        
        # 2. Commit the changes
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # 3. Push to the current branch
        subprocess.run(["git", "push"], check=True)
        
        print(f"Successfully committed: '{commit_message}' and pushed.")
    
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # If a message is passed, use it. Otherwise, use the default.
    msg = sys.argv[1] if len(sys.argv) > 1 else None
    git_commit_and_push(msg)