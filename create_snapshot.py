from pathlib import Path

def create_snapshot():
    # Anchor the script to its actual directory
    project_root = Path(__file__).parent.resolve()
    
    exclude = {'.venv', 'venv', 'env', '.git', '__pycache__', 'artifacts', 'logs', 'models', 'node_modules'}
    snapshot_filename = 'project_snapshot.txt'
    snapshot_path = project_root / snapshot_filename

    print(f"Generating snapshot in {snapshot_filename}...")

    # Define text extensions in lowercase for comparison
    text_extensions = {'.py', '.md', '.txt', '.yaml', '.yml', '.json', '.html', '.css', '.js', '.ts', '.tsx', '.jsx'}

    with open(snapshot_path, 'w', encoding='utf-8') as f:
        f.write("# Project Snapshot\n\n")
        
        # pathlib's rglob allows us to iterate cleanly
        for file_path in project_root.rglob('*'):
            # Skip directories and the snapshot files themselves
            if file_path.is_dir() or file_path.name in [Path(__file__).name, snapshot_filename]:
                continue
                
            # Check if any parent directory of the file is in the exclusion list
            if any(part in exclude for part in file_path.relative_to(project_root).parts):
                continue
                
            display_path = file_path.relative_to(project_root)
            f.write(f"--- \n### File: {display_path}\n")
            
            # Case-insensitive extension check
            if file_path.suffix.lower() in text_extensions:
                try:
                    # Using errors='replace' to guard against encoding crashes
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as sf:
                        snippet = sf.read(500) # 500 characters for context
                        
                        # Preserve formatting but wrap it nicely for LLM comprehension
                        f.write("```\n")
                        f.write(snippet)
                        if len(snippet) == 500:
                            f.write("\n... [Truncated]")
                        f.write("\n```\n\n")  # Fixed line here
                except Exception as e:
                    f.write(f"Could not read snippet: {e}\n\n")
            else:
                f.write("[Binary or Non-Text File]\n\n")

    print(f"Successfully created {snapshot_filename}")

if __name__ == '__main__':
    create_snapshot()