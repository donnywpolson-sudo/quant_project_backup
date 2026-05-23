from pathlib import Path

def create_snapshot():
    project_root = Path(__file__).parent.resolve()
    
    # Files to always capture in full (Audit Requirement)
    important_files = {'ES.yaml', 'CL.yaml', 'ZB.yaml', 'run_full_pipeline.py', 'config.py'}
    
    exclude = {'.venv', 'venv', 'env', '.git', '__pycache__', 'artifacts', 'logs', 'models', 'node_modules'}
    snapshot_filename = 'project_snapshot.txt'
    
    with open(project_root / snapshot_filename, 'w', encoding='utf-8') as f:
        f.write("# Project Snapshot\n\n")
        
        for file_path in project_root.rglob('*'):
            # Filter
            if file_path.is_dir() or file_path.name in [Path(__file__).name, snapshot_filename]:
                continue
            if any(part in exclude for part in file_path.relative_to(project_root).parts):
                continue
                
            f.write(f"--- \n### File: {file_path.relative_to(project_root)}\n")
            
            # Logic: Full dump for critical, snippet for others
            if file_path.name in important_files:
                f.write("```\n" + file_path.read_text(encoding='utf-8', errors='replace') + "\n```\n\n")
            else:
                snippet = file_path.read_text(encoding='utf-8', errors='replace')[:500]
                f.write("```\n" + snippet + ("\n... [Truncated]" if len(snippet)==500 else "") + "\n```\n\n")

    print(f"Successfully created {snapshot_filename}")

if __name__ == '__main__':
    create_snapshot()