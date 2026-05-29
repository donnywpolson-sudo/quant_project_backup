import os

# --- CONFIGURATION ---
# 1. Find the directory where this script lives (the "audit" folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Set the root directory to one level up from the script's folder
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

# 3. Save the output file inside the "audit" folder
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "full_code.txt")

# Folders to completely ignore
IGNORE_DIRS = {
    '.git', '.venv', 'venv', 'env', 'node_modules', '__pycache__', 
    '.idea', '.vscode', 'build', 'dist', 'coverage', '.next'
}

# File extensions to ignore
IGNORE_EXTENSIONS = {
    '.pyc', '.pyo', '.exe', '.dll', '.so', '.dylib', 
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.webp',
    '.mp3', '.mp4', '.wav', '.avi', '.mov',
    '.zip', '.tar', '.gz', '.rar', '.7z',
    '.pdf', '.doc', '.docx', '.xls', '.xlsx',
    '.ttf', '.otf', '.woff', '.woff2',
    '.db', '.sqlite3', '.log', '.lock'
}

# Specific files to ignore (Updated to match your new file names)
IGNORE_FILES = {
    'generate_prompt.py', 
    'full_code.txt', 
    '.env', 
    '.env.local'
}

def generate_context():
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        # 1. GENERATE DIRECTORY TREE
        outfile.write("# PROJECT DIRECTORY STRUCTURE\n")
        outfile.write("=============================\n\n")
        
        for root, dirs, files in os.walk(ROOT_DIR):
            # Modify dirs in-place to prevent visiting ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
            
            # Safely calculate depth based on relative path to root
            rel_root = os.path.relpath(root, ROOT_DIR)
            if rel_root == '.':
                level = 0
            else:
                level = rel_root.count(os.sep) + 1
                
            indent = ' ' * 4 * level
            folder_name = os.path.basename(root)
            
            if folder_name or root == ROOT_DIR:
                outfile.write(f"{indent}{folder_name if folder_name else 'ROOT'}/\n")
            
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                if f not in IGNORE_FILES and not f.startswith('.'):
                    outfile.write(f"{subindent}{f}\n")

        # 2. GENERATE FILE CONTENTS
        outfile.write("\n\n# FILE CONTENTS\n")
        outfile.write("=================\n\n")
        
        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
            
            for file in files:
                if file in IGNORE_FILES:
                    continue
                
                ext = os.path.splitext(file)[1].lower()
                if ext in IGNORE_EXTENSIONS:
                    continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, ROOT_DIR)

                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        
                        outfile.write(f"--- START OF FILE: {rel_path} ---\n")
                        outfile.write(content)
                        if not content.endswith('\n'):
                            outfile.write('\n')
                        outfile.write(f"--- END OF FILE: {rel_path} ---\n\n")
                        
                except UnicodeDecodeError:
                    outfile.write(f"--- SKIPPED BINARY OR UNREADABLE FILE: {rel_path} ---\n\n")
                except Exception as e:
                    outfile.write(f"--- ERROR READING FILE: {rel_path} (Error: {e}) ---\n\n")

    print(f"Success! Project context written to:\n{OUTPUT_FILE}")

if __name__ == "__main__":
    generate_context()