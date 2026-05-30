import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "full_code.txt")

MAX_FILE_CHARS = 80_000

INCLUDE_EXTENSIONS = {
    ".py", ".yaml", ".yml", ".toml", ".json", ".md", ".txt"
}

IGNORE_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".idea", ".vscode", "build", "dist", "coverage", ".next",
    "output", "outputs", "artifacts", "logs", "cache", ".cache",
    "data", "raw_data", "processed_data",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".kilo"
}

IGNORE_FILES = {
    "generate_prompt.py",
    "full_code.txt",
    ".env",
    ".env.local",
    "credentials.json",
    "secrets.json",
    "token.json",
    "kilo.jsonc",
}

SECRET_PATTERNS = [
    (re.compile(r"db-[A-Za-z0-9_\-]{20,}"), "db-REDACTED"),
    (re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)['\"]?[^'\"\n]+"), r"\1REDACTED"),
    (re.compile(r"(?i)(secret\s*[:=]\s*)['\"]?[^'\"\n]+"), r"\1REDACTED"),
    (re.compile(r"(?i)(password\s*[:=]\s*)['\"]?[^'\"\n]+"), r"\1REDACTED"),
    (re.compile(r"(?i)(token\s*[:=]\s*)['\"]?[^'\"\n]+"), r"\1REDACTED"),
]

def should_skip_file(file):
    if file in IGNORE_FILES or file.startswith("."):
        return True
    ext = os.path.splitext(file)[1].lower()
    return ext not in INCLUDE_EXTENSIONS

def redact(text):
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def generate_context():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        outfile.write("# PROJECT DIRECTORY STRUCTURE\n")
        outfile.write("=============================\n\n")

        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

            rel_root = os.path.relpath(root, ROOT_DIR)
            level = 0 if rel_root == "." else rel_root.count(os.sep) + 1
            indent = " " * 4 * level
            outfile.write(f"{indent}{os.path.basename(root) or 'ROOT'}/\n")

            subindent = " " * 4 * (level + 1)
            for f in sorted(files):
                if not should_skip_file(f):
                    outfile.write(f"{subindent}{f}\n")

        outfile.write("\n\n# FILE CONTENTS\n")
        outfile.write("=================\n\n")

        for root, dirs, files in os.walk(ROOT_DIR):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith(".")]

            for file in sorted(files):
                if should_skip_file(file):
                    continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, ROOT_DIR)

                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        content = infile.read(MAX_FILE_CHARS + 1)

                    truncated = len(content) > MAX_FILE_CHARS
                    content = content[:MAX_FILE_CHARS]
                    content = redact(content)

                    outfile.write(f"--- START OF FILE: {rel_path} ---\n")
                    outfile.write(content)
                    if not content.endswith("\n"):
                        outfile.write("\n")
                    if truncated:
                        outfile.write(f"\n--- TRUNCATED: {rel_path} exceeded {MAX_FILE_CHARS} chars ---\n")
                    outfile.write(f"--- END OF FILE: {rel_path} ---\n\n")

                except UnicodeDecodeError:
                    outfile.write(f"--- SKIPPED UNREADABLE FILE: {rel_path} ---\n\n")
                except Exception as e:
                    outfile.write(f"--- ERROR READING FILE: {rel_path} ({e}) ---\n\n")

    print(f"Success. Project context written to:\n{OUTPUT_FILE}")

if __name__ == "__main__":
    generate_context()