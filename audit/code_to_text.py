import os
import re

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "audit",
    "venv",
    ".venv",
}

TARGET_EXTENSIONS = (".py", ".yaml", ".yml")

MAX_LITERAL = 2000


def compress_large_literals(content: str) -> str:
    # Large triple-quoted strings
    content = re.sub(
        r'"""[\s\S]{2000,}?"""',
        '"""TRUNCATED_LARGE_STRING"""',
        content,
    )

    content = re.sub(
        r"'''[\s\S]{2000,}?'''",
        "'''TRUNCATED_LARGE_STRING'''",
        content,
    )

    return content


def normalize_spacing(content: str) -> str:
    # Collapse excessive blank lines only
    return re.sub(r"\n{3,}", "\n\n", content)


def should_skip(root: str) -> bool:
    parts = set(root.split(os.sep))
    return bool(parts & EXCLUDE_DIRS)


def git_ingest(output_dir="audit", output_filename="full_code.txt"):
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as outfile:

        for root, dirs, files in os.walk("."):

            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            if should_skip(root):
                continue

            for file in files:

                if not file.endswith(TARGET_EXTENSIONS):
                    continue

                path = os.path.join(root, file)

                outfile.write("=" * 80 + "\n")
                outfile.write(f"FILE: {path}\n")
                outfile.write("=" * 80 + "\n\n")

                try:
                    with open(path, "r", encoding="utf-8") as infile:
                        content = infile.read()

                    content = compress_large_literals(content)
                    content = normalize_spacing(content)

                    outfile.write(content)

                except Exception as e:
                    outfile.write(f"ERROR_READING_FILE: {e}")

                outfile.write("\n\n")

    print(f"Ingestion complete: {output_path}")


if __name__ == "__main__":
    git_ingest()
