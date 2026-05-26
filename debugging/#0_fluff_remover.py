import ast
from pathlib import Path
import re


def remove_docstrings(source: str) -> str:
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.body:
                continue

            first = node.body[0]

            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                # Remove docstring safely
                node.body.pop(0)

    return ast.unparse(tree)


def clean_text(text: str) -> str:
    # ONLY remove clearly decorative comments (safe patterns)
    text = re.sub(r'^\s*#\s*[=\-]{3,}.*$', '', text, flags=re.MULTILINE)

    # Remove standalone emoji checklist comments
    text = re.sub(r'^\s*#\s*✅.*$', '', text, flags=re.MULTILINE)

    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def clean_file(path: Path):
    original = path.read_text(encoding='utf-8')

    try:
        cleaned = remove_docstrings(original)
        cleaned = clean_text(cleaned)

        # ✅ CRITICAL: verify code still parses
        ast.parse(cleaned)

    except Exception:
        # If anything goes wrong, DO NOT overwrite
        print(f"Skipped (unsafe): {path}")
        return

    path.write_text(cleaned, encoding='utf-8')


def clean_repo(root='./'):
    for file in Path(root).rglob('*.py'):
        clean_file(file)


if __name__ == '__main__':
    clean_repo()