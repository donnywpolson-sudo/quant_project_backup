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
                node.body.pop(0)
    return ast.unparse(tree)

def clean_text(text: str) -> str:
    text = re.sub('^\\s*#\\s*[=\\-]{3,}.*$', '', text, flags=re.MULTILINE)
    text = re.sub('^\\s*#\\s*✅.*$', '', text, flags=re.MULTILINE)
    text = re.sub('\\n{3,}', '\n\n', text)
    return text

def clean_file(path: Path):
    original = path.read_text(encoding='utf-8')
    try:
        cleaned = remove_docstrings(original)
        cleaned = clean_text(cleaned)
        ast.parse(cleaned)
    except Exception:
        print(f'Skipped (unsafe): {path}')
        return
    path.write_text(cleaned, encoding='utf-8')

def clean_repo(root='./'):
    for file in Path(root).rglob('*.py'):
        clean_file(file)
if __name__ == '__main__':
    clean_repo()