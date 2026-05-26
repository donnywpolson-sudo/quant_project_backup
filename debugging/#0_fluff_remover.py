import ast
from pathlib import Path
import re

def remove_docstrings(source):
    parsed = ast.parse(source)
    for node in ast.walk(parsed):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            if node.body and isinstance(node.body[0], ast.Expr):
                if isinstance(node.body[0].value, ast.Str):
                    node.body[0] = ast.Pass()
    return ast.unparse(parsed)

def clean_file(path):
    text = Path(path).read_text()
    text = remove_docstrings(text)
    text = re.sub('#.*={3,}.*', '', text)
    text = re.sub('#.*-{3,}.*', '', text)
    text = re.sub('# *✅.*', '', text)
    text = re.sub('\\n{3,}', '\n\n', text)
    Path(path).write_text(text)

def clean_repo(root='./'):
    for file in Path(root).rglob('*.py'):
        clean_file(file)
if __name__ == '__main__':
    clean_repo()