"""Size limits for the code generation backend functions."""

import ast
import itertools
from pathlib import Path

import pytest

import thermodiff

MAX_CODE_LINES = 50
MAX_COMPLEXITY = 10
PACKAGE_ROOT = Path(thermodiff.__file__).parent
BACKEND_FILES = sorted(
    [*(PACKAGE_ROOT / "codegen").glob("*.py"), PACKAGE_ROOT / "chain.py"]
)


def _code_lines(node: ast.AST, source_lines: list[str]) -> int:
    """Count the code lines of ``node``, excluding docstring and comments."""
    body = node.body
    first = body[0]
    has_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    start = first.end_lineno + 1 if has_docstring else node.lineno
    stop = node.end_lineno
    span = itertools.islice(source_lines, start - 1, stop)
    return sum(
        1
        for line in span
        if line.strip() and not line.lstrip().startswith("#")
    )


def _functions(path: Path):
    """Yield every function definition of ``path``."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


@pytest.mark.parametrize("path", BACKEND_FILES, ids=lambda p: p.name)
def test_functions_have_at_most_fifty_code_lines(path):
    """No backend function exceeds the code line limit."""
    source_lines = path.read_text().splitlines()
    too_long = {
        node.name: _code_lines(node, source_lines)
        for node in _functions(path)
        if _code_lines(node, source_lines) > MAX_CODE_LINES
    }
    assert not too_long


@pytest.mark.parametrize("path", BACKEND_FILES, ids=lambda p: p.name)
def test_functions_have_bounded_complexity(path):
    """No backend function exceeds the McCabe complexity limit."""
    mccabe = pytest.importorskip("mccabe")
    visitor = mccabe.PathGraphingAstVisitor()
    visitor.preorder(ast.parse(path.read_text()), visitor)
    too_complex = {
        graph.entity: graph.complexity()
        for graph in visitor.graphs.values()
        if graph.complexity() > MAX_COMPLEXITY
    }
    assert not too_complex
