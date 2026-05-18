"""Lint zero-adjacent ``jnp.where`` branches that contain division."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
import tokenize


SUPPRESSION_TAG = "jax-where-division-ok"


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    message: str


def _is_jnp_where(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "where":
        return False
    value = func.value
    if isinstance(value, ast.Name):
        return value.id == "jnp"
    if isinstance(value, ast.Attribute) and value.attr == "numpy":
        return isinstance(value.value, ast.Name) and value.value.id == "jax"
    return False


def _contains_division(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.BinOp) and isinstance(child.op, ast.Div)
        for child in ast.walk(node)
    )


def _is_zero_literal(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value in (0, 0.0)


def _condition_mentions_zero(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Compare):
            if _is_zero_literal(child.left):
                return True
            if any(_is_zero_literal(comparator) for comparator in child.comparators):
                return True
    return False


def _suppressed(comment_lines: dict[int, str], node: ast.AST) -> bool:
    end_lineno = getattr(node, "end_lineno", node.lineno)
    first_line = max(1, node.lineno - 2)
    return any(
        SUPPRESSION_TAG in comment_lines.get(line_number, "")
        for line_number in range(first_line, end_lineno + 1)
    )


def _comment_lines(path: Path) -> dict[int, str]:
    comments: dict[int, str] = {}
    with tokenize.open(path) as handle:
        for token in tokenize.generate_tokens(handle.readline):
            if token.type == tokenize.COMMENT:
                comments[token.start[0]] = token.string
    return comments


def lint_source(
    path: Path, source: str, comment_lines: dict[int, str] | None = None
) -> list[Finding]:
    tree = ast.parse(source, filename=str(path))
    comments = comment_lines if comment_lines is not None else {}
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not _is_jnp_where(node):
            continue
        if len(node.args) < 3:
            continue
        condition, true_branch, false_branch = node.args[:3]
        if not _condition_mentions_zero(condition):
            continue
        if not (_contains_division(true_branch) or _contains_division(false_branch)):
            continue
        if _suppressed(comments, node):
            continue
        findings.append(
            Finding(
                path=path,
                line=node.lineno,
                message=(
                    "jnp.where with a zero-adjacent predicate selects a division "
                    "branch; use a safe denominator/double-where idiom or add "
                    f"#{SUPPRESSION_TAG}: <reason> after review."
                ),
            )
        )
    return findings


def _iter_python_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
        elif path.suffix == ".py":
            files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    findings: list[Finding] = []
    for path in _iter_python_files(args.paths):
        source = path.read_text(encoding="utf-8")
        findings.extend(lint_source(path, source, _comment_lines(path)))

    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.message}", file=sys.stderr)
    return 1 if findings else 0
