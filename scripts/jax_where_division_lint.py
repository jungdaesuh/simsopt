"""Lint zero-adjacent ``jnp.where`` divisions that need explicit review."""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

_ALLOW_COMMENT = "jax-where-division-ok"


@dataclass(frozen=True)
class Finding:
    path: Path
    lineno: int
    message: str


def _is_jnp_where(call: ast.Call) -> bool:
    func = call.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "where"
        and isinstance(func.value, ast.Name)
        and func.value.id == "jnp"
    )


def _contains_division(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.BinOp) and isinstance(child.op, ast.Div)
        for child in ast.walk(node)
    )


def _is_zero_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value == 0 or node.value == 0.0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_zero_literal(node.operand)
    return False


def _predicate_is_zero_adjacent(node: ast.AST) -> bool:
    if not isinstance(node, ast.Compare):
        return False
    comparators = (node.left, *node.comparators)
    return any(_is_zero_literal(comparator) for comparator in comparators)


def _has_allow_comment(comments: Mapping[int, str], lineno: int) -> bool:
    same_line = comments.get(lineno, "")
    previous_line = comments.get(lineno - 1, "")
    return _ALLOW_COMMENT in same_line or _ALLOW_COMMENT in previous_line


def lint_source(path: Path, source: str, comments: Mapping[int, str]) -> list[Finding]:
    tree = ast.parse(source, filename=str(path))
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_jnp_where(node):
            continue
        if len(node.args) < 3:
            continue
        predicate, true_branch, false_branch = node.args[:3]
        if not _predicate_is_zero_adjacent(predicate):
            continue
        division_with_zero_branch = (
            _contains_division(true_branch)
            and _is_zero_literal(false_branch)
            or _contains_division(false_branch)
            and _is_zero_literal(true_branch)
        )
        if not division_with_zero_branch:
            continue
        if _has_allow_comment(comments, node.lineno):
            continue
        findings.append(
            Finding(
                path=path,
                lineno=node.lineno,
                message=(
                    "zero-adjacent jnp.where division needs a "
                    "jax-where-division-ok review comment"
                ),
            )
        )
    return findings


def _comments_by_line(source: str) -> dict[int, str]:
    return {
        line_number: line
        for line_number, line in enumerate(source.splitlines(), start=1)
        if "#" in line
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    all_findings: list[Finding] = []
    for path in args.paths:
        source = path.read_text(encoding="utf-8")
        all_findings.extend(lint_source(path, source, _comments_by_line(source)))

    for finding in all_findings:
        print(f"{finding.path}:{finding.lineno}: {finding.message}")
    return 1 if all_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
