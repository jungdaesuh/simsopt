from __future__ import annotations

from pathlib import Path

from simsopt._maintenance import jax_where_division_lint as _LINT


def _findings(source: str):
    comments = {
        line_number: line
        for line_number, line in enumerate(source.splitlines(), start=1)
        if "#" in line
    }
    return _LINT.lint_source(Path("example.py"), source, comments)


def test_flags_zero_adjacent_where_division_branch():
    findings = _findings(
        """
import jax.numpy as jnp

def f(x):
    return jnp.where(x > 0.0, 1.0 / x, 0.0)
"""
    )

    assert len(findings) == 1


def test_allows_reviewed_where_division_branch():
    findings = _findings(
        """
import jax.numpy as jnp

def f(x):
    # jax-where-division-ok: denominator is clamped before this selection.
    return jnp.where(x > 0.0, 1.0 / x, 0.0)
""",
    )

    assert findings == []


def test_ignores_division_without_zero_adjacent_predicate():
    findings = _findings(
        """
import jax.numpy as jnp

def f(mask, x):
    return jnp.where(mask, 1.0 / x, 0.0)
"""
    )

    assert findings == []
