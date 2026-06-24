"""Regression tests for the AST-based cached-kernel callback compatibility check.

The check guards JAX's persistent compilation cache: a cached kernel that
references a host/debug callback primitive (``jax.debug.callback`` and friends)
poisons the on-disk cache. These tests pin the detector's *observable* behavior
on synthetic kernels. Several cases (aliased imports, ``partial`` application,
decorators) are exactly the ones a naive substring scan answered wrong with a
false PASS; the comment/docstring/identifier-collision cases are the ones it
answered wrong with a false FAIL.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from benchmarks.check_cached_kernel_callback_compatibility import (
    FORBIDDEN_CALLBACK_QUALNAMES,
    _find_target_function,
    _forbidden_callbacks_in_function,
    evaluate_cached_kernel_callback_compatibility,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _forbidden_in(tmp_path: Path, source: str, symbol: str = "kernel") -> tuple[str, ...]:
    module_path = tmp_path / "synthetic_module.py"
    module_path.write_text(textwrap.dedent(source), encoding="utf-8")
    function = _find_target_function(module_path, symbol)
    return _forbidden_callbacks_in_function(function)


# --- detection: true positives (must be caught) ---------------------------------


def test_direct_dotted_callback_is_detected(tmp_path: Path) -> None:
    found = _forbidden_in(
        tmp_path,
        """
        import jax

        def kernel(x):
            jax.debug.callback(host_fn, x)
            return x
        """,
    )
    assert found == ("jax.debug.callback",)


def test_aliased_callback_import_is_detected(tmp_path: Path) -> None:
    # A naive substring scan over the body sees only ``cb(...)`` and PASSes —
    # this is the dangerous false-PASS the AST alias resolution closes.
    found = _forbidden_in(
        tmp_path,
        """
        from jax.debug import callback as cb

        def kernel(x):
            cb(host_fn, x)
            return x
        """,
    )
    assert found == ("jax.debug.callback",)


def test_partial_application_of_pure_callback_is_detected(tmp_path: Path) -> None:
    found = _forbidden_in(
        tmp_path,
        """
        import functools
        import jax

        def kernel(x):
            bridge = functools.partial(jax.pure_callback, host_fn, x.shape)
            return bridge(x)
        """,
    )
    assert found == ("jax.pure_callback",)


def test_aliased_host_callback_module_call_is_detected(tmp_path: Path) -> None:
    found = _forbidden_in(
        tmp_path,
        """
        import jax.experimental.host_callback as hcb

        def kernel(x):
            hcb.call(host_fn, x)
            return x
        """,
    )
    assert found == ("jax.experimental.host_callback.call",)


def test_star_imported_callback_is_detected(tmp_path: Path) -> None:
    # ``from jax import *`` supplies ``pure_callback`` by bare name; the detector
    # must treat the star module as a candidate prefix for unbound names.
    found = _forbidden_in(
        tmp_path,
        """
        from jax import *

        def kernel(x):
            return pure_callback(host_fn, x)
        """,
    )
    assert found == ("jax.pure_callback",)


def test_decorator_callback_is_detected(tmp_path: Path) -> None:
    # ``node.lineno`` points at ``def``, so a body-only source slice excludes the
    # decorator. The AST walk includes ``decorator_list``, so it is caught.
    found = _forbidden_in(
        tmp_path,
        """
        import jax

        @jax.debug.callback
        def kernel(x):
            return x
        """,
    )
    assert found == ("jax.debug.callback",)


def test_alias_imported_in_except_fallback_is_detected(tmp_path: Path) -> None:
    # The aliasing import lives in an ``except`` branch; scope traversal must
    # descend into the handler body, not only the ``try`` body.
    found = _forbidden_in(
        tmp_path,
        """
        try:
            import unavailable_module as cb
        except ImportError:
            from jax.debug import callback as cb

        def kernel(x):
            cb(host_fn, x)
            return x
        """,
    )
    assert found == ("jax.debug.callback",)


def test_async_target_function_is_resolved_and_scanned(tmp_path: Path) -> None:
    found = _forbidden_in(
        tmp_path,
        """
        import jax

        async def kernel(x):
            jax.debug.print("{}", x)
            return x
        """,
    )
    assert found == ("jax.debug.print",)


# --- detection: true negatives (must NOT be flagged) ----------------------------


def test_callback_name_in_comment_is_not_detected(tmp_path: Path) -> None:
    # A substring scan FAILs this callback-free kernel; the AST scan does not.
    found = _forbidden_in(
        tmp_path,
        """
        def kernel(x):
            # deliberately avoid jax.debug.callback in this cached kernel
            return x + 1
        """,
    )
    assert found == ()


def test_callback_name_in_docstring_is_not_detected(tmp_path: Path) -> None:
    found = _forbidden_in(
        tmp_path,
        '''
        def kernel(x):
            "This kernel must not use jax.pure_callback."
            return x
        ''',
    )
    assert found == ()


def test_identifier_collision_is_not_detected(tmp_path: Path) -> None:
    # ``id_print``/``host_callback`` are substrings of innocent identifiers.
    found = _forbidden_in(
        tmp_path,
        """
        def kernel(x):
            id_print_counter = 0
            host_callback_registry = {}
            return x + id_print_counter
        """,
    )
    assert found == ()


def test_parameter_shadowing_an_alias_is_not_falsely_flagged(tmp_path: Path) -> None:
    # The function param ``cb`` shadows the module import inside the body, so
    # ``cb(0)`` is a local call, not the imported callback.
    found = _forbidden_in(
        tmp_path,
        """
        from jax.debug import callback as cb

        def kernel(cb):
            return cb(0)
        """,
    )
    assert found == ()


def test_clean_kernel_has_no_findings(tmp_path: Path) -> None:
    found = _forbidden_in(
        tmp_path,
        """
        import jax.numpy as jnp

        def kernel(x):
            return jnp.sum(x ** 2)
        """,
    )
    assert found == ()


# --- resolver fail-loud + real-tree canary --------------------------------------


def test_duplicate_symbol_raises(tmp_path: Path) -> None:
    module_path = tmp_path / "dup.py"
    module_path.write_text(
        textwrap.dedent(
            """
            def kernel(x):
                return x

            def kernel(y):
                return y
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="found 2"):
        _find_target_function(module_path, "kernel")


def test_every_forbidden_qualname_is_fully_dotted() -> None:
    # Each entry must be a canonical dotted path (the resolver compares against
    # these verbatim); a bare token here would silently never match.
    assert all("." in qualname for qualname in FORBIDDEN_CALLBACK_QUALNAMES)


def test_real_repo_targets_resolve_and_pass() -> None:
    # Canary: every hard-coded target symbol still resolves to exactly one
    # function in the live tree and is callback-free / carries its guard.
    result = evaluate_cached_kernel_callback_compatibility(_REPO_ROOT)
    assert result["passed"] is True, result["failures"]
