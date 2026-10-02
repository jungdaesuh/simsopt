"""The GSCO undo test is upstream's in both the C++ solver and the JAX kernel.

Until 2026-09-21 the branch carried a symmetric undo test (commit ``c20277ccd``)
that was not upstream's, recorded as a scoped deviation. On the user's decision
C2 the C++ went back to upstream's spelling and the JAX kernel follows it. This
test fails if either kernel returns to the symmetric form.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import inspect
from pathlib import Path

from simsopt_jax.core import wireframe_workflow

REPO_ROOT = Path(__file__).resolve().parents[3]
SOLVER = REPO_ROOT / "src" / "simsoptpp" / "wireframe_optimization.cpp"
BRANCH_UNDO_TEST = "((opt_ind + nLoops) % twoNLoops) == opt_ind_prev"
UPSTREAM_UNDO_TEST = "(opt_ind + nLoops % (twoNLoops)) == opt_ind_prev"


def test_both_kernels_use_upstreams_undo_test() -> None:
    solver = SOLVER.read_text()
    assert solver.count(UPSTREAM_UNDO_TEST) == 1
    assert BRANCH_UNDO_TEST not in solver
    kernel = inspect.getsource(wireframe_workflow._gsco_opposite_candidate_index)
    # Upstream's test compares the previous index with ``opt_ind + nLoops`` without
    # wrapping; a modulo over the doubled loop count is the symmetric branch form.
    assert "% _runtime_init_scalar(2 * n_loops" not in kernel
    assert "return opt_ind + _runtime_init_scalar(n_loops, opt_ind.dtype)" in kernel
