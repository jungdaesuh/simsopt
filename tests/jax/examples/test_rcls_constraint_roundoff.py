"""Reduction-order tolerance cannot excuse changed inputs or infeasible currents."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases.native_wireframe_rcls_basic import (
    CONSTRAINT_FEASIBILITY_RELATIVE_LIMIT,
    _constraint_feasibility_limit,
    _constraint_feasible,
    _initial_constraint_comparison,
    _reduction_operation_count,
)
from simsopt_jax.parity_tolerances import PARITY_LADDER_TOLERANCES

ROUNDOFF_BUCKET = PARITY_LADDER_TOLERANCES["rcls_constraint_roundoff"]

#: Unit roundoff of float64, as the helper under test spells it
#: (``native_wireframe_rcls_basic.py``: ``np.finfo(np.float64).eps / 2.0``).
#: Higham's ``(n - 1) u S`` is stated in ``u``, not in ``eps``.
UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0

#: Two independently ordered evaluations are compared, each with its own error.
LANE_PAIR = 2.0

#: Sparse rows in the shape the shipped constraint matrix has: at
#: ``native_default`` ``C`` is (95, 192) with 4 to 16 nonzeros per row.
SPARSE_ROWS = 12
SPARSE_COLUMNS = 192
SPARSE_NONZEROS = 4
SPARSE_TRIALS = 20


def _operands() -> dict[str, np.ndarray]:
    return {
        "constraint_matrix": np.asarray([[1.0, -1.0], [0.0, 0.0]]),
        "constraint_target": np.asarray([[0.0], [0.0]]),
        "initial_currents": np.asarray([[312500.0], [312500.0]]),
        "free_segments": np.asarray([0, 1]),
    }


def _dense_operands() -> dict[str, np.ndarray]:
    """Operands whose every row has a strictly positive rounding scale."""
    generator = np.random.default_rng(20260920)
    constraint = generator.standard_normal((5, 7))
    currents = 3.125e5 * generator.standard_normal((7, 1))
    return {
        "constraint_matrix": constraint,
        "constraint_target": constraint @ currents,
        "initial_currents": currents,
        "free_segments": np.arange(7),
    }


def _units(arrays: dict[str, np.ndarray], residual: np.ndarray) -> np.ndarray:
    return _initial_constraint_comparison(arrays, residual)[
        "initial:constraint_roundoff_units"
    ]


def _pair_bound(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Recover the published bound from the helper itself, not from a copy."""
    probe = np.ones_like(arrays["constraint_target"])
    return 1.0 / _units(arrays, probe)


def _pair_route_passes(
    arrays: dict[str, np.ndarray],
    left_residual: np.ndarray,
    right_residual: np.ndarray,
) -> bool:
    """Apply the real bucket exactly as ``arbiter.py`` applies it."""
    return bool(
        np.allclose(
            _units(arrays, left_residual),
            _units(arrays, right_residual),
            rtol=ROUNDOFF_BUCKET["rtol"],
            atol=ROUNDOFF_BUCKET["atol"],
        )
    )


def _residual(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate ``C x0 - b`` the way both lanes do: one dense product."""
    return (
        arrays["constraint_matrix"]
        @ arrays["initial_currents"][arrays["free_segments"]]
        - arrays["constraint_target"]
    )


def _block_sparse_operands(
    generator: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """A padded system, its zero-column-free twin, and the EXACT residual.

    Both systems describe the same arithmetic on the same numbers: the padded
    matrix multiplies and sums ``SPARSE_COLUMNS`` entries of which all but
    ``SPARSE_NONZEROS`` are exactly zero, the compacted one multiplies and sums
    the nonzero ones only. The exact residual is evaluated with
    :class:`fractions.Fraction`, so it contains no rounding at all.
    """
    columns = generator.choice(SPARSE_COLUMNS, size=SPARSE_NONZEROS, replace=False)
    values = generator.standard_normal((SPARSE_ROWS, SPARSE_NONZEROS))
    currents = 3.125e5 * generator.standard_normal((SPARSE_COLUMNS, 1))
    padded = np.zeros((SPARSE_ROWS, SPARSE_COLUMNS))
    padded[:, columns] = values
    target = np.zeros((SPARSE_ROWS, 1))
    exact = np.asarray(
        [
            [
                float(
                    sum(
                        (
                            Fraction(float(values[row, term]))
                            * Fraction(float(currents[columns[term], 0]))
                            for term in range(SPARSE_NONZEROS)
                        ),
                        Fraction(0),
                    )
                    - Fraction(float(target[row, 0]))
                )
            ]
            for row in range(SPARSE_ROWS)
        ]
    )
    padded_arrays = {
        "constraint_matrix": padded,
        "constraint_target": target,
        "initial_currents": currents,
        "free_segments": np.arange(SPARSE_COLUMNS),
    }
    compact_arrays = {
        "constraint_matrix": padded[:, columns],
        "constraint_target": target,
        "initial_currents": currents[columns],
        "free_segments": np.arange(SPARSE_NONZEROS),
    }
    return padded_arrays, compact_arrays, exact


def test_roundoff_bucket_is_the_a_priori_bound_and_is_not_tuned() -> None:
    # `atol 1.0` means "one a-priori pair bound". Retuning it toward an
    # observed drift (0.0839 units cross-lane on the retained native_default
    # receipt) would turn a derived bound into a measured one; this test exists
    # to make that edit fail.
    assert ROUNDOFF_BUCKET["rtol"] == 0.0
    assert ROUNDOFF_BUCKET["atol"] == 1.0
    assert ROUNDOFF_BUCKET["requires_same_state"] is True


def test_zero_columns_add_no_rounding_so_they_do_not_enlarge_the_bound() -> None:
    # An exact zero entry rounds nothing: `0.0 * x` is exact and `s + 0.0` is
    # exact, for every summation order and blocking. So a padded row and its
    # compacted twin perform the same INEXACT operations, and the bound built
    # from the operation count must not depend on the padding.
    # Measured first: the two evaluations stay within one compacted pair bound
    # of each other (they would not if padding really added 188 rounding
    # operations), and their exact values are identical.
    # Asserted second: same count, and the same bound up to the rounding of the
    # bound's own scale sum. That sum has `nnz + 1` non-negative terms (the
    # nonzero products and |b|), and padding leaves those terms and their
    # products untouched, so the two evaluations differ only in the ORDER numpy
    # sums them in: `(n - 1) u = nnz * u` per ordering, twice for two orderings.
    # Measured worst over the 20 trials: 4.68e-16 against this 8.88e-16.
    generator = np.random.default_rng(20260920)
    scale_sum_rtol = LANE_PAIR * float(SPARSE_NONZEROS) * UNIT_ROUNDOFF
    for _ in range(SPARSE_TRIALS):
        padded_arrays, compact_arrays, exact = _block_sparse_operands(generator)
        compact_bound = _pair_bound(compact_arrays)
        difference = np.abs(_residual(padded_arrays) - _residual(compact_arrays))
        assert np.all(difference <= compact_bound)
        np.testing.assert_array_equal(
            _reduction_operation_count(padded_arrays["constraint_matrix"]),
            _reduction_operation_count(compact_arrays["constraint_matrix"]),
        )
        np.testing.assert_allclose(
            _pair_bound(padded_arrays),
            compact_bound,
            rtol=scale_sum_rtol,
            atol=0.0,
        )
        assert np.all(np.abs(_residual(padded_arrays) - exact) <= compact_bound)


def test_bound_covers_the_rounding_of_a_dense_evaluation_of_sparse_rows() -> None:
    # Soundness against exact arithmetic: half the pair bound is the one-lane
    # bound, and it must cover the real error of the dense product. The worst
    # measured case uses about 0.49 of it, so an operation count below the
    # nonzero count (the subtraction alone divides the bound by five here)
    # makes this test fail.
    generator = np.random.default_rng(20260921)
    for _ in range(SPARSE_TRIALS):
        padded_arrays, _, exact = _block_sparse_operands(generator)
        one_lane_bound = 0.5 * _pair_bound(padded_arrays)
        assert np.all(np.abs(_residual(padded_arrays) - exact) <= one_lane_bound)


def test_the_bound_follows_the_nonzero_count_at_an_equal_rounding_scale() -> None:
    # Two rows with the SAME rounding scale `|c|^T |x| + |b|` (two terms of 1.0
    # against four of 0.5, on a constant current vector, all exact in binary)
    # but different nonzero counts. The bound must be strictly larger for the
    # row that performs more inexact operations; a matrix-wide count (the
    # padded width, or the largest row) makes the two equal. The second
    # assertion is the per-row property: row 0's bound does not change because
    # a denser row 1 exists next to it.
    currents = np.full((SPARSE_COLUMNS, 1), 3.125e5)

    def arrays_for(matrix: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "constraint_matrix": matrix,
            "constraint_target": np.zeros((matrix.shape[0], 1)),
            "initial_currents": currents,
            "free_segments": np.arange(SPARSE_COLUMNS),
        }

    two_terms = np.zeros((1, SPARSE_COLUMNS))
    two_terms[0, :2] = 1.0
    four_terms = np.zeros((1, SPARSE_COLUMNS))
    four_terms[0, 4:8] = 0.5
    matrix = np.concatenate((two_terms, four_terms))
    scale = np.abs(matrix) @ np.abs(currents)
    assert scale[0, 0] == scale[1, 0]
    bound = _pair_bound(arrays_for(matrix))
    assert bound[1, 0] > bound[0, 0]
    assert bound[0, 0] == _pair_bound(arrays_for(two_terms))[0, 0]


def test_the_bound_judges_the_lane_difference_not_one_lanes_residual() -> None:
    # An infeasible initial current vector makes one lane's residual larger
    # than the bound in EXACT arithmetic, so no rounding bound could cover it
    # -- exactly the situation of the shipped operands, whose exact initial
    # residual is 2.9103830456733704e-09 on 84 of 95 rows. What the bound
    # covers is the difference between two lanes, so two lanes that agree pass
    # the real route while a single lane's units exceed 1.
    generator = np.random.default_rng(20260923)
    padded_arrays, _, exact = _block_sparse_operands(generator)
    padded_arrays["constraint_target"] = (
        padded_arrays["constraint_target"] + 4.0 * exact
    )
    residual = _residual(padded_arrays)
    assert np.abs(residual).max() > _pair_bound(padded_arrays).max()
    assert np.abs(_units(padded_arrays, residual)).max() > 1.0
    assert _pair_route_passes(padded_arrays, residual, residual)


def test_pair_difference_just_below_the_bound_passes() -> None:
    arrays = _dense_operands()
    bound = _pair_bound(arrays)
    zero = np.zeros_like(bound)
    assert _pair_route_passes(arrays, zero, 0.999 * bound)


def test_pair_difference_just_above_the_bound_fails() -> None:
    arrays = _dense_operands()
    bound = _pair_bound(arrays)
    zero = np.zeros_like(bound)
    assert not _pair_route_passes(arrays, zero, 1.001 * bound)


def test_one_row_over_the_bound_fails_even_with_every_other_row_equal() -> None:
    arrays = _dense_operands()
    bound = _pair_bound(arrays)
    perturbed = np.zeros_like(bound)
    perturbed[2, 0] = 1.001 * bound[2, 0]
    assert not _pair_route_passes(arrays, np.zeros_like(bound), perturbed)


def test_roundoff_scaled_residual_preserves_exact_zero_rows() -> None:
    arrays = _operands()
    residual = np.asarray([[2.0e-12], [0.0]])
    result = _initial_constraint_comparison(arrays, residual)
    assert result["initial:constraint_satisfied"]
    assert 0 < result["initial:constraint_roundoff_units"][0, 0] < 1
    assert result["initial:constraint_roundoff_units"][1, 0] == 0
    np.testing.assert_array_equal(residual, [[2e-12], [0.0]])


def test_impossible_residual_on_zero_row_fails_closed() -> None:
    result = _initial_constraint_comparison(_operands(), np.asarray([[0.0], [1e-15]]))
    assert not np.isfinite(result["initial:constraint_roundoff_units"][1, 0])


def test_constraint_error_larger_than_roundoff_fails_the_pair_bound() -> None:
    arrays = _operands()
    result = _initial_constraint_comparison(arrays, np.asarray([[1e-6], [0.0]]))
    assert not result["initial:constraint_satisfied"]
    assert not _pair_route_passes(
        arrays,
        np.asarray([[0.0], [0.0]]),
        np.asarray([[1e-6], [0.0]]),
    )


def test_shared_physical_violation_is_rejected_even_when_lanes_agree() -> None:
    arrays = _operands()
    arrays["initial_currents"] *= 1e10
    violation = np.asarray([[1e-3], [0.0]])
    result = _initial_constraint_comparison(arrays, violation)
    # Huge cancelling terms make this smaller than the rounding bound, and two
    # lanes that agree on it pass the pair route -- only the physical gate
    # rejects it.
    assert abs(result["initial:constraint_roundoff_units"][0, 0]) < 1
    assert _pair_route_passes(arrays, violation, violation)
    assert not result["initial:constraint_satisfied"]


def test_feasibility_limit_is_one_relative_policy_scaled_by_the_target() -> None:
    small = np.asarray([[0.25], [0.5]])
    large = np.asarray([[2.5e6], [0.0]])
    assert _constraint_feasibility_limit(small) == (
        CONSTRAINT_FEASIBILITY_RELATIVE_LIMIT
    )
    assert _constraint_feasibility_limit(large) == (
        CONSTRAINT_FEASIBILITY_RELATIVE_LIMIT * 2.5e6
    )


def test_feasibility_gate_rejects_over_limit_and_non_finite_residuals() -> None:
    target = np.asarray([[2.5e6], [0.0]])
    limit = _constraint_feasibility_limit(target)
    assert _constraint_feasible(np.asarray([[limit], [0.0]]), target)
    assert not _constraint_feasible(
        np.asarray([[np.nextafter(limit, 1.0)], [0.0]]),
        target,
    )
    assert not _constraint_feasible(np.asarray([[np.nan], [0.0]]), target)


@pytest.mark.parametrize(
    "case_id", ("native-wireframe-rcls-basic", "native-wireframe-rcls-with-ports")
)
def test_both_rcls_contracts_require_exact_operands_at_each_scale(case_id: str) -> None:
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "examples/jax/parity_manifest.json").read_text())
    relationship = next(
        row for row in manifest["relationships"] if row["case_id"] == case_id
    )
    groups = [relationship["comparison_routes"]]
    groups.extend(
        contract["comparison_routes"]
        for contract in relationship["scale_contracts"].values()
    )
    for routes in groups:
        for phase, observable in (
            ("construction", "constraint_matrix"),
            ("construction", "constraint_target"),
            ("construction", "free_segments"),
            ("initial", "currents"),
        ):
            matches = [
                row
                for row in routes
                if row["phase"] == phase and row["observable"] == observable
            ]
            assert len(matches) == 3
            assert all(
                row["comparator"] == "exact" and row["applicable"] for row in matches
            )
        roundoff = [
            row for row in routes if row["observable"] == "constraint_roundoff_units"
        ]
        assert len(roundoff) == 3
        assert all(
            row["applicable"] and row["tolerance_bucket"] == "rcls_constraint_roundoff"
            for row in roundoff
        )
        satisfied = [
            row for row in routes if row["observable"] == "constraint_satisfied"
        ]
        assert len(satisfied) == 3
        assert all(
            row["applicable"] and row["comparator"] == "exact" for row in satisfied
        )
