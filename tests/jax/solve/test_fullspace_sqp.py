from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax.solve.fullspace_sqp as sqp_module
from simsopt_jax.geo.optimizers.dense_sqp import (
    DenseSQPOptions,
    DenseSQPResult,
    PreparedDenseSQP,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceConstraints,
    FullSpaceEvaluation,
    FullSpaceKKTPrimitives,
    FullSpaceObservables,
    FullSpaceProblem,
    FullSpaceRawTerms,
)
from simsopt_jax.solve.fullspace import FullSpaceScaling
from simsopt_jax.solve.fullspace_sqp import (
    PreparedCfsSqp1,
    cfs_sqp1_endpoint_diagnostics,
    cfs_sqp1_joint_linearization,
    cfs_sqp1_joint_value_constraints,
    cfs_sqp1_joint_vjp_rows,
    cfs_sqp1_raw_multipliers,
    prepare_cfs_sqp1,
)

jax.config.update("jax_enable_x64", True)

_STATE_SIZE = 716
_EQUALITY_SIZE = 255
_CONSTRAINT_MATRIX = jnp.pad(
    jnp.diag(jnp.linspace(1.0, 2.0, _EQUALITY_SIZE, dtype=jnp.float64)),
    ((0, 0), (0, _STATE_SIZE - _EQUALITY_SIZE)),
)


def _evaluation(state: jax.Array, _problem: object) -> FullSpaceEvaluation:
    constraints = _CONSTRAINT_MATRIX @ state
    objective = 0.5 * jnp.vdot(state, state)
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    return FullSpaceEvaluation(
        raw_terms=FullSpaceRawTerms(
            non_qs=objective,
            residual=zero,
            iota=zero,
            major_radius=zero,
            length=zero,
        ),
        weighted_total=objective,
        constraints=FullSpaceConstraints(
            boozer=constraints[:-1],
            volume=constraints[-1],
        ),
        observables=FullSpaceObservables(
            iota=state[-2],
            G=state[-1],
            volume=constraints[-1],
            major_radius=zero,
            total_length=zero,
            non_qs_ratio=objective,
            boozer_residual_scalar=zero,
            boozer_residual_rms=zero,
        ),
    )


def _kkt(
    state: jax.Array,
    raw_multipliers: jax.Array,
    _problem: object,
) -> FullSpaceKKTPrimitives:
    constraints = _CONSTRAINT_MATRIX @ state
    stationarity = state + _CONSTRAINT_MATRIX.T @ raw_multipliers
    return FullSpaceKKTPrimitives(
        objective_value=0.5 * jnp.vdot(state, state),
        constraint_residual=constraints,
        stationarity_residual=stationarity,
        primal_feasibility_inf=jnp.max(jnp.abs(constraints)),
        stationarity_inf=jnp.max(jnp.abs(stationarity)),
        all_finite=(
            jnp.all(jnp.isfinite(state))
            & jnp.all(jnp.isfinite(raw_multipliers))
            & jnp.all(jnp.isfinite(constraints))
            & jnp.all(jnp.isfinite(stationarity))
        ),
    )


@pytest.fixture(autouse=True)
def _install_test_physics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sqp_module, "evaluate_fullspace", _evaluation)
    monkeypatch.setattr(sqp_module, "fullspace_kkt_primitives", _kkt)


def _problem() -> object:
    return SimpleNamespace(layout=SimpleNamespace(total_dof_count=_STATE_SIZE))


def _scaling() -> FullSpaceScaling:
    return FullSpaceScaling(
        bootstrap_anchor=jnp.linspace(-0.2, 0.2, _STATE_SIZE, dtype=jnp.float64),
        variable_scale=jnp.linspace(0.01, 2.0, _STATE_SIZE, dtype=jnp.float64),
        constraint_inverse_scale=jnp.linspace(
            0.5, 1.5, _EQUALITY_SIZE, dtype=jnp.float64
        ),
    )


def test_joint_callback_is_exact_phi_and_scaled_boozer_then_volume() -> None:
    scaling = _scaling()
    optimizer_coordinates = jnp.linspace(
        -1.0e-3, 1.0e-3, _STATE_SIZE, dtype=jnp.float64
    )
    physical_state = (
        scaling.bootstrap_anchor + scaling.variable_scale * optimizer_coordinates
    )

    objective, scaled_constraints = cfs_sqp1_joint_value_constraints(
        optimizer_coordinates,
        cast("FullSpaceProblem", _problem()),
        scaling,
    )

    np.testing.assert_allclose(
        objective, 0.5 * jnp.vdot(physical_state, physical_state)
    )
    np.testing.assert_allclose(
        scaled_constraints,
        scaling.constraint_inverse_scale * (_CONSTRAINT_MATRIX @ physical_state),
    )
    assert objective.dtype == jnp.float64
    assert scaled_constraints.shape == (_EQUALITY_SIZE,)


def test_exact_256_joint_vjp_rows_have_frozen_order_and_scaling() -> None:
    scaling = _scaling()
    optimizer_coordinates = jnp.linspace(
        -1.0e-3, 1.0e-3, _STATE_SIZE, dtype=jnp.float64
    )
    physical_state = (
        scaling.bootstrap_anchor + scaling.variable_scale * optimizer_coordinates
    )

    linearization = cfs_sqp1_joint_linearization(
        optimizer_coordinates,
        cast("FullSpaceProblem", _problem()),
        scaling,
    )
    rows = linearization.joint_vjp_rows

    expected_objective_row = physical_state * scaling.variable_scale
    expected_constraint_rows = (
        scaling.constraint_inverse_scale[:, None]
        * _CONSTRAINT_MATRIX
        * scaling.variable_scale[None, :]
    )
    assert rows.shape == (1 + _EQUALITY_SIZE, _STATE_SIZE)
    assert rows.dtype == jnp.float64
    np.testing.assert_allclose(
        linearization.physical_objective,
        0.5 * jnp.vdot(physical_state, physical_state),
    )
    np.testing.assert_allclose(
        linearization.scaled_constraints,
        scaling.constraint_inverse_scale * (_CONSTRAINT_MATRIX @ physical_state),
    )
    np.testing.assert_array_equal(linearization.objective_gradient, rows[0])
    np.testing.assert_array_equal(linearization.constraint_jacobian, rows[1:])
    np.testing.assert_allclose(rows[0], expected_objective_row, rtol=1.0e-13)
    np.testing.assert_allclose(rows[1:], expected_constraint_rows, rtol=1.0e-13)
    assert linearization.all_finite

    compatibility_rows = cfs_sqp1_joint_vjp_rows(
        optimizer_coordinates,
        cast("FullSpaceProblem", _problem()),
        scaling,
    )
    np.testing.assert_array_equal(compatibility_rows, rows)


def test_scaled_duals_and_final_raw_kkt_are_formed_independently() -> None:
    scaling = _scaling()
    optimizer_coordinates = jnp.zeros(_STATE_SIZE, dtype=jnp.float64)
    scaled_multipliers = jnp.linspace(-0.1, 0.1, _EQUALITY_SIZE, dtype=jnp.float64)
    expected_raw = scaling.constraint_inverse_scale * scaled_multipliers

    raw = cfs_sqp1_raw_multipliers(scaled_multipliers, scaling)
    diagnostics = cfs_sqp1_endpoint_diagnostics(
        optimizer_coordinates,
        scaled_multipliers,
        cast("FullSpaceProblem", _problem()),
        scaling,
    )

    np.testing.assert_array_equal(raw, expected_raw)
    np.testing.assert_allclose(
        diagnostics.raw_stationarity_residual,
        scaling.bootstrap_anchor + _CONSTRAINT_MATRIX.T @ expected_raw,
    )
    assert diagnostics.raw_constraints.shape == (_EQUALITY_SIZE,)
    assert diagnostics.all_finite


@dataclass
class _FakePreparedDenseSQP:
    endpoint: jax.Array
    multipliers: jax.Array
    initial_converged: bool = False

    def run(
        self,
        initial_coordinates: jax.Array,
        initial_multipliers: jax.Array | None = None,
    ) -> DenseSQPResult:
        assert initial_coordinates.shape == (_STATE_SIZE,)
        assert initial_multipliers is not None
        assert initial_multipliers.shape == (_EQUALITY_SIZE,)
        scaling = _scaling()
        physical_state = (
            scaling.bootstrap_anchor + scaling.variable_scale * self.endpoint
        )
        raw_multipliers = scaling.constraint_inverse_scale * self.multipliers
        scaled_constraints = scaling.constraint_inverse_scale * (
            _CONSTRAINT_MATRIX @ physical_state
        )
        raw_stationarity = physical_state + _CONSTRAINT_MATRIX.T @ raw_multipliers
        kkt_solves = 0 if self.initial_converged else 1
        factor_diagnostic = jnp.asarray(
            jnp.nan if self.initial_converged else 0.0,
            dtype=jnp.float64,
        )
        return cast(
            "DenseSQPResult",
            SimpleNamespace(
                optimizer_coordinates=self.endpoint,
                multipliers=self.multipliers,
                bfgs_matrix=jnp.eye(_STATE_SIZE, dtype=jnp.float64),
                objective=0.5 * jnp.vdot(physical_state, physical_state),
                constraints=scaled_constraints,
                objective_gradient=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
                constraint_jacobian=jnp.zeros(
                    (_EQUALITY_SIZE, _STATE_SIZE), dtype=jnp.float64
                ),
                stationarity=scaling.variable_scale * raw_stationarity,
                converged=jnp.asarray(True),
                fatal=jnp.asarray(False),
                failed=jnp.asarray(False),
                status=jnp.asarray(1, dtype=jnp.int32),
                iterations=jnp.asarray(1, dtype=jnp.int32),
                joint_evaluations=jnp.asarray(2, dtype=jnp.int32),
                derivative_builds=jnp.asarray(2, dtype=jnp.int32),
                kkt_solves=jnp.asarray(kkt_solves, dtype=jnp.int32),
                line_search_evaluations=jnp.asarray(1, dtype=jnp.int32),
                rejected_nonfinite_trials=jnp.asarray(0, dtype=jnp.int32),
                bfgs_resets=jnp.asarray(0, dtype=jnp.int32),
                regularization_uses=jnp.asarray(1, dtype=jnp.int32),
                final_kkt_relative_residual=factor_diagnostic,
                final_kkt_reciprocal_condition=factor_diagnostic,
                final_kkt_solution_scaled_residual=factor_diagnostic,
                final_schur_relative_residual=factor_diagnostic,
                selected_regularization=factor_diagnostic,
                regularization_candidates_tested=jnp.asarray(
                    kkt_solves, dtype=jnp.int32
                ),
                merit_penalty=jnp.asarray(1.0, dtype=jnp.float64),
                all_accepted_states_finite=jnp.asarray(True),
                all_finite=jnp.asarray(True),
            ),
        )


def test_prepare_binds_frozen_options_zero_duals_and_conservative_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaling = _scaling()
    captured: dict[str, object] = {}
    endpoint = jnp.zeros(_STATE_SIZE, dtype=jnp.float64)
    multipliers = jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64)

    def fake_prepare(
        joint: object,
        x0: jax.Array,
        *,
        options: object,
    ) -> _FakePreparedDenseSQP:
        captured.update(joint=joint, x0=x0, options=options)
        return _FakePreparedDenseSQP(endpoint=endpoint, multipliers=multipliers)

    monkeypatch.setattr(
        sqp_module,
        "fullspace_scaling_from_bootstrap",
        lambda _bootstrap, _problem: scaling,
    )
    monkeypatch.setattr(sqp_module, "prepare_dense_sqp", fake_prepare)
    bootstrap = scaling.bootstrap_anchor
    prepared = prepare_cfs_sqp1(
        cast("FullSpaceProblem", _problem()),
        bootstrap,
        bootstrap,
    )

    options = cast("DenseSQPOptions", captured["options"])
    expected_tolerance = 1.0e-7 * float(jnp.min(scaling.variable_scale))
    assert prepared.optimizer_stationarity_tolerance == pytest.approx(
        expected_tolerance,
        rel=0.0,
        abs=0.0,
    )
    assert options.stationarity_tolerance == pytest.approx(expected_tolerance)
    assert options.maximum_iterations == 100
    assert options.reverse_row_batch_width == 8
    assert options.initial_bfgs_identity_scale == 1.0
    assert options.schur_relative_residual_tolerance == 1.0e-10
    assert options.kkt_forward_error_tolerance == 1.0e-7
    assert options.kkt_solution_scaled_residual_tolerance == 1.0e-10
    assert options.merit_multiplier_margin == 1.0
    assert options.maximum_identity_retries == 1
    np.testing.assert_array_equal(
        prepared.initial_scaled_multipliers,
        jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
    )

    result = prepared.run()
    assert result.schema_version == "single-stage-fullspace-cfs-sqp1-result-v1"
    assert result.status == 1
    assert result.iterations == 1
    assert result.all_finite
    assert result.solver_result_consistent
    assert result.endpoint.raw_multipliers.shape == (_EQUALITY_SIZE,)
    assert result.optimizer_stationarity_tolerance == expected_tolerance
    assert result.final_kkt_reciprocal_condition == 0.0
    assert result.final_kkt_solution_scaled_residual == 0.0


@pytest.mark.parametrize("maximum_iterations", [1, 10, 100])
def test_prepare_accepts_only_frozen_canary_iteration_budgets(
    monkeypatch: pytest.MonkeyPatch,
    maximum_iterations: int,
) -> None:
    scaling = _scaling()
    captured: dict[str, object] = {}

    def fake_prepare(
        _joint: object,
        _x0: jax.Array,
        *,
        options: object,
    ) -> _FakePreparedDenseSQP:
        captured["options"] = options
        return _FakePreparedDenseSQP(
            endpoint=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
            multipliers=jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
        )

    monkeypatch.setattr(
        sqp_module,
        "fullspace_scaling_from_bootstrap",
        lambda _bootstrap, _problem: scaling,
    )
    monkeypatch.setattr(sqp_module, "prepare_dense_sqp", fake_prepare)

    prepare_cfs_sqp1(
        cast("FullSpaceProblem", _problem()),
        scaling.bootstrap_anchor,
        scaling.bootstrap_anchor,
        maximum_iterations=maximum_iterations,
    )

    options = cast("DenseSQPOptions", captured["options"])
    assert options.maximum_iterations == maximum_iterations


@pytest.mark.parametrize("maximum_iterations", [0, 2, 99, 101, True])
def test_prepare_rejects_nonfrozen_canary_iteration_budgets(
    maximum_iterations: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"maximum_iterations must be one of \{1, 10, 100\}",
    ):
        prepare_cfs_sqp1(
            cast("FullSpaceProblem", _problem()),
            _scaling().bootstrap_anchor,
            _scaling().bootstrap_anchor,
            maximum_iterations=maximum_iterations,
        )


def test_solver_dispatch_and_endpoint_finalization_are_separate() -> None:
    scaling = _scaling()
    optimizer = _FakePreparedDenseSQP(
        endpoint=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        multipliers=jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
    )
    prepared = PreparedCfsSqp1(
        problem=cast("FullSpaceProblem", _problem()),
        scaling=scaling,
        initial_optimizer_coordinates=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        initial_scaled_multipliers=jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
        optimizer_stationarity_tolerance=1.0e-9,
        optimizer=cast("PreparedDenseSQP", optimizer),
    )

    solver_result = prepared.run_solver()
    finalized = prepared.finalize_result(solver_result)

    assert finalized.optimizer is solver_result
    np.testing.assert_array_equal(
        finalized.physical_state,
        scaling.bootstrap_anchor,
    )
    assert finalized.solver_result_consistent

    compatibility_result = prepared.run()
    np.testing.assert_array_equal(
        compatibility_result.optimizer.optimizer_coordinates,
        solver_result.optimizer_coordinates,
    )


def test_unused_factor_diagnostics_do_not_poison_initial_point_finiteness() -> None:
    scaling = _scaling()
    prepared = PreparedCfsSqp1(
        problem=cast("FullSpaceProblem", _problem()),
        scaling=scaling,
        initial_optimizer_coordinates=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        initial_scaled_multipliers=jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
        optimizer_stationarity_tolerance=1.0e-9,
        optimizer=cast(
            "PreparedDenseSQP",
            _FakePreparedDenseSQP(
                endpoint=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
                multipliers=jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
                initial_converged=True,
            ),
        ),
    )

    result = prepared.run()

    assert result.kkt_solves == 0
    assert jnp.isnan(result.final_kkt_relative_residual)
    assert jnp.isnan(result.final_kkt_reciprocal_condition)
    assert jnp.isnan(result.final_kkt_solution_scaled_residual)
    assert result.all_finite
