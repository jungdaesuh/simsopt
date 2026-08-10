from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax.solve.fullspace_filter_trust_region as ftr_module
import simsopt_jax.solve.fullspace_sqp as sqp_module
from simsopt_jax.geo.optimizers.filter_trust_region_sqp import (
    FilterTrustRegionSQPOptions,
    FilterTrustRegionSQPResult,
    PreparedFilterTrustRegionSQP,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceConstraints,
    FullSpaceEvaluation,
    FullSpaceKKTPrimitives,
    FullSpaceObservables,
    FullSpaceProblem,
    FullSpaceRawTerms,
)
from simsopt_jax.solve.fullspace import FullSpaceRoute, FullSpaceScaling
from simsopt_jax.solve.fullspace_filter_trust_region import (
    PreparedCfsFtr1,
    prepare_cfs_ftr1,
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
        bootstrap_anchor=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        variable_scale=jnp.linspace(0.01, 2.0, _STATE_SIZE, dtype=jnp.float64),
        constraint_inverse_scale=jnp.linspace(
            0.5,
            1.5,
            _EQUALITY_SIZE,
            dtype=jnp.float64,
        ),
    )


def _optimizer_result(
    coordinates: jax.Array,
    multipliers: jax.Array,
    *,
    normal_relative_residual: float = 0.0,
    normal_forward_error_bound: float = 0.0,
    multiplier_forward_error_bound: float = 0.0,
    accepted_iterations: int = 10,
) -> FilterTrustRegionSQPResult:
    scaling = _scaling()
    physical_state = scaling.variable_scale * coordinates
    raw_multipliers = scaling.constraint_inverse_scale * multipliers
    constraints = scaling.constraint_inverse_scale * (
        _CONSTRAINT_MATRIX @ physical_state
    )
    stationarity = scaling.variable_scale * (
        physical_state + _CONSTRAINT_MATRIX.T @ raw_multipliers
    )
    return cast(
        "FilterTrustRegionSQPResult",
        SimpleNamespace(
            optimizer_coordinates=coordinates,
            multipliers=multipliers,
            bfgs_matrix=jnp.eye(_STATE_SIZE, dtype=jnp.float64),
            objective=0.5 * jnp.vdot(physical_state, physical_state),
            constraints=constraints,
            objective_gradient=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
            constraint_jacobian=jnp.zeros(
                (_EQUALITY_SIZE, _STATE_SIZE),
                dtype=jnp.float64,
            ),
            stationarity=stationarity,
            radius=jnp.asarray(1.0, dtype=jnp.float64),
            status=jnp.asarray(1, dtype=jnp.int32),
            iterations=jnp.asarray(10, dtype=jnp.int32),
            accepted_iterations=jnp.asarray(accepted_iterations, dtype=jnp.int32),
            joint_evaluations=jnp.asarray(21, dtype=jnp.int32),
            derivative_builds=jnp.asarray(11, dtype=jnp.int32),
            final_normal_relative_residual=jnp.asarray(
                normal_relative_residual,
                dtype=jnp.float64,
            ),
            final_normal_forward_error_bound=jnp.asarray(
                normal_forward_error_bound,
                dtype=jnp.float64,
            ),
            final_tangency_relative_residual=jnp.asarray(0.0, dtype=jnp.float64),
            final_multiplier_projection_relative_residual=jnp.asarray(
                0.0,
                dtype=jnp.float64,
            ),
            final_multiplier_projection_forward_error_bound=jnp.asarray(
                multiplier_forward_error_bound,
                dtype=jnp.float64,
            ),
            converged=jnp.asarray(True),
            failed=jnp.asarray(False),
            fatal=jnp.asarray(False),
            all_accepted_states_finite=jnp.asarray(True),
            all_finite=jnp.asarray(True),
        ),
    )


@dataclass
class _FakePreparedFtr:
    result: FilterTrustRegionSQPResult

    def run(
        self,
        initial_coordinates: jax.Array,
        initial_multipliers: jax.Array | None = None,
    ) -> FilterTrustRegionSQPResult:
        assert initial_coordinates.shape == (_STATE_SIZE,)
        assert initial_multipliers is not None
        assert initial_multipliers.shape == (_EQUALITY_SIZE,)
        return self.result


def test_prepare_reuses_frozen_fullspace_physics_and_binds_ftr_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaling = _scaling()
    captured: dict[str, object] = {}
    coordinates = jnp.zeros(_STATE_SIZE, dtype=jnp.float64)
    optimizer_result = _optimizer_result(
        coordinates,
        jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
    )

    def fake_prepare(
        joint: object,
        x0: jax.Array,
        *,
        options: FilterTrustRegionSQPOptions,
    ) -> _FakePreparedFtr:
        captured.update(joint=joint, x0=x0, options=options)
        return _FakePreparedFtr(optimizer_result)

    monkeypatch.setattr(
        ftr_module,
        "fullspace_scaling_from_bootstrap",
        lambda _bootstrap, _problem: scaling,
    )
    monkeypatch.setattr(ftr_module, "prepare_filter_trust_region_sqp", fake_prepare)

    prepared = prepare_cfs_ftr1(
        cast("FullSpaceProblem", _problem()),
        scaling.bootstrap_anchor,
        scaling.bootstrap_anchor,
        maximum_iterations=10,
    )

    options = cast("FilterTrustRegionSQPOptions", captured["options"])
    assert options.maximum_iterations == 10
    assert options.maximum_joint_evaluations == 1200
    assert options.reverse_row_batch_width == 8
    assert options.initial_bfgs_identity_scale == 1.0
    assert options.curvature_fraction == 0.2
    assert options.gram_relative_residual_tolerance == 1.0e-10
    assert options.multiplier_projection_relative_residual_tolerance == 1.0e-10
    assert options.linear_solve_forward_error_tolerance == 1.0e-7
    assert options.acceptance_ratio == 0.1
    assert options.radius_shrink_ratio == 0.25
    assert prepared.optimizer_stationarity_tolerance == pytest.approx(1.0e-9)
    np.testing.assert_array_equal(
        prepared.initial_scaled_multipliers,
        jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
    )

    joint = cast(
        "Callable[[jax.Array], tuple[jax.Array, jax.Array]]",
        captured["joint"],
    )
    objective, constraints = joint(coordinates)
    assert objective == 0.0
    np.testing.assert_array_equal(
        constraints,
        jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
    )

    result = prepared.run()
    assert result.schema_version == "single-stage-fullspace-cfs-ftr1-result-v1"
    assert result.route is FullSpaceRoute.CFS_FTR1
    assert result.solver_result_consistent
    assert result.converged


@pytest.mark.parametrize("maximum_iterations", [0, 1, 11, 99, 101, True])
def test_prepare_rejects_nonfrozen_iteration_budgets(
    maximum_iterations: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"maximum_iterations must be one of \{10, 100\}",
    ):
        prepare_cfs_ftr1(
            cast("FullSpaceProblem", _problem()),
            _scaling().bootstrap_anchor,
            _scaling().bootstrap_anchor,
            maximum_iterations=maximum_iterations,
        )


def test_solver_dispatch_and_raw_endpoint_finalization_are_separate() -> None:
    scaling = _scaling()
    optimizer_result = _optimizer_result(
        jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
    )
    prepared = PreparedCfsFtr1(
        problem=cast("FullSpaceProblem", _problem()),
        scaling=scaling,
        initial_optimizer_coordinates=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        initial_scaled_multipliers=jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
        optimizer_stationarity_tolerance=1.0e-9,
        optimizer=cast(
            "PreparedFilterTrustRegionSQP",
            _FakePreparedFtr(optimizer_result),
        ),
    )

    solver_result = prepared.run_solver()
    finalized = prepared.finalize_result(solver_result)

    assert finalized.optimizer is solver_result
    np.testing.assert_array_equal(finalized.physical_state, scaling.bootstrap_anchor)
    np.testing.assert_array_equal(
        finalized.raw_multipliers,
        jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
    )
    assert finalized.solver_result_consistent


def test_independent_raw_kkt_rejects_false_generic_convergence() -> None:
    scaling = _scaling()
    coordinates = jnp.zeros(_STATE_SIZE, dtype=jnp.float64).at[-1].set(1.0)
    optimizer_result = _optimizer_result(
        coordinates,
        jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
    )
    prepared = PreparedCfsFtr1(
        problem=cast("FullSpaceProblem", _problem()),
        scaling=scaling,
        initial_optimizer_coordinates=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        initial_scaled_multipliers=jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
        optimizer_stationarity_tolerance=1.0e-9,
        optimizer=cast(
            "PreparedFilterTrustRegionSQP",
            _FakePreparedFtr(optimizer_result),
        ),
    )

    result = prepared.run()

    assert result.optimizer.converged
    assert result.raw_kkt_stationarity_infinity_norm > 1.0e-7
    assert not result.converged


def test_excessive_retained_solve_residual_rejects_generic_convergence() -> None:
    scaling = _scaling()
    optimizer_result = _optimizer_result(
        jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
        normal_relative_residual=2.0e-10,
    )
    prepared = PreparedCfsFtr1(
        problem=cast("FullSpaceProblem", _problem()),
        scaling=scaling,
        initial_optimizer_coordinates=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        initial_scaled_multipliers=jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
        optimizer_stationarity_tolerance=1.0e-9,
        optimizer=cast(
            "PreparedFilterTrustRegionSQP",
            _FakePreparedFtr(optimizer_result),
        ),
    )

    result = prepared.run()

    assert result.optimizer.converged
    assert result.all_finite
    assert not result.solve_certificates_valid
    assert not result.converged


@pytest.mark.parametrize(
    ("normal_bound", "multiplier_bound"),
    ((2.0e-7, 0.0), (0.0, 2.0e-7)),
)
def test_excessive_forward_error_bound_rejects_generic_convergence(
    normal_bound: float,
    multiplier_bound: float,
) -> None:
    scaling = _scaling()
    optimizer_result = _optimizer_result(
        jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
        normal_forward_error_bound=normal_bound,
        multiplier_forward_error_bound=multiplier_bound,
    )
    prepared = PreparedCfsFtr1(
        problem=cast("FullSpaceProblem", _problem()),
        scaling=scaling,
        initial_optimizer_coordinates=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        initial_scaled_multipliers=jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
        optimizer_stationarity_tolerance=1.0e-9,
        optimizer=cast(
            "PreparedFilterTrustRegionSQP",
            _FakePreparedFtr(optimizer_result),
        ),
    )

    result = prepared.run()

    assert result.optimizer.converged
    assert result.all_finite
    assert not result.solve_certificates_valid
    assert not result.converged


def test_zero_step_nonfinite_certificate_cannot_bypass_finalization() -> None:
    scaling = _scaling()
    optimizer_result = _optimizer_result(
        jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
        normal_forward_error_bound=float("nan"),
        accepted_iterations=0,
    )
    prepared = PreparedCfsFtr1(
        problem=cast("FullSpaceProblem", _problem()),
        scaling=scaling,
        initial_optimizer_coordinates=jnp.zeros(_STATE_SIZE, dtype=jnp.float64),
        initial_scaled_multipliers=jnp.zeros(_EQUALITY_SIZE, dtype=jnp.float64),
        optimizer_stationarity_tolerance=1.0e-9,
        optimizer=cast(
            "PreparedFilterTrustRegionSQP",
            _FakePreparedFtr(optimizer_result),
        ),
    )

    result = prepared.run()

    assert not result.solve_certificates_valid
    assert not result.all_finite
    assert not result.converged
