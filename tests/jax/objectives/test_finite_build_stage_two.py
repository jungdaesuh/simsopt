"""Pure-JAX finite-build Stage-II objective contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import numpy as np
import pytest
from simsopt._core.optimizable import Optimizable
from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    apply_symmetries_to_currents,
    apply_symmetries_to_curves,
)
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    SurfaceRZFourier,
    create_equally_spaced_curves,
    create_multifilament_grid,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt_jax.core import compute_filament_offsets
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives import (
    FiniteBuildStageTwoConfig,
    finite_build_stage_two_diagnostics,
    make_finite_build_stage_two_objective,
)
from simsopt_jax_adapters.objectives.finite_build_stage_two import (
    FINITE_BUILD_DIAGNOSTIC_FIELDS,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

# Packing layout is owned by ``FINITE_BUILD_DIAGNOSTIC_FIELDS``.
_MINIMUM_CLEARANCE_INDEX = FINITE_BUILD_DIAGNOSTIC_FIELDS.index("minimum_clearance")
_STATE_IDS = ("initial", "perturbed_a", "perturbed_b")


@dataclass(frozen=True, slots=True)
class _FiniteBuildCase:
    """Independent native evaluator, JAX programs, and frozen parameter states."""

    native_objective: Optimizable
    native_distance: CurveCurveDistance
    objective: Callable[[jax.Array], jax.Array]
    diagnostics: Callable[[jax.Array], jax.Array]
    states: dict[str, np.ndarray]


def _apply_state(case: _FiniteBuildCase, state_id: str) -> np.ndarray:
    """Stage one frozen state on the native evaluator and return it."""
    parameters = case.states[state_id]
    case.native_objective.x = parameters
    return parameters


@pytest.fixture(scope="module")
def finite_build_case() -> _FiniteBuildCase:
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=1,
        nfp=1,
        stellsym=True,
        quadpoints_phi=np.linspace(0.0, 0.5, 4, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 4, endpoint=False),
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.25)
    base_curves = create_equally_spaced_curves(
        2,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.55,
        order=2,
        numquadpoints=8,
        use_jax_curve=False,
    )
    base_currents = [Current(5.0e4) for _ in base_curves]
    base_currents[0].fix_all()
    filaments_per_base = 2
    base_filaments = sum(
        (
            create_multifilament_grid(
                curve,
                numfilaments_n=2,
                numfilaments_b=1,
                gapsize_n=0.02,
                gapsize_b=0.04,
                rotation_order=1,
            )
            for curve in base_curves
        ),
        [],
    )
    filament_currents = sum(
        ([current] * filaments_per_base for current in base_currents),
        [],
    )
    curves = apply_symmetries_to_curves(base_curves, surface.nfp, True)
    filament_curves = apply_symmetries_to_curves(
        base_filaments,
        surface.nfp,
        True,
    )
    currents = apply_symmetries_to_currents(
        filament_currents,
        surface.nfp,
        True,
    )
    coils = [
        Coil(curve, current)
        for curve, current in zip(filament_curves, currents, strict=True)
    ]
    initial_lengths = np.asarray(
        [CurveLength(curve).J() for curve in base_curves],
        dtype=np.float64,
    )

    native_field = BiotSavart(coils)
    native_flux = SquaredFlux(surface, native_field)
    native_lengths = [CurveLength(curve) for curve in base_curves]
    native_distance = CurveCurveDistance(curves, 0.1)
    native_objective = (
        native_flux
        + 1.0e-2
        * sum(
            QuadraticPenalty(length, target, "max")
            for length, target in zip(
                native_lengths,
                initial_lengths,
                strict=True,
            )
        )
        + 10.0 * native_distance
    )

    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    config = FiniteBuildStageTwoConfig(
        num_base_curves=2,
        filament_offsets=compute_filament_offsets(
            numfilaments_n=2,
            numfilaments_b=1,
            gapsize_n=0.02,
            gapsize_b=0.04,
        ),
        symmetry_copies=2,
        length_targets=tuple(float(value) for value in initial_lengths),
        length_weight=1.0e-2,
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=10.0,
    )
    flux_spec = flux.fixed_surface_flux_spec()
    objective = make_finite_build_stage_two_objective(field, flux_spec, config)
    diagnostics = finite_build_stage_two_diagnostics(field, flux_spec, config)

    # ``taylor_direction`` convention of the finite-build parity input bundle.
    parameters = np.asarray(field.x, dtype=np.float64)
    direction = np.random.RandomState(1).uniform(size=parameters.shape)
    states = {
        "initial": parameters,
        "perturbed_a": parameters + 1.0e-4 * direction,
        "perturbed_b": parameters - 1.0e-3 * direction,
    }
    return _FiniteBuildCase(
        native_objective=native_objective,
        native_distance=native_distance,
        objective=objective,
        diagnostics=diagnostics,
        states=states,
    )


@pytest.mark.parametrize("state_id", _STATE_IDS)
def test_finite_build_objective_matches_native_value_and_gradient(
    finite_build_case: _FiniteBuildCase,
    state_id: str,
) -> None:
    parameters = _apply_state(finite_build_case, state_id)
    native_value = finite_build_case.native_objective.J()
    native_gradient = np.asarray(
        finite_build_case.native_objective.dJ(),
        dtype=np.float64,
    )
    jax_value, jax_gradient = jax.value_and_grad(finite_build_case.objective)(
        jax.device_put(parameters)
    )

    np.testing.assert_allclose(jax_value, native_value, rtol=2.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(
        jax_gradient,
        native_gradient,
        rtol=2.0e-8,
        atol=2.0e-10,
    )


@pytest.mark.parametrize("state_id", _STATE_IDS)
def test_finite_build_minimum_clearance_matches_native_shortest_distance(
    finite_build_case: _FiniteBuildCase,
    state_id: str,
) -> None:
    parameters = _apply_state(finite_build_case, state_id)
    native_clearance = finite_build_case.native_distance.shortest_distance()
    diagnostics = np.asarray(
        finite_build_case.diagnostics(jax.device_put(parameters)),
        dtype=np.float64,
    )

    assert native_clearance > 0.0
    np.testing.assert_allclose(
        diagnostics[_MINIMUM_CLEARANCE_INDEX],
        native_clearance,
        rtol=1.0e-14,
    )
