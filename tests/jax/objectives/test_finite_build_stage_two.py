"""Pure-JAX finite-build Stage-II objective contracts."""

from __future__ import annotations

import jax
import numpy as np
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
    make_finite_build_stage_two_objective,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX


def test_finite_build_objective_matches_native_value_and_gradient() -> None:
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
    objective = make_finite_build_stage_two_objective(
        field,
        flux.fixed_surface_flux_spec(),
        config,
    )
    parameters = np.asarray(field.x, dtype=np.float64)
    direction = np.sin(np.arange(parameters.size, dtype=np.float64) + 1.0)
    perturbed = parameters + 1.0e-4 * direction

    native_objective.x = perturbed
    native_value = native_objective.J()
    native_gradient = np.asarray(native_objective.dJ(), dtype=np.float64)
    jax_value, jax_gradient = jax.value_and_grad(objective)(jax.device_put(perturbed))

    np.testing.assert_allclose(jax_value, native_value, rtol=2.0e-11, atol=1.0e-12)
    np.testing.assert_allclose(
        jax_gradient,
        native_gradient,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
