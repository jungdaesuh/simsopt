"""Strict-transfer coverage for the exact stochastic Stage-II workflow.

The training samples are materialized on the host first, as the shipped mirror
does; the rest of the mirror's JAX lane -- geometry, device staging, the
stochastic objective, the SciPy L-BFGS-B solve over it at the native example's
policy, and the endpoint read-back -- then runs inside
``jax.transfer_guard("disallow")``.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt.field import Coil, Current, coils_via_symmetries
from simsopt.geo import (
    Curve,
    GaussianSampler,
    RotatedCurve,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt_contracts.optimization_endpoint import (
    certify_optimization_endpoint,
    normalized_terminal_status,
)
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import (
    StochasticPerturbationBundle,
    StochasticStageTwoConfiguration,
    materialize_stochastic_coil_perturbations,
    stochastic_stage_two_configuration,
)
from simsopt_jax.examples.stochastic_stage_two import solve_stochastic_stage_two
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    StochasticCoilPerturbations,
    make_stochastic_stage_two_objective,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import TraceableScalarProblem
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

SURFACE_INPUT = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "test_files"
    / "input.LandremanPaul2021_QA"
)


def _geometry(
    configuration: StochasticStageTwoConfiguration,
) -> tuple[SurfaceRZFourier, list[Curve], list[Coil]]:
    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        range="full torus",
        nphi=configuration.surface_nphi,
        ntheta=configuration.surface_ntheta,
    )
    surface.fix_all()
    base_curves = create_equally_spaced_curves(
        configuration.num_base_curves,
        surface.nfp,
        stellsym=True,
        R0=configuration.major_radius,
        R1=configuration.minor_radius,
        order=configuration.curve_order,
        numquadpoints=configuration.curve_quadrature,
    )
    base_currents = [Current(configuration.initial_current) for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    return surface, base_curves, coils


def _training_samples(
    configuration: StochasticStageTwoConfiguration,
) -> StochasticPerturbationBundle:
    """The mirror's host-side sample materialization, symmetry layout included."""
    _surface, base_curves, coils = _geometry(configuration)
    source_indices = tuple(index % len(base_curves) for index in range(len(coils)))
    rotations = []
    for coil, source_index in zip(coils, source_indices, strict=True):
        curve = coil.curve
        if curve is base_curves[source_index]:
            rotations.append(np.eye(3, dtype=np.float64))
        else:
            assert isinstance(curve, RotatedCurve)
            assert curve.curve is base_curves[source_index]
            rotations.append(np.asarray(curve.rotmat, dtype=np.float64))
    return materialize_stochastic_coil_perturbations(
        GaussianSampler(
            base_curves[0].quadpoints,
            sigma=configuration.perturbation_sigma,
            length_scale=configuration.perturbation_length_scale,
            n_derivs=1,
        ),
        source_indices=source_indices,
        rotations=np.stack(rotations),
        base_curve_count=len(base_curves),
        sample_count=configuration.training_sample_count,
        seed=configuration.training_seed,
    )


def test_stochastic_stage_two_has_explicit_device_transfer_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration = stochastic_stage_two_configuration("bounded")
    samples = _training_samples(configuration)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")

    with jax.transfer_guard("disallow"):
        surface, _base_curves, coils = _geometry(configuration)
        field = BiotSavartJAX(coils)
        flux_spec = SquaredFluxJAX(surface, field).fixed_surface_flux_spec()
        device = get_runtime_jax_device()
        objective = make_stochastic_stage_two_objective(
            field,
            flux_spec,
            StochasticCoilPerturbations(
                gamma=jax.device_put(samples.gamma, device),
                gammadash=jax.device_put(samples.gammadash, device),
            ),
            jax.device_put(
                np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)),
                device,
            ),
            jax.device_put(
                np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)),
                device,
            ),
            StageTwoObjectiveConfig(
                num_base_curves=configuration.num_base_curves,
                length_weight=configuration.length_weight,
                curve_curve_minimum_distance=configuration.curve_curve_threshold,
                curve_curve_weight=configuration.curve_curve_weight,
                curvature_threshold=configuration.curvature_threshold,
                curvature_weight=configuration.curvature_weight,
                mean_squared_curvature_threshold=(
                    configuration.mean_squared_curvature_threshold
                ),
                mean_squared_curvature_weight=(
                    configuration.mean_squared_curvature_weight
                ),
                arclength_variation_weight=configuration.arclength_variation_weight,
            ),
        )
        initial_parameters = jax.device_put(
            np.asarray(field.x, dtype=np.float64), device
        )
        problem = TraceableScalarProblem(objective_fn=objective, x=initial_parameters)
        initial_objective, initial_gradient = problem.value_and_grad(initial_parameters)
        optimizer = solve_stochastic_stage_two(
            problem,
            driver=Driver.SCIPY_LBFGSB,
            max_steps=configuration.max_steps,
            maxcor=configuration.lbfgs_history_size,
            tol=configuration.rtol,
        )
        final_objective, final_gradient = problem.value_and_grad(problem.x)
        (
            initial_objective,
            initial_gradient,
            final_parameters,
            final_objective,
            final_gradient,
        ) = jax.device_get(
            jax.block_until_ready(
                (
                    initial_objective,
                    initial_gradient,
                    problem.x,
                    final_objective,
                    final_gradient,
                )
            )
        )

    # The whole lane ran under the guard. Its bounded budget ends on the iteration
    # cap, which the label reports; ``failed`` would mean the run or its scientific
    # predicate broke.
    scientific_predicate = bool(
        np.isfinite(final_objective)
        and final_objective < initial_objective
        and np.all(np.isfinite(final_gradient))
    )
    observation = normalized_terminal_status(
        scientific_predicate=scientific_predicate,
        stage_stopping_reasons=(
            certify_optimization_endpoint(
                status_convention="scipy-lbfgsb",
                provider_success=bool(optimizer.success),
                provider_status=int(optimizer.status),
                iterations=int(optimizer.nit),
                max_iterations=configuration.max_steps,
                initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
                final_gradient_inf_norm=float(np.max(np.abs(final_gradient))),
                parameters_finite=bool(np.all(np.isfinite(final_parameters))),
                observables_finite=bool(np.isfinite(final_objective)),
                inner_success=True,
            ).stopping_reason,
        ),
    )
    assert observation.normalized_status == "budget_exhausted"
    assert observation.success is False
