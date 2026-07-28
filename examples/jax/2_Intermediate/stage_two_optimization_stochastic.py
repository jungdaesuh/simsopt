"""JAX port of ``examples/2_Intermediate/stage_two_optimization_stochastic.py``.

The host materializes the native PCG64DXSM systematic and statistical coil
perturbations once, including the original draw order and symmetry action.
The sample tensors are immutable and hash-bound.  Their stochastic flux mean
is accumulated by a rematerialized ``jax.lax.scan`` so optimization remains
JAX-native on CPU or GPU without retaining every sample field in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Coil, Current, coils_via_symmetries
from simsopt.geo import (
    Curve,
    GaussianSampler,
    RotatedCurve,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt_jax.examples import (
    ExampleResult,
    materialize_stochastic_coil_perturbations,
    run_example,
    scalar_example_driver,
)
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    StochasticCoilPerturbations,
    make_stochastic_stage_two_objective,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import (
    TraceableArrayFunction,
    TraceableScalarProblem,
    serial_solve_jax,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

EXAMPLE_ID = "native-stage-two-optimization-stochastic"
NATIVE_ITERATIONS = 400
_STOCHASTIC_LBFGS_HISTORY_SIZE = 10
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


@dataclass(frozen=True, slots=True)
class _SampleIdentity:
    fingerprint: str
    training_fingerprint: str
    out_of_sample_fingerprint: str
    generator: str
    ordering: str
    dtype: str
    byte_order: str
    seeds: tuple[int, int]
    training_shape: tuple[int, ...]
    out_of_sample_shape: tuple[int, ...]


def _symmetry_layout(
    coils: list[Coil],
    base_curves: list[Curve],
) -> tuple[tuple[int, ...], np.ndarray]:
    source_indices = tuple(
        final_index % len(base_curves) for final_index in range(len(coils))
    )
    rotations: list[np.ndarray] = []
    for coil, source_index in zip(coils, source_indices, strict=True):
        curve = coil.curve
        if curve is base_curves[source_index]:
            rotations.append(np.eye(3, dtype=np.float64))
        else:
            if not isinstance(curve, RotatedCurve) or (
                curve.curve is not base_curves[source_index]
            ):
                raise TypeError("symmetry-expanded coils must use RotatedCurve")
            rotations.append(np.asarray(curve.rotmat, dtype=np.float64))
    return source_indices, np.stack(rotations)


def _objective_config() -> StageTwoObjectiveConfig:
    return StageTwoObjectiveConfig(
        num_base_curves=4,
        length_weight=1.0e-6,
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=10.0,
        curvature_threshold=5.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=5.0,
        mean_squared_curvature_weight=1.0e-6,
        arclength_variation_weight=1.0e-2,
    )


def _flux_only_config() -> StageTwoObjectiveConfig:
    return StageTwoObjectiveConfig(num_base_curves=4)


def _build_problem(
    max_steps: int,
) -> tuple[
    BiotSavartJAX,
    SquaredFluxJAX,
    jax.Array,
    jax.Array,
    StochasticCoilPerturbations,
    StochasticCoilPerturbations,
    _SampleIdentity,
]:
    native_scale = max_steps >= NATIVE_ITERATIONS
    surface = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        range="full torus",
        nphi=64 if native_scale else 4,
        ntheta=16 if native_scale else 4,
    )
    base_curves = create_equally_spaced_curves(
        4,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=24 if native_scale else 2,
        numquadpoints=360 if native_scale else 16,
    )
    base_currents = [Current(1.0e5) for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        True,
    )
    source_indices, rotations = _symmetry_layout(coils, base_curves)
    sampler = GaussianSampler(
        base_curves[0].quadpoints,
        sigma=1.0e-3,
        length_scale=0.5,
        n_derivs=1,
    )
    training_bundle = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=source_indices,
        rotations=rotations,
        base_curve_count=len(base_curves),
        sample_count=16 if native_scale else 2,
        seed=0,
    )
    out_of_sample_bundle = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=source_indices,
        rotations=rotations,
        base_curve_count=len(base_curves),
        sample_count=256 if native_scale else 4,
        seed=1,
    )
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3))
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3))
    )
    training = StochasticCoilPerturbations(
        gamma=jax.device_put(training_bundle.gamma),
        gammadash=jax.device_put(training_bundle.gammadash),
    )
    out_of_sample = StochasticCoilPerturbations(
        gamma=jax.device_put(out_of_sample_bundle.gamma),
        gammadash=jax.device_put(out_of_sample_bundle.gammadash),
    )
    identity_digest = hashlib.sha256()
    identity_digest.update(bytes.fromhex(training_bundle.sha256))
    identity_digest.update(bytes.fromhex(out_of_sample_bundle.sha256))
    sample_identity = _SampleIdentity(
        fingerprint=identity_digest.hexdigest(),
        training_fingerprint=training_bundle.sha256,
        out_of_sample_fingerprint=out_of_sample_bundle.sha256,
        generator=training_bundle.generator,
        ordering=training_bundle.ordering,
        dtype=training_bundle.dtype,
        byte_order=training_bundle.byte_order,
        seeds=(training_bundle.seed, out_of_sample_bundle.seed),
        training_shape=training_bundle.gamma.shape,
        out_of_sample_shape=out_of_sample_bundle.gamma.shape,
    )
    return (
        field,
        flux,
        surface_gamma,
        surface_normal,
        training,
        out_of_sample,
        sample_identity,
    )


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    (
        field,
        flux,
        surface_gamma,
        surface_normal,
        training,
        out_of_sample,
        sample_identity,
    ) = _build_problem(max_steps)
    flux_spec = flux.fixed_surface_flux_spec()
    objective = make_stochastic_stage_two_objective(
        field,
        flux_spec,
        training,
        surface_gamma,
        surface_normal,
        _objective_config(),
    )
    training_flux = make_stochastic_stage_two_objective(
        field,
        flux_spec,
        training,
        surface_gamma,
        surface_normal,
        _flux_only_config(),
    )
    out_of_sample_flux = make_stochastic_stage_two_objective(
        field,
        flux_spec,
        out_of_sample,
        surface_gamma,
        surface_normal,
        _flux_only_config(),
    )
    nominal_flux = flux.traceable_objective()
    initial_device = jax.device_put(np.asarray(field.x, dtype=np.float64))
    problem = TraceableScalarProblem(objective_fn=objective, x=initial_device)
    initial_objective_device, initial_gradient_device = problem.value_and_grad(
        initial_device
    )
    driver = scalar_example_driver()
    if driver == Driver.SIMSOPT_LBFGSB:
        result = serial_solve_jax(
            problem,
            driver=driver,
            max_steps=max_steps,
            maxcor=_STOCHASTIC_LBFGS_HISTORY_SIZE,
            rtol=1.0e-15,
            atol=1.0e-8,
            require_success=False,
        )
    else:
        result = serial_solve_jax(
            problem,
            driver=driver,
            max_steps=max_steps,
            line_search_max_steps=40,
            rtol=1.0e-15,
            atol=1.0e-8,
            require_success=False,
        )
    solution_device = problem.x
    final_objective_device, final_gradient_device = problem.value_and_grad(
        solution_device
    )
    diagnostic = TraceableArrayFunction(
        function_fn=lambda parameters: jnp.stack(
            (
                training_flux(parameters),
                nominal_flux(parameters),
                out_of_sample_flux(parameters),
            )
        ),
        x=solution_device,
    )
    flux_diagnostics_device = diagnostic(solution_device)
    (
        initial,
        initial_gradient,
        solution,
        final_gradient,
        objective_values,
    ) = jax.device_get(
        jax.block_until_ready(
            (
                initial_device,
                initial_gradient_device,
                solution_device,
                final_gradient_device,
                jnp.concatenate(
                    (
                        jnp.stack(
                            (
                                initial_objective_device,
                                final_objective_device,
                            )
                        ),
                        flux_diagnostics_device,
                    )
                ),
            )
        )
    )
    initial = np.asarray(initial, dtype=np.float64)
    initial_gradient = np.asarray(initial_gradient, dtype=np.float64)
    solution = np.asarray(solution, dtype=np.float64)
    final_gradient = np.asarray(final_gradient, dtype=np.float64)
    objective_values = np.asarray(
        objective_values,
        dtype=np.float64,
    )
    (
        initial_objective,
        final_objective,
        training_objective,
        nominal_objective,
        out_of_sample_objective,
    ) = (float(value) for value in objective_values)
    solver_accepted = result.status in (0, 1)
    scientific_success = bool(
        solver_accepted
        and np.all(np.isfinite(solution))
        and np.all(np.isfinite(final_gradient))
        and np.all(np.isfinite(objective_values))
        and final_objective < initial_objective
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "sample_fingerprint": sample_identity.fingerprint,
            "training_sample_fingerprint": sample_identity.training_fingerprint,
            "out_of_sample_fingerprint": sample_identity.out_of_sample_fingerprint,
            "sample_generator": sample_identity.generator,
            "sample_ordering": sample_identity.ordering,
            "sample_dtype": sample_identity.dtype,
            "sample_byte_order": sample_identity.byte_order,
            "sample_seeds": sample_identity.seeds,
            "training_sample_shape": sample_identity.training_shape,
            "out_of_sample_shape": sample_identity.out_of_sample_shape,
            "sample_metadata": {
                "fingerprint": sample_identity.training_fingerprint,
                "generator": sample_identity.generator,
                "ordering": sample_identity.ordering,
                "dtype": sample_identity.dtype,
                "byte_order": sample_identity.byte_order,
                "seed": sample_identity.seeds[0],
                "shape": sample_identity.training_shape,
            },
            "out_of_sample_metadata": {
                "fingerprint": sample_identity.out_of_sample_fingerprint,
                "generator": sample_identity.generator,
                "ordering": sample_identity.ordering,
                "dtype": sample_identity.dtype,
                "byte_order": sample_identity.byte_order,
                "seed": sample_identity.seeds[1],
                "shape": sample_identity.out_of_sample_shape,
            },
            "initial_parameters": tuple(float(value) for value in initial),
            "initial_objective": initial_objective,
            "initial_gradient": tuple(float(value) for value in initial_gradient),
            "solution": tuple(float(value) for value in solution),
            "final_objective": final_objective,
            "final_gradient": tuple(float(value) for value in final_gradient),
            "training_flux_objective": training_objective,
            "nominal_flux_objective": nominal_objective,
            "out_of_sample_objective": out_of_sample_objective,
            "solver_success": bool(result.success),
            "solver_status": result.status,
            "solver_iterations": result.nit,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-stage-two-stochastic-",
        bounded_steps=20,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
