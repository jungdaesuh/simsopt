"""Hybrid JAX port of ``examples/3_Advanced/single_stage_optimization.py``.

VMEC and its finite-difference equilibrium derivatives remain on the host/MPI
lane.  Coil-field evaluation, local quadratic flux, engineering penalties, and
the mixed surface/coil derivative execute as one pure-JAX kernel on the
selected CPU or GPU.  The SIMSOPT-owned L-BFGS loop stays on the host, avoiding
a whole-optimization JIT and device-resident optimizer-history growth.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from numbers import Real
from pathlib import Path
from typing import Iterator, cast

import jax
import numpy as np
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import create_equally_spaced_curves
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.util import MpiPartition
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax.geo.optimizer_host_lbfgs import minimize_lbfgs_host_core
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    SurfaceRZFourierDofContract,
    make_dynamic_surface_stage_two_objective,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.mhd.vmec_host import (
    VmecHostEvaluation,
    boundary_sha256,
    hybrid_result_is_scientifically_successful,
    validate_vmec_host_evaluation,
    vmec_result_is_receiptable,
)

EXAMPLE_ID = "native-single-stage-vmec-hybrid"
NATIVE_ITERATIONS = 10
INPUT = (
    Path(__file__).resolve().parents[2]
    / "3_Advanced"
    / "inputs"
    / "input.nfp4_QH_warm_start"
)


@dataclass(frozen=True, slots=True)
class HybridConfiguration:
    """Numerical contract shared by VMEC, JAX, and the evidence receipt."""

    nphi_vmec: int
    ntheta_vmec: int
    coil_order: int
    coil_quadpoints: int
    max_surface_mode: int
    finite_difference_method: str = "forward"
    finite_difference_abs_step: float = 1.0e-7
    finite_difference_rel_step: float = 0.0
    jacobian_failure_threshold: float = 100.0
    aspect_ratio_target: float = 7.0
    aspect_ratio_weight: float = 1.0
    coils_objective_weight: float = 1.0e3

    def fingerprint(self, *, input_sha256: str, mpi_world_size: int) -> str:
        payload = {
            **asdict(self),
            "input_sha256": input_sha256,
            "mpi_world_size": int(mpi_world_size),
            "host_solver": "SIMSOPT_LBFGS",
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _configuration(native_scale: bool) -> HybridConfiguration:
    return HybridConfiguration(
        nphi_vmec=34 if native_scale else 8,
        ntheta_vmec=34 if native_scale else 8,
        coil_order=7 if native_scale else 2,
        coil_quadpoints=128 if native_scale else 16,
        max_surface_mode=1,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _process_working_directory(path: Path) -> Iterator[None]:
    """Own VMEC's process-scoped working directory for this solve."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _stage_two_config() -> StageTwoObjectiveConfig:
    return StageTwoObjectiveConfig(
        num_base_curves=3,
        length_weight=1.0e-8,
        individual_length_target=3.3,
        individual_length_weight=0.1,
        curve_curve_minimum_distance=0.08,
        curve_curve_weight=1.0,
        curvature_threshold=7.0,
        curvature_weight=1.0e-3,
        mean_squared_curvature_threshold=10.0,
        mean_squared_curvature_weight=1.0e-3,
    )


def _host_value_and_gradient(
    compiled_value_and_gradient,
    coil_dofs: np.ndarray,
    surface_dofs: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    coil_device = jax.device_put(np.asarray(coil_dofs, dtype=np.float64))
    surface_device = jax.device_put(np.asarray(surface_dofs, dtype=np.float64))
    value_device, gradients_device = compiled_value_and_gradient(
        coil_device,
        surface_device,
    )
    jax.block_until_ready((value_device, gradients_device))
    coil_gradient_device, surface_gradient_device = gradients_device
    return (
        float(np.asarray(jax.device_get(value_device), dtype=np.float64)),
        np.asarray(jax.device_get(coil_gradient_device), dtype=np.float64),
        np.asarray(jax.device_get(surface_gradient_device), dtype=np.float64),
    )


def solve(output_directory: Path, max_steps: int) -> ExampleResult:
    mpi = MpiPartition()
    native_scale = max_steps >= NATIVE_ITERATIONS
    config = _configuration(native_scale)
    vmec_directory = output_directory / "vmec"
    if mpi.proc0_world:
        vmec_directory.mkdir(parents=True, exist_ok=True)
    mpi.comm_world.Barrier()

    with _process_working_directory(vmec_directory):
        vmec = Vmec(
            str(INPUT),
            mpi=mpi,
            verbose=False,
            nphi=config.nphi_vmec,
            ntheta=config.ntheta_vmec,
            range_surface="half period",
        )
        surface = vmec.boundary
        surface.fix_all()
        surface.fixed_range(
            mmin=0,
            mmax=config.max_surface_mode,
            nmin=-config.max_surface_mode,
            nmax=config.max_surface_mode,
            fixed=False,
        )
        surface.fix("rc(0,0)")

        base_curves = create_equally_spaced_curves(
            3,
            surface.nfp,
            stellsym=True,
            R0=1.0,
            R1=0.6,
            order=config.coil_order,
            numquadpoints=config.coil_quadpoints,
            use_jax_curve=False,
        )
        base_currents = [Current(1.0e5) for _ in base_curves]
        base_currents[0].fix_all()
        coils = coils_via_symmetries(
            base_curves,
            base_currents,
            surface.nfp,
            True,
        )
        field = BiotSavartJAX(coils)
        surface_contract = SurfaceRZFourierDofContract.from_surface(surface)
        stage_two_objective = make_dynamic_surface_stage_two_objective(
            field,
            surface_contract,
            _stage_two_config(),
            definition="local",
        )
        compiled_value_and_gradient = jax.jit(
            jax.value_and_grad(stage_two_objective, argnums=(0, 1))
        )

        quasisymmetry = QuasisymmetryRatioResidual(
            vmec,
            np.linspace(0.0, 1.0, 11),
            helicity_m=1,
            helicity_n=-1,
        )
        equilibrium_problem = LeastSquaresProblem.from_tuples(
            (
                (
                    vmec.aspect,
                    cast(Real, config.aspect_ratio_target),
                    cast(Real, config.aspect_ratio_weight),
                ),
                (
                    quasisymmetry.residuals,
                    cast(Real, 0.0),
                    cast(Real, 1.0),
                ),
            )
        )

        coil_initial = np.asarray(field.x, dtype=np.float64)
        surface_initial = np.asarray(equilibrium_problem.x, dtype=np.float64)
        jax_elapsed_seconds = 0.0

        def coil_value_and_gradient(
            coil_dofs: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            nonlocal jax_elapsed_seconds
            started = time.perf_counter()
            value, coil_gradient, _surface_gradient = _host_value_and_gradient(
                compiled_value_and_gradient,
                coil_dofs,
                surface_initial,
            )
            jax_elapsed_seconds += time.perf_counter() - started
            return value, coil_gradient

        coil_solution = np.array(coil_initial, copy=True)
        if mpi.proc0_world:
            coil_initial_value_and_gradient = coil_value_and_gradient(coil_initial)
            coil_result = minimize_lbfgs_host_core(
                coil_value_and_gradient,
                coil_initial,
                maxiter=NATIVE_ITERATIONS,
                maxcor=min(NATIVE_ITERATIONS, 300),
                ftol=0.0,
                gtol=1.0e-8,
                maxls=20,
                initial_value_and_grad=coil_initial_value_and_gradient,
            )
            coil_solution = np.asarray(coil_result.x_k, dtype=np.float64)
        mpi.comm_world.Bcast(coil_solution, root=0)

        initial = np.concatenate((coil_solution, surface_initial))
        solution = np.array(initial, copy=True)
        vmec_elapsed_seconds = 0.0
        vmec_evaluation_count = 0
        vmec_failed_evaluation_count = 0
        latest_vmec_evaluation: VmecHostEvaluation | None = None
        coil_size = coil_solution.size
        joint_initial_value_and_gradient: tuple[float, np.ndarray] | None = None
        log_file = str(output_directory / "vmec_finite_difference")

        with MPIFiniteDifference(
            equilibrium_problem.objective,
            mpi,
            diff_method=config.finite_difference_method,
            abs_step=config.finite_difference_abs_step,
            rel_step=config.finite_difference_rel_step,
            log_file=log_file,
        ) as equilibrium_jacobian:
            if mpi.proc0_world:

                def joint_value_and_gradient(
                    parameters: np.ndarray,
                ) -> tuple[float, np.ndarray]:
                    nonlocal jax_elapsed_seconds
                    nonlocal latest_vmec_evaluation
                    nonlocal vmec_elapsed_seconds
                    nonlocal vmec_evaluation_count
                    nonlocal vmec_failed_evaluation_count

                    coil_dofs = np.asarray(parameters[:coil_size], dtype=np.float64)
                    surface_dofs = np.asarray(
                        parameters[coil_size:],
                        dtype=np.float64,
                    )
                    equilibrium_problem.x = surface_dofs
                    vmec.need_to_run_code = True
                    iteration_before = int(vmec.iter)
                    wall_started_ns = time.time_ns()
                    vmec_started = time.perf_counter()
                    vmec_objective = float(equilibrium_problem.objective())
                    output_file = Path(vmec.output_file)
                    vmec_evaluation_count += 1
                    if not vmec_result_is_receiptable(
                        objective=vmec_objective,
                        failure_threshold=config.jacobian_failure_threshold,
                        output_file=output_file,
                    ):
                        vmec_elapsed_seconds += time.perf_counter() - vmec_started
                        vmec_failed_evaluation_count += 1
                        return (
                            config.jacobian_failure_threshold,
                            np.zeros_like(parameters),
                        )
                    surface_hash = boundary_sha256(surface_dofs)
                    empty_gradient = np.zeros_like(surface_dofs)
                    evaluation = VmecHostEvaluation.from_output(
                        boundary_dofs=surface_dofs,
                        output_file=output_file,
                        vmec_iteration_before=iteration_before,
                        vmec_iteration_after=int(vmec.iter),
                        mpi_world_size=int(mpi.comm_world.size),
                        objective=vmec_objective,
                        gradient=empty_gradient,
                        elapsed_seconds=time.perf_counter() - vmec_started,
                        evaluation_count=vmec_evaluation_count,
                        started_ns=wall_started_ns,
                    )
                    equilibrium_gradient = np.ravel(
                        equilibrium_jacobian.jac(surface_dofs)
                    )
                    vmec_elapsed_seconds += time.perf_counter() - vmec_started
                    evaluation = replace(
                        evaluation,
                        gradient=np.ascontiguousarray(
                            equilibrium_gradient,
                            dtype=np.float64,
                        ),
                    )
                    validate_vmec_host_evaluation(
                        evaluation,
                        expected_boundary_sha256=surface_hash,
                        expected_mpi_world_size=int(mpi.comm_world.size),
                        expected_gradient_size=surface_dofs.size,
                    )
                    latest_vmec_evaluation = evaluation

                    jax_started = time.perf_counter()
                    (
                        stage_two_value,
                        coil_gradient,
                        mixed_surface_gradient,
                    ) = _host_value_and_gradient(
                        compiled_value_and_gradient,
                        coil_dofs,
                        surface_dofs,
                    )
                    jax_elapsed_seconds += time.perf_counter() - jax_started
                    objective = (
                        vmec_objective + config.coils_objective_weight * stage_two_value
                    )
                    gradient = np.concatenate(
                        (
                            config.coils_objective_weight * coil_gradient,
                            equilibrium_gradient
                            + config.coils_objective_weight * mixed_surface_gradient,
                        )
                    )
                    if (
                        not np.isfinite(objective)
                        or objective > config.jacobian_failure_threshold
                    ):
                        return (
                            config.jacobian_failure_threshold,
                            np.zeros_like(gradient),
                        )
                    return objective, gradient

                joint_initial_value_and_gradient = joint_value_and_gradient(initial)
                joint_result = minimize_lbfgs_host_core(
                    joint_value_and_gradient,
                    initial,
                    maxiter=max_steps,
                    maxcor=min(max_steps, 300),
                    ftol=0.0,
                    gtol=1.0e-8,
                    maxls=20,
                    initial_value_and_grad=joint_initial_value_and_gradient,
                )
                solution = np.asarray(joint_result.x_k, dtype=np.float64)
                final_objective, final_gradient = joint_value_and_gradient(solution)

        mpi.comm_world.Bcast(solution, root=0)
        result: ExampleResult | None = None
        if mpi.proc0_world:
            if (
                latest_vmec_evaluation is None
                or joint_initial_value_and_gradient is None
            ):
                raise RuntimeError("The root process did not produce hybrid evidence.")
            coil_final = solution[:coil_size]
            surface_final = solution[coil_size:]
            jax_started = time.perf_counter()
            (
                final_stage_two_objective,
                _final_coil_gradient,
                final_mixed_surface_gradient,
            ) = _host_value_and_gradient(
                compiled_value_and_gradient,
                coil_final,
                surface_final,
            )
            jax_elapsed_seconds += time.perf_counter() - jax_started
            platform = jax.devices()[0].platform
            failure_threshold_triggered = bool(
                final_objective >= config.jacobian_failure_threshold
            )
            scientific_success = hybrid_result_is_scientifically_successful(
                initial_objective=joint_initial_value_and_gradient[0],
                final_objective=final_objective,
                final_gradient=final_gradient,
                failure_threshold=config.jacobian_failure_threshold,
                expected_boundary_sha256=boundary_sha256(surface_final),
                final_boundary_sha256=latest_vmec_evaluation.boundary_sha256,
            )
            result = ExampleResult(
                example_id=EXAMPLE_ID,
                observables={
                    "execution_scope": (
                        "jax_slice_gpu" if platform == "gpu" else "jax_slice_cpu"
                    ),
                    "vmec_platform": "cpu_mpi",
                    "jax_platform": platform,
                    "vmec_elapsed_seconds": vmec_elapsed_seconds,
                    "jax_elapsed_seconds": jax_elapsed_seconds,
                    "boundary_sha256": latest_vmec_evaluation.boundary_sha256,
                    "vmec_output_sha256": latest_vmec_evaluation.output_sha256,
                    "configuration_sha256": config.fingerprint(
                        input_sha256=_sha256(INPUT),
                        mpi_world_size=int(mpi.comm_world.size),
                    ),
                    "vmec_objective": latest_vmec_evaluation.objective,
                    "stage_two_objective": final_stage_two_objective,
                    "mixed_surface_gradient": tuple(
                        float(value) for value in final_mixed_surface_gradient
                    ),
                    "initial_objective": joint_initial_value_and_gradient[0],
                    "final_objective": final_objective,
                    "failure_threshold_triggered": failure_threshold_triggered,
                    "solution": tuple(float(value) for value in solution),
                    "gradient": tuple(float(value) for value in final_gradient),
                    "vmec_evaluations": vmec_evaluation_count,
                    "vmec_failed_evaluations": vmec_failed_evaluation_count,
                    "solver_iterations": int(joint_result.k),
                    "solver_evaluations": int(joint_result.nfev),
                    "outer_solver_success": bool(joint_result.converged),
                },
                status="ok" if scientific_success else "failed",
            )
        broadcast_result = mpi.comm_world.bcast(result, root=0)
        if not isinstance(broadcast_result, ExampleResult):
            raise TypeError("The root process did not broadcast the hybrid result.")
        return broadcast_result


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-vmec-hybrid-",
        bounded_steps=1,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
