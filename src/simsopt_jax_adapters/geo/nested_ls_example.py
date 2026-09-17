"""Shared serial NCSX nested-LS example construction and outer transaction.

The native and JAX tutorials use the same NCSX configuration convention, a
resolution-selected surface seed, and the same serial nine-term coil objective.
This module owns the accepted-iterate transaction: successful line-search
probes are staged by their exact float64 coil bytes and become warm starts only
when SciPy reports that exact point as accepted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize
from simsopt._core import load
from simsopt.configs.zoo import get_data
from simsopt.examples import ExampleResult
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier, Volume
from simsopt.geo.boozersurface import BoozerSurface

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NestedLsOuterCandidateStore,
    nested_ls_outer_rejection_barrier,
)
from simsopt_jax_adapters.geo.nested_ls_ncsx import (
    NCSX_EXAMPLE_JSON,
    NcsxNestedLsBranchJump,
    NcsxNestedLsInnerSolveFailed,
    NcsxNestedLsProblem,
    NcsxNestedLsSelfIntersecting,
    clone_surface_xyz_tensor_fourier,
    commit_ncsx_anchor,
    ncsx_native_outer_value_and_grad,
    ncsx_problem_from_jax_boozers,
    ncsx_problem_from_native_boozers,
    ncsx_reduced_schur_outer_value_and_grad,
    remap_tensor_fourier_index,
    restore_ncsx_anchor,
    upsample_surface_xyz_tensor_fourier,
)

ExampleBackend = Literal["native", "jax"]
OuterEvaluator = Callable[
    [NcsxNestedLsProblem, NDArray[np.float64]],
    tuple[float, NDArray[np.float64]],
]
_REJECTION_DISTANCE_SCALE = 1.0


@dataclass(frozen=True, slots=True)
class NestedLsExampleResolution:
    """Fourier and quadrature resolution for one documented example scale."""

    mpol: int
    ntor: int
    nphi: int
    ntheta: int


@dataclass(frozen=True, slots=True)
class NestedLsExampleCandidate:
    """A feasible trial state waiting for an exact SciPy callback match."""

    objective: float
    gradient: NDArray[np.float64]
    coil_dofs: NDArray[np.float64]
    surface_dofs: tuple[NDArray[np.float64], ...]
    iotas: tuple[float, ...]
    g_values: tuple[float, ...]
    inner_success: bool
    inner_reduced_gradient_l2: float | None
    inner_full_penalty_jacobian_l2: float | None


@dataclass(frozen=True, slots=True)
class NestedLsExampleRun:
    """Observable result of one serial moving-coil nested-LS outer run."""

    backend: ExampleBackend
    inner_policy: str
    resolution: NestedLsExampleResolution
    initial_objective: float
    final_objective: float
    final_gradient_l2: float
    outer_iterations: int
    outer_evaluations: int
    rejected_evaluations: int
    final_iota: float
    final_g: float
    inner_success: bool
    inner_reduced_gradient_l2: float | None
    inner_full_penalty_jacobian_l2: float | None
    accepted_coil_delta_l2: float
    accepted_nonzero_coil_movement: bool
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    stopping_reason: str
    budget_exhausted: bool
    runtime_platform: str
    runtime_device: str
    runtime_precision: str
    completed_feasible_step: bool


def nested_ls_example_resolution(smoke: bool) -> NestedLsExampleResolution:
    """Return the bounded 7x7 smoke or documented 48x48 tutorial grid."""

    if smoke:
        return NestedLsExampleResolution(mpol=2, ntor=2, nphi=7, ntheta=7)
    return NestedLsExampleResolution(mpol=6, ntor=6, nphi=48, ntheta=48)


def _resample_surface(
    surface: SurfaceXYZTensorFourier,
    resolution: NestedLsExampleResolution,
) -> SurfaceXYZTensorFourier:
    if resolution.mpol >= surface.mpol and resolution.ntor >= surface.ntor:
        return upsample_surface_xyz_tensor_fourier(
            surface,
            mpol=resolution.mpol,
            ntor=resolution.ntor,
            nphi=resolution.nphi,
            ntheta=resolution.ntheta,
        )

    phi = np.linspace(0.0, 1.0 / surface.nfp, resolution.nphi, endpoint=False)
    theta = np.linspace(0.0, 1.0, resolution.ntheta, endpoint=False)
    truncated = SurfaceXYZTensorFourier(
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        mpol=resolution.mpol,
        ntor=resolution.ntor,
        quadpoints_phi=phi,
        quadpoints_theta=theta,
        clamped_dims=list(surface.clamped_dims),
    )
    for coefficients in (truncated.xcs, truncated.ycs, truncated.zcs):
        coefficients[:, :] = 0.0
    for destination_i in range(2 * resolution.mpol + 1):
        poloidal_mode = (
            destination_i
            if destination_i <= resolution.mpol
            else destination_i - resolution.mpol
        )
        if poloidal_mode > surface.mpol:
            continue
        source_i = remap_tensor_fourier_index(
            destination_i, resolution.mpol, surface.mpol
        )
        for destination_j in range(2 * resolution.ntor + 1):
            toroidal_mode = (
                destination_j
                if destination_j <= resolution.ntor
                else destination_j - resolution.ntor
            )
            if toroidal_mode > surface.ntor:
                continue
            source_j = remap_tensor_fourier_index(
                destination_j, resolution.ntor, surface.ntor
            )
            truncated.xcs[destination_i, destination_j] = surface.xcs[
                source_i, source_j
            ]
            truncated.ycs[destination_i, destination_j] = surface.ycs[
                source_i, source_j
            ]
            truncated.zcs[destination_i, destination_j] = surface.zcs[
                source_i, source_j
            ]
    truncated.local_full_x = truncated.get_dofs()
    truncated.invalidate_cache()
    return truncated


def _load_ncsx_inputs(
    resolution: NestedLsExampleResolution,
) -> tuple[
    object,
    object,
    object,
    SurfaceXYZTensorFourier,
    float,
    float,
    float,
    object,
]:
    if resolution == nested_ls_example_resolution(True):
        base_curves, base_currents, magnetic_axis, nfp, biot_savart = get_data("ncsx")
        base_currents[0].fix_all()
        surface = SurfaceXYZTensorFourier(
            mpol=resolution.mpol,
            ntor=resolution.ntor,
            nfp=nfp,
            stellsym=True,
            quadpoints_phi=np.linspace(0.0, 1.0 / nfp, resolution.nphi, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, resolution.ntheta, endpoint=False),
        )
        surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
        current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
        g_value = 2.0 * np.pi * current_sum * (2.0e-7)
        return (
            base_curves,
            biot_savart.coils,
            [coil.curve for coil in biot_savart.coils],
            surface,
            -0.406,
            g_value,
            1.0,
            base_currents,
        )
    loaded = load(str(NCSX_EXAMPLE_JSON))
    (
        base_curves,
        base_currents,
        coils,
        curves,
        surfaces,
        boozer_surfaces,
        residuals,
    ) = loaded
    base_currents[0].fix_all()
    surface = _resample_surface(surfaces[0], resolution)
    return (
        base_curves,
        coils,
        curves,
        surface,
        float(residuals[0]["iota"]),
        float(residuals[0]["G"]),
        float(boozer_surfaces[0].constraint_weight),
        base_currents,
    )


def _land_native_boozer(
    coils: object,
    surface: SurfaceXYZTensorFourier,
    *,
    iota: float,
    g_value: float,
    constraint_weight: float,
) -> tuple[BoozerSurface, float, float]:
    label = Volume(surface)
    native = BoozerSurface(
        BiotSavart(coils),
        surface,
        label,
        float(label.J()),
        constraint_weight=constraint_weight,
        options={"verbose": False, "weight_inv_modB": True},
    )
    native.need_to_run_code = True
    landed = native.run_code(iota, g_value)
    if landed is None or not bool(landed["success"]):
        raise RuntimeError("NCSX example could not land its initial banana surface.")
    return native, float(landed["iota"]), float(landed["G"])


def build_native_nested_ls_example(
    resolution: NestedLsExampleResolution,
) -> NcsxNestedLsProblem:
    """Build a serial native banana nested-LS outer at one NCSX surface."""

    (
        base_curves,
        coils,
        curves,
        surface,
        iota,
        g_value,
        constraint_weight,
        _base_currents,
    ) = _load_ncsx_inputs(resolution)
    native, landed_iota, landed_g = _land_native_boozer(
        coils,
        surface,
        iota=iota,
        g_value=g_value,
        constraint_weight=constraint_weight,
    )
    return ncsx_problem_from_native_boozers(
        [native],
        iotas=[landed_iota],
        g_values=[landed_g],
        base_curves=base_curves,
        curves=curves,
    )


def build_jax_nested_ls_example(
    resolution: NestedLsExampleResolution,
) -> NcsxNestedLsProblem:
    """Build a serial JAX nested-LS outer from the same native banana land."""

    (
        base_curves,
        coils,
        curves,
        surface,
        iota,
        g_value,
        constraint_weight,
        _base_currents,
    ) = _load_ncsx_inputs(resolution)
    native, landed_iota, landed_g = _land_native_boozer(
        coils,
        surface,
        iota=iota,
        g_value=g_value,
        constraint_weight=constraint_weight,
    )
    jax_surface = clone_surface_xyz_tensor_fourier(native.surface)
    label = Volume(jax_surface)
    jax_boozer = BoozerSurfaceJAX(
        BiotSavartJAX(coils),
        jax_surface,
        label,
        float(native.targetlabel),
        constraint_weight=constraint_weight,
        options={
            "verbose": False,
            "weight_inv_modB": True,
            "optimizer_backend": "ondevice",
        },
    )
    return ncsx_problem_from_jax_boozers(
        [jax_boozer],
        iotas=[landed_iota],
        g_values=[landed_g],
        base_curves=base_curves,
        curves=curves,
        natives=[native],
    )


def _live_surface_dofs(problem: NcsxNestedLsProblem) -> tuple[NDArray[np.float64], ...]:
    surfaces: list[NDArray[np.float64]] = []
    for state in problem.surfaces:
        if state.jax_boozer is not None:
            surface = state.jax_boozer.surface
        elif state.native is not None:
            surface = state.native.surface
        else:
            raise ValueError("nested-LS example surface state has no Boozer surface.")
        surfaces.append(np.array(surface.get_dofs(), dtype=np.float64, copy=True))
    return tuple(surfaces)


def _inner_measurement(
    problem: NcsxNestedLsProblem,
) -> tuple[bool, float | None, float | None]:
    if problem.last_inner is not None:
        runtime = problem.last_run_code
        if runtime is None:
            raise RuntimeError(
                "nested-LS JAX evaluation did not publish full packed stationarity."
            )
        return (
            bool(problem.last_inner.success),
            float(np.linalg.norm(problem.last_inner.reduced_gradient)),
            float(
                np.linalg.norm(
                    np.asarray(runtime["full_packed_stationarity"], dtype=np.float64)
                )
            ),
        )
    inner = problem.last_native_inner
    if inner is None:
        raise RuntimeError(
            "nested-LS native evaluation did not publish full packed stationarity."
        )
    return (
        bool(inner["success"]),
        None,
        float(
            np.linalg.norm(
                np.asarray(inner["full_packed_stationarity"], dtype=np.float64)
            )
        ),
    )


def _snapshot_candidate(
    problem: NcsxNestedLsProblem,
    *,
    objective: float,
    gradient: NDArray[np.float64],
    coil_dofs: NDArray[np.float64],
) -> NestedLsExampleCandidate:
    (
        inner_success,
        inner_reduced_gradient_l2,
        inner_full_penalty_jacobian_l2,
    ) = _inner_measurement(problem)
    return NestedLsExampleCandidate(
        objective=float(objective),
        gradient=np.array(gradient, dtype=np.float64, copy=True),
        coil_dofs=np.array(coil_dofs, dtype=np.float64, copy=True),
        surface_dofs=_live_surface_dofs(problem),
        iotas=tuple(float(state.iota) for state in problem.surfaces),
        g_values=tuple(float(state.G) for state in problem.surfaces),
        inner_success=inner_success,
        inner_reduced_gradient_l2=inner_reduced_gradient_l2,
        inner_full_penalty_jacobian_l2=inner_full_penalty_jacobian_l2,
    )


def _install_candidate(
    problem: NcsxNestedLsProblem,
    candidate: NestedLsExampleCandidate,
) -> None:
    if problem.biotsavart is not None:
        problem.biotsavart.x = np.array(
            candidate.coil_dofs, dtype=np.float64, copy=True
        )
    if problem.native_biotsavart is not None:
        problem.native_biotsavart.x = np.array(
            candidate.coil_dofs,
            dtype=np.float64,
            copy=True,
        )
    for state, dofs, iota, g_value in zip(
        problem.surfaces,
        candidate.surface_dofs,
        candidate.iotas,
        candidate.g_values,
        strict=True,
    ):
        if state.jax_boozer is not None:
            state.jax_boozer.surface.set_dofs(
                np.array(dofs, dtype=np.float64, copy=True)
            )
        if state.native is not None:
            state.native.surface.set_dofs(np.array(dofs, dtype=np.float64, copy=True))
        state.iota = iota
        state.G = g_value
    commit_ncsx_anchor(problem)


def _outer_coil_dofs(problem: NcsxNestedLsProblem) -> NDArray[np.float64]:
    biot_savart = problem.biotsavart or problem.native_biotsavart
    if biot_savart is None:
        raise ValueError("nested-LS example has no outer Biot-Savart state.")
    return np.array(biot_savart.x, dtype=np.float64, copy=True)


def run_nested_ls_example_outer(
    problem: NcsxNestedLsProblem,
    *,
    backend: ExampleBackend,
    inner_policy: str,
    resolution: NestedLsExampleResolution,
    max_steps: int,
    evaluate: OuterEvaluator,
) -> NestedLsExampleRun:
    """Run L-BFGS-B while committing only SciPy's exact accepted candidates."""

    start = _outer_coil_dofs(problem)
    candidates = NestedLsOuterCandidateStore[NestedLsExampleCandidate](start)
    rejected_evaluations = 0
    initial_objective: float | None = None

    def objective(coil_dofs: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        nonlocal initial_objective, rejected_evaluations
        point = np.array(coil_dofs, dtype=np.float64, copy=True)
        try:
            value, gradient = evaluate(problem, point)
        except (
            NcsxNestedLsInnerSolveFailed,
            NcsxNestedLsBranchJump,
            NcsxNestedLsSelfIntersecting,
        ) as error:
            restore_ncsx_anchor(problem)
            if isinstance(
                error, NcsxNestedLsInnerSolveFailed
            ) and error.exit_status in (
                "runtime_install_changed_surface",
                "runtime_install_changed_y",
                "runtime_install_failed",
            ):
                raise
            if not candidates.is_primed or candidates.committed_matches(point):
                raise
            rejected_evaluations += 1
            incumbent = candidates.committed
            return nested_ls_outer_rejection_barrier(
                anchor_value=incumbent.objective,
                anchor_parameters=incumbent.coil_dofs,
                trial_parameters=point,
                scale=_REJECTION_DISTANCE_SCALE,
            )
        candidate = _snapshot_candidate(
            problem,
            objective=value,
            gradient=gradient,
            coil_dofs=point,
        )
        primed = candidates.record(point, candidate)
        if primed:
            initial_objective = float(value)
            _install_candidate(problem, candidate)
        else:
            _install_candidate(problem, candidates.committed)
        return float(value), np.array(gradient, dtype=np.float64, copy=True)

    def accepted(coil_dofs: NDArray[np.float64]) -> None:
        _install_candidate(
            problem, candidates.accept(np.asarray(coil_dofs, dtype=np.float64))
        )

    result: OptimizeResult = minimize(
        objective,
        start,
        method="L-BFGS-B",
        jac=True,
        callback=accepted,
        options={"maxiter": max_steps, "maxcor": min(max_steps, 20), "ftol": 0.0},
    )
    if initial_objective is None or not candidates.is_primed:
        raise RuntimeError("nested-LS example outer did not evaluate its start point.")
    final_candidate = candidates.committed
    _install_candidate(problem, final_candidate)
    accepted_coil_delta_l2 = float(np.linalg.norm(final_candidate.coil_dofs - start))
    accepted_nonzero_coil_movement = accepted_coil_delta_l2 > 0.0
    if backend == "native":
        runtime_platform = "cpu"
        runtime_device = "native-cpu"
        runtime_precision = "fp64"
    else:
        runtime_platform = jax.default_backend()
        runtime_device = str(jax.devices()[0])
        runtime_precision = "fp64" if bool(jax.config.jax_enable_x64) else "fp32"
    optimizer_success = bool(result.success)
    budget_exhausted = bool(
        not optimizer_success
        and int(result.status) == 1
        and int(result.nit) >= max_steps
    )
    stopping_reason = (
        "converged"
        if optimizer_success
        else "iteration_limit"
        if budget_exhausted
        else "solver_terminated"
    )
    completed_feasible_step = bool(
        np.isfinite(initial_objective)
        and np.isfinite(final_candidate.objective)
        and np.isfinite(np.linalg.norm(final_candidate.gradient))
        and np.isfinite(final_candidate.iotas[0])
        and np.isfinite(final_candidate.g_values[0])
        and final_candidate.inner_success
        and (
            final_candidate.inner_reduced_gradient_l2 is not None
            or final_candidate.inner_full_penalty_jacobian_l2 is not None
        )
        and all(
            np.isfinite(measurement)
            for measurement in (
                final_candidate.inner_reduced_gradient_l2,
                final_candidate.inner_full_penalty_jacobian_l2,
            )
            if measurement is not None
        )
        and accepted_nonzero_coil_movement
        and final_candidate.objective < initial_objective
        and (optimizer_success or budget_exhausted)
    )
    return NestedLsExampleRun(
        backend=backend,
        inner_policy=inner_policy,
        resolution=resolution,
        initial_objective=initial_objective,
        final_objective=float(final_candidate.objective),
        final_gradient_l2=float(np.linalg.norm(final_candidate.gradient)),
        outer_iterations=int(result.nit),
        outer_evaluations=int(result.nfev),
        rejected_evaluations=rejected_evaluations,
        final_iota=final_candidate.iotas[0],
        final_g=final_candidate.g_values[0],
        inner_success=final_candidate.inner_success,
        inner_reduced_gradient_l2=final_candidate.inner_reduced_gradient_l2,
        inner_full_penalty_jacobian_l2=final_candidate.inner_full_penalty_jacobian_l2,
        accepted_coil_delta_l2=accepted_coil_delta_l2,
        accepted_nonzero_coil_movement=accepted_nonzero_coil_movement,
        optimizer_success=optimizer_success,
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        stopping_reason=stopping_reason,
        budget_exhausted=budget_exhausted,
        runtime_platform=runtime_platform,
        runtime_device=runtime_device,
        runtime_precision=runtime_precision,
        completed_feasible_step=completed_feasible_step,
    )


def run_native_nested_ls_example(
    *,
    smoke: bool,
    max_steps: int,
) -> NestedLsExampleRun:
    """Run the public native banana BFGS/Newton serial tutorial."""

    resolution = nested_ls_example_resolution(smoke)
    return run_nested_ls_example_outer(
        build_native_nested_ls_example(resolution),
        backend="native",
        inner_policy="banana_bfgs_then_newton",
        resolution=resolution,
        max_steps=max_steps,
        evaluate=ncsx_native_outer_value_and_grad,
    )


def run_jax_nested_ls_example(
    *,
    smoke: bool,
    max_steps: int,
) -> NestedLsExampleRun:
    """Run the JAX reduced-Schur serial tutorial over moving coil DOFs.

    The native twin uses banana BFGS/Newton while this route uses reduced
    Schur Newton. They share NCSX input, the nine-term outer objective, and
    accepted-iterate transaction, but their inner policies are intentionally
    reported as different and are not timing-comparable.
    """

    resolution = nested_ls_example_resolution(smoke)
    return run_nested_ls_example_outer(
        build_jax_nested_ls_example(resolution),
        backend="jax",
        inner_policy="reduced_schur_newton",
        resolution=resolution,
        max_steps=max_steps,
        evaluate=ncsx_reduced_schur_outer_value_and_grad,
    )


def nested_ls_example_result(
    run: NestedLsExampleRun,
    *,
    example_id: str,
) -> ExampleResult:
    """Return common example observables without reimplementing the envelope."""

    return ExampleResult(
        example_id=example_id,
        status="ok" if run.completed_feasible_step else "failed",
        observables={
            "backend": run.backend,
            "runtime_device": run.runtime_device,
            "inner_policy": run.inner_policy,
            "initial_objective": run.initial_objective,
            "final_objective": run.final_objective,
            "final_gradient_l2": run.final_gradient_l2,
            "final_iota": run.final_iota,
            "final_G": run.final_g,
            "inner_success": run.inner_success,
            "inner_reduced_gradient_l2": run.inner_reduced_gradient_l2,
            "inner_full_penalty_jacobian_l2": run.inner_full_penalty_jacobian_l2,
            "outer_iterations": run.outer_iterations,
            "outer_evaluations": run.outer_evaluations,
            "rejected_evaluations": run.rejected_evaluations,
            "accepted_coil_delta_l2": run.accepted_coil_delta_l2,
            "accepted_nonzero_coil_movement": run.accepted_nonzero_coil_movement,
            "completed_feasible_step": run.completed_feasible_step,
            "optimizer_success": run.optimizer_success,
            "optimizer_status": run.optimizer_status,
            "optimizer_message": run.optimizer_message,
            "stopping_reason": run.stopping_reason,
            "budget_exhausted": run.budget_exhausted,
            "mpol": run.resolution.mpol,
            "ntor": run.resolution.ntor,
            "nphi": run.resolution.nphi,
            "ntheta": run.resolution.ntheta,
        },
    )
