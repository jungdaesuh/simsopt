"""Small, deterministic quasi-Newton fixtures.

The fixtures deliberately contain no mutable input files, VMEC state, or
randomness.  ``coil47`` and ``boozer`` are source-owned VMEC-free physics
fixtures with both JAX and SIMSOPT-native objective callbacks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np

ArrayObjective = Callable[[jax.Array], jax.Array]
ValueAndGrad = Callable[[jax.Array], tuple[jax.Array, jax.Array]]
NativeValueAndGrad = Callable[[np.ndarray], tuple[float, np.ndarray]]
NativeReset = Callable[[], None]
MetadataValue: TypeAlias = (
    str | int | float | bool | tuple[int, ...] | tuple[float, ...]
)
FixtureMetadata: TypeAlias = tuple[tuple[str, MetadataValue], ...]


@dataclass(frozen=True)
class ScientificEndpointEvidence:
    """Fixture-owned scientific evidence evaluated outside timed solver work."""

    inner_success: bool
    observables: FixtureMetadata


ScientificEndpoint = Callable[[np.ndarray], ScientificEndpointEvidence]


class _TraceableEndpointSolver(Protocol):
    """Full inner solve used to certify one endpoint outside timed work."""

    def __call__(
        self,
        coil_source: object,
        sdofs: jax.Array,
        iota: jax.Array,
        G: jax.Array,
        *,
        materialize_dense_linearization: bool,
    ) -> Mapping[str, object]: ...


def _certified_traceable_endpoint(
    x: np.ndarray,
    *,
    coil_set_spec_from_dofs: Callable[[jax.Array], object],
    run_code_traceable: _TraceableEndpointSolver,
    install_traceable_solved_runtime_state: Callable[[Mapping[str, object]], object],
    surface_dofs: np.ndarray,
    iota_seed: float,
    G_seed: float,
    reporting_metrics_from_solution: Callable[..., Mapping[str, jax.Array]],
) -> ScientificEndpointEvidence:
    """Certify endpoint observables from a fresh solve at the endpoint coils."""

    coil_dofs = jnp.asarray(x, dtype=jnp.float64)
    solved = run_code_traceable(
        coil_set_spec_from_dofs(coil_dofs),
        jax.device_put(np.asarray(surface_dofs, dtype=np.float64)),
        jax.device_put(np.asarray(iota_seed, dtype=np.float64)),
        jax.device_put(np.asarray(G_seed, dtype=np.float64)),
        materialize_dense_linearization=False,
    )
    jax.block_until_ready(solved)
    inner_success = bool(np.asarray(jax.device_get(solved["success"])))
    if not inner_success:
        return ScientificEndpointEvidence(inner_success=False, observables=())

    install_traceable_solved_runtime_state(solved)
    solved_x = cast(jax.Array, solved["x"])
    solver_success = cast(jax.Array, solved["success"])
    metrics = reporting_metrics_from_solution(
        coil_dofs,
        solved_x,
        solver_success,
        include_distance_metrics=False,
    )
    jax.block_until_ready(metrics)
    return ScientificEndpointEvidence(
        inner_success=True,
        observables=tuple(
            (key, float(np.asarray(jax.device_get(metrics[key]))))
            for key in (
                "final_boozer_residual",
                "final_non_qs",
                "final_iota",
                "final_volume",
            )
        ),
    )


@dataclass(frozen=True)
class ObjectiveTrialEvaluation:
    """Objective-owned evidence for one opt-in diagnostic evaluation."""

    raw_objective: float | None
    raw_objective_certified: bool
    filtered_objective: float | None
    gradient: np.ndarray
    gradient_source: Literal["candidate", "baseline", "unavailable"]
    predictor_kind: str | None
    predictor_success: bool | None
    primal_success: bool
    adjoint_success: bool | None
    newton_success: bool
    newton_stop_reason_code: int | None
    newton_accepted_iterations: int | None
    newton_attempted_iterations: int | None
    newton_last_linear_solve_success: bool | None
    inner_penalty_residual_l2: float | None
    inner_final_gradient_inf_norm: float | None


ObjectiveTrialEvaluator = Callable[[np.ndarray], ObjectiveTrialEvaluation]


class AcceptedIncumbentHostValueAndGrad(Protocol):
    """Host objective controller whose state advances only after acceptance."""

    def value_and_grad(self, parameters: np.ndarray) -> tuple[float, np.ndarray]: ...

    def accept(self, parameters: np.ndarray) -> None: ...


AcceptedIncumbentHostValueAndGradFactory = Callable[
    [], AcceptedIncumbentHostValueAndGrad
]


@dataclass(frozen=True)
class Fixture:
    """A reproducible scalar objective and its initial point.

    ``native_reset`` restores mutable native-provider state before each
    independent timed run; a provider without mutable state may leave it unset.
    """

    name: str
    objective: ArrayObjective
    initial: np.ndarray
    expected_dimension: int
    source: str
    certificate: str
    method: Literal["bfgs", "lbfgs"]
    value_and_grad: ValueAndGrad | None = None
    native_value_and_grad: NativeValueAndGrad | None = None
    metadata: FixtureMetadata = ()
    native_reset: NativeReset | None = None
    scientific_endpoint: ScientificEndpoint | None = None
    native_scientific_endpoint: ScientificEndpoint | None = None
    trial_evaluator: ObjectiveTrialEvaluator | None = None
    native_trial_evaluator: ObjectiveTrialEvaluator | None = None
    accepted_incumbent_host_value_and_grad: (
        AcceptedIncumbentHostValueAndGradFactory | None
    ) = None


@dataclass(frozen=True)
class FixtureDefinition:
    builder: Callable[[], Fixture]
    method: Literal["bfgs", "lbfgs"]


def quadratic47() -> Fixture:
    """Return a 47-variable strictly convex quadratic contract fixture."""

    dimension = 47
    center = jnp.linspace(-0.35, 0.35, dimension, dtype=jnp.float64)
    curvature = jnp.linspace(1.0, 3.0, dimension, dtype=jnp.float64)

    def objective(x: jax.Array) -> jax.Array:
        delta = x - center
        return 0.5 * jnp.sum(curvature * delta * delta)

    initial = np.linspace(0.75, -0.55, dimension, dtype=np.float64)
    return Fixture(
        name="quadratic47",
        objective=objective,
        initial=initial,
        expected_dimension=dimension,
        source="synthetic_contract_quadratic",
        certificate="solver-runtime-only; not a coil-physics parity certificate",
        method="lbfgs",
    )


def coil47_physics() -> Fixture:
    """Return the bounded VMEC-free coil preoptimization contract.

    The curve/current construction follows the single-stage example.  A fixed
    analytic surface keeps this fixture independent of VMEC, MPI, and mutable
    input files while retaining the full 47-DOF coil-to-flux JAX objective.
    """

    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
    from simsopt.objectives import SquaredFlux
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    nfp = 4
    surface_quadpoints = 8
    curve_count = 3
    curve_order = 2
    curve_quadpoints = 16
    current_value = 1.0e5
    quadrature = np.linspace(
        0.0,
        1.0,
        surface_quadpoints,
        endpoint=False,
        dtype=np.float64,
    )
    surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=True,
        mpol=1,
        ntor=1,
        quadpoints_phi=quadrature,
        quadpoints_theta=quadrature,
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.3)
    surface.set_zs(1, 0, 0.3)
    surface.fix_all()
    target = np.linspace(
        0.1,
        0.2,
        surface_quadpoints * surface_quadpoints,
        dtype=np.float64,
    ).reshape((surface_quadpoints, surface_quadpoints))
    curves = create_equally_spaced_curves(
        curve_count,
        nfp,
        stellsym=True,
        R0=1.0,
        R1=0.6,
        order=curve_order,
        numquadpoints=curve_quadpoints,
        use_jax_curve=False,
    )
    currents = [Current(1.0) * current_value for _ in curves]
    currents[0].fix_all()
    coils = coils_via_symmetries(curves, currents, nfp, True)
    native_field = BiotSavart(coils)
    native_flux = SquaredFlux(surface, native_field, target=target)
    field = BiotSavartJAX(coils)
    objective = SquaredFluxJAX(surface, field, target=target).traceable_objective()
    initial = np.asarray(field.x, dtype=np.float64)
    if initial.size != 47:
        raise RuntimeError(f"coil47 fixture expected 47 free DOFs, got {initial.size}")

    def native_value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        native_flux.x = np.asarray(x, dtype=np.float64)
        return float(native_flux.J()), np.asarray(native_flux.dJ(), dtype=np.float64)

    metadata: FixtureMetadata = (
        ("source_example", "examples/jax/3_Advanced/single_stage_optimization.py"),
        ("surface_kind", "analytic_surface_rz_fourier"),
        ("surface_nfp", nfp),
        ("surface_mpol", 1),
        ("surface_ntor", 1),
        ("surface_quadpoints", surface_quadpoints),
        ("curve_count", curve_count),
        ("curve_order", curve_order),
        ("curve_quadpoints", curve_quadpoints),
        ("current_value", current_value),
        ("fixed_current_index", 0),
        ("objective_definition", "quadratic flux"),
        (
            "native_objective_provider",
            "simsopt.field.BiotSavart + simsopt.objectives.SquaredFlux",
        ),
        ("target_min", 0.1),
        ("target_max", 0.2),
        ("dtype", "float64"),
        ("seed", 0),
    )
    return Fixture(
        name="coil47",
        objective=objective,
        initial=initial,
        expected_dimension=47,
        source="source_owned_fixed_surface_coil_flux",
        certificate=(
            "source-owned VMEC-free coil physics; native/JAX objective parity"
        ),
        method="lbfgs",
        native_value_and_grad=native_value_and_grad,
        metadata=metadata,
    )


def boozer_physics() -> Fixture:
    """Return the bounded VMEC-free Boozer vacuum objective contract."""

    from simsopt.configs import get_data
    from simsopt.field import BiotSavart
    from simsopt.geo import (
        BoozerResidual,
        BoozerSurface,
        CurveLength,
        Iotas,
        MajorRadius,
        NonQuasiSymmetricRatio,
        SurfaceXYZTensorFourier,
        Volume,
    )
    from simsopt.objectives import QuadraticPenalty
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
    from simsopt_jax_adapters.geo.surface_objectives_traceable import (
        TraceableObjectiveTrialResult,
        make_traceable_objective_runtime_bundle,
        make_traceable_objective_session,
    )

    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data(
        "ncsx",
        coil_order=3,
        magnetic_axis_order=3,
        points_per_period=8,
    )
    base_currents[0].fix_all()
    field = BiotSavartJAX(native_field.coils)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    g0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    resolution = 1
    surface = SurfaceXYZTensorFourier(
        mpol=resolution,
        ntor=resolution,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(
            0.0,
            1.0 / nfp,
            2 * resolution + 1,
            endpoint=False,
        ),
        quadpoints_theta=np.linspace(
            0.0,
            1.0,
            2 * resolution + 1,
            endpoint=False,
        ),
    )
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
    volume_label = Volume(surface)
    volume_target = float(volume_label.J())

    boozer_surface = BoozerSurfaceJAX(
        field,
        surface,
        volume_label,
        volume_target,
        options={
            "optimizer_backend": "ondevice",
            "newton_maxiter": 20,
            "newton_tol": 1.0e-13,
            "verbose": False,
        },
    )
    surface_dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
    traceable_result = boozer_surface.run_code_traceable(
        field.coil_set_spec(),
        jax.device_put(surface_dofs),
        jax.device_put(np.asarray(-0.406, dtype=np.float64)),
        jax.device_put(np.asarray(g0, dtype=np.float64)),
    )
    jax.block_until_ready(traceable_result)
    if not bool(np.asarray(jax.device_get(traceable_result["success"]))):
        raise RuntimeError("Boozer traceable fixture route did not converge")
    boozer_surface.install_traceable_solved_runtime_state(traceable_result)
    iota_target = float(np.asarray(jax.device_get(traceable_result["iota"])))
    objective_config: dict[str, object] = {
        "non_qs_weight": 1.0,
        "residual_weight": 1.0,
        "iota_weight": 1.0,
        "major_radius_weight": 1.0,
        "length_weight": 1.0,
        "curvature_weight": 0.0,
        "curve_curve_weight": 0.0,
        "curve_surface_weight": 0.0,
        "surface_vessel_weight": 0.0,
        "non_qs_quadpoints_phi": np.linspace(
            0.0,
            1.0 / nfp,
            8,
            endpoint=False,
        ),
        "non_qs_quadpoints_theta": np.linspace(0.0, 1.0, 8, endpoint=False),
        "non_qs_axis": 0,
        "optimized_coil_index": 0,
        "length_coil_indices": (0, 1, 2),
        "length_target": float(sum(CurveLength(curve).J() for curve in base_curves)),
        "curvature_threshold": 0.0,
        "curvature_p_norm": 2.0,
        "major_radius_target": float(surface.major_radius()),
        "curve_curve_threshold": 0.0,
        "curve_surface_threshold": 0.0,
        "vessel_gamma": np.asarray(surface.gamma(), dtype=np.float64),
        "surface_vessel_threshold": 0.0,
    }
    session = make_traceable_objective_session(
        boozer_surface,
        field,
        iota_target,
        outer_objective_config=objective_config,
    )
    runtime_bundle = make_traceable_objective_runtime_bundle(
        boozer_surface,
        field,
        iota_target,
        outer_objective_config=objective_config,
        session=session,
    )
    objective = cast(Callable[[jax.Array], jax.Array], runtime_bundle["objective"])
    value_and_grad = cast(
        Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        runtime_bundle["value_and_grad"],
    )
    reporting_metrics_from_solution = cast(
        Callable[..., Mapping[str, jax.Array]],
        runtime_bundle["reporting_metrics_from_solution"],
    )
    runtime_state = cast(Mapping[str, object], session["state"])
    endpoint_coil_set_spec_from_dofs = cast(
        Callable[[jax.Array], object], runtime_state["coil_set_spec_from_dofs"]
    )
    endpoint_solver = cast(_TraceableEndpointSolver, boozer_surface.run_code_traceable)

    def trial_evaluator(x: np.ndarray) -> ObjectiveTrialEvaluation:
        traceable_trial_evaluator = cast(
            Callable[[jax.Array], TraceableObjectiveTrialResult],
            session.trial_evaluator(),
        )
        result = traceable_trial_evaluator(jnp.asarray(x, dtype=jnp.float64))
        gradient = np.asarray(result.gradient, dtype=np.float64)
        raw_objective = float(result.raw_objective_value)
        filtered_objective = float(result.filtered_objective_value)
        raw_certified = bool(
            result.primal_success
            and result.actual_adjoint_success
            and result.gradient_is_finite
            and np.isfinite(raw_objective)
        )
        return ObjectiveTrialEvaluation(
            raw_objective=raw_objective if np.isfinite(raw_objective) else None,
            raw_objective_certified=raw_certified,
            filtered_objective=(
                filtered_objective if np.isfinite(filtered_objective) else None
            ),
            gradient=gradient,
            gradient_source=cast(
                Literal["candidate", "baseline", "unavailable"],
                result.gradient_source,
            ),
            predictor_kind=boozer_surface.boozer_type,
            predictor_success=result.predictor_success,
            primal_success=result.primal_success,
            adjoint_success=result.actual_adjoint_success,
            newton_success=result.newton_success,
            newton_stop_reason_code=result.newton_stop_reason_code,
            newton_accepted_iterations=result.newton_iterations,
            newton_attempted_iterations=result.newton_attempted_iterations,
            newton_last_linear_solve_success=(result.newton_last_linear_solve_success),
            inner_penalty_residual_l2=(
                result.inner_penalty_residual_l2
                if result.inner_penalty_residual_l2 is not None
                else None
            ),
            inner_final_gradient_inf_norm=(
                result.final_gradient_inf_norm
                if result.final_gradient_inf_norm is not None
                else None
            ),
        )

    def scientific_endpoint(x: np.ndarray) -> ScientificEndpointEvidence:
        return _certified_traceable_endpoint(
            x,
            coil_set_spec_from_dofs=endpoint_coil_set_spec_from_dofs,
            run_code_traceable=endpoint_solver,
            install_traceable_solved_runtime_state=(
                boozer_surface.install_traceable_solved_runtime_state
            ),
            surface_dofs=surface_dofs,
            iota_seed=-0.406,
            G_seed=g0,
            reporting_metrics_from_solution=reporting_metrics_from_solution,
        )

    native_curves, native_currents, native_axis, native_nfp, native_field = get_data(
        "ncsx",
        coil_order=3,
        magnetic_axis_order=3,
        points_per_period=8,
    )
    native_currents[0].fix_all()
    native_current_sum = native_nfp * sum(
        abs(current.get_value()) for current in native_currents
    )
    native_g0 = (
        2.0 * np.pi * native_current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    )
    native_surface = SurfaceXYZTensorFourier(
        mpol=resolution,
        ntor=resolution,
        stellsym=True,
        nfp=native_nfp,
        quadpoints_phi=np.linspace(
            0.0,
            1.0 / native_nfp,
            2 * resolution + 1,
            endpoint=False,
        ),
        quadpoints_theta=np.linspace(
            0.0,
            1.0,
            2 * resolution + 1,
            endpoint=False,
        ),
    )
    native_surface.fit_to_curve(native_axis, 0.1, flip_theta=True)
    native_volume = Volume(native_surface)
    native_solver = BoozerSurface(
        native_field,
        native_surface,
        native_volume,
        float(native_volume.J()),
    )
    native_initial_solution = native_solver.solve_residual_equation_exactly_newton(
        tol=1.0e-13,
        maxiter=20,
        iota=-0.406,
        G=native_g0,
    )
    native_iota_target = float(native_initial_solution["iota"])
    native_initial_surface_dofs = np.asarray(
        native_solver.surface.x,
        dtype=np.float64,
    ).copy()
    native_initial_iota = float(native_initial_solution["iota"])
    native_initial_G = float(native_initial_solution["G"])

    def reset_native() -> None:
        """Restore the native inner solve before each timed outer run."""
        native_solver.surface.x = native_initial_surface_dofs.copy()
        native_solver.res["iota"] = native_initial_iota
        native_solver.res["G"] = native_initial_G
        native_solver.set_recompute_flag()

    native_major_radius = MajorRadius(native_solver)
    native_lengths = [CurveLength(curve) for curve in native_curves]
    native_non_qs = NonQuasiSymmetricRatio(
        native_solver,
        BiotSavart(native_field.coils),
        sDIM=4,
    )
    native_residual = BoozerResidual(native_solver, native_field)
    native_iota_penalty = QuadraticPenalty(
        Iotas(native_solver),
        native_iota_target,
        "identity",
    )
    native_radius_penalty = QuadraticPenalty(
        native_major_radius,
        float(native_major_radius.J()),
        "identity",
    )
    native_length_penalty = QuadraticPenalty(
        sum(native_lengths),
        float(sum(native_lengths).J()),
        "max",
    )
    native_objective = (
        native_non_qs
        + native_residual
        + native_iota_penalty
        + native_radius_penalty
        + native_length_penalty
    )

    def native_trial_evaluator(x: np.ndarray) -> ObjectiveTrialEvaluation:
        previous_surface = np.asarray(native_solver.surface.x, dtype=np.float64)
        previous_iota = float(native_solver.res["iota"])
        previous_g = float(native_solver.res["G"])
        native_objective.x = np.asarray(x, dtype=np.float64)
        value = float(native_objective.J())
        gradient = np.asarray(native_objective.dJ(), dtype=np.float64)
        inner_success = bool(native_solver.res["success"])
        if not inner_success:
            native_solver.surface.x = previous_surface
            native_solver.res["iota"] = previous_iota
            native_solver.res["G"] = previous_g
        gradient_finite = bool(np.all(np.isfinite(gradient)))
        return ObjectiveTrialEvaluation(
            raw_objective=value if np.isfinite(value) else None,
            raw_objective_certified=bool(
                inner_success and np.isfinite(value) and gradient_finite
            ),
            filtered_objective=value if inner_success else 1.0e3,
            gradient=gradient,
            gradient_source="candidate" if gradient_finite else "unavailable",
            predictor_kind=None,
            predictor_success=None,
            primal_success=inner_success,
            adjoint_success=None,
            newton_success=inner_success,
            newton_stop_reason_code=None,
            newton_accepted_iterations=None,
            newton_attempted_iterations=None,
            newton_last_linear_solve_success=None,
            inner_penalty_residual_l2=None,
            inner_final_gradient_inf_norm=None,
        )

    def native_value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        evaluation = native_trial_evaluator(x)
        if evaluation.filtered_objective is None:
            return float("nan"), evaluation.gradient
        return evaluation.filtered_objective, evaluation.gradient

    def native_scientific_endpoint(x: np.ndarray) -> ScientificEndpointEvidence:
        reset_native()
        native_objective.x = np.asarray(x, dtype=np.float64)
        native_objective.J()
        inner_success = bool(native_solver.res["success"])
        if not inner_success:
            return ScientificEndpointEvidence(inner_success=False, observables=())
        return ScientificEndpointEvidence(
            inner_success=True,
            observables=(
                ("final_boozer_residual", float(native_residual.J())),
                ("final_non_qs", float(native_non_qs.J())),
                ("final_iota", float(native_solver.res["iota"])),
                ("final_volume", float(native_volume.J())),
            ),
        )

    initial = np.asarray(field.x, dtype=np.float64)
    traceable_iota = float(np.asarray(jax.device_get(traceable_result["iota"])))
    traceable_G = float(np.asarray(jax.device_get(traceable_result["G"])))
    metadata: FixtureMetadata = (
        (
            "source_example",
            "examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py",
        ),
        ("field_source", "simsopt.configs.get_data('ncsx')"),
        ("nfp", int(nfp)),
        ("coil_order", 3),
        ("magnetic_axis_order", 3),
        ("points_per_period", 8),
        ("surface_mpol", resolution),
        ("surface_ntor", resolution),
        ("surface_dof_count", int(surface_dofs.size)),
        ("newton_maxiter", 20),
        ("bfgs_maxiter", 2),
        ("dtype", "float64"),
        ("seed", 0),
        (
            "native_objective_provider",
            "simsopt.geo.BoozerSurface + native outer objective",
        ),
        ("traceable_run_code_success", True),
        ("traceable_inner_iota", traceable_iota),
        ("traceable_inner_G", traceable_G),
    )
    return Fixture(
        name="boozer",
        objective=objective,
        initial=initial,
        expected_dimension=int(initial.size),
        source="source_owned_boozer_vacuum",
        certificate="source-owned VMEC-free Boozer physics; native/JAX objective parity",
        method="bfgs",
        value_and_grad=value_and_grad,
        native_value_and_grad=native_value_and_grad,
        native_reset=reset_native,
        scientific_endpoint=scientific_endpoint,
        native_scientific_endpoint=native_scientific_endpoint,
        trial_evaluator=trial_evaluator,
        native_trial_evaluator=native_trial_evaluator,
        accepted_incumbent_host_value_and_grad=(
            session.accepted_incumbent_host_value_and_grad
        ),
        metadata=metadata,
    )


def rosenbrock() -> Fixture:
    """Return the fixed two-variable Rosenbrock trajectory fixture."""

    def objective(x: jax.Array) -> jax.Array:
        return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

    return Fixture(
        name="rosenbrock",
        objective=objective,
        initial=np.asarray((-1.2, 1.0), dtype=np.float64),
        expected_dimension=2,
        source="tests/jax/solve/test_lbfgsb_trajectory_parity.py",
        certificate="accepted-step trajectory contract",
        method="lbfgs",
    )


def bfgs_quadratic() -> Fixture:
    """Return the quadratic contract with the dense-BFGS method selected."""

    base = quadratic47()
    return Fixture(
        name="bfgs_quadratic",
        objective=base.objective,
        initial=base.initial,
        expected_dimension=base.expected_dimension,
        source=base.source,
        certificate="solver-runtime-only; deterministic dense-BFGS contract",
        method="bfgs",
    )


_FIXTURE_DEFINITIONS: dict[str, FixtureDefinition] = {
    "coil47": FixtureDefinition(coil47_physics, "lbfgs"),
    "boozer": FixtureDefinition(boozer_physics, "bfgs"),
    "bfgs_quadratic": FixtureDefinition(bfgs_quadratic, "bfgs"),
    "rosenbrock": FixtureDefinition(rosenbrock, "lbfgs"),
}


def fixture_method(name: str) -> Literal["bfgs", "lbfgs"]:
    """Return solver metadata without constructing expensive physics state."""

    try:
        return _FIXTURE_DEFINITIONS[name].method
    except KeyError as exc:
        raise ValueError(f"unknown custom quasi-Newton fixture: {name}") from exc


def fixture(name: str) -> Fixture:
    """Resolve a named deterministic fixture without dynamic imports."""

    try:
        definition = _FIXTURE_DEFINITIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown custom quasi-Newton fixture: {name}") from exc
    fixture_case = definition.builder()
    if fixture_case.method != definition.method:
        raise RuntimeError(f"fixture method metadata drifted for {name!r}")
    return fixture_case
