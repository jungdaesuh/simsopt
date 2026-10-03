"""Native SIMSOPT against its JAX mirror at one shared starting state.

Each case builds an official upstream example's native problem and the JAX
mirror's own construction at the same small-size start state, evaluates the
objective (and its gradient where the mirror defines one) once on each side,
and compares them: native against JAX CPU, and against JAX GPU when CUDA is
present. No reference data is stored; both sides are computed live.

Tolerances are the initial-phase values the example parity contracts declared:
the ``native_workflow`` same-state values of ``simsopt_jax.parity_tolerances``
(``direct_kernel`` for QFM), value or derivative by the observable's name,
applied as ``allclose(native, jax)``.

Not covered here, because no initial-state comparison applies (``surf_vol_area``
declared none; the three tracing examples start from the shared seed input) or
because their own tests compare them with native SIMSOPT: ``surf_vol_area``
and the tracing examples (``tests/integration/test_jax_official_tiny_least_squares.py``,
``tests/integration/test_jax_tracing_*``), ``permanent_magnet_MUSE`` and ``_QA``
(``tests/integration/test_jax_permanent_magnet_{arbvec,qa}_native_parity.py``),
``stage_two_optimization_stochastic``
(``tests/jax/examples/test_stochastic_matched_state_parity.py``) and
``wireframe_rcls_with_ports`` (``tests/integration/test_jax_wireframe_rcls_matches_native.py``).
The planned single-stage example has no mirror to compare.
"""

from __future__ import annotations

from jax_test_support import (
    enable_strict_parity_backend,
    fixture_jax_runtime_guard,  # noqa: F401
    fixture_parity_lane,  # noqa: F401
    parity_device,
)

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt.configs import get_data
from simsopt.field import (
    B2Energy,
    BiotSavart,
    Coil,
    Current,
    LpCurveForce,
    ToroidalField,
    apply_symmetries_to_currents,
    apply_symmetries_to_curves,
    coils_via_symmetries,
)
from simsopt.geo import (
    Area,
    BoozerSurface,
    CurveCurveDistance,
    CurveLength,
    CurveRZFourier,
    CurveSurfaceDistance,
    CurveXYZFourier,
    FramedCurveCentroid,
    FrameRotation,
    Iotas,
    LinkingNumber,
    LPBinormalCurvatureStrainPenalty,
    LpCurveCurvature,
    LPTorsionalStrainPenalty,
    MajorRadius,
    MeanSquaredCurvature,
    NonQuasiSymmetricRatio,
    PermanentMagnetGrid,
    QfmSurface,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    ToroidalWireframe,
    Volume,
    create_equally_spaced_curves,
    create_equally_spaced_planar_curves,
    create_multifilament_grid,
)
from simsopt.objectives import LeastSquaresProblem, QuadraticPenalty, SquaredFlux
from simsopt.objectives.functions import Identity
from simsopt.solve import GPMO
from simsopt.solve.wireframe_optimization import bnorm_obj_matrices, gsco_wireframe
from simsopt.util import (
    FocusData,
    FocusPlasmaBnormal,
    discretize_polarizations,
    orientation_phi,
    polarization_axes,
    read_focus_coils,
)
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core import compute_filament_offsets
from simsopt_jax.examples import (
    solve_strain_rotation,
    solve_wireframe_rcls,
    standard_stage_two_state,
)
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_CONSTRAINT_WEIGHT,
    OFFICIAL_INITIAL_IOTA,
    OFFICIAL_LBFGS_MAXITER,
    OFFICIAL_LS_MAXITER,
    OFFICIAL_QA_INITIAL_IOTA,
    OFFICIAL_QA_NEWTON_MAXITER,
    OFFICIAL_QA_NON_QS_RESOLUTION,
    OFFICIAL_QA_RESIDUAL_WEIGHT,
    OFFICIAL_QA_SURFACE_DISTANCE,
    OFFICIAL_SOLVER_TOLERANCE,
    OFFICIAL_SURFACE_DISTANCE,
    boozer_official_options,
)
from simsopt_jax.examples.official_tiny_least_squares import (
    CONTROLLED_CURVE_INITIAL_FULL,
    curve_length_residual,
    quadratic_residual,
    value_and_jacobian,
)
from simsopt_jax.examples.qfm_host_scipy import (
    QFM_SURFACE_RESOLUTION,
    build_qfm_host_kernels,
)
from simsopt_jax.examples.stage_two_minimal import minimal_stage_two_state
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.objectives import StageTwoObjectiveConfig
from simsopt_jax.parity_tolerances import parity_ladder_tolerances
from simsopt_jax.solve.permanent_magnet import (
    GPMOPublicResult,
    GPMO_ArbVec_backtracking_jax,
    GPMO_baseline_jax,
)
from simsopt_jax_adapters.examples.gpmo_rules import gpmo_history_period
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_qa_problem import BoozerQAProblem
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.objectives import (
    FiniteBuildStageTwoConfig,
    ForceStageTwoConfig,
    make_finite_build_stage_two_objective,
    make_force_stage_two_length_penalty,
    make_force_stage_two_objective,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX
from simsopt_jax_adapters.solve.wireframe import (
    bnorm_obj_matrices_jax,
    gsco_wireframe_jax,
)

TEST_DATA = Path(__file__).resolve().parents[1] / "test_files"
_MU0 = 4.0e-7 * np.pi
_DERIVATIVE_TOKENS = ("gradient", "jacobian", "derivative")

Observables = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class SameStateCase:
    """One mirror: its native and JAX evaluations at the shared start state."""

    example_id: str
    native: Callable[[], Observables]
    jax: Callable[[], Observables]
    tolerance_bucket: str = "native_workflow"


def _tolerance(bucket: str, observable: str) -> tuple[float, float]:
    """The declared initial-phase tolerance of one observable."""
    tolerance = parity_ladder_tolerances(bucket)
    keys = ("rtol", "atol")
    if bucket == "native_workflow":
        derivative = any(token in observable for token in _DERIVATIVE_TOKENS)
        prefix = "same_state_derivative" if derivative else "same_state_value"
        keys = (f"{prefix}_rtol", f"{prefix}_atol")
    rtol, atol = tolerance[keys[0]], tolerance[keys[1]]
    assert isinstance(rtol, float) and isinstance(atol, float)
    return rtol, atol


def _host(values: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value, dtype=np.float64)
        for name, value in jax.device_get(dict(values)).items()
    }


def _value_and_gradient(objective) -> Observables:
    return {
        "objective": np.asarray(objective.J()),
        "objective_gradient": objective.dJ(),
    }


def _jax_value_and_gradient(objective, parameters: np.ndarray) -> Observables:
    value, gradient = jax.jit(jax.value_and_grad(objective))(parameters)
    return _host({"objective": value, "objective_gradient": gradient})


# --- 1_Simple/just_a_quadratic.py and minimize_curve_length.py ---------------

_QUADRATIC_START = np.zeros(3)
_QUADRATIC_TARGETS = np.asarray((1.0, 2.0, 3.0))
_QUADRATIC_WEIGHTS = np.asarray((1.0, 2.0, 3.0))


def _quadratic_native() -> Observables:
    identities = tuple(Identity() for _ in _QUADRATIC_START)
    problem = LeastSquaresProblem.from_tuples(
        [
            (identity.f, float(target), float(weight))
            for identity, target, weight in zip(
                identities, _QUADRATIC_TARGETS, _QUADRATIC_WEIGHTS, strict=True
            )
        ]
    )
    problem.x = _QUADRATIC_START
    return {
        "residual": problem.residuals(),
        "objective_sum_squares": np.asarray(problem.objective()),
    }


def _quadratic_jax() -> Observables:
    residual, _jacobian = value_and_jacobian(
        quadratic_residual(_QUADRATIC_TARGETS, _QUADRATIC_WEIGHTS), _QUADRATIC_START
    )
    return {
        "residual": residual,
        "objective_sum_squares": np.asarray(residual @ residual),
    }


def _start_curve() -> CurveRZFourier:
    curve = CurveRZFourier(100, 4, 5, True)
    curve.x = np.asarray(CONTROLLED_CURVE_INITIAL_FULL)
    curve.fix(0)
    return curve


def _curve_length_native() -> Observables:
    objective = CurveLength(_start_curve())
    return {
        "length": np.asarray(objective.J()),
        "residual_jacobian": np.asarray(objective.dJ())[np.newaxis, :],
    }


def _curve_length_jax() -> Observables:
    curve = _start_curve()
    free_positions = np.flatnonzero(curve.local_dofs_free_status)
    full_dofs = np.asarray(curve.local_full_x, dtype=np.float64)
    residual = curve_length_residual(
        full_dofs,
        np.asarray(curve.quadpoints, dtype=np.float64),
        free_positions,
        order=curve.order,
        nfp=curve.nfp,
        stellsym=curve.stellsym,
    )
    value, jacobian = value_and_jacobian(residual, full_dofs[free_positions])
    return {"length": np.asarray(value[0]), "residual_jacobian": jacobian}


# --- Stage-II family: Landreman-Paul QA surface, Fourier coils ---------------


def _qa_surface(resolution: int = 4) -> SurfaceRZFourier:
    surface = SurfaceRZFourier.from_vmec_input(
        str(TEST_DATA / "input.LandremanPaul2021_QA"),
        range="half period",
        nphi=resolution,
        ntheta=resolution,
    )
    surface.fix_all()
    return surface


def _stage_two_geometry(
    *,
    curve_factory: Callable[..., list] = create_equally_spaced_curves,
    num_base_curves: int = 4,
    numquadpoints: int = 16,
    current_scale: float | None = None,
    regularization: float | None = None,
):
    """The official QA coil set; the defaults are ``stage_two_optimization``'s."""
    surface = _qa_surface()
    base_curves = curve_factory(
        num_base_curves,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=2,
        numquadpoints=numquadpoints,
    )
    # Official minimal: ``Current(1.0) * 1e5``; the other scripts: ``Current(1e5)``.
    base_currents = [
        Current(1.0e5) if current_scale is None else Current(1.0) * current_scale
        for _ in base_curves
    ]
    base_currents[0].fix_all()
    regularizations = (
        None if regularization is None else [regularization] * num_base_curves
    )
    coils = coils_via_symmetries(
        base_curves, base_currents, surface.nfp, True, regularizations
    )
    return surface, base_curves, coils


def _surface_arrays(surface: SurfaceRZFourier) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(surface.gamma()).reshape((-1, 3)),
        np.asarray(surface.normal()).reshape((-1, 3)),
    )


def _native_penalties(config: StageTwoObjectiveConfig, surface, coils, weight):
    """The official scripts' coil penalties, spelled natively from one config.

    As in the JAX objective, a term with zero weight is absent.
    """
    base_curves = [coil.curve for coil in coils[: config.num_base_curves]]
    curves = [coil.curve for coil in coils]
    total_length = sum(CurveLength(curve) for curve in base_curves)
    if config.length_target is not None:
        total_length = QuadraticPenalty(
            total_length, config.length_target, config.length_target_mode
        )
    terms = (
        (weight, total_length),
        (
            config.curve_curve_weight,
            CurveCurveDistance(
                curves,
                config.curve_curve_minimum_distance,
                num_basecurves=config.num_base_curves,
            ),
        ),
        (
            config.curve_surface_weight,
            CurveSurfaceDistance(
                curves, surface, config.curve_surface_minimum_distance
            ),
        ),
        (
            config.curvature_weight,
            sum(
                LpCurveCurvature(c, 2, config.curvature_threshold) for c in base_curves
            ),
        ),
        (
            config.mean_squared_curvature_weight,
            sum(
                QuadraticPenalty(
                    MeanSquaredCurvature(c),
                    config.mean_squared_curvature_threshold,
                    config.mean_squared_curvature_target_mode,
                )
                for c in base_curves
            ),
        ),
        (config.linking_number_weight, LinkingNumber(curves)),
    )
    return sum(weight * term for weight, term in terms if weight != 0.0)


def _standard_native(geometry, config, length_weight) -> Observables:
    surface, _base_curves, coils = geometry()
    return _value_and_gradient(
        SquaredFlux(surface, BiotSavart(coils))
        + _native_penalties(config, surface, coils, length_weight)
    )


def _standard_jax(geometry, config, length_weight) -> Observables:
    surface, _base_curves, coils = geometry()
    field = BiotSavartJAX(coils)
    gamma, normal = _surface_arrays(surface)
    state = standard_stage_two_state(
        field=field,
        flux_spec=SquaredFluxJAX(surface, field).fixed_surface_flux_spec(),
        surface_gamma=gamma,
        surface_normal=normal,
        parameters=np.asarray(field.x),
        regularization_config=config,
        length_weight=length_weight,
    )
    return _host(
        {"objective": state.objective, "objective_gradient": state.objective_gradient}
    )


#: ``2_Intermediate/stage_two_optimization.py``, first-stage length weight.
_STANDARD = (
    _stage_two_geometry,
    StageTwoObjectiveConfig(
        num_base_curves=4,
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=1000.0,
        curve_surface_minimum_distance=0.3,
        curve_surface_weight=10.0,
        curvature_threshold=5.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=5.0,
        mean_squared_curvature_weight=1.0e-6,
    ),
    1.0e-6,
)
#: ``2_Intermediate/stage_two_optimization_planar_coils.py``, first stage.
_PLANAR = (
    functools.partial(
        _stage_two_geometry,
        curve_factory=create_equally_spaced_planar_curves,
        numquadpoints=32,
    ),
    StageTwoObjectiveConfig(
        num_base_curves=4,
        length_target=10.4,
        length_target_mode="identity",
        curve_curve_minimum_distance=0.08,
        curve_curve_weight=1000.0,
        curve_surface_minimum_distance=0.12,
        curve_surface_weight=10.0,
        curvature_threshold=10.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=10.0,
        mean_squared_curvature_target_mode="identity",
        mean_squared_curvature_weight=1.0e-6,
        linking_number_weight=1.0,
    ),
    10.0,
)

# --- 1_Simple/stage_two_optimization_minimal.py ------------------------------

_MINIMAL_GEOMETRY = functools.partial(_stage_two_geometry, current_scale=1.0e5)
_MINIMAL_CONFIG = StageTwoObjectiveConfig(num_base_curves=4, length_target=18.0)


def _minimal_native() -> Observables:
    surface, _base_curves, coils = _MINIMAL_GEOMETRY()
    flux = SquaredFlux(surface, BiotSavart(coils))
    return {
        **_value_and_gradient(
            flux + _native_penalties(_MINIMAL_CONFIG, surface, coils, 1.0)
        ),
        "squared_flux": np.asarray(flux.J()),
    }


def _minimal_jax() -> Observables:
    surface, _base_curves, coils = _MINIMAL_GEOMETRY()
    field = BiotSavartJAX(coils)
    gamma, normal = _surface_arrays(surface)
    state = minimal_stage_two_state(
        field=field,
        flux_spec=SquaredFluxJAX(surface, field).fixed_surface_flux_spec(),
        surface_gamma=gamma,
        surface_normal=normal,
        num_base_curves=4,
        length_weight=1.0,
        length_target=18.0,
    )(np.asarray(field.x))
    return _host(
        {
            "objective": state.objective,
            "objective_gradient": state.objective_gradient,
            "squared_flux": state.squared_flux,
        }
    )


# --- 3_Advanced/stage_two_optimization_finitebuild.py ------------------------

_FILAMENTS = (2, 3)
_FILAMENT_GAPS = (0.02, 0.04)


def _finite_build_geometry():
    """Shared-frame multifilament packs around four base coils."""
    surface = _qa_surface()
    base_curves = create_equally_spaced_curves(
        4, surface.nfp, stellsym=True, R0=1.0, R1=0.7, order=2, numquadpoints=8
    )
    filaments = int(np.prod(_FILAMENTS))
    unit_currents = [Current(1.0) for _ in base_curves]
    unit_currents[0].fix_all()
    currents = [current * (1.0e5 / filaments) for current in unit_currents]
    packs = [
        create_multifilament_grid(c, *_FILAMENTS, *_FILAMENT_GAPS, rotation_order=1)
        for c in base_curves
    ]
    coils = [
        Coil(curve, current)
        for curve, current in zip(
            apply_symmetries_to_curves(
                [filament for pack in packs for filament in pack], surface.nfp, True
            ),
            apply_symmetries_to_currents(
                [current for current in currents for _ in range(filaments)],
                surface.nfp,
                True,
            ),
            strict=True,
        )
    ]
    config = FiniteBuildStageTwoConfig(
        num_base_curves=4,
        filament_offsets=compute_filament_offsets(
            numfilaments_n=_FILAMENTS[0],
            numfilaments_b=_FILAMENTS[1],
            gapsize_n=_FILAMENT_GAPS[0],
            gapsize_b=_FILAMENT_GAPS[1],
        ),
        symmetry_copies=2 * surface.nfp,
        length_targets=tuple(float(CurveLength(c).J()) for c in base_curves),
        length_weight=1.0e-2,
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=10.0,
    )
    return surface, base_curves, coils, config


def _finite_build_native() -> Observables:
    surface, base_curves, coils, config = _finite_build_geometry()
    lengths = [CurveLength(curve) for curve in base_curves]
    distance = CurveCurveDistance(
        apply_symmetries_to_curves(base_curves, surface.nfp, True),
        config.curve_curve_minimum_distance,
    )
    return _value_and_gradient(
        SquaredFlux(surface, BiotSavart(coils))
        + config.length_weight
        * sum(
            QuadraticPenalty(length, target, "max")
            for length, target in zip(lengths, config.length_targets, strict=True)
        )
        + config.curve_curve_weight * distance
    )


def _finite_build_jax() -> Observables:
    surface, _base_curves, coils, config = _finite_build_geometry()
    field = BiotSavartJAX(coils)
    flux_spec = SquaredFluxJAX(surface, field).fixed_surface_flux_spec()
    return _jax_value_and_gradient(
        make_finite_build_stage_two_objective(field, flux_spec, config),
        np.asarray(field.x),
    )


# --- 3_Advanced/coil_forces.py -----------------------------------------------

_FORCE_RADIUS = 0.05**2 / np.sqrt(np.e)
_FORCE_LENGTH_WEIGHT = 1.0e-3
_FORCE_GEOMETRY = functools.partial(
    _stage_two_geometry,
    num_base_curves=3,
    numquadpoints=8,
    regularization=_FORCE_RADIUS,
)
_FORCE_CONFIG = StageTwoObjectiveConfig(
    num_base_curves=3,
    length_target=17.4,
    curve_curve_minimum_distance=0.1,
    curve_curve_weight=1000.0,
    curve_surface_minimum_distance=0.3,
    curve_surface_weight=10.0,
    curvature_threshold=5.0,
    curvature_weight=1.0e-6,
    mean_squared_curvature_threshold=5.0,
    mean_squared_curvature_weight=1.0e-6,
)
_FORCE_TERMS = ForceStageTwoConfig(
    num_force_coils=3,
    force_weight=1.0e-2,
    vacuum_energy_weight=1.0e-4,
    force_power=4.0,
    force_threshold=0.0,
)


def _coil_forces_native() -> Observables:
    surface, _base_curves, coils = _FORCE_GEOMETRY()
    terms = _FORCE_TERMS
    force = LpCurveForce(
        coils[: terms.num_force_coils],
        coils,
        p=terms.force_power,
        threshold=terms.force_threshold,
    )
    return _value_and_gradient(
        SquaredFlux(surface, BiotSavart(coils))
        + _native_penalties(_FORCE_CONFIG, surface, coils, _FORCE_LENGTH_WEIGHT)
        + terms.force_weight * force
        + terms.vacuum_energy_weight * B2Energy(coils)
    )


def _coil_forces_jax() -> Observables:
    surface, base_curves, coils = _FORCE_GEOMETRY()
    field = BiotSavartJAX(coils)
    objective = make_force_stage_two_objective(
        field,
        SquaredFluxJAX(surface, field).traceable_objective(),
        *_surface_arrays(surface),
        np.stack([np.asarray(c.quadpoints) for c in base_curves]),
        np.full(len(coils), _FORCE_RADIUS),
        _FORCE_CONFIG,
        _FORCE_TERMS,
    )
    length_penalty = make_force_stage_two_length_penalty(field, _FORCE_CONFIG)
    return _jax_value_and_gradient(
        lambda x: objective(x) + length_penalty(x, _FORCE_LENGTH_WEIGHT),
        np.asarray(field.x),
    )


# --- 2_Intermediate/strain_optimization.py -----------------------------------

_STRAIN_ORDER = 10
_STRAIN_WIDTH = 1.0e-3
_STRAIN_THRESHOLD = 2.0e-3


def _strain_curve() -> CurveXYZFourier:
    """Coil 1 of HSX, scaled by 0.1 and fixed, as the official script builds it."""
    source = get_data("hsx", coil_order=10, points_per_period=10)[0][1]
    curve = CurveXYZFourier(source.quadpoints, source.order)
    curve.x = np.asarray(source.x) * 0.1
    curve.fix_all()
    return curve


def _strain_native() -> Observables:
    curve = _strain_curve()
    framed = FramedCurveCentroid(curve, FrameRotation(curve.quadpoints, _STRAIN_ORDER))
    options = {"width": _STRAIN_WIDTH, "p": 2, "threshold": _STRAIN_THRESHOLD}
    objective = LPTorsionalStrainPenalty(
        framed, **options
    ) + LPBinormalCurvatureStrainPenalty(framed, **options)
    objective.x = np.zeros(2 * _STRAIN_ORDER + 1)
    return {"objective": np.asarray(objective.J()), "gradient": objective.dJ()}


def _strain_jax() -> Observables:
    curve = _strain_curve()
    # The mirror publishes its start state through its solve, which evaluates
    # ``initial`` before the optimizer's first (and here only) iteration.
    initial = solve_strain_rotation(
        quadpoints=np.asarray(curve.quadpoints),
        gamma=curve.gamma(),
        gammadash=curve.gammadash(),
        gammadashdash=curve.gammadashdash(),
        initial_parameters=np.zeros(2 * _STRAIN_ORDER + 1),
        rotation_order=_STRAIN_ORDER,
        objective_width=_STRAIN_WIDTH,
        reporting_width=3.0e-3,
        torsional_threshold=_STRAIN_THRESHOLD,
        curvature_threshold=_STRAIN_THRESHOLD,
        maxiter=1,
        maxfun=15000,
        gtol=1.0e-20,
        ftol=1.0e-20,
        maxcor=10,
        maxls=20,
    ).initial
    return _host({"objective": initial.objective, "gradient": initial.gradient})


# --- 1_Simple/qfm.py ---------------------------------------------------------


def _qfm_geometry():
    _, _, magnetic_axis, nfp, field = get_data("ncsx")
    resolution = QFM_SURFACE_RESOLUTION["bounded"]
    size = resolution.quadrature_size
    surface = SurfaceRZFourier(
        mpol=resolution.order,
        ntor=resolution.order,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, size, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, size, endpoint=False),
    )
    surface.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    return field, surface


def _qfm_native() -> Observables:
    field, surface = _qfm_geometry()
    volume = Volume(surface)
    qfm = QfmSurface(field, surface, volume, float(volume.J()))
    value, gradient = qfm.qfm_objective(np.asarray(surface.x), derivatives=1)
    return {"qfm_value": np.asarray(value), "qfm_gradient": np.asarray(gradient)}


def _qfm_jax() -> Observables:
    field, surface = _qfm_geometry()
    kernels = build_qfm_host_kernels(
        initial_parameters=np.asarray(surface.x),
        quadpoints_phi=np.asarray(surface.quadpoints_phi),
        quadpoints_theta=np.asarray(surface.quadpoints_theta),
        coil_set_spec=BiotSavartJAX(field.coils).coil_set_spec(),
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )
    value, gradient = kernels.host_qfm()(np.asarray(surface.x))
    return {"qfm_value": np.asarray(value), "qfm_gradient": gradient}


# --- 2_Intermediate/boozer.py and boozerQA.py: NCSX Boozer start -------------


def _ncsx_boozer_start(distance: float, **coil_options: int):
    base_curves, base_currents, magnetic_axis, nfp, field = get_data(
        "ncsx", **coil_options
    )
    base_currents[0].fix_all()
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    surface = SurfaceXYZTensorFourier(
        mpol=2,
        ntor=2,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 5, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
    )
    surface.fit_to_curve(magnetic_axis, distance, flip_theta=True)
    G0 = 2.0 * np.pi * current_sum * (_MU0 / (2.0 * np.pi))
    return base_curves, field, surface, int(nfp), float(G0)


def _boozer_start():
    _, field, surface, _, G0 = _ncsx_boozer_start(OFFICIAL_SURFACE_DISTANCE)
    start = np.concatenate((surface.get_dofs(), (OFFICIAL_INITIAL_IOTA, G0)))
    return field, surface, Area(surface), start


def _boozer_native() -> Observables:
    field, surface, area, start = _boozer_start()
    solver = BoozerSurface(field, surface, area, float(area.J()))
    residual, jacobian = solver._get_residual_vector_and_jacobian(
        start, OFFICIAL_CONSTRAINT_WEIGHT, True, True
    )
    return {"residual": residual, "jacobian": jacobian}


def _boozer_jax() -> Observables:
    field, surface, area, start = _boozer_start()
    jax_field = BiotSavartJAX(field.coils)
    solver = BoozerSurfaceJAX(
        jax_field,
        surface,
        area,
        float(area.J()),
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
        options=boozer_official_options(
            rough_maxiter=OFFICIAL_LBFGS_MAXITER,
            ls_maxiter=OFFICIAL_LS_MAXITER,
            tolerance=OFFICIAL_SOLVER_TOLERANCE,
        ),
    )
    kernels = solver._get_penalty_kernel_bundle(
        optimize_G=True,
        weight_inv_modB=True,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
    )
    x = explicit_device_array(start, dtype=np.float64)
    coils = jax_field.coil_set_spec()
    return _host(
        {"residual": kernels.residual(x, coils), "jacobian": kernels.jacobian(x, coils)}
    )


_BOOZER_QA_START = functools.partial(
    _ncsx_boozer_start,
    OFFICIAL_QA_SURFACE_DISTANCE,
    coil_order=3,
    magnetic_axis_order=3,
    points_per_period=8,
)
_BOOZER_QA_NEWTON_TOLERANCE = 1.0e-10


def _boozer_qa_native() -> Observables:
    base_curves, field, surface, _, G0 = _BOOZER_QA_START()
    volume = Volume(surface)
    solver = BoozerSurface(field, surface, volume, float(volume.J()))
    solution = solver.solve_residual_equation_exactly_newton(
        tol=_BOOZER_QA_NEWTON_TOLERANCE,
        maxiter=OFFICIAL_QA_NEWTON_MAXITER,
        iota=OFFICIAL_QA_INITIAL_IOTA,
        G=G0,
    )
    major_radius = MajorRadius(solver)
    length = sum(CurveLength(curve) for curve in base_curves)
    objective = (
        NonQuasiSymmetricRatio(
            solver, BiotSavart(field.coils), sDIM=OFFICIAL_QA_NON_QS_RESOLUTION
        )
        + QuadraticPenalty(Iotas(solver), float(solution["iota"]), "identity")
        + QuadraticPenalty(major_radius, float(major_radius.J()), "identity")
        + QuadraticPenalty(length, float(length.J()), "max")
    )
    return {**_value_and_gradient(objective), "iota": np.asarray(solution["iota"])}


def _boozer_qa_jax() -> Observables:
    base_curves, field, surface, nfp, G0 = _BOOZER_QA_START()
    problem = BoozerQAProblem(
        base_curves=base_curves,
        native_field=field,
        surface=surface,
        nfp=nfp,
        initial_G=G0,
        initial_iota=OFFICIAL_QA_INITIAL_IOTA,
        boozer_options={
            "newton_maxiter": OFFICIAL_QA_NEWTON_MAXITER,
            "newton_tol": _BOOZER_QA_NEWTON_TOLERANCE,
            "verbose": False,
        },
        non_qs_resolution=OFFICIAL_QA_NON_QS_RESOLUTION,
        residual_weight=OFFICIAL_QA_RESIDUAL_WEIGHT,
    )
    value, gradient = problem.value_and_gradient(problem.initial_coil_dofs)
    return {
        "objective": np.asarray(value),
        "objective_gradient": np.asarray(gradient),
        "iota": np.asarray(problem.iota_target),
    }


# --- Wireframe examples: QA plasma, NESCOIL or offset winding surface --------
# Each side computes the response matrix itself (C++ ``WireframeField`` against
# ``WireframeFieldJAX``). GSCO runs zero iterations, so its history holds only
# the start state each kernel evaluated.


def _poloidal_current(plasma: SurfaceRZFourier) -> float:
    return -2.0 * np.pi * plasma.get_rc(0, 0) / _MU0


def _nescoil_wireframe(n_phi: int) -> ToroidalWireframe:
    surface = SurfaceRZFourier.from_nescoil_input(
        TEST_DATA / "nescin.LandremanPaul2021_QA", "current"
    )
    return ToroidalWireframe(surface, n_phi, 8)


@dataclass(frozen=True)
class _GscoSeed:
    tf_coils: int
    lambda_s: float
    current_fraction: float | None  # sector saddle: fraction of I_pol
    break_width: int | None = None


_GSCO_MODULAR = _GscoSeed(tf_coils=6, lambda_s=1.0e-6, current_fraction=None)
_GSCO_SADDLE = _GscoSeed(3, 10.0**-6.5, current_fraction=0.05, break_width=2)


def _gsco_geometry(seed: _GscoSeed):
    plasma = _qa_surface()
    wireframe = _nescoil_wireframe(18)
    poloidal = _poloidal_current(plasma)
    tf_current = poloidal / (2 * wireframe.nfp * seed.tf_coils)
    wireframe.add_tfcoil_currents(seed.tf_coils, tf_current)
    if seed.break_width is not None:
        wireframe.set_toroidal_breaks(
            seed.tf_coils, seed.break_width, allow_pol_current=True
        )
    wireframe.set_poloidal_current(poloidal)
    fraction = seed.current_fraction
    current = abs(tf_current if fraction is None else fraction * poloidal)
    # lambda_S, no_crossing, match_current, default/max current, max_iter, print
    arguments = (seed.lambda_s, True, False, current, 1.1 * current, 0, 1)
    return wireframe, plasma, arguments


def _gsco_native(seed: _GscoSeed) -> Observables:
    wireframe, plasma, arguments = _gsco_geometry(seed)
    response, target = bnorm_obj_matrices(wireframe, plasma, verbose=False)
    *_, normal, _, total, _ = gsco_wireframe(
        wireframe, response, target, *arguments, verbose=False
    )
    return {
        "response_matrix": response,
        "normal_objective": normal[0],
        "total_objective": total[0],
    }


def _gsco_jax(seed: _GscoSeed) -> Observables:
    wireframe, plasma, arguments = _gsco_geometry(seed)
    response, target = bnorm_obj_matrices_jax(wireframe, plasma, verbose=False)
    result = gsco_wireframe_jax(
        wireframe, response, target, *arguments, record_every=1, verbose=False
    )
    return _host(
        {
            "response_matrix": response,
            "normal_objective": result.f_B_history[0],
            "total_objective": result.f_history[0],
        }
    )


def _multistep(field_type, matrices) -> Observables:
    """``3_Advanced/wireframe_gsco_multistep.py`` from zero currents: the start
    state is the response and the planar-TF normal field each side computes."""
    plasma = _qa_surface()
    wireframe = _nescoil_wireframe(24)
    wireframe.set_toroidal_breaks(3, 4, allow_pol_current=True)
    wireframe.set_poloidal_current(0.0)
    curves = create_equally_spaced_curves(3, plasma.nfp, True, R0=1.0, R1=0.85)
    tf_current = -_poloidal_current(plasma) / (2 * 3 * plasma.nfp)
    coils = coils_via_symmetries(
        curves, [Current(tf_current) for _ in curves], plasma.nfp, True
    )
    response, target = matrices(
        wireframe, plasma, ext_field=field_type(coils), verbose=False
    )
    return _host({"response_matrix": response, "target": target})


_RCLS_REGULARIZATION = 1.0e-10


def _rcls_geometry():
    """``2_Intermediate/wireframe_rcls_basic.py``. Start state: the minimum-norm
    currents that satisfy the poloidal-current constraint."""
    plasma = _qa_surface(16)
    surface = SurfaceRZFourier.from_vmec_input(
        str(TEST_DATA / "input.LandremanPaul2021_QA")
    )
    surface.extend_via_projected_normal(0.3)
    wireframe = ToroidalWireframe(surface, 4, 6)
    wireframe.set_poloidal_current(_poloidal_current(plasma))
    constraint, target = wireframe.constraint_matrices(
        assume_no_crossings=False, remove_constrained_segments=True
    )
    currents = np.zeros((wireframe.n_segments, 1))
    currents[wireframe.unconstrained_segments()] = constraint.T @ np.linalg.solve(
        constraint @ constraint.T, np.reshape(target, (-1, 1))
    )
    return wireframe, plasma, currents


def _rcls_native() -> Observables:
    wireframe, plasma, currents = _rcls_geometry()
    response, target = bnorm_obj_matrices(wireframe, plasma, verbose=False)
    residual = response @ currents - target
    normal_objective = 0.5 * np.vdot(residual, residual)
    regularization = 0.5 * _RCLS_REGULARIZATION**2 * np.vdot(currents, currents)
    return {
        "response_matrix": response,
        "normal_field_residual": residual,
        "total_objective": np.asarray(normal_objective + regularization),
    }


def _rcls_jax() -> Observables:
    wireframe, plasma, currents = _rcls_geometry()
    response, target = bnorm_obj_matrices_jax(wireframe, plasma, verbose=False)
    area = np.linalg.norm(plasma.normal(), axis=2)
    initial = solve_wireframe_rcls(
        wireframe=wireframe,
        response=response,
        target=target,
        regularization=_RCLS_REGULARIZATION,
        initial_currents=currents,
        plasma_points=plasma.gamma().reshape((-1, 3)),
        plasma_unit_normal=plasma.unitnormal().reshape((-1, 3)),
        plasma_area_weights=area.ravel() / area.size,
        wireframe_nodes=np.stack(wireframe.nodes),
        wireframe_segments=np.asarray(wireframe.segments, dtype=np.int32),
        wireframe_segment_signs=np.asarray(wireframe.seg_signs, dtype=np.float64),
        assume_no_crossings=False,
    ).initial
    return _host(
        {
            "response_matrix": response,
            "normal_field_residual": initial.normal_field_residual,
            "total_objective": initial.total_objective,
        }
    )


# --- Permanent-magnet examples: the first GPMO step from the start moments ---
# Every mirror stages the native host grid (``PermanentMagnetGridJAX.from_cpu``)
# and starts from zero moments, so the start residual is one host formula. What
# each solver computes at the start state is its first greedy step: it scores
# every candidate against the start residual and places the best one. Native
# GPMO records half the squared residual; JAX records the full one.


def _gpmo_step_native(
    grid: PermanentMagnetGrid, algorithm: str, **kwargs
) -> Observables:
    errors, _, _ = GPMO(grid, algorithm, K=1, nhistory=1, verbose=True, **kwargs)
    return {
        "step_one_moments": grid.m.reshape((-1, 3)),
        "step_one_residual": grid.A_obj @ grid.m - grid.b_obj,
        "step_one_objective_sum_squares": np.asarray(2.0 * errors[-1]),
    }


def _gpmo_step_jax(result: GPMOPublicResult) -> Observables:
    return _host(
        {
            "step_one_moments": result.m,
            "step_one_residual": result.residual,
            "step_one_objective_sum_squares": result.residual_history[-1],
        }
    )


def _pm_simple_grid() -> PermanentMagnetGrid:
    """``1_Simple/permanent_magnet_simple.py`` at its CI resolution."""
    surface = SurfaceRZFourier.from_wout(
        str(TEST_DATA / "wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc"),
        range="half period",
        nphi=2,
        ntheta=2,
    )
    field = ToroidalField(R0=1.0, B0=_MU0 * 3.7713e6 / (2.0 * np.pi))
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(field.B().reshape((2, 2, 3)) * surface.unitnormal(), axis=2)
    return PermanentMagnetGrid.geo_setup_from_famus(
        surface,
        normal_field,
        TEST_DATA / "init_orient_pm_nonorm_5E4_q4_dp.focus",
        coordinate_flag="cylindrical",
        downsample=100,
    )


def _pm_simple_native() -> Observables:
    return _gpmo_step_native(_pm_simple_grid(), "baseline", single_direction=-1)


def _pm_simple_jax() -> Observables:
    grid = PermanentMagnetGridJAX.from_cpu(_pm_simple_grid())
    return _gpmo_step_jax(GPMO_baseline_jax(grid, K=1, record_every=1))


def _pm4stell_grid() -> PermanentMagnetGrid:
    """``2_Intermediate/permanent_magnet_PM4Stell.py`` at its CI resolution."""
    plasma = TEST_DATA / "c09r00_B_axis_half_tesla_PM4Stell.plasma"
    magnets = TEST_DATA / "magpie_trial104b_PM4Stell.focus"
    surface = SurfaceRZFourier.from_focus(plasma, range="half period", nphi=2, ntheta=2)
    curves, currents, count = read_focus_coils(
        TEST_DATA / "tf_only_half_tesla_symmetry_baxis_PM4Stell.focus"
    )
    field = BiotSavart([Coil(curves[index], currents[index]) for index in range(count)])
    field.set_points(surface.gamma().reshape((-1, 3)))
    coil_normal = np.sum(field.B().reshape((2, 2, 3)) * surface.unitnormal(), axis=2)
    plasma_normal = FocusPlasmaBnormal(plasma).bnormal_grid(2, 2, "half period")
    magnet_data = FocusData(magnets, downsample=100)
    orientation = orientation_phi(TEST_DATA / "magpie_trial104b_corners_PM4Stell.csv")
    # The positive half of each official polarization family.
    families = [polarization_axes([name]) for name in ("face", "fe_ftri", "fc_ftri")]
    axes = np.concatenate([axes[: len(types) // 2] for axes, types in families])
    types = np.concatenate(
        [types[: len(types) // 2] + index for index, (_, types) in enumerate(families)]
    )
    discretize_polarizations(
        magnet_data, orientation[: magnet_data.nMagnets], axes, types
    )
    return PermanentMagnetGrid.geo_setup_from_famus(
        surface,
        plasma_normal + coil_normal,
        magnets,
        pol_vectors=np.stack(
            (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z), axis=-1
        ),
        m_maxima=5.0 / _MU0,
        downsample=100,
    )


_PM4STELL_OPTIONS = {
    "Nadjacent": 10,
    "backtracking": 200,
    "thresh_angle": np.pi,
    "max_nMagnets": 20,
}


def _pm4stell_native() -> Observables:
    grid = _pm4stell_grid()
    return _gpmo_step_native(
        grid,
        "ArbVec_backtracking",
        dipole_grid_xyz=grid.dipole_grid_xyz,
        **_PM4STELL_OPTIONS,
    )


def _pm4stell_jax() -> Observables:
    result = GPMO_ArbVec_backtracking_jax(
        PermanentMagnetGridJAX.from_cpu(_pm4stell_grid()),
        K=1,
        record_every=gpmo_history_period(iterations=1, history_count=1),
        **_PM4STELL_OPTIONS,
    )
    return _gpmo_step_jax(result)


CASES = (
    SameStateCase("native-just-a-quadratic", _quadratic_native, _quadratic_jax),
    SameStateCase(
        "native-minimize-curve-length", _curve_length_native, _curve_length_jax
    ),
    SameStateCase(
        "native-stage-two-optimization",
        functools.partial(_standard_native, *_STANDARD),
        functools.partial(_standard_jax, *_STANDARD),
    ),
    SameStateCase(
        "native-stage-two-optimization-planar-coils",
        functools.partial(_standard_native, *_PLANAR),
        functools.partial(_standard_jax, *_PLANAR),
    ),
    SameStateCase(
        "native-stage-two-optimization-minimal", _minimal_native, _minimal_jax
    ),
    SameStateCase(
        "native-stage-two-optimization-finitebuild",
        _finite_build_native,
        _finite_build_jax,
    ),
    SameStateCase("native-coil-forces", _coil_forces_native, _coil_forces_jax),
    SameStateCase("native-strain-optimization", _strain_native, _strain_jax),
    SameStateCase("native-qfm", _qfm_native, _qfm_jax, "direct_kernel"),
    SameStateCase("native-boozer", _boozer_native, _boozer_jax),
    SameStateCase("native-boozerqa", _boozer_qa_native, _boozer_qa_jax),
    SameStateCase(
        "native-wireframe-gsco-modular",
        functools.partial(_gsco_native, _GSCO_MODULAR),
        functools.partial(_gsco_jax, _GSCO_MODULAR),
    ),
    SameStateCase(
        "native-wireframe-gsco-sector-saddle",
        functools.partial(_gsco_native, _GSCO_SADDLE),
        functools.partial(_gsco_jax, _GSCO_SADDLE),
    ),
    SameStateCase(
        "native-wireframe-gsco-multistep",
        functools.partial(_multistep, BiotSavart, bnorm_obj_matrices),
        functools.partial(_multistep, BiotSavartJAX, bnorm_obj_matrices_jax),
    ),
    SameStateCase("native-wireframe-rcls-basic", _rcls_native, _rcls_jax),
    SameStateCase("native-permanent-magnet-simple", _pm_simple_native, _pm_simple_jax),
    SameStateCase("native-permanent-magnet-pm4stell", _pm4stell_native, _pm4stell_jax),
)


@pytest.mark.parametrize("case", CASES, ids=[case.example_id for case in CASES])
def test_mirror_matches_native_at_the_shared_start_state(
    case: SameStateCase,
    parity_lane: str,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    parity_device(parity_lane)  # skips a lane this host does not have
    native = case.native()
    enable_strict_parity_backend(monkeypatch, request, parity_lane, precision="fp64")
    mirror = case.jax()

    assert set(mirror) == set(native)
    for observable, native_value in native.items():
        rtol, atol = _tolerance(case.tolerance_bucket, observable)
        np.testing.assert_allclose(
            native_value,
            mirror[observable],
            rtol=rtol,
            atol=atol,
            err_msg=f"{case.example_id}:{observable}",
        )
