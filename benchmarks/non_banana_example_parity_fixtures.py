"""Fixture builders for the non-banana example CPU C++/JAX parity harness.

This module defines the immutable fixture specifications listed in
``docs/non_banana_example_cpp_jax_cpu_parity_plan_2026-05-12.md`` and
provides constructors that return per-lane mutable objects from a shared
spec. CPU and JAX lanes are constructed as independent instances. No
optimizer execution, no VTK side effects, and no GPU code paths are
performed here.

The fixture registry covers every fixture ID enumerated by the plan.
Supported fixtures expose ``build_lanes`` that returns CPU and JAX
artifacts. Unsupported / CPU-only / support-gate fixtures expose
``build_lanes`` that raises a :class:`FixtureNotSupportedError` carrying
the classification reason; the benchmark records that information in the
output artifact rather than failing the run.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_FILES = REPO_ROOT / "tests" / "test_files"
EXAMPLES = REPO_ROOT / "examples"

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Classifications


SUPPORTED = "supported"
UNSUPPORTED_NATIVE_JAX = "unsupported_native_jax"
SUPPORT_GATE = "support_gate"
CPU_ONLY = "cpu_only_diagnostic"
OUT_OF_SCOPE = "out_of_scope"


class FixtureNotSupportedError(RuntimeError):
    """Raised by build_lanes() when a fixture has no native JAX lane.

    The benchmark records the message as the fixture's classification
    reason; it is not treated as a parity failure.
    """


# ---------------------------------------------------------------------------
# Data containers


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    source_example: str
    classification: str
    classification_reason: str = ""
    inputs: Mapping[str, Any] = field(default_factory=dict)
    # ``fixture_kind`` selects the comparison surface in the harness.
    # ``"biot_savart_squared_flux"`` (default) covers field B, B·n, surface
    # geometry, and SquaredFlux objective + gradient.
    # ``"boozer_surface_fixed_state"`` selects fixed-state Boozer residual +
    # labels (Area/Volume/ToroidalFlux) at an unsolved (iota, G, surface)
    # triplet — no inner solve is run.
    fixture_kind: str = "biot_savart_squared_flux"


@dataclass(frozen=True)
class LaneArtifact:
    lane: str
    objective_total: Optional[float]
    objective_native_subtotal: Optional[float]
    components: Mapping[str, float]
    gradient: Optional[np.ndarray]
    gradient_norm: Optional[float]
    active_dof_names: Tuple[str, ...]
    active_dof_hash: str
    fixed_free_mask_hash: str
    native_curve_spec_hashes: Tuple[str, ...]
    surface_point_hash: str
    unit_normal_hash: str
    field_B_hash: str
    field_B_max: float
    field_B_mean: float
    Bdotn_array_hash: str
    Bdotn_max: float
    Bdotn_mean: float
    raw_arrays: Mapping[
        str, np.ndarray
    ]  # for parity comparison (B, gamma, normal, Bdotn)
    timing: Mapping[str, float]


@dataclass(frozen=True)
class FixtureBuild:
    spec: FixtureSpec
    cpu_lane: LaneArtifact
    jax_lane: LaneArtifact
    unsupported_components: Tuple[str, ...]
    # Optional native-supported subproblem evaluators (CPU and JAX) used by
    # the perturbation diagnostic. Both must accept a flat free-DOF vector
    # matching ``cpu_lane.active_dof_names`` / ``jax_lane.active_dof_names``
    # (asserted positionally equal at compare time) and return ``float(J)``
    # for the *native-supported* portion of the objective only — i.e., the
    # unsupported components listed for a partial fixture are excluded from
    # this callable on both sides.
    cpu_native_subproblem_J: Optional[Callable[[np.ndarray], float]] = None
    jax_native_subproblem_J: Optional[Callable[[np.ndarray], float]] = None
    # Initial active free-DOF vector for the native subproblem.
    x0: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Hashing helpers


def _hash_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _hash_mask(mask: Sequence[bool]) -> str:
    payload = np.asarray(mask, dtype=np.uint8).tobytes()
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Native-spec contract validation


def _curve_supports_native_jax(curve) -> bool:
    """Mirror simsopt.field.biotsavart_jax_backend._supports_native_curve_geometry.

    Operates on a *base* curve obtained after unwrapping any RotatedCurve
    wrappers, exactly as BiotSavartJAX does internally.
    """
    if callable(getattr(curve, "to_spec", None)):
        return True
    surface = getattr(curve, "surf", None)
    return (
        surface is not None
        and getattr(curve, "surf_type", None) == "RZ_Fourier"
        and callable(getattr(surface, "surface_spec", None))
    )


def _verify_jax_native_spec_contract(coils) -> Tuple[str, ...]:
    """Return per-base-curve spec hashes or raise if any curve lacks native support.

    Mirrors BiotSavartJAX's internal contract: each coil is unwrapped to its
    base curve (stripping RotatedCurve wrappers) and the base curve is then
    checked against the same ``_supports_native_curve_geometry`` predicate
    the field adapter uses.
    """
    from simsopt.field.biotsavart_jax_backend import _unwrap_coil_curve_and_current

    hashes = []
    seen_base_ids: dict[int, str] = {}
    for idx, coil in enumerate(coils):
        base_curve, _rotmat, _current, _scale = _unwrap_coil_curve_and_current(coil)
        if not _curve_supports_native_jax(base_curve):
            raise FixtureNotSupportedError(
                f"Coil[{idx}] base curve type {type(base_curve).__name__} does "
                "not expose an immutable native JAX spec (to_spec()); "
                "BiotSavartJAX cannot be constructed for this fixture."
            )
        cid = id(base_curve)
        cached = seen_base_ids.get(cid)
        if cached is not None:
            hashes.append(cached)
            continue
        curve_x = np.asarray(base_curve.x, dtype=np.float64)
        payload = (
            type(base_curve).__name__.encode("utf-8")
            + b"|"
            + curve_x.tobytes()
            + b"|"
            + np.asarray(base_curve.quadpoints, dtype=np.float64).tobytes()
        )
        digest = hashlib.sha256(payload).hexdigest()
        seen_base_ids[cid] = digest
        hashes.append(digest)
    return tuple(hashes)


# ---------------------------------------------------------------------------
# Shared lane construction helpers


def _cpu_imports():
    """Import CPU-side simsopt entry points lazily.

    Lazy so the benchmark can probe `jax.config.jax_enable_x64` before
    triggering any heavy imports.
    """
    field_mod = importlib.import_module("simsopt.field")
    objectives_mod = importlib.import_module("simsopt.objectives")
    geo_mod = importlib.import_module("simsopt.geo")
    return field_mod, objectives_mod, geo_mod


def _jax_imports():
    bs_jax_mod = importlib.import_module("simsopt.field.biotsavart_jax_backend")
    flux_jax_mod = importlib.import_module("simsopt.objectives.fluxobjective_jax")
    return bs_jax_mod, flux_jax_mod


def _flatten_components(components: Mapping[str, float]) -> Mapping[str, float]:
    return {str(name): float(value) for name, value in components.items()}


def _build_cpu_lane(
    *,
    surface,
    coils,
    jf_cpu,
    bs_cpu,
    target_array: Optional[np.ndarray],
    extra_components: Mapping[str, float],
    setup_seconds: float,
) -> LaneArtifact:
    """Construct a CPU-lane artifact from already-built simsopt objects."""
    import time

    surface_gamma = np.asarray(surface.gamma(), dtype=np.float64)
    surface_unit_normal = np.asarray(surface.unitnormal(), dtype=np.float64)

    nphi, ntheta = surface_gamma.shape[:2]

    start_exec = time.perf_counter()
    field_B_flat = np.asarray(bs_cpu.B(), dtype=np.float64)
    field_B = field_B_flat.reshape(nphi, ntheta, 3)
    Bdotn = np.sum(field_B * surface_unit_normal, axis=2)
    if target_array is not None:
        Bdotn_for_metric = Bdotn - np.asarray(target_array, dtype=np.float64)
    else:
        Bdotn_for_metric = Bdotn
    j_total = float(jf_cpu.J())
    grad = np.asarray(jf_cpu.dJ(), dtype=np.float64)
    exec_seconds = time.perf_counter() - start_exec

    dof_names = tuple(jf_cpu.dof_names)
    free_mask = np.concatenate(
        [
            np.asarray(o.local_dofs_free_status, dtype=bool)
            for o in jf_cpu.unique_dof_lineage
        ]
    )

    components = dict(extra_components)
    components["SquaredFlux"] = j_total

    raw_arrays = {
        "field_B": field_B,
        "surface_gamma": surface_gamma,
        "surface_unit_normal": surface_unit_normal,
        "Bdotn": Bdotn,
        "Bdotn_target_subtracted": Bdotn_for_metric,
        "gradient": grad,
        "objective_total": np.array([j_total], dtype=np.float64),
    }

    return LaneArtifact(
        lane="cpu_cpp",
        objective_total=j_total,
        objective_native_subtotal=j_total,
        components=_flatten_components(components),
        gradient=grad,
        gradient_norm=float(np.linalg.norm(grad)),
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(np.asarray(jf_cpu.x, dtype=np.float64)),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(surface_gamma),
        unit_normal_hash=_hash_array(surface_unit_normal),
        field_B_hash=_hash_array(field_B),
        field_B_max=float(np.max(np.abs(field_B))),
        field_B_mean=float(np.mean(np.abs(field_B))),
        Bdotn_array_hash=_hash_array(Bdotn_for_metric),
        Bdotn_max=float(np.max(np.abs(Bdotn_for_metric))),
        Bdotn_mean=float(np.mean(np.abs(Bdotn_for_metric))),
        raw_arrays=raw_arrays,
        timing={"setup_s": float(setup_seconds), "execute_s": float(exec_seconds)},
    )


def _build_jax_lane(
    *,
    surface,
    coils,
    bs_jax,
    jf_jax,
    target_array: Optional[np.ndarray],
    extra_components: Mapping[str, float],
    setup_seconds: float,
) -> LaneArtifact:
    """Construct a JAX-lane artifact from already-built BiotSavartJAX/SquaredFluxJAX."""
    import time
    import jax

    native_spec_hashes = _verify_jax_native_spec_contract(coils)

    surface_gamma = np.asarray(surface.gamma(), dtype=np.float64)
    surface_unit_normal = np.asarray(surface.unitnormal(), dtype=np.float64)
    nphi, ntheta = surface_gamma.shape[:2]

    start_compile = time.perf_counter()
    j_total_first = float(jf_jax.J())
    jax.block_until_ready(np.float64(j_total_first))
    compile_plus_first_s = time.perf_counter() - start_compile

    start_exec = time.perf_counter()
    # Force fresh evaluation (clear cache) to time steady-state.
    jf_jax.new_x = True
    j_total = float(jf_jax.J())
    jf_jax.new_x = True
    grad = np.asarray(jf_jax.dJ(), dtype=np.float64)
    jax.block_until_ready(grad)
    exec_seconds = time.perf_counter() - start_exec

    field_B_flat = np.asarray(bs_jax.B(), dtype=np.float64)
    field_B = field_B_flat.reshape(nphi, ntheta, 3)
    Bdotn = np.sum(field_B * surface_unit_normal, axis=2)
    if target_array is not None:
        Bdotn_for_metric = Bdotn - np.asarray(target_array, dtype=np.float64)
    else:
        Bdotn_for_metric = Bdotn

    dof_names = tuple(jf_jax.dof_names)
    free_mask = np.concatenate(
        [
            np.asarray(o.local_dofs_free_status, dtype=bool)
            for o in jf_jax.unique_dof_lineage
        ]
    )

    components = dict(extra_components)
    components["SquaredFluxJAX"] = j_total

    raw_arrays = {
        "field_B": field_B,
        "surface_gamma": surface_gamma,
        "surface_unit_normal": surface_unit_normal,
        "Bdotn": Bdotn,
        "Bdotn_target_subtracted": Bdotn_for_metric,
        "gradient": grad,
        "objective_total": np.array([j_total], dtype=np.float64),
    }

    return LaneArtifact(
        lane="jax_cpu",
        objective_total=j_total,
        objective_native_subtotal=j_total,
        components=_flatten_components(components),
        gradient=grad,
        gradient_norm=float(np.linalg.norm(grad)),
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(np.asarray(jf_jax.x, dtype=np.float64)),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=native_spec_hashes,
        surface_point_hash=_hash_array(surface_gamma),
        unit_normal_hash=_hash_array(surface_unit_normal),
        field_B_hash=_hash_array(field_B),
        field_B_max=float(np.max(np.abs(field_B))),
        field_B_mean=float(np.mean(np.abs(field_B))),
        Bdotn_array_hash=_hash_array(Bdotn_for_metric),
        Bdotn_max=float(np.max(np.abs(Bdotn_for_metric))),
        Bdotn_mean=float(np.mean(np.abs(Bdotn_for_metric))),
        raw_arrays=raw_arrays,
        timing={
            "setup_s": float(setup_seconds),
            "compile_plus_first_s": float(compile_plus_first_s),
            "execute_s": float(exec_seconds),
        },
    )


# ---------------------------------------------------------------------------
# Phase 1 — P0 minimal Stage-II fixture


def _build_minimal_stage2_state():
    """Recreate the exact initial state of stage_two_optimization_minimal.py."""
    import time

    start_setup = time.perf_counter()
    field_mod, objectives_mod, geo_mod = _cpu_imports()
    SurfaceRZFourier = geo_mod.SurfaceRZFourier
    create_equally_spaced_curves = geo_mod.create_equally_spaced_curves
    CurveLength = geo_mod.CurveLength
    Current = field_mod.Current
    coils_via_symmetries = field_mod.coils_via_symmetries
    BiotSavart = field_mod.BiotSavart
    SquaredFlux = objectives_mod.SquaredFlux
    QuadraticPenalty = objectives_mod.QuadraticPenalty

    nphi = 32
    ntheta = 32
    ncoils = 4
    R0 = 1.0
    R1 = 0.5
    order = 5
    length_target = 18.0
    length_weight = 1.0

    filename = TESTS_FILES / "input.LandremanPaul2021_QA"
    surface = SurfaceRZFourier.from_vmec_input(
        str(filename),
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )

    base_curves = create_equally_spaced_curves(
        ncoils,
        surface.nfp,
        stellsym=True,
        R0=R0,
        R1=R1,
        order=order,
    )
    base_currents = [Current(1.0) * 1e5 for _ in range(ncoils)]
    base_currents[0].fix_all()

    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(surface.gamma().reshape((-1, 3)))

    jf_cpu = SquaredFlux(surface, bs_cpu)
    jls = [CurveLength(c) for c in base_curves]
    length_penalty = QuadraticPenalty(sum(jls), length_target, "max")
    jf_full = jf_cpu + length_weight * length_penalty

    length_penalty_value = float(length_penalty.J())
    cpu_total_value = float(jf_full.J())

    extra_components_cpu = {
        "SquaredFlux": float(jf_cpu.J()),
        "QuadraticPenalty_over_sum_CurveLength_max": length_penalty_value,
        "JF_total_cpu": cpu_total_value,
    }
    setup_seconds_cpu = time.perf_counter() - start_setup

    cpu_lane = _build_cpu_lane(
        surface=surface,
        coils=coils,
        jf_cpu=jf_cpu,
        bs_cpu=bs_cpu,
        target_array=None,
        extra_components=extra_components_cpu,
        setup_seconds=setup_seconds_cpu,
    )

    # Build the JAX lane from the same coils. Build new coil objects to keep
    # cpu and jax fields/objectives strictly independent.
    start_jax_setup = time.perf_counter()
    base_curves_jax = create_equally_spaced_curves(
        ncoils,
        surface.nfp,
        stellsym=True,
        R0=R0,
        R1=R1,
        order=order,
    )
    base_currents_jax = [Current(1.0) * 1e5 for _ in range(ncoils)]
    base_currents_jax[0].fix_all()
    coils_jax = coils_via_symmetries(
        base_curves_jax, base_currents_jax, surface.nfp, True
    )

    bs_jax_mod, flux_jax_mod = _jax_imports()
    BiotSavartJAX = bs_jax_mod.BiotSavartJAX
    SquaredFluxJAX = flux_jax_mod.SquaredFluxJAX

    bs_jax = BiotSavartJAX(coils_jax)
    jf_jax = SquaredFluxJAX(surface, bs_jax)
    setup_seconds_jax = time.perf_counter() - start_jax_setup
    jax_lane = _build_jax_lane(
        surface=surface,
        coils=coils_jax,
        bs_jax=bs_jax,
        jf_jax=jf_jax,
        target_array=None,
        extra_components={},
        setup_seconds=setup_seconds_jax,
    )

    # Native-subproblem evaluators for the perturbation diagnostic. Both
    # callables share the same free-DOF basis (SquaredFlux/SquaredFluxJAX
    # only) — i.e., the unsupported length penalty is intentionally not
    # included so the comparison stays on the native-supported portion of J.
    x0 = np.asarray(jf_cpu.x, dtype=np.float64).copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        jf_cpu.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_cpu.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jf_jax.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_jax.J())

    spec = MINIMAL_STAGE2_FLUX_LENGTH_GAP_SPEC
    return FixtureBuild(
        spec=spec,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=("QuadraticPenalty_over_sum_CurveLength_max",),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
    )


# ---------------------------------------------------------------------------
# Phase 3 — P1 full Stage-II composite fixture
#
# Source: examples/2_Intermediate/stage_two_optimization.py
#
# Native-JAX support today is limited to ``SquaredFlux`` / ``SquaredFluxJAX``
# on the field side. The CPU lane mirrors the example's full composite for
# documentation, but the JAX lane only runs the native-supported portion;
# the remaining components (curve length / distance / curvature penalties
# and the per-curve max-quadratic curvature penalty) are listed in
# ``unsupported_components`` and excluded from the gradient comparison
# basis. Verdict is therefore always ``"partial"`` while those wrappers
# remain CPU-only.


def _build_full_stage2_composite():
    """Recreate the initial state of stage_two_optimization.py.

    Plan parameters (verbatim from the upstream example): ``ncoils=4``,
    ``R0=1.0``, ``R1=0.5``, ``order=5``, ``LENGTH_WEIGHT=1e-6``,
    ``CC_THRESHOLD=0.1``, ``CC_WEIGHT=1000``, ``CS_THRESHOLD=0.3``,
    ``CS_WEIGHT=10``, ``CURVATURE_THRESHOLD=5``, ``CURVATURE_WEIGHT=1e-6``,
    ``MSC_THRESHOLD=5``, ``MSC_WEIGHT=1e-6``, ``nphi=32``, ``ntheta=32``,
    surface ``input.LandremanPaul2021_QA`` half-period.
    """
    import time

    start_setup = time.perf_counter()
    field_mod, objectives_mod, geo_mod = _cpu_imports()
    SurfaceRZFourier = geo_mod.SurfaceRZFourier
    create_equally_spaced_curves = geo_mod.create_equally_spaced_curves
    CurveLength = geo_mod.CurveLength
    CurveCurveDistance = geo_mod.CurveCurveDistance
    CurveSurfaceDistance = geo_mod.CurveSurfaceDistance
    LpCurveCurvature = geo_mod.LpCurveCurvature
    MeanSquaredCurvature = geo_mod.MeanSquaredCurvature
    Current = field_mod.Current
    coils_via_symmetries = field_mod.coils_via_symmetries
    BiotSavart = field_mod.BiotSavart
    SquaredFlux = objectives_mod.SquaredFlux
    QuadraticPenalty = objectives_mod.QuadraticPenalty

    nphi = 32
    ntheta = 32
    ncoils = 4
    R0 = 1.0
    R1 = 0.5
    order = 5
    LENGTH_WEIGHT = 1e-6
    CC_THRESHOLD = 0.1
    CC_WEIGHT = 1000
    CS_THRESHOLD = 0.3
    CS_WEIGHT = 10
    CURVATURE_THRESHOLD = 5.0
    CURVATURE_WEIGHT = 1e-6
    MSC_THRESHOLD = 5
    MSC_WEIGHT = 1e-6

    filename = TESTS_FILES / "input.LandremanPaul2021_QA"
    surface = SurfaceRZFourier.from_vmec_input(
        str(filename),
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )

    base_curves = create_equally_spaced_curves(
        ncoils, surface.nfp, stellsym=True, R0=R0, R1=R1, order=order
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    base_currents[0].fix_all()

    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(surface.gamma().reshape((-1, 3)))

    jf_cpu_sf = SquaredFlux(surface, bs_cpu)
    curves = [c.curve for c in coils]
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, surface, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    msc_quadratic_terms = [QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs]
    jf_full = (
        jf_cpu_sf
        + LENGTH_WEIGHT * sum(Jls)
        + CC_WEIGHT * Jccdist
        + CS_WEIGHT * Jcsdist
        + CURVATURE_WEIGHT * sum(Jcs)
        + MSC_WEIGHT * sum(msc_quadratic_terms)
    )

    squared_flux_value = float(jf_cpu_sf.J())
    sum_length_value = float(sum(J.J() for J in Jls))
    ccdist_value = float(Jccdist.J())
    csdist_value = float(Jcsdist.J())
    curvature_sum_value = float(sum(J.J() for J in Jcs))
    msc_quadratic_sum_value = float(sum(J.J() for J in msc_quadratic_terms))
    composite_total_value = float(jf_full.J())

    # Plan §"Math, Physics, And Computation Gates" requires composite
    # objective reporting to record BOTH raw component values (before
    # weights) AND weighted component values, plus the composite total
    # ``JF_total_cpu``. The raw entries below carry ``_raw`` suffixes; the
    # ``_weighted`` entries reproduce the weights applied by the upstream
    # example (LENGTH_WEIGHT * sum_CurveLength, CC_WEIGHT * ccdist, etc.).
    extra_components_cpu = {
        "SquaredFlux": squared_flux_value,
        "sum_CurveLength_raw": sum_length_value,
        "CurveCurveDistance_raw": ccdist_value,
        "CurveSurfaceDistance_raw": csdist_value,
        "sum_LpCurveCurvature_raw": curvature_sum_value,
        "sum_QuadraticPenalty_MeanSquaredCurvature_max_raw": msc_quadratic_sum_value,
        "sum_CurveLength_weighted": LENGTH_WEIGHT * sum_length_value,
        "CurveCurveDistance_weighted": CC_WEIGHT * ccdist_value,
        "CurveSurfaceDistance_weighted": CS_WEIGHT * csdist_value,
        "sum_LpCurveCurvature_weighted": CURVATURE_WEIGHT * curvature_sum_value,
        "sum_QuadraticPenalty_MeanSquaredCurvature_max_weighted": (
            MSC_WEIGHT * msc_quadratic_sum_value
        ),
        "JF_total_cpu": composite_total_value,
    }
    setup_seconds_cpu = time.perf_counter() - start_setup

    cpu_lane = _build_cpu_lane(
        surface=surface,
        coils=coils,
        jf_cpu=jf_cpu_sf,
        bs_cpu=bs_cpu,
        target_array=None,
        extra_components=extra_components_cpu,
        setup_seconds=setup_seconds_cpu,
    )

    # JAX lane — build independent coils so neither lane mutates the other's
    # Optimizable tree. SquaredFluxJAX is the only native-supported component
    # of the full composite; the other terms are listed in
    # ``unsupported_components`` and excluded from the gradient comparison.
    start_jax_setup = time.perf_counter()
    base_curves_jax = create_equally_spaced_curves(
        ncoils, surface.nfp, stellsym=True, R0=R0, R1=R1, order=order
    )
    base_currents_jax = [Current(1e5) for _ in range(ncoils)]
    base_currents_jax[0].fix_all()
    coils_jax = coils_via_symmetries(
        base_curves_jax, base_currents_jax, surface.nfp, True
    )

    bs_jax_mod, flux_jax_mod = _jax_imports()
    BiotSavartJAX = bs_jax_mod.BiotSavartJAX
    SquaredFluxJAX = flux_jax_mod.SquaredFluxJAX

    bs_jax = BiotSavartJAX(coils_jax)
    jf_jax = SquaredFluxJAX(surface, bs_jax)
    setup_seconds_jax = time.perf_counter() - start_jax_setup

    jax_lane = _build_jax_lane(
        surface=surface,
        coils=coils_jax,
        bs_jax=bs_jax,
        jf_jax=jf_jax,
        target_array=None,
        extra_components={},
        setup_seconds=setup_seconds_jax,
    )

    x0 = np.asarray(jf_cpu_sf.x, dtype=np.float64).copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        jf_cpu_sf.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_cpu_sf.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jf_jax.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_jax.J())

    return FixtureBuild(
        spec=FULL_STAGE2_COMPOSITE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(
            "sum_CurveLength",
            "CurveCurveDistance",
            "CurveSurfaceDistance",
            "sum_LpCurveCurvature",
            "sum_QuadraticPenalty_MeanSquaredCurvature_max",
        ),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
    )


# ---------------------------------------------------------------------------
# Phase 4 — P2 planar Stage-II composite fixture
#
# Source: examples/2_Intermediate/stage_two_optimization_planar_coils.py
#
# Same partial-supported approach as Phase 3 with planar coils. The
# ``CurvePlanarFourier`` family exposes ``to_spec()`` (verified via the
# native-spec contract check inside ``_build_jax_lane``), so the JAX lane
# constructs successfully. ``LinkingNumber``, the planar Stage-II length /
# distance / curvature penalties, and the planar-fixture-specific
# quadratic-penalty wrapping of the length sum all stay CPU-only and are
# reported in ``unsupported_components``.


def _build_planar_stage2_composite():
    import time

    start_setup = time.perf_counter()
    field_mod, objectives_mod, geo_mod = _cpu_imports()
    SurfaceRZFourier = geo_mod.SurfaceRZFourier
    create_equally_spaced_planar_curves = geo_mod.create_equally_spaced_planar_curves
    CurveLength = geo_mod.CurveLength
    CurveCurveDistance = geo_mod.CurveCurveDistance
    CurveSurfaceDistance = geo_mod.CurveSurfaceDistance
    LpCurveCurvature = geo_mod.LpCurveCurvature
    MeanSquaredCurvature = geo_mod.MeanSquaredCurvature
    LinkingNumber = geo_mod.LinkingNumber
    Current = field_mod.Current
    coils_via_symmetries = field_mod.coils_via_symmetries
    BiotSavart = field_mod.BiotSavart
    SquaredFlux = objectives_mod.SquaredFlux
    QuadraticPenalty = objectives_mod.QuadraticPenalty

    nphi = 32
    ntheta = 32
    ncoils = 4
    R0 = 1.0
    R1 = 0.5
    order = 5
    LENGTH_WEIGHT = 10.0
    LENGTH_QP_TARGET = 2.6 * ncoils  # mirrors upstream planar example
    CC_THRESHOLD = 0.08
    CC_WEIGHT = 1000
    CS_THRESHOLD = 0.12
    CS_WEIGHT = 10
    CURVATURE_THRESHOLD = 10.0
    CURVATURE_WEIGHT = 1e-6
    MSC_THRESHOLD = 10
    MSC_WEIGHT = 1e-6

    filename = TESTS_FILES / "input.LandremanPaul2021_QA"
    surface = SurfaceRZFourier.from_vmec_input(
        str(filename),
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )

    base_curves = create_equally_spaced_planar_curves(
        ncoils, surface.nfp, stellsym=True, R0=R0, R1=R1, order=order
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    base_currents[0].fix_all()

    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(surface.gamma().reshape((-1, 3)))

    jf_cpu_sf = SquaredFlux(surface, bs_cpu)
    curves = [c.curve for c in coils]
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
    Jcsdist = CurveSurfaceDistance(curves, surface, CS_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    msc_quadratic_terms = [QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs]
    length_quadratic_penalty = QuadraticPenalty(sum(Jls), LENGTH_QP_TARGET)
    linkNum = LinkingNumber(curves)

    jf_full = (
        jf_cpu_sf
        + LENGTH_WEIGHT * length_quadratic_penalty
        + CC_WEIGHT * Jccdist
        + CS_WEIGHT * Jcsdist
        + CURVATURE_WEIGHT * sum(Jcs)
        + MSC_WEIGHT * sum(msc_quadratic_terms)
        + linkNum
    )

    squared_flux_value = float(jf_cpu_sf.J())
    length_qp_value = float(length_quadratic_penalty.J())
    ccdist_value = float(Jccdist.J())
    csdist_value = float(Jcsdist.J())
    curvature_sum_value = float(sum(J.J() for J in Jcs))
    msc_quadratic_sum_value = float(sum(J.J() for J in msc_quadratic_terms))
    link_number_value = float(linkNum.J())
    composite_total_value = float(jf_full.J())

    # Plan §"Math, Physics, And Computation Gates" requires composite
    # objective reporting to record raw component values, weighted
    # component values, and the composite total. Planar fixture weights:
    # LENGTH_WEIGHT * length_quadratic_penalty, CC_WEIGHT * ccdist,
    # CS_WEIGHT * csdist, CURVATURE_WEIGHT * curvature_sum,
    # MSC_WEIGHT * msc_quadratic_sum, and LinkingNumber has no weight
    # multiplier (it enters the composite with weight 1).
    extra_components_cpu = {
        "SquaredFlux": squared_flux_value,
        "QuadraticPenalty_over_sum_CurveLength_identity_raw": length_qp_value,
        "CurveCurveDistance_raw": ccdist_value,
        "CurveSurfaceDistance_raw": csdist_value,
        "sum_LpCurveCurvature_raw": curvature_sum_value,
        "sum_QuadraticPenalty_MeanSquaredCurvature_identity_raw": msc_quadratic_sum_value,
        "LinkingNumber_raw": link_number_value,
        "QuadraticPenalty_over_sum_CurveLength_identity_weighted": (
            LENGTH_WEIGHT * length_qp_value
        ),
        "CurveCurveDistance_weighted": CC_WEIGHT * ccdist_value,
        "CurveSurfaceDistance_weighted": CS_WEIGHT * csdist_value,
        "sum_LpCurveCurvature_weighted": CURVATURE_WEIGHT * curvature_sum_value,
        "sum_QuadraticPenalty_MeanSquaredCurvature_identity_weighted": (
            MSC_WEIGHT * msc_quadratic_sum_value
        ),
        "LinkingNumber_weighted": link_number_value,
        "JF_total_cpu": composite_total_value,
    }
    setup_seconds_cpu = time.perf_counter() - start_setup

    cpu_lane = _build_cpu_lane(
        surface=surface,
        coils=coils,
        jf_cpu=jf_cpu_sf,
        bs_cpu=bs_cpu,
        target_array=None,
        extra_components=extra_components_cpu,
        setup_seconds=setup_seconds_cpu,
    )

    # JAX lane: build independent planar coils so the JAX field adapter
    # owns disjoint Curve/Current Optimizable nodes.
    start_jax_setup = time.perf_counter()
    base_curves_jax = create_equally_spaced_planar_curves(
        ncoils, surface.nfp, stellsym=True, R0=R0, R1=R1, order=order
    )
    base_currents_jax = [Current(1e5) for _ in range(ncoils)]
    base_currents_jax[0].fix_all()
    coils_jax = coils_via_symmetries(
        base_curves_jax, base_currents_jax, surface.nfp, True
    )

    bs_jax_mod, flux_jax_mod = _jax_imports()
    BiotSavartJAX = bs_jax_mod.BiotSavartJAX
    SquaredFluxJAX = flux_jax_mod.SquaredFluxJAX

    bs_jax = BiotSavartJAX(coils_jax)
    jf_jax = SquaredFluxJAX(surface, bs_jax)
    setup_seconds_jax = time.perf_counter() - start_jax_setup

    jax_lane = _build_jax_lane(
        surface=surface,
        coils=coils_jax,
        bs_jax=bs_jax,
        jf_jax=jf_jax,
        target_array=None,
        extra_components={},
        setup_seconds=setup_seconds_jax,
    )

    x0 = np.asarray(jf_cpu_sf.x, dtype=np.float64).copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        jf_cpu_sf.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_cpu_sf.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jf_jax.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_jax.J())

    return FixtureBuild(
        spec=PLANAR_STAGE2_COMPOSITE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(
            "QuadraticPenalty_over_sum_CurveLength_identity",
            "CurveCurveDistance",
            "CurveSurfaceDistance",
            "sum_LpCurveCurvature",
            "sum_QuadraticPenalty_MeanSquaredCurvature_identity",
            "LinkingNumber",
        ),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
    )


# ---------------------------------------------------------------------------
# Phase 2 — P1 saved CWS local-flux fixture


def _build_cws_saved_local_flux(*, nfp: int):
    """Load the CWS saved BiotSavart artifact and compare local SquaredFlux."""
    import time

    start_setup = time.perf_counter()
    field_mod, objectives_mod, geo_mod = _cpu_imports()
    SurfaceRZFourier = geo_mod.SurfaceRZFourier
    SquaredFlux = objectives_mod.SquaredFlux
    BiotSavart = field_mod.BiotSavart
    load_module = importlib.import_module("simsopt._core")
    simsopt_load = load_module.load

    if nfp == 2:
        case_dir = "optimization_cws_singlestage_nfp2_QA_ncoils3_axiTorus"
        coils_filename = "biot_savart_opt_maxmode3.json"
        vmec_filename = "input.maxmode3"
        spec = CWS_SAVED_LOCAL_FLUX_NFP2_SPEC
    elif nfp == 3:
        case_dir = "optimization_cws_singlestage_nfp3_QA_ncoils4_axiTorus"
        coils_filename = "biot_savart_opt_maxmode4.json"
        vmec_filename = "input.maxmode4"
        spec = CWS_SAVED_LOCAL_FLUX_NFP3_SPEC
    else:
        raise ValueError(f"Unsupported CWS nfp: {nfp}")

    coils_path = EXAMPLES / "3_Advanced" / case_dir / "coils" / coils_filename
    vmec_path = EXAMPLES / "3_Advanced" / case_dir / vmec_filename
    if not coils_path.exists() or not vmec_path.exists():
        raise FixtureNotSupportedError(
            f"CWS artifact missing for nfp={nfp}; expected {coils_path} and "
            f"{vmec_path}."
        )

    surface = SurfaceRZFourier.from_vmec_input(
        str(vmec_path),
        range="full torus",
        nphi=64,
        ntheta=32,
    )

    def _load_independent_coil_list():
        """Load a fresh BiotSavart from disk and return its independent coil list.

        Calling ``simsopt.load`` twice gives two disjoint coil trees so the
        CPU and JAX field adapters never share mutable Curve/Current
        Optimizable nodes. This honors the plan's lane-independence rule
        even though both lanes deserialize from the same JSON artifact.
        """
        try:
            loaded = simsopt_load(str(coils_path))
        except TypeError as exc:
            if "CurveCWSFourier" not in str(exc):
                raise
            # Upstream simsopt JSON loader currently fails to reconstruct
            # ``CurveCWSFourier`` instances when handing them to the
            # simsoptpp ``Coil`` constructor. The plan requires fail-closed
            # classification: report the fixture as unsupported rather than
            # allowing a silent partial pass.
            raise FixtureNotSupportedError(
                "Upstream simsopt.load() cannot reconstruct CurveCWSFourier "
                f"from {coils_path.name}: {exc}. Native JAX parity for "
                "this saved-CWS artifact is gated on fixing the upstream "
                "JSON deserialization of CurveCWSFourier (out of scope for "
                "this non-banana parity plan)."
            ) from exc
        return list(loaded.coils)

    coils_cpu = _load_independent_coil_list()
    bs_cpu = BiotSavart(coils_cpu)
    bs_cpu.set_points(surface.gamma().reshape((-1, 3)))
    jf_cpu = SquaredFlux(surface, bs_cpu, definition="local")

    setup_seconds_cpu = time.perf_counter() - start_setup
    cpu_lane = _build_cpu_lane(
        surface=surface,
        coils=coils_cpu,
        jf_cpu=jf_cpu,
        bs_cpu=bs_cpu,
        target_array=None,
        extra_components={},
        setup_seconds=setup_seconds_cpu,
    )

    # JAX lane: load a *second* coil list from the same artifact so the
    # JAX field adapter owns disjoint Curve/Current Optimizable nodes.
    start_jax_setup = time.perf_counter()
    bs_jax_mod, flux_jax_mod = _jax_imports()
    BiotSavartJAX = bs_jax_mod.BiotSavartJAX
    SquaredFluxJAX = flux_jax_mod.SquaredFluxJAX

    coils_jax = _load_independent_coil_list()
    bs_jax = BiotSavartJAX(coils_jax)
    jf_jax = SquaredFluxJAX(surface, bs_jax, definition="local")
    setup_seconds_jax = time.perf_counter() - start_jax_setup

    jax_lane = _build_jax_lane(
        surface=surface,
        coils=coils_jax,
        bs_jax=bs_jax,
        jf_jax=jf_jax,
        target_array=None,
        extra_components={},
        setup_seconds=setup_seconds_jax,
    )

    x0 = np.asarray(jf_cpu.x, dtype=np.float64).copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        jf_cpu.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_cpu.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jf_jax.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_jax.J())

    return FixtureBuild(
        spec=spec,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
    )


# ---------------------------------------------------------------------------
# Phase 6 — P2 Boozer surface fixed-state residual + label fixture
#
# Source: examples/2_Intermediate/boozer.py
#
# Fixed-state parity: build the NCSX initial guess (tube-around-axis surface,
# iota=-0.4, G0 from coil currents) and compare CPU
# ``boozer_surface_residual`` against the JAX ``boozer_residual_vector``
# kernel at the same unsolved state. The CPU and JAX sides also compute
# Area/Volume/ToroidalFlux labels independently for parity. No BFGS/LM solve
# is executed — the comparison is purely on the pre-solve residual and
# label values at a fixed (surface, iota, G) triplet.


def _build_boozer_surface_basic():
    """Build the NCSX boozer fixed-state fixture (residual + labels).

    Mirrors the opening setup of ``examples/2_Intermediate/boozer.py``: the
    NCSX configuration is loaded via ``simsopt.configs.get_data('ncsx')``,
    a ``SurfaceXYZTensorFourier`` is fit to the magnetic axis with minor
    radius 0.10, ``iota=-0.4`` and ``G0`` from the coil current sum are
    used as the residual decision variables. The CPU oracle computes
    ``boozer_surface_residual`` with ``weight_inv_modB=False`` (the
    example default); the JAX side computes ``boozer_residual_vector`` on
    the same surface tangents using ``BiotSavartJAX`` for the magnetic
    field. Both sides also compute Area, Volume, and ToroidalFlux labels
    on independent coil trees so neither lane mutates the other's
    Optimizable nodes.
    """
    import time

    start_setup = time.perf_counter()
    field_mod, _objectives_mod, geo_mod = _cpu_imports()
    BiotSavart = field_mod.BiotSavart
    SurfaceXYZTensorFourier = geo_mod.SurfaceXYZTensorFourier
    Area = geo_mod.Area
    Volume = geo_mod.Volume
    ToroidalFlux = geo_mod.ToroidalFlux
    boozer_surface_residual = geo_mod.boozer_surface_residual

    from simsopt.configs import get_data

    # CPU lane data
    base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
    bs_tf_cpu = BiotSavart(bs.coils)
    current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1e-7 / (2.0 * np.pi))
    iota_value = -0.4

    mpol = 5
    ntor = 5
    stellsym = True
    phis = np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False)
    surface_cpu = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface_cpu.fit_to_curve(ma, 0.10, flip_theta=True)

    surface_gamma_cpu = np.asarray(surface_cpu.gamma(), dtype=np.float64)
    surface_unit_normal_cpu = np.asarray(surface_cpu.unitnormal(), dtype=np.float64)
    nphi_cpu, ntheta_cpu = surface_gamma_cpu.shape[:2]

    # CPU field at surface points (used for B parity + B·n bookkeeping).
    bs.set_points(surface_gamma_cpu.reshape(-1, 3))
    field_B_cpu = np.asarray(bs.B(), dtype=np.float64).reshape(nphi_cpu, ntheta_cpu, 3)
    Bdotn_cpu = np.sum(field_B_cpu * surface_unit_normal_cpu, axis=2)

    # CPU residual + labels.
    r_cpu = np.asarray(
        boozer_surface_residual(
            surface_cpu,
            iota_value,
            G0,
            bs,
            derivatives=0,
            weight_inv_modB=False,
        )[0],
        dtype=np.float64,
    )
    area_cpu_value = float(Area(surface_cpu).J())
    volume_cpu_value = float(Volume(surface_cpu).J())
    toroidal_flux_cpu_value = float(ToroidalFlux(surface_cpu, bs_tf_cpu).J())
    setup_seconds_cpu = time.perf_counter() - start_setup

    cpu_components = {
        "boozer_residual_norm": float(np.linalg.norm(r_cpu)),
        "area": area_cpu_value,
        "volume": volume_cpu_value,
        "toroidal_flux": toroidal_flux_cpu_value,
        "iota": float(iota_value),
        "G": float(G0),
    }
    cpu_dof_names = tuple(surface_cpu.dof_names)
    cpu_free_mask = np.asarray(surface_cpu.local_dofs_free_status, dtype=bool)
    cpu_active_dofs = np.asarray(surface_cpu.x, dtype=np.float64)
    cpu_raw_arrays = {
        "field_B": field_B_cpu,
        "surface_gamma": surface_gamma_cpu,
        "surface_unit_normal": surface_unit_normal_cpu,
        "Bdotn": Bdotn_cpu,
        "boozer_residual": r_cpu,
        "area": np.array([area_cpu_value], dtype=np.float64),
        "volume": np.array([volume_cpu_value], dtype=np.float64),
        "toroidal_flux": np.array([toroidal_flux_cpu_value], dtype=np.float64),
    }

    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=None,
        objective_native_subtotal=None,
        components=_flatten_components(cpu_components),
        gradient=None,
        gradient_norm=None,
        active_dof_names=cpu_dof_names,
        active_dof_hash=_hash_array(cpu_active_dofs),
        fixed_free_mask_hash=_hash_mask(cpu_free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(surface_gamma_cpu),
        unit_normal_hash=_hash_array(surface_unit_normal_cpu),
        field_B_hash=_hash_array(field_B_cpu),
        field_B_max=float(np.max(np.abs(field_B_cpu))),
        field_B_mean=float(np.mean(np.abs(field_B_cpu))),
        Bdotn_array_hash=_hash_array(Bdotn_cpu),
        Bdotn_max=float(np.max(np.abs(Bdotn_cpu))),
        Bdotn_mean=float(np.mean(np.abs(Bdotn_cpu))),
        raw_arrays=cpu_raw_arrays,
        timing={"setup_s": float(setup_seconds_cpu), "execute_s": 0.0},
    )

    # JAX lane: rebuild NCSX independently so coil DOF trees stay disjoint.
    start_jax_setup = time.perf_counter()
    bs_jax_mod, _flux_jax_mod = _jax_imports()
    BiotSavartJAX = bs_jax_mod.BiotSavartJAX
    from simsopt.geo.boozer_residual_jax import boozer_residual_vector
    from simsopt.geo.label_constraints_jax import (
        area_jax as area_jax_fn,
        toroidal_flux_jax as toroidal_flux_jax_fn,
        volume_jax as volume_jax_fn,
    )

    (
        base_curves_jax,
        base_currents_jax,
        ma_jax,
        nfp_jax,
        bs_jax_field,
    ) = get_data("ncsx")
    current_sum_jax = nfp_jax * sum(abs(c.get_value()) for c in base_currents_jax)
    G0_jax = 2.0 * np.pi * current_sum_jax * (4.0 * np.pi * 1e-7 / (2.0 * np.pi))

    # The JAX surface DOFs must match the CPU lane positionally (otherwise
    # the boozer_residual comparison is meaningless). Fit_to_curve is
    # deterministic for the same magnetic axis + radius, so the JAX-side
    # surface comes out byte-equal to the CPU side.
    surface_jax = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp_jax,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface_jax.fit_to_curve(ma_jax, 0.10, flip_theta=True)
    surface_gamma_jax = np.asarray(surface_jax.gamma(), dtype=np.float64)
    surface_xphi_jax = np.asarray(surface_jax.gammadash1(), dtype=np.float64)
    surface_xtheta_jax = np.asarray(surface_jax.gammadash2(), dtype=np.float64)
    surface_normal_jax = np.asarray(surface_jax.normal(), dtype=np.float64)
    surface_unit_normal_jax = np.asarray(surface_jax.unitnormal(), dtype=np.float64)
    nphi_jax, ntheta_jax = surface_gamma_jax.shape[:2]

    # Verify native-spec contract on the JAX coils before building B/A.
    coils_jax_independent = list(bs_jax_field.coils)
    native_spec_hashes = _verify_jax_native_spec_contract(coils_jax_independent)

    # JAX B at the surface points (raveled).
    bs_jax_B = BiotSavartJAX(coils_jax_independent)
    bs_jax_B.set_points(surface_gamma_jax.reshape(-1, 3))
    field_B_jax_flat = np.asarray(bs_jax_B.B(), dtype=np.float64)
    field_B_jax = field_B_jax_flat.reshape(nphi_jax, ntheta_jax, 3)

    # JAX residual.
    r_jax = np.asarray(
        boozer_residual_vector(
            G0_jax,
            iota_value,
            field_B_jax,
            surface_xphi_jax,
            surface_xtheta_jax,
            weight_inv_modB=False,
        ),
        dtype=np.float64,
    )

    # JAX labels: Area, Volume use unnormalized surface normal; ToroidalFlux
    # uses A at the idx=0 phi slice (matching the CPU ToroidalFlux default).
    area_jax_value = float(area_jax_fn(surface_normal_jax))
    volume_jax_value = float(volume_jax_fn(surface_gamma_jax, surface_normal_jax))

    # Build a *second* independent BiotSavartJAX for ToroidalFlux to mirror
    # the CPU side's separate ``bs_tf`` (also a separate BiotSavart in the
    # example). The CPU ToroidalFlux uses the idx=0 phi slice of gamma.
    bs_jax_tf = BiotSavartJAX(list(bs_jax_field.coils))
    tf_gamma_slice = surface_gamma_jax[0]
    bs_jax_tf.set_points(np.ascontiguousarray(tf_gamma_slice))
    A_jax_slice = np.asarray(bs_jax_tf.A(), dtype=np.float64)
    toroidal_flux_jax_value = float(
        toroidal_flux_jax_fn(
            A_jax_slice,
            surface_xtheta_jax[0],
            ntheta_jax,
        )
    )
    Bdotn_jax = np.sum(field_B_jax * surface_unit_normal_jax, axis=2)

    setup_seconds_jax = time.perf_counter() - start_jax_setup

    jax_components = {
        "boozer_residual_norm": float(np.linalg.norm(r_jax)),
        "area": area_jax_value,
        "volume": volume_jax_value,
        "toroidal_flux": toroidal_flux_jax_value,
        "iota": float(iota_value),
        "G": float(G0_jax),
    }
    jax_dof_names = tuple(surface_jax.dof_names)
    jax_free_mask = np.asarray(surface_jax.local_dofs_free_status, dtype=bool)
    jax_active_dofs = np.asarray(surface_jax.x, dtype=np.float64)
    jax_raw_arrays = {
        "field_B": field_B_jax,
        "surface_gamma": surface_gamma_jax,
        "surface_unit_normal": surface_unit_normal_jax,
        "Bdotn": Bdotn_jax,
        "boozer_residual": r_jax,
        "area": np.array([area_jax_value], dtype=np.float64),
        "volume": np.array([volume_jax_value], dtype=np.float64),
        "toroidal_flux": np.array([toroidal_flux_jax_value], dtype=np.float64),
    }

    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=None,
        objective_native_subtotal=None,
        components=_flatten_components(jax_components),
        gradient=None,
        gradient_norm=None,
        active_dof_names=jax_dof_names,
        active_dof_hash=_hash_array(jax_active_dofs),
        fixed_free_mask_hash=_hash_mask(jax_free_mask),
        native_curve_spec_hashes=native_spec_hashes,
        surface_point_hash=_hash_array(surface_gamma_jax),
        unit_normal_hash=_hash_array(surface_unit_normal_jax),
        field_B_hash=_hash_array(field_B_jax),
        field_B_max=float(np.max(np.abs(field_B_jax))),
        field_B_mean=float(np.mean(np.abs(field_B_jax))),
        Bdotn_array_hash=_hash_array(Bdotn_jax),
        Bdotn_max=float(np.max(np.abs(Bdotn_jax))),
        Bdotn_mean=float(np.mean(np.abs(Bdotn_jax))),
        raw_arrays=jax_raw_arrays,
        timing={
            "setup_s": float(setup_seconds_jax),
            "execute_s": 0.0,
        },
    )

    # No gradient-based perturbation diagnostic for this fixture: the
    # comparison is a fixed-state residual+label snapshot, so we leave
    # cpu_native_subproblem_J/jax_native_subproblem_J unset.
    return FixtureBuild(
        spec=BOOZER_SURFACE_BASIC_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
        cpu_native_subproblem_J=None,
        jax_native_subproblem_J=None,
        x0=None,
    )


# ---------------------------------------------------------------------------
# Phase 6 — P2 Boozer QA scalar fixed-solved-state fixture
#
# Source: examples/2_Intermediate/boozerQA.py
#
# Fixed-solved-state parity: solve the Boozer surface once on the CPU side
# via ``BoozerSurface.solve_residual_equation_exactly_newton`` (the same
# Newton solve the upstream example uses), then evaluate the QA scalar values
# at that solved (surface, iota, G) triplet on both CPU and JAX lanes. We do
# not solve a second time on the JAX side because the plan instructions
# explicitly require keeping solver-path differences separate from value
# parity, and because public ``BoozerSurfaceJAX`` wrapper/adjoint parity
# belongs to the dedicated BoozerSurfaceJAX tests, not this copied-state
# fixture.
#
# The CPU lane evaluates the upstream wrappers (``Iotas``, ``MajorRadius``,
# ``NonQuasiSymmetricRatio``, and ``sum(CurveLength)``) on the solved
# BoozerSurface. The JAX lane recomputes the corresponding scalars as pure
# JAX functions:
#
#   * ``Iotas``: returns the solved iota directly. JAX side reads the same
#     scalar; the comparison gates that both lanes report the same iota.
#   * ``MajorRadius``: evaluated through
#     ``surface_major_radius_jax_from_dofs`` on the surface spec built from
#     the solved DOFs.
#   * ``NonQuasiSymmetricRatio``: evaluated through
#     ``_qs_ratio_pure`` on the auxiliary sDIM=20 quadrature grid using a
#     ``BiotSavartJAX`` ``coil_set_spec``.
#   * Length penalty (``sum(CurveLength)``): no native JAX path today;
#     classified in ``unsupported_components``.
#
# Verdict is therefore ``"partial"`` (length penalty unsupported) once all
# native comparisons pass. This fixture does not claim public
# ``BoozerSurfaceJAX`` wrapper or adjoint parity.


def _build_boozer_qa_wrappers():
    """Build the NCSX BoozerQA fixed-solved-state scalar fixture.

    Mirrors the opening setup of ``examples/2_Intermediate/boozerQA.py``:
    the NCSX configuration is loaded via ``simsopt.configs.get_data``,
    a ``SurfaceXYZTensorFourier`` is fit to the magnetic axis with minor
    radius 0.10, the surface is solved with
    ``BoozerSurface.solve_residual_equation_exactly_newton`` (Newton tol
    ``1e-13``, ``maxiter=20``, initial ``iota=-0.406``), and the QA
    upstream CPU wrappers are then evaluated at the solved state and compared
    to JAX scalar helpers over the copied solved-state DOFs.

    Resolution is held at ``mpol=ntor=5`` to keep the fixture cheap
    (Newton converges in ~7 iterations, total fixture build time well
    under one second on CPU). The upstream example uses ``mpol=ntor=6``;
    the lower resolution is justified here because the parity claim is on
    same-state scalar values derived from the converged state and not on
    solver path behavior.
    """
    import time

    start_setup = time.perf_counter()
    field_mod, _objectives_mod, geo_mod = _cpu_imports()
    BiotSavart = field_mod.BiotSavart
    SurfaceXYZTensorFourier = geo_mod.SurfaceXYZTensorFourier
    BoozerSurface = geo_mod.BoozerSurface
    Volume = geo_mod.Volume
    MajorRadius = geo_mod.MajorRadius
    CurveLength = geo_mod.CurveLength
    NonQuasiSymmetricRatio = geo_mod.NonQuasiSymmetricRatio
    Iotas = geo_mod.Iotas

    from simsopt.configs import get_data

    # CPU lane: load NCSX, fit surface, solve via exact Newton, evaluate
    # the four upstream quantities (Iotas, MajorRadius,
    # NonQuasiSymmetricRatio, sum(CurveLength)) at the solved state.
    base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
    current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1e-7 / (2.0 * np.pi))
    iota_initial = -0.406

    mpol = 5
    ntor = 5
    stellsym = True
    minor_radius = 0.10
    sDIM = 20  # NonQuasiSymmetricRatio auxiliary-surface half-resolution.
    phis = np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False)
    surface_cpu = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface_cpu.fit_to_curve(ma, minor_radius, flip_theta=True)

    vol_cpu = Volume(surface_cpu)
    vol_target_cpu = vol_cpu.J()
    boozer_surface_cpu = BoozerSurface(bs, surface_cpu, vol_cpu, vol_target_cpu)
    res_cpu = boozer_surface_cpu.solve_residual_equation_exactly_newton(
        tol=1e-13,
        maxiter=20,
        iota=iota_initial,
        G=G0,
    )
    if not res_cpu["success"]:
        raise FixtureNotSupportedError(
            "Phase 6 boozerQA wrappers fixture: CPU exact Newton solve did "
            f"not converge (iter={res_cpu.get('iter')}, "
            f"residual_norm="
            f"{float(np.linalg.norm(np.asarray(res_cpu.get('residual', []), dtype=np.float64))):.3e}"
            "). The fixture builder requires a converged Newton state."
        )
    solved_iota = float(res_cpu["iota"])
    solved_G = float(res_cpu["G"])

    # Surface geometry at the solved state (for hashing and arrays).
    surface_gamma_cpu = np.asarray(surface_cpu.gamma(), dtype=np.float64)
    surface_unit_normal_cpu = np.asarray(surface_cpu.unitnormal(), dtype=np.float64)
    nphi_cpu, ntheta_cpu = surface_gamma_cpu.shape[:2]

    bs.set_points(surface_gamma_cpu.reshape(-1, 3))
    field_B_cpu = np.asarray(bs.B(), dtype=np.float64).reshape(nphi_cpu, ntheta_cpu, 3)
    Bdotn_cpu = np.sum(field_B_cpu * surface_unit_normal_cpu, axis=2)

    # Upstream CPU wrappers at the solved state. Iotas.J() returns the
    # solved iota; MajorRadius.J() returns the surface major radius;
    # NonQuasiSymmetricRatio uses a fresh BiotSavart (matches the upstream
    # boozerQA.py pattern of ``bs_nonQS = BiotSavart(bs.coils)``).
    iotas_cpu_value = float(Iotas(boozer_surface_cpu).J())
    major_radius_cpu_value = float(MajorRadius(boozer_surface_cpu).J())
    bs_nonQS_cpu = BiotSavart(bs.coils)
    nqs_cpu_value = float(
        NonQuasiSymmetricRatio(boozer_surface_cpu, bs_nonQS_cpu, sDIM=sDIM).J()
    )
    length_sum_cpu_value = float(sum(CurveLength(c).J() for c in base_curves))

    setup_seconds_cpu = time.perf_counter() - start_setup

    cpu_components = {
        "iota": iotas_cpu_value,
        "major_radius": major_radius_cpu_value,
        "nq_symmetric_ratio": nqs_cpu_value,
        "G": solved_G,
        # Length penalty is CPU-only; it is included in the CPU components
        # for traceability but excluded from native cross-lane comparison
        # via the unsupported_components list returned below.
        "sum_CurveLength": length_sum_cpu_value,
    }
    cpu_dof_names = tuple(surface_cpu.dof_names)
    cpu_free_mask = np.asarray(surface_cpu.local_dofs_free_status, dtype=bool)
    cpu_active_dofs = np.asarray(surface_cpu.x, dtype=np.float64)
    cpu_raw_arrays = {
        "field_B": field_B_cpu,
        "surface_gamma": surface_gamma_cpu,
        "surface_unit_normal": surface_unit_normal_cpu,
        "Bdotn": Bdotn_cpu,
        "iota": np.array([iotas_cpu_value], dtype=np.float64),
        "major_radius": np.array([major_radius_cpu_value], dtype=np.float64),
        "nq_symmetric_ratio": np.array([nqs_cpu_value], dtype=np.float64),
    }

    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=None,
        objective_native_subtotal=None,
        components=_flatten_components(cpu_components),
        gradient=None,
        gradient_norm=None,
        active_dof_names=cpu_dof_names,
        active_dof_hash=_hash_array(cpu_active_dofs),
        fixed_free_mask_hash=_hash_mask(cpu_free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(surface_gamma_cpu),
        unit_normal_hash=_hash_array(surface_unit_normal_cpu),
        field_B_hash=_hash_array(field_B_cpu),
        field_B_max=float(np.max(np.abs(field_B_cpu))),
        field_B_mean=float(np.mean(np.abs(field_B_cpu))),
        Bdotn_array_hash=_hash_array(Bdotn_cpu),
        Bdotn_max=float(np.max(np.abs(Bdotn_cpu))),
        Bdotn_mean=float(np.mean(np.abs(Bdotn_cpu))),
        raw_arrays=cpu_raw_arrays,
        timing={"setup_s": float(setup_seconds_cpu), "execute_s": 0.0},
    )

    # JAX lane: rebuild NCSX independently so the JAX coil tree is disjoint
    # from the CPU one; copy the solved surface DOFs over from the CPU
    # lane so both lanes evaluate the wrappers at byte-equal surface state.
    start_jax_setup = time.perf_counter()
    bs_jax_mod, _flux_jax_mod = _jax_imports()
    BiotSavartJAX = bs_jax_mod.BiotSavartJAX
    from simsopt.geo.boozersurface_jax import _generic_surface_scatter_operator
    from simsopt.geo.surfaceobjectives_jax import (
        _qs_ratio_pure,
        surface_major_radius_jax_from_dofs,
    )

    (
        base_curves_jax,
        base_currents_jax,
        ma_jax,
        nfp_jax,
        bs_jax_field,
    ) = get_data("ncsx")
    # Construct the JAX-side surface from byte-equal DOFs so the wrappers
    # see the same converged state. ``set_dofs`` is the public mutation API
    # for SurfaceXYZTensorFourier; the surface_spec() snapshot afterwards
    # is what the pure-JAX scalar helpers consume.
    surface_jax = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nfp=nfp_jax,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface_jax.set_dofs(np.asarray(surface_cpu.get_dofs(), dtype=np.float64))
    surface_gamma_jax = np.asarray(surface_jax.gamma(), dtype=np.float64)
    surface_unit_normal_jax = np.asarray(surface_jax.unitnormal(), dtype=np.float64)
    nphi_jax, ntheta_jax = surface_gamma_jax.shape[:2]

    coils_jax_independent = list(bs_jax_field.coils)
    native_spec_hashes = _verify_jax_native_spec_contract(coils_jax_independent)

    # Native JAX B at the surface points (raveled). Used only for B parity
    # bookkeeping; the wrapper values do not require this B array directly,
    # but it gates that BiotSavartJAX construction succeeded and produces a
    # finite field over the surface.
    bs_jax_B = BiotSavartJAX(coils_jax_independent)
    bs_jax_B.set_points(surface_gamma_jax.reshape(-1, 3))
    field_B_jax_flat = np.asarray(bs_jax_B.B(), dtype=np.float64)
    field_B_jax = field_B_jax_flat.reshape(nphi_jax, ntheta_jax, 3)
    Bdotn_jax = np.sum(field_B_jax * surface_unit_normal_jax, axis=2)

    # Iotas scalar: equal to the solved iota. This comparison gates that both
    # lanes agree on the scalar state reported by the CPU wrapper extraction.
    # It is intentionally a cross-lane sanity check rather than a public
    # BoozerSurfaceJAX wrapper or adjoint check.
    iotas_jax_value = float(solved_iota)

    # MajorRadius scalar: pure JAX over the surface spec built from the
    # solved DOFs.
    surface_spec_jax = surface_jax.surface_spec()
    sdofs_jax = np.asarray(surface_jax.get_dofs(), dtype=np.float64)
    major_radius_jax_value = float(
        surface_major_radius_jax_from_dofs(surface_spec_jax, sdofs_jax)
    )

    # NonQuasiSymmetricRatio scalar: pure JAX through
    # ``_qs_ratio_pure`` on an auxiliary sDIM=20 quadrature grid (matching
    # the upstream NonQuasiSymmetricRatio default and the CPU oracle
    # configuration above). The auxiliary grid is independent of the
    # solved surface quadrature grid.
    aux_phi = np.linspace(0.0, 1.0 / nfp_jax, 2 * sDIM, endpoint=False)
    aux_theta = np.linspace(0.0, 1.0, 2 * sDIM, endpoint=False)
    bs_jax_nonQS = BiotSavartJAX(list(bs_jax_field.coils))
    coil_set_spec_nqs = bs_jax_nonQS.coil_set_spec()
    # ``_qs_ratio_pure`` requires the same surface_kind / scatter_indices
    # contract that BoozerSurfaceJAX would set up internally: for stellsym
    # ``xyztensorfourier`` surfaces, scatter indices are the generic
    # scatter operator built from ``stellsym_scatter_indices`` rather than
    # the raw integer mask used for RZFourier surfaces.
    scatter_indices = (
        _generic_surface_scatter_operator(mpol, ntor) if stellsym else None
    )
    nqs_jax_value = float(
        _qs_ratio_pure(
            sdofs_jax,
            coil_set_spec_nqs,
            quadpoints_phi=aux_phi,
            quadpoints_theta=aux_theta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp_jax,
            stellsym=stellsym,
            scatter_indices=scatter_indices,
            surface_kind="xyztensorfourier",
            axis=0,
        )
    )

    setup_seconds_jax = time.perf_counter() - start_jax_setup

    jax_components = {
        "iota": iotas_jax_value,
        "major_radius": major_radius_jax_value,
        "nq_symmetric_ratio": nqs_jax_value,
        "G": solved_G,
    }
    jax_dof_names = tuple(surface_jax.dof_names)
    jax_free_mask = np.asarray(surface_jax.local_dofs_free_status, dtype=bool)
    jax_active_dofs = np.asarray(surface_jax.x, dtype=np.float64)
    jax_raw_arrays = {
        "field_B": field_B_jax,
        "surface_gamma": surface_gamma_jax,
        "surface_unit_normal": surface_unit_normal_jax,
        "Bdotn": Bdotn_jax,
        "iota": np.array([iotas_jax_value], dtype=np.float64),
        "major_radius": np.array([major_radius_jax_value], dtype=np.float64),
        "nq_symmetric_ratio": np.array([nqs_jax_value], dtype=np.float64),
    }

    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=None,
        objective_native_subtotal=None,
        components=_flatten_components(jax_components),
        gradient=None,
        gradient_norm=None,
        active_dof_names=jax_dof_names,
        active_dof_hash=_hash_array(jax_active_dofs),
        fixed_free_mask_hash=_hash_mask(jax_free_mask),
        native_curve_spec_hashes=native_spec_hashes,
        surface_point_hash=_hash_array(surface_gamma_jax),
        unit_normal_hash=_hash_array(surface_unit_normal_jax),
        field_B_hash=_hash_array(field_B_jax),
        field_B_max=float(np.max(np.abs(field_B_jax))),
        field_B_mean=float(np.mean(np.abs(field_B_jax))),
        Bdotn_array_hash=_hash_array(Bdotn_jax),
        Bdotn_max=float(np.max(np.abs(Bdotn_jax))),
        Bdotn_mean=float(np.mean(np.abs(Bdotn_jax))),
        raw_arrays=jax_raw_arrays,
        timing={
            "setup_s": float(setup_seconds_jax),
            "execute_s": 0.0,
        },
    )

    # The length-penalty wrapper (``sum(CurveLength)`` in upstream
    # ``boozerQA.py``) has no pure-JAX implementation today: CurveLength
    # is a CPU-only Optimizable and the QuadraticPenalty around it
    # depends on the same CPU-only objective. The plan §"Math, Physics,
    # And Computation Gates" requires this term to be listed in
    # unsupported_components rather than evaluated through a CPU
    # substitute on the JAX side, so the verdict will be ``"partial"``
    # once the three native comparisons (iota, major_radius,
    # nq_symmetric_ratio) pass.
    return FixtureBuild(
        spec=BOOZER_QA_WRAPPERS_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=("sum_CurveLength",),
        cpu_native_subproblem_J=None,
        jax_native_subproblem_J=None,
        x0=None,
    )


# ---------------------------------------------------------------------------
# Unsupported / support-gate classification builders


def _raise_unsupported(message: str) -> Callable[[], FixtureBuild]:
    def _factory() -> FixtureBuild:
        raise FixtureNotSupportedError(message)

    return _factory


# ---------------------------------------------------------------------------
# Phase 5 — Position/orientation support gate
#
# Source: examples/1_Simple/optimize_coil_position_orientation.py
#
# The plan requires that the harness build the CPU TF+windowpane coil
# fixture *without* running the optimizer and then emit a precise
# unsupported-native-JAX result because ``OrientedCurveXYZFourier`` does
# not implement ``to_spec()``. We do exactly that here: the CPU fixture
# is materialized so the failure carries the actual missing-spec class
# name and the active DOF list the JAX lane would need to mirror once
# native support lands.


def _build_position_orientation_support_gate_probe():
    """Build the CPU TF+windowpane fixture and probe JAX native-spec support.

    Returns a builder that always raises ``FixtureNotSupportedError`` with
    the precise unsupported curve class. The CPU fixture is constructed
    only for the purpose of producing a detailed support-gate message that
    documents what the JAX lane would need to mirror; the CPU objects are
    not returned because the harness's ``unsupported`` verdict does not
    carry lane data.
    """
    field_mod, objectives_mod, geo_mod = _cpu_imports()
    SurfaceRZFourier = geo_mod.SurfaceRZFourier
    create_equally_spaced_curves = geo_mod.create_equally_spaced_curves
    create_equally_spaced_oriented_curves = (
        geo_mod.create_equally_spaced_oriented_curves
    )
    Current = field_mod.Current
    ScaledCurrent = importlib.import_module("simsopt.field.coil").ScaledCurrent
    coils_via_symmetries = field_mod.coils_via_symmetries
    BiotSavart = field_mod.BiotSavart
    SquaredFlux = objectives_mod.SquaredFlux

    nphi = 32
    ntheta = 32
    n_tf_coils = 4
    n_wp_coils = 2
    R0 = 1.0
    R1 = 0.5

    filename = TESTS_FILES / "input.LandremanPaul2021_QA"
    surface = SurfaceRZFourier.from_vmec_input(
        str(filename),
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )

    base_tf_curves = create_equally_spaced_curves(
        n_tf_coils, surface.nfp, stellsym=True, R0=R0, R1=R1, order=2
    )
    base_wp_curves = create_equally_spaced_oriented_curves(
        n_wp_coils, surface.nfp, R0=(R0 + R1) * 1.01, R1=R1 / 10, Z0=0, order=2
    )
    base_tf_currents = [ScaledCurrent(Current(1.0), 1e5) for _ in range(n_tf_coils)]
    base_wp_currents = [ScaledCurrent(Current(1.0), 1e3) for _ in range(n_wp_coils)]

    # DOF layout mirrors examples/1_Simple/optimize_coil_position_orientation.py
    for curve in base_tf_curves:
        curve.fix_all()
    for curve in base_wp_curves:
        curve.fix_all()
        for xyz in ("x0", "y0", "z0"):
            curve.unfix(xyz)
        for ypr in ("yaw", "pitch", "roll"):
            curve.unfix(ypr)
    for current in base_tf_currents:
        current.unfix_all()
    for current in base_wp_currents:
        current.unfix_all()
    base_tf_currents[0].fix_all()

    tf_coils = coils_via_symmetries(base_tf_curves, base_tf_currents, surface.nfp, True)
    wp_coils = coils_via_symmetries(base_wp_curves, base_wp_currents, surface.nfp, True)
    coils = tf_coils + wp_coils
    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(surface.gamma().reshape((-1, 3)))
    jf_cpu = SquaredFlux(surface, bs_cpu)
    active_dof_names = tuple(jf_cpu.dof_names)

    # Probe the native-spec contract that BiotSavartJAX would enforce. We
    # call the same predicate used internally so the support-gate message
    # mirrors the actual rejection path.
    offending = None
    for coil in coils:
        from simsopt.field.biotsavart_jax_backend import _unwrap_coil_curve_and_current

        base_curve, _rotmat, _current, _scale = _unwrap_coil_curve_and_current(coil)
        if not _curve_supports_native_jax(base_curve):
            offending = type(base_curve).__name__
            break

    if offending is None:
        # Should not happen with current source — included for forward
        # compatibility: if OrientedCurveXYZFourier later acquires
        # ``to_spec()``, this fixture must be flipped to SUPPORTED with a
        # real build_lanes path instead of staying in the support-gate
        # branch. The raise type must be ``FixtureNotSupportedError`` (not
        # a bare ``RuntimeError``) so the harness driver records this as
        # ``verdict='unsupported'`` per the prior commit's fail-closed
        # contract — see ``_evaluate_unsupported_fixture`` in
        # ``non_banana_example_cpp_jax_cpu_parity.py`` and Phase 7's
        # analogous upgrade-path branch below.
        raise FixtureNotSupportedError(
            "Phase 5 support-gate probe found no missing native spec; "
            "OrientedCurveXYZFourier appears to expose to_spec() now. "
            "Flip POSITION_ORIENTATION_FLUX_SUPPORT_GATE_SPEC to SUPPORTED "
            "and implement a real build_lanes constructor."
        )

    raise FixtureNotSupportedError(
        f"Phase 5 support gate: {offending} does not implement to_spec() "
        f"(checked via _supports_native_curve_geometry). BiotSavartJAX "
        f"rejects this curve family until immutable native spec support "
        f"is added. Active free DOFs the JAX lane would mirror once "
        f"supported (count={len(active_dof_names)}): "
        f"{', '.join(active_dof_names[:6])}{'...' if len(active_dof_names) > 6 else ''}."
    )


# ---------------------------------------------------------------------------
# Phase 7 — Finite-build multifilament support gate
#
# Source: examples/3_Advanced/stage_two_optimization_finitebuild.py
#
# Multifilament curves generated by ``create_multifilament_grid`` need a
# native immutable spec on every filament before the JAX field adapter
# can be constructed. We materialize the base curves at low resolution
# and probe ``_supports_native_curve_geometry`` exactly the way
# BiotSavartJAX does internally; if a single filament rejects the
# contract the fixture is reported as a support gate.


def _build_finitebuild_support_gate_probe():
    field_mod, _objectives_mod, geo_mod = _cpu_imports()
    SurfaceRZFourier = geo_mod.SurfaceRZFourier
    create_equally_spaced_curves = geo_mod.create_equally_spaced_curves
    Current = field_mod.Current

    # Try to import the finite-build helpers. If the module is missing
    # entirely, that is itself a support-gate signal.
    try:
        finitebuild_mod = importlib.import_module("simsopt.geo.finitebuild")
    except ImportError as exc:
        raise FixtureNotSupportedError(
            "Phase 7 support gate (finite build): "
            f"simsopt.geo.finitebuild import failed: {exc}. Native JAX "
            "parity for multifilament coils requires both finitebuild and "
            "an immutable spec on every filament curve."
        ) from exc

    create_multifilament_grid = getattr(
        finitebuild_mod, "create_multifilament_grid", None
    )
    coil_mod = importlib.import_module("simsopt.field.coil")
    apply_symmetries_to_curves = getattr(coil_mod, "apply_symmetries_to_curves", None)
    apply_symmetries_to_currents = getattr(
        coil_mod, "apply_symmetries_to_currents", None
    )
    if (
        create_multifilament_grid is None
        or apply_symmetries_to_curves is None
        or apply_symmetries_to_currents is None
    ):
        raise FixtureNotSupportedError(
            "Phase 7 support gate (finite build): missing one of "
            "simsopt.geo.finitebuild.create_multifilament_grid or "
            "simsopt.field.coil.apply_symmetries_to_curve(nt)s in this build."
        )

    ncoils = 3
    R0 = 1.0
    R1 = 0.5
    order = 5
    numfilaments_n = 2
    numfilaments_b = 3
    gapsize_n = 0.02
    gapsize_b = 0.04
    rot_order = 1
    nfil = numfilaments_n * numfilaments_b

    filename = TESTS_FILES / "input.LandremanPaul2021_QA"
    surface = SurfaceRZFourier.from_vmec_input(
        str(filename),
        range="half period",
        nphi=16,
        ntheta=16,
    )

    base_curves = create_equally_spaced_curves(
        ncoils, surface.nfp, stellsym=True, R0=R0, R1=R1, order=order
    )
    base_currents = []
    for i in range(ncoils):
        curr = Current(1.0)
        if i == 0:
            curr.fix_all()
        base_currents.append(curr * (1e5 / nfil))

    # ``create_multifilament_grid`` takes one base curve and returns a list
    # of filament curves; concatenate over all base curves to get the full
    # expanded filament set, mirroring the upstream finitebuild example.
    filament_curves = []
    for c in base_curves:
        filament_curves.extend(
            create_multifilament_grid(
                c,
                numfilaments_n,
                numfilaments_b,
                gapsize_n,
                gapsize_b,
                rotation_order=rot_order,
            )
        )
    filament_currents = []
    for current in base_currents:
        filament_currents.extend([current] * nfil)

    curves_fb = apply_symmetries_to_curves(filament_curves, surface.nfp, True)
    # The current symmetry-expansion is exercised here for completeness, but
    # only the symmetry-expanded curves are probed for native-spec support.
    apply_symmetries_to_currents(filament_currents, surface.nfp, True)

    # Mirror BiotSavartJAX's native-spec check on each symmetry-expanded
    # filament. The field adapter strips ``RotatedCurve`` wrappers before
    # probing ``_supports_native_curve_geometry``, so the probe must do
    # the same: ``RotatedCurve`` itself never exposes ``to_spec()``, but
    # the base curve underneath might. ``CurveFilament`` further wraps a
    # ``FramedCurve(Centroid|Frenet)`` which depends on this build of
    # simsopt for native-spec support.
    from simsopt.geo.curve import RotatedCurve

    offending = None
    for curve in curves_fb:
        base = curve
        while isinstance(base, RotatedCurve):
            base = base.curve
        if not _curve_supports_native_jax(base):
            offending = type(base).__name__
            break

    if offending is not None:
        raise FixtureNotSupportedError(
            f"Phase 7 support gate (finite build): {offending} (one of the "
            f"expanded multifilament curves) does not implement to_spec() "
            "(checked via _supports_native_curve_geometry). Multifilament "
            "BiotSavartJAX construction needs an immutable native spec on "
            "every filament curve."
        )

    # If we get here, every filament passes the native-spec contract.
    # That signals an upgrade path: this fixture should be flipped to
    # SUPPORTED in a follow-up plan that also writes a build_lanes
    # constructor mirroring the upstream finitebuild example.
    raise FixtureNotSupportedError(
        "Phase 7 support gate (finite build): all expanded filament "
        "curves expose native specs, but a full build_lanes constructor "
        "for the finite-build composite (flux + curve length penalty + "
        "MeanSquaredCurvature + MinimumDistance + filament-arclength "
        "variation penalty) has not been implemented in this harness yet."
    )


# ---------------------------------------------------------------------------
# Specs


MINIMAL_STAGE2_FLUX_LENGTH_GAP_SPEC = FixtureSpec(
    fixture_id="minimal_stage2_flux_length_gap",
    source_example="examples/1_Simple/stage_two_optimization_minimal.py",
    classification=SUPPORTED,
    classification_reason=(
        "SquaredFlux/SquaredFluxJAX value/gradient parity is supported. "
        "QuadraticPenalty(sum(CurveLength), 'max') has no native JAX "
        "implementation; the length term is reported in unsupported_components."
    ),
    inputs={
        "ncoils": 4,
        "R0": 1.0,
        "R1": 0.5,
        "order": 5,
        "LENGTH_TARGET": 18.0,
        "LENGTH_WEIGHT": 1.0,
        "nphi": 32,
        "ntheta": 32,
        "vmec_input": "tests/test_files/input.LandremanPaul2021_QA",
        "surface_range": "half period",
        "definition": "quadratic flux",
        "target": None,
    },
)


CWS_SAVED_LOCAL_FLUX_NFP2_SPEC = FixtureSpec(
    fixture_id="cws_saved_local_flux_nfp2",
    source_example="examples/3_Advanced/curves_CWS_example.py",
    classification=SUPPORTED,
    classification_reason=(
        "Saved BiotSavart artifact + local-flux SquaredFluxJAX path is "
        "fully native-JAX supported."
    ),
    inputs={
        "case_dir": "optimization_cws_singlestage_nfp2_QA_ncoils3_axiTorus",
        "coils_file": "biot_savart_opt_maxmode3.json",
        "vmec_input": "input.maxmode3",
        "nfp": 2,
        "nphi": 64,
        "ntheta": 32,
        "surface_range": "full torus",
        "definition": "local",
    },
)


CWS_SAVED_LOCAL_FLUX_NFP3_SPEC = FixtureSpec(
    fixture_id="cws_saved_local_flux_nfp3",
    source_example="examples/3_Advanced/curves_CWS_example.py",
    classification=SUPPORTED,
    classification_reason=(
        "Saved BiotSavart artifact + local-flux SquaredFluxJAX path is "
        "fully native-JAX supported."
    ),
    inputs={
        "case_dir": "optimization_cws_singlestage_nfp3_QA_ncoils4_axiTorus",
        "coils_file": "biot_savart_opt_maxmode4.json",
        "vmec_input": "input.maxmode4",
        "nfp": 3,
        "nphi": 64,
        "ntheta": 32,
        "surface_range": "full torus",
        "definition": "local",
    },
)


FULL_STAGE2_COMPOSITE_SPEC = FixtureSpec(
    fixture_id="full_stage2_composite",
    source_example="examples/2_Intermediate/stage_two_optimization.py",
    classification=SUPPORTED,
    classification_reason=(
        "Composite Stage-II flux portion is native-supported via "
        "SquaredFlux/SquaredFluxJAX. CurveLength, CurveCurveDistance, "
        "CurveSurfaceDistance, LpCurveCurvature, and "
        "QuadraticPenalty(MeanSquaredCurvature, 'max') do not have native "
        "JAX wrappers today; they are listed in unsupported_components and "
        "the verdict is 'partial' until those wrappers exist."
    ),
    inputs={
        "ncoils": 4,
        "R0": 1.0,
        "R1": 0.5,
        "order": 5,
        "LENGTH_WEIGHT": 1e-6,
        "CC_THRESHOLD": 0.1,
        "CC_WEIGHT": 1000,
        "CS_THRESHOLD": 0.3,
        "CS_WEIGHT": 10,
        "CURVATURE_THRESHOLD": 5.0,
        "CURVATURE_WEIGHT": 1e-6,
        "MSC_THRESHOLD": 5,
        "MSC_WEIGHT": 1e-6,
        "nphi": 32,
        "ntheta": 32,
        "vmec_input": "tests/test_files/input.LandremanPaul2021_QA",
        "surface_range": "half period",
        "definition": "quadratic flux",
        "target": None,
    },
)


PLANAR_STAGE2_COMPOSITE_SPEC = FixtureSpec(
    fixture_id="planar_stage2_composite",
    source_example="examples/2_Intermediate/stage_two_optimization_planar_coils.py",
    classification=SUPPORTED,
    classification_reason=(
        "CurvePlanarFourier exposes a native immutable JAX spec, so "
        "BiotSavartJAX + SquaredFluxJAX construct successfully. "
        "LinkingNumber and the shared geometry penalties used by the "
        "planar fixture have no native JAX wrappers today; they are "
        "listed in unsupported_components and the verdict is 'partial' "
        "until those wrappers exist."
    ),
    inputs={
        "ncoils": 4,
        "R0": 1.0,
        "R1": 0.5,
        "order": 5,
        "LENGTH_WEIGHT": 10.0,
        "LENGTH_QP_TARGET": 2.6 * 4,
        "CC_THRESHOLD": 0.08,
        "CC_WEIGHT": 1000,
        "CS_THRESHOLD": 0.12,
        "CS_WEIGHT": 10,
        "CURVATURE_THRESHOLD": 10.0,
        "CURVATURE_WEIGHT": 1e-6,
        "MSC_THRESHOLD": 10,
        "MSC_WEIGHT": 1e-6,
        "nphi": 32,
        "ntheta": 32,
        "vmec_input": "tests/test_files/input.LandremanPaul2021_QA",
        "surface_range": "half period",
        "definition": "quadratic flux",
        "target": None,
    },
)


POSITION_ORIENTATION_FLUX_SUPPORT_GATE_SPEC = FixtureSpec(
    fixture_id="position_orientation_flux_support_gate",
    source_example="examples/1_Simple/optimize_coil_position_orientation.py",
    classification=SUPPORT_GATE,
    classification_reason=(
        "OrientedCurveXYZFourier does not implement to_spec(); BiotSavartJAX "
        "rejects this curve family until immutable native spec support is "
        "added. The probe builds the CPU TF+windowpane fixture without "
        "running the optimizer and records the exact rejecting curve type."
    ),
    inputs={
        "n_tf_coils": 4,
        "n_wp_coils": 2,
        "R0": 1.0,
        "R1": 0.5,
        "tf_curve_order": 2,
        "wp_curve_order": 2,
        "wp_active_dofs": ("x0", "y0", "z0", "yaw", "pitch", "roll"),
        "tf_geometry_fixed": True,
        "wp_geometry_fixed_except_position_orientation": True,
        "tf_current_seed_fixed": True,
        "vmec_input": "tests/test_files/input.LandremanPaul2021_QA",
        "surface_range": "half period",
        "nphi": 32,
        "ntheta": 32,
    },
)


BOOZER_SURFACE_BASIC_SPEC = FixtureSpec(
    fixture_id="boozer_surface_basic",
    source_example="examples/2_Intermediate/boozer.py",
    classification=SUPPORTED,
    classification_reason=(
        "Fixed-state Boozer residual parity at the NCSX initial guess "
        "(tube-around-axis surface, iota=-0.4, G0 from coil currents): "
        "CPU ``boozer_surface_residual`` (weight_inv_modB=False) versus "
        "JAX ``boozer_residual_vector`` via ``BiotSavartJAX``. The label "
        "comparison covers Area and Volume against "
        "``simsopt.geo.label_constraints_jax.{area_jax,volume_jax}`` and "
        "ToroidalFlux against ``toroidal_flux_jax`` consuming the JAX "
        "vector potential A(idx=0 phi slice). No BFGS/LM solve is run; "
        "the comparison is purely on the pre-solve residual vector and "
        "the three label scalars at a fixed (surface, iota, G) triplet."
    ),
    fixture_kind="boozer_surface_fixed_state",
    inputs={
        "config_name": "ncsx",
        "mpol": 5,
        "ntor": 5,
        "stellsym": True,
        "minor_radius": 0.10,
        "fit_to_curve_flip_theta": True,
        "iota_value": -0.4,
        "weight_inv_modB": False,
        "G0_source": "2π * (nfp * Σ|I_k|) * μ0/(2π)",
        "labels": ("area", "volume", "toroidal_flux"),
    },
)


BOOZER_QA_WRAPPERS_SPEC = FixtureSpec(
    fixture_id="boozer_qa_wrappers",
    source_example="examples/2_Intermediate/boozerQA.py",
    classification=SUPPORTED,
    classification_reason=(
        "Fixed-solved-state QA scalar parity at the NCSX boozerQA "
        "configuration (NCSX coils, SurfaceXYZTensorFourier at mpol=ntor=5 "
        "fit to magnetic axis with minor radius 0.10, surface solved via "
        "BoozerSurface.solve_residual_equation_exactly_newton at tol=1e-13 "
        "starting from iota=-0.406 and G0=2π·(nfp·Σ|I_k|)·μ0/(2π)). At the "
        "converged state the CPU lane evaluates upstream wrappers Iotas, "
        "MajorRadius, and NonQuasiSymmetricRatio (auxiliary surface "
        "sDIM=20), and the JAX lane recomputes the corresponding scalar "
        "values from the solved iota plus pure-JAX helpers "
        "(surface_major_radius_jax_from_dofs, _qs_ratio_pure) on a fresh "
        "BiotSavartJAX coil_set_spec. This fixture does not claim public "
        "BoozerSurfaceJAX wrapper or adjoint parity. The CPU-side length "
        "penalty sum(CurveLength) has no native JAX path today and is "
        "listed in unsupported_components, so the verdict is 'partial'."
    ),
    fixture_kind="boozer_qa_wrappers_solved_state",
    inputs={
        "config_name": "ncsx",
        "mpol": 5,
        "ntor": 5,
        "stellsym": True,
        "minor_radius": 0.10,
        "fit_to_curve_flip_theta": True,
        "iota_initial": -0.406,
        "G0_source": "2π * (nfp * Σ|I_k|) * μ0/(2π)",
        "solver": "BoozerSurface.solve_residual_equation_exactly_newton",
        "solver_tol": 1e-13,
        "solver_maxiter": 20,
        "label": "volume",
        "wrapper_set": ("iota", "major_radius", "nq_symmetric_ratio"),
        "non_quasisymmetric_sDIM": 20,
        "length_penalty_classification": "sum_CurveLength (unsupported)",
    },
)


FINITE_BETA_TARGET_FLUX_SPEC = FixtureSpec(
    fixture_id="finite_beta_target_flux",
    source_example="examples/2_Intermediate/stage_two_optimization_finite_beta.py",
    classification=UNSUPPORTED_NATIVE_JAX,
    classification_reason=(
        "SquaredFluxJAX accepts a target normal-field array, so the JAX "
        "side is supportable in principle. The blocker is the "
        "VirtualCasing.from_vmec(...) preprocessing step: it requires a "
        "VMEC run and is not wired into the non-banana parity harness "
        "today. Once a cached vcasing_*.nc artifact is checked in for "
        "the W7-X target equilibrium and a load path is added to the "
        "harness, this fixture should be flipped to SUPPORTED with a "
        "partial verdict (SquaredFluxJAX with target array native; CPU-"
        "only length QuadraticPenalty(identity) listed as unsupported)."
    ),
)


FINITEBUILD_MULTIFILAMENT_SUPPORT_GATE_SPEC = FixtureSpec(
    fixture_id="finitebuild_multifilament_support_gate",
    source_example="examples/3_Advanced/stage_two_optimization_finitebuild.py",
    classification=SUPPORT_GATE,
    classification_reason=(
        "Finite-build multifilament curves are expanded by "
        "simsopt.geo.finitebuild.create_multifilament_grid. The probe "
        "materializes a low-resolution grid and checks each expanded "
        "filament against the same _supports_native_curve_geometry "
        "predicate BiotSavartJAX uses internally. The fixture stays "
        "support_gate either because a filament lacks to_spec() or "
        "because a full multifilament native-JAX build_lanes constructor "
        "(flux + length + curvature + filament-arclength variation + "
        "min-distance penalty) has not been wired into this harness yet."
    ),
    inputs={
        "ncoils": 3,
        "R0": 1.0,
        "R1": 0.5,
        "order": 5,
        "numfilaments_n": 2,
        "numfilaments_b": 3,
        "gapsize_n": 0.02,
        "gapsize_b": 0.04,
        "rotation_order": 1,
        "vmec_input": "tests/test_files/input.LandremanPaul2021_QA",
        "surface_range": "half period",
        "nphi": 16,
        "ntheta": 16,
    },
)


QFM_SURFACE_SPEC = FixtureSpec(
    fixture_id="qfm_surface",
    source_example="examples/1_Simple/qfm.py",
    classification=UNSUPPORTED_NATIVE_JAX,
    classification_reason=(
        "QfmResidualJAX exists at src/simsopt/geo/surfaceobjectives_jax.py "
        "and has parity coverage elsewhere in the JAX-port test suite. "
        "Wiring a per-fixture report for QFM-residual + label parity is a "
        "follow-up plan because the harness LaneArtifact does not yet "
        "carry QFM residual vectors / label values / surface DOF "
        "comparisons; those are not BiotSavart/SquaredFlux quantities."
    ),
)


# ---------------------------------------------------------------------------
# Registry


@dataclass(frozen=True)
class FixtureRecord:
    spec: FixtureSpec
    builder: Callable[[], FixtureBuild]


def _unsupported_classification_builder(
    spec: FixtureSpec,
) -> Callable[[], FixtureBuild]:
    """Return a builder that raises FixtureNotSupportedError(reason)."""
    return _raise_unsupported(
        f"Fixture {spec.fixture_id!r} ({spec.classification}): "
        f"{spec.classification_reason}"
    )


FIXTURE_REGISTRY: Mapping[str, FixtureRecord] = {
    MINIMAL_STAGE2_FLUX_LENGTH_GAP_SPEC.fixture_id: FixtureRecord(
        spec=MINIMAL_STAGE2_FLUX_LENGTH_GAP_SPEC,
        builder=_build_minimal_stage2_state,
    ),
    CWS_SAVED_LOCAL_FLUX_NFP2_SPEC.fixture_id: FixtureRecord(
        spec=CWS_SAVED_LOCAL_FLUX_NFP2_SPEC,
        builder=lambda: _build_cws_saved_local_flux(nfp=2),
    ),
    CWS_SAVED_LOCAL_FLUX_NFP3_SPEC.fixture_id: FixtureRecord(
        spec=CWS_SAVED_LOCAL_FLUX_NFP3_SPEC,
        builder=lambda: _build_cws_saved_local_flux(nfp=3),
    ),
    FULL_STAGE2_COMPOSITE_SPEC.fixture_id: FixtureRecord(
        spec=FULL_STAGE2_COMPOSITE_SPEC,
        builder=_build_full_stage2_composite,
    ),
    PLANAR_STAGE2_COMPOSITE_SPEC.fixture_id: FixtureRecord(
        spec=PLANAR_STAGE2_COMPOSITE_SPEC,
        builder=_build_planar_stage2_composite,
    ),
    POSITION_ORIENTATION_FLUX_SUPPORT_GATE_SPEC.fixture_id: FixtureRecord(
        spec=POSITION_ORIENTATION_FLUX_SUPPORT_GATE_SPEC,
        builder=_build_position_orientation_support_gate_probe,
    ),
    BOOZER_SURFACE_BASIC_SPEC.fixture_id: FixtureRecord(
        spec=BOOZER_SURFACE_BASIC_SPEC,
        builder=_build_boozer_surface_basic,
    ),
    BOOZER_QA_WRAPPERS_SPEC.fixture_id: FixtureRecord(
        spec=BOOZER_QA_WRAPPERS_SPEC,
        builder=_build_boozer_qa_wrappers,
    ),
    FINITE_BETA_TARGET_FLUX_SPEC.fixture_id: FixtureRecord(
        spec=FINITE_BETA_TARGET_FLUX_SPEC,
        builder=_unsupported_classification_builder(FINITE_BETA_TARGET_FLUX_SPEC),
    ),
    FINITEBUILD_MULTIFILAMENT_SUPPORT_GATE_SPEC.fixture_id: FixtureRecord(
        spec=FINITEBUILD_MULTIFILAMENT_SUPPORT_GATE_SPEC,
        builder=_build_finitebuild_support_gate_probe,
    ),
    QFM_SURFACE_SPEC.fixture_id: FixtureRecord(
        spec=QFM_SURFACE_SPEC,
        builder=_unsupported_classification_builder(QFM_SURFACE_SPEC),
    ),
}


def fixture_ids() -> Tuple[str, ...]:
    return tuple(FIXTURE_REGISTRY.keys())


def supported_fixture_ids() -> Tuple[str, ...]:
    return tuple(
        fid
        for fid, record in FIXTURE_REGISTRY.items()
        if record.spec.classification == SUPPORTED
    )


def get_fixture(fixture_id: str) -> FixtureRecord:
    if fixture_id not in FIXTURE_REGISTRY:
        valid = ", ".join(sorted(FIXTURE_REGISTRY))
        raise KeyError(f"Unknown fixture_id {fixture_id!r}. Expected one of: {valid}.")
    return FIXTURE_REGISTRY[fixture_id]
