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


SURFACE_SCALAR = "surface_scalar"
QFM = "qfm"
PERMANENT_MAGNET = "pm"
PERMANENT_MAGNET_RELAX_AND_SPLIT = "pm_relax_and_split"
WIREFRAME = "wireframe"
WIREFRAME_GSCO = "wireframe_gsco"
TRACING = "tracing"
STRAIN = "strain"
COIL_FORCE_ENERGY = "coil_force_energy"

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
    rationale: str = ""
    acceptance_criteria: Tuple[str, ...] = ()
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


def _file_input_metadata(path: Path) -> Mapping[str, Any]:
    relpath = path.relative_to(REPO_ROOT)
    payload = path.read_bytes()
    return {
        "path": str(relpath),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def fixed_state_input_hash(inputs: Mapping[str, Any]) -> str:
    """Return a stable hash for fixture input metadata and dense arrays."""
    import json

    def _normalize(value):
        if isinstance(value, np.ndarray):
            arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
            return {
                "array_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                "dtype": str(arr.dtype),
                "shape": list(arr.shape),
            }
        if isinstance(value, Mapping):
            return {str(k): _normalize(v) for k, v in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            return [_normalize(v) for v in value]
        return value

    payload = json.dumps(_normalize(inputs), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


FINITE_BETA_TARGET_ARRAY_PATH = (
    TESTS_FILES / "finite_beta_w7x_B_external_normal_nphi32_ntheta32.npy"
)


def _load_finite_beta_target_array() -> np.ndarray:
    return np.ascontiguousarray(
        np.load(FINITE_BETA_TARGET_ARRAY_PATH, allow_pickle=False),
        dtype=np.float64,
    )


def _finite_beta_target_metadata() -> Mapping[str, Any]:
    target = _load_finite_beta_target_array()
    return {
        "path": str(FINITE_BETA_TARGET_ARRAY_PATH.relative_to(REPO_ROOT)),
        "shape": list(target.shape),
        "dtype": str(target.dtype),
        "array_sha256": _hash_array(target),
        "file": _file_input_metadata(FINITE_BETA_TARGET_ARRAY_PATH),
        "source": "VirtualCasing.from_vmec",
        "source_vmec": (
            "tests/test_files/"
            "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
        ),
        "src_nphi": 80,
        "trgt_nphi": 32,
        "trgt_ntheta": 32,
    }


def gpu_readiness_metadata(*, proven: bool = False) -> Mapping[str, Any]:
    """Mark fixture rows as CPU-only until an explicit GPU artifact exists."""
    return {
        "gpu_ready": bool(proven),
        "gpu_proven": bool(proven),
        "gpu_artifact": None,
    }


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
    objective_component_name: str = "SquaredFlux",
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
    components[objective_component_name] = j_total

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
    objective_component_name: str = "SquaredFluxJAX",
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
    components[objective_component_name] = j_total

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


def _build_scalar_lane(
    *,
    lane: str,
    objective_total: float,
    components: Mapping[str, float],
    gradient: np.ndarray,
    active_dof_names: Sequence[str],
    active_dofs: np.ndarray,
    raw_arrays: Mapping[str, np.ndarray],
    setup_seconds: float,
    execute_seconds: float,
    native_curve_spec_hashes: Sequence[str] = (),
) -> LaneArtifact:
    zero_array = np.zeros((0,), dtype=np.float64)
    return LaneArtifact(
        lane=lane,
        objective_total=float(objective_total),
        objective_native_subtotal=float(objective_total),
        components=_flatten_components(components),
        gradient=np.asarray(gradient, dtype=np.float64),
        gradient_norm=float(np.linalg.norm(gradient)),
        active_dof_names=tuple(active_dof_names),
        active_dof_hash=_hash_array(np.asarray(active_dofs, dtype=np.float64)),
        fixed_free_mask_hash=_hash_mask(np.ones_like(active_dofs, dtype=bool)),
        native_curve_spec_hashes=tuple(native_curve_spec_hashes),
        surface_point_hash=_hash_array(zero_array),
        unit_normal_hash=_hash_array(zero_array),
        field_B_hash=_hash_array(zero_array),
        field_B_max=0.0,
        field_B_mean=0.0,
        Bdotn_array_hash=_hash_array(zero_array),
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays=dict(raw_arrays),
        timing={"setup_s": float(setup_seconds), "execute_s": float(execute_seconds)},
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
        jf_cpu=jf_full,
        bs_cpu=bs_cpu,
        target_array=None,
        extra_components=extra_components_cpu,
        setup_seconds=setup_seconds_cpu,
        objective_component_name="JF_total_cpu",
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
    from simsopt.geo.curveobjectives_jax import CurveLengthJAX

    bs_jax = BiotSavartJAX(coils_jax)
    jf_jax = SquaredFluxJAX(surface, bs_jax)
    jls_jax = [CurveLengthJAX(c) for c in base_curves_jax]
    length_penalty_jax = QuadraticPenalty(sum(jls_jax), length_target, "max")
    jf_full_jax = jf_jax + length_weight * length_penalty_jax
    length_penalty_jax_value = float(length_penalty_jax.J())
    setup_seconds_jax = time.perf_counter() - start_jax_setup
    jax_lane = _build_jax_lane(
        surface=surface,
        coils=coils_jax,
        bs_jax=bs_jax,
        jf_jax=jf_full_jax,
        target_array=None,
        extra_components={
            "SquaredFluxJAX": float(jf_jax.J()),
            "QuadraticPenalty_over_sum_CurveLength_max": length_penalty_jax_value,
        },
        setup_seconds=setup_seconds_jax,
        objective_component_name="JF_total_jax",
    )

    x0 = np.asarray(jf_full.x, dtype=np.float64).copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        jf_full.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_full.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jf_full_jax.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_full_jax.J())

    spec = MINIMAL_STAGE2_FLUX_LENGTH_GAP_SPEC
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
# Wave 1 — examples/1_Simple/surf_vol_area.py


def _build_surface_area_volume_simple():
    """Recreate the side-effect-free fixed state from surf_vol_area.py."""
    import time
    import jax
    from simsopt.geo import Area, SurfaceRZFourier, Volume
    from simsopt.geo.surfaceobjectives_jax import AreaJAX, VolumeJAX

    start_cpu = time.perf_counter()
    cpu_surface = SurfaceRZFourier()
    cpu_surface.fix("rc(0,0)")
    area_cpu = Area(cpu_surface)
    volume_cpu = Volume(cpu_surface)
    cpu_grad_area = np.asarray(area_cpu.dJ_by_dsurfacecoefficients(), dtype=np.float64)
    cpu_grad_volume = np.asarray(
        volume_cpu.dJ_by_dsurfacecoefficients(), dtype=np.float64
    )
    cpu_setup = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    jax_surface = SurfaceRZFourier()
    jax_surface.fix("rc(0,0)")
    jax_surface.set_dofs(np.asarray(cpu_surface.get_dofs(), dtype=np.float64))

    area_jax = AreaJAX(jax_surface)
    volume_jax = VolumeJAX(jax_surface)
    jax_area_value = area_jax.J()
    jax_volume_value = volume_jax.J()
    jax_grad_area = np.asarray(area_jax.dJ_by_dsurfacecoefficients(), dtype=np.float64)
    jax_grad_volume = np.asarray(
        volume_jax.dJ_by_dsurfacecoefficients(), dtype=np.float64
    )
    jax.block_until_ready(jax_grad_area)
    jax.block_until_ready(jax_grad_volume)

    base_dofs = np.asarray(cpu_surface.get_dofs(), dtype=np.float64)
    perturbations = np.zeros((2, base_dofs.size), dtype=np.float64)
    if base_dofs.size >= 1:
        perturbations[0, 0] = 1.0e-3
    if base_dofs.size >= 2:
        perturbations[1, 1] = -7.0e-4
    else:
        perturbations[1, 0] = -7.0e-4

    cpu_perturbed_area = []
    cpu_perturbed_volume = []
    jax_perturbed_area = []
    jax_perturbed_volume = []
    for delta in perturbations:
        perturbed_dofs = base_dofs + delta
        cpu_surface.set_dofs(perturbed_dofs)
        jax_surface.set_dofs(perturbed_dofs)
        cpu_perturbed_area.append(float(area_cpu.J()))
        cpu_perturbed_volume.append(float(volume_cpu.J()))
        jax_area_delta = area_jax.J()
        jax_volume_delta = volume_jax.J()
        jax.block_until_ready(jax_area_delta)
        jax.block_until_ready(jax_volume_delta)
        jax_perturbed_area.append(float(jax_area_delta))
        jax_perturbed_volume.append(float(jax_volume_delta))
    cpu_surface.set_dofs(base_dofs)
    jax_surface.set_dofs(base_dofs)
    jax_setup = time.perf_counter() - start_jax

    cpu_dofs = np.asarray(cpu_surface.get_dofs(), dtype=np.float64)
    jax_dofs = np.asarray(jax_surface.get_dofs(), dtype=np.float64)
    dof_names_cpu = tuple(cpu_surface.dof_names)
    dof_names_jax = tuple(jax_surface.dof_names)
    free_mask_cpu = np.asarray(cpu_surface.local_dofs_free_status, dtype=bool)
    free_mask_jax = np.asarray(jax_surface.local_dofs_free_status, dtype=bool)

    cpu_gamma = np.asarray(cpu_surface.gamma(), dtype=np.float64)
    cpu_normal = np.asarray(cpu_surface.unitnormal(), dtype=np.float64)
    jax_gamma = np.asarray(jax_surface.gamma(), dtype=np.float64)
    jax_normal = np.asarray(jax_surface.unitnormal(), dtype=np.float64)
    empty_field = np.zeros((*cpu_gamma.shape[:2], 3), dtype=np.float64)
    empty_scalar_grid = np.zeros(cpu_gamma.shape[:2], dtype=np.float64)

    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=float(area_cpu.J() + volume_cpu.J()),
        objective_native_subtotal=float(area_cpu.J() + volume_cpu.J()),
        components={
            "area": float(area_cpu.J()),
            "volume": float(volume_cpu.J()),
        },
        gradient=np.concatenate([cpu_grad_area, cpu_grad_volume]),
        gradient_norm=float(
            np.linalg.norm(np.concatenate([cpu_grad_area, cpu_grad_volume]))
        ),
        active_dof_names=dof_names_cpu,
        active_dof_hash=_hash_array(cpu_dofs),
        fixed_free_mask_hash=_hash_mask(free_mask_cpu),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(cpu_gamma),
        unit_normal_hash=_hash_array(cpu_normal),
        field_B_hash=_hash_array(empty_field),
        field_B_max=0.0,
        field_B_mean=0.0,
        Bdotn_array_hash=_hash_array(empty_scalar_grid),
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays={
            "surface_gamma": cpu_gamma,
            "surface_unit_normal": cpu_normal,
            "area_gradient": cpu_grad_area,
            "volume_gradient": cpu_grad_volume,
            "area_perturbed_values": np.asarray(cpu_perturbed_area, dtype=np.float64),
            "volume_perturbed_values": np.asarray(
                cpu_perturbed_volume, dtype=np.float64
            ),
            "objective_total": np.array([float(area_cpu.J() + volume_cpu.J())]),
        },
        timing={"setup_s": float(cpu_setup), "execute_s": 0.0},
    )
    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=float(jax_area_value + jax_volume_value),
        objective_native_subtotal=float(jax_area_value + jax_volume_value),
        components={
            "area": float(jax_area_value),
            "volume": float(jax_volume_value),
        },
        gradient=np.concatenate([jax_grad_area, jax_grad_volume]),
        gradient_norm=float(
            np.linalg.norm(np.concatenate([jax_grad_area, jax_grad_volume]))
        ),
        active_dof_names=dof_names_jax,
        active_dof_hash=_hash_array(jax_dofs),
        fixed_free_mask_hash=_hash_mask(free_mask_jax),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(jax_gamma),
        unit_normal_hash=_hash_array(jax_normal),
        field_B_hash=_hash_array(empty_field),
        field_B_max=0.0,
        field_B_mean=0.0,
        Bdotn_array_hash=_hash_array(empty_scalar_grid),
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays={
            "surface_gamma": jax_gamma,
            "surface_unit_normal": jax_normal,
            "area_gradient": jax_grad_area,
            "volume_gradient": jax_grad_volume,
            "area_perturbed_values": np.asarray(jax_perturbed_area, dtype=np.float64),
            "volume_perturbed_values": np.asarray(
                jax_perturbed_volume, dtype=np.float64
            ),
            "objective_total": np.array([float(jax_area_value + jax_volume_value)]),
        },
        timing={"setup_s": float(jax_setup), "execute_s": 0.0},
    )

    x0 = cpu_dofs.copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        cpu_surface.set_dofs(np.asarray(dofs, dtype=np.float64))
        return float(area_cpu.J() + volume_cpu.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jax_surface.set_dofs(np.asarray(dofs, dtype=np.float64))
        return float(area_jax.J() + volume_jax.J())

    return FixtureBuild(
        spec=SURFACE_AREA_VOLUME_SIMPLE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
    )


# ---------------------------------------------------------------------------
# Wave 2 — examples/1_Simple/qfm.py


def _build_qfm_surface_fixed_state():
    """Build the qfm.py initial residual/label state without running solvers."""
    import time
    import jax
    from simsopt.configs import get_data
    from simsopt.field import BiotSavart
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import Area, QfmResidual, SurfaceRZFourier, ToroidalFlux, Volume
    from simsopt.geo.surfaceobjectives_jax import AreaJAX, QfmResidualJAX, VolumeJAX

    start_cpu = time.perf_counter()
    _base_curves, _base_currents, ma, nfp, bs = get_data("ncsx")
    bs_cpu = BiotSavart(bs.coils)
    bs_tf_cpu = BiotSavart(bs.coils)
    phis = np.linspace(0.0, 1.0 / nfp, 25, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 25, endpoint=False)
    surface_cpu = SurfaceRZFourier(
        mpol=5,
        ntor=5,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface_cpu.fit_to_curve(ma, 0.2, flip_theta=True)
    qfm_cpu = QfmResidual(surface_cpu, bs_cpu)
    volume_cpu = Volume(surface_cpu)
    area_cpu = Area(surface_cpu)
    tf_cpu = ToroidalFlux(surface_cpu, bs_tf_cpu)
    cpu_grad = np.asarray(qfm_cpu.dJ_by_dsurfacecoefficients(), dtype=np.float64)
    gamma_cpu = np.asarray(surface_cpu.gamma(), dtype=np.float64)
    normal_cpu = np.asarray(surface_cpu.unitnormal(), dtype=np.float64)
    bs_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    field_B_cpu = np.asarray(bs_cpu.B(), dtype=np.float64).reshape(25, 25, 3)
    setup_cpu = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    _base_curves_jax, _base_currents_jax, ma_jax, _nfp_jax, bs_jax_source = get_data(
        "ncsx"
    )
    surface_jax = SurfaceRZFourier(
        mpol=5,
        ntor=5,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface_jax.fit_to_curve(ma_jax, 0.2, flip_theta=True)
    surface_jax.set_dofs(np.asarray(surface_cpu.get_dofs(), dtype=np.float64))
    bs_jax = BiotSavartJAX(bs_jax_source.coils)
    bs_tf_jax = BiotSavartJAX(bs_jax_source.coils)
    qfm_jax = QfmResidualJAX(surface_jax, bs_jax)
    area_jax = AreaJAX(surface_jax)
    volume_jax = VolumeJAX(surface_jax)
    tf_jax = ToroidalFlux(surface_jax, bs_tf_jax)
    qfm_value_jax = qfm_jax.J()
    jax_grad = np.asarray(qfm_jax.dJ_by_dsurfacecoefficients(), dtype=np.float64)
    jax.block_until_ready(jax_grad)
    gamma_jax = np.asarray(surface_jax.gamma(), dtype=np.float64)
    normal_jax = np.asarray(surface_jax.unitnormal(), dtype=np.float64)
    bs_jax.set_points(gamma_jax.reshape((-1, 3)))
    field_B_jax = np.asarray(bs_jax.B(), dtype=np.float64).reshape(25, 25, 3)
    setup_jax = time.perf_counter() - start_jax

    dofs_cpu = np.asarray(surface_cpu.get_dofs(), dtype=np.float64)
    dofs_jax = np.asarray(surface_jax.get_dofs(), dtype=np.float64)
    free_mask_cpu = np.asarray(surface_cpu.local_dofs_free_status, dtype=bool)
    free_mask_jax = np.asarray(surface_jax.local_dofs_free_status, dtype=bool)
    bdotn_cpu = np.sum(field_B_cpu * normal_cpu, axis=2)
    bdotn_jax = np.sum(field_B_jax * normal_jax, axis=2)

    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=float(qfm_cpu.J()),
        objective_native_subtotal=float(qfm_cpu.J()),
        components={
            "qfm_residual": float(qfm_cpu.J()),
            "area": float(area_cpu.J()),
            "volume": float(volume_cpu.J()),
            "toroidal_flux": float(tf_cpu.J()),
        },
        gradient=cpu_grad,
        gradient_norm=float(np.linalg.norm(cpu_grad)),
        active_dof_names=tuple(surface_cpu.dof_names),
        active_dof_hash=_hash_array(dofs_cpu),
        fixed_free_mask_hash=_hash_mask(free_mask_cpu),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma_cpu),
        unit_normal_hash=_hash_array(normal_cpu),
        field_B_hash=_hash_array(field_B_cpu),
        field_B_max=float(np.max(np.abs(field_B_cpu))),
        field_B_mean=float(np.mean(np.abs(field_B_cpu))),
        Bdotn_array_hash=_hash_array(bdotn_cpu),
        Bdotn_max=float(np.max(np.abs(bdotn_cpu))),
        Bdotn_mean=float(np.mean(np.abs(bdotn_cpu))),
        raw_arrays={
            "surface_gamma": gamma_cpu,
            "surface_unit_normal": normal_cpu,
            "field_B": field_B_cpu,
            "Bdotn": bdotn_cpu,
            "qfm_gradient": cpu_grad,
            "objective_total": np.array([float(qfm_cpu.J())]),
        },
        timing={"setup_s": float(setup_cpu), "execute_s": 0.0},
    )
    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=float(qfm_value_jax),
        objective_native_subtotal=float(qfm_value_jax),
        components={
            "qfm_residual": float(qfm_value_jax),
            "area": float(area_jax.J()),
            "volume": float(volume_jax.J()),
            "toroidal_flux": float(tf_jax.J()),
        },
        gradient=jax_grad,
        gradient_norm=float(np.linalg.norm(jax_grad)),
        active_dof_names=tuple(surface_jax.dof_names),
        active_dof_hash=_hash_array(dofs_jax),
        fixed_free_mask_hash=_hash_mask(free_mask_jax),
        native_curve_spec_hashes=_verify_jax_native_spec_contract(bs_jax_source.coils),
        surface_point_hash=_hash_array(gamma_jax),
        unit_normal_hash=_hash_array(normal_jax),
        field_B_hash=_hash_array(field_B_jax),
        field_B_max=float(np.max(np.abs(field_B_jax))),
        field_B_mean=float(np.mean(np.abs(field_B_jax))),
        Bdotn_array_hash=_hash_array(bdotn_jax),
        Bdotn_max=float(np.max(np.abs(bdotn_jax))),
        Bdotn_mean=float(np.mean(np.abs(bdotn_jax))),
        raw_arrays={
            "surface_gamma": gamma_jax,
            "surface_unit_normal": normal_jax,
            "field_B": field_B_jax,
            "Bdotn": bdotn_jax,
            "qfm_gradient": jax_grad,
            "objective_total": np.array([float(qfm_value_jax)]),
        },
        timing={"setup_s": float(setup_jax), "execute_s": 0.0},
    )

    x0 = dofs_cpu.copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        surface_cpu.set_dofs(np.asarray(dofs, dtype=np.float64))
        qfm_cpu.invalidate_cache()
        return float(qfm_cpu.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        surface_jax.set_dofs(np.asarray(dofs, dtype=np.float64))
        qfm_jax.invalidate_cache()
        return float(qfm_jax.J())

    return FixtureBuild(
        spec=QFM_SURFACE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=("QfmSurface_host_solver",),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
    )


# ---------------------------------------------------------------------------
# Wave 3 — examples/1_Simple/permanent_magnet_simple.py


def _pm_normal_norms(pm_cpu) -> np.ndarray:
    normal = np.asarray(pm_cpu.plasma_boundary.normal(), dtype=np.float64)
    return np.ravel(np.sqrt(np.sum(normal * normal, axis=-1)))


def _pm_history_sample_indices(*, K: int, nhistory: int) -> np.ndarray:
    period = max(1, int(K / nhistory))
    return np.asarray(
        [
            step
            for step in range(K)
            if (step % period) == 0 or step == 0 or step == (K - 1)
        ],
        dtype=np.int64,
    )


def _pm_history_components(
    *,
    A_obj: np.ndarray,
    b_obj: np.ndarray,
    m_history: np.ndarray,
    normal_norms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_history = np.reshape(
        np.asarray(m_history, dtype=np.float64), (m_history.shape[0], -1)
    )
    residual_history = flat_history @ np.asarray(A_obj, dtype=np.float64).T
    residual_history -= np.asarray(b_obj, dtype=np.float64)[None, :]
    R2_history = 0.5 * np.sum(residual_history * residual_history, axis=1)
    Bn_history = np.sum(
        np.abs(residual_history) * np.sqrt(normal_norms)[None, :], axis=1
    ) / np.sqrt(residual_history.shape[1])
    return R2_history, Bn_history


def _sample_jax_gpmo_history(
    *,
    result_jax,
    grid_jax,
    normal_norms: np.ndarray,
    K: int,
    nhistory: int,
    initial_m: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = _pm_history_sample_indices(K=K, nhistory=nhistory)
    m_history_steps = np.asarray(result_jax.m_history, dtype=np.float64)[indices]
    if initial_m is not None:
        m_history_steps = np.concatenate(
            [np.asarray(initial_m, dtype=np.float64)[None, :, :], m_history_steps],
            axis=0,
        )
    R2_history, Bn_history = _pm_history_components(
        A_obj=np.asarray(grid_jax.A_obj, dtype=np.float64),
        b_obj=np.asarray(grid_jax.b_obj, dtype=np.float64),
        m_history=m_history_steps,
        normal_norms=normal_norms,
    )
    return R2_history, Bn_history, np.moveaxis(m_history_steps, 0, -1)


def _build_pm_simple_fixed_state_gpmo_baseline():
    """Build reduced permanent_magnet_simple baseline GPMO parity fixture."""
    import time
    import jax
    from simsopt.field import DipoleField, ToroidalField
    from simsopt.field.dipole_field_jax import DipoleFieldJAX
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
    from simsopt.geo.permanent_magnet_grid_jax import PermanentMagnetGridJAX
    from simsopt.solve import GPMO
    from simsopt.solve.permanent_magnet_optimization_jax import GPMO_baseline_jax

    start_cpu = time.perf_counter()
    nphi = 2
    ntheta = 2
    downsample = 100
    K = 4
    reg_l2 = 0.0
    single_direction = -1
    surface_filename = TESTS_FILES / "wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc"
    famus_filename = TESTS_FILES / "init_orient_pm_nonorm_5E4_q4_dp.focus"

    surface_cpu = SurfaceRZFourier.from_wout(
        surface_filename,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    net_poloidal_current_Amperes = 3.7713e6
    mu0 = 4 * np.pi * 1e-7
    RB = mu0 * net_poloidal_current_Amperes / (2 * np.pi)
    bs_cpu = ToroidalField(R0=1, B0=RB)
    gamma_cpu = np.asarray(surface_cpu.gamma(), dtype=np.float64)
    normal_cpu = np.asarray(surface_cpu.unitnormal(), dtype=np.float64)
    bs_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    Bnormal_cpu = np.sum(
        np.asarray(bs_cpu.B(), dtype=np.float64).reshape((nphi, ntheta, 3))
        * normal_cpu,
        axis=2,
    )
    pm_cpu = PermanentMagnetGrid.geo_setup_from_famus(
        surface_cpu,
        Bnormal_cpu,
        famus_filename,
        coordinate_flag="cylindrical",
        downsample=downsample,
    )
    R2_history, Bn_history, m_history = GPMO(
        pm_cpu,
        algorithm="baseline",
        K=K,
        reg_l2=reg_l2,
        single_direction=single_direction,
        nhistory=1,
        verbose=True,
    )
    dipole_cpu = DipoleField(
        pm_cpu.dipole_grid_xyz,
        pm_cpu.m,
        nfp=surface_cpu.nfp,
        coordinate_flag=pm_cpu.coordinate_flag,
        m_maxima=pm_cpu.m_maxima,
    )
    dipole_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    dipole_B_cpu = np.asarray(dipole_cpu.B(), dtype=np.float64).reshape(
        (nphi, ntheta, 3)
    )
    dipole_Bn_cpu = np.sum(dipole_B_cpu * normal_cpu, axis=2)
    residual_cpu = np.asarray(pm_cpu.A_obj @ pm_cpu.m - pm_cpu.b_obj, dtype=np.float64)
    cpu_objective = float(0.5 * np.dot(residual_cpu, residual_cpu))
    setup_cpu = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    grid_jax = PermanentMagnetGridJAX.from_cpu(pm_cpu)
    result_jax = GPMO_baseline_jax(
        grid_jax,
        K=K,
        reg_l2=reg_l2,
        single_direction=single_direction,
    )
    normal_norms = _pm_normal_norms(pm_cpu)
    jax_R2_history, jax_Bn_history, jax_m_history = _sample_jax_gpmo_history(
        result_jax=result_jax,
        grid_jax=grid_jax,
        normal_norms=normal_norms,
        K=K,
        nhistory=1,
    )
    m_jax = np.asarray(result_jax.m, dtype=np.float64).reshape(pm_cpu.ndipoles * 3)
    residual_jax = np.asarray(result_jax.residual, dtype=np.float64)
    jax.block_until_ready(result_jax.residual)
    dipole_jax = DipoleFieldJAX(
        pm_cpu.dipole_grid_xyz,
        np.asarray(result_jax.m, dtype=np.float64),
        stellsym=surface_cpu.stellsym,
        nfp=surface_cpu.nfp,
        coordinate_flag=pm_cpu.coordinate_flag,
        m_maxima=pm_cpu.m_maxima,
    )
    dipole_jax.set_points(gamma_cpu.reshape((-1, 3)))
    dipole_B_jax = np.asarray(dipole_jax.B(), dtype=np.float64).reshape(
        (nphi, ntheta, 3)
    )
    dipole_Bn_jax = np.sum(dipole_B_jax * normal_cpu, axis=2)
    jax_objective = float(0.5 * np.dot(residual_jax, residual_jax))
    setup_jax = time.perf_counter() - start_jax

    empty_grad = np.zeros(0, dtype=np.float64)
    free_mask = np.ones(pm_cpu.ndipoles * 3, dtype=bool)
    dof_names = tuple(f"pm_m[{idx}]" for idx in range(pm_cpu.ndipoles * 3))
    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=cpu_objective,
        objective_native_subtotal=cpu_objective,
        components={
            "pm_objective": cpu_objective,
            "ndipoles": float(pm_cpu.ndipoles),
            "K": float(K),
            "algorithm_variant": 0.0,
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(np.asarray(pm_cpu.m, dtype=np.float64)),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma_cpu),
        unit_normal_hash=_hash_array(normal_cpu),
        field_B_hash=_hash_array(dipole_B_cpu),
        field_B_max=float(np.max(np.abs(dipole_B_cpu))),
        field_B_mean=float(np.mean(np.abs(dipole_B_cpu))),
        Bdotn_array_hash=_hash_array(dipole_Bn_cpu),
        Bdotn_max=float(np.max(np.abs(dipole_Bn_cpu))),
        Bdotn_mean=float(np.mean(np.abs(dipole_Bn_cpu))),
        raw_arrays={
            "surface_gamma": gamma_cpu,
            "surface_unit_normal": normal_cpu,
            "A_obj": np.asarray(pm_cpu.A_obj, dtype=np.float64),
            "b_obj": np.asarray(pm_cpu.b_obj, dtype=np.float64),
            "ATb": np.asarray(pm_cpu.ATb, dtype=np.float64),
            "m_maxima": np.asarray(pm_cpu.m_maxima, dtype=np.float64),
            "dipole_grid_xyz": np.asarray(pm_cpu.dipole_grid_xyz, dtype=np.float64),
            "m": np.asarray(pm_cpu.m, dtype=np.float64),
            "residual": residual_cpu,
            "R2_history": np.asarray(R2_history, dtype=np.float64),
            "Bn_history": np.asarray(Bn_history, dtype=np.float64),
            "m_history": np.asarray(m_history, dtype=np.float64),
            "dipole_B": dipole_B_cpu,
            "dipole_Bn": dipole_Bn_cpu,
            "objective_total": np.array([cpu_objective], dtype=np.float64),
        },
        timing={"setup_s": float(setup_cpu), "execute_s": 0.0},
    )
    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=jax_objective,
        objective_native_subtotal=jax_objective,
        components={
            "pm_objective": jax_objective,
            "ndipoles": float(grid_jax.ndipoles),
            "K": float(K),
            "algorithm_variant": 0.0,
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(m_jax),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma_cpu),
        unit_normal_hash=_hash_array(normal_cpu),
        field_B_hash=_hash_array(dipole_B_jax),
        field_B_max=float(np.max(np.abs(dipole_B_jax))),
        field_B_mean=float(np.mean(np.abs(dipole_B_jax))),
        Bdotn_array_hash=_hash_array(dipole_Bn_jax),
        Bdotn_max=float(np.max(np.abs(dipole_Bn_jax))),
        Bdotn_mean=float(np.mean(np.abs(dipole_Bn_jax))),
        raw_arrays={
            "surface_gamma": gamma_cpu,
            "surface_unit_normal": normal_cpu,
            "A_obj": np.asarray(grid_jax.A_obj, dtype=np.float64),
            "b_obj": np.asarray(grid_jax.b_obj, dtype=np.float64),
            "ATb": np.asarray(grid_jax.ATb, dtype=np.float64).reshape(-1),
            "m_maxima": np.asarray(grid_jax.m_maxima, dtype=np.float64),
            "dipole_grid_xyz": np.asarray(grid_jax.dipole_grid_xyz, dtype=np.float64),
            "m": m_jax,
            "x": np.asarray(result_jax.x, dtype=np.float64),
            "residual": residual_jax,
            "R2_history": jax_R2_history,
            "Bn_history": jax_Bn_history,
            "m_history": jax_m_history,
            "residual_history": np.asarray(
                result_jax.residual_history, dtype=np.float64
            ),
            "selected_dipoles": np.asarray(
                result_jax.selected_dipoles, dtype=np.float64
            ),
            "selected_components": np.asarray(
                result_jax.selected_components, dtype=np.float64
            ),
            "selected_signs": np.asarray(result_jax.selected_signs, dtype=np.float64),
            "dipole_B": dipole_B_jax,
            "dipole_Bn": dipole_Bn_jax,
            "objective_total": np.array([jax_objective], dtype=np.float64),
        },
        timing={"setup_s": float(setup_jax), "execute_s": 0.0},
    )
    return FixtureBuild(
        spec=PM_SIMPLE_FIXED_STATE_GPMO_BASELINE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
    )


def _build_pm_muse_famus_arbvec_backtracking():
    """Build reduced MUSE FAMUS ArbVec_backtracking PM parity fixture."""
    import time
    import jax
    from simsopt.field import BiotSavart, Coil, Current, DipoleField
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
    from simsopt.solve import GPMO
    from simsopt.util import FocusData, discretize_polarizations, polarization_axes
    from simsopt.util.coil_optimization_helper_functions import read_focus_coils
    from simsopt.util.permanent_magnet_helper_functions import initialize_default_kwargs
    from simsopt.field.dipole_field_jax import DipoleFieldJAX
    from simsopt.geo.permanent_magnet_grid_jax import PermanentMagnetGridJAX
    from simsopt.solve.permanent_magnet_optimization_jax import (
        GPMO_ArbVec_backtracking_jax,
    )

    start_cpu = time.perf_counter()

    nphi = 2
    ntheta = 2
    downsample = 100
    dr = 0.01
    K = 5
    backtracking = 2
    max_nMagnets = 4
    nAdjacent = 1
    thresh_angle = np.pi
    surface_filename = TESTS_FILES / "input.muse"
    famus_filename = TESTS_FILES / "zot80.focus"
    coil_filename = TESTS_FILES / "muse_tf_coils.focus"

    surface_cpu = SurfaceRZFourier.from_focus(
        surface_filename,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    base_curves, base_currents0, ncoils = read_focus_coils(coil_filename)
    total_current_value = float(
        np.sum([current.get_value() for current in base_currents0])
    )
    base_currents = [
        (Current(total_current_value / ncoils * 1e-5) * 1e5)
        for _idx in range(ncoils - 1)
    ]
    total_current = Current(total_current_value)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = [Coil(base_curves[idx], base_currents[idx]) for idx in range(ncoils)]
    for curve in base_curves:
        curve.fix_all()
    bs_cpu = BiotSavart(coils)
    gamma_cpu = np.asarray(surface_cpu.gamma(), dtype=np.float64)
    normal_cpu = np.asarray(surface_cpu.unitnormal(), dtype=np.float64)
    bs_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    Bnormal_cpu = np.sum(
        np.asarray(bs_cpu.B(), dtype=np.float64).reshape((nphi, ntheta, 3))
        * normal_cpu,
        axis=2,
    )

    mag_data = FocusData(famus_filename, downsample=downsample)
    pol_axes_f, pol_type_f = polarization_axes(["face"])
    ntype_f = int(len(pol_type_f) / 2)
    pol_axes = pol_axes_f[:ntype_f, :]
    pol_type = pol_type_f[:ntype_f]
    ophi = np.arctan2(mag_data.oy, mag_data.ox)
    discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
    pol_vectors = np.zeros((mag_data.nMagnets, len(pol_type), 3))
    pol_vectors[:, :, 0] = mag_data.pol_x
    pol_vectors[:, :, 1] = mag_data.pol_y
    pol_vectors[:, :, 2] = mag_data.pol_z

    pm_cpu = PermanentMagnetGrid.geo_setup_from_famus(
        surface_cpu,
        Bnormal_cpu,
        famus_filename,
        pol_vectors=pol_vectors,
        downsample=downsample,
        dr=dr,
    )
    kwargs = initialize_default_kwargs("GPMO")
    kwargs.update(
        K=K,
        nhistory=1,
        backtracking=backtracking,
        Nadjacent=nAdjacent,
        dipole_grid_xyz=np.ascontiguousarray(pm_cpu.dipole_grid_xyz),
        max_nMagnets=max_nMagnets,
        thresh_angle=thresh_angle,
    )
    R2_history, Bn_history, m_history = GPMO(pm_cpu, "ArbVec_backtracking", **kwargs)
    min_history_index = int(np.argmin(R2_history))
    pm_cpu.m = np.ravel(m_history[:, :, min_history_index])
    dipole_cpu = DipoleField(
        pm_cpu.dipole_grid_xyz,
        pm_cpu.m,
        nfp=surface_cpu.nfp,
        coordinate_flag=pm_cpu.coordinate_flag,
        m_maxima=pm_cpu.m_maxima,
    )
    dipole_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    dipole_B_cpu = np.asarray(dipole_cpu.B(), dtype=np.float64).reshape(
        (nphi, ntheta, 3)
    )
    dipole_Bn_cpu = np.sum(dipole_B_cpu * normal_cpu, axis=2)
    residual_cpu = np.asarray(pm_cpu.A_obj @ pm_cpu.m - pm_cpu.b_obj, dtype=np.float64)
    cpu_objective = float(0.5 * np.dot(residual_cpu, residual_cpu))
    setup_cpu = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    grid_jax = PermanentMagnetGridJAX.from_cpu(pm_cpu)
    result_jax = GPMO_ArbVec_backtracking_jax(
        grid_jax,
        K=K,
        Nadjacent=nAdjacent,
        backtracking=backtracking,
        thresh_angle=thresh_angle,
        max_nMagnets=max_nMagnets,
        pol_vectors=pol_vectors,
    )
    jax.block_until_ready(result_jax.m)
    m_jax = np.asarray(result_jax.m, dtype=np.float64).reshape(pm_cpu.ndipoles * 3)
    residual_jax = np.asarray(result_jax.residual, dtype=np.float64)
    dipole_jax = DipoleFieldJAX(
        pm_cpu.dipole_grid_xyz,
        np.asarray(result_jax.m, dtype=np.float64),
        stellsym=surface_cpu.stellsym,
        nfp=surface_cpu.nfp,
        coordinate_flag=pm_cpu.coordinate_flag,
        m_maxima=pm_cpu.m_maxima,
    )
    dipole_jax.set_points(gamma_cpu.reshape((-1, 3)))
    dipole_B_jax = np.asarray(dipole_jax.B(), dtype=np.float64).reshape(
        (nphi, ntheta, 3)
    )
    dipole_Bn_jax = np.sum(dipole_B_jax * normal_cpu, axis=2)
    jax_objective = float(0.5 * np.dot(residual_jax, residual_jax))
    normal_norms = _pm_normal_norms(pm_cpu)
    jax_R2_history, jax_Bn_history, jax_m_history = _sample_jax_gpmo_history(
        result_jax=result_jax,
        grid_jax=grid_jax,
        normal_norms=normal_norms,
        K=K,
        nhistory=1,
        initial_m=np.zeros((pm_cpu.ndipoles, 3), dtype=np.float64),
    )
    setup_jax = time.perf_counter() - start_jax

    empty_grad = np.zeros(0, dtype=np.float64)
    free_mask = np.ones(pm_cpu.ndipoles * 3, dtype=bool)
    dof_names = tuple(f"pm_m[{idx}]" for idx in range(pm_cpu.ndipoles * 3))

    def _pm_lane(lane, objective, components, active_hash, field_B, Bdotn, raw, setup):
        return LaneArtifact(
            lane=lane,
            objective_total=objective,
            objective_native_subtotal=objective,
            components=components,
            gradient=empty_grad,
            gradient_norm=0.0,
            active_dof_names=dof_names,
            active_dof_hash=active_hash,
            fixed_free_mask_hash=_hash_mask(free_mask),
            native_curve_spec_hashes=(),
            surface_point_hash=_hash_array(gamma_cpu),
            unit_normal_hash=_hash_array(normal_cpu),
            field_B_hash=_hash_array(field_B),
            field_B_max=float(np.max(np.abs(field_B))),
            field_B_mean=float(np.mean(np.abs(field_B))),
            Bdotn_array_hash=_hash_array(Bdotn),
            Bdotn_max=float(np.max(np.abs(Bdotn))),
            Bdotn_mean=float(np.mean(np.abs(Bdotn))),
            raw_arrays=raw,
            timing={"setup_s": float(setup), "execute_s": 0.0},
        )

    cpu_raw = {
        "surface_gamma": gamma_cpu,
        "surface_unit_normal": normal_cpu,
        "A_obj": np.asarray(pm_cpu.A_obj, dtype=np.float64),
        "b_obj": np.asarray(pm_cpu.b_obj, dtype=np.float64),
        "ATb": np.asarray(pm_cpu.ATb, dtype=np.float64),
        "m_maxima": np.asarray(pm_cpu.m_maxima, dtype=np.float64),
        "dipole_grid_xyz": np.asarray(pm_cpu.dipole_grid_xyz, dtype=np.float64),
        "m": np.asarray(pm_cpu.m, dtype=np.float64),
        "residual": residual_cpu,
        "R2_history": np.asarray(R2_history, dtype=np.float64),
        "Bn_history": np.asarray(Bn_history, dtype=np.float64),
        "m_history": np.asarray(m_history, dtype=np.float64),
        "dipole_B": dipole_B_cpu,
        "dipole_Bn": dipole_Bn_cpu,
        "objective_total": np.array([cpu_objective], dtype=np.float64),
    }
    jax_raw = {
        "surface_gamma": gamma_cpu,
        "surface_unit_normal": normal_cpu,
        "A_obj": np.asarray(grid_jax.A_obj, dtype=np.float64),
        "b_obj": np.asarray(grid_jax.b_obj, dtype=np.float64),
        "ATb": np.asarray(grid_jax.ATb, dtype=np.float64).reshape(-1),
        "m_maxima": np.asarray(grid_jax.m_maxima, dtype=np.float64),
        "dipole_grid_xyz": np.asarray(grid_jax.dipole_grid_xyz, dtype=np.float64),
        "m": m_jax,
        "residual": residual_jax,
        "R2_history": jax_R2_history,
        "Bn_history": jax_Bn_history,
        "m_history": jax_m_history,
        "residual_history": np.asarray(result_jax.residual_history, dtype=np.float64),
        "selected_dipoles": np.asarray(result_jax.selected_dipoles, dtype=np.float64),
        "selected_vector_indices": np.asarray(
            result_jax.selected_vector_indices, dtype=np.float64
        ),
        "selected_signs": np.asarray(result_jax.selected_signs, dtype=np.float64),
        "dipole_B": dipole_B_jax,
        "dipole_Bn": dipole_Bn_jax,
        "objective_total": np.array([jax_objective], dtype=np.float64),
    }
    cpu_lane = _pm_lane(
        "cpu_cpp",
        cpu_objective,
        {
            "pm_objective": cpu_objective,
            "ndipoles": float(pm_cpu.ndipoles),
            "K": float(K),
            "algorithm_variant": 3.0,
        },
        _hash_array(np.asarray(pm_cpu.m, dtype=np.float64)),
        dipole_B_cpu,
        dipole_Bn_cpu,
        cpu_raw,
        setup_cpu,
    )
    jax_lane = _pm_lane(
        "jax_cpu",
        jax_objective,
        {
            "pm_objective": jax_objective,
            "ndipoles": float(grid_jax.ndipoles),
            "K": float(K),
            "algorithm_variant": 3.0,
        },
        _hash_array(m_jax),
        dipole_B_jax,
        dipole_Bn_jax,
        jax_raw,
        setup_jax,
    )
    return FixtureBuild(
        spec=PM_MUSE_FAMUS_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
    )


def _build_pm_pm4stell_arbvec_backtracking():
    """Build reduced PM4Stell ArbVec_backtracking PM parity fixture."""
    import time
    import jax
    from simsopt.field import BiotSavart, Coil, DipoleField
    from simsopt.field.dipole_field_jax import DipoleFieldJAX
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
    from simsopt.geo.permanent_magnet_grid_jax import PermanentMagnetGridJAX
    from simsopt.solve import GPMO
    from simsopt.solve.permanent_magnet_optimization_jax import (
        GPMO_ArbVec_backtracking_jax,
    )
    from simsopt.util import (
        FocusData,
        FocusPlasmaBnormal,
        initialize_default_kwargs,
        read_focus_coils,
    )
    from simsopt.util.polarization_project import (
        discretize_polarizations,
        orientation_phi,
        polarization_axes,
    )

    start_cpu = time.perf_counter()
    nphi = 2
    ntheta = 2
    downsample = 100
    K = 5
    backtracking = 2
    max_nMagnets = 4
    nAdjacent = 10
    thresh_angle = np.pi
    surface_filename = TESTS_FILES / "c09r00_B_axis_half_tesla_PM4Stell.plasma"
    coil_filename = TESTS_FILES / "tf_only_half_tesla_symmetry_baxis_PM4Stell.focus"
    famus_filename = TESTS_FILES / "magpie_trial104b_PM4Stell.focus"
    corners_filename = TESTS_FILES / "magpie_trial104b_corners_PM4Stell.csv"

    surface_cpu = SurfaceRZFourier.from_focus(
        surface_filename,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    Bnormal_plasma = FocusPlasmaBnormal(surface_filename).bnormal_grid(
        nphi, ntheta, "half period"
    )
    base_curves, base_currents, ncoils = read_focus_coils(coil_filename)
    coils = [Coil(base_curves[idx], base_currents[idx]) for idx in range(ncoils)]
    base_currents[0].fix_all()
    for curve in base_curves:
        curve.fix_all()
    bs_cpu = BiotSavart(coils)
    gamma_cpu = np.asarray(surface_cpu.gamma(), dtype=np.float64)
    normal_cpu = np.asarray(surface_cpu.unitnormal(), dtype=np.float64)
    bs_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    Bnormal_tfcoils = np.sum(
        np.asarray(bs_cpu.B(), dtype=np.float64).reshape((nphi, ntheta, 3))
        * normal_cpu,
        axis=2,
    )
    Bnormal_cpu = Bnormal_plasma + Bnormal_tfcoils

    mag_data = FocusData(famus_filename, downsample=downsample)
    nMagnets_total = mag_data.nMagnets
    pol_axes = np.zeros((0, 3))
    pol_type = np.zeros(0, dtype=int)
    for axes_request, type_offset in (
        (["face"], 0),
        (["fe_ftri"], 1),
        (["fc_ftri"], 2),
    ):
        next_axes, next_type = polarization_axes(axes_request)
        ntype = int(len(next_type) / 2)
        pol_axes = np.concatenate((pol_axes, next_axes[:ntype, :]), axis=0)
        pol_type = np.concatenate((pol_type, next_type[:ntype] + type_offset))

    ophi = orientation_phi(corners_filename)[:nMagnets_total]
    discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
    pol_vectors = np.zeros((nMagnets_total, len(pol_type), 3))
    pol_vectors[:, :, 0] = mag_data.pol_x
    pol_vectors[:, :, 1] = mag_data.pol_y
    pol_vectors[:, :, 2] = mag_data.pol_z

    B_max = 5.0
    mu0 = 4 * np.pi * 1e-7
    m_maxima = B_max / mu0
    pm_cpu = PermanentMagnetGrid.geo_setup_from_famus(
        surface_cpu,
        Bnormal_cpu,
        famus_filename,
        pol_vectors=pol_vectors,
        m_maxima=m_maxima,
        downsample=downsample,
    )
    kwargs = initialize_default_kwargs("GPMO")
    kwargs.update(
        K=K,
        nhistory=1,
        backtracking=backtracking,
        Nadjacent=nAdjacent,
        dipole_grid_xyz=np.ascontiguousarray(pm_cpu.dipole_grid_xyz),
        max_nMagnets=max_nMagnets,
        thresh_angle=thresh_angle,
    )
    R2_history, Bn_history, m_history = GPMO(pm_cpu, "ArbVec_backtracking", **kwargs)
    dipole_cpu = DipoleField(
        pm_cpu.dipole_grid_xyz,
        pm_cpu.m,
        nfp=surface_cpu.nfp,
        coordinate_flag=pm_cpu.coordinate_flag,
        m_maxima=pm_cpu.m_maxima,
    )
    dipole_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    dipole_B_cpu = np.asarray(dipole_cpu.B(), dtype=np.float64).reshape(
        (nphi, ntheta, 3)
    )
    dipole_Bn_cpu = np.sum(dipole_B_cpu * normal_cpu, axis=2)
    residual_cpu = np.asarray(pm_cpu.A_obj @ pm_cpu.m - pm_cpu.b_obj, dtype=np.float64)
    cpu_objective = float(0.5 * np.dot(residual_cpu, residual_cpu))
    setup_cpu = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    grid_jax = PermanentMagnetGridJAX.from_cpu(pm_cpu)
    result_jax = GPMO_ArbVec_backtracking_jax(
        grid_jax,
        K=K,
        Nadjacent=nAdjacent,
        backtracking=backtracking,
        thresh_angle=thresh_angle,
        max_nMagnets=max_nMagnets,
        pol_vectors=pol_vectors,
    )
    jax.block_until_ready(result_jax.m)
    m_jax = np.asarray(result_jax.m, dtype=np.float64).reshape(pm_cpu.ndipoles * 3)
    residual_jax = np.asarray(result_jax.residual, dtype=np.float64)
    dipole_jax = DipoleFieldJAX(
        pm_cpu.dipole_grid_xyz,
        np.asarray(result_jax.m, dtype=np.float64),
        stellsym=surface_cpu.stellsym,
        nfp=surface_cpu.nfp,
        coordinate_flag=pm_cpu.coordinate_flag,
        m_maxima=pm_cpu.m_maxima,
    )
    dipole_jax.set_points(gamma_cpu.reshape((-1, 3)))
    dipole_B_jax = np.asarray(dipole_jax.B(), dtype=np.float64).reshape(
        (nphi, ntheta, 3)
    )
    dipole_Bn_jax = np.sum(dipole_B_jax * normal_cpu, axis=2)
    jax_objective = float(0.5 * np.dot(residual_jax, residual_jax))
    normal_norms = _pm_normal_norms(pm_cpu)
    jax_R2_history, jax_Bn_history, jax_m_history = _sample_jax_gpmo_history(
        result_jax=result_jax,
        grid_jax=grid_jax,
        normal_norms=normal_norms,
        K=K,
        nhistory=1,
        initial_m=np.zeros((pm_cpu.ndipoles, 3), dtype=np.float64),
    )
    setup_jax = time.perf_counter() - start_jax

    empty_grad = np.zeros(0, dtype=np.float64)
    free_mask = np.ones(pm_cpu.ndipoles * 3, dtype=bool)
    dof_names = tuple(f"pm_m[{idx}]" for idx in range(pm_cpu.ndipoles * 3))

    def _pm_lane(lane, objective, components, active_hash, field_B, Bdotn, raw, setup):
        return LaneArtifact(
            lane=lane,
            objective_total=objective,
            objective_native_subtotal=objective,
            components=components,
            gradient=empty_grad,
            gradient_norm=0.0,
            active_dof_names=dof_names,
            active_dof_hash=active_hash,
            fixed_free_mask_hash=_hash_mask(free_mask),
            native_curve_spec_hashes=(),
            surface_point_hash=_hash_array(gamma_cpu),
            unit_normal_hash=_hash_array(normal_cpu),
            field_B_hash=_hash_array(field_B),
            field_B_max=float(np.max(np.abs(field_B))),
            field_B_mean=float(np.mean(np.abs(field_B))),
            Bdotn_array_hash=_hash_array(Bdotn),
            Bdotn_max=float(np.max(np.abs(Bdotn))),
            Bdotn_mean=float(np.mean(np.abs(Bdotn))),
            raw_arrays=raw,
            timing={"setup_s": float(setup), "execute_s": 0.0},
        )

    cpu_raw = {
        "surface_gamma": gamma_cpu,
        "surface_unit_normal": normal_cpu,
        "A_obj": np.asarray(pm_cpu.A_obj, dtype=np.float64),
        "b_obj": np.asarray(pm_cpu.b_obj, dtype=np.float64),
        "ATb": np.asarray(pm_cpu.ATb, dtype=np.float64),
        "m_maxima": np.asarray(pm_cpu.m_maxima, dtype=np.float64),
        "dipole_grid_xyz": np.asarray(pm_cpu.dipole_grid_xyz, dtype=np.float64),
        "m": np.asarray(pm_cpu.m, dtype=np.float64),
        "residual": residual_cpu,
        "R2_history": np.asarray(R2_history, dtype=np.float64),
        "Bn_history": np.asarray(Bn_history, dtype=np.float64),
        "m_history": np.asarray(m_history, dtype=np.float64),
        "dipole_B": dipole_B_cpu,
        "dipole_Bn": dipole_Bn_cpu,
        "objective_total": np.array([cpu_objective], dtype=np.float64),
    }
    jax_raw = {
        "surface_gamma": gamma_cpu,
        "surface_unit_normal": normal_cpu,
        "A_obj": np.asarray(grid_jax.A_obj, dtype=np.float64),
        "b_obj": np.asarray(grid_jax.b_obj, dtype=np.float64),
        "ATb": np.asarray(grid_jax.ATb, dtype=np.float64).reshape(-1),
        "m_maxima": np.asarray(grid_jax.m_maxima, dtype=np.float64),
        "dipole_grid_xyz": np.asarray(grid_jax.dipole_grid_xyz, dtype=np.float64),
        "m": m_jax,
        "residual": residual_jax,
        "R2_history": jax_R2_history,
        "Bn_history": jax_Bn_history,
        "m_history": jax_m_history,
        "residual_history": np.asarray(result_jax.residual_history, dtype=np.float64),
        "selected_dipoles": np.asarray(result_jax.selected_dipoles, dtype=np.float64),
        "selected_vector_indices": np.asarray(
            result_jax.selected_vector_indices, dtype=np.float64
        ),
        "selected_signs": np.asarray(result_jax.selected_signs, dtype=np.float64),
        "dipole_B": dipole_B_jax,
        "dipole_Bn": dipole_Bn_jax,
        "objective_total": np.array([jax_objective], dtype=np.float64),
    }
    cpu_lane = _pm_lane(
        "cpu_cpp",
        cpu_objective,
        {
            "pm_objective": cpu_objective,
            "ndipoles": float(pm_cpu.ndipoles),
            "K": float(K),
            "algorithm_variant": 3.0,
        },
        _hash_array(np.asarray(pm_cpu.m, dtype=np.float64)),
        dipole_B_cpu,
        dipole_Bn_cpu,
        cpu_raw,
        setup_cpu,
    )
    jax_lane = _pm_lane(
        "jax_cpu",
        jax_objective,
        {
            "pm_objective": jax_objective,
            "ndipoles": float(grid_jax.ndipoles),
            "K": float(K),
            "algorithm_variant": 3.0,
        },
        _hash_array(m_jax),
        dipole_B_jax,
        dipole_Bn_jax,
        jax_raw,
        setup_jax,
    )
    return FixtureBuild(
        spec=PM_PM4STELL_BACKTRACKING_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
    )


def _build_pm_qa_relax_and_split_fixed_state():
    """Build reduced QA relax-and-split permanent-magnet parity fixture."""
    import time
    import jax
    from simsopt.field import BiotSavart, Current, DipoleField, coils_via_symmetries
    from simsopt.field.dipole_field_jax import DipoleFieldJAX
    from simsopt.geo import (
        PermanentMagnetGrid,
        SurfaceRZFourier,
        create_equally_spaced_curves,
    )
    from simsopt.geo.permanent_magnet_grid_jax import PermanentMagnetGridJAX
    from simsopt.solve import relax_and_split
    from simsopt.solve.permanent_magnet_optimization_jax import relax_and_split_jax
    from simsopt.util import initialize_default_kwargs

    start_cpu = time.perf_counter()
    nphi = 4
    ntheta = 4
    dr = 0.05
    coff = 0.1
    poff = 0.05
    ncoils = 8
    coil_R0 = 1.0
    coil_R1 = 0.65
    coil_order = 5
    coil_numquadpoints = 128
    total_current_value = 187500.0
    reg_l0 = 0.05
    nu = 1e10
    max_iter = 2
    max_iter_RS = 2
    threshold_passes = 2
    surface_filename = TESTS_FILES / "input.LandremanPaul2021_QA_lowres"

    surface_cpu = SurfaceRZFourier.from_vmec_input(
        surface_filename,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    surface_inner = SurfaceRZFourier.from_vmec_input(
        surface_filename,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    surface_outer = SurfaceRZFourier.from_vmec_input(
        surface_filename,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    surface_inner.extend_via_projected_normal(poff)
    surface_outer.extend_via_projected_normal(poff + coff)

    base_curves = create_equally_spaced_curves(
        ncoils,
        surface_cpu.nfp,
        stellsym=True,
        R0=coil_R0,
        R1=coil_R1,
        order=coil_order,
        numquadpoints=coil_numquadpoints,
    )
    base_currents = [
        (Current(total_current_value / ncoils * 1e-5) * 1e5)
        for _idx in range(ncoils - 1)
    ]
    total_current = Current(total_current_value)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    for curve in base_curves:
        curve.fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface_cpu.nfp, True)
    bs_cpu = BiotSavart(coils)
    gamma_cpu = np.asarray(surface_cpu.gamma(), dtype=np.float64)
    normal_cpu = np.asarray(surface_cpu.unitnormal(), dtype=np.float64)
    bs_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    Bnormal_cpu = np.sum(
        np.asarray(bs_cpu.B(), dtype=np.float64).reshape((nphi, ntheta, 3))
        * normal_cpu,
        axis=2,
    )

    pm_cpu = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
        surface_cpu,
        Bnormal_cpu,
        surface_inner,
        surface_outer,
        dr=dr,
        coordinate_flag="cylindrical",
    )
    reg_l0_scaled, _reg_l1, _reg_l2, nu_scaled = pm_cpu.rescale_for_opt(
        reg_l0, 0.0, 0.0, nu
    )
    grid_jax = PermanentMagnetGridJAX.from_cpu(pm_cpu)

    total_rs_history = []
    total_m_history = []
    total_m_proxy_history = []
    total_jax_rs_history = []
    total_jax_m_history = []
    total_jax_m_proxy_history = []
    m0_cpu = np.zeros(pm_cpu.ndipoles * 3, dtype=np.float64)
    jax_result = None
    m0_jax = None
    for pass_index in range(threshold_passes):
        pass_reg_l0 = reg_l0_scaled * (pass_index + 1) / threshold_passes
        kwargs = initialize_default_kwargs()
        kwargs.update(
            nu=nu_scaled,
            max_iter=max_iter,
            max_iter_RS=max_iter_RS,
            reg_l0=pass_reg_l0,
            verbose=True,
        )
        rs_history, m_history, m_proxy_history = relax_and_split(
            pm_cpu, m0=m0_cpu, **kwargs
        )
        jax_result = relax_and_split_jax(
            grid_jax,
            m0_jax,
            max_iter=max_iter,
            max_iter_RS=max_iter_RS,
            nu=nu_scaled,
            reg_l0=pass_reg_l0,
        )
        jax.block_until_ready(jax_result.m)
        total_rs_history.append(np.asarray(rs_history, dtype=np.float64))
        total_m_history.append(np.asarray(m_history, dtype=np.float64))
        total_m_proxy_history.append(np.asarray(m_proxy_history, dtype=np.float64))
        total_jax_rs_history.append(np.asarray(jax_result.errors, dtype=np.float64))
        total_jax_m_history.append(np.asarray(jax_result.m_history, dtype=np.float64))
        total_jax_m_proxy_history.append(
            np.asarray(jax_result.m_proxy_history, dtype=np.float64).reshape(
                max_iter, -1
            )
        )
        m0_cpu = pm_cpu.m
        m0_jax = jax_result.m

    m_jax = np.asarray(jax_result.m, dtype=np.float64).reshape(pm_cpu.ndipoles * 3)
    m_proxy_jax = np.asarray(jax_result.m_proxy, dtype=np.float64).reshape(
        pm_cpu.ndipoles * 3
    )
    residual_cpu = np.asarray(pm_cpu.A_obj @ pm_cpu.m - pm_cpu.b_obj, dtype=np.float64)
    residual_proxy_cpu = np.asarray(
        pm_cpu.A_obj @ pm_cpu.m_proxy - pm_cpu.b_obj, dtype=np.float64
    )
    residual_jax = np.asarray(grid_jax.A_obj @ m_jax - grid_jax.b_obj, dtype=np.float64)
    residual_proxy_jax = np.asarray(
        grid_jax.A_obj @ m_proxy_jax - grid_jax.b_obj, dtype=np.float64
    )
    cpu_objective = float(0.5 * np.dot(residual_cpu, residual_cpu))
    cpu_proxy_objective = float(0.5 * np.dot(residual_proxy_cpu, residual_proxy_cpu))
    jax_objective = float(0.5 * np.dot(residual_jax, residual_jax))
    jax_proxy_objective = float(0.5 * np.dot(residual_proxy_jax, residual_proxy_jax))

    dipole_cpu = DipoleField(
        pm_cpu.dipole_grid_xyz,
        pm_cpu.m,
        nfp=surface_cpu.nfp,
        coordinate_flag=pm_cpu.coordinate_flag,
        m_maxima=pm_cpu.m_maxima,
    )
    dipole_proxy_cpu = DipoleField(
        pm_cpu.dipole_grid_xyz,
        pm_cpu.m_proxy,
        nfp=surface_cpu.nfp,
        coordinate_flag=pm_cpu.coordinate_flag,
        m_maxima=pm_cpu.m_maxima,
    )
    dipole_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    dipole_proxy_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    dipole_B_cpu = np.asarray(dipole_cpu.B(), dtype=np.float64).reshape(
        (nphi, ntheta, 3)
    )
    dipole_proxy_B_cpu = np.asarray(dipole_proxy_cpu.B(), dtype=np.float64).reshape(
        (nphi, ntheta, 3)
    )
    dipole_Bn_cpu = np.sum(dipole_B_cpu * normal_cpu, axis=2)
    dipole_proxy_Bn_cpu = np.sum(dipole_proxy_B_cpu * normal_cpu, axis=2)
    setup_cpu = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    dipole_jax = DipoleFieldJAX(
        pm_cpu.dipole_grid_xyz,
        m_jax.reshape(pm_cpu.ndipoles, 3),
        stellsym=surface_cpu.stellsym,
        nfp=surface_cpu.nfp,
        coordinate_flag=pm_cpu.coordinate_flag,
        m_maxima=pm_cpu.m_maxima,
    )
    dipole_proxy_jax = DipoleFieldJAX(
        pm_cpu.dipole_grid_xyz,
        m_proxy_jax.reshape(pm_cpu.ndipoles, 3),
        stellsym=surface_cpu.stellsym,
        nfp=surface_cpu.nfp,
        coordinate_flag=pm_cpu.coordinate_flag,
        m_maxima=pm_cpu.m_maxima,
    )
    dipole_jax.set_points(gamma_cpu.reshape((-1, 3)))
    dipole_proxy_jax.set_points(gamma_cpu.reshape((-1, 3)))
    dipole_B_jax = np.asarray(dipole_jax.B(), dtype=np.float64).reshape(
        (nphi, ntheta, 3)
    )
    dipole_proxy_B_jax = np.asarray(dipole_proxy_jax.B(), dtype=np.float64).reshape(
        (nphi, ntheta, 3)
    )
    dipole_Bn_jax = np.sum(dipole_B_jax * normal_cpu, axis=2)
    dipole_proxy_Bn_jax = np.sum(dipole_proxy_B_jax * normal_cpu, axis=2)
    setup_jax = time.perf_counter() - start_jax

    empty_grad = np.zeros(0, dtype=np.float64)
    free_mask = np.ones(pm_cpu.ndipoles * 3, dtype=bool)
    dof_names = tuple(f"pm_m[{idx}]" for idx in range(pm_cpu.ndipoles * 3))

    def _pm_lane(lane, objective, components, active_hash, field_B, Bdotn, raw, setup):
        return LaneArtifact(
            lane=lane,
            objective_total=objective,
            objective_native_subtotal=objective,
            components=components,
            gradient=empty_grad,
            gradient_norm=0.0,
            active_dof_names=dof_names,
            active_dof_hash=active_hash,
            fixed_free_mask_hash=_hash_mask(free_mask),
            native_curve_spec_hashes=(),
            surface_point_hash=_hash_array(gamma_cpu),
            unit_normal_hash=_hash_array(normal_cpu),
            field_B_hash=_hash_array(field_B),
            field_B_max=float(np.max(np.abs(field_B))),
            field_B_mean=float(np.mean(np.abs(field_B))),
            Bdotn_array_hash=_hash_array(Bdotn),
            Bdotn_max=float(np.max(np.abs(Bdotn))),
            Bdotn_mean=float(np.mean(np.abs(Bdotn))),
            raw_arrays=raw,
            timing={"setup_s": float(setup), "execute_s": 0.0},
        )

    cpu_raw = {
        "surface_gamma": gamma_cpu,
        "surface_unit_normal": normal_cpu,
        "A_obj": np.asarray(pm_cpu.A_obj, dtype=np.float64),
        "b_obj": np.asarray(pm_cpu.b_obj, dtype=np.float64),
        "ATb": np.asarray(pm_cpu.ATb, dtype=np.float64),
        "m_maxima": np.asarray(pm_cpu.m_maxima, dtype=np.float64),
        "dipole_grid_xyz": np.asarray(pm_cpu.dipole_grid_xyz, dtype=np.float64),
        "m": np.asarray(pm_cpu.m, dtype=np.float64),
        "m_proxy": np.asarray(pm_cpu.m_proxy, dtype=np.float64),
        "residual": residual_cpu,
        "residual_proxy": residual_proxy_cpu,
        "RS_history": np.asarray(total_rs_history, dtype=np.float64),
        "m_history": np.asarray(total_m_history, dtype=np.float64),
        "m_proxy_history": np.asarray(total_m_proxy_history, dtype=np.float64),
        "dipole_B": dipole_B_cpu,
        "dipole_proxy_B": dipole_proxy_B_cpu,
        "dipole_Bn": dipole_Bn_cpu,
        "dipole_proxy_Bn": dipole_proxy_Bn_cpu,
        "objective_total": np.array([cpu_objective], dtype=np.float64),
        "objective_proxy": np.array([cpu_proxy_objective], dtype=np.float64),
    }
    jax_raw = {
        "surface_gamma": gamma_cpu,
        "surface_unit_normal": normal_cpu,
        "A_obj": np.asarray(grid_jax.A_obj, dtype=np.float64),
        "b_obj": np.asarray(grid_jax.b_obj, dtype=np.float64),
        "ATb": np.asarray(grid_jax.ATb, dtype=np.float64).reshape(-1),
        "m_maxima": np.asarray(grid_jax.m_maxima, dtype=np.float64),
        "dipole_grid_xyz": np.asarray(grid_jax.dipole_grid_xyz, dtype=np.float64),
        "m": m_jax,
        "m_proxy": m_proxy_jax,
        "residual": residual_jax,
        "residual_proxy": residual_proxy_jax,
        "RS_history": np.asarray(total_jax_rs_history, dtype=np.float64),
        "m_history": np.asarray(total_jax_m_history, dtype=np.float64),
        "m_proxy_history": np.asarray(total_jax_m_proxy_history, dtype=np.float64),
        "dipole_B": dipole_B_jax,
        "dipole_proxy_B": dipole_proxy_B_jax,
        "dipole_Bn": dipole_Bn_jax,
        "dipole_proxy_Bn": dipole_proxy_Bn_jax,
        "objective_total": np.array([jax_objective], dtype=np.float64),
        "objective_proxy": np.array([jax_proxy_objective], dtype=np.float64),
    }
    cpu_lane = _pm_lane(
        "cpu_cpp",
        cpu_objective,
        {
            "pm_objective": cpu_objective,
            "pm_proxy_objective": cpu_proxy_objective,
            "ndipoles": float(pm_cpu.ndipoles),
            "max_iter": float(max_iter),
            "max_iter_RS": float(max_iter_RS),
            "threshold_passes": float(threshold_passes),
            "algorithm_variant": 4.0,
        },
        _hash_array(np.asarray(pm_cpu.m, dtype=np.float64)),
        dipole_B_cpu,
        dipole_Bn_cpu,
        cpu_raw,
        setup_cpu,
    )
    jax_lane = _pm_lane(
        "jax_cpu",
        jax_objective,
        {
            "pm_objective": jax_objective,
            "pm_proxy_objective": jax_proxy_objective,
            "ndipoles": float(grid_jax.ndipoles),
            "max_iter": float(max_iter),
            "max_iter_RS": float(max_iter_RS),
            "threshold_passes": float(threshold_passes),
            "algorithm_variant": 4.0,
        },
        _hash_array(m_jax),
        dipole_B_jax,
        dipole_Bn_jax,
        jax_raw,
        setup_jax,
    )
    return FixtureBuild(
        spec=PM_QA_FIXED_STATE_GPMO_ARB_VEC_OR_MULTI_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(
            "qa_coil_current_optimization",
            "qa_plot_and_famus_outputs",
        ),
    )


# ---------------------------------------------------------------------------
# Wave 4 — examples/2_Intermediate/wireframe_rcls_basic.py


def _build_wireframe_rcls_basic_fixed_state():
    """Build reduced wireframe_rcls_basic RCLS parity fixture."""
    import time
    import jax
    from simsopt.field import WireframeField
    from simsopt.field.wireframefield_jax import WireframeFieldJAX
    from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
    from simsopt.solve import bnorm_obj_matrices, optimize_wireframe
    from simsopt.solve.wireframe_optimization_jax import (
        bnorm_obj_matrices_jax,
        optimize_wireframe_jax,
    )

    start_cpu = time.perf_counter()
    wf_n_phi = 8
    wf_n_theta = 12
    wf_surf_dist = 0.3
    field_on_axis = 1.0
    regularization_w = 1e-10
    filename_equil = TESTS_FILES / "input.LandremanPaul2021_QA"
    plas_n_phi = 32
    plas_n_theta = 32
    surf_plas_cpu = SurfaceRZFourier.from_vmec_input(
        filename_equil,
        nphi=plas_n_phi,
        ntheta=plas_n_theta,
        range="half period",
    )
    surf_wf_cpu = SurfaceRZFourier.from_vmec_input(filename_equil)
    surf_wf_cpu.extend_via_projected_normal(wf_surf_dist)
    wf_cpu = ToroidalWireframe(surf_wf_cpu, wf_n_phi, wf_n_theta)
    mu0 = 4.0 * np.pi * 1e-7
    pol_cur = -2.0 * np.pi * surf_plas_cpu.get_rc(0, 0) * field_on_axis / mu0
    wf_cpu.set_poloidal_current(pol_cur)
    params = {"reg_W": regularization_w}
    A_cpu, b_cpu = bnorm_obj_matrices(wf_cpu, surf_plas_cpu, verbose=False)
    res_cpu = optimize_wireframe(
        wf_cpu,
        "rcls",
        params,
        surf_plas=surf_plas_cpu,
        verbose=False,
    )
    wire_field_cpu = WireframeField(wf_cpu)
    gamma_cpu = np.asarray(surf_plas_cpu.gamma(), dtype=np.float64)
    normal_cpu = np.asarray(surf_plas_cpu.unitnormal(), dtype=np.float64)
    wire_field_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    field_B_cpu = np.asarray(wire_field_cpu.B(), dtype=np.float64).reshape(
        (plas_n_phi, plas_n_theta, 3)
    )
    field_dB_cpu = np.asarray(wire_field_cpu.dB_by_dX(), dtype=np.float64)
    Bnormal_cpu = np.sum(field_B_cpu * normal_cpu, axis=2)
    setup_cpu = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    surf_plas_jax = SurfaceRZFourier.from_vmec_input(
        filename_equil,
        nphi=plas_n_phi,
        ntheta=plas_n_theta,
        range="half period",
    )
    surf_wf_jax = SurfaceRZFourier.from_vmec_input(filename_equil)
    surf_wf_jax.extend_via_projected_normal(wf_surf_dist)
    wf_jax = ToroidalWireframe(surf_wf_jax, wf_n_phi, wf_n_theta)
    wf_jax.set_poloidal_current(pol_cur)
    A_jax, b_jax = bnorm_obj_matrices_jax(wf_jax, surf_plas_jax, verbose=False)
    res_jax = optimize_wireframe_jax(
        wf_jax,
        "rcls",
        params,
        surf_plas=surf_plas_jax,
        verbose=False,
    )
    wire_field_jax = WireframeFieldJAX(wf_jax)
    gamma_jax = np.asarray(surf_plas_jax.gamma(), dtype=np.float64)
    normal_jax = np.asarray(surf_plas_jax.unitnormal(), dtype=np.float64)
    wire_field_jax.set_points(gamma_jax.reshape((-1, 3)))
    field_B_jax = np.asarray(wire_field_jax.B(), dtype=np.float64).reshape(
        (plas_n_phi, plas_n_theta, 3)
    )
    field_dB_jax = np.asarray(wire_field_jax.dB_by_dX(), dtype=np.float64)
    jax.block_until_ready(field_dB_jax)
    Bnormal_jax = np.sum(field_B_jax * normal_jax, axis=2)
    setup_jax = time.perf_counter() - start_jax

    free_mask = np.ones(wf_cpu.n_segments, dtype=bool)
    dof_names = tuple(f"wireframe_current[{idx}]" for idx in range(wf_cpu.n_segments))
    empty_grad = np.zeros(0, dtype=np.float64)
    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=float(res_cpu["f"]),
        objective_native_subtotal=float(res_cpu["f"]),
        components={
            "f_B": float(res_cpu["f_B"]),
            "f_R": float(res_cpu["f_R"]),
            "f": float(res_cpu["f"]),
            "constraints_satisfied": float(bool(wf_cpu.check_constraints())),
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(np.asarray(res_cpu["x"], dtype=np.float64)),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma_cpu),
        unit_normal_hash=_hash_array(normal_cpu),
        field_B_hash=_hash_array(field_B_cpu),
        field_B_max=float(np.max(np.abs(field_B_cpu))),
        field_B_mean=float(np.mean(np.abs(field_B_cpu))),
        Bdotn_array_hash=_hash_array(Bnormal_cpu),
        Bdotn_max=float(np.max(np.abs(Bnormal_cpu))),
        Bdotn_mean=float(np.mean(np.abs(Bnormal_cpu))),
        raw_arrays={
            "surface_gamma": gamma_cpu,
            "surface_unit_normal": normal_cpu,
            "Amat": np.asarray(A_cpu, dtype=np.float64),
            "bvec": np.asarray(b_cpu, dtype=np.float64),
            "x": np.asarray(res_cpu["x"], dtype=np.float64),
            "field_B": field_B_cpu,
            "field_dB_by_dX": field_dB_cpu,
            "Bnormal": Bnormal_cpu,
            "constraints_satisfied": np.array(
                [float(bool(wf_cpu.check_constraints()))]
            ),
            "objective_total": np.array([float(res_cpu["f"])], dtype=np.float64),
        },
        timing={"setup_s": float(setup_cpu), "execute_s": 0.0},
    )
    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=float(res_jax["f"]),
        objective_native_subtotal=float(res_jax["f"]),
        components={
            "f_B": float(res_jax["f_B"]),
            "f_R": float(res_jax["f_R"]),
            "f": float(res_jax["f"]),
            "constraints_satisfied": float(bool(wf_jax.check_constraints())),
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(np.asarray(res_jax["x"], dtype=np.float64)),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma_jax),
        unit_normal_hash=_hash_array(normal_jax),
        field_B_hash=_hash_array(field_B_jax),
        field_B_max=float(np.max(np.abs(field_B_jax))),
        field_B_mean=float(np.mean(np.abs(field_B_jax))),
        Bdotn_array_hash=_hash_array(Bnormal_jax),
        Bdotn_max=float(np.max(np.abs(Bnormal_jax))),
        Bdotn_mean=float(np.mean(np.abs(Bnormal_jax))),
        raw_arrays={
            "surface_gamma": gamma_jax,
            "surface_unit_normal": normal_jax,
            "Amat": np.asarray(A_jax, dtype=np.float64),
            "bvec": np.asarray(b_jax, dtype=np.float64),
            "x": np.asarray(res_jax["x"], dtype=np.float64),
            "field_B": field_B_jax,
            "field_dB_by_dX": field_dB_jax,
            "Bnormal": Bnormal_jax,
            "constraints_satisfied": np.array(
                [float(bool(wf_jax.check_constraints()))]
            ),
            "objective_total": np.array([float(res_jax["f"])], dtype=np.float64),
        },
        timing={"setup_s": float(setup_jax), "execute_s": 0.0},
    )
    return FixtureBuild(
        spec=WIREFRAME_RCLS_BASIC_FIXED_STATE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=("RCLS_current_vector_nonunique_nullspace",),
    )


def _build_wireframe_rcls_ports_constraint_fixed_state():
    """Build reduced RCLS port-constraint fixture from wireframe_rcls_with_ports."""
    import time
    import jax

    from simsopt.field import WireframeField
    from simsopt.field.wireframefield_jax import WireframeFieldJAX
    from simsopt.geo import CircularPort, PortSet, SurfaceRZFourier, ToroidalWireframe
    from simsopt.solve import bnorm_obj_matrices, optimize_wireframe
    from simsopt.solve.wireframe_optimization_jax import (
        bnorm_obj_matrices_jax,
        optimize_wireframe_jax,
    )

    wf_n_phi = 8
    wf_n_theta = 12
    wf_surf_dist = 0.3
    plas_n_phi = 16
    plas_n_theta = 16
    port_phis = (np.pi / 8.0, 3.0 * np.pi / 8.0)
    port_thetas = (np.pi / 4.0, 7.0 * np.pi / 4.0)
    port_ir = 0.1
    port_thick = 0.005
    port_gap = 0.04
    port_l0 = -0.15
    port_l1 = 0.15
    field_on_axis = 1.0
    regularization_w = 1e-10
    filename_equil = TESTS_FILES / "input.LandremanPaul2021_QA"

    def _build_wireframe_state():
        surf_plas = SurfaceRZFourier.from_vmec_input(
            filename_equil,
            nphi=plas_n_phi,
            ntheta=plas_n_theta,
            range="half period",
        )
        surf_wf = SurfaceRZFourier.from_vmec_input(filename_equil)
        surf_wf.extend_via_projected_normal(wf_surf_dist)
        wf = ToroidalWireframe(surf_wf, wf_n_phi, wf_n_theta)

        ports = PortSet()
        gamma = surf_wf.gamma()
        normal = surf_wf.normal()
        for phi in port_phis:
            phi_nearest = int(
                np.argmin(np.abs((0.5 / np.pi) * phi - surf_wf.quadpoints_phi))
            )
            for theta in port_thetas:
                theta_nearest = int(
                    np.argmin(np.abs((0.5 / np.pi) * theta - surf_wf.quadpoints_theta))
                )
                ox, oy, oz = gamma[phi_nearest, theta_nearest]
                ax, ay, az = normal[phi_nearest, theta_nearest]
                ports.add_ports(
                    [
                        CircularPort(
                            ox=ox,
                            oy=oy,
                            oz=oz,
                            ax=ax,
                            ay=ay,
                            az=az,
                            ir=port_ir,
                            thick=port_thick,
                            l0=port_l0,
                            l1=port_l1,
                        )
                    ]
                )
        ports = ports.repeat_via_symmetries(surf_wf.nfp, True)
        wf.constrain_colliding_segments(ports.collides, gap=port_gap)

        mu0 = 4.0 * np.pi * 1e-7
        pol_cur = -2.0 * np.pi * surf_plas.get_rc(0, 0) * field_on_axis / mu0
        wf.set_poloidal_current(pol_cur)
        return surf_plas, wf, ports

    start_cpu = time.perf_counter()
    surf_plas_cpu, wf_cpu, ports_cpu = _build_wireframe_state()
    params = {"reg_W": regularization_w}
    A_cpu, b_cpu = bnorm_obj_matrices(wf_cpu, surf_plas_cpu, verbose=False)
    res_cpu = optimize_wireframe(
        wf_cpu,
        "rcls",
        params,
        surf_plas=surf_plas_cpu,
        verbose=False,
    )
    wire_field_cpu = WireframeField(wf_cpu)
    gamma_cpu = np.asarray(surf_plas_cpu.gamma(), dtype=np.float64)
    normal_cpu = np.asarray(surf_plas_cpu.unitnormal(), dtype=np.float64)
    wire_field_cpu.set_points(gamma_cpu.reshape((-1, 3)))
    field_B_cpu = np.asarray(wire_field_cpu.B(), dtype=np.float64).reshape(
        (plas_n_phi, plas_n_theta, 3)
    )
    field_dB_cpu = np.asarray(wire_field_cpu.dB_by_dX(), dtype=np.float64)
    Bnormal_cpu = np.sum(field_B_cpu * normal_cpu, axis=2)
    constraint_shape_cpu = np.asarray(
        wf_cpu.constraint_matrices()[0].shape,
        dtype=np.float64,
    )
    setup_cpu = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    surf_plas_jax, wf_jax, ports_jax = _build_wireframe_state()
    A_jax, b_jax = bnorm_obj_matrices_jax(wf_jax, surf_plas_jax, verbose=False)
    res_jax = optimize_wireframe_jax(
        wf_jax,
        "rcls",
        params,
        surf_plas=surf_plas_jax,
        verbose=False,
    )
    wire_field_jax = WireframeFieldJAX(wf_jax)
    gamma_jax = np.asarray(surf_plas_jax.gamma(), dtype=np.float64)
    normal_jax = np.asarray(surf_plas_jax.unitnormal(), dtype=np.float64)
    wire_field_jax.set_points(gamma_jax.reshape((-1, 3)))
    field_B_jax = np.asarray(wire_field_jax.B(), dtype=np.float64).reshape(
        (plas_n_phi, plas_n_theta, 3)
    )
    field_dB_jax = np.asarray(wire_field_jax.dB_by_dX(), dtype=np.float64)
    jax.block_until_ready(field_dB_jax)
    Bnormal_jax = np.sum(field_B_jax * normal_jax, axis=2)
    constraint_shape_jax = np.asarray(
        wf_jax.constraint_matrices()[0].shape,
        dtype=np.float64,
    )
    setup_jax = time.perf_counter() - start_jax

    free_mask = np.ones(wf_cpu.n_segments, dtype=bool)
    dof_names = tuple(f"wireframe_current[{idx}]" for idx in range(wf_cpu.n_segments))
    empty_grad = np.zeros(0, dtype=np.float64)
    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=float(res_cpu["f"]),
        objective_native_subtotal=float(res_cpu["f"]),
        components={
            "f_B": float(res_cpu["f_B"]),
            "f_R": float(res_cpu["f_R"]),
            "f": float(res_cpu["f"]),
            "constraints_satisfied": float(bool(wf_cpu.check_constraints())),
            "constraint_rows": float(constraint_shape_cpu[0]),
            "port_count": float(len(ports_cpu.ports)),
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(np.asarray(res_cpu["x"], dtype=np.float64)),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma_cpu),
        unit_normal_hash=_hash_array(normal_cpu),
        field_B_hash=_hash_array(field_B_cpu),
        field_B_max=float(np.max(np.abs(field_B_cpu))),
        field_B_mean=float(np.mean(np.abs(field_B_cpu))),
        Bdotn_array_hash=_hash_array(Bnormal_cpu),
        Bdotn_max=float(np.max(np.abs(Bnormal_cpu))),
        Bdotn_mean=float(np.mean(np.abs(Bnormal_cpu))),
        raw_arrays={
            "surface_gamma": gamma_cpu,
            "surface_unit_normal": normal_cpu,
            "Amat": np.asarray(A_cpu, dtype=np.float64),
            "bvec": np.asarray(b_cpu, dtype=np.float64),
            "x": np.asarray(res_cpu["x"], dtype=np.float64),
            "field_B": field_B_cpu,
            "field_dB_by_dX": field_dB_cpu,
            "Bnormal": Bnormal_cpu,
            "constraints_satisfied": np.array(
                [float(bool(wf_cpu.check_constraints()))]
            ),
            "constraint_matrix_shape": constraint_shape_cpu,
            "objective_total": np.array([float(res_cpu["f"])], dtype=np.float64),
        },
        timing={"setup_s": float(setup_cpu), "execute_s": 0.0},
    )
    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=float(res_jax["f"]),
        objective_native_subtotal=float(res_jax["f"]),
        components={
            "f_B": float(res_jax["f_B"]),
            "f_R": float(res_jax["f_R"]),
            "f": float(res_jax["f"]),
            "constraints_satisfied": float(bool(wf_jax.check_constraints())),
            "constraint_rows": float(constraint_shape_jax[0]),
            "port_count": float(len(ports_jax.ports)),
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(np.asarray(res_jax["x"], dtype=np.float64)),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma_jax),
        unit_normal_hash=_hash_array(normal_jax),
        field_B_hash=_hash_array(field_B_jax),
        field_B_max=float(np.max(np.abs(field_B_jax))),
        field_B_mean=float(np.mean(np.abs(field_B_jax))),
        Bdotn_array_hash=_hash_array(Bnormal_jax),
        Bdotn_max=float(np.max(np.abs(Bnormal_jax))),
        Bdotn_mean=float(np.mean(np.abs(Bnormal_jax))),
        raw_arrays={
            "surface_gamma": gamma_jax,
            "surface_unit_normal": normal_jax,
            "Amat": np.asarray(A_jax, dtype=np.float64),
            "bvec": np.asarray(b_jax, dtype=np.float64),
            "x": np.asarray(res_jax["x"], dtype=np.float64),
            "field_B": field_B_jax,
            "field_dB_by_dX": field_dB_jax,
            "Bnormal": Bnormal_jax,
            "constraints_satisfied": np.array(
                [float(bool(wf_jax.check_constraints()))]
            ),
            "constraint_matrix_shape": constraint_shape_jax,
            "objective_total": np.array([float(res_jax["f"])], dtype=np.float64),
        },
        timing={"setup_s": float(setup_jax), "execute_s": 0.0},
    )
    return FixtureBuild(
        spec=WIREFRAME_RCLS_PORTS_CONSTRAINT_GATE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=("RCLS_current_vector_nonunique_nullspace",),
    )


def _build_wireframe_gsco_modular_fixed_state():
    """Build a deterministic reduced GSCO history fixture."""
    import time
    import jax
    import simsoptpp as sopp

    from simsopt.solve.wireframe_optimization_jax import (
        greedy_stellarator_coil_optimization_jax,
    )

    rng = np.random.default_rng(3104)
    A = np.ascontiguousarray(rng.standard_normal(size=(5, 6)))
    b = np.ascontiguousarray(rng.standard_normal(size=(5, 1)))
    loops = np.ascontiguousarray(np.array([[0, 1, 2, 3], [2, 3, 4, 5]], dtype=np.int64))
    free_loops = np.ascontiguousarray(np.ones(2, dtype=np.int64))
    segments = np.ascontiguousarray(
        np.array(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]],
            dtype=np.int64,
        )
    )
    connections = np.ascontiguousarray(
        np.array(
            [[0, 3, 4, 0], [0, 1, 5, 0], [1, 2, 4, 0], [2, 3, 5, 0]],
            dtype=np.int64,
        )
    )
    x_init = np.ascontiguousarray(np.zeros((6, 1), dtype=np.float64))
    loop_count_init = np.ascontiguousarray(np.zeros(2, dtype=np.int64))
    no_crossing = False
    no_new_coils = False
    match_current = False
    default_current = 0.2
    max_current = np.inf
    max_loop_count = 0
    lambda_s = 0.15
    max_iter = 5

    start_cpu = time.perf_counter()
    (
        x_cpu,
        loop_count_cpu,
        iter_hist_cpu,
        curr_hist_cpu,
        loop_hist_cpu,
        f_B_hist_cpu,
        f_S_hist_cpu,
        f_hist_cpu,
    ) = sopp.GSCO(
        no_crossing,
        no_new_coils,
        match_current,
        A,
        b,
        abs(default_current),
        abs(max_current),
        abs(max_loop_count),
        loops,
        free_loops,
        segments,
        connections,
        lambda_s,
        max_iter,
        x_init,
        loop_count_init,
        max_iter + 1,
    )
    setup_cpu = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    result_jax = greedy_stellarator_coil_optimization_jax(
        no_crossing,
        no_new_coils,
        match_current,
        A,
        b,
        default_current,
        max_current,
        max_loop_count,
        loops,
        free_loops,
        segments,
        connections,
        lambda_s,
        max_iter,
        x_init,
        loop_count_init,
    )
    jax.block_until_ready(result_jax.x)
    setup_jax = time.perf_counter() - start_jax
    history_length = int(np.asarray(result_jax.history_length, dtype=np.int64))
    history_slice = slice(0, history_length)

    zero_array = np.zeros((0,), dtype=np.float64)
    empty_grad = np.zeros(0, dtype=np.float64)
    dof_names = tuple(f"wireframe_current[{idx}]" for idx in range(x_cpu.size))
    free_mask = np.ones(x_cpu.size, dtype=bool)
    cpu_raw = {
        "A_obj": A,
        "b_obj": np.reshape(b, (-1,)),
        "x": np.asarray(x_cpu, dtype=np.float64).reshape(-1),
        "loop_count": np.asarray(loop_count_cpu, dtype=np.float64),
        "iter_hist": np.asarray(iter_hist_cpu, dtype=np.float64),
        "curr_hist": np.asarray(curr_hist_cpu, dtype=np.float64),
        "loop_hist": np.asarray(loop_hist_cpu, dtype=np.float64),
        "f_B_hist": np.asarray(f_B_hist_cpu, dtype=np.float64),
        "f_S_hist": np.asarray(f_S_hist_cpu, dtype=np.float64),
        "f_hist": np.asarray(f_hist_cpu, dtype=np.float64),
        "flags": np.array(
            [
                float(no_crossing),
                float(no_new_coils),
                float(match_current),
                float(max_loop_count),
            ],
            dtype=np.float64,
        ),
        "objective_total": np.array([float(f_hist_cpu[-1])], dtype=np.float64),
    }
    jax_raw = {
        "A_obj": A,
        "b_obj": np.reshape(b, (-1,)),
        "x": np.asarray(result_jax.x, dtype=np.float64).reshape(-1),
        "loop_count": np.asarray(result_jax.loop_count, dtype=np.float64),
        "iter_hist": np.asarray(result_jax.iter_history, dtype=np.float64)[
            history_slice
        ],
        "curr_hist": np.asarray(result_jax.curr_history, dtype=np.float64)[
            history_slice
        ],
        "loop_hist": np.asarray(result_jax.loop_history, dtype=np.float64)[
            history_slice
        ],
        "f_B_hist": np.asarray(result_jax.f_B_history, dtype=np.float64)[history_slice],
        "f_S_hist": np.asarray(result_jax.f_S_history, dtype=np.float64)[history_slice],
        "f_hist": np.asarray(result_jax.f_history, dtype=np.float64)[history_slice],
        "flags": np.array(
            [
                float(no_crossing),
                float(no_new_coils),
                float(match_current),
                float(max_loop_count),
            ],
            dtype=np.float64,
        ),
        "objective_total": np.array(
            [
                float(
                    np.asarray(result_jax.f_history, dtype=np.float64)[
                        history_length - 1
                    ]
                )
            ],
            dtype=np.float64,
        ),
    }

    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=float(f_hist_cpu[-1]),
        objective_native_subtotal=float(f_hist_cpu[-1]),
        components={
            "history_length": float(len(iter_hist_cpu)),
            "no_crossing": float(no_crossing),
            "no_new_coils": float(no_new_coils),
            "match_current": float(match_current),
            "max_loop_count": float(max_loop_count),
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(cpu_raw["x"]),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(zero_array),
        unit_normal_hash=_hash_array(zero_array),
        field_B_hash=_hash_array(zero_array),
        field_B_max=0.0,
        field_B_mean=0.0,
        Bdotn_array_hash=_hash_array(zero_array),
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays=cpu_raw,
        timing={"setup_s": float(setup_cpu), "execute_s": 0.0},
    )
    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=float(jax_raw["objective_total"][0]),
        objective_native_subtotal=float(jax_raw["objective_total"][0]),
        components={
            "history_length": float(history_length),
            "no_crossing": float(no_crossing),
            "no_new_coils": float(no_new_coils),
            "match_current": float(match_current),
            "max_loop_count": float(max_loop_count),
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(jax_raw["x"]),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(zero_array),
        unit_normal_hash=_hash_array(zero_array),
        field_B_hash=_hash_array(zero_array),
        field_B_max=0.0,
        field_B_mean=0.0,
        Bdotn_array_hash=_hash_array(zero_array),
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays=jax_raw,
        timing={"setup_s": float(setup_jax), "execute_s": 0.0},
    )
    return FixtureBuild(
        spec=WIREFRAME_GSCO_MODULAR_FIXED_STATE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
    )


def _build_wireframe_gsco_sector_saddle_fixed_state():
    """Build reduced GSCO sector/saddle fixture from the public example path."""
    import time
    import jax

    from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
    from simsopt.solve import optimize_wireframe
    from simsopt.solve.wireframe_optimization_jax import optimize_wireframe_jax

    filename_equil = TESTS_FILES / "input.LandremanPaul2021_QA"
    filename_wf_surf = TESTS_FILES / "nescin.LandremanPaul2021_QA"
    wf_n_phi = 18
    wf_n_theta = 8
    plas_n = 4
    n_tf_coils_hp = 3
    break_width = 2
    gsco_cur_frac = 0.05
    field_on_axis = 1.0
    lambda_s = 10**-6.5
    max_iter = 5
    print_interval = max_iter + 1

    def _build_state():
        surf_plas = SurfaceRZFourier.from_vmec_input(
            filename_equil,
            nphi=plas_n,
            ntheta=plas_n,
            range="half period",
        )
        surf_wf = SurfaceRZFourier.from_nescoil_input(filename_wf_surf, "current")
        wf = ToroidalWireframe(surf_wf, wf_n_phi, wf_n_theta)
        mu0 = 4.0 * np.pi * 1e-7
        pol_cur = -2.0 * np.pi * surf_plas.get_rc(0, 0) * field_on_axis / mu0
        tfcoil_current = pol_cur / (2 * wf.nfp * n_tf_coils_hp)
        wf.add_tfcoil_currents(n_tf_coils_hp, tfcoil_current)
        wf.set_toroidal_breaks(
            n_tf_coils_hp,
            break_width,
            allow_pol_current=True,
        )
        wf.set_poloidal_current(pol_cur)
        params = {
            "lambda_S": lambda_s,
            "max_iter": max_iter,
            "print_interval": print_interval,
            "no_crossing": True,
            "default_current": abs(gsco_cur_frac * pol_cur),
            "max_current": 1.1 * abs(gsco_cur_frac * pol_cur),
        }
        initial_currents = np.asarray(wf.currents, dtype=np.float64).copy()
        free_cells = np.asarray(wf.get_free_cells(form="logical"), dtype=np.float64)
        return surf_plas, wf, params, initial_currents, free_cells

    start_cpu = time.perf_counter()
    surf_cpu, wf_cpu, params_cpu, initial_currents_cpu, free_cells_cpu = _build_state()
    res_cpu = optimize_wireframe(
        wf_cpu,
        "gsco",
        params_cpu,
        surf_plas=surf_cpu,
        verbose=False,
    )
    gamma_cpu = np.asarray(surf_cpu.gamma(), dtype=np.float64)
    normal_cpu = np.asarray(surf_cpu.unitnormal(), dtype=np.float64)
    res_cpu["wframe_field"].set_points(gamma_cpu.reshape((-1, 3)))
    field_B_cpu = np.asarray(res_cpu["wframe_field"].B(), dtype=np.float64).reshape(
        (plas_n, plas_n, 3)
    )
    Bnormal_cpu = np.sum(field_B_cpu * normal_cpu, axis=2)
    setup_cpu = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    surf_jax, wf_jax, params_jax, initial_currents_jax, free_cells_jax = _build_state()
    res_jax = optimize_wireframe_jax(
        wf_jax,
        "gsco",
        params_jax,
        surf_plas=surf_jax,
        verbose=False,
    )
    gamma_jax = np.asarray(surf_jax.gamma(), dtype=np.float64)
    normal_jax = np.asarray(surf_jax.unitnormal(), dtype=np.float64)
    res_jax["wframe_field"].set_points(gamma_jax.reshape((-1, 3)))
    field_B_jax = np.asarray(res_jax["wframe_field"].B(), dtype=np.float64).reshape(
        (plas_n, plas_n, 3)
    )
    jax.block_until_ready(field_B_jax)
    Bnormal_jax = np.sum(field_B_jax * normal_jax, axis=2)
    setup_jax = time.perf_counter() - start_jax

    empty_grad = np.zeros(0, dtype=np.float64)
    x_cpu = np.asarray(res_cpu["x"], dtype=np.float64).reshape(-1)
    x_jax = np.asarray(res_jax["x"], dtype=np.float64).reshape(-1)
    dof_names = tuple(f"wireframe_current[{idx}]" for idx in range(x_cpu.size))
    free_mask = np.ones(x_cpu.size, dtype=bool)
    flags = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    constraints_cpu = np.array([float(bool(wf_cpu.check_constraints()))])
    constraints_jax = np.array([float(bool(wf_jax.check_constraints()))])
    cpu_raw = {
        "A_obj": np.asarray(res_cpu["Amat"], dtype=np.float64),
        "b_obj": np.asarray(res_cpu["bvec"], dtype=np.float64).reshape(-1),
        "x": x_cpu,
        "loop_count": np.asarray(res_cpu["loop_count"], dtype=np.float64),
        "iter_hist": np.asarray(res_cpu["iter_hist"], dtype=np.float64),
        "curr_hist": np.asarray(res_cpu["curr_hist"], dtype=np.float64),
        "loop_hist": np.asarray(res_cpu["loop_hist"], dtype=np.float64),
        "f_B_hist": np.asarray(res_cpu["f_B_hist"], dtype=np.float64),
        "f_S_hist": np.asarray(res_cpu["f_S_hist"], dtype=np.float64),
        "f_hist": np.asarray(res_cpu["f_hist"], dtype=np.float64),
        "flags": flags,
        "free_cells": free_cells_cpu,
        "initial_currents": initial_currents_cpu,
        "constraints_satisfied": constraints_cpu,
        "field_B": field_B_cpu,
        "Bnormal": Bnormal_cpu,
        "objective_total": np.array([float(res_cpu["f"])], dtype=np.float64),
    }
    jax_raw = {
        "A_obj": np.asarray(res_jax["Amat"], dtype=np.float64),
        "b_obj": np.asarray(res_jax["bvec"], dtype=np.float64).reshape(-1),
        "x": x_jax,
        "loop_count": np.asarray(res_jax["loop_count"], dtype=np.float64),
        "iter_hist": np.asarray(res_jax["iter_hist"], dtype=np.float64),
        "curr_hist": np.asarray(res_jax["curr_hist"], dtype=np.float64),
        "loop_hist": np.asarray(res_jax["loop_hist"], dtype=np.float64),
        "f_B_hist": np.asarray(res_jax["f_B_hist"], dtype=np.float64),
        "f_S_hist": np.asarray(res_jax["f_S_hist"], dtype=np.float64),
        "f_hist": np.asarray(res_jax["f_hist"], dtype=np.float64),
        "flags": flags,
        "free_cells": free_cells_jax,
        "initial_currents": initial_currents_jax,
        "constraints_satisfied": constraints_jax,
        "field_B": field_B_jax,
        "Bnormal": Bnormal_jax,
        "objective_total": np.array([float(res_jax["f"])], dtype=np.float64),
    }
    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=float(res_cpu["f"]),
        objective_native_subtotal=float(res_cpu["f"]),
        components={
            "history_length": float(len(res_cpu["iter_hist"])),
            "no_crossing": 1.0,
            "no_new_coils": 0.0,
            "match_current": 0.0,
            "max_loop_count": 0.0,
            "default_current": float(params_cpu["default_current"]),
            "max_current": float(params_cpu["max_current"]),
            "lambda_S": float(lambda_s),
            "constraints_satisfied": float(constraints_cpu[0]),
            "free_cell_count": float(np.sum(free_cells_cpu)),
            "total_cell_count": float(free_cells_cpu.size),
            "initial_current_nonzero_count": float(
                np.count_nonzero(initial_currents_cpu)
            ),
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(cpu_raw["x"]),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma_cpu),
        unit_normal_hash=_hash_array(normal_cpu),
        field_B_hash=_hash_array(field_B_cpu),
        field_B_max=float(np.max(np.abs(field_B_cpu))),
        field_B_mean=float(np.mean(np.abs(field_B_cpu))),
        Bdotn_array_hash=_hash_array(Bnormal_cpu),
        Bdotn_max=float(np.max(np.abs(Bnormal_cpu))),
        Bdotn_mean=float(np.mean(np.abs(Bnormal_cpu))),
        raw_arrays=cpu_raw,
        timing={"setup_s": float(setup_cpu), "execute_s": 0.0},
    )
    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=float(res_jax["f"]),
        objective_native_subtotal=float(res_jax["f"]),
        components={
            "history_length": float(len(res_jax["iter_hist"])),
            "no_crossing": 1.0,
            "no_new_coils": 0.0,
            "match_current": 0.0,
            "max_loop_count": 0.0,
            "default_current": float(params_jax["default_current"]),
            "max_current": float(params_jax["max_current"]),
            "lambda_S": float(lambda_s),
            "constraints_satisfied": float(constraints_jax[0]),
            "free_cell_count": float(np.sum(free_cells_jax)),
            "total_cell_count": float(free_cells_jax.size),
            "initial_current_nonzero_count": float(
                np.count_nonzero(initial_currents_jax)
            ),
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(jax_raw["x"]),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma_jax),
        unit_normal_hash=_hash_array(normal_jax),
        field_B_hash=_hash_array(field_B_jax),
        field_B_max=float(np.max(np.abs(field_B_jax))),
        field_B_mean=float(np.mean(np.abs(field_B_jax))),
        Bdotn_array_hash=_hash_array(Bnormal_jax),
        Bdotn_max=float(np.max(np.abs(Bnormal_jax))),
        Bdotn_mean=float(np.mean(np.abs(Bnormal_jax))),
        raw_arrays=jax_raw,
        timing={"setup_s": float(setup_jax), "execute_s": 0.0},
    )
    return FixtureBuild(
        spec=WIREFRAME_GSCO_SECTOR_SADDLE_FIXED_STATE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
    )


def _build_wireframe_gsco_multistep_reduced_diagnostic():
    """Build the first immutable GSCO step from wireframe_gsco_multistep.py."""
    import time

    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import (
        SurfaceRZFourier,
        ToroidalWireframe,
        create_equally_spaced_curves,
    )
    from simsopt.solve import optimize_wireframe
    from simsopt.solve.wireframe_optimization_jax import optimize_wireframe_jax

    filename_equil = TESTS_FILES / "input.LandremanPaul2021_QA"
    filename_wf_surf = TESTS_FILES / "nescin.LandremanPaul2021_QA"
    wf_n_phi = 24
    wf_n_theta = 8
    plas_n = 4
    n_tf_coils_hp = 3
    break_width = 4
    init_gsco_cur_frac = 0.2
    field_on_axis = 1.0
    lambda_s = 1e-7
    max_iter = 5
    print_interval = max_iter + 1

    def _build_state(ext_field_cls):
        surf_plas = SurfaceRZFourier.from_vmec_input(
            filename_equil,
            nphi=plas_n,
            ntheta=plas_n,
            range="half period",
        )
        surf_wf = SurfaceRZFourier.from_nescoil_input(filename_wf_surf, "current")
        wf = ToroidalWireframe(surf_wf, wf_n_phi, wf_n_theta)
        mu0 = 4.0 * np.pi * 1e-7
        pol_cur = -2.0 * np.pi * surf_plas.get_rc(0, 0) * field_on_axis / mu0
        wf.set_toroidal_breaks(
            n_tf_coils_hp,
            break_width,
            allow_pol_current=True,
        )
        wf.set_poloidal_current(0)
        tf_curves = create_equally_spaced_curves(
            n_tf_coils_hp,
            surf_plas.nfp,
            True,
            R0=1.0,
            R1=0.85,
        )
        tf_curr = [
            Current(-pol_cur / (2 * n_tf_coils_hp * surf_plas.nfp))
            for _idx in range(n_tf_coils_hp)
        ]
        tf_coils = coils_via_symmetries(tf_curves, tf_curr, surf_plas.nfp, True)
        params = {
            "lambda_S": lambda_s,
            "max_iter": max_iter,
            "print_interval": print_interval,
            "no_crossing": True,
            "max_loop_count": 1,
            "loop_count_init": None,
            "default_current": abs(init_gsco_cur_frac * pol_cur),
            "max_current": 1.1 * abs(init_gsco_cur_frac * pol_cur),
        }
        return surf_plas, wf, ext_field_cls(tf_coils), params

    start_cpu = time.perf_counter()
    surf_cpu, wf_cpu, mf_cpu, params_cpu = _build_state(BiotSavart)
    res_cpu = optimize_wireframe(
        wf_cpu,
        "gsco",
        params_cpu,
        surf_plas=surf_cpu,
        ext_field=mf_cpu,
        verbose=False,
    )
    setup_cpu = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    surf_jax, wf_jax, mf_jax, params_jax = _build_state(BiotSavartJAX)
    res_jax = optimize_wireframe_jax(
        wf_jax,
        "gsco",
        params_jax,
        surf_plas=surf_jax,
        ext_field=mf_jax,
        verbose=False,
    )
    setup_jax = time.perf_counter() - start_jax

    zero_array = np.zeros((0,), dtype=np.float64)
    empty_grad = np.zeros(0, dtype=np.float64)
    x_cpu = np.asarray(res_cpu["x"], dtype=np.float64).reshape(-1)
    x_jax = np.asarray(res_jax["x"], dtype=np.float64).reshape(-1)
    dof_names = tuple(f"wireframe_current[{idx}]" for idx in range(x_cpu.size))
    free_mask = np.ones(x_cpu.size, dtype=bool)
    flags = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    gamma_cpu = np.asarray(surf_cpu.gamma(), dtype=np.float64)
    gamma_jax = np.asarray(surf_jax.gamma(), dtype=np.float64)
    normal_cpu = np.asarray(surf_cpu.unitnormal(), dtype=np.float64)
    normal_jax = np.asarray(surf_jax.unitnormal(), dtype=np.float64)
    cpu_raw = {
        "A_obj": np.asarray(res_cpu["Amat"], dtype=np.float64),
        "b_obj": np.asarray(res_cpu["bvec"], dtype=np.float64).reshape(-1),
        "x": x_cpu,
        "loop_count": np.asarray(res_cpu["loop_count"], dtype=np.float64),
        "iter_hist": np.asarray(res_cpu["iter_hist"], dtype=np.float64),
        "curr_hist": np.asarray(res_cpu["curr_hist"], dtype=np.float64),
        "loop_hist": np.asarray(res_cpu["loop_hist"], dtype=np.float64),
        "f_B_hist": np.asarray(res_cpu["f_B_hist"], dtype=np.float64),
        "f_S_hist": np.asarray(res_cpu["f_S_hist"], dtype=np.float64),
        "f_hist": np.asarray(res_cpu["f_hist"], dtype=np.float64),
        "flags": flags,
        "objective_total": np.array([float(res_cpu["f"])], dtype=np.float64),
    }
    jax_raw = {
        "A_obj": np.asarray(res_jax["Amat"], dtype=np.float64),
        "b_obj": np.asarray(res_jax["bvec"], dtype=np.float64).reshape(-1),
        "x": x_jax,
        "loop_count": np.asarray(res_jax["loop_count"], dtype=np.float64),
        "iter_hist": np.asarray(res_jax["iter_hist"], dtype=np.float64),
        "curr_hist": np.asarray(res_jax["curr_hist"], dtype=np.float64),
        "loop_hist": np.asarray(res_jax["loop_hist"], dtype=np.float64),
        "f_B_hist": np.asarray(res_jax["f_B_hist"], dtype=np.float64),
        "f_S_hist": np.asarray(res_jax["f_S_hist"], dtype=np.float64),
        "f_hist": np.asarray(res_jax["f_hist"], dtype=np.float64),
        "flags": flags,
        "objective_total": np.array([float(res_jax["f"])], dtype=np.float64),
    }
    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=float(res_cpu["f"]),
        objective_native_subtotal=float(res_cpu["f"]),
        components={
            "history_length": float(len(res_cpu["iter_hist"])),
            "no_crossing": 1.0,
            "no_new_coils": 0.0,
            "match_current": 0.0,
            "max_loop_count": 1.0,
            "default_current": float(params_cpu["default_current"]),
            "max_current": float(params_cpu["max_current"]),
            "lambda_S": float(lambda_s),
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(cpu_raw["x"]),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma_cpu),
        unit_normal_hash=_hash_array(normal_cpu),
        field_B_hash=_hash_array(zero_array),
        field_B_max=0.0,
        field_B_mean=0.0,
        Bdotn_array_hash=_hash_array(zero_array),
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays=cpu_raw,
        timing={"setup_s": float(setup_cpu), "execute_s": 0.0},
    )
    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=float(res_jax["f"]),
        objective_native_subtotal=float(res_jax["f"]),
        components={
            "history_length": float(len(res_jax["iter_hist"])),
            "no_crossing": 1.0,
            "no_new_coils": 0.0,
            "match_current": 0.0,
            "max_loop_count": 1.0,
            "default_current": float(params_jax["default_current"]),
            "max_current": float(params_jax["max_current"]),
            "lambda_S": float(lambda_s),
        },
        gradient=empty_grad,
        gradient_norm=0.0,
        active_dof_names=dof_names,
        active_dof_hash=_hash_array(jax_raw["x"]),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma_jax),
        unit_normal_hash=_hash_array(normal_jax),
        field_B_hash=_hash_array(zero_array),
        field_B_max=0.0,
        field_B_mean=0.0,
        Bdotn_array_hash=_hash_array(zero_array),
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays=jax_raw,
        timing={"setup_s": float(setup_jax), "execute_s": 0.0},
    )
    return FixtureBuild(
        spec=WIREFRAME_GSCO_MULTISTEP_REDUCED_DIAGNOSTIC_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(
            "wireframe_multistep_mutation_loop",
            "wireframe_small_coil_pruning",
            "wireframe_final_adjustment_step",
            "wireframe_plot_and_vtk_outputs",
        ),
    )


# ---------------------------------------------------------------------------
# Phase 3 — P1 full Stage-II composite fixture
#
# Source: examples/2_Intermediate/stage_two_optimization.py
#
# The JAX lane now includes the same fixed-state composite objective terms as
# the CPU lane through public curve-objective JAX wrappers. This row is a full
# fixed-state CPU/C++ vs JAX CPU objective and gradient comparison.


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
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import (
        CurveCurveDistance,
        CurveLength,
        CurveSurfaceDistance,
        LpCurveCurvature,
        MeanSquaredCurvature,
        SurfaceRZFourier,
        create_equally_spaced_curves,
    )
    from simsopt.objectives import QuadraticPenalty, SquaredFlux
    from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX
    from simsopt.geo.curveobjectives_jax import (
        CurveCurveDistanceJAX,
        CurveLengthJAX,
        CurveSurfaceDistanceJAX,
        LpCurveCurvatureJAX,
        MeanSquaredCurvatureJAX,
    )

    start_setup = time.perf_counter()
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
        jf_cpu=jf_full,
        bs_cpu=bs_cpu,
        target_array=None,
        extra_components=extra_components_cpu,
        setup_seconds=setup_seconds_cpu,
        objective_component_name="JF_total_cpu",
    )

    # JAX lane — build independent coils so neither lane mutates the other's
    # Optimizable tree.
    start_jax_setup = time.perf_counter()
    base_curves_jax = create_equally_spaced_curves(
        ncoils, surface.nfp, stellsym=True, R0=R0, R1=R1, order=order
    )
    base_currents_jax = [Current(1e5) for _ in range(ncoils)]
    base_currents_jax[0].fix_all()
    coils_jax = coils_via_symmetries(
        base_curves_jax, base_currents_jax, surface.nfp, True
    )

    bs_jax = BiotSavartJAX(coils_jax)
    jf_jax = SquaredFluxJAX(surface, bs_jax)
    curves_jax = [c.curve for c in coils_jax]
    Jls_jax = [CurveLengthJAX(c) for c in base_curves_jax]
    Jccdist_jax = CurveCurveDistanceJAX(
        curves_jax,
        CC_THRESHOLD,
        num_basecurves=ncoils,
    )
    Jcsdist_jax = CurveSurfaceDistanceJAX(curves_jax, surface, CS_THRESHOLD)
    Jcs_jax = [LpCurveCurvatureJAX(c, 2, CURVATURE_THRESHOLD) for c in base_curves_jax]
    Jmscs_jax = [MeanSquaredCurvatureJAX(c) for c in base_curves_jax]
    msc_quadratic_terms_jax = [
        QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs_jax
    ]
    jf_full_jax = (
        jf_jax
        + LENGTH_WEIGHT * sum(Jls_jax)
        + CC_WEIGHT * Jccdist_jax
        + CS_WEIGHT * Jcsdist_jax
        + CURVATURE_WEIGHT * sum(Jcs_jax)
        + MSC_WEIGHT * sum(msc_quadratic_terms_jax)
    )
    sum_length_value_jax = float(sum(J.J() for J in Jls_jax))
    ccdist_value_jax = float(Jccdist_jax.J())
    csdist_value_jax = float(Jcsdist_jax.J())
    curvature_sum_value_jax = float(sum(J.J() for J in Jcs_jax))
    msc_quadratic_sum_value_jax = float(sum(J.J() for J in msc_quadratic_terms_jax))
    setup_seconds_jax = time.perf_counter() - start_jax_setup

    jax_lane = _build_jax_lane(
        surface=surface,
        coils=coils_jax,
        bs_jax=bs_jax,
        jf_jax=jf_full_jax,
        target_array=None,
        extra_components={
            "SquaredFluxJAX": float(jf_jax.J()),
            "sum_CurveLength_raw": sum_length_value_jax,
            "CurveCurveDistance_raw": ccdist_value_jax,
            "CurveSurfaceDistance_raw": csdist_value_jax,
            "sum_LpCurveCurvature_raw": curvature_sum_value_jax,
            "sum_QuadraticPenalty_MeanSquaredCurvature_max_raw": (
                msc_quadratic_sum_value_jax
            ),
            "sum_CurveLength_weighted": LENGTH_WEIGHT * sum_length_value_jax,
            "CurveCurveDistance_weighted": CC_WEIGHT * ccdist_value_jax,
            "CurveSurfaceDistance_weighted": CS_WEIGHT * csdist_value_jax,
            "sum_LpCurveCurvature_weighted": (
                CURVATURE_WEIGHT * curvature_sum_value_jax
            ),
            "sum_QuadraticPenalty_MeanSquaredCurvature_max_weighted": (
                MSC_WEIGHT * msc_quadratic_sum_value_jax
            ),
        },
        setup_seconds=setup_seconds_jax,
        objective_component_name="JF_total_jax",
    )

    x0 = np.asarray(jf_full.x, dtype=np.float64).copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        jf_full.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_full.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jf_full_jax.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_full_jax.J())

    return FixtureBuild(
        spec=FULL_STAGE2_COMPOSITE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
    )


# ---------------------------------------------------------------------------
# Phase 4 — P2 planar Stage-II composite fixture
#
# Source: examples/2_Intermediate/stage_two_optimization_planar_coils.py
#
# ``CurvePlanarFourier`` exposes ``to_spec()`` and the public JAX
# curve-objective wrappers cover the fixed-state planar length, distance,
# curvature, and linking-number penalties. This row compares the full
# fixed-state planar composite.


def _build_planar_stage2_composite():
    import time
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import (
        CurveCurveDistance,
        CurveLength,
        CurveSurfaceDistance,
        LinkingNumber,
        LpCurveCurvature,
        MeanSquaredCurvature,
        SurfaceRZFourier,
        create_equally_spaced_planar_curves,
    )
    from simsopt.objectives import QuadraticPenalty, SquaredFlux
    from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX
    from simsopt.geo.curveobjectives_jax import (
        CurveCurveDistanceJAX,
        CurveLengthJAX,
        CurveSurfaceDistanceJAX,
        LinkingNumberJAX,
        LpCurveCurvatureJAX,
        MeanSquaredCurvatureJAX,
    )

    start_setup = time.perf_counter()
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
        jf_cpu=jf_full,
        bs_cpu=bs_cpu,
        target_array=None,
        extra_components=extra_components_cpu,
        setup_seconds=setup_seconds_cpu,
        objective_component_name="JF_total_cpu",
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

    bs_jax = BiotSavartJAX(coils_jax)
    jf_jax = SquaredFluxJAX(surface, bs_jax)
    curves_jax = [c.curve for c in coils_jax]
    Jls_jax = [CurveLengthJAX(c) for c in base_curves_jax]
    length_quadratic_penalty_jax = QuadraticPenalty(sum(Jls_jax), LENGTH_QP_TARGET)
    Jccdist_jax = CurveCurveDistanceJAX(
        curves_jax,
        CC_THRESHOLD,
        num_basecurves=ncoils,
    )
    Jcsdist_jax = CurveSurfaceDistanceJAX(curves_jax, surface, CS_THRESHOLD)
    Jcs_jax = [LpCurveCurvatureJAX(c, 2, CURVATURE_THRESHOLD) for c in base_curves_jax]
    Jmscs_jax = [MeanSquaredCurvatureJAX(c) for c in base_curves_jax]
    msc_quadratic_terms_jax = [QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs_jax]
    linkNum_jax = LinkingNumberJAX(curves_jax)
    jf_full_jax = (
        jf_jax
        + LENGTH_WEIGHT * length_quadratic_penalty_jax
        + CC_WEIGHT * Jccdist_jax
        + CS_WEIGHT * Jcsdist_jax
        + CURVATURE_WEIGHT * sum(Jcs_jax)
        + MSC_WEIGHT * sum(msc_quadratic_terms_jax)
        + linkNum_jax
    )
    length_qp_value_jax = float(length_quadratic_penalty_jax.J())
    ccdist_value_jax = float(Jccdist_jax.J())
    csdist_value_jax = float(Jcsdist_jax.J())
    curvature_sum_value_jax = float(sum(J.J() for J in Jcs_jax))
    msc_quadratic_sum_value_jax = float(sum(J.J() for J in msc_quadratic_terms_jax))
    link_number_value_jax = float(linkNum_jax.J())
    setup_seconds_jax = time.perf_counter() - start_jax_setup

    jax_lane = _build_jax_lane(
        surface=surface,
        coils=coils_jax,
        bs_jax=bs_jax,
        jf_jax=jf_full_jax,
        target_array=None,
        extra_components={
            "SquaredFluxJAX": float(jf_jax.J()),
            "QuadraticPenalty_over_sum_CurveLength_identity_raw": (length_qp_value_jax),
            "CurveCurveDistance_raw": ccdist_value_jax,
            "CurveSurfaceDistance_raw": csdist_value_jax,
            "sum_LpCurveCurvature_raw": curvature_sum_value_jax,
            "sum_QuadraticPenalty_MeanSquaredCurvature_identity_raw": (
                msc_quadratic_sum_value_jax
            ),
            "LinkingNumber_raw": link_number_value_jax,
            "QuadraticPenalty_over_sum_CurveLength_identity_weighted": (
                LENGTH_WEIGHT * length_qp_value_jax
            ),
            "CurveCurveDistance_weighted": CC_WEIGHT * ccdist_value_jax,
            "CurveSurfaceDistance_weighted": CS_WEIGHT * csdist_value_jax,
            "sum_LpCurveCurvature_weighted": (
                CURVATURE_WEIGHT * curvature_sum_value_jax
            ),
            "sum_QuadraticPenalty_MeanSquaredCurvature_identity_weighted": (
                MSC_WEIGHT * msc_quadratic_sum_value_jax
            ),
            "LinkingNumber_weighted": link_number_value_jax,
        },
        setup_seconds=setup_seconds_jax,
        objective_component_name="JF_total_jax",
    )

    x0 = np.asarray(jf_full.x, dtype=np.float64).copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        jf_full.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_full.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jf_full_jax.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_full_jax.J())

    return FixtureBuild(
        spec=PLANAR_STAGE2_COMPOSITE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
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
#   * Length penalty (``sum(CurveLength)``): evaluated through
#     ``CurveLengthJAX`` over an independently loaded NCSX curve tree.
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
        "sum_CurveLength": np.array([length_sum_cpu_value], dtype=np.float64),
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
    from simsopt.geo.curveobjectives_jax import CurveLengthJAX

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
    length_sum_jax_value = float(sum(CurveLengthJAX(c).J() for c in base_curves_jax))

    setup_seconds_jax = time.perf_counter() - start_jax_setup

    jax_components = {
        "iota": iotas_jax_value,
        "major_radius": major_radius_jax_value,
        "nq_symmetric_ratio": nqs_jax_value,
        "G": solved_G,
        "sum_CurveLength": length_sum_jax_value,
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
        "sum_CurveLength": np.array([length_sum_jax_value], dtype=np.float64),
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

    return FixtureBuild(
        spec=BOOZER_QA_WRAPPERS_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
        cpu_native_subproblem_J=None,
        jax_native_subproblem_J=None,
        x0=None,
    )


# ---------------------------------------------------------------------------
# Wave 5 — examples/1_Simple/tracing_fieldlines_QA.py


def _build_tracing_fieldlines_qa_reduced_endpoint():
    """Reduced InterpolatedField/compute_fieldlines fixture from QA tracing."""
    import time
    import simsopt
    from simsopt.field import InterpolatedField
    from simsopt.field.interpolated_field_jax import InterpolatedFieldJAX
    from simsopt.field.tracing import (
        LevelsetStoppingCriterion,
        SurfaceClassifier,
        _compute_fieldlines_jax,
        compute_fieldlines,
    )
    from simsopt.geo import SurfaceRZFourier

    nphi = 32
    ntheta = 16
    interp_degree = 2
    interp_n_r = 5
    interp_n_phi = 8
    interp_n_z = 4
    R0 = [1.24]
    Z0 = [0.0]
    tmax = 20.0
    tol = 1e-12

    start_cpu = time.perf_counter()
    surface = SurfaceRZFourier.from_vmec_input(
        str(TESTS_FILES / "input.LandremanPaul2021_QA"),
        nphi=nphi,
        ntheta=ntheta,
        range="full torus",
    )
    nfp = surface.nfp
    bs_cpu = simsopt.load(EXAMPLES / "1_Simple" / "inputs" / "biot_savart_opt.json")
    gamma = np.asarray(surface.gamma(), dtype=np.float64)
    rs = np.linalg.norm(gamma[:, :, 0:2], axis=2)
    zs = gamma[:, :, 2]
    rrange = (float(np.min(rs)), float(np.max(rs)), interp_n_r)
    phirange = (0.0, float(2.0 * np.pi / nfp), interp_n_phi)
    zrange = (0.0, float(np.max(zs)), interp_n_z)
    phis = [float(0.25 * 2.0 * np.pi / nfp)]
    sc_fieldline = SurfaceClassifier(surface, h=0.1, p=2)

    def skip(rs_skip, phis_skip, zs_skip):
        rphiz = np.asarray([rs_skip, phis_skip, zs_skip]).T.copy()
        return list((sc_fieldline.evaluate_rphiz(rphiz) < -0.05).flatten())

    stopping_criteria = [LevelsetStoppingCriterion(sc_fieldline.dist)]

    cpu_field = InterpolatedField(
        bs_cpu,
        interp_degree,
        rrange,
        phirange,
        zrange,
        True,
        nfp=nfp,
        stellsym=True,
        skip=skip,
    )
    cpu_field.set_points(gamma.reshape((-1, 3)))
    cpu_field_B = np.asarray(cpu_field.B(), dtype=np.float64)
    cpu_tys, cpu_hits = compute_fieldlines(
        cpu_field,
        R0,
        Z0,
        tmax=tmax,
        tol=tol,
        phis=phis,
        stopping_criteria=stopping_criteria,
    )
    cpu_setup = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    bs_jax_source = simsopt.load(
        EXAMPLES / "1_Simple" / "inputs" / "biot_savart_opt.json"
    )
    jax_field = InterpolatedFieldJAX(
        bs_jax_source,
        interp_degree,
        rrange,
        phirange,
        zrange,
        True,
        nfp=nfp,
        stellsym=True,
        skip=skip,
    )
    jax_field.set_points(gamma.reshape((-1, 3)))
    jax_field_B = np.asarray(jax_field.B(), dtype=np.float64)
    jax_tys, jax_hits = _compute_fieldlines_jax(
        jax_field,
        R0,
        Z0,
        tmax=tmax,
        tol=tol,
        phis=phis,
        stopping_criteria=stopping_criteria,
        comm=None,
    )
    jax_setup = time.perf_counter() - start_jax

    cpu_traj = np.asarray(cpu_tys[0], dtype=np.float64)
    jax_traj = np.asarray(jax_tys[0], dtype=np.float64)
    cpu_phi_hits = np.asarray(cpu_hits[0], dtype=np.float64)
    jax_phi_hits = np.asarray(jax_hits[0], dtype=np.float64)
    hit_count = min(int(cpu_phi_hits.shape[0]), int(jax_phi_hits.shape[0]))
    cpu_hit_xyz = cpu_phi_hits[:hit_count, 2:5]
    jax_hit_xyz = jax_phi_hits[:hit_count, 2:5]
    cpu_t_final = float(cpu_traj[-1, 0])
    jax_t_final = float(jax_traj[-1, 0])
    cpu_status_code = _tracing_status_code(cpu_traj, tmax)
    jax_status_code = _tracing_status_code(jax_traj, tmax)

    empty_grid = np.zeros(gamma.shape[:2], dtype=np.float64)
    active_dofs = np.array([], dtype=np.float64)
    free_mask = np.array([], dtype=bool)
    cpu_raw_arrays = {
        "field_B": cpu_field_B.reshape(gamma.shape),
        "surface_gamma": gamma,
        "surface_unit_normal": np.asarray(surface.unitnormal(), dtype=np.float64),
        "Bdotn": empty_grid,
        "trajectory_endpoint": cpu_traj[-1:, 1:4],
        "trajectory_t_final": np.array([cpu_t_final], dtype=np.float64),
        "trajectory_status_code": np.array([cpu_status_code], dtype=np.float64),
        "phi_hit_xyz": cpu_hit_xyz,
        "phi_hit_count": np.array([cpu_phi_hits.shape[0]], dtype=np.float64),
        "trajectory_step_count": np.array([cpu_traj.shape[0]], dtype=np.float64),
    }
    jax_raw_arrays = {
        "field_B": jax_field_B.reshape(gamma.shape),
        "surface_gamma": gamma,
        "surface_unit_normal": np.asarray(surface.unitnormal(), dtype=np.float64),
        "Bdotn": empty_grid,
        "trajectory_endpoint": jax_traj[-1:, 1:4],
        "trajectory_t_final": np.array([jax_t_final], dtype=np.float64),
        "trajectory_status_code": np.array([jax_status_code], dtype=np.float64),
        "phi_hit_xyz": jax_hit_xyz,
        "phi_hit_count": np.array([jax_phi_hits.shape[0]], dtype=np.float64),
        "trajectory_step_count": np.array([jax_traj.shape[0]], dtype=np.float64),
    }

    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=None,
        objective_native_subtotal=None,
        components=_flatten_components({"tmax": tmax, "phi_hit_count": hit_count}),
        gradient=None,
        gradient_norm=None,
        active_dof_names=(),
        active_dof_hash=_hash_array(active_dofs),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma),
        unit_normal_hash=_hash_array(cpu_raw_arrays["surface_unit_normal"]),
        field_B_hash=_hash_array(cpu_raw_arrays["field_B"]),
        field_B_max=float(np.max(np.abs(cpu_raw_arrays["field_B"]))),
        field_B_mean=float(np.mean(np.abs(cpu_raw_arrays["field_B"]))),
        Bdotn_array_hash=_hash_array(empty_grid),
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays=cpu_raw_arrays,
        timing={"setup_s": float(cpu_setup), "execute_s": 0.0},
    )
    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=None,
        objective_native_subtotal=None,
        components=_flatten_components({"tmax": tmax, "phi_hit_count": hit_count}),
        gradient=None,
        gradient_norm=None,
        active_dof_names=(),
        active_dof_hash=_hash_array(active_dofs),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma),
        unit_normal_hash=_hash_array(jax_raw_arrays["surface_unit_normal"]),
        field_B_hash=_hash_array(jax_raw_arrays["field_B"]),
        field_B_max=float(np.max(np.abs(jax_raw_arrays["field_B"]))),
        field_B_mean=float(np.mean(np.abs(jax_raw_arrays["field_B"]))),
        Bdotn_array_hash=_hash_array(empty_grid),
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays=jax_raw_arrays,
        timing={"setup_s": float(jax_setup), "execute_s": 0.0},
    )

    return FixtureBuild(
        spec=TRACING_FIELDLINES_QA_REDUCED_ENDPOINT_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
    )


def _build_tracing_fieldlines_ncsx_reduced_endpoint():
    """Reduced InterpolatedField/compute_fieldlines fixture from NCSX tracing."""
    import time
    from simsopt.configs import get_data
    from simsopt.field import InterpolatedField
    from simsopt.field.interpolated_field_jax import InterpolatedFieldJAX
    from simsopt.field.tracing import (
        LevelsetStoppingCriterion,
        SurfaceClassifier,
        _compute_fieldlines_jax,
        compute_fieldlines,
    )
    from simsopt.geo import SurfaceRZFourier

    mpol = 5
    ntor = 5
    nphi = 32
    ntheta = 12
    interp_degree = 2
    interp_n_r = 5
    interp_n_phi = 8
    interp_n_z = 4
    tmax = 20.0
    tol = 1e-12
    surface_radius = 0.70

    start_cpu = time.perf_counter()
    _base_curves, _currents, axis_cpu, nfp, bs_cpu = get_data("ncsx")
    surface = SurfaceRZFourier.from_nphi_ntheta(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        range="full torus",
        nphi=nphi,
        ntheta=ntheta,
    )
    surface.fit_to_curve(axis_cpu, surface_radius, flip_theta=False)
    gamma = np.asarray(surface.gamma(), dtype=np.float64)
    unit_normal = np.asarray(surface.unitnormal(), dtype=np.float64)
    rs = np.linalg.norm(gamma[:, :, 0:2], axis=2)
    zs = gamma[:, :, 2]
    rrange = (float(np.min(rs)), float(np.max(rs)), interp_n_r)
    phirange = (0.0, float(2.0 * np.pi / nfp), interp_n_phi)
    zrange = (0.0, float(np.max(zs)), interp_n_z)
    axis_gamma = np.asarray(axis_cpu.gamma(), dtype=np.float64)
    R0 = [float(axis_gamma[0, 0])]
    Z0 = [float(axis_gamma[0, 2])]
    phis = [float(0.25 * 2.0 * np.pi / nfp)]
    sc_fieldline = SurfaceClassifier(surface, h=0.1, p=2)

    def skip(rs_skip, phis_skip, zs_skip):
        rphiz = np.asarray([rs_skip, phis_skip, zs_skip]).T.copy()
        return list((sc_fieldline.evaluate_rphiz(rphiz) < -0.05).flatten())

    stopping_criteria = [LevelsetStoppingCriterion(sc_fieldline.dist)]

    cpu_field = InterpolatedField(
        bs_cpu,
        interp_degree,
        rrange,
        phirange,
        zrange,
        True,
        nfp=nfp,
        stellsym=True,
        skip=skip,
    )
    cpu_field.set_points(gamma.reshape((-1, 3)))
    cpu_field_B = np.asarray(cpu_field.B(), dtype=np.float64)
    cpu_tys, cpu_hits = compute_fieldlines(
        cpu_field,
        R0,
        Z0,
        tmax=tmax,
        tol=tol,
        phis=phis,
        stopping_criteria=stopping_criteria,
    )
    cpu_setup = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    _base_curves_jax, _currents_jax, _axis_jax, nfp_jax, bs_jax_source = get_data(
        "ncsx"
    )
    jax_field = InterpolatedFieldJAX(
        bs_jax_source,
        interp_degree,
        rrange,
        phirange,
        zrange,
        True,
        nfp=nfp_jax,
        stellsym=True,
        skip=skip,
    )
    jax_field.set_points(gamma.reshape((-1, 3)))
    jax_field_B = np.asarray(jax_field.B(), dtype=np.float64)
    jax_tys, jax_hits = _compute_fieldlines_jax(
        jax_field,
        R0,
        Z0,
        tmax=tmax,
        tol=tol,
        phis=phis,
        stopping_criteria=stopping_criteria,
        comm=None,
    )
    jax_setup = time.perf_counter() - start_jax

    cpu_traj = np.asarray(cpu_tys[0], dtype=np.float64)
    jax_traj = np.asarray(jax_tys[0], dtype=np.float64)
    cpu_phi_hits = np.asarray(cpu_hits[0], dtype=np.float64)
    jax_phi_hits = np.asarray(jax_hits[0], dtype=np.float64)
    hit_count = min(int(cpu_phi_hits.shape[0]), int(jax_phi_hits.shape[0]))
    cpu_t_final = float(cpu_traj[-1, 0])
    jax_t_final = float(jax_traj[-1, 0])
    cpu_status_code = _tracing_status_code(cpu_traj, tmax)
    jax_status_code = _tracing_status_code(jax_traj, tmax)

    empty_grid = np.zeros(gamma.shape[:2], dtype=np.float64)
    active_dofs = np.array([], dtype=np.float64)
    free_mask = np.array([], dtype=bool)
    cpu_raw_arrays = {
        "field_B": cpu_field_B.reshape(gamma.shape),
        "surface_gamma": gamma,
        "surface_unit_normal": unit_normal,
        "Bdotn": empty_grid,
        "trajectory_endpoint": cpu_traj[-1:, 1:4],
        "trajectory_t_final": np.array([cpu_t_final], dtype=np.float64),
        "trajectory_status_code": np.array([cpu_status_code], dtype=np.float64),
        "phi_hit_xyz": cpu_phi_hits[:hit_count, 2:5],
        "phi_hit_count": np.array([cpu_phi_hits.shape[0]], dtype=np.float64),
        "trajectory_step_count": np.array([cpu_traj.shape[0]], dtype=np.float64),
    }
    jax_raw_arrays = {
        "field_B": jax_field_B.reshape(gamma.shape),
        "surface_gamma": gamma,
        "surface_unit_normal": unit_normal,
        "Bdotn": empty_grid,
        "trajectory_endpoint": jax_traj[-1:, 1:4],
        "trajectory_t_final": np.array([jax_t_final], dtype=np.float64),
        "trajectory_status_code": np.array([jax_status_code], dtype=np.float64),
        "phi_hit_xyz": jax_phi_hits[:hit_count, 2:5],
        "phi_hit_count": np.array([jax_phi_hits.shape[0]], dtype=np.float64),
        "trajectory_step_count": np.array([jax_traj.shape[0]], dtype=np.float64),
    }
    components = _flatten_components(
        {
            "tmax": tmax,
            "tol": tol,
            "phi_hit_count": hit_count,
            "fieldline_count": 1,
            "nfp": nfp,
        }
    )
    cpu_lane = LaneArtifact(
        lane="cpu_cpp",
        objective_total=None,
        objective_native_subtotal=None,
        components=components,
        gradient=None,
        gradient_norm=None,
        active_dof_names=(),
        active_dof_hash=_hash_array(active_dofs),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma),
        unit_normal_hash=_hash_array(unit_normal),
        field_B_hash=_hash_array(cpu_raw_arrays["field_B"]),
        field_B_max=float(np.max(np.abs(cpu_raw_arrays["field_B"]))),
        field_B_mean=float(np.mean(np.abs(cpu_raw_arrays["field_B"]))),
        Bdotn_array_hash=_hash_array(empty_grid),
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays=cpu_raw_arrays,
        timing={"setup_s": float(cpu_setup), "execute_s": 0.0},
    )
    jax_lane = LaneArtifact(
        lane="jax_cpu",
        objective_total=None,
        objective_native_subtotal=None,
        components=components,
        gradient=None,
        gradient_norm=None,
        active_dof_names=(),
        active_dof_hash=_hash_array(active_dofs),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(gamma),
        unit_normal_hash=_hash_array(unit_normal),
        field_B_hash=_hash_array(jax_raw_arrays["field_B"]),
        field_B_max=float(np.max(np.abs(jax_raw_arrays["field_B"]))),
        field_B_mean=float(np.mean(np.abs(jax_raw_arrays["field_B"]))),
        Bdotn_array_hash=_hash_array(empty_grid),
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays=jax_raw_arrays,
        timing={"setup_s": float(jax_setup), "execute_s": 0.0},
    )

    return FixtureBuild(
        spec=TRACING_FIELDLINES_NCSX_REDUCED_ENDPOINT_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
    )


def _tracing_event_state_columns(hits: np.ndarray) -> np.ndarray:
    hits_arr = np.asarray(hits, dtype=np.float64)
    if hits_arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return hits_arr.reshape((-1, hits_arr.shape[-1]))[:, 2:5]


def _tracing_event_count(hits: np.ndarray) -> int:
    hits_arr = np.asarray(hits, dtype=np.float64)
    if hits_arr.size == 0:
        return 0
    return int(hits_arr.reshape((-1, hits_arr.shape[-1])).shape[0])


def _tracing_status_code(traj: np.ndarray, tmax: float) -> int:
    return 0 if float(traj[-1, 0]) >= float(tmax) - 1e-15 else 1


def _tracing_lane_artifact(
    *,
    lane: str,
    components: Mapping[str, float],
    raw_arrays: Mapping[str, np.ndarray],
    field_hash_array: np.ndarray,
    setup_seconds: float,
) -> LaneArtifact:
    active_dofs = np.array([], dtype=np.float64)
    free_mask = np.array([], dtype=bool)
    Bdotn = np.asarray(raw_arrays["Bdotn"], dtype=np.float64)
    field_values = np.asarray(field_hash_array, dtype=np.float64)
    return LaneArtifact(
        lane=lane,
        objective_total=None,
        objective_native_subtotal=None,
        components=_flatten_components(components),
        gradient=None,
        gradient_norm=None,
        active_dof_names=(),
        active_dof_hash=_hash_array(active_dofs),
        fixed_free_mask_hash=_hash_mask(free_mask),
        native_curve_spec_hashes=(),
        surface_point_hash=_hash_array(raw_arrays["surface_gamma"]),
        unit_normal_hash=_hash_array(raw_arrays["surface_unit_normal"]),
        field_B_hash=_hash_array(field_values),
        field_B_max=float(np.max(np.abs(field_values))),
        field_B_mean=float(np.mean(np.abs(field_values))),
        Bdotn_array_hash=_hash_array(Bdotn),
        Bdotn_max=float(np.max(np.abs(Bdotn))) if Bdotn.size else 0.0,
        Bdotn_mean=float(np.mean(np.abs(Bdotn))) if Bdotn.size else 0.0,
        raw_arrays=raw_arrays,
        timing={"setup_s": float(setup_seconds), "execute_s": 0.0},
    )


def _build_tracing_particle_gc_vac_reduced_endpoint():
    """Reduced InterpolatedField particle-GC fixture from tracing_particle."""
    import time
    from math import sqrt

    from simsopt.configs import get_data
    from simsopt.field import InterpolatedField
    from simsopt.field.interpolated_field_jax import InterpolatedFieldJAX
    from simsopt.field.sampling import draw_uniform_on_curve
    from simsopt.field.tracing import (
        LevelsetStoppingCriterion,
        SurfaceClassifier,
        _trace_particles_jax_guiding_center_vacuum,
        trace_particles,
    )
    from simsopt.geo import SurfaceRZFourier
    from simsopt.util.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS

    mpol = 5
    ntor = 5
    nphi = 32
    ntheta = 12
    interp_degree = 2
    interp_n = 5
    nparticles = 1
    seed = 1
    surface_radius = 0.20
    tmax = 1e-7
    tol = 1e-9
    Ekin = 5000.0 * ONE_EV
    mass = PROTON_MASS
    charge = ELEMENTARY_CHARGE

    start_cpu = time.perf_counter()
    _base_curves, _base_currents, ma, nfp, bs_cpu = get_data("ncsx")
    surface = SurfaceRZFourier.from_nphi_ntheta(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        range="full torus",
        nphi=nphi,
        ntheta=ntheta,
    )
    surface.fit_to_curve(ma, surface_radius, flip_theta=False)
    classifier = SurfaceClassifier(surface, h=0.1, p=2)
    gamma = np.asarray(surface.gamma(), dtype=np.float64)
    unit_normal = np.asarray(surface.unitnormal(), dtype=np.float64)
    rs = np.linalg.norm(gamma[:, :, 0:2], axis=2)
    zs = gamma[:, :, 2]
    rrange = (float(np.min(rs)), float(np.max(rs)), interp_n)
    phirange = (0.0, float(2.0 * np.pi / nfp), 2 * interp_n)
    zrange = (0.0, float(np.max(zs)), interp_n // 2)
    phis = [float((i / 4.0) * (2.0 * np.pi / nfp)) for i in range(4)]

    speed_total = sqrt(2.0 * Ekin / mass)
    np.random.seed(seed)
    pitch = np.random.uniform(low=-1.0, high=1.0, size=(nparticles,))
    speed_par = pitch * speed_total
    xyz_inits, _ = draw_uniform_on_curve(ma, nparticles, safetyfactor=10)
    stopping_criteria = [LevelsetStoppingCriterion(classifier.dist)]

    cpu_field = InterpolatedField(
        bs_cpu,
        interp_degree,
        rrange,
        phirange,
        zrange,
        True,
        nfp=nfp,
        stellsym=True,
    )
    cpu_tys, cpu_hits = trace_particles(
        cpu_field,
        xyz_inits,
        speed_par,
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        phis=phis,
        stopping_criteria=stopping_criteria,
        mode="gc_vac",
        forget_exact_path=True,
    )
    cpu_field.set_points(gamma.reshape((-1, 3)))
    cpu_field_B = np.asarray(cpu_field.B(), dtype=np.float64).reshape(gamma.shape)
    cpu_field_GradAbsB = np.asarray(cpu_field.GradAbsB(), dtype=np.float64).reshape(
        gamma.shape
    )
    cpu_setup = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    _base_curves_jax, _base_currents_jax, _ma_jax, nfp_jax, bs_jax = get_data("ncsx")
    jax_field = InterpolatedFieldJAX(
        bs_jax,
        interp_degree,
        rrange,
        phirange,
        zrange,
        True,
        nfp=nfp_jax,
        stellsym=True,
    )
    jax_tys, jax_hits = _trace_particles_jax_guiding_center_vacuum(
        jax_field,
        xyz_inits,
        speed_par,
        speed_total,
        tmax=tmax,
        mass=mass,
        charge=charge,
        tol=tol,
        comm=None,
        phis=phis,
        stopping_criteria=stopping_criteria,
        mode="gc_vac",
        forget_exact_path=True,
    )
    jax_field.set_points(gamma.reshape((-1, 3)))
    jax_field_B = np.asarray(jax_field.B(), dtype=np.float64).reshape(gamma.shape)
    jax_field_GradAbsB = np.asarray(jax_field.GradAbsB(), dtype=np.float64).reshape(
        gamma.shape
    )
    jax_setup = time.perf_counter() - start_jax

    cpu_traj = np.asarray(cpu_tys[0], dtype=np.float64)
    jax_traj = np.asarray(jax_tys[0], dtype=np.float64)
    cpu_phi_hits = np.asarray(cpu_hits[0], dtype=np.float64)
    jax_phi_hits = np.asarray(jax_hits[0], dtype=np.float64)
    hit_count = min(
        _tracing_event_count(cpu_phi_hits), _tracing_event_count(jax_phi_hits)
    )
    empty_grid = np.zeros(gamma.shape[:2], dtype=np.float64)
    cpu_raw_arrays = {
        "field_B": cpu_field_B,
        "field_GradAbsB": cpu_field_GradAbsB,
        "surface_gamma": gamma,
        "surface_unit_normal": unit_normal,
        "Bdotn": empty_grid,
        "trajectory_endpoint": cpu_traj[-1:, 1:5],
        "trajectory_t_final": np.array([cpu_traj[-1, 0]], dtype=np.float64),
        "trajectory_status_code": np.array(
            [_tracing_status_code(cpu_traj, tmax)], dtype=np.float64
        ),
        "phi_hit_xyz": _tracing_event_state_columns(cpu_phi_hits)[:hit_count],
        "phi_hit_count": np.array(
            [_tracing_event_count(cpu_phi_hits)], dtype=np.float64
        ),
        "trajectory_step_count": np.array([cpu_traj.shape[0]], dtype=np.float64),
    }
    jax_raw_arrays = {
        "field_B": jax_field_B,
        "field_GradAbsB": jax_field_GradAbsB,
        "surface_gamma": gamma,
        "surface_unit_normal": unit_normal,
        "Bdotn": empty_grid,
        "trajectory_endpoint": jax_traj[-1:, 1:5],
        "trajectory_t_final": np.array([jax_traj[-1, 0]], dtype=np.float64),
        "trajectory_status_code": np.array(
            [_tracing_status_code(jax_traj, tmax)], dtype=np.float64
        ),
        "phi_hit_xyz": _tracing_event_state_columns(jax_phi_hits)[:hit_count],
        "phi_hit_count": np.array(
            [_tracing_event_count(jax_phi_hits)], dtype=np.float64
        ),
        "trajectory_step_count": np.array([jax_traj.shape[0]], dtype=np.float64),
    }
    components = {
        "tmax": tmax,
        "tol": tol,
        "particle_count": nparticles,
        "phi_hit_count": hit_count,
        "nfp": nfp,
        "Ekin_eV": 5000.0,
        "seed": seed,
    }

    return FixtureBuild(
        spec=TRACING_PARTICLE_GC_VAC_REDUCED_ENDPOINT_SPEC,
        cpu_lane=_tracing_lane_artifact(
            lane="cpu_cpp",
            components=components,
            raw_arrays=cpu_raw_arrays,
            field_hash_array=cpu_field_B,
            setup_seconds=cpu_setup,
        ),
        jax_lane=_tracing_lane_artifact(
            lane="jax_cpu",
            components=components,
            raw_arrays=jax_raw_arrays,
            field_hash_array=jax_field_B,
            setup_seconds=jax_setup,
        ),
        unsupported_components=(),
    )


@dataclass(frozen=True)
class _CachedBoozXformState:
    asym: bool
    nfp: int
    mboz: int
    nboz: int
    ns_b: int
    ns_in: int
    s_in: np.ndarray
    s_b: np.ndarray
    xm_b: np.ndarray
    xn_b: np.ndarray
    iota: np.ndarray
    Boozer_G: np.ndarray
    Boozer_I: np.ndarray
    bmnc_b: np.ndarray
    rmnc_b: np.ndarray
    zmns_b: np.ndarray
    numns_b: np.ndarray
    bmns_b: np.ndarray
    rmns_b: np.ndarray
    zmnc_b: np.ndarray
    numnc_b: np.ndarray


def _cached_boozer_from_boozmn(wout_file: Path, boozmn_file: Path):
    """Build a Boozer object from checked-in BOOZXFORM output."""
    from scipy.io import netcdf_file
    from simsopt.mhd import Boozer, Vmec

    vmec = Vmec(str(wout_file))
    boozmn = netcdf_file(boozmn_file, mmap=False)
    try:
        jlist = np.asarray(boozmn.variables["jlist"][()], dtype=np.int64).copy()
        surface_count = int(jlist.size)
        ns_full = int(np.asarray(boozmn.variables["ns_b"][()]).item())
        mode_count = int(np.asarray(boozmn.variables["mnboz_b"][()]).item())
        s_in = (jlist.astype(np.float64) - 1.5) / float(ns_full - 1)
        radial_indices = jlist - 1

        def _flux_function(name: str) -> np.ndarray:
            values = np.asarray(boozmn.variables[name][()], dtype=np.float64)
            return values[radial_indices].copy()

        def _mode_matrix(name: str) -> np.ndarray:
            values = np.asarray(boozmn.variables[name][()], dtype=np.float64)
            return values[:, :mode_count].T.copy()

        asym = bool(np.asarray(boozmn.variables["lasym__logical__"][()]).item())
        empty_asym = np.zeros((mode_count, surface_count), dtype=np.float64)
        bx = _CachedBoozXformState(
            asym=asym,
            nfp=int(np.asarray(boozmn.variables["nfp_b"][()]).item()),
            mboz=int(np.asarray(boozmn.variables["mboz_b"][()]).item()),
            nboz=int(np.asarray(boozmn.variables["nboz_b"][()]).item()),
            ns_b=surface_count,
            ns_in=surface_count,
            s_in=s_in,
            s_b=s_in.copy(),
            xm_b=np.asarray(boozmn.variables["ixm_b"][()], dtype=np.float64)[
                :mode_count
            ].copy(),
            xn_b=np.asarray(boozmn.variables["ixn_b"][()], dtype=np.float64)[
                :mode_count
            ].copy(),
            iota=_flux_function("iota_b"),
            Boozer_G=_flux_function("bvco_b"),
            Boozer_I=_flux_function("buco_b"),
            bmnc_b=_mode_matrix("bmnc_b"),
            rmnc_b=_mode_matrix("rmnc_b"),
            zmns_b=_mode_matrix("zmns_b"),
            numns_b=_mode_matrix("pmns_b"),
            bmns_b=empty_asym.copy(),
            rmns_b=empty_asym.copy(),
            zmnc_b=empty_asym.copy(),
            numnc_b=empty_asym.copy(),
        )
    finally:
        boozmn.close()

    booz = Boozer.__new__(Boozer)
    booz.equil = vmec
    booz.bx = bx
    booz.mpi = None
    booz.s = set(float(value) for value in s_in)
    booz.s_to_index = {float(value): index for index, value in enumerate(s_in)}
    booz.need_to_run_code = False
    booz._calls = 0
    return booz


def _build_tracing_boozer_gc_reduced_endpoint():
    """Reduced InterpolatedBoozerField GC fixture from tracing_boozer."""
    import time

    from simsopt.field import (
        BoozerRadialInterpolant,
        InterpolatedBoozerField,
        MaxToroidalFluxStoppingCriterion,
        MinToroidalFluxStoppingCriterion,
        ToroidalTransitStoppingCriterion,
        trace_particles_boozer,
    )
    from simsopt.field.boozermagneticfield_jax import InterpolatedBoozerFieldJAX
    from simsopt.field.tracing import _trace_particles_boozer_jax
    from simsopt.jax_core.tracing import _BOOZER_RHS_EVAL_KEYS
    from simsopt.util.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS

    wout_file = TESTS_FILES / "wout_circular_tokamak_reference.nc"
    boozmn_file = TESTS_FILES / "boozmn_circular_tokamak.nc"
    order = 3
    degree = 2
    grid_n = 5
    stz_inits = np.array([[0.30, 0.0, 0.0]], dtype=np.float64)
    field_points = np.array([[0.30, 0.0, 0.0], [0.40, 0.20, 0.10]], dtype=np.float64)
    tmax = 1e-7
    tol = 1e-9
    Ekin = 1000.0 * ONE_EV
    mass = PROTON_MASS
    charge = ELEMENTARY_CHARGE
    speed_total = np.sqrt(2.0 * Ekin / mass)
    speed_par = np.array([0.6 * speed_total], dtype=np.float64)

    start_cpu = time.perf_counter()
    booz_cpu = _cached_boozer_from_boozmn(wout_file, boozmn_file)
    bri_cpu = BoozerRadialInterpolant(
        booz_cpu,
        order=order,
        mpol=int(booz_cpu.bx.mboz),
        ntor=int(booz_cpu.bx.nboz),
        rescale=True,
        enforce_vacuum=True,
    )
    nfp = int(booz_cpu.equil.wout.nfp)
    srange = (0.0, 1.0, grid_n)
    thetarange = (0.0, np.pi, grid_n)
    zetarange = (0.0, float(2.0 * np.pi / nfp), grid_n)
    zetas = [float(0.25 * 2.0 * np.pi / nfp)]
    stopping_criteria = [
        MinToroidalFluxStoppingCriterion(0.01),
        MaxToroidalFluxStoppingCriterion(0.99),
        ToroidalTransitStoppingCriterion(100, True),
    ]
    cpu_field = InterpolatedBoozerField(
        bri_cpu,
        degree,
        srange,
        thetarange,
        zetarange,
        True,
        nfp=nfp,
        stellsym=True,
    )
    cpu_tys, cpu_hits = trace_particles_boozer(
        cpu_field,
        stz_inits,
        speed_par,
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        zetas=zetas,
        stopping_criteria=stopping_criteria,
        mode="gc_vac",
        forget_exact_path=True,
    )
    cpu_field.set_points(field_points)
    cpu_modB = np.asarray(cpu_field.modB(), dtype=np.float64)
    cpu_setup = time.perf_counter() - start_cpu

    start_jax = time.perf_counter()
    booz_jax = _cached_boozer_from_boozmn(wout_file, boozmn_file)
    bri_jax = BoozerRadialInterpolant(
        booz_jax,
        order=order,
        mpol=int(booz_jax.bx.mboz),
        ntor=int(booz_jax.bx.nboz),
        rescale=True,
        enforce_vacuum=True,
    )
    jax_field = InterpolatedBoozerFieldJAX(
        bri_jax,
        degree,
        srange,
        thetarange,
        zetarange,
        True,
        nfp=nfp,
        stellsym=True,
        scalars=_BOOZER_RHS_EVAL_KEYS,
    )
    jax_tys, jax_hits = _trace_particles_boozer_jax(
        jax_field,
        stz_inits,
        speed_par,
        speed_total,
        tmax=tmax,
        mass=mass,
        charge=charge,
        tol=tol,
        comm=None,
        zetas=zetas,
        stopping_criteria=stopping_criteria,
        mode="gc_vac",
        forget_exact_path=True,
    )
    jax_field.set_points(field_points)
    jax_modB = np.asarray(jax_field.modB(), dtype=np.float64)
    jax_setup = time.perf_counter() - start_jax

    cpu_traj = np.asarray(cpu_tys[0], dtype=np.float64)
    jax_traj = np.asarray(jax_tys[0], dtype=np.float64)
    cpu_zeta_hits = np.asarray(cpu_hits[0], dtype=np.float64)
    jax_zeta_hits = np.asarray(jax_hits[0], dtype=np.float64)
    hit_count = min(
        _tracing_event_count(cpu_zeta_hits), _tracing_event_count(jax_zeta_hits)
    )
    field_points_grid = field_points.reshape((field_points.shape[0], 1, 3))
    empty_grid = np.zeros((field_points.shape[0], 1), dtype=np.float64)
    unit_normal = np.zeros_like(field_points_grid)
    cpu_raw_arrays = {
        "field_modB": cpu_modB,
        "surface_gamma": field_points_grid,
        "surface_unit_normal": unit_normal,
        "Bdotn": empty_grid,
        "trajectory_endpoint": cpu_traj[-1:, 1:5],
        "trajectory_t_final": np.array([cpu_traj[-1, 0]], dtype=np.float64),
        "trajectory_status_code": np.array(
            [_tracing_status_code(cpu_traj, tmax)], dtype=np.float64
        ),
        "phi_hit_xyz": _tracing_event_state_columns(cpu_zeta_hits)[:hit_count],
        "phi_hit_count": np.array(
            [_tracing_event_count(cpu_zeta_hits)], dtype=np.float64
        ),
        "trajectory_step_count": np.array([cpu_traj.shape[0]], dtype=np.float64),
    }
    jax_raw_arrays = {
        "field_modB": jax_modB,
        "surface_gamma": field_points_grid,
        "surface_unit_normal": unit_normal,
        "Bdotn": empty_grid,
        "trajectory_endpoint": jax_traj[-1:, 1:5],
        "trajectory_t_final": np.array([jax_traj[-1, 0]], dtype=np.float64),
        "trajectory_status_code": np.array(
            [_tracing_status_code(jax_traj, tmax)], dtype=np.float64
        ),
        "phi_hit_xyz": _tracing_event_state_columns(jax_zeta_hits)[:hit_count],
        "phi_hit_count": np.array(
            [_tracing_event_count(jax_zeta_hits)], dtype=np.float64
        ),
        "trajectory_step_count": np.array([jax_traj.shape[0]], dtype=np.float64),
    }
    components = {
        "tmax": tmax,
        "tol": tol,
        "particle_count": 1,
        "zeta_hit_count": hit_count,
        "nfp": nfp,
        "Ekin_eV": 1000.0,
        "interpolation_degree": degree,
    }

    return FixtureBuild(
        spec=TRACING_BOOZER_GC_REDUCED_ENDPOINT_SPEC,
        cpu_lane=_tracing_lane_artifact(
            lane="cpu_cpp",
            components=components,
            raw_arrays=cpu_raw_arrays,
            field_hash_array=cpu_modB,
            setup_seconds=cpu_setup,
        ),
        jax_lane=_tracing_lane_artifact(
            lane="jax_cpu",
            components=components,
            raw_arrays=jax_raw_arrays,
            field_hash_array=jax_modB,
            setup_seconds=jax_setup,
        ),
        unsupported_components=("VMEC_input_external_solver",),
    )


# ---------------------------------------------------------------------------
# Unsupported / support-gate classification builders


def _raise_unsupported(message: str) -> Callable[[], FixtureBuild]:
    def _factory() -> FixtureBuild:
        raise FixtureNotSupportedError(message)

    return _factory


# ---------------------------------------------------------------------------
# Phase 5 — Position/orientation fixed-state fixture
#
# Source: examples/1_Simple/optimize_coil_position_orientation.py
#
# The harness builds the reduced TF+windowpane coil fixture without running the
# optimizer, preserving the example's active position/orientation DOFs while
# comparing the fixed-state flux path on CPU/C++ and JAX CPU.


def _build_position_orientation_flux_fixed_state():
    """Build reduced TF+windowpane flux parity without optimizer execution."""
    import time
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.field.coil import ScaledCurrent
    from simsopt.geo import (
        SurfaceRZFourier,
        create_equally_spaced_curves,
        create_equally_spaced_oriented_curves,
    )
    from simsopt.objectives import SquaredFlux
    from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX

    nphi = 32
    ntheta = 32
    n_tf_coils = 4
    n_wp_coils = 2
    R0 = 1.0
    R1 = 0.5

    start_cpu = time.perf_counter()
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

    setup_cpu = time.perf_counter() - start_cpu
    cpu_lane = _build_cpu_lane(
        surface=surface,
        coils=coils,
        jf_cpu=jf_cpu,
        bs_cpu=bs_cpu,
        target_array=None,
        extra_components={
            "active_free_dof_count": float(len(active_dof_names)),
            "tf_coil_count": float(len(tf_coils)),
            "wp_coil_count": float(len(wp_coils)),
        },
        setup_seconds=setup_cpu,
    )

    start_jax = time.perf_counter()
    base_tf_curves_jax = create_equally_spaced_curves(
        n_tf_coils, surface.nfp, stellsym=True, R0=R0, R1=R1, order=2
    )
    base_wp_curves_jax = create_equally_spaced_oriented_curves(
        n_wp_coils, surface.nfp, R0=(R0 + R1) * 1.01, R1=R1 / 10, Z0=0, order=2
    )
    base_tf_currents_jax = [ScaledCurrent(Current(1.0), 1e5) for _ in range(n_tf_coils)]
    base_wp_currents_jax = [ScaledCurrent(Current(1.0), 1e3) for _ in range(n_wp_coils)]
    for curve in base_tf_curves_jax:
        curve.fix_all()
    for curve in base_wp_curves_jax:
        curve.fix_all()
        for xyz in ("x0", "y0", "z0"):
            curve.unfix(xyz)
        for ypr in ("yaw", "pitch", "roll"):
            curve.unfix(ypr)
    for current in base_tf_currents_jax:
        current.unfix_all()
    for current in base_wp_currents_jax:
        current.unfix_all()
    base_tf_currents_jax[0].fix_all()
    tf_coils_jax = coils_via_symmetries(
        base_tf_curves_jax, base_tf_currents_jax, surface.nfp, True
    )
    wp_coils_jax = coils_via_symmetries(
        base_wp_curves_jax, base_wp_currents_jax, surface.nfp, True
    )
    coils_jax = tf_coils_jax + wp_coils_jax
    bs_jax = BiotSavartJAX(coils_jax)
    jf_jax = SquaredFluxJAX(surface, bs_jax)
    setup_jax = time.perf_counter() - start_jax
    jax_lane = _build_jax_lane(
        surface=surface,
        coils=coils_jax,
        bs_jax=bs_jax,
        jf_jax=jf_jax,
        target_array=None,
        extra_components={
            "active_free_dof_count": float(len(active_dof_names)),
            "tf_coil_count": float(len(tf_coils_jax)),
            "wp_coil_count": float(len(wp_coils_jax)),
        },
        setup_seconds=setup_jax,
    )

    x0 = np.asarray(jf_cpu.x, dtype=np.float64).copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        jf_cpu.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_cpu.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jf_jax.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_jax.J())

    return FixtureBuild(
        spec=POSITION_ORIENTATION_FLUX_SUPPORT_GATE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
    )


# ---------------------------------------------------------------------------
# Wave 6 — finite-beta target normal-field fixture
#
# Source: examples/2_Intermediate/stage_two_optimization_finite_beta.py
#
# The expensive virtual-casing preprocessing is represented by a deterministic
# cached target-normal-field array. The fixture compares only the native
# SquaredFlux/SquaredFluxJAX target-array subproblem plus fixed-state length
# identity penalties through public CurveLengthJAX wrappers.


def _build_finite_beta_target_flux_fixed_state():
    """Build W7-X finite-beta target-flux parity from cached virtual casing."""
    import time
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo import CurveLength, SurfaceRZFourier, create_equally_spaced_curves
    from simsopt.geo.curveobjectives_jax import CurveLengthJAX
    from simsopt.mhd import Vmec
    from simsopt.objectives import QuadraticPenalty, SquaredFlux
    from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX

    nphi = 32
    ntheta = 32
    ncoils = 5
    R0 = 5.5
    R1 = 1.25
    order = 6
    numquadpoints = 128
    length_penalty_weight = 1.0

    target_array = _load_finite_beta_target_array()
    vmec_file = (
        TESTS_FILES / "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
    )

    surface = SurfaceRZFourier.from_wout(
        str(vmec_file),
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    total_current = Vmec(str(vmec_file)).external_current() / (2 * surface.nfp)

    def _build_coils():
        base_curves = create_equally_spaced_curves(
            ncoils,
            surface.nfp,
            stellsym=True,
            R0=R0,
            R1=R1,
            order=order,
            numquadpoints=numquadpoints,
        )
        base_currents = [Current(total_current / ncoils) for _ in range(ncoils)]
        for current in base_currents:
            current.fix_all()
        coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
        return base_curves, coils

    start_cpu = time.perf_counter()
    base_curves, coils = _build_coils()
    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(surface.gamma().reshape((-1, 3)))
    jf_cpu = SquaredFlux(surface, bs_cpu, target=target_array)
    length_objectives = [CurveLength(curve) for curve in base_curves]
    length_targets = [length.J() for length in length_objectives]
    length_penalties = [
        QuadraticPenalty(length, target, "identity")
        for length, target in zip(length_objectives, length_targets)
    ]
    length_penalty_sum = sum(length_penalties)
    jf_full = jf_cpu + length_penalty_weight * length_penalty_sum
    length_penalty_value = float(sum(penalty.J() for penalty in length_penalties))
    cpu_total_value = float(jf_full.J())
    setup_seconds_cpu = time.perf_counter() - start_cpu

    cpu_lane = _build_cpu_lane(
        surface=surface,
        coils=coils,
        jf_cpu=jf_full,
        bs_cpu=bs_cpu,
        target_array=target_array,
        extra_components={
            "SquaredFlux": float(jf_cpu.J()),
            "sum_QuadraticPenalty_CurveLength_identity": length_penalty_value,
            "sum_QuadraticPenalty_CurveLength_identity_weighted": (
                length_penalty_weight * length_penalty_value
            ),
            "JF_total_cpu": cpu_total_value,
            "target_array_min": float(np.min(target_array)),
            "target_array_max": float(np.max(target_array)),
            "target_array_mean": float(np.mean(target_array)),
        },
        setup_seconds=setup_seconds_cpu,
        objective_component_name="JF_total_cpu",
    )

    start_jax = time.perf_counter()
    base_curves_jax, coils_jax = _build_coils()
    bs_jax = BiotSavartJAX(coils_jax)
    jf_jax = SquaredFluxJAX(surface, bs_jax, target=target_array)
    length_objectives_jax = [CurveLengthJAX(curve) for curve in base_curves_jax]
    length_penalties_jax = [
        QuadraticPenalty(length, target, "identity")
        for length, target in zip(length_objectives_jax, length_targets)
    ]
    length_penalty_sum_jax = sum(length_penalties_jax)
    jf_full_jax = jf_jax + length_penalty_weight * length_penalty_sum_jax
    length_penalty_value_jax = float(
        sum(penalty.J() for penalty in length_penalties_jax)
    )
    setup_seconds_jax = time.perf_counter() - start_jax

    jax_lane = _build_jax_lane(
        surface=surface,
        coils=coils_jax,
        bs_jax=bs_jax,
        jf_jax=jf_full_jax,
        target_array=target_array,
        extra_components={
            "SquaredFluxJAX": float(jf_jax.J()),
            "sum_QuadraticPenalty_CurveLength_identity": length_penalty_value_jax,
            "sum_QuadraticPenalty_CurveLength_identity_weighted": (
                length_penalty_weight * length_penalty_value_jax
            ),
            "target_array_min": float(np.min(target_array)),
            "target_array_max": float(np.max(target_array)),
            "target_array_mean": float(np.mean(target_array)),
        },
        setup_seconds=setup_seconds_jax,
        objective_component_name="JF_total_jax",
    )

    x0 = np.asarray(jf_full.x, dtype=np.float64).copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        jf_full.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_full.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jf_full_jax.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_full_jax.J())

    return FixtureBuild(
        spec=FINITE_BETA_TARGET_FLUX_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
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
    import time
    from simsopt.field import BiotSavart, Coil, Current
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.field.coil import (
        apply_symmetries_to_curves,
        apply_symmetries_to_currents,
    )
    from simsopt.geo import CurveCurveDistance, CurveLength, SurfaceRZFourier
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.geo.curveobjectives_jax import CurveCurveDistanceJAX, CurveLengthJAX
    from simsopt.geo.finitebuild import create_multifilament_grid
    from simsopt.objectives import QuadraticPenalty, SquaredFlux
    from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX

    ncoils = 4
    R0 = 1.0
    R1 = 0.7
    order = 5
    length_pen = 1e-2
    dist_min = 0.1
    dist_pen = 10.0
    numfilaments_n = 2
    numfilaments_b = 3
    gapsize_n = 0.02
    gapsize_b = 0.04
    rot_order = 1
    nfil = numfilaments_n * numfilaments_b

    filename = TESTS_FILES / "input.LandremanPaul2021_QA"
    start_cpu = time.perf_counter()
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
    currents_fb = apply_symmetries_to_currents(filament_currents, surface.nfp, True)
    curves = apply_symmetries_to_curves(base_curves, surface.nfp, True)

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

    coils_fb = [Coil(c, curr) for (c, curr) in zip(curves_fb, currents_fb)]
    bs_cpu = BiotSavart(coils_fb)
    bs_cpu.set_points(surface.gamma().reshape((-1, 3)))
    jf_cpu = SquaredFlux(surface, bs_cpu)
    jls = [CurveLength(c) for c in base_curves]
    length_targets = [j.J() for j in jls]
    length_penalties = [
        QuadraticPenalty(jl, target, "max") for jl, target in zip(jls, length_targets)
    ]
    jdist = CurveCurveDistance(curves, dist_min)
    jf_full = jf_cpu + length_pen * sum(length_penalties) + dist_pen * jdist
    setup_cpu = time.perf_counter() - start_cpu
    cpu_lane = _build_cpu_lane(
        surface=surface,
        coils=coils_fb,
        jf_cpu=jf_full,
        bs_cpu=bs_cpu,
        target_array=None,
        extra_components={
            "SquaredFlux": float(jf_cpu.J()),
            "sum_QuadraticPenalty_CurveLength_max": float(
                length_pen * sum(j.J() for j in length_penalties)
            ),
            "CurveCurveDistance": float(dist_pen * jdist.J()),
        },
        setup_seconds=setup_cpu,
        objective_component_name="JF_total_cpu",
    )

    start_jax = time.perf_counter()
    base_curves_jax = create_equally_spaced_curves(
        ncoils, surface.nfp, stellsym=True, R0=R0, R1=R1, order=order
    )
    base_currents_jax = []
    for i in range(ncoils):
        curr = Current(1.0)
        if i == 0:
            curr.fix_all()
        base_currents_jax.append(curr * (1e5 / nfil))
    filament_curves_jax = []
    for c in base_curves_jax:
        filament_curves_jax.extend(
            create_multifilament_grid(
                c,
                numfilaments_n,
                numfilaments_b,
                gapsize_n,
                gapsize_b,
                rotation_order=rot_order,
            )
        )
    filament_currents_jax = []
    for current in base_currents_jax:
        filament_currents_jax.extend([current] * nfil)
    curves_fb_jax = apply_symmetries_to_curves(filament_curves_jax, surface.nfp, True)
    currents_fb_jax = apply_symmetries_to_currents(
        filament_currents_jax, surface.nfp, True
    )
    coils_fb_jax = [Coil(c, curr) for (c, curr) in zip(curves_fb_jax, currents_fb_jax)]
    bs_jax = BiotSavartJAX(coils_fb_jax)
    jf_jax = SquaredFluxJAX(surface, bs_jax)
    curves_jax = apply_symmetries_to_curves(base_curves_jax, surface.nfp, True)
    jls_jax = [CurveLengthJAX(c) for c in base_curves_jax]
    length_penalties_jax = [
        QuadraticPenalty(jl, target, "max")
        for jl, target in zip(jls_jax, length_targets)
    ]
    jdist_jax = CurveCurveDistanceJAX(curves_jax, dist_min)
    jf_full_jax = jf_jax + length_pen * sum(length_penalties_jax) + dist_pen * jdist_jax
    setup_jax = time.perf_counter() - start_jax
    jax_lane = _build_jax_lane(
        surface=surface,
        coils=coils_fb_jax,
        bs_jax=bs_jax,
        jf_jax=jf_full_jax,
        target_array=None,
        extra_components={
            "SquaredFluxJAX": float(jf_jax.J()),
            "sum_QuadraticPenalty_CurveLength_max": float(
                length_pen * sum(j.J() for j in length_penalties_jax)
            ),
            "CurveCurveDistance": float(dist_pen * jdist_jax.J()),
            "filament_count": float(len(coils_fb_jax)),
            "base_curve_count": float(len(base_curves_jax)),
        },
        setup_seconds=setup_jax,
        objective_component_name="JF_total_jax",
    )

    x0 = np.asarray(jf_full.x, dtype=np.float64).copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        jf_full.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_full.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jf_full_jax.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_full_jax.J())

    return FixtureBuild(
        spec=FINITEBUILD_MULTIFILAMENT_SUPPORT_GATE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
    )


# ---------------------------------------------------------------------------
# Phase 8 — strain optimization fixed-state rotation fixture


def _build_strain_optimization_fixed_state():
    """Build the HSX strain example at its initial fixed rotation state."""
    import time

    from simsopt.configs import get_data
    from simsopt.geo import CurveXYZFourier, FrameRotation, FramedCurveCentroid
    from simsopt.geo.framedcurve_jax import FrameRotationJAX, FramedCurveCentroidJAX
    from simsopt.geo.strain_optimization import (
        CoilStrain,
        LPBinormalCurvatureStrainPenalty,
        LPTorsionalStrainPenalty,
    )

    start_setup = time.perf_counter()
    base_curves, _base_currents, _ma, _nfp, _bs = get_data(
        "hsx",
        coil_order=10,
        points_per_period=10,
    )
    source_curve = base_curves[1]
    scale_factor = 0.1
    rotation_order = 10
    objective_width = 1e-3
    diagnostic_width = 3e-3
    tor_threshold = 0.002
    cur_threshold = 0.002

    curve_cpu = CurveXYZFourier(source_curve.quadpoints, source_curve.order)
    curve_cpu.x = np.asarray(source_curve.x, dtype=np.float64) * scale_factor
    curve_cpu.fix_all()
    rotation_cpu = FrameRotation(curve_cpu.quadpoints, rotation_order)
    framed_cpu = FramedCurveCentroid(curve_cpu, rotation_cpu)
    jtor_cpu = LPTorsionalStrainPenalty(
        framed_cpu,
        width=objective_width,
        p=2,
        threshold=tor_threshold,
    )
    jbin_cpu = LPBinormalCurvatureStrainPenalty(
        framed_cpu,
        width=objective_width,
        p=2,
        threshold=cur_threshold,
    )
    jf_cpu = jtor_cpu + jbin_cpu
    strain_cpu = CoilStrain(framed_cpu, diagnostic_width)
    x0 = np.asarray(jf_cpu.x, dtype=np.float64).copy()
    active_names = tuple(f"FrameRotation:x{i}" for i in range(x0.size))
    setup_seconds = time.perf_counter() - start_setup

    start_cpu = time.perf_counter()
    cpu_torsional_strain = np.asarray(strain_cpu.torsional_strain(), dtype=np.float64)
    cpu_binormal_strain = np.asarray(
        strain_cpu.binormal_curvature_strain(),
        dtype=np.float64,
    )
    cpu_torsional_penalty = float(jtor_cpu.J())
    cpu_binormal_penalty = float(jbin_cpu.J())
    cpu_total = float(jf_cpu.J())
    cpu_gradient = np.asarray(jf_cpu.dJ(), dtype=np.float64)
    cpu_seconds = time.perf_counter() - start_cpu

    start_jax_setup = time.perf_counter()
    curve_jax = CurveXYZFourier(source_curve.quadpoints, source_curve.order)
    curve_jax.x = np.asarray(source_curve.x, dtype=np.float64) * scale_factor
    curve_jax.fix_all()
    rotation_jax = FrameRotationJAX(curve_jax.quadpoints, rotation_order)
    rotation_jax.x = x0.copy()
    framed_jax = FramedCurveCentroidJAX(curve_jax, rotation_jax)
    jtor_jax = LPTorsionalStrainPenalty(
        framed_jax,
        width=objective_width,
        p=2,
        threshold=tor_threshold,
    )
    jbin_jax = LPBinormalCurvatureStrainPenalty(
        framed_jax,
        width=objective_width,
        p=2,
        threshold=cur_threshold,
    )
    jf_jax = jtor_jax + jbin_jax
    strain_jax = CoilStrain(framed_jax, diagnostic_width)
    jax_setup_seconds = time.perf_counter() - start_jax_setup

    start_jax = time.perf_counter()
    jax_torsional_strain = np.asarray(strain_jax.torsional_strain(), dtype=np.float64)
    jax_binormal_strain = np.asarray(
        strain_jax.binormal_curvature_strain(),
        dtype=np.float64,
    )
    jax_torsional_penalty = float(jtor_jax.J())
    jax_binormal_penalty = float(jbin_jax.J())
    jax_total = float(jf_jax.J())
    jax_gradient = np.asarray(jf_jax.dJ(), dtype=np.float64)
    jax_seconds = time.perf_counter() - start_jax

    cpu_lane = _build_scalar_lane(
        lane="cpu_cpp",
        objective_total=cpu_total,
        components={
            "torsional_penalty": cpu_torsional_penalty,
            "binormal_curvature_penalty": cpu_binormal_penalty,
            "max_torsional_strain": float(np.max(cpu_torsional_strain)),
            "max_binormal_curvature_strain": float(np.max(cpu_binormal_strain)),
        },
        gradient=cpu_gradient,
        active_dof_names=active_names,
        active_dofs=x0,
        raw_arrays={
            "torsional_strain": cpu_torsional_strain,
            "binormal_curvature_strain": cpu_binormal_strain,
            "gradient": cpu_gradient,
            "objective_total": np.asarray([cpu_total], dtype=np.float64),
        },
        setup_seconds=setup_seconds,
        execute_seconds=cpu_seconds,
    )
    jax_lane = _build_scalar_lane(
        lane="jax_cpu",
        objective_total=float(jax_total),
        components={
            "torsional_penalty": float(jax_torsional_penalty),
            "binormal_curvature_penalty": float(jax_binormal_penalty),
            "max_torsional_strain": float(np.max(np.asarray(jax_torsional_strain))),
            "max_binormal_curvature_strain": float(
                np.max(np.asarray(jax_binormal_strain))
            ),
        },
        gradient=np.asarray(jax_gradient, dtype=np.float64),
        active_dof_names=active_names,
        active_dofs=x0,
        raw_arrays={
            "torsional_strain": jax_torsional_strain,
            "binormal_curvature_strain": jax_binormal_strain,
            "gradient": np.asarray(jax_gradient, dtype=np.float64),
            "objective_total": np.asarray([float(jax_total)], dtype=np.float64),
        },
        setup_seconds=jax_setup_seconds,
        execute_seconds=jax_seconds,
    )

    def _cpu_native_J(dofs: np.ndarray) -> float:
        jf_cpu.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_cpu.J())

    def _jax_native_J(dofs: np.ndarray) -> float:
        jf_jax.x = np.asarray(dofs, dtype=np.float64)
        return float(jf_jax.J())

    return FixtureBuild(
        spec=STRAIN_OPTIMIZATION_SUPPORT_GATE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
    )


def _build_coil_force_energy_fixed_state():
    """Build the coil-forces example's native force/energy subproblem."""
    import time

    from simsopt.field.selffield import regularization_circ

    field_mod, _objectives_mod, geo_mod = _cpu_imports()
    Current = field_mod.Current
    B2Energy = field_mod.B2Energy
    LpCurveForce = field_mod.LpCurveForce
    coils_via_symmetries = field_mod.coils_via_symmetries
    create_equally_spaced_curves = geo_mod.create_equally_spaced_curves

    nfp = 2
    stellsym = True
    ncoils = 3
    R0 = 1.0
    R1 = 0.5
    order = 5
    numquadpoints = 16
    current_amplitude = 1.0e5
    p = 4.0
    threshold = 0.0
    force_weight = 1.0e-2
    energy_weight = 1.0e-4

    def _build_terms(force_cls, energy_cls):
        start_setup = time.perf_counter()
        base_curves = create_equally_spaced_curves(
            ncoils,
            nfp,
            stellsym=stellsym,
            R0=R0,
            R1=R1,
            order=order,
            numquadpoints=numquadpoints,
            use_jax_curve=False,
        )
        base_currents = [Current(current_amplitude) for _ in range(ncoils)]
        base_currents[0].fix_all()
        coils = coils_via_symmetries(
            base_curves,
            base_currents,
            nfp,
            stellsym,
            regularizations=[regularization_circ(0.05)] * ncoils,
        )
        base_coils = coils[:ncoils]
        force_obj = force_cls(base_coils, coils, p=p, threshold=threshold)
        energy_obj = energy_cls(coils)
        return force_obj, energy_obj, coils, time.perf_counter() - start_setup

    def _independent_lp_curve_force_oracle(target_coils, source_coils) -> float:
        total = 0.0
        for coil in target_coils:
            gammadash_norm = np.linalg.norm(coil.curve.gammadash(), axis=1)
            force_norm_mn_per_m = np.linalg.norm(coil.force(source_coils), axis=1) / 1e6
            total += (
                np.sum(
                    np.maximum(force_norm_mn_per_m - threshold, 0.0) ** p
                    * gammadash_norm
                )
                / gammadash_norm.shape[0]
                / p
            )
        return float(total)

    def _independent_b2_energy_oracle(coils) -> float:
        gammas = np.asarray([coil.curve.gamma() for coil in coils], dtype=np.float64)
        gammadashs = np.asarray(
            [coil.curve.gammadash() for coil in coils],
            dtype=np.float64,
        )
        currents = np.asarray(
            [coil.current.get_value() for coil in coils], dtype=np.float64
        )
        regularizations = np.asarray(
            [coil.regularization for coil in coils],
            dtype=np.float64,
        )
        ncoils_local = gammas.shape[0]
        nquad = gammas.shape[1]
        inductance = np.empty((ncoils_local, ncoils_local), dtype=np.float64)
        for i in range(ncoils_local):
            for j in range(ncoils_local):
                separation = gammas[j][None, :, :] - gammas[i][:, None, :]
                distance = np.linalg.norm(separation, axis=-1)
                if i == j:
                    distance = np.sqrt(distance**2 + regularizations[i])
                tangent_dot = np.sum(
                    gammadashs[j][None, :, :] * gammadashs[i][:, None, :],
                    axis=-1,
                )
                inductance[i, j] = 1e-7 * np.sum(tangent_dot / distance) / nquad**2
        return float(
            0.5 * np.sum(currents[:, None] * currents[None, :] * inductance) / 1e6
        )

    def _evaluate(force_obj, energy_obj):
        start_exec = time.perf_counter()
        force_value = float(force_obj.J())
        energy_value = float(energy_obj.J())
        force_gradient = np.asarray(force_obj.dJ(), dtype=np.float64)
        energy_gradient = np.asarray(energy_obj.dJ(), dtype=np.float64)
        gradient = force_weight * force_gradient + energy_weight * energy_gradient
        total = force_weight * force_value + energy_weight * energy_value
        return {
            "force_value": force_value,
            "energy_value": energy_value,
            "total": total,
            "force_gradient": force_gradient,
            "energy_gradient": energy_gradient,
            "gradient": gradient,
            "execute_seconds": time.perf_counter() - start_exec,
        }

    force_cpu, energy_cpu, coils_cpu, cpu_setup_seconds = _build_terms(
        LpCurveForce,
        B2Energy,
    )
    force_jax, energy_jax, coils_jax, jax_setup_seconds = _build_terms(
        LpCurveForce,
        B2Energy,
    )
    cpu_eval = _evaluate(force_cpu, energy_cpu)
    jax_eval = _evaluate(force_jax, energy_jax)
    force_oracle_value = _independent_lp_curve_force_oracle(
        coils_cpu[:ncoils],
        coils_cpu,
    )
    energy_oracle_value = _independent_b2_energy_oracle(coils_cpu)

    cpu_lane = _build_scalar_lane(
        lane="cpu_cpp",
        objective_total=cpu_eval["total"],
        components={
            "LpCurveForce": cpu_eval["force_value"],
            "B2Energy": cpu_eval["energy_value"],
            "LpCurveForce_independent_oracle": force_oracle_value,
            "B2Energy_independent_oracle": energy_oracle_value,
            "FORCE_WEIGHT": force_weight,
            "B2Energy_WEIGHT": energy_weight,
        },
        gradient=cpu_eval["gradient"],
        active_dof_names=tuple(force_cpu.dof_names),
        active_dofs=np.asarray(force_cpu.x, dtype=np.float64),
        raw_arrays={
            "lp_curve_force_gradient": cpu_eval["force_gradient"],
            "b2_energy_gradient": cpu_eval["energy_gradient"],
            "gradient": cpu_eval["gradient"],
            "objective_total": np.asarray([cpu_eval["total"]], dtype=np.float64),
            "lp_curve_force_independent_oracle": np.asarray(
                [force_oracle_value],
                dtype=np.float64,
            ),
            "b2_energy_independent_oracle": np.asarray(
                [energy_oracle_value],
                dtype=np.float64,
            ),
        },
        setup_seconds=cpu_setup_seconds,
        execute_seconds=cpu_eval["execute_seconds"],
    )
    jax_lane = _build_scalar_lane(
        lane="jax_cpu",
        objective_total=jax_eval["total"],
        components={
            "LpCurveForce": jax_eval["force_value"],
            "B2Energy": jax_eval["energy_value"],
            "FORCE_WEIGHT": force_weight,
            "B2Energy_WEIGHT": energy_weight,
        },
        gradient=jax_eval["gradient"],
        active_dof_names=tuple(force_jax.dof_names),
        active_dofs=np.asarray(force_jax.x, dtype=np.float64),
        raw_arrays={
            "lp_curve_force_gradient": jax_eval["force_gradient"],
            "b2_energy_gradient": jax_eval["energy_gradient"],
            "gradient": jax_eval["gradient"],
            "objective_total": np.asarray([jax_eval["total"]], dtype=np.float64),
        },
        setup_seconds=jax_setup_seconds,
        execute_seconds=jax_eval["execute_seconds"],
        native_curve_spec_hashes=_verify_jax_native_spec_contract(coils_jax),
    )
    x0 = np.asarray(force_cpu.x, dtype=np.float64).copy()

    def _cpu_native_J(dofs: np.ndarray) -> float:
        x = np.asarray(dofs, dtype=np.float64)
        force_cpu.x = x
        energy_cpu.x = x
        return force_weight * float(force_cpu.J()) + energy_weight * float(
            energy_cpu.J()
        )

    def _jax_native_J(dofs: np.ndarray) -> float:
        x = np.asarray(dofs, dtype=np.float64)
        force_jax.x = x
        energy_jax.x = x
        return force_weight * float(force_jax.J()) + energy_weight * float(
            energy_jax.J()
        )

    return FixtureBuild(
        spec=COIL_FORCES_SUPPORT_GATE_SPEC,
        cpu_lane=cpu_lane,
        jax_lane=jax_lane,
        unsupported_components=(),
        cpu_native_subproblem_J=_cpu_native_J,
        jax_native_subproblem_J=_jax_native_J,
        x0=x0,
    )


# ---------------------------------------------------------------------------
# Specs


MINIMAL_STAGE2_FLUX_LENGTH_GAP_SPEC = FixtureSpec(
    fixture_id="minimal_stage2_flux_length_gap",
    source_example="examples/1_Simple/stage_two_optimization_minimal.py",
    classification=SUPPORTED,
    classification_reason=(
        "SquaredFlux/SquaredFluxJAX and "
        "QuadraticPenalty(sum(CurveLengthJAX), 'max') value/gradient parity "
        "are supported in the fixed-state CPU/C++ vs JAX CPU comparison."
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


SURFACE_AREA_VOLUME_SIMPLE_SPEC = FixtureSpec(
    fixture_id="surface_area_volume_simple",
    source_example="examples/1_Simple/surf_vol_area.py",
    classification=SUPPORTED,
    classification_reason=(
        "Side-effect-free fixed-state Area/Volume parity using the initial "
        "SurfaceRZFourier state from the example. The optimizer, JSON save/load, "
        "and second centered-difference solve remain outside this fixture."
    ),
    rationale=(
        "This is the lowest-cost example-derived fixture for scalar surface "
        "objective values and surface-DOF gradients."
    ),
    acceptance_criteria=(
        "Area and Volume values match CPU C++ oracles.",
        "Area and Volume surface gradients match CPU C++ oracles.",
        "CPU and JAX lanes use independent surface objects with identical DOFs.",
        "No unsupported components are recorded.",
    ),
    fixture_kind=SURFACE_SCALAR,
    inputs={
        "desired_area": 8.0,
        "desired_volume": 0.6,
        "surface": "SurfaceRZFourier()",
        "fixed_dofs": ("rc(0,0)",),
        "side_effects_excluded": (
            "least_squares_serial_solve",
            "surf.save",
            "simsopt.save",
            "simsopt.load",
        ),
    },
)


CWS_SAVED_LOCAL_FLUX_NFP2_SPEC = FixtureSpec(
    fixture_id="cws_saved_local_flux_nfp2",
    source_example="examples/3_Advanced/curves_CWS_example.py",
    classification=SUPPORTED,
    classification_reason=(
        "Saved BiotSavart artifact + local-flux SquaredFluxJAX path is "
        "supported after legacy CurveCWSFourier JSON reconstruction."
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
        "supported after legacy CurveCWSFourier JSON reconstruction."
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
        "The fixed-state composite Stage-II objective is native-supported via "
        "SquaredFluxJAX plus public JAX wrappers for CurveLength, "
        "CurveCurveDistance, CurveSurfaceDistance, LpCurveCurvature, and "
        "QuadraticPenalty(MeanSquaredCurvatureJAX, 'max'); no unsupported "
        "components are recorded for this row."
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
        "CurvePlanarFourier exposes a native immutable JAX spec, and the "
        "fixed-state planar composite is native-supported via SquaredFluxJAX "
        "plus public JAX wrappers for CurveLength, CurveCurveDistance, "
        "CurveSurfaceDistance, LpCurveCurvature, MeanSquaredCurvature, and "
        "LinkingNumber. No components are listed as unsupported for this "
        "fixture."
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
    classification=SUPPORTED,
    classification_reason=(
        "OrientedCurveXYZFourier exposes an immutable native spec, so the "
        "fixed-state TF+windowpane SquaredFlux/SquaredFluxJAX path can be "
        "compared without running the optimizer."
    ),
    rationale=(
        "This upgrades the prior oriented-curve support gate while preserving "
        "the example's active position/orientation DOF boundary."
    ),
    acceptance_criteria=(
        "Active free DOF mapping includes windowpane position and orientation.",
        "BiotSavart field arrays match CPU C++ oracles.",
        "SquaredFlux values and gradients match CPU C++ oracles.",
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
        "MajorRadius, NonQuasiSymmetricRatio (auxiliary surface sDIM=20), "
        "and sum(CurveLength). The JAX lane recomputes the corresponding "
        "scalar values from the solved iota plus pure-JAX helpers "
        "(surface_major_radius_jax_from_dofs, _qs_ratio_pure) on a fresh "
        "BiotSavartJAX coil_set_spec and CurveLengthJAX over an independently "
        "loaded NCSX curve tree. This fixture does not claim public "
        "BoozerSurfaceJAX wrapper or adjoint parity."
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
        "wrapper_set": (
            "iota",
            "major_radius",
            "nq_symmetric_ratio",
            "sum_CurveLength",
        ),
        "non_quasisymmetric_sDIM": 20,
        "length_penalty_classification": "sum_CurveLength via CurveLengthJAX",
    },
)


FINITE_BETA_TARGET_FLUX_SPEC = FixtureSpec(
    fixture_id="finite_beta_target_flux",
    source_example="examples/2_Intermediate/stage_two_optimization_finite_beta.py",
    classification=SUPPORTED,
    classification_reason=(
        "Finite-beta W7-X fixed-state target-flux fixture. The virtual "
        "casing preprocessing output is represented by a deterministic "
        "cached B_external_normal target array; CPU SquaredFlux and "
        "SquaredFluxJAX receive the same target array. The fixed-state "
        "length identity penalties are included through CurveLengthJAX, so "
        "no unsupported components are recorded for this row."
    ),
    rationale=(
        "This closes the target-array SquaredFluxJAX support surface without "
        "running VMEC or virtual-casing preprocessing inside the parity harness."
    ),
    acceptance_criteria=(
        "Cached virtual-casing target array path, shape, and hash are recorded.",
        "CPU SquaredFlux and SquaredFluxJAX compare with the same target array.",
        "Length identity penalties are included in the CPU/JAX objective and "
        "gradient comparison.",
    ),
    inputs={
        "nphi": 32,
        "ntheta": 32,
        "ncoils": 5,
        "R0": 5.5,
        "R1": 1.25,
        "order": 6,
        "numquadpoints": 128,
        "surface_range": "half period",
        "current_mode": "fixed equal scalar currents from VMEC external_current",
        "vmec_file": (
            "tests/test_files/"
            "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
        ),
        "virtual_casing_target": _finite_beta_target_metadata(),
        "length_penalty": "sum_QuadraticPenalty_CurveLength_identity",
        "input_file_hashes": {
            "vmec": _file_input_metadata(
                TESTS_FILES
                / "wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc"
            ),
            "virtual_casing_target": _file_input_metadata(
                FINITE_BETA_TARGET_ARRAY_PATH
            ),
        },
    },
)


FINITEBUILD_MULTIFILAMENT_SUPPORT_GATE_SPEC = FixtureSpec(
    fixture_id="finitebuild_multifilament_support_gate",
    source_example="examples/3_Advanced/stage_two_optimization_finitebuild.py",
    classification=SUPPORTED,
    classification_reason=(
        "Finite-build multifilament curves are expanded by "
        "simsopt.geo.finitebuild.create_multifilament_grid. The fixed-state "
        "fixture materializes a reduced grid, checks every expanded filament "
        "against the same _supports_native_curve_geometry predicate "
        "BiotSavartJAX uses internally, and compares the field/flux objective "
        "plus CurveLengthJAX quadratic penalties and CurveCurveDistanceJAX. "
        "No unsupported components are recorded for this row."
    ),
    rationale=(
        "This closes the previous finite-build support probe by proving the "
        "native multifilament field/flux subproblem and the fixed-state "
        "curve-length and curve-distance penalties."
    ),
    acceptance_criteria=(
        "Every symmetry-expanded finite-build filament has an immutable native "
        "JAX geometry spec.",
        "CPU/C++ and JAX CPU agree for field_B, Bdotn, SquaredFlux, curve "
        "penalties, native subtotal, and gradient on the shared fixed-state "
        "multifilament subproblem.",
        "No unsupported components are recorded.",
    ),
    inputs={
        "ncoils": 4,
        "R0": 1.0,
        "R1": 0.7,
        "order": 5,
        "length_penalty_weight": 1e-2,
        "distance_min": 0.1,
        "distance_penalty_weight": 10.0,
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
    classification=SUPPORTED,
    classification_reason=(
        "Fixed-state QfmResidual/QfmResidualJAX and label parity are "
        "supported for the qfm.py initial NCSX surface. QfmSurface solver "
        "orchestration remains CPU-only and is reported as "
        "QfmSurface_host_solver."
    ),
    rationale=(
        "This closes the previous unsupported QFM row without claiming the "
        "host SciPy QfmSurface optimizer path."
    ),
    acceptance_criteria=(
        "QfmResidual values and surface gradients match CPU C++ oracles.",
        "Area, Volume, and ToroidalFlux label values match CPU C++ oracles.",
        "QfmSurface_host_solver is the only unsupported component.",
    ),
    fixture_kind=QFM,
    inputs={
        "config_name": "ncsx",
        "mpol": 5,
        "ntor": 5,
        "stellsym": True,
        "quadpoints_phi": 25,
        "quadpoints_theta": 25,
        "minor_radius": 0.2,
        "fit_to_curve_flip_theta": True,
        "labels": ("volume", "area", "toroidal_flux"),
        "unsupported_solver": "QfmSurface_host_solver",
        "post_constraint_target_state": (
            "not_reconstructable_without_host_scipy_QfmSurface"
        ),
    },
)


PM_SIMPLE_FIXED_STATE_GPMO_BASELINE_SPEC = FixtureSpec(
    fixture_id="pm_simple_fixed_state_gpmo_baseline",
    source_example="examples/1_Simple/permanent_magnet_simple.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced fixed-state permanent_magnet_simple baseline GPMO fixture. "
        "Compares immutable PermanentMagnetGridJAX payloads, baseline GPMO "
        "final moments/residuals, and DipoleField/DipoleFieldJAX field output."
    ),
    rationale=(
        "This is the first example-derived permanent-magnet fixture that uses "
        "real FAMUS-derived input while keeping CI cost bounded."
    ),
    acceptance_criteria=(
        "Grid payload arrays match the CPU PermanentMagnetGrid oracle.",
        "Baseline GPMO final moments and residual objective match.",
        "DipoleFieldJAX field and Bdotn match the CPU DipoleField oracle.",
        "No generic GPMO dispatcher parity is claimed.",
    ),
    fixture_kind=PERMANENT_MAGNET,
    inputs={
        "nphi": 2,
        "ntheta": 2,
        "downsample": 100,
        "surface": "tests/test_files/wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc",
        "famus": "tests/test_files/init_orient_pm_nonorm_5E4_q4_dp.focus",
        "input_file_hashes": {
            "surface": _file_input_metadata(
                TESTS_FILES / "wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc"
            ),
            "famus": _file_input_metadata(
                TESTS_FILES / "init_orient_pm_nonorm_5E4_q4_dp.focus"
            ),
        },
        "coordinate_flag": "cylindrical",
        "algorithm_variant": "baseline",
        "K": 4,
        "reg_l2": 0.0,
        "single_direction": -1,
        "memory_budget": {
            "ci_profile": "reduced",
            "surface_points": 4,
            "downsample": 100,
            "selected_dipoles": 4,
        },
    },
)


WIREFRAME_RCLS_BASIC_FIXED_STATE_SPEC = FixtureSpec(
    fixture_id="wireframe_rcls_basic_fixed_state",
    source_example="examples/2_Intermediate/wireframe_rcls_basic.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced fixed-state RCLS wireframe fixture preserving the example's "
        "surf_plas input mode for both CPU optimize_wireframe and "
        "optimize_wireframe_jax."
    ),
    rationale=(
        "This is the first wireframe example with a direct JAX solve wrapper "
        "and C++ field oracle over real example surfaces."
    ),
    acceptance_criteria=(
        "bnorm objective matrices match.",
        "RCLS objective components and constraint values match.",
        "WireframeFieldJAX B and dB_by_dX match the CPU WireframeField oracle.",
        "Raw RCLS current-vector identity is not claimed; equivalent nullspace "
        "solutions are listed as unsupported.",
    ),
    fixture_kind=WIREFRAME,
    inputs={
        "wf_n_phi": 8,
        "wf_n_theta": 12,
        "wf_surf_dist": 0.3,
        "field_on_axis": 1.0,
        "regularization_w": 1e-10,
        "plas_n_phi": 32,
        "plas_n_theta": 32,
        "vmec_input": "tests/test_files/input.LandremanPaul2021_QA",
        "input_file_hashes": {
            "vmec_input": _file_input_metadata(
                TESTS_FILES / "input.LandremanPaul2021_QA"
            ),
        },
        "surface_range": "half period",
        "algorithm": "rcls",
        "input_mode": "surf_plas",
        "memory_budget": {
            "ci_profile": "reduced",
            "wf_grid": (8, 12),
            "plas_grid": (32, 32),
        },
    },
)


TRACING_FIELDLINES_QA_REDUCED_ENDPOINT_SPEC = FixtureSpec(
    fixture_id="tracing_fieldlines_qa_reduced_endpoint",
    source_example="examples/1_Simple/tracing_fieldlines_QA.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced InterpolatedField/compute_fieldlines fixture for the QA "
        "Poincare example. CPU uses InterpolatedField + upstream "
        "compute_fieldlines; JAX uses InterpolatedFieldJAX.jax_B_at through "
        "the JAX tracing backend. The fixture uses the example spelling "
        "LevelsetStoppingCriterion(sc_fieldline.dist) plus the example skip "
        "callback; the raw distance-grid adapter is recovered through the "
        "SurfaceClassifier metadata registry."
    ),
    rationale=(
        "This exercises the example's interpolated-field tracing path without "
        "running VTK/plot side effects or long Poincare trajectories."
    ),
    acceptance_criteria=(
        "InterpolatedFieldJAX exposes jax_B_at for the JAX tracing backend.",
        "Interpolated B on the reduced QA surface matches the CPU oracle.",
        "Reduced fieldline endpoint and first phi-plane hit coordinates match.",
        "Raw LevelsetStoppingCriterion distance-grid and skip adapters are exercised.",
    ),
    fixture_kind=TRACING,
    inputs={
        "nphi": 32,
        "ntheta": 16,
        "degree": 2,
        "interpolation_grid": (5, 8, 4),
        "R0": (1.24,),
        "Z0": (0.0,),
        "tmax": 20.0,
        "tol": 1e-12,
        "phi_fraction_of_field_period": 0.25,
        "surface_classifier_h": 0.1,
        "surface_classifier_p": 2,
        "skip_threshold": -0.05,
        "vmec_input": "tests/test_files/input.LandremanPaul2021_QA",
        "biot_savart": "examples/1_Simple/inputs/biot_savart_opt.json",
        "input_file_hashes": {
            "vmec_input": _file_input_metadata(
                TESTS_FILES / "input.LandremanPaul2021_QA"
            ),
            "biot_savart": _file_input_metadata(
                EXAMPLES / "1_Simple" / "inputs" / "biot_savart_opt.json"
            ),
        },
        "memory_budget": {
            "ci_profile": "reduced",
            "fieldlines": 1,
            "tmax": 20.0,
            "interpolation_grid": (5, 8, 4),
        },
    },
)


PM_QA_FIXED_STATE_GPMO_ARB_VEC_OR_MULTI_SPEC = FixtureSpec(
    fixture_id="pm_qa_fixed_state_gpmo_arbvec_or_multi",
    source_example="examples/2_Intermediate/permanent_magnet_QA.py",
    classification=SUPPORTED,
    classification_reason=(
        "The historical fixture id is stale: permanent_magnet_QA.py uses "
        "relax_and_split, not GPMO. This reduced fixed-state row preserves the "
        "example relax-and-split threshold pass structure and compares the grid, "
        "final dense/proxy moments, scalar history, residuals, objectives, and "
        "dipole fields against relax_and_split_jax. Host coil-current optimization "
        "and output writing remain unsupported orchestration components."
    ),
    rationale=(
        "This upgrades the QA row by pinning the actual public example optimizer "
        "family instead of forcing the stale GPMO fixture name."
    ),
    acceptance_criteria=(
        "QA row uses relax_and_split on CPU and relax_and_split_jax on JAX.",
        "CPU/C++ and JAX CPU agree on grid payload, final m, final m_proxy, "
        "scalar history, residuals, objectives, and dipole fields.",
        "Host orchestration gaps remain named in unsupported_components.",
    ),
    inputs={
        "nphi": 4,
        "ntheta": 4,
        "dr": 0.05,
        "coff": 0.1,
        "poff": 0.05,
        "ncoils": 8,
        "coil_R0": 1.0,
        "coil_R1": 0.65,
        "coil_order": 5,
        "coil_numquadpoints": 128,
        "total_current_A": 187500.0,
        "reg_l0": 0.05,
        "nu": 1e10,
        "max_iter": 2,
        "max_iter_RS": 2,
        "threshold_passes": 2,
        "algorithm": "relax_and_split",
        "surface_input": "tests/test_files/input.LandremanPaul2021_QA_lowres",
        "input_file_hashes": {
            "surface": _file_input_metadata(
                TESTS_FILES / "input.LandremanPaul2021_QA_lowres"
            ),
        },
    },
    fixture_kind=PERMANENT_MAGNET_RELAX_AND_SPLIT,
)


PM_MUSE_FAMUS_SPEC = FixtureSpec(
    fixture_id="pm_muse_famus",
    source_example="examples/2_Intermediate/permanent_magnet_MUSE.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced MUSE FAMUS fixture preserves the public ArbVec_backtracking "
        "algorithm family and compares the fixed-state grid, final moments, "
        "residual, objective, R2/Bn histories, moment history, and dipole field "
        "against the matching explicit JAX wrapper."
    ),
    rationale=(
        "This upgrades the MUSE row without importing the plotting/file-writing "
        "example script or comparing it against the wrong GPMO family."
    ),
    acceptance_criteria=(
        "MUSE row uses ArbVec_backtracking on both CPU and JAX lanes.",
        "CPU/C++ and JAX CPU agree on grid payload, final m, residual, objective, "
        "R2/Bn histories, moment history, and dipole field.",
    ),
    inputs={
        "nphi": 2,
        "ntheta": 2,
        "downsample": 100,
        "K": 5,
        "backtracking": 2,
        "max_nMagnets": 4,
        "Nadjacent": 1,
        "algorithm": "ArbVec_backtracking",
        "surface_input": "tests/test_files/input.muse",
        "famus_input": "tests/test_files/zot80.focus",
        "coil_input": "tests/test_files/muse_tf_coils.focus",
        "input_file_hashes": {
            "surface": _file_input_metadata(TESTS_FILES / "input.muse"),
            "famus": _file_input_metadata(TESTS_FILES / "zot80.focus"),
            "coils": _file_input_metadata(TESTS_FILES / "muse_tf_coils.focus"),
        },
    },
    fixture_kind=PERMANENT_MAGNET,
)


PM_PM4STELL_BACKTRACKING_SPEC = FixtureSpec(
    fixture_id="pm_pm4stell_backtracking",
    source_example="examples/2_Intermediate/permanent_magnet_PM4Stell.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced PM4Stell fixture preserves the public ArbVec_backtracking "
        "algorithm family with PM4Stell face/edge/corner triplet polarizations "
        "and compares the fixed-state grid, final moments, residual, objective, "
        "R2/Bn histories, moment history, and dipole field against the matching "
        "explicit JAX wrapper."
    ),
    rationale=(
        "This upgrades the PM4Stell row without importing the plotting/file-writing "
        "example script or comparing it against the wrong GPMO family."
    ),
    acceptance_criteria=(
        "PM4Stell row uses ArbVec_backtracking on both CPU and JAX lanes.",
        "CPU/C++ and JAX CPU agree on grid payload, final m, residual, objective, "
        "R2/Bn histories, moment history, and dipole field.",
    ),
    inputs={
        "nphi": 2,
        "ntheta": 2,
        "downsample": 100,
        "K": 5,
        "backtracking": 2,
        "max_nMagnets": 4,
        "Nadjacent": 10,
        "algorithm": "ArbVec_backtracking",
        "surface_input": "tests/test_files/c09r00_B_axis_half_tesla_PM4Stell.plasma",
        "coil_input": "tests/test_files/tf_only_half_tesla_symmetry_baxis_PM4Stell.focus",
        "famus_input": "tests/test_files/magpie_trial104b_PM4Stell.focus",
        "corners_input": "tests/test_files/magpie_trial104b_corners_PM4Stell.csv",
        "input_file_hashes": {
            "surface": _file_input_metadata(
                TESTS_FILES / "c09r00_B_axis_half_tesla_PM4Stell.plasma"
            ),
            "coils": _file_input_metadata(
                TESTS_FILES / "tf_only_half_tesla_symmetry_baxis_PM4Stell.focus"
            ),
            "famus": _file_input_metadata(
                TESTS_FILES / "magpie_trial104b_PM4Stell.focus"
            ),
            "corners": _file_input_metadata(
                TESTS_FILES / "magpie_trial104b_corners_PM4Stell.csv"
            ),
        },
    },
    fixture_kind=PERMANENT_MAGNET,
)


WIREFRAME_RCLS_PORTS_CONSTRAINT_GATE_SPEC = FixtureSpec(
    fixture_id="wireframe_rcls_ports_constraint_gate",
    source_example="examples/2_Intermediate/wireframe_rcls_with_ports.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced port-constrained RCLS fixture preserves the example's port "
        "collision masks, poloidal-current constraint, and surf_plas input mode. "
        "CPU optimize_wireframe and optimize_wireframe_jax agree on matrices, "
        "objective components, constraint shape/satisfaction, and fields; raw "
        "current-vector identity remains unsupported due the RCLS nullspace."
    ),
    rationale=(
        "This upgrades the port example from a support gate without importing "
        "the plotting/VTK example script or treating nullspace-equivalent "
        "current vectors as parity evidence."
    ),
    acceptance_criteria=(
        "Port collision constraints are applied before both CPU and JAX RCLS solves.",
        "bnorm objective matrices and constraint matrix shape match.",
        "RCLS objective components and constraint satisfaction match.",
        "WireframeFieldJAX B and dB_by_dX match the CPU WireframeField oracle.",
        "Raw RCLS current-vector identity remains named unsupported.",
    ),
    fixture_kind=WIREFRAME,
    inputs={
        "wf_n_phi": 8,
        "wf_n_theta": 12,
        "wf_surf_dist": 0.3,
        "plas_n_phi": 16,
        "plas_n_theta": 16,
        "port_phis": ("pi/8", "3*pi/8"),
        "port_thetas": ("pi/4", "7*pi/4"),
        "port_ir": 0.1,
        "port_thick": 0.005,
        "port_gap": 0.04,
        "port_l0": -0.15,
        "port_l1": 0.15,
        "field_on_axis": 1.0,
        "regularization_w": 1e-10,
        "vmec_input": "tests/test_files/input.LandremanPaul2021_QA",
        "input_file_hashes": {
            "vmec_input": _file_input_metadata(
                TESTS_FILES / "input.LandremanPaul2021_QA"
            ),
        },
        "surface_range": "half period",
        "algorithm": "rcls",
        "input_mode": "surf_plas",
        "memory_budget": {
            "ci_profile": "reduced",
            "wf_grid": (8, 12),
            "plas_grid": (16, 16),
        },
    },
)


WIREFRAME_GSCO_MODULAR_FIXED_STATE_SPEC = FixtureSpec(
    fixture_id="wireframe_gsco_modular_fixed_state",
    source_example="examples/2_Intermediate/wireframe_gsco_modular.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced deterministic GSCO fixed-state row compares the C++ "
        "simsoptpp.GSCO history contract against the pure-JAX "
        "greedy_stellarator_coil_optimization_jax kernel. Plotting and "
        "example file-writing side effects stay out of scope."
    ),
    rationale=(
        "This provides the first GSCO pass row by preserving the algorithm "
        "history fields and constraint flags that decide the greedy loop."
    ),
    acceptance_criteria=(
        "C++ and JAX agree on final x and final loop count.",
        "C++ and JAX agree on iter_hist, curr_hist, loop_hist, f_B_hist, "
        "f_S_hist, and f_hist.",
        "Constraint flags no_crossing, no_new_coils, match_current, and "
        "max_loop_count are recorded in the fixture payload.",
    ),
    inputs={
        "reduced_fixture_seed": 3104,
        "A_shape": (5, 6),
        "b_shape": (5, 1),
        "no_crossing": False,
        "no_new_coils": False,
        "match_current": False,
        "default_current": 0.2,
        "max_current": "inf",
        "max_loop_count": 0,
        "lambda_S": 0.15,
        "max_iter": 5,
    },
    fixture_kind=WIREFRAME_GSCO,
)


WIREFRAME_GSCO_SECTOR_SADDLE_FIXED_STATE_SPEC = FixtureSpec(
    fixture_id="wireframe_gsco_sector_saddle_fixed_state",
    source_example="examples/2_Intermediate/wireframe_gsco_sector_saddle.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced sector/saddle GSCO fixture preserves the example's TF-coil "
        "initial currents, toroidal break constraints, poloidal-current "
        "constraint, and public surf_plas input mode; CPU optimize_wireframe "
        "and optimize_wireframe_jax agree on matrices, constraint masks, "
        "history fields, final currents, fields, and Bnormal."
    ),
    rationale=(
        "This pins the example's wedge-restricted saddle-coil setup without "
        "importing the plotting/VTK script side effects or relying on full "
        "example-size GSCO iteration counts in CI."
    ),
    acceptance_criteria=(
        "CPU and JAX build the same reduced BNORM wireframe and plasma surface.",
        "TF-coil initial currents, toroidal break free-cell masks, and "
        "poloidal-current constraints are applied before both solves.",
        "CPU and JAX agree on Amat, bvec, final x, final loop count, GSCO "
        "history arrays, final field B, and Bnormal.",
        "Plotting and VTK outputs remain non-parity side effects.",
    ),
    inputs={
        "input_file_hashes": {
            "vmec_input": _file_input_metadata(
                TESTS_FILES / "input.LandremanPaul2021_QA"
            ),
            "wireframe_surface": _file_input_metadata(
                TESTS_FILES / "nescin.LandremanPaul2021_QA"
            ),
        },
        "wf_n_phi": 18,
        "wf_n_theta": 8,
        "plas_n": 4,
        "n_tf_coils_hp": 3,
        "break_width": 2,
        "gsco_cur_frac": 0.05,
        "field_on_axis": 1.0,
        "lambda_S": 10**-6.5,
        "max_iter": 5,
        "print_interval": 6,
        "no_crossing": True,
        "default_current_source": "abs(gsco_cur_frac * pol_cur)",
        "max_current_source": "1.1 * abs(gsco_cur_frac * pol_cur)",
        "optimize_wireframe_input_mode": "surf_plas",
        "n_segments": 288,
        "n_cells": 144,
    },
    fixture_kind=WIREFRAME_GSCO,
)


WIREFRAME_GSCO_MULTISTEP_REDUCED_DIAGNOSTIC_SPEC = FixtureSpec(
    fixture_id="wireframe_gsco_multistep_reduced_diagnostic",
    source_example="examples/3_Advanced/wireframe_gsco_multistep.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced first-step GSCO diagnostic is supported through the public "
        "optimize_wireframe surf_plas/ext_field path. The full mutating "
        "multistep loop remains explicitly unsupported."
    ),
    rationale=(
        "This pins the example's first immutable GSCO step using the "
        "LandremanPaul QA plasma surface, BNORM wireframe surface, toroidal "
        "break constraints, and fixed TF-coil external field without claiming "
        "coil-pruning or final-adjustment parity."
    ),
    acceptance_criteria=(
        "CPU and JAX preserve the example surf_plas/ext_field input mode.",
        "CPU and JAX agree on Amat, bvec, final x, final loop count, and "
        "GSCO history arrays for the reduced first step.",
        "Full multistep mutation, small-coil pruning, final adjustment, and "
        "plot/VTK side effects remain named unsupported components.",
    ),
    inputs={
        "input_file_hashes": {
            "vmec_input": _file_input_metadata(
                TESTS_FILES / "input.LandremanPaul2021_QA"
            ),
            "wireframe_surface": _file_input_metadata(
                TESTS_FILES / "nescin.LandremanPaul2021_QA"
            ),
        },
        "wf_n_phi": 24,
        "wf_n_theta": 8,
        "plas_n": 4,
        "n_tf_coils_hp": 3,
        "break_width": 4,
        "init_gsco_cur_frac": 0.2,
        "field_on_axis": 1.0,
        "lambda_S": 1e-7,
        "max_iter": 5,
        "print_interval": 6,
        "no_crossing": True,
        "max_loop_count": 1,
        "loop_count_init": None,
        "optimize_wireframe_input_mode": "surf_plas_ext_field",
        "n_segments": 384,
        "n_cells": 192,
    },
    fixture_kind=WIREFRAME_GSCO,
)


TRACING_FIELDLINES_NCSX_REDUCED_ENDPOINT_SPEC = FixtureSpec(
    fixture_id="tracing_fieldlines_ncsx_reduced_endpoint",
    source_example="examples/1_Simple/tracing_fieldlines_NCSX.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced NCSX InterpolatedField fixture preserves the example's "
        "tube-around-axis surface, native interpolated field, initial magnetic "
        "axis seed, and phi-plane event semantics. CPU compute_fieldlines and "
        "the JAX fieldline route agree on field values, endpoint, final time, "
        "status, hit coordinates, and hit count while exercising the example's "
        "SurfaceClassifier skip callback and raw levelset-distance stopping "
        "adapter."
    ),
    rationale=(
        "This pins the numerical fieldline surface of tracing_fieldlines_NCSX "
        "without importing MPI, VTK, plotting, or the full-size example path."
    ),
    acceptance_criteria=(
        "CPU and JAX build independent NCSX BiotSavart-backed interpolated fields.",
        "Reduced endpoint, final integration time/status, and phi-plane hit "
        "coordinates match the CPU oracle.",
        "Raw SurfaceClassifier skip and LevelsetStoppingCriterion.dist adapter "
        "behavior is included in the supported fixture path.",
    ),
    inputs={
        "source_config": "simsopt.configs.get_data('ncsx')",
        "mpol": 5,
        "ntor": 5,
        "surface_nphi": 32,
        "surface_ntheta": 12,
        "surface_radius": 0.70,
        "surface_classifier_h": 0.1,
        "surface_classifier_p": 2,
        "skip_threshold": -0.05,
        "interp_degree": 2,
        "interp_grid": (5, 8, 4),
        "fieldlines": 1,
        "R0_source": "magnetic_axis_gamma[0,0]",
        "Z0_source": "magnetic_axis_gamma[0,2]",
        "phis": ("0.25 * 2*pi/nfp",),
        "tmax": 20.0,
        "tol": 1e-12,
        "max_steps": 4000,
        "max_phi_hits": 4096,
    },
    fixture_kind=TRACING,
)


TRACING_PARTICLE_GC_VAC_REDUCED_ENDPOINT_SPEC = FixtureSpec(
    fixture_id="tracing_particle_gc_vac_reduced_endpoint",
    source_example="examples/1_Simple/tracing_particle.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced NCSX particle guiding-center fixture preserves the example's "
        "InterpolatedField, SurfaceClassifier loss surface, axis-seeded "
        "particle initialization, mode='gc_vac', phi planes, and "
        "forget_exact_path endpoint contract. CPU trace_particles and the "
        "JAX guiding-center route agree on interpolated B, GradAbsB, endpoint, "
        "final time, status, and hit count while exercising the example's raw "
        "LevelsetStoppingCriterion(sc_particle.dist) adapter through the "
        "SurfaceClassifier metadata registry."
    ),
    rationale=(
        "This pins the native numerical particle-tracing surface of "
        "tracing_particle.py without MPI, VTK, plotting, or the full example "
        "particle count."
    ),
    acceptance_criteria=(
        "CPU and JAX build independent NCSX BiotSavart-backed interpolated fields.",
        "Reduced particle endpoint, final integration time/status, and phi-hit "
        "count match the CPU oracle.",
        "InterpolatedFieldJAX exposes the GradAbsB path required by the "
        "JAX guiding-center route.",
        "The raw LevelsetStoppingCriterion(sc_particle.dist) adapter is included "
        "in the supported fixture path.",
    ),
    inputs={
        "source_config": "simsopt.configs.get_data('ncsx')",
        "mpol": 5,
        "ntor": 5,
        "surface_nphi": 32,
        "surface_ntheta": 12,
        "surface_radius": 0.20,
        "surface_classifier_h": 0.1,
        "surface_classifier_p": 2,
        "interp_degree": 2,
        "interp_grid": (5, 10, 2),
        "particles": 1,
        "seed": 1,
        "mode": "gc_vac",
        "phis": tuple(f"{i}/4 * 2*pi/nfp" for i in range(4)),
        "tmax": 1e-7,
        "tol": 1e-9,
        "Ekin_eV": 5000.0,
        "forget_exact_path": True,
    },
    fixture_kind=TRACING,
)


TRACING_BOOZER_GC_REDUCED_ENDPOINT_SPEC = FixtureSpec(
    fixture_id="tracing_boozer_gc_reduced_endpoint",
    source_example="examples/2_Intermediate/tracing_boozer.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced Boozer guiding-center fixture preserves the example's "
        "BoozerRadialInterpolant -> InterpolatedBoozerField -> "
        "trace_particles_boozer contract from cached VMEC wout and BOOZXFORM "
        "boozmn states. CPU InterpolatedBoozerField and "
        "InterpolatedBoozerFieldJAX agree on modB, endpoint, final time, "
        "status, and zeta-hit count; running the example's VMEC/BOOZXFORM "
        "input path remains an external-solver blocker for this CI fixture."
    ),
    rationale=(
        "This proves the native Boozer-coordinate tracing path without requiring "
        "the VMEC or BOOZXFORM Python extensions, plotting, or resonance "
        "post-processing."
    ),
    acceptance_criteria=(
        "CPU and JAX construct independent BoozerRadialInterpolant-backed "
        "InterpolatedBoozerField objects from the same cached wout state.",
        "Reduced GC-vac endpoint, final integration time/status, and zeta-hit "
        "count match the CPU oracle.",
        "The JAX route consumes InterpolatedBoozerFieldJAX frozen-state scalar "
        "evaluators rather than falling back to the CPU field.",
        "The unavailable VMEC input-file solve remains named unsupported.",
    ),
    inputs={
        "input_file_hashes": {
            "example_vmec_input": _file_input_metadata(
                EXAMPLES / "2_Intermediate" / "inputs" / "input.LandremanPaul2021_QH"
            ),
            "cached_wout": _file_input_metadata(
                TESTS_FILES / "wout_circular_tokamak_reference.nc"
            ),
            "cached_boozmn": _file_input_metadata(
                TESTS_FILES / "boozmn_circular_tokamak.nc"
            ),
        },
        "order": 3,
        "cached_mboz": 48,
        "cached_nboz": 0,
        "interp_degree": 2,
        "interp_grid": (5, 5, 5),
        "particles": 1,
        "stz_init": ((0.30, 0.0, 0.0),),
        "zetas": ("0.25 * 2*pi/nfp",),
        "mode": "gc_vac",
        "tmax": 1e-7,
        "tol": 1e-9,
        "Ekin_eV": 1000.0,
        "parallel_speed_fraction": 0.6,
        "forget_exact_path": True,
    },
    fixture_kind=TRACING,
)


STRAIN_OPTIMIZATION_SUPPORT_GATE_SPEC = FixtureSpec(
    fixture_id="strain_optimization_support_gate",
    source_example="examples/2_Intermediate/strain_optimization.py",
    classification=SUPPORTED,
    classification_reason=(
        "Fixed-state rotation-only strain objective is supported as a pass "
        "fixture. The JAX lane uses public FrameRotationJAX and "
        "FramedCurveCentroidJAX wrappers, including their dframe_* VJP methods, "
        "through the same public strain-penalty classes as the CPU lane."
    ),
    rationale=(
        "This mirrors the example's initial fixed curve and rotation-only "
        "objective without running L-BFGS-B."
    ),
    acceptance_criteria=(
        "Torsional and binormal-curvature strain arrays match.",
        "Torsional and binormal penalty values match.",
        "Rotation-DOF gradient matches the CPU objective gradient.",
        "Public FramedCurveJAX VJP wrapper path is exercised.",
    ),
    fixture_kind=STRAIN,
    inputs={
        "config": "hsx",
        "coil_order": 10,
        "points_per_period": 10,
        "curve_index": 1,
        "scale_factor": 0.1,
        "rotation_order": 10,
        "curve_dofs": "fixed_all",
        "optimized_dofs": "FrameRotation only",
        "objective_width": 1e-3,
        "diagnostic_width": 3e-3,
        "tor_threshold": 0.002,
        "cur_threshold": 0.002,
        "side_effects_excluded": ("scipy.optimize.minimize",),
    },
)


COIL_FORCES_SUPPORT_GATE_SPEC = FixtureSpec(
    fixture_id="coil_forces_support_gate",
    source_example="examples/3_Advanced/coil_forces.py",
    classification=SUPPORTED,
    classification_reason=(
        "Reduced fixed-state coil force/energy fixture is supported as a "
        "pass public-wrapper parity row. LpCurveForce and B2Energy are the "
        "public JAX-kernel-backed wrappers. This row compares independent object graphs and also gates "
        "the JAX public values against independent CPU oracles: "
        "RegularizedCoil.force integration for LpCurveForce and a NumPy "
        "inductance-matrix loop for B2Energy."
    ),
    rationale=(
        "This mirrors the example's initial native force and B2-energy terms "
        "without optimizer execution, VTK output, or CPU-only geometric "
        "penalties."
    ),
    acceptance_criteria=(
        "Public LpCurveForce and B2Energy resolve through the field lazy exports.",
        "Independent CPU and JAX coil trees compare LpCurveForce and B2Energy values.",
        "JAX public force and energy values match independent CPU oracle values.",
        "Per-component and weighted native-subtotal gradients compare in the "
        "same active DOF basis.",
    ),
    fixture_kind=COIL_FORCE_ENERGY,
    inputs={
        "nfp": 2,
        "stellsym": True,
        "ncoils": 3,
        "R0": 1.0,
        "R1": 0.5,
        "order": 5,
        "numquadpoints": 16,
        "current_amplitude": 1.0e5,
        "fixed_current_indices": (0,),
        "regularization": "regularization_circ(0.05)",
        "LpCurveForce_p": 4.0,
        "LpCurveForce_threshold": 0.0,
        "force_independent_oracle": "RegularizedCoil.force integration",
        "energy_independent_oracle": "NumPy inductance-matrix loop",
        "FORCE_WEIGHT": 1.0e-2,
        "B2Energy_WEIGHT": 1.0e-4,
        "side_effects_excluded": (
            "scipy.optimize.minimize",
            "coils_to_vtk",
            "surface.to_vtk",
        ),
    },
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
    SURFACE_AREA_VOLUME_SIMPLE_SPEC.fixture_id: FixtureRecord(
        spec=SURFACE_AREA_VOLUME_SIMPLE_SPEC,
        builder=_build_surface_area_volume_simple,
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
        builder=_build_position_orientation_flux_fixed_state,
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
        builder=_build_finite_beta_target_flux_fixed_state,
    ),
    FINITEBUILD_MULTIFILAMENT_SUPPORT_GATE_SPEC.fixture_id: FixtureRecord(
        spec=FINITEBUILD_MULTIFILAMENT_SUPPORT_GATE_SPEC,
        builder=_build_finitebuild_support_gate_probe,
    ),
    QFM_SURFACE_SPEC.fixture_id: FixtureRecord(
        spec=QFM_SURFACE_SPEC,
        builder=_build_qfm_surface_fixed_state,
    ),
    PM_SIMPLE_FIXED_STATE_GPMO_BASELINE_SPEC.fixture_id: FixtureRecord(
        spec=PM_SIMPLE_FIXED_STATE_GPMO_BASELINE_SPEC,
        builder=_build_pm_simple_fixed_state_gpmo_baseline,
    ),
    WIREFRAME_RCLS_BASIC_FIXED_STATE_SPEC.fixture_id: FixtureRecord(
        spec=WIREFRAME_RCLS_BASIC_FIXED_STATE_SPEC,
        builder=_build_wireframe_rcls_basic_fixed_state,
    ),
    TRACING_FIELDLINES_QA_REDUCED_ENDPOINT_SPEC.fixture_id: FixtureRecord(
        spec=TRACING_FIELDLINES_QA_REDUCED_ENDPOINT_SPEC,
        builder=_build_tracing_fieldlines_qa_reduced_endpoint,
    ),
    PM_QA_FIXED_STATE_GPMO_ARB_VEC_OR_MULTI_SPEC.fixture_id: FixtureRecord(
        spec=PM_QA_FIXED_STATE_GPMO_ARB_VEC_OR_MULTI_SPEC,
        builder=_build_pm_qa_relax_and_split_fixed_state,
    ),
    PM_MUSE_FAMUS_SPEC.fixture_id: FixtureRecord(
        spec=PM_MUSE_FAMUS_SPEC,
        builder=_build_pm_muse_famus_arbvec_backtracking,
    ),
    PM_PM4STELL_BACKTRACKING_SPEC.fixture_id: FixtureRecord(
        spec=PM_PM4STELL_BACKTRACKING_SPEC,
        builder=_build_pm_pm4stell_arbvec_backtracking,
    ),
    WIREFRAME_RCLS_PORTS_CONSTRAINT_GATE_SPEC.fixture_id: FixtureRecord(
        spec=WIREFRAME_RCLS_PORTS_CONSTRAINT_GATE_SPEC,
        builder=_build_wireframe_rcls_ports_constraint_fixed_state,
    ),
    WIREFRAME_GSCO_MODULAR_FIXED_STATE_SPEC.fixture_id: FixtureRecord(
        spec=WIREFRAME_GSCO_MODULAR_FIXED_STATE_SPEC,
        builder=_build_wireframe_gsco_modular_fixed_state,
    ),
    WIREFRAME_GSCO_SECTOR_SADDLE_FIXED_STATE_SPEC.fixture_id: FixtureRecord(
        spec=WIREFRAME_GSCO_SECTOR_SADDLE_FIXED_STATE_SPEC,
        builder=_build_wireframe_gsco_sector_saddle_fixed_state,
    ),
    WIREFRAME_GSCO_MULTISTEP_REDUCED_DIAGNOSTIC_SPEC.fixture_id: FixtureRecord(
        spec=WIREFRAME_GSCO_MULTISTEP_REDUCED_DIAGNOSTIC_SPEC,
        builder=_build_wireframe_gsco_multistep_reduced_diagnostic,
    ),
    TRACING_FIELDLINES_NCSX_REDUCED_ENDPOINT_SPEC.fixture_id: FixtureRecord(
        spec=TRACING_FIELDLINES_NCSX_REDUCED_ENDPOINT_SPEC,
        builder=_build_tracing_fieldlines_ncsx_reduced_endpoint,
    ),
    TRACING_PARTICLE_GC_VAC_REDUCED_ENDPOINT_SPEC.fixture_id: FixtureRecord(
        spec=TRACING_PARTICLE_GC_VAC_REDUCED_ENDPOINT_SPEC,
        builder=_build_tracing_particle_gc_vac_reduced_endpoint,
    ),
    TRACING_BOOZER_GC_REDUCED_ENDPOINT_SPEC.fixture_id: FixtureRecord(
        spec=TRACING_BOOZER_GC_REDUCED_ENDPOINT_SPEC,
        builder=_build_tracing_boozer_gc_reduced_endpoint,
    ),
    STRAIN_OPTIMIZATION_SUPPORT_GATE_SPEC.fixture_id: FixtureRecord(
        spec=STRAIN_OPTIMIZATION_SUPPORT_GATE_SPEC,
        builder=_build_strain_optimization_fixed_state,
    ),
    COIL_FORCES_SUPPORT_GATE_SPEC.fixture_id: FixtureRecord(
        spec=COIL_FORCES_SUPPORT_GATE_SPEC,
        builder=_build_coil_force_energy_fixed_state,
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
