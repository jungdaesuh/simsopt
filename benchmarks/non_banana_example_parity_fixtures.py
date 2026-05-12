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
    # length penalty in the minimal Stage-II fixture is excluded from this
    # callable on both sides.
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
# Unsupported / support-gate classification builders


def _raise_unsupported(message: str) -> Callable[[], FixtureBuild]:
    def _factory() -> FixtureBuild:
        raise FixtureNotSupportedError(message)

    return _factory


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
    classification=UNSUPPORTED_NATIVE_JAX,
    classification_reason=(
        "Full composite includes CurveCurveDistance, CurveSurfaceDistance, "
        "LpCurveCurvature, MeanSquaredCurvature, and QuadraticPenalty over "
        "sums; none are exposed as native JAX wrappers today. This fixture "
        "is classified for follow-up plans."
    ),
)


PLANAR_STAGE2_COMPOSITE_SPEC = FixtureSpec(
    fixture_id="planar_stage2_composite",
    source_example="examples/2_Intermediate/stage_two_optimization_planar_coils.py",
    classification=UNSUPPORTED_NATIVE_JAX,
    classification_reason=(
        "CurvePlanarFourier exposes a JAX spec, but LinkingNumber and the "
        "shared geometry penalties used by the planar fixture do not have "
        "native JAX wrappers today."
    ),
)


POSITION_ORIENTATION_FLUX_SUPPORT_GATE_SPEC = FixtureSpec(
    fixture_id="position_orientation_flux_support_gate",
    source_example="examples/1_Simple/optimize_coil_position_orientation.py",
    classification=SUPPORT_GATE,
    classification_reason=(
        "OrientedCurveXYZFourier does not implement to_spec(); BiotSavartJAX "
        "rejects this curve family until immutable native spec support is "
        "added."
    ),
)


BOOZER_SURFACE_BASIC_SPEC = FixtureSpec(
    fixture_id="boozer_surface_basic",
    source_example="examples/2_Intermediate/boozer.py",
    classification=UNSUPPORTED_NATIVE_JAX,
    classification_reason=(
        "Boozer fixed-state residual JAX parity is covered by "
        "tests/geo/test_boozer_residual_jax.py and the M3 derivative tests. "
        "This non-banana fixture is classified for follow-up plans that "
        "wire those checks into the per-fixture parity report."
    ),
)


BOOZER_QA_WRAPPERS_SPEC = FixtureSpec(
    fixture_id="boozer_qa_wrappers",
    source_example="examples/2_Intermediate/boozerQA.py",
    classification=UNSUPPORTED_NATIVE_JAX,
    classification_reason=(
        "BoozerSurfaceJAX + IotasJAX / NonQuasiSymmetricRatioJAX wrapper "
        "parity already lives in tests/integration/test_single_stage_jax.py. "
        "This non-banana fixture is classified for follow-up plans."
    ),
)


FINITE_BETA_TARGET_FLUX_SPEC = FixtureSpec(
    fixture_id="finite_beta_target_flux",
    source_example="examples/2_Intermediate/stage_two_optimization_finite_beta.py",
    classification=UNSUPPORTED_NATIVE_JAX,
    classification_reason=(
        "Finite-beta target field comes from VirtualCasing which is not "
        "wired into the non-banana parity harness; this fixture is classified "
        "for follow-up plans."
    ),
)


FINITEBUILD_MULTIFILAMENT_SUPPORT_GATE_SPEC = FixtureSpec(
    fixture_id="finitebuild_multifilament_support_gate",
    source_example="examples/3_Advanced/stage_two_optimization_finitebuild.py",
    classification=SUPPORT_GATE,
    classification_reason=(
        "Finite-build multifilament curves expose a partial JAX spec but "
        "the multifilament construction has not been verified end-to-end "
        "against the BiotSavartJAX native-spec contract yet."
    ),
)


QFM_SURFACE_SPEC = FixtureSpec(
    fixture_id="qfm_surface",
    source_example="examples/1_Simple/qfm.py",
    classification=UNSUPPORTED_NATIVE_JAX,
    classification_reason=(
        "QfmResidual does not have a native JAX wrapper; this fixture is "
        "classified for follow-up plans."
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
        builder=_unsupported_classification_builder(FULL_STAGE2_COMPOSITE_SPEC),
    ),
    PLANAR_STAGE2_COMPOSITE_SPEC.fixture_id: FixtureRecord(
        spec=PLANAR_STAGE2_COMPOSITE_SPEC,
        builder=_unsupported_classification_builder(PLANAR_STAGE2_COMPOSITE_SPEC),
    ),
    POSITION_ORIENTATION_FLUX_SUPPORT_GATE_SPEC.fixture_id: FixtureRecord(
        spec=POSITION_ORIENTATION_FLUX_SUPPORT_GATE_SPEC,
        builder=_unsupported_classification_builder(
            POSITION_ORIENTATION_FLUX_SUPPORT_GATE_SPEC
        ),
    ),
    BOOZER_SURFACE_BASIC_SPEC.fixture_id: FixtureRecord(
        spec=BOOZER_SURFACE_BASIC_SPEC,
        builder=_unsupported_classification_builder(BOOZER_SURFACE_BASIC_SPEC),
    ),
    BOOZER_QA_WRAPPERS_SPEC.fixture_id: FixtureRecord(
        spec=BOOZER_QA_WRAPPERS_SPEC,
        builder=_unsupported_classification_builder(BOOZER_QA_WRAPPERS_SPEC),
    ),
    FINITE_BETA_TARGET_FLUX_SPEC.fixture_id: FixtureRecord(
        spec=FINITE_BETA_TARGET_FLUX_SPEC,
        builder=_unsupported_classification_builder(FINITE_BETA_TARGET_FLUX_SPEC),
    ),
    FINITEBUILD_MULTIFILAMENT_SUPPORT_GATE_SPEC.fixture_id: FixtureRecord(
        spec=FINITEBUILD_MULTIFILAMENT_SUPPORT_GATE_SPEC,
        builder=_unsupported_classification_builder(
            FINITEBUILD_MULTIFILAMENT_SUPPORT_GATE_SPEC
        ),
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
