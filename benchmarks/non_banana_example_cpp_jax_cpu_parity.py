"""Non-banana example CPU C++/JAX parity benchmark (Phase 0–8 implementation).

Implements the harness described by
``docs/non_banana_example_cpp_jax_cpu_parity_plan_2026-05-12.md``:

* Phase 0 — baseline metadata + fixture-contract gates (x64, no-GPU,
  native-spec).
* Phase 1 — P0 ``minimal_stage2_flux_length_gap`` fixed-state parity
  (SquaredFlux / SquaredFluxJAX, B, B·n, surface geometry, gradient,
  deterministic perturbation diagnostics).
* Phase 2 — P1 ``cws_saved_local_flux_nfp{2,3}`` saved-artifact fixtures
  compare reconstructed local-flux states after the CurveCWSFourier
  deserializer gap was routed around in the fixture layer.
* Phase 3/4 — full and planar Stage-II fixed-state fixtures. Their
  native-supported ``SquaredFlux`` subproblems are compared, while CPU-only
  geometry penalties are listed in ``unsupported_components``.
* Phase 5/7 — position/orientation and finite-build support gates with live
  CPU fixture probes.
* Phase 6/7/8 — the basic Boozer fixed-state fixture is wired for
  residual/label parity, the BoozerQA wrappers fixture is wired for
  fixed-solved-state parity of Iotas / MajorRadius /
  NonQuasiSymmetricRatio, and finite-beta / QFM / force-energy rows record
  partial parity with their remaining host-solver or independent-oracle
  blockers named explicitly.

CPU is the default execution mode. A separate ``jax_gpu`` mode is available
only when the process is launched with the explicit CUDA parity environment
and a CPU baseline artifact is supplied for JAX CPU vs JAX GPU comparison.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

LANE_CPU_CPP = "cpu_cpp"
LANE_JAX_CPU = "jax_cpu"
LANE_JAX_GPU = "jax_gpu"
SUPPORTED_LANES = frozenset((LANE_CPU_CPP, LANE_JAX_CPU, LANE_JAX_GPU))

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def _expose_current_tree_to_simsopt_editable_finder() -> None:
    """Make this repo's ``src/simsopt`` visible to scikit-build editables.

    Some parity environments install a simsoptpp-backed editable SIMSOPT
    through scikit-build. Its meta-path finder intercepts ``simsopt.*``
    before normal ``sys.path`` resolution, so adding ``src/`` is not enough
    for direct CLI runs. This benchmark is current-tree evidence, therefore
    the local Python modules are the authoritative source side.
    """
    simsopt_src = SRC_ROOT / "simsopt"
    if not simsopt_src.exists():
        raise RuntimeError(f"Missing local simsopt source tree: {simsopt_src}")

    source_files: dict[str, str] = {}
    package_locations: dict[str, str] = {}
    for py_file in simsopt_src.rglob("*.py"):
        rel_path = py_file.relative_to(simsopt_src)
        module_parts = rel_path.parent.parts
        if py_file.name != "__init__.py":
            module_parts = rel_path.with_suffix("").parts
        module_name = "simsopt"
        if module_parts:
            module_name += "." + ".".join(module_parts)

        source_files[module_name] = str(py_file)
        if py_file.name == "__init__.py":
            package_locations[module_name] = str(py_file.parent)

    patched = False
    for finder in sys.meta_path:
        known_source_files = getattr(finder, "known_source_files", None)
        submodule_locations = getattr(finder, "submodule_search_locations", None)
        if not isinstance(known_source_files, dict) or not isinstance(
            submodule_locations, dict
        ):
            continue
        known_source_files.update(source_files)
        for package_name, package_path in package_locations.items():
            submodule_locations.setdefault(package_name, set()).add(package_path)
        patched = True

    if patched:
        importlib.invalidate_caches()


_expose_current_tree_to_simsopt_editable_finder()


def _preimport_selected_lanes(argv: Sequence[str]) -> Sequence[str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--lanes",
        default=f"{LANE_CPU_CPP},{LANE_JAX_CPU}",
    )
    parsed, _ = parser.parse_known_args(argv)
    return tuple(part.strip() for part in parsed.lanes.split(",") if part.strip())


def _requested_jax_platform() -> str:
    """Return the single JAX platform this process is allowed to initialize."""
    requested = os.environ.get("SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM", "cpu")
    if requested not in {"cpu", "cuda"}:
        raise RuntimeError(
            "SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM must be "
            f"'cpu' or 'cuda'; got {requested!r}."
        )
    if requested == "cuda" and LANE_JAX_GPU not in _preimport_selected_lanes(
        sys.argv[1:]
    ):
        return "cpu"
    return requested


_REQUESTED_JAX_PLATFORM = _requested_jax_platform()

# ``jax`` is configured at import time so subsequent imports see exactly one
# parity runtime. CPU remains the default, even if a parent shell exported a
# broader platform list. CUDA requires both SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=cuda
# and a jax_gpu lane before this module is imported.
os.environ["JAX_PLATFORMS"] = _REQUESTED_JAX_PLATFORM
os.environ["JAX_ENABLE_X64"] = "1"

import jax  # noqa: E402  (after env-var setup)
import jaxlib  # noqa: E402  (after env-var setup)

jax.config.update("jax_platforms", _REQUESTED_JAX_PLATFORM)
jax.config.update("jax_enable_x64", True)

from benchmarks.non_banana_example_parity_fixtures import (  # noqa: E402
    FixtureBuild,
    FixtureNotSupportedError,
    FixtureRecord,
    LaneArtifact,
    SCHEMA_VERSION,
    SUPPORTED,
    fixture_ids,
    fixed_state_input_hash,
    get_fixture,
    gpu_readiness_metadata,
    supported_fixture_ids,
)
from benchmarks.validation_ladder_common import (  # noqa: E402
    current_xla_cuda_metadata,
    query_nvidia_smi_facts,
)
from benchmarks.validation_ladder_contract import (  # noqa: E402
    parity_ladder_tolerances,
)


CUDA_DEVICE_PLATFORMS = frozenset(("cuda", "gpu"))


# ---------------------------------------------------------------------------
# Tolerance mapping (mirrors plan §"Default Tolerance Mapping").


_TOLERANCE_BUCKETS = {
    "field_B": "direct_kernel",
    "field_GradAbsB": "direct_kernel",
    "field_modB": "direct_kernel",
    "surface_gamma": "direct_kernel",
    "surface_unit_normal": "direct_kernel",
    "Bdotn": "direct_kernel",
    "objective_native_subtotal": "ls_wrapper_gradient",
    "SquaredFlux": "ls_wrapper_gradient",
    "SquaredFluxJAX": "ls_wrapper_gradient",
    "gradient": "ls_wrapper_gradient",
    # Phase 6 fixed-state Boozer fixture: residual vector and labels are
    # direct kernel comparisons (no LS/gradient wrapper involved).
    "boozer_residual": "direct_kernel",
    "area": "direct_kernel",
    "volume": "direct_kernel",
    "area_gradient": "derivative_heavy",
    "volume_gradient": "derivative_heavy",
    "qfm_residual": "direct_kernel",
    "qfm_gradient": "derivative_heavy",
    "pm_grid_payload": "direct_kernel",
    "pm_moments": "direct_kernel",
    "pm_residual": "direct_kernel",
    "pm_proxy_residual": "direct_kernel",
    "pm_objective": "direct_kernel",
    "pm_proxy_objective": "direct_kernel",
    "pm_history": "direct_kernel",
    "pm_dipole_field_B": "direct_kernel",
    "pm_proxy_dipole_field_B": "direct_kernel",
    "pm_dipole_Bdotn": "direct_kernel",
    "pm_proxy_dipole_Bdotn": "direct_kernel",
    "wireframe_matrix": "direct_kernel",
    "wireframe_current": "direct_kernel",
    "wireframe_objective": "direct_kernel",
    "wireframe_constraints": "direct_kernel",
    "wireframe_field_B": "direct_kernel",
    "wireframe_field_dB_by_dX": "derivative_heavy",
    "wireframe_Bnormal": "direct_kernel",
    "wireframe_gsco_flags": "direct_kernel",
    "wireframe_gsco_history": "direct_kernel",
    "wireframe_gsco_solution": "direct_kernel",
    "trajectory_endpoint": "event_time_tracing",
    "trajectory_t_final": "event_time_tracing",
    "trajectory_status_code": "direct_kernel",
    "phi_hit_xyz": "event_time_tracing",
    "phi_hit_count": "direct_kernel",
    "toroidal_flux": "direct_kernel",
    "LpCurveForce": "direct_kernel",
    "B2Energy": "direct_kernel",
    "lp_curve_force_gradient": "derivative_heavy",
    "b2_energy_gradient": "derivative_heavy",
    # Phase 6 boozerQA wrappers fixture: each wrapper value is compared as
    # a direct-kernel scalar at the shared solved (surface, iota, G)
    # state. The JAX-side recomputations are pure JAX (no LS solver, no
    # adjoint), so direct_kernel is the appropriate bucket.
    "iota": "direct_kernel",
    "major_radius": "direct_kernel",
    "nq_symmetric_ratio": "direct_kernel",
}


_DOF_NAME_COUNTER_RE = __import__("re").compile(r"^([A-Za-z_][A-Za-z_]*)(\d+)(:.*)$")


def _strip_dof_name_counter(name: str) -> str:
    """Strip simsopt's per-instance counter from a DOF name.

    Example: ``"CurveXYZFourier5:xs(1)"`` -> ``"CurveXYZFourier:xs(1)"``.
    Names that do not match the expected pattern are returned unchanged.
    """
    match = _DOF_NAME_COUNTER_RE.match(str(name))
    if match is None:
        return str(name)
    cls_part, _counter, dof_part = match.groups()
    return f"{cls_part}{dof_part}"


def _tolerance_for(quantity: str) -> tuple[str, float, float]:
    bucket = _TOLERANCE_BUCKETS.get(quantity, "direct_kernel")
    tolerances = parity_ladder_tolerances(bucket)
    if bucket == "event_time_tracing":
        return (
            bucket,
            float(tolerances["state_vector_rtol"]),
            float(tolerances["state_vector_atol"]),
        )
    if "first_derivative_rtol" in tolerances and "rtol" not in tolerances:
        rtol = float(tolerances["first_derivative_rtol"])
        atol = float(tolerances["first_derivative_atol"])
    else:
        rtol = float(tolerances["rtol"])
        atol = float(tolerances["atol"])
    return bucket, rtol, atol


# ---------------------------------------------------------------------------
# Comparison helpers.


def _compare_array(
    *,
    cpu_arr: np.ndarray,
    jax_arr: np.ndarray,
    quantity: str,
    component: str,
    active_dof_names: Sequence[str],
) -> Mapping[str, Any]:
    bucket, rtol, atol = _tolerance_for(quantity)
    cpu = np.asarray(cpu_arr, dtype=np.float64)
    jax_a = np.asarray(jax_arr, dtype=np.float64)

    if cpu.shape != jax_a.shape:
        return {
            "quantity": quantity,
            "component": component,
            "source_example": None,
            "cpu_cpp_value": None,
            "jax_cpu_value": None,
            "tolerance_bucket": bucket,
            "rtol": rtol,
            "atol": atol,
            "tolerance_rtol": rtol,
            "tolerance_atol": atol,
            "max_abs_diff": None,
            "max_rel_diff": None,
            "argmax_index": None,
            "argmax_dof_name": None,
            "verdict": "fail",
            "failure_reason": f"shape mismatch cpu={cpu.shape} jax={jax_a.shape}",
        }

    diff = jax_a - cpu
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    denom = atol + rtol * np.abs(cpu)
    rel_excess = abs_diff - denom
    argmax_flat = int(rel_excess.argmax()) if rel_excess.size else 0
    if rel_excess.size:
        argmax_index = np.unravel_index(argmax_flat, cpu.shape) if cpu.ndim else (0,)
    else:
        argmax_index = ()
    # Always use the (atol-cushioned) relative formula. ``atol`` keeps the
    # denominator strictly positive for entries that are exactly zero, so
    # there is no need to branch on a single argmax entry being zero.
    if abs_diff.size:
        max_rel = float((abs_diff / (np.abs(cpu) + atol)).max())
    else:
        max_rel = 0.0

    argmax_dof_name = None
    if quantity == "gradient" and cpu.ndim == 1 and len(active_dof_names) == cpu.size:
        argmax_dof_name = str(active_dof_names[argmax_flat])

    passed = bool(np.all(abs_diff <= denom))
    verdict = "pass" if passed else "fail"
    entry = {
        "quantity": quantity,
        "component": component,
        "source_example": None,
        "cpu_cpp_value": cpu.tolist(),
        "jax_cpu_value": jax_a.tolist(),
        "tolerance_bucket": bucket,
        "rtol": rtol,
        "atol": atol,
        "tolerance_rtol": rtol,
        "tolerance_atol": atol,
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "argmax_index": list(int(i) for i in np.atleast_1d(argmax_index).tolist()),
        "argmax_dof_name": argmax_dof_name,
        "verdict": verdict,
    }
    return entry


def _compare_scalar(
    *,
    cpu_value: float,
    jax_value: float,
    quantity: str,
    component: str,
) -> Mapping[str, Any]:
    bucket, rtol, atol = _tolerance_for(quantity)
    abs_diff = abs(jax_value - cpu_value)
    denom = atol + rtol * abs(cpu_value)
    rel = abs_diff / (abs(cpu_value) + atol)
    passed = abs_diff <= denom
    return {
        "quantity": quantity,
        "component": component,
        "source_example": None,
        "cpu_cpp_value": float(cpu_value),
        "jax_cpu_value": float(jax_value),
        "tolerance_bucket": bucket,
        "rtol": rtol,
        "atol": atol,
        "tolerance_rtol": rtol,
        "tolerance_atol": atol,
        "max_abs_diff": float(abs_diff),
        "max_rel_diff": float(rel),
        "argmax_index": None,
        "argmax_dof_name": None,
        "verdict": "pass" if passed else "fail",
    }


def _retarget_comparison_entry(
    entry: Mapping[str, Any],
    *,
    left_lane: str,
    right_lane: str,
) -> Mapping[str, Any]:
    """Attach explicit lane labels and lane-specific value keys."""
    retargeted = dict(entry)
    left_value = retargeted.pop("cpu_cpp_value")
    right_value = retargeted.pop("jax_cpu_value")
    retargeted["left_lane"] = left_lane
    retargeted["right_lane"] = right_lane
    retargeted["left_value"] = left_value
    retargeted["right_value"] = right_value
    retargeted[f"{left_lane}_value"] = left_value
    retargeted[f"{right_lane}_value"] = right_value
    return retargeted


def _retarget_comparison_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    left_lane: str,
    right_lane: str,
) -> Sequence[Mapping[str, Any]]:
    return tuple(
        _retarget_comparison_entry(
            entry,
            left_lane=left_lane,
            right_lane=right_lane,
        )
        for entry in entries
    )


def _compare_json_values(
    *,
    left_value: Any,
    right_value: Any,
    quantity: str,
    component: str,
    left_lane: str,
    right_lane: str,
) -> Mapping[str, Any]:
    if left_value is None or right_value is None:
        bucket, rtol, atol = _tolerance_for(quantity)
        return {
            "quantity": quantity,
            "component": component,
            "source_example": None,
            "left_lane": left_lane,
            "right_lane": right_lane,
            "left_value": left_value,
            "right_value": right_value,
            f"{left_lane}_value": left_value,
            f"{right_lane}_value": right_value,
            "tolerance_bucket": bucket,
            "rtol": rtol,
            "atol": atol,
            "tolerance_rtol": rtol,
            "tolerance_atol": atol,
            "max_abs_diff": None,
            "max_rel_diff": None,
            "argmax_index": None,
            "argmax_dof_name": None,
            "verdict": "fail",
            "failure_reason": "missing baseline or GPU comparison value",
        }

    left = np.asarray(left_value, dtype=np.float64)
    right = np.asarray(right_value, dtype=np.float64)
    if left.ndim == 0 and right.ndim == 0:
        entry = _compare_scalar(
            cpu_value=float(left),
            jax_value=float(right),
            quantity=quantity,
            component=component,
        )
    else:
        entry = _compare_array(
            cpu_arr=left,
            jax_arr=right,
            quantity=quantity,
            component=component,
            active_dof_names=(),
        )
    return _retarget_comparison_entry(
        entry,
        left_lane=left_lane,
        right_lane=right_lane,
    )


def _block_lane_artifact_outputs(lane: LaneArtifact) -> None:
    """Synchronize lane arrays before they are serialized into an artifact."""
    arrays = tuple(lane.raw_arrays.values())
    if lane.gradient is None:
        jax.block_until_ready(arrays)
    else:
        jax.block_until_ready((*arrays, lane.gradient))


# ---------------------------------------------------------------------------
# Fixture evaluation.


@dataclass
class FixtureResult:
    fixture_id: str
    source_example: str
    classification: str
    classification_reason: str
    fixture_inputs: Mapping[str, Any]
    dof_contract: Mapping[str, Any]
    native_spec_contract: Mapping[str, Any]
    lanes: Mapping[str, Mapping[str, Any]]
    comparisons: Mapping[str, Sequence[Mapping[str, Any]]]
    unsupported_components: Sequence[str]
    mixed_lane_diagnostics: Sequence[str]
    perturbation_diagnostics: Optional[Mapping[str, Any]]
    verdict: str
    passed: bool
    failures: Sequence[str]
    error: Optional[str] = None


def _lane_to_jsonable(
    lane: LaneArtifact,
    *,
    lane_name: Optional[str] = None,
    gpu_proven: bool = False,
) -> Mapping[str, Any]:
    emitted_lane_name = lane_name or lane.lane
    return {
        "lane": emitted_lane_name,
        "objective_total": lane.objective_total,
        "objective_native_subtotal": lane.objective_native_subtotal,
        "components": dict(lane.components),
        "gradient_norm": lane.gradient_norm,
        "active_dof_names": list(lane.active_dof_names),
        "active_dof_hash": lane.active_dof_hash,
        "fixed_free_mask_hash": lane.fixed_free_mask_hash,
        "native_curve_spec_hashes": list(lane.native_curve_spec_hashes),
        "surface_point_hash": lane.surface_point_hash,
        "unit_normal_hash": lane.unit_normal_hash,
        "field_B_hash": lane.field_B_hash,
        "field_B_max": lane.field_B_max,
        "field_B_mean": lane.field_B_mean,
        "Bdotn_array_hash": lane.Bdotn_array_hash,
        "Bdotn_max": lane.Bdotn_max,
        "Bdotn_mean": lane.Bdotn_mean,
        "gpu_readiness": dict(gpu_readiness_metadata(proven=gpu_proven)),
        "timing": dict(lane.timing),
    }


def _supported_comparisons(build: FixtureBuild) -> Sequence[Mapping[str, Any]]:
    """Compute comparison entries for every native-supported quantity.

    Branches on ``build.spec.fixture_kind``: the default
    ``biot_savart_squared_flux`` kind compares surface geometry, field B,
    B·n, the SquaredFlux scalar, the objective_native_subtotal, and the
    gradient. The ``boozer_surface_fixed_state`` kind compares surface
    geometry, field B, the Boozer residual vector, and the Area / Volume /
    ToroidalFlux scalars; the SquaredFlux / gradient comparisons are
    skipped because they are not part of the Boozer fixed-state contract.
    The ``boozer_qa_wrappers_solved_state`` kind compares surface
    geometry, field B, and the four native-supported QA scalar values
    corresponding to the upstream wrappers (Iotas, MajorRadius,
    NonQuasiSymmetricRatio, and sum(CurveLength)) at the CPU-solved state.
    The JAX lane uses the solved iota scalar plus pure-JAX helper functions
    over the copied solved surface DOFs and ``CurveLengthJAX`` over an
    independent NCSX curve tree; it does not claim public
    ``BoozerSurfaceJAX`` wrapper or adjoint parity. Gradients are not compared
    in this fixture.
    """
    cpu = build.cpu_lane
    jax_lane = build.jax_lane
    fixture_kind = build.spec.fixture_kind

    if fixture_kind == "boozer_surface_fixed_state":
        return _boozer_fixed_state_comparisons(cpu, jax_lane)
    if fixture_kind == "boozer_qa_wrappers_solved_state":
        return _boozer_qa_wrappers_comparisons(cpu, jax_lane)
    if fixture_kind == "surface_scalar":
        return _surface_scalar_comparisons(cpu, jax_lane)
    if fixture_kind == "qfm":
        return _qfm_comparisons(cpu, jax_lane)
    if fixture_kind == "pm":
        return _pm_comparisons(cpu, jax_lane)
    if fixture_kind == "pm_relax_and_split":
        return _pm_relax_and_split_comparisons(cpu, jax_lane)
    if fixture_kind == "wireframe":
        return _wireframe_comparisons(cpu, jax_lane)
    if fixture_kind == "wireframe_gsco":
        return _wireframe_gsco_comparisons(cpu, jax_lane)
    if fixture_kind == "tracing":
        return _tracing_comparisons(cpu, jax_lane)
    if fixture_kind == "strain":
        return _strain_comparisons(cpu, jax_lane)
    if fixture_kind == "coil_force_energy":
        return _coil_force_energy_comparisons(cpu, jax_lane)

    comparisons = []

    # Surface geometry first (independent of field/objective).
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_gamma"],
            jax_arr=jax_lane.raw_arrays["surface_gamma"],
            quantity="surface_gamma",
            component="surface",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_unit_normal"],
            jax_arr=jax_lane.raw_arrays["surface_unit_normal"],
            quantity="surface_unit_normal",
            component="surface",
            active_dof_names=cpu.active_dof_names,
        )
    )

    # Field-level parity.
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["field_B"],
            jax_arr=jax_lane.raw_arrays["field_B"],
            quantity="field_B",
            component="biot_savart",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["Bdotn"],
            jax_arr=jax_lane.raw_arrays["Bdotn"],
            quantity="Bdotn",
            component="biot_savart",
            active_dof_names=cpu.active_dof_names,
        )
    )

    # Wrapper objective + gradient.
    comparisons.append(
        _compare_scalar(
            cpu_value=cpu.components.get("SquaredFlux"),
            jax_value=jax_lane.components.get("SquaredFluxJAX"),
            quantity="SquaredFlux",
            component="objective",
        )
    )
    # Native-supported objective subtotal: for the current fixtures this
    # equals SquaredFlux, but the explicit comparison gates the
    # ``objective_native_subtotal`` lane field so future composites that
    # add native components surface a real cross-lane check.
    comparisons.append(
        _compare_scalar(
            cpu_value=cpu.objective_native_subtotal,
            jax_value=jax_lane.objective_native_subtotal,
            quantity="objective_native_subtotal",
            component="objective",
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["gradient"],
            jax_arr=jax_lane.raw_arrays["gradient"],
            quantity="gradient",
            component="objective",
            active_dof_names=cpu.active_dof_names,
        )
    )
    return comparisons


def _surface_scalar_comparisons(
    cpu: LaneArtifact,
    jax_lane: LaneArtifact,
) -> Sequence[Mapping[str, Any]]:
    """Compare Area/Volume example quantities at fixed surface states."""
    return [
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_gamma"],
            jax_arr=jax_lane.raw_arrays["surface_gamma"],
            quantity="surface_gamma",
            component="surface",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_unit_normal"],
            jax_arr=jax_lane.raw_arrays["surface_unit_normal"],
            quantity="surface_unit_normal",
            component="surface",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_scalar(
            cpu_value=cpu.components["area"],
            jax_value=jax_lane.components["area"],
            quantity="area",
            component="surface_scalar",
        ),
        _compare_scalar(
            cpu_value=cpu.components["volume"],
            jax_value=jax_lane.components["volume"],
            quantity="volume",
            component="surface_scalar",
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["area_gradient"],
            jax_arr=jax_lane.raw_arrays["area_gradient"],
            quantity="area_gradient",
            component="surface_scalar",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["volume_gradient"],
            jax_arr=jax_lane.raw_arrays["volume_gradient"],
            quantity="volume_gradient",
            component="surface_scalar",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["area_perturbed_values"],
            jax_arr=jax_lane.raw_arrays["area_perturbed_values"],
            quantity="area_perturbed_values",
            component="surface_scalar",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["volume_perturbed_values"],
            jax_arr=jax_lane.raw_arrays["volume_perturbed_values"],
            quantity="volume_perturbed_values",
            component="surface_scalar",
            active_dof_names=cpu.active_dof_names,
        ),
    ]


def _strain_comparisons(
    cpu: LaneArtifact,
    jax_lane: LaneArtifact,
) -> Sequence[Mapping[str, Any]]:
    """Compare fixed-state strain quantities for the rotation-only example."""
    return [
        _compare_array(
            cpu_arr=cpu.raw_arrays["torsional_strain"],
            jax_arr=jax_lane.raw_arrays["torsional_strain"],
            quantity="torsional_strain",
            component="strain",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["binormal_curvature_strain"],
            jax_arr=jax_lane.raw_arrays["binormal_curvature_strain"],
            quantity="binormal_curvature_strain",
            component="strain",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_scalar(
            cpu_value=cpu.components["torsional_penalty"],
            jax_value=jax_lane.components["torsional_penalty"],
            quantity="torsional_penalty",
            component="strain_objective",
        ),
        _compare_scalar(
            cpu_value=cpu.components["binormal_curvature_penalty"],
            jax_value=jax_lane.components["binormal_curvature_penalty"],
            quantity="binormal_curvature_penalty",
            component="strain_objective",
        ),
        _compare_scalar(
            cpu_value=cpu.objective_native_subtotal,
            jax_value=jax_lane.objective_native_subtotal,
            quantity="objective_native_subtotal",
            component="strain_objective",
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["gradient"],
            jax_arr=jax_lane.raw_arrays["gradient"],
            quantity="gradient",
            component="strain_objective",
            active_dof_names=cpu.active_dof_names,
        ),
    ]


def _coil_force_energy_comparisons(
    cpu: LaneArtifact,
    jax_lane: LaneArtifact,
) -> Sequence[Mapping[str, Any]]:
    """Compare fixed-state coil force and magnetic-energy wrappers."""
    return [
        _compare_scalar(
            cpu_value=cpu.components["LpCurveForce"],
            jax_value=jax_lane.components["LpCurveForceJAX"],
            quantity="LpCurveForce",
            component="force_objective",
        ),
        _compare_scalar(
            cpu_value=cpu.components["LpCurveForce_independent_oracle"],
            jax_value=jax_lane.components["LpCurveForceJAX"],
            quantity="LpCurveForce_independent_oracle",
            component="force_objective",
        ),
        _compare_scalar(
            cpu_value=cpu.components["B2Energy"],
            jax_value=jax_lane.components["B2EnergyJAX"],
            quantity="B2Energy",
            component="energy_objective",
        ),
        _compare_scalar(
            cpu_value=cpu.components["B2Energy_independent_oracle"],
            jax_value=jax_lane.components["B2EnergyJAX"],
            quantity="B2Energy_independent_oracle",
            component="energy_objective",
        ),
        _compare_scalar(
            cpu_value=cpu.objective_native_subtotal,
            jax_value=jax_lane.objective_native_subtotal,
            quantity="objective_native_subtotal",
            component="coil_force_energy_objective",
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["lp_curve_force_gradient"],
            jax_arr=jax_lane.raw_arrays["lp_curve_force_gradient"],
            quantity="lp_curve_force_gradient",
            component="force_objective",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["b2_energy_gradient"],
            jax_arr=jax_lane.raw_arrays["b2_energy_gradient"],
            quantity="b2_energy_gradient",
            component="energy_objective",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["gradient"],
            jax_arr=jax_lane.raw_arrays["gradient"],
            quantity="gradient",
            component="coil_force_energy_objective",
            active_dof_names=cpu.active_dof_names,
        ),
    ]


def _qfm_comparisons(
    cpu: LaneArtifact,
    jax_lane: LaneArtifact,
) -> Sequence[Mapping[str, Any]]:
    """Compare fixed-state QFM residual and example label quantities."""
    return [
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_gamma"],
            jax_arr=jax_lane.raw_arrays["surface_gamma"],
            quantity="surface_gamma",
            component="surface",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_unit_normal"],
            jax_arr=jax_lane.raw_arrays["surface_unit_normal"],
            quantity="surface_unit_normal",
            component="surface",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["field_B"],
            jax_arr=jax_lane.raw_arrays["field_B"],
            quantity="field_B",
            component="biot_savart",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["Bdotn"],
            jax_arr=jax_lane.raw_arrays["Bdotn"],
            quantity="Bdotn",
            component="qfm_residual",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_scalar(
            cpu_value=cpu.components["qfm_residual"],
            jax_value=jax_lane.components["qfm_residual"],
            quantity="qfm_residual",
            component="qfm_residual",
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["qfm_gradient"],
            jax_arr=jax_lane.raw_arrays["qfm_gradient"],
            quantity="qfm_gradient",
            component="qfm_residual",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_scalar(
            cpu_value=cpu.components["area"],
            jax_value=jax_lane.components["area"],
            quantity="area",
            component="label",
        ),
        _compare_scalar(
            cpu_value=cpu.components["volume"],
            jax_value=jax_lane.components["volume"],
            quantity="volume",
            component="label",
        ),
        _compare_scalar(
            cpu_value=cpu.components["toroidal_flux"],
            jax_value=jax_lane.components["toroidal_flux"],
            quantity="toroidal_flux",
            component="label",
        ),
    ]


def _pm_comparisons(
    cpu: LaneArtifact,
    jax_lane: LaneArtifact,
) -> Sequence[Mapping[str, Any]]:
    """Compare reduced permanent-magnet fixed-state payload and result arrays."""
    algorithm_component = {
        0.0: "GPMO_baseline",
        3.0: "GPMO_ArbVec_backtracking",
    }[cpu.components["algorithm_variant"]]
    comparisons = [
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_gamma"],
            jax_arr=jax_lane.raw_arrays["surface_gamma"],
            quantity="surface_gamma",
            component="pm_surface",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_unit_normal"],
            jax_arr=jax_lane.raw_arrays["surface_unit_normal"],
            quantity="surface_unit_normal",
            component="pm_surface",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["A_obj"],
            jax_arr=jax_lane.raw_arrays["A_obj"],
            quantity="pm_grid_payload",
            component="A_obj",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["b_obj"],
            jax_arr=jax_lane.raw_arrays["b_obj"],
            quantity="pm_grid_payload",
            component="b_obj",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["m_maxima"],
            jax_arr=jax_lane.raw_arrays["m_maxima"],
            quantity="pm_grid_payload",
            component="m_maxima",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["dipole_grid_xyz"],
            jax_arr=jax_lane.raw_arrays["dipole_grid_xyz"],
            quantity="pm_grid_payload",
            component="dipole_grid_xyz",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["m"],
            jax_arr=jax_lane.raw_arrays["m"],
            quantity="pm_moments",
            component=algorithm_component,
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["residual"],
            jax_arr=jax_lane.raw_arrays["residual"],
            quantity="pm_residual",
            component=algorithm_component,
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_scalar(
            cpu_value=cpu.components["pm_objective"],
            jax_value=jax_lane.components["pm_objective"],
            quantity="pm_objective",
            component=algorithm_component,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["R2_history"],
            jax_arr=jax_lane.raw_arrays["R2_history"],
            quantity="pm_history",
            component=f"{algorithm_component}_R2_history",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["Bn_history"],
            jax_arr=jax_lane.raw_arrays["Bn_history"],
            quantity="pm_history",
            component=f"{algorithm_component}_Bn_history",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["dipole_B"],
            jax_arr=jax_lane.raw_arrays["dipole_B"],
            quantity="pm_dipole_field_B",
            component="DipoleField",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["dipole_Bn"],
            jax_arr=jax_lane.raw_arrays["dipole_Bn"],
            quantity="pm_dipole_Bdotn",
            component="DipoleField",
            active_dof_names=cpu.active_dof_names,
        ),
    ]
    if cpu.components["algorithm_variant"] in (0.0, 3.0):
        comparisons.append(
            _compare_array(
                cpu_arr=cpu.raw_arrays["m_history"],
                jax_arr=jax_lane.raw_arrays["m_history"],
                quantity="pm_history",
                component=f"{algorithm_component}_m_history",
                active_dof_names=cpu.active_dof_names,
            ),
        )
    return comparisons


def _pm_relax_and_split_comparisons(
    cpu: LaneArtifact,
    jax_lane: LaneArtifact,
) -> Sequence[Mapping[str, Any]]:
    """Compare reduced permanent-magnet relax-and-split payload and final states."""
    algorithm_component = "relax_and_split"
    return [
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_gamma"],
            jax_arr=jax_lane.raw_arrays["surface_gamma"],
            quantity="surface_gamma",
            component="pm_surface",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_unit_normal"],
            jax_arr=jax_lane.raw_arrays["surface_unit_normal"],
            quantity="surface_unit_normal",
            component="pm_surface",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["A_obj"],
            jax_arr=jax_lane.raw_arrays["A_obj"],
            quantity="pm_grid_payload",
            component="A_obj",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["b_obj"],
            jax_arr=jax_lane.raw_arrays["b_obj"],
            quantity="pm_grid_payload",
            component="b_obj",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["m_maxima"],
            jax_arr=jax_lane.raw_arrays["m_maxima"],
            quantity="pm_grid_payload",
            component="m_maxima",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["dipole_grid_xyz"],
            jax_arr=jax_lane.raw_arrays["dipole_grid_xyz"],
            quantity="pm_grid_payload",
            component="dipole_grid_xyz",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["m"],
            jax_arr=jax_lane.raw_arrays["m"],
            quantity="pm_moments",
            component=algorithm_component,
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["m_proxy"],
            jax_arr=jax_lane.raw_arrays["m_proxy"],
            quantity="pm_moments",
            component=f"{algorithm_component}_proxy",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["residual"],
            jax_arr=jax_lane.raw_arrays["residual"],
            quantity="pm_residual",
            component=algorithm_component,
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["residual_proxy"],
            jax_arr=jax_lane.raw_arrays["residual_proxy"],
            quantity="pm_proxy_residual",
            component=algorithm_component,
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_scalar(
            cpu_value=cpu.components["pm_objective"],
            jax_value=jax_lane.components["pm_objective"],
            quantity="pm_objective",
            component=algorithm_component,
        ),
        _compare_scalar(
            cpu_value=cpu.components["pm_proxy_objective"],
            jax_value=jax_lane.components["pm_proxy_objective"],
            quantity="pm_proxy_objective",
            component=algorithm_component,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["RS_history"],
            jax_arr=jax_lane.raw_arrays["RS_history"],
            quantity="pm_history",
            component=f"{algorithm_component}_RS_history",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["m_history"],
            jax_arr=jax_lane.raw_arrays["m_history"],
            quantity="pm_history",
            component=f"{algorithm_component}_m_history",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["m_proxy_history"],
            jax_arr=jax_lane.raw_arrays["m_proxy_history"],
            quantity="pm_history",
            component=f"{algorithm_component}_m_proxy_history",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["dipole_B"],
            jax_arr=jax_lane.raw_arrays["dipole_B"],
            quantity="pm_dipole_field_B",
            component="DipoleField",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["dipole_proxy_B"],
            jax_arr=jax_lane.raw_arrays["dipole_proxy_B"],
            quantity="pm_proxy_dipole_field_B",
            component="DipoleField",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["dipole_Bn"],
            jax_arr=jax_lane.raw_arrays["dipole_Bn"],
            quantity="pm_dipole_Bdotn",
            component="DipoleField",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["dipole_proxy_Bn"],
            jax_arr=jax_lane.raw_arrays["dipole_proxy_Bn"],
            quantity="pm_proxy_dipole_Bdotn",
            component="DipoleField",
            active_dof_names=cpu.active_dof_names,
        ),
    ]


def _wireframe_comparisons(
    cpu: LaneArtifact,
    jax_lane: LaneArtifact,
) -> Sequence[Mapping[str, Any]]:
    """Compare fixed-state wireframe RCLS matrices, solve output, and field."""
    comparisons = [
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_gamma"],
            jax_arr=jax_lane.raw_arrays["surface_gamma"],
            quantity="surface_gamma",
            component="wireframe_surface",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_unit_normal"],
            jax_arr=jax_lane.raw_arrays["surface_unit_normal"],
            quantity="surface_unit_normal",
            component="wireframe_surface",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["Amat"],
            jax_arr=jax_lane.raw_arrays["Amat"],
            quantity="wireframe_matrix",
            component="Amat",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["bvec"],
            jax_arr=jax_lane.raw_arrays["bvec"],
            quantity="wireframe_matrix",
            component="bvec",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_scalar(
            cpu_value=cpu.components["f_B"],
            jax_value=jax_lane.components["f_B"],
            quantity="wireframe_objective",
            component="f_B",
        ),
        _compare_scalar(
            cpu_value=cpu.components["f_R"],
            jax_value=jax_lane.components["f_R"],
            quantity="wireframe_objective",
            component="f_R",
        ),
        _compare_scalar(
            cpu_value=cpu.components["f"],
            jax_value=jax_lane.components["f"],
            quantity="wireframe_objective",
            component="f",
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["constraints_satisfied"],
            jax_arr=jax_lane.raw_arrays["constraints_satisfied"],
            quantity="wireframe_constraints",
            component="check_constraints",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["field_B"],
            jax_arr=jax_lane.raw_arrays["field_B"],
            quantity="wireframe_field_B",
            component="WireframeField",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["field_dB_by_dX"],
            jax_arr=jax_lane.raw_arrays["field_dB_by_dX"],
            quantity="wireframe_field_dB_by_dX",
            component="WireframeField",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["Bnormal"],
            jax_arr=jax_lane.raw_arrays["Bnormal"],
            quantity="wireframe_Bnormal",
            component="WireframeField",
            active_dof_names=cpu.active_dof_names,
        ),
    ]
    if (
        "constraint_matrix_shape" in cpu.raw_arrays
        and "constraint_matrix_shape" in jax_lane.raw_arrays
    ):
        comparisons.append(
            _compare_array(
                cpu_arr=cpu.raw_arrays["constraint_matrix_shape"],
                jax_arr=jax_lane.raw_arrays["constraint_matrix_shape"],
                quantity="wireframe_constraints",
                component="constraint_matrix_shape",
                active_dof_names=cpu.active_dof_names,
            )
        )
    return comparisons


def _wireframe_gsco_comparisons(
    cpu: LaneArtifact,
    jax_lane: LaneArtifact,
) -> Sequence[Mapping[str, Any]]:
    """Compare deterministic reduced GSCO fixed-state histories."""
    comparisons = [
        _compare_array(
            cpu_arr=cpu.raw_arrays["A_obj"],
            jax_arr=jax_lane.raw_arrays["A_obj"],
            quantity="wireframe_matrix",
            component="GSCO_Amat",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["b_obj"],
            jax_arr=jax_lane.raw_arrays["b_obj"],
            quantity="wireframe_matrix",
            component="GSCO_bvec",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["flags"],
            jax_arr=jax_lane.raw_arrays["flags"],
            quantity="wireframe_gsco_flags",
            component="constraint_flags",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["x"],
            jax_arr=jax_lane.raw_arrays["x"],
            quantity="wireframe_gsco_solution",
            component="final_x",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["loop_count"],
            jax_arr=jax_lane.raw_arrays["loop_count"],
            quantity="wireframe_gsco_solution",
            component="final_loop_count",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["iter_hist"],
            jax_arr=jax_lane.raw_arrays["iter_hist"],
            quantity="wireframe_gsco_history",
            component="iter_hist",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["curr_hist"],
            jax_arr=jax_lane.raw_arrays["curr_hist"],
            quantity="wireframe_gsco_history",
            component="curr_hist",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["loop_hist"],
            jax_arr=jax_lane.raw_arrays["loop_hist"],
            quantity="wireframe_gsco_history",
            component="loop_hist",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["f_B_hist"],
            jax_arr=jax_lane.raw_arrays["f_B_hist"],
            quantity="wireframe_gsco_history",
            component="f_B_hist",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["f_S_hist"],
            jax_arr=jax_lane.raw_arrays["f_S_hist"],
            quantity="wireframe_gsco_history",
            component="f_S_hist",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_array(
            cpu_arr=cpu.raw_arrays["f_hist"],
            jax_arr=jax_lane.raw_arrays["f_hist"],
            quantity="wireframe_gsco_history",
            component="f_hist",
            active_dof_names=cpu.active_dof_names,
        ),
        _compare_scalar(
            cpu_value=cpu.objective_native_subtotal,
            jax_value=jax_lane.objective_native_subtotal,
            quantity="objective_native_subtotal",
            component="GSCO",
        ),
    ]
    optional_array_comparisons = (
        (
            "free_cells",
            "wireframe_gsco_constraints",
            "free_cell_mask",
        ),
        (
            "initial_currents",
            "wireframe_gsco_constraints",
            "initial_currents",
        ),
        (
            "constraints_satisfied",
            "wireframe_gsco_constraints",
            "check_constraints",
        ),
        (
            "field_B",
            "wireframe_field_B",
            "WireframeField",
        ),
        (
            "Bnormal",
            "wireframe_Bnormal",
            "WireframeField",
        ),
    )
    for raw_key, quantity, component in optional_array_comparisons:
        if raw_key in cpu.raw_arrays and raw_key in jax_lane.raw_arrays:
            comparisons.append(
                _compare_array(
                    cpu_arr=cpu.raw_arrays[raw_key],
                    jax_arr=jax_lane.raw_arrays[raw_key],
                    quantity=quantity,
                    component=component,
                    active_dof_names=cpu.active_dof_names,
                )
            )
    return comparisons


def _with_source_example(
    comparisons: Sequence[Mapping[str, Any]],
    source_example: str,
) -> Sequence[Mapping[str, Any]]:
    return [dict(entry, source_example=source_example) for entry in comparisons]


def _boozer_fixed_state_comparisons(
    cpu: LaneArtifact,
    jax_lane: LaneArtifact,
) -> Sequence[Mapping[str, Any]]:
    """Comparisons specific to the fixed-state Boozer fixture.

    Compares:
      * surface geometry (gamma, unit normal) — same DOF state by
        construction, so byte parity is expected;
      * field B at the surface points;
      * the Boozer residual vector (no inner solve);
      * Area, Volume, and ToroidalFlux scalar labels.

    No SquaredFlux scalar, no gradient comparison — those are not part of
    the Boozer fixed-state contract for this fixture.
    """
    comparisons = []
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_gamma"],
            jax_arr=jax_lane.raw_arrays["surface_gamma"],
            quantity="surface_gamma",
            component="surface",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_unit_normal"],
            jax_arr=jax_lane.raw_arrays["surface_unit_normal"],
            quantity="surface_unit_normal",
            component="surface",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["field_B"],
            jax_arr=jax_lane.raw_arrays["field_B"],
            quantity="field_B",
            component="biot_savart",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["boozer_residual"],
            jax_arr=jax_lane.raw_arrays["boozer_residual"],
            quantity="boozer_residual",
            component="boozer",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_scalar(
            cpu_value=cpu.components["area"],
            jax_value=jax_lane.components["area"],
            quantity="area",
            component="label",
        )
    )
    comparisons.append(
        _compare_scalar(
            cpu_value=cpu.components["volume"],
            jax_value=jax_lane.components["volume"],
            quantity="volume",
            component="label",
        )
    )
    comparisons.append(
        _compare_scalar(
            cpu_value=cpu.components["toroidal_flux"],
            jax_value=jax_lane.components["toroidal_flux"],
            quantity="toroidal_flux",
            component="label",
        )
    )
    return comparisons


def _tracing_comparisons(
    cpu: LaneArtifact,
    jax_lane: LaneArtifact,
) -> Sequence[Mapping[str, Any]]:
    comparisons = []
    if "field_B" in cpu.raw_arrays and "field_B" in jax_lane.raw_arrays:
        comparisons.append(
            _compare_array(
                cpu_arr=cpu.raw_arrays["field_B"],
                jax_arr=jax_lane.raw_arrays["field_B"],
                quantity="field_B",
                component="interpolated_field",
                active_dof_names=cpu.active_dof_names,
            )
        )
    if "field_GradAbsB" in cpu.raw_arrays and "field_GradAbsB" in jax_lane.raw_arrays:
        comparisons.append(
            _compare_array(
                cpu_arr=cpu.raw_arrays["field_GradAbsB"],
                jax_arr=jax_lane.raw_arrays["field_GradAbsB"],
                quantity="field_GradAbsB",
                component="interpolated_field",
                active_dof_names=cpu.active_dof_names,
            )
        )
    if "field_modB" in cpu.raw_arrays and "field_modB" in jax_lane.raw_arrays:
        comparisons.append(
            _compare_array(
                cpu_arr=cpu.raw_arrays["field_modB"],
                jax_arr=jax_lane.raw_arrays["field_modB"],
                quantity="field_modB",
                component="interpolated_boozer_field",
                active_dof_names=cpu.active_dof_names,
            )
        )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["trajectory_endpoint"],
            jax_arr=jax_lane.raw_arrays["trajectory_endpoint"],
            quantity="trajectory_endpoint",
            component="compute_fieldlines",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["trajectory_t_final"],
            jax_arr=jax_lane.raw_arrays["trajectory_t_final"],
            quantity="trajectory_t_final",
            component="compute_fieldlines",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["trajectory_status_code"],
            jax_arr=jax_lane.raw_arrays["trajectory_status_code"],
            quantity="trajectory_status_code",
            component="compute_fieldlines",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["phi_hit_xyz"],
            jax_arr=jax_lane.raw_arrays["phi_hit_xyz"],
            quantity="phi_hit_xyz",
            component="compute_fieldlines",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["phi_hit_count"],
            jax_arr=jax_lane.raw_arrays["phi_hit_count"],
            quantity="phi_hit_count",
            component="compute_fieldlines",
            active_dof_names=cpu.active_dof_names,
        )
    )
    return comparisons


def _boozer_qa_wrappers_comparisons(
    cpu: LaneArtifact,
    jax_lane: LaneArtifact,
) -> Sequence[Mapping[str, Any]]:
    """Comparisons specific to the boozerQA fixed-solved-state scalar fixture.

    Compares:
      * surface geometry (gamma, unit normal) — same surface DOF state by
        construction (JAX side imports CPU-solved DOFs), so byte parity is
        expected at the direct_kernel bucket;
      * field B at the surface points — gates that BiotSavartJAX
        reproduces the CPU BiotSavart magnetic field at the same surface
        points;
      * Iotas scalar — degenerate cross-lane comparison (both lanes report
        the same CPU-solved iota);
      * MajorRadius scalar — exercises the pure-JAX
        ``surface_major_radius_jax_from_dofs`` against the CPU
        ``MajorRadius.J()`` oracle at the same surface DOFs;
      * NonQuasiSymmetricRatio scalar — exercises the pure-JAX
        ``_qs_ratio_pure`` against the CPU ``NonQuasiSymmetricRatio.J()``
        oracle at the same surface DOFs + auxiliary sDIM grid.
      * sum(CurveLength) scalar — exercises ``CurveLengthJAX`` over an
        independently loaded NCSX curve tree against the CPU ``CurveLength``
        wrapper sum used by the example.

    This fixture does not claim public ``BoozerSurfaceJAX`` wrapper or
    adjoint parity. No gradient comparison is included in this fixed-solved
    state fixture.
    """
    comparisons = []
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_gamma"],
            jax_arr=jax_lane.raw_arrays["surface_gamma"],
            quantity="surface_gamma",
            component="surface",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["surface_unit_normal"],
            jax_arr=jax_lane.raw_arrays["surface_unit_normal"],
            quantity="surface_unit_normal",
            component="surface",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_array(
            cpu_arr=cpu.raw_arrays["field_B"],
            jax_arr=jax_lane.raw_arrays["field_B"],
            quantity="field_B",
            component="biot_savart",
            active_dof_names=cpu.active_dof_names,
        )
    )
    comparisons.append(
        _compare_scalar(
            cpu_value=cpu.components["iota"],
            jax_value=jax_lane.components["iota"],
            quantity="iota",
            component="wrapper",
        )
    )
    comparisons.append(
        _compare_scalar(
            cpu_value=cpu.components["major_radius"],
            jax_value=jax_lane.components["major_radius"],
            quantity="major_radius",
            component="wrapper",
        )
    )
    comparisons.append(
        _compare_scalar(
            cpu_value=cpu.components["nq_symmetric_ratio"],
            jax_value=jax_lane.components["nq_symmetric_ratio"],
            quantity="nq_symmetric_ratio",
            component="wrapper",
        )
    )
    comparisons.append(
        _compare_scalar(
            cpu_value=cpu.components["sum_CurveLength"],
            jax_value=jax_lane.components["sum_CurveLength"],
            quantity="sum_CurveLength",
            component="curve_objective",
        )
    )
    return comparisons


def _run_perturbation_diagnostic(build: FixtureBuild) -> Optional[Mapping[str, Any]]:
    """Run the plan-required seed=1 Taylor central-difference sweep.

    Returns ``None`` if the fixture did not expose native subproblem
    evaluators. Otherwise applies a deterministic seed=1 random direction
    plus eps in {1e-3, 1e-4, 1e-5, 1e-6, 1e-7} to both CPU and JAX
    subproblems and records per-eps slopes.
    """
    if (
        build.cpu_native_subproblem_J is None
        or build.jax_native_subproblem_J is None
        or build.x0 is None
    ):
        return None

    import hashlib

    x0 = np.asarray(build.x0, dtype=np.float64).copy()
    grad_jax = (
        np.asarray(build.jax_lane.gradient, dtype=np.float64)
        if build.jax_lane.gradient is not None
        else np.zeros_like(x0)
    )
    grad_cpu = (
        np.asarray(build.cpu_lane.gradient, dtype=np.float64)
        if build.cpu_lane.gradient is not None
        else np.zeros_like(x0)
    )
    if grad_jax.size != x0.size or grad_cpu.size != x0.size:
        return {
            "seed": 1,
            "direction_hash": None,
            "samples": [],
            "directional_derivative_grad_jax": None,
            "directional_derivative_grad_cpu": None,
            "note": (
                "gradient size does not match active DOF basis; "
                "perturbation diagnostic skipped for this fixture."
            ),
        }

    rng = np.random.default_rng(1)
    direction = rng.uniform(size=x0.shape).astype(np.float64)
    direction_hash = hashlib.sha256(direction.tobytes()).hexdigest()

    samples = []
    for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        # Evaluate at +eps then -eps. The fixture's evaluators reset jf.x
        # each call, so the sequence is stateless across eps values aside
        # from the cache invalidation inside Optimizable.
        j_plus_cpu = build.cpu_native_subproblem_J(x0 + eps * direction)
        j_minus_cpu = build.cpu_native_subproblem_J(x0 - eps * direction)
        j_plus_jax = build.jax_native_subproblem_J(x0 + eps * direction)
        j_minus_jax = build.jax_native_subproblem_J(x0 - eps * direction)
        slope_cpu = (j_plus_cpu - j_minus_cpu) / (2.0 * eps)
        slope_jax = (j_plus_jax - j_minus_jax) / (2.0 * eps)
        samples.append(
            {
                "eps": eps,
                "cpu_central_diff": float(slope_cpu),
                "jax_central_diff": float(slope_jax),
                "abs_diff": float(abs(slope_jax - slope_cpu)),
            }
        )
    # Restore x0 on both lanes so post-diagnostic state matches pre-diag.
    build.cpu_native_subproblem_J(x0)
    build.jax_native_subproblem_J(x0)

    return {
        "seed": 1,
        "direction_hash": direction_hash,
        "samples": samples,
        "directional_derivative_grad_jax": float(np.dot(grad_jax, direction)),
        "directional_derivative_grad_cpu": float(np.dot(grad_cpu, direction)),
    }


def _evaluate_supported_fixture(
    record: FixtureRecord,
    *,
    jax_lane_name: str = LANE_JAX_CPU,
) -> FixtureResult:
    build = record.builder()
    spec = build.spec
    if jax_lane_name == LANE_JAX_GPU:
        _block_lane_artifact_outputs(build.jax_lane)

    comparisons = _with_source_example(
        _supported_comparisons(build),
        spec.source_example,
    )
    comparison_key = f"{LANE_CPU_CPP}_vs_{jax_lane_name}"
    if jax_lane_name != LANE_JAX_CPU:
        comparisons = _retarget_comparison_entries(
            comparisons,
            left_lane=LANE_CPU_CPP,
            right_lane=jax_lane_name,
        )
    failures = [
        f"{entry['quantity']}/{entry['component']}: max_abs_diff="
        f"{entry['max_abs_diff']!r} rtol={entry['tolerance_rtol']:.2e}"
        for entry in comparisons
        if entry["verdict"] == "fail"
    ]
    if failures:
        verdict = "fail"
    elif build.unsupported_components:
        verdict = "partial"
    else:
        verdict = "pass"

    # simsopt's auto-generated dof names carry a per-instance counter
    # (e.g. ``CurveXYZFourier1:xc(0)`` vs ``CurveXYZFourier5:xc(0)``).
    # Stripping the counter from the class-name prefix yields the
    # *structural* name that must match between independently constructed
    # lanes; positional equality of those structural names is what makes
    # element-wise gradient comparison well-defined.
    cpu_struct_names = tuple(
        _strip_dof_name_counter(n) for n in build.cpu_lane.active_dof_names
    )
    jax_struct_names = tuple(
        _strip_dof_name_counter(n) for n in build.jax_lane.active_dof_names
    )
    dof_basis_aligned = cpu_struct_names == jax_struct_names

    dof_contract = {
        "active_dof_names_cpu": list(build.cpu_lane.active_dof_names),
        "active_dof_names_jax": list(build.jax_lane.active_dof_names),
        "active_dof_structural_names_cpu": list(cpu_struct_names),
        "active_dof_structural_names_jax": list(jax_struct_names),
        "active_dof_basis_aligned": dof_basis_aligned,
        "active_dof_hash_cpu": build.cpu_lane.active_dof_hash,
        "active_dof_hash_jax": build.jax_lane.active_dof_hash,
        "fixed_free_mask_hash_cpu": build.cpu_lane.fixed_free_mask_hash,
        "fixed_free_mask_hash_jax": build.jax_lane.fixed_free_mask_hash,
        "fixture_input_hash": fixed_state_input_hash(spec.inputs),
    }
    native_spec_contract = {
        "native_curve_spec_hashes": list(build.jax_lane.native_curve_spec_hashes),
        "spec_count": len(build.jax_lane.native_curve_spec_hashes),
    }
    if not dof_basis_aligned:
        failures.append(
            "active_dof_structural_names mismatch between CPU and JAX lanes; "
            "cross-lane gradient comparison requires a documented basis "
            "mapping that is not present in this fixture."
        )
        verdict = "fail"

    perturbation = _run_perturbation_diagnostic(build) if dof_basis_aligned else None

    return FixtureResult(
        fixture_id=spec.fixture_id,
        source_example=spec.source_example,
        classification=spec.classification,
        classification_reason=spec.classification_reason,
        fixture_inputs=dict(spec.inputs),
        dof_contract=dof_contract,
        native_spec_contract=native_spec_contract,
        lanes={
            LANE_CPU_CPP: _lane_to_jsonable(build.cpu_lane),
            jax_lane_name: _lane_to_jsonable(
                build.jax_lane,
                lane_name=jax_lane_name,
                gpu_proven=False,
            ),
        },
        comparisons={comparison_key: list(comparisons)},
        unsupported_components=list(build.unsupported_components),
        mixed_lane_diagnostics=[],
        perturbation_diagnostics=perturbation,
        verdict=verdict,
        passed=(verdict in ("pass", "partial")),
        failures=failures,
    )


def _evaluate_unsupported_fixture(
    record: FixtureRecord,
    error_message: str,
) -> FixtureResult:
    """Build a fail-closed ``unsupported`` record for a runtime gap.

    A ``FixtureNotSupportedError`` is always a contract gap (upstream
    deserialization, missing artifacts, missing native spec), never a
    parity failure. The verdict is therefore always ``"unsupported"``
    regardless of whether the spec was *declared* supported up front.
    The classification string on the record (``SUPPORTED``,
    ``SUPPORT_GATE``, ``UNSUPPORTED_NATIVE_JAX``, ...) preserves the
    plan's declared intent for the reader.
    """
    spec = record.spec
    return FixtureResult(
        fixture_id=spec.fixture_id,
        source_example=spec.source_example,
        classification=spec.classification,
        classification_reason=spec.classification_reason,
        fixture_inputs=dict(spec.inputs),
        dof_contract={},
        native_spec_contract={},
        lanes={},
        comparisons={"cpu_cpp_vs_jax_cpu": []},
        unsupported_components=[],
        mixed_lane_diagnostics=[],
        perturbation_diagnostics=None,
        verdict="unsupported",
        passed=False,
        failures=[],
        error=error_message,
    )


def _filter_result_for_lanes(
    result: FixtureResult,
    lanes: Sequence[str],
) -> FixtureResult:
    selected = set(lanes)
    filtered_lanes = {
        lane_name: lane_payload
        for lane_name, lane_payload in result.lanes.items()
        if lane_name in selected
    }
    comparisons = dict(result.comparisons)
    parity_pairs = {
        f"{LANE_CPU_CPP}_vs_{LANE_JAX_CPU}": {LANE_CPU_CPP, LANE_JAX_CPU},
        f"{LANE_CPU_CPP}_vs_{LANE_JAX_GPU}": {LANE_CPU_CPP, LANE_JAX_GPU},
    }
    has_parity_pair = False
    for comparison_key, required_lanes in parity_pairs.items():
        if comparison_key not in comparisons:
            continue
        comparison_has_pair = required_lanes <= set(filtered_lanes)
        has_parity_pair = has_parity_pair or comparison_has_pair
        if not comparison_has_pair:
            comparisons[comparison_key] = []
    if not has_parity_pair:
        if result.verdict in ("pass", "partial"):
            return replace(
                result,
                lanes=filtered_lanes,
                comparisons=comparisons,
                verdict="fail",
                passed=False,
                failures=(
                    *result.failures,
                    "selected lanes omit the required parity pair for verdict",
                ),
            )
    return replace(result, lanes=filtered_lanes, comparisons=comparisons)


# ---------------------------------------------------------------------------
# Run-level metadata.


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        text=True,
    ).strip()


def _git_branch() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(REPO_ROOT),
        text=True,
    ).strip()


def _dirty_tree_summary() -> Mapping[str, Any]:
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT),
        text=True,
    )
    lines = [line for line in porcelain.splitlines() if line.strip()]
    return {
        "available": True,
        "is_dirty": bool(lines),
        "entry_count": len(lines),
        "entries": lines,
    }


def _python_version() -> str:
    return sys.version.split()[0]


def _jax_version() -> str:
    return jax.__version__


def _jaxlib_version() -> str:
    return jaxlib.__version__


def _simsopt_version() -> str:
    import simsopt

    return getattr(simsopt, "__version__", "editable")


def _jax_devices_metadata() -> Sequence[Mapping[str, Any]]:
    return [
        {
            "platform": d.platform,
            "device_kind": getattr(d, "device_kind", ""),
            "id": getattr(d, "id", None),
            "process_index": getattr(d, "process_index", None),
            "platform_version": getattr(
                getattr(d, "client", None),
                "platform_version",
                None,
            ),
        }
        for d in jax.devices()
    ]


def _jax_platform_versions() -> Sequence[str]:
    versions = tuple(
        str(device["platform_version"])
        for device in _jax_devices_metadata()
        if device["platform_version"]
    )
    return tuple(sorted(set(versions)))


def _is_cuda_device(device) -> bool:
    return getattr(device, "platform", None) in CUDA_DEVICE_PLATFORMS


def _assert_no_gpu_devices() -> None:
    bad = [d for d in jax.devices() if _is_cuda_device(d)]
    if bad:
        raise RuntimeError(
            "Non-banana parity harness is CPU-only; refusing to run with "
            f"non-CPU JAX devices visible: {bad!r}"
        )


def _assert_gpu_runtime_contract() -> None:
    required_env = {
        "SIMSOPT_BACKEND_MODE": "jax_gpu_parity",
        "SIMSOPT_JAX_PLATFORM": "cuda",
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_X64": "1",
        "SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM": "cuda",
    }
    mismatches = {
        name: {"expected": expected, "actual": os.environ.get(name)}
        for name, expected in required_env.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "jax_gpu lane requires the explicit CUDA parity environment; "
            f"mismatches={mismatches!r}"
        )
    backend = jax.default_backend()
    cuda_devices = [device for device in jax.devices() if _is_cuda_device(device)]
    if backend not in CUDA_DEVICE_PLATFORMS or not cuda_devices:
        raise RuntimeError(
            "jax_gpu lane requires an active CUDA JAX backend; "
            f"default_backend={backend!r}, devices={jax.devices()!r}"
        )


def _gpu_transfer_guard_probe() -> Mapping[str, Any]:
    seed = jax.device_put(np.asarray([1.0], dtype=np.float64))
    with jax.transfer_guard("disallow"):
        value = jax.jit(lambda x: x + 1.0)(seed)
        value.block_until_ready()
    host_value = np.asarray(jax.device_get(value), dtype=np.float64)
    return {
        "status": "pass",
        "mode": "disallow",
        "explicit_device_get_value": host_value.tolist(),
    }


def _query_nvidia_compute_capabilities() -> Sequence[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = tuple(line.strip() for line in result.stdout.splitlines() if line.strip())
    if not values:
        raise RuntimeError("nvidia-smi did not report any GPU compute capability.")
    return values


def _assert_x64() -> None:
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "Non-banana parity harness requires JAX_ENABLE_X64=1; jax x64 "
            "is currently disabled."
        )


def _gpu_runtime_metadata() -> Mapping[str, Any]:
    smi_facts = query_nvidia_smi_facts()
    if smi_facts is None:
        raise RuntimeError("jax_gpu lane requires nvidia-smi GPU provenance.")
    compute_capabilities = _query_nvidia_compute_capabilities()
    smi_gpus = smi_facts["nvidia_smi_gpus"]
    platform_versions = _jax_platform_versions()
    return {
        "jax_cuda_wheel_runtime_line": (
            platform_versions[0]
            if platform_versions
            else f"jaxlib {jaxlib.__version__}"
        ),
        "cuda_runtime_version_visible_to_jax": smi_facts.get("cuda_runtime_version"),
        "nvidia_driver_version": smi_facts.get("cuda_driver_version"),
        "device_name": smi_gpus[0]["name"],
        "compute_capability": compute_capabilities[0],
        "compute_capabilities": list(compute_capabilities),
        "transfer_guard": "disallow",
        "xla_cuda": current_xla_cuda_metadata(),
        "nvidia_smi": smi_facts,
        "transfer_guard_probe": _gpu_transfer_guard_probe(),
    }


def build_run_metadata(
    *,
    git_sha_override: Optional[str],
    lanes: Sequence[str] = (LANE_CPU_CPP, LANE_JAX_CPU),
) -> Mapping[str, Any]:
    _assert_x64()
    lane_set = set(lanes)
    if LANE_JAX_GPU in lane_set:
        _assert_gpu_runtime_contract()
        gpu_runtime = _gpu_runtime_metadata()
    else:
        _assert_no_gpu_devices()
        gpu_runtime = None
    jax_devices = list(_jax_devices_metadata())
    return {
        "git_head": git_sha_override or _git_head(),
        "git_branch": _git_branch(),
        "dirty_tree_summary": _dirty_tree_summary(),
        "jax_platform": jax_devices[0]["platform"],
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_backend": jax_devices[0]["platform"],
        "jax_devices": jax_devices,
        "requested_jax_platform": _REQUESTED_JAX_PLATFORM,
        "gpu_runtime": gpu_runtime,
        "python_version": _python_version(),
        "jax_version": _jax_version(),
        "jaxlib_version": _jaxlib_version(),
        "simsopt_version": _simsopt_version(),
        "platform": platform.platform(),
        "host_machine": platform.machine(),
        "executable": sys.executable,
        "version_probe_command": (
            'conda run -n jax python -c "import jax, jaxlib; '
            'print(jax.__version__, jaxlib.__version__)"'
        ),
        "lane_schema": {
            "cpu_cpp": {"required": True, "artifact_kind": "cpu_oracle"},
            "jax_cpu": {"required": True, "artifact_kind": "jax_cpu_candidate"},
            "jax_gpu": {
                "required": False,
                "artifact_kind": "jax_gpu_followup",
                "status": "runtime_required",
                "required_environment": {
                    "SIMSOPT_BACKEND_MODE": "jax_gpu_parity",
                    "SIMSOPT_JAX_PLATFORM": "cuda",
                    "JAX_PLATFORMS": "cuda",
                    "JAX_ENABLE_X64": "1",
                    "SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM": "cuda",
                },
                "required_provenance_fields": (
                    "jax_version",
                    "jaxlib_version",
                    "jax_cuda_wheel_runtime_line",
                    "cuda_runtime_version_visible_to_jax",
                    "nvidia_driver_version",
                    "device_name",
                    "compute_capability",
                    "transfer_guard",
                ),
                "must_reuse_fixture_input_hash": True,
                "cannot_upgrade_cpu_unsupported": True,
                "separate_artifact_required": True,
                "first_proof_lane": "jax_gpu_parity",
                "disallowed_first_proof_lane": "jax_gpu_fast",
            },
        },
    }


def _load_baseline_payload(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"Baseline artifact schema mismatch: {payload.get('schema_version')!r}"
        )
    return payload


def _baseline_fixture_by_id(
    baseline_payload: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    fixtures = baseline_payload.get("fixtures")
    if not isinstance(fixtures, list):
        raise RuntimeError("Baseline artifact is missing a fixtures list.")
    by_id = {}
    for entry in fixtures:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Baseline fixture entry is not an object: {entry!r}")
        fixture_id = entry.get("fixture_id")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise RuntimeError(
                f"Baseline fixture entry is missing fixture_id: {entry!r}"
            )
        by_id[fixture_id] = entry
    return by_id


def _right_value_for_comparison(
    entry: Mapping[str, Any],
    *,
    lane_name: str,
) -> Any:
    lane_key = f"{lane_name}_value"
    if lane_key in entry:
        return entry[lane_key]
    if lane_name == LANE_JAX_CPU:
        return entry.get("jax_cpu_value")
    if lane_name == LANE_JAX_GPU:
        return entry.get("jax_gpu_value")
    return entry.get("right_value")


def _jax_cpu_vs_jax_gpu_comparisons(
    *,
    baseline_entry: Mapping[str, Any],
    gpu_result: FixtureResult,
) -> Sequence[Mapping[str, Any]]:
    baseline_hash = (
        baseline_entry.get("dof_contract", {}).get("fixture_input_hash")
        if isinstance(baseline_entry.get("dof_contract"), dict)
        else None
    )
    gpu_hash = gpu_result.dof_contract.get("fixture_input_hash")
    if baseline_hash != gpu_hash:
        raise RuntimeError(
            f"{gpu_result.fixture_id}: baseline fixture_input_hash {baseline_hash!r} "
            f"does not match GPU fixture_input_hash {gpu_hash!r}."
        )

    baseline_comparisons = baseline_entry.get("comparisons", {}).get(
        f"{LANE_CPU_CPP}_vs_{LANE_JAX_CPU}",
        [],
    )
    gpu_comparisons = gpu_result.comparisons.get(
        f"{LANE_CPU_CPP}_vs_{LANE_JAX_GPU}",
        [],
    )
    if len(baseline_comparisons) != len(gpu_comparisons):
        raise RuntimeError(
            f"{gpu_result.fixture_id}: baseline comparison count "
            f"{len(baseline_comparisons)} does not match GPU comparison count "
            f"{len(gpu_comparisons)}."
        )

    comparisons = []
    for baseline_comparison, gpu_comparison in zip(
        baseline_comparisons,
        gpu_comparisons,
    ):
        baseline_key = (
            baseline_comparison.get("quantity"),
            baseline_comparison.get("component"),
        )
        gpu_key = (
            gpu_comparison.get("quantity"),
            gpu_comparison.get("component"),
        )
        if baseline_key != gpu_key:
            raise RuntimeError(
                f"{gpu_result.fixture_id}: comparison mismatch "
                f"{baseline_key!r} != {gpu_key!r}."
            )
        comparisons.append(
            _compare_json_values(
                left_value=_right_value_for_comparison(
                    baseline_comparison,
                    lane_name=LANE_JAX_CPU,
                ),
                right_value=_right_value_for_comparison(
                    gpu_comparison,
                    lane_name=LANE_JAX_GPU,
                ),
                quantity=str(gpu_comparison["quantity"]),
                component=str(gpu_comparison["component"]),
                left_lane=LANE_JAX_CPU,
                right_lane=LANE_JAX_GPU,
            )
        )
    return tuple(comparisons)


# ---------------------------------------------------------------------------
# Run loop.


def run_fixtures(
    fixture_ids_to_run: Sequence[str],
    *,
    git_sha_override: Optional[str] = None,
    lanes: Sequence[str] = ("cpu_cpp", "jax_cpu"),
    baseline_json: Optional[Path] = None,
) -> Mapping[str, Any]:
    lane_set = tuple(lanes)
    unsupported_lanes = tuple(lane for lane in lane_set if lane not in SUPPORTED_LANES)
    if unsupported_lanes:
        raise RuntimeError(f"Unsupported parity lane(s): {unsupported_lanes!r}.")
    if LANE_JAX_CPU in lane_set and LANE_JAX_GPU in lane_set:
        raise RuntimeError(
            "Select jax_gpu in a separate CUDA process and pass --baseline-json "
            "for jax_cpu_vs_jax_gpu comparisons."
        )
    if LANE_JAX_GPU in lane_set and baseline_json is None:
        raise RuntimeError("jax_gpu lane requires --baseline-json.")

    baseline_by_id = None
    if baseline_json is not None:
        baseline_by_id = _baseline_fixture_by_id(_load_baseline_payload(baseline_json))

    metadata = build_run_metadata(git_sha_override=git_sha_override, lanes=lane_set)
    metadata = dict(metadata)
    metadata["selected_lanes"] = list(lane_set)
    if baseline_json is not None:
        metadata["baseline_json"] = str(baseline_json)
    fixtures = []
    jax_lane_name = LANE_JAX_GPU if LANE_JAX_GPU in lane_set else LANE_JAX_CPU
    for fid in fixture_ids_to_run:
        record = get_fixture(fid)
        if record.spec.classification == SUPPORTED:
            try:
                result = _evaluate_supported_fixture(
                    record,
                    jax_lane_name=jax_lane_name,
                )
                if baseline_by_id is not None and LANE_JAX_GPU in lane_set:
                    baseline_entry = baseline_by_id.get(result.fixture_id)
                    if baseline_entry is None:
                        raise RuntimeError(
                            f"{result.fixture_id}: missing from baseline artifact."
                        )
                    jax_gpu_comparisons = _jax_cpu_vs_jax_gpu_comparisons(
                        baseline_entry=baseline_entry,
                        gpu_result=result,
                    )
                    result = replace(
                        result,
                        comparisons={
                            **dict(result.comparisons),
                            f"{LANE_JAX_CPU}_vs_{LANE_JAX_GPU}": list(
                                jax_gpu_comparisons
                            ),
                        },
                        failures=(
                            *result.failures,
                            *(
                                f"{entry['quantity']}/{entry['component']} "
                                f"{LANE_JAX_CPU}_vs_{LANE_JAX_GPU}: "
                                f"max_abs_diff={entry['max_abs_diff']!r} "
                                f"rtol={entry['tolerance_rtol']:.2e}"
                                for entry in jax_gpu_comparisons
                                if entry["verdict"] == "fail"
                            ),
                        ),
                    )
                    if result.failures:
                        result = replace(
                            result,
                            verdict="fail",
                            passed=False,
                        )
                    gpu_lane = dict(result.lanes[LANE_JAX_GPU])
                    gpu_lane["gpu_readiness"] = dict(
                        gpu_readiness_metadata(proven=result.passed)
                    )
                    result = replace(
                        result,
                        lanes={
                            **dict(result.lanes),
                            LANE_JAX_GPU: gpu_lane,
                        },
                    )
            except FixtureNotSupportedError as exc:
                result = _evaluate_unsupported_fixture(record, str(exc))
            except Exception as exc:  # report failure without aborting batch
                result = FixtureResult(
                    fixture_id=record.spec.fixture_id,
                    source_example=record.spec.source_example,
                    classification=record.spec.classification,
                    classification_reason=record.spec.classification_reason,
                    fixture_inputs=dict(record.spec.inputs),
                    dof_contract={},
                    native_spec_contract={},
                    lanes={},
                    comparisons={"cpu_cpp_vs_jax_cpu": []},
                    unsupported_components=[],
                    mixed_lane_diagnostics=[],
                    perturbation_diagnostics=None,
                    verdict="fail",
                    passed=False,
                    failures=[f"unexpected error: {type(exc).__name__}: {exc}"],
                    error=f"{type(exc).__name__}: {exc}",
                )
        else:
            try:
                record.builder()
                result = _evaluate_unsupported_fixture(
                    record, "unsupported-classification builder did not raise"
                )
            except FixtureNotSupportedError as exc:
                result = _evaluate_unsupported_fixture(record, str(exc))
            except Exception as exc:  # noqa: BLE001 — match SUPPORTED branch
                result = FixtureResult(
                    fixture_id=record.spec.fixture_id,
                    source_example=record.spec.source_example,
                    classification=record.spec.classification,
                    classification_reason=record.spec.classification_reason,
                    fixture_inputs=dict(record.spec.inputs),
                    dof_contract={},
                    native_spec_contract={},
                    lanes={},
                    comparisons={"cpu_cpp_vs_jax_cpu": []},
                    unsupported_components=[],
                    mixed_lane_diagnostics=[],
                    perturbation_diagnostics=None,
                    verdict="fail",
                    passed=False,
                    failures=[
                        f"unexpected error in unsupported-classification builder: "
                        f"{type(exc).__name__}: {exc}"
                    ],
                    error=f"{type(exc).__name__}: {exc}",
                )

        fixtures.append(_filter_result_for_lanes(result, lane_set).__dict__)

    return {
        "schema_version": SCHEMA_VERSION,
        "harness": "non_banana_example_cpp_jax_cpu_parity",
        "metadata": dict(metadata),
        "fixtures": fixtures,
    }


# ---------------------------------------------------------------------------
# CLI.


def _resolve_fixture_selection(arg: str) -> Sequence[str]:
    if arg == "all-supported":
        return supported_fixture_ids()
    if arg == "all":
        return fixture_ids()
    return tuple(part.strip() for part in arg.split(",") if part.strip())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Run the non-banana example CPU C++/JAX parity harness."),
    )
    parser.add_argument(
        "--fixtures",
        default="all-supported",
        help=(
            "Comma-separated fixture IDs, or one of "
            "'all-supported' (default) or 'all' (includes unsupported "
            "classification records)."
        ),
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help="Override the recorded git HEAD (defaults to git rev-parse HEAD).",
    )
    parser.add_argument(
        "--dirty-policy",
        choices=("record",),
        default="record",
        help=(
            "Dirty-tree policy. Only 'record' is supported in this harness; "
            "the dirty-tree summary is always written to the JSON artifact."
        ),
    )
    parser.add_argument(
        "--lanes",
        default="cpu_cpp,jax_cpu",
        help=(
            "Comma-separated lane selector. CPU runs use cpu_cpp,jax_cpu. "
            "CUDA follow-up runs use cpu_cpp,jax_gpu with --baseline-json."
        ),
    )
    parser.add_argument(
        "--baseline-json",
        default=None,
        help=(
            "CPU artifact JSON used to compare jax_cpu against jax_gpu in "
            "CUDA follow-up runs."
        ),
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "Optional path for the JSON artifact. If omitted, the artifact "
            "is printed to stdout."
        ),
    )
    args = parser.parse_args(argv)

    fixture_selection = _resolve_fixture_selection(args.fixtures)
    payload = run_fixtures(
        fixture_selection,
        git_sha_override=args.git_sha,
        lanes=tuple(part.strip() for part in args.lanes.split(",") if part.strip()),
        baseline_json=Path(args.baseline_json) if args.baseline_json else None,
    )
    text = json.dumps(payload, indent=2, sort_keys=False, default=_json_default)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n")
        print(f"Wrote parity artifact: {out_path}")
    else:
        print(text)

    has_failure = any(
        fixture.get("verdict") == "fail" for fixture in payload["fixtures"]
    )
    return 1 if has_failure else 0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    sys.exit(main())
