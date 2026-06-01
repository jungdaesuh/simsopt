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

CPU is the default execution mode. Separate ``jax_gpu`` and ``jax_mps`` modes
are available only when the process is launched with the matching accelerator
environment and a CPU baseline artifact is supplied for JAX CPU vs accelerator
comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

LANE_CPU_CPP = "cpu_cpp"
LANE_JAX_CPU = "jax_cpu"
LANE_JAX_GPU = "jax_gpu"
LANE_JAX_MPS = "jax_mps"
FOLLOWUP_JAX_LANES = frozenset((LANE_JAX_GPU, LANE_JAX_MPS))
SUPPORTED_LANES = frozenset((LANE_CPU_CPP, LANE_JAX_CPU, *FOLLOWUP_JAX_LANES))

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
    if requested not in {"cpu", "cuda", "mps"}:
        raise RuntimeError(
            "SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM must be "
            f"'cpu', 'cuda', or 'mps'; got {requested!r}."
        )
    selected_lanes = _preimport_selected_lanes(sys.argv[1:])
    if requested == "cuda" and LANE_JAX_GPU not in selected_lanes:
        raise RuntimeError(
            "SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM='cuda' requires lane "
            f"{LANE_JAX_GPU!r}; selected_lanes={selected_lanes!r}."
        )
    if requested == "mps" and LANE_JAX_MPS not in selected_lanes:
        raise RuntimeError(
            "SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM='mps' requires lane "
            f"{LANE_JAX_MPS!r}; selected_lanes={selected_lanes!r}."
        )
    return requested


_REQUESTED_JAX_PLATFORM = _requested_jax_platform()


def _requested_jax_enable_x64(requested_platform: str) -> str:
    if requested_platform == "mps":
        return "0"
    if (
        requested_platform == "cpu"
        and os.environ.get("SIMSOPT_BACKEND_MODE") == "jax_cpu_float32_smoke"
    ):
        return "0"
    return "1"


_REQUESTED_JAX_ENABLE_X64 = _requested_jax_enable_x64(_REQUESTED_JAX_PLATFORM)

# ``jax`` is configured at import time so subsequent imports see exactly one
# parity runtime. CPU remains the default, even if a parent shell exported a
# broader platform list. Accelerator follow-ups require both
# SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM=<platform> and the matching lane before
# this module is imported.
os.environ["JAX_PLATFORMS"] = _REQUESTED_JAX_PLATFORM
os.environ["JAX_ENABLE_X64"] = _REQUESTED_JAX_ENABLE_X64

import jax  # noqa: E402  (after env-var setup)
import jaxlib  # noqa: E402  (after env-var setup)

jax.config.update("jax_platforms", _REQUESTED_JAX_PLATFORM)
jax.config.update("jax_enable_x64", _REQUESTED_JAX_ENABLE_X64 == "1")

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
from benchmarks.run_code_benchmark_common import (  # noqa: E402
    artifact_host_array,
)
from benchmarks.validation_ladder_common import (  # noqa: E402
    current_xla_cuda_metadata,
    get_git_sha,
    query_nvidia_smi_facts,
)
from benchmarks.validation_ladder_contract import (  # noqa: E402
    FLOAT32_SMOKE_TOLERANCE_TIER,
    comparison_failure_gates_verdict,
    comparison_failure_is_diagnostic,
    quantity_parity_tolerance,
    quantity_uses_gradient_tolerance,
)
from simsopt.backend import (  # noqa: E402
    get_backend_policy,
    get_tolerance_tier,
    get_transfer_guard,
)
from simsopt.backend.dtypes import runtime_host_dtype  # noqa: E402


CUDA_DEVICE_PLATFORMS = frozenset(("cuda", "gpu"))
MPS_DEVICE_PLATFORMS = frozenset(("mps",))


JAX_MPS_FLOAT32_ONLY_AUTHORITY = {
    "source": "tillahoffmann/jax-mps README",
    "url": "https://github.com/tillahoffmann/jax-mps#footnotes",
    "constraint": "MLX only supports float32.",
    "effect": "jax_mps_smoke is a float32 smoke lane, not float64 production parity.",
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
    return quantity_parity_tolerance(quantity, runtime_tier=get_tolerance_tier())


def _runtime_host_np_dtype() -> np.dtype:
    return np.dtype(runtime_host_dtype())


def _runtime_policy_metadata() -> Mapping[str, Any]:
    policy = get_backend_policy()
    return {
        "backend_mode": policy.mode,
        "runtime_dtype": policy.runtime_dtype,
        "host_dtype": policy.host_dtype,
        "tolerance_tier": policy.tolerance_tier,
        "parity_mode": policy.parity_mode,
    }


def _lane_numeric_contract(lane_name: str) -> Mapping[str, str]:
    policy = get_backend_policy()
    if lane_name == LANE_CPU_CPP:
        source_precision = "float64_oracle"
        device_platform = "host_cpu_cpp"
    else:
        source_precision = policy.runtime_dtype
        device_platform = policy.jax_platform
    return {
        "source_precision": source_precision,
        "artifact_dtype": _runtime_host_np_dtype().name,
        "runtime_dtype": policy.runtime_dtype,
        "device_platform": device_platform,
    }


def _finite_array_summary(value: Any) -> Mapping[str, Any]:
    array = artifact_host_array(value)
    if not np.issubdtype(array.dtype, np.number):
        return {"checked": False, "all_finite": True, "nonfinite_count": 0}
    finite = np.isfinite(array)
    return {
        "checked": True,
        "all_finite": bool(np.all(finite)),
        "nonfinite_count": int(array.size - int(np.count_nonzero(finite))),
    }


def _lane_finite_summary(lane: LaneArtifact) -> Mapping[str, Any]:
    fields: dict[str, Mapping[str, Any]] = {}
    scalar_fields = {
        "objective_total": lane.objective_total,
        "objective_native_subtotal": lane.objective_native_subtotal,
        "gradient_norm": lane.gradient_norm,
        "field_B_max": lane.field_B_max,
        "field_B_mean": lane.field_B_mean,
        "Bdotn_max": lane.Bdotn_max,
        "Bdotn_mean": lane.Bdotn_mean,
        **dict(lane.components),
    }
    for name, value in scalar_fields.items():
        if value is not None:
            fields[name] = _finite_array_summary(value)
    for name, value in lane.raw_arrays.items():
        fields[f"raw_arrays.{name}"] = _finite_array_summary(value)
    if lane.gradient is not None:
        fields["gradient"] = _finite_array_summary(lane.gradient)

    checked_fields = {
        name: summary for name, summary in fields.items() if summary["checked"]
    }
    nonfinite_fields = {
        name: summary
        for name, summary in checked_fields.items()
        if not summary["all_finite"]
    }
    return {
        "all_finite": not nonfinite_fields,
        "checked_field_count": len(checked_fields),
        "nonfinite_field_count": len(nonfinite_fields),
        "nonfinite_fields": nonfinite_fields,
    }


def _lane_finite_failures(lane_name: str, lane: LaneArtifact) -> Sequence[str]:
    summary = _lane_finite_summary(lane)
    if summary["all_finite"]:
        return ()
    return (
        f"{lane_name} has non-finite fixed-state fields: "
        f"{sorted(summary['nonfinite_fields'])}",
    )


ARTIFACT_MATERIALIZATION_TRANSFER_GUARD = "device_to_host_allow"
HOST_MATERIALIZATION_PURPOSE = "json_artifact"


def _runtime_host_float_array(value: Any) -> np.ndarray:
    array = artifact_host_array(value)
    if array.dtype.kind == "f":
        return np.asarray(array, dtype=_runtime_host_np_dtype())
    return array


def _host_float_scalar(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(artifact_host_array(value, dtype=_runtime_host_np_dtype()))


def _host_hash_array(value: Any) -> str:
    array = np.ascontiguousarray(_runtime_host_float_array(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


_METADATA_RAW_ARRAY_KEYS_BY_FIXTURE_KIND: Mapping[str, Mapping[str, Optional[str]]] = {
    "biot_savart_squared_flux": {
        "surface_point": "surface_gamma",
        "unit_normal": "surface_unit_normal",
        "field_B": "field_B",
        "Bdotn": "Bdotn_target_subtracted",
    },
    "qfm": {
        "surface_point": "surface_gamma",
        "unit_normal": "surface_unit_normal",
        "field_B": "field_B",
        "Bdotn": "Bdotn",
    },
    "boozer_surface_fixed_state": {
        "surface_point": "surface_gamma",
        "unit_normal": "surface_unit_normal",
        "field_B": "field_B",
        "Bdotn": None,
    },
    "boozer_qa_wrappers_solved_state": {
        "surface_point": "surface_gamma",
        "unit_normal": "surface_unit_normal",
        "field_B": "field_B",
        "Bdotn": None,
    },
    "surface_scalar": {
        "surface_point": "surface_gamma",
        "unit_normal": "surface_unit_normal",
        "field_B": None,
        "Bdotn": None,
    },
    "pm": {
        "surface_point": "surface_gamma",
        "unit_normal": "surface_unit_normal",
        "field_B": "dipole_B",
        "Bdotn": "dipole_Bn",
    },
    "pm_relax_and_split": {
        "surface_point": "surface_gamma",
        "unit_normal": "surface_unit_normal",
        "field_B": "dipole_B",
        "Bdotn": "dipole_Bn",
    },
    "wireframe": {
        "surface_point": None,
        "unit_normal": None,
        "field_B": None,
        "Bdotn": None,
    },
    "wireframe_gsco": {
        "surface_point": None,
        "unit_normal": None,
        "field_B": None,
        "Bdotn": None,
    },
    "tracing": {
        "surface_point": None,
        "unit_normal": None,
        "field_B": None,
        "Bdotn": None,
    },
    "strain": {
        "surface_point": None,
        "unit_normal": None,
        "field_B": None,
        "Bdotn": None,
    },
    "coil_force_energy": {
        "surface_point": None,
        "unit_normal": None,
        "field_B": None,
        "Bdotn": None,
    },
}


def _metadata_raw_array(
    raw_arrays: Mapping[str, np.ndarray],
    *,
    fixture_kind: str,
    metadata_name: str,
) -> Optional[np.ndarray]:
    source_keys = _METADATA_RAW_ARRAY_KEYS_BY_FIXTURE_KIND[fixture_kind]
    raw_array_key = source_keys[metadata_name]
    if raw_array_key is None:
        return None
    return raw_arrays[raw_array_key]


def _lane_with_runtime_host_dtype(
    lane: LaneArtifact,
    *,
    fixture_kind: str,
) -> LaneArtifact:
    raw_arrays = {
        name: _runtime_host_float_array(value)
        for name, value in lane.raw_arrays.items()
    }
    gradient = None
    if lane.gradient is not None:
        gradient = _runtime_host_float_array(lane.gradient)
    components = {
        name: _host_float_scalar(value) for name, value in lane.components.items()
    }
    surface_gamma = _metadata_raw_array(
        raw_arrays,
        fixture_kind=fixture_kind,
        metadata_name="surface_point",
    )
    surface_unit_normal = _metadata_raw_array(
        raw_arrays,
        fixture_kind=fixture_kind,
        metadata_name="unit_normal",
    )
    field_B = _metadata_raw_array(
        raw_arrays,
        fixture_kind=fixture_kind,
        metadata_name="field_B",
    )
    bdotn = _metadata_raw_array(
        raw_arrays,
        fixture_kind=fixture_kind,
        metadata_name="Bdotn",
    )
    lane_updates = {
        "objective_total": _host_float_scalar(lane.objective_total),
        "objective_native_subtotal": _host_float_scalar(lane.objective_native_subtotal),
        "components": components,
        "gradient": gradient,
        "gradient_norm": (
            None if gradient is None else float(np.linalg.norm(np.asarray(gradient)))
        ),
        "field_B_max": _host_float_scalar(lane.field_B_max),
        "field_B_mean": _host_float_scalar(lane.field_B_mean),
        "Bdotn_max": _host_float_scalar(lane.Bdotn_max),
        "Bdotn_mean": _host_float_scalar(lane.Bdotn_mean),
        "raw_arrays": raw_arrays,
    }
    if surface_gamma is not None:
        lane_updates["surface_point_hash"] = _host_hash_array(surface_gamma)
    if surface_unit_normal is not None:
        lane_updates["unit_normal_hash"] = _host_hash_array(surface_unit_normal)
    if field_B is not None:
        lane_updates["field_B_hash"] = _host_hash_array(field_B)
        lane_updates["field_B_max"] = float(np.max(np.abs(field_B)))
        lane_updates["field_B_mean"] = float(np.mean(np.abs(field_B)))
    if bdotn is not None:
        lane_updates["Bdotn_array_hash"] = _host_hash_array(bdotn)
        lane_updates["Bdotn_max"] = float(np.max(np.abs(bdotn)))
        lane_updates["Bdotn_mean"] = float(np.mean(np.abs(bdotn)))
    return replace(lane, **lane_updates)


def _build_with_runtime_host_dtype(build: FixtureBuild) -> FixtureBuild:
    return replace(
        build,
        cpu_lane=_lane_with_runtime_host_dtype(
            build.cpu_lane,
            fixture_kind=build.spec.fixture_kind,
        ),
        jax_lane=_lane_with_runtime_host_dtype(
            build.jax_lane,
            fixture_kind=build.spec.fixture_kind,
        ),
    )


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
    cpu = _runtime_host_float_array(cpu_arr)
    jax_a = _runtime_host_float_array(jax_arr)

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
    if bucket == FLOAT32_SMOKE_TOLERANCE_TIER and quantity_uses_gradient_tolerance(
        quantity
    ):
        entry["diagnostic_only"] = True
        entry["diagnostic_reason"] = "float32_smoke_gradient_not_production_parity_gate"
    return entry


def _compare_derived_normal_projection(
    *,
    cpu_field: np.ndarray,
    jax_field: np.ndarray,
    cpu_normal: np.ndarray,
    jax_normal: np.ndarray,
    cpu_projection: np.ndarray,
    jax_projection: np.ndarray,
    quantity: str,
    component: str,
    active_dof_names: Sequence[str],
) -> Mapping[str, Any]:
    """Compare a host-side B.normal projection using source-array budgets."""
    entry = dict(
        _compare_array(
            cpu_arr=cpu_projection,
            jax_arr=jax_projection,
            quantity=quantity,
            component=component,
            active_dof_names=active_dof_names,
        )
    )
    cpu_field_arr = _runtime_host_float_array(cpu_field)
    jax_field_arr = _runtime_host_float_array(jax_field)
    cpu_normal_arr = _runtime_host_float_array(cpu_normal)
    jax_normal_arr = _runtime_host_float_array(jax_normal)
    cpu_projection_arr = _runtime_host_float_array(cpu_projection)
    jax_projection_arr = _runtime_host_float_array(jax_projection)
    expected_projection_shape = cpu_field_arr.shape[:-1]
    source_shapes_match = (
        cpu_field_arr.shape == jax_field_arr.shape
        and cpu_field_arr.shape == cpu_normal_arr.shape
        and cpu_normal_arr.shape == jax_normal_arr.shape
        and cpu_projection_arr.shape == jax_projection_arr.shape
        and cpu_projection_arr.shape == expected_projection_shape
    )
    if not source_shapes_match:
        entry["verdict"] = "fail"
        entry["failure_reason"] = (
            "normal projection source shape mismatch: "
            f"cpu_field={cpu_field_arr.shape}, jax_field={jax_field_arr.shape}, "
            f"cpu_normal={cpu_normal_arr.shape}, jax_normal={jax_normal_arr.shape}, "
            f"cpu_projection={cpu_projection_arr.shape}, "
            f"jax_projection={jax_projection_arr.shape}"
        )
        return entry

    _, field_rtol, field_atol = _tolerance_for("wireframe_field_B")
    _, normal_rtol, normal_atol = _tolerance_for("surface_unit_normal")
    field_budget = field_atol + field_rtol * np.abs(cpu_field_arr)
    normal_budget = normal_atol + normal_rtol * np.abs(cpu_normal_arr)
    projection_budget = np.sum(
        np.abs(cpu_normal_arr) * field_budget
        + np.abs(cpu_field_arr) * normal_budget
        + field_budget * normal_budget,
        axis=-1,
    )
    abs_diff = np.abs(jax_projection_arr - cpu_projection_arr)
    excess = abs_diff - projection_budget
    projection_argmax_flat = int(excess.argmax()) if excess.size else 0
    if excess.size:
        projection_argmax_index = np.unravel_index(
            projection_argmax_flat,
            excess.shape,
        )
    else:
        projection_argmax_index = ()
    entry["derived_from_quantities"] = [
        "wireframe_field_B",
        "surface_unit_normal",
    ]
    entry["derived_projection_budget_max"] = (
        float(projection_budget.max()) if projection_budget.size else 0.0
    )
    entry["derived_projection_budget_at_argmax"] = (
        float(projection_budget[projection_argmax_index])
        if projection_budget.size
        else 0.0
    )
    entry["derived_projection_max_excess"] = (
        float(excess[projection_argmax_index]) if excess.size else 0.0
    )
    entry["derived_projection_argmax_index"] = [
        int(i) for i in np.atleast_1d(projection_argmax_index).tolist()
    ]
    entry["verdict"] = "pass" if bool(np.all(abs_diff <= projection_budget)) else "fail"
    return entry


def _compare_scalar(
    *,
    cpu_value: float,
    jax_value: float,
    quantity: str,
    component: str,
) -> Mapping[str, Any]:
    bucket, rtol, atol = _tolerance_for(quantity)
    cpu_value = float(np.asarray(cpu_value, dtype=_runtime_host_np_dtype()))
    jax_value = float(np.asarray(jax_value, dtype=_runtime_host_np_dtype()))
    abs_diff = abs(jax_value - cpu_value)
    denom = atol + rtol * abs(cpu_value)
    rel = abs_diff / (abs(cpu_value) + atol)
    passed = abs_diff <= denom
    entry = {
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
    if bucket == FLOAT32_SMOKE_TOLERANCE_TIER and quantity_uses_gradient_tolerance(
        quantity
    ):
        entry["diagnostic_only"] = True
        entry["diagnostic_reason"] = "float32_smoke_gradient_not_production_parity_gate"
    return entry


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

    left = _runtime_host_float_array(left_value)
    right = _runtime_host_float_array(right_value)
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
        "numeric_contract": dict(_lane_numeric_contract(emitted_lane_name)),
        "finite_summary": dict(_lane_finite_summary(lane)),
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
            jax_value=jax_lane.components["LpCurveForce"],
            quantity="LpCurveForce",
            component="force_objective",
        ),
        _compare_scalar(
            cpu_value=cpu.components["LpCurveForce_independent_oracle"],
            jax_value=jax_lane.components["LpCurveForce"],
            quantity="LpCurveForce_independent_oracle",
            component="force_objective",
        ),
        _compare_scalar(
            cpu_value=cpu.components["B2Energy"],
            jax_value=jax_lane.components["B2Energy"],
            quantity="B2Energy",
            component="energy_objective",
        ),
        _compare_scalar(
            cpu_value=cpu.components["B2Energy_independent_oracle"],
            jax_value=jax_lane.components["B2Energy"],
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
        _compare_derived_normal_projection(
            cpu_field=cpu.raw_arrays["field_B"],
            jax_field=jax_lane.raw_arrays["field_B"],
            cpu_normal=cpu.raw_arrays["surface_unit_normal"],
            jax_normal=jax_lane.raw_arrays["surface_unit_normal"],
            cpu_projection=cpu.raw_arrays["Bnormal"],
            jax_projection=jax_lane.raw_arrays["Bnormal"],
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
            "surface_unit_normal",
            "surface_unit_normal",
            "wireframe_surface",
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
            if raw_key == "Bnormal":
                comparisons.append(
                    _compare_derived_normal_projection(
                        cpu_field=cpu.raw_arrays["field_B"],
                        jax_field=jax_lane.raw_arrays["field_B"],
                        cpu_normal=cpu.raw_arrays["surface_unit_normal"],
                        jax_normal=jax_lane.raw_arrays["surface_unit_normal"],
                        cpu_projection=cpu.raw_arrays[raw_key],
                        jax_projection=jax_lane.raw_arrays[raw_key],
                        quantity=quantity,
                        component=component,
                        active_dof_names=cpu.active_dof_names,
                    )
                )
            else:
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

    host_dtype = _runtime_host_np_dtype()
    x0 = np.asarray(build.x0, dtype=host_dtype).copy()
    grad_jax = (
        np.asarray(build.jax_lane.gradient, dtype=host_dtype)
        if build.jax_lane.gradient is not None
        else np.zeros_like(x0)
    )
    grad_cpu = (
        np.asarray(build.cpu_lane.gradient, dtype=host_dtype)
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
    direction = rng.uniform(size=x0.shape).astype(host_dtype)
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
    build = _build_with_runtime_host_dtype(record.builder())
    spec = build.spec
    if jax_lane_name in FOLLOWUP_JAX_LANES:
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
        if comparison_failure_gates_verdict(entry)
    ]
    if failures:
        verdict = "fail"
    elif any(comparison_failure_is_diagnostic(entry) for entry in comparisons):
        verdict = "partial"
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
    finite_failures = (
        *_lane_finite_failures(LANE_CPU_CPP, build.cpu_lane),
        *_lane_finite_failures(jax_lane_name, build.jax_lane),
    )
    if finite_failures:
        failures.extend(finite_failures)
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
        f"{LANE_CPU_CPP}_vs_{LANE_JAX_MPS}": {LANE_CPU_CPP, LANE_JAX_MPS},
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
    return get_git_sha()


def _has_git_metadata() -> bool:
    return (REPO_ROOT / ".git").exists()


def _git_branch() -> str:
    if not _has_git_metadata():
        return "source-archive"
    return subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=str(REPO_ROOT),
        text=True,
    ).strip()


def _dirty_tree_summary() -> Mapping[str, Any]:
    if not _has_git_metadata() and "SIMSOPT_GIT_STATUS_SHORT" in os.environ:
        porcelain = os.environ["SIMSOPT_GIT_STATUS_SHORT"]
    else:
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


def _is_mps_device(device) -> bool:
    return getattr(device, "platform", None) in MPS_DEVICE_PLATFORMS


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


def _assert_mps_runtime_contract() -> None:
    required_env = {
        "SIMSOPT_BACKEND_MODE": "jax_mps_smoke",
        "SIMSOPT_JAX_PLATFORM": "mps",
        "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
        "JAX_PLATFORMS": "mps",
        "JAX_ENABLE_X64": "0",
        "SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM": "mps",
    }
    mismatches = {
        name: {"expected": expected, "actual": os.environ.get(name)}
        for name, expected in required_env.items()
        if os.environ.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "jax_mps lane requires the explicit MPS smoke environment; "
            f"mismatches={mismatches!r}"
        )
    backend = jax.default_backend()
    mps_devices = [device for device in jax.devices() if _is_mps_device(device)]
    if backend not in MPS_DEVICE_PLATFORMS or not mps_devices:
        raise RuntimeError(
            "jax_mps lane requires an active MPS JAX backend; "
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


def _assert_cpu_runtime_contract() -> None:
    policy = get_backend_policy()
    if policy.mode == "jax_cpu_float32_smoke":
        required_env = {
            "SIMSOPT_BACKEND_MODE": "jax_cpu_float32_smoke",
            "SIMSOPT_JAX_PLATFORM": "cpu",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "0",
            "SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM": "cpu",
        }
        mismatches = {
            name: {"expected": expected, "actual": os.environ.get(name)}
            for name, expected in required_env.items()
            if os.environ.get(name) != expected
        }
        if mismatches:
            raise RuntimeError(
                "jax_cpu_float32_smoke lane requires the explicit CPU float32 "
                f"smoke environment; mismatches={mismatches!r}"
            )
        backend = jax.default_backend()
        if backend != "cpu" or any(_is_cuda_device(device) for device in jax.devices()):
            raise RuntimeError(
                "jax_cpu_float32_smoke lane requires an active CPU JAX backend; "
                f"default_backend={backend!r}, devices={jax.devices()!r}"
            )
        return

    _assert_x64()
    _assert_no_gpu_devices()


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


def _mps_transfer_guard_probe() -> Mapping[str, Any]:
    seed = jax.device_put(np.asarray([1.0], dtype=np.float32))
    with jax.transfer_guard("disallow"):
        value = jax.jit(lambda x: x + np.float32(1.0))(seed)
        value.block_until_ready()
    host_value = np.asarray(jax.device_get(value), dtype=np.float32)
    return {
        "status": "pass",
        "mode": "disallow",
        "explicit_device_get_value": host_value.tolist(),
    }


def _mps_runtime_metadata() -> Mapping[str, Any]:
    mps_devices = [
        device for device in _jax_devices_metadata() if device["platform"] == "mps"
    ]
    platform_versions = _jax_platform_versions()
    return {
        **_runtime_policy_metadata(),
        "jax_mps_version": importlib_metadata.version("jax-mps"),
        "device_name": mps_devices[0]["device_kind"],
        "platform_versions": list(platform_versions),
        "transfer_guard": get_transfer_guard(),
        "compute_transfer_guard": get_transfer_guard(),
        "transfer_guard_probe": _mps_transfer_guard_probe(),
        "float64_production_lane_exclusion": dict(JAX_MPS_FLOAT32_ONLY_AUTHORITY),
    }


def _mps_readiness_metadata(*, proven: bool = False) -> Mapping[str, Any]:
    return {
        **_runtime_policy_metadata(),
        "mps_ready": bool(proven),
        "mps_proven": bool(proven),
        "first_proof_lane": "jax_mps_smoke",
        "float64_production_lane_exclusion": dict(JAX_MPS_FLOAT32_ONLY_AUTHORITY),
    }


def build_run_metadata(
    *,
    git_sha_override: Optional[str],
    lanes: Sequence[str] = (LANE_CPU_CPP, LANE_JAX_CPU),
) -> Mapping[str, Any]:
    lane_set = set(lanes)
    if LANE_JAX_MPS in lane_set:
        _assert_mps_runtime_contract()
        gpu_runtime = None
        mps_runtime = _mps_runtime_metadata()
    elif LANE_JAX_GPU in lane_set:
        _assert_x64()
        _assert_gpu_runtime_contract()
        gpu_runtime = _gpu_runtime_metadata()
        mps_runtime = None
    else:
        _assert_cpu_runtime_contract()
        gpu_runtime = None
        mps_runtime = None
    jax_devices = list(_jax_devices_metadata())
    policy_metadata = _runtime_policy_metadata()
    return {
        "git_head": git_sha_override or _git_head(),
        "git_branch": _git_branch(),
        "dirty_tree_summary": _dirty_tree_summary(),
        "jax_platform": jax_devices[0]["platform"],
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_backend": jax_devices[0]["platform"],
        "jax_devices": jax_devices,
        "requested_jax_platform": _REQUESTED_JAX_PLATFORM,
        **policy_metadata,
        "runtime_host_dtype": _runtime_host_np_dtype().name,
        "runtime_tolerance_tier": policy_metadata["tolerance_tier"],
        "artifact_materialization_transfer_guard": (
            ARTIFACT_MATERIALIZATION_TRANSFER_GUARD
        ),
        "host_materialization_purpose": HOST_MATERIALIZATION_PURPOSE,
        "gpu_runtime": gpu_runtime,
        "mps_runtime": mps_runtime,
        "python_version": _python_version(),
        "jax_version": _jax_version(),
        "jaxlib_version": _jaxlib_version(),
        "simsopt_version": _simsopt_version(),
        "platform": platform.platform(),
        "host_machine": platform.machine(),
        "executable": sys.executable,
        "version_probe_command": (
            f"{shlex.quote(sys.executable)} -c "
            "'import jax, jaxlib; print(jax.__version__, jaxlib.__version__)'"
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
            "jax_mps": {
                "required": False,
                "artifact_kind": "jax_mps_followup",
                "status": "runtime_required",
                "required_environment": {
                    "SIMSOPT_BACKEND_MODE": "jax_mps_smoke",
                    "SIMSOPT_JAX_PLATFORM": "mps",
                    "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
                    "JAX_PLATFORMS": "mps",
                    "JAX_ENABLE_X64": "0",
                    "SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM": "mps",
                },
                "required_provenance_fields": (
                    "backend_mode",
                    "jax_version",
                    "jaxlib_version",
                    "jax_mps_version",
                    "device_name",
                    "runtime_dtype",
                    "host_dtype",
                    "tolerance_tier",
                    "transfer_guard",
                    "compute_transfer_guard",
                ),
                "must_reuse_fixture_input_hash": True,
                "cannot_upgrade_cpu_unsupported": True,
                "separate_artifact_required": True,
                "first_proof_lane": "jax_mps_smoke",
                "tolerance_tier": FLOAT32_SMOKE_TOLERANCE_TIER,
                "gradient_diagnostic_only": True,
                "production_parity": False,
                "float64_production_lane_exclusion": dict(
                    JAX_MPS_FLOAT32_ONLY_AUTHORITY
                ),
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
    return entry[lane_key]


def _jax_cpu_vs_followup_comparisons(
    *,
    baseline_entry: Mapping[str, Any],
    followup_result: FixtureResult,
    followup_lane_name: str,
) -> Sequence[Mapping[str, Any]]:
    if followup_lane_name not in FOLLOWUP_JAX_LANES:
        raise RuntimeError(f"Unsupported JAX follow-up lane: {followup_lane_name!r}.")
    baseline_hash = (
        baseline_entry.get("dof_contract", {}).get("fixture_input_hash")
        if isinstance(baseline_entry.get("dof_contract"), dict)
        else None
    )
    followup_hash = followup_result.dof_contract.get("fixture_input_hash")
    if baseline_hash != followup_hash:
        raise RuntimeError(
            f"{followup_result.fixture_id}: baseline fixture_input_hash "
            f"{baseline_hash!r} does not match {followup_lane_name} "
            f"fixture_input_hash {followup_hash!r}."
        )

    baseline_comparisons = baseline_entry.get("comparisons", {}).get(
        f"{LANE_CPU_CPP}_vs_{LANE_JAX_CPU}",
        [],
    )
    followup_comparisons = followup_result.comparisons.get(
        f"{LANE_CPU_CPP}_vs_{followup_lane_name}",
        [],
    )
    if len(baseline_comparisons) != len(followup_comparisons):
        raise RuntimeError(
            f"{followup_result.fixture_id}: baseline comparison count "
            f"{len(baseline_comparisons)} does not match {followup_lane_name} "
            f"comparison count {len(followup_comparisons)}."
        )

    baseline_by_key = {
        (comparison.get("quantity"), comparison.get("component")): comparison
        for comparison in baseline_comparisons
    }
    followup_by_key = {
        (comparison.get("quantity"), comparison.get("component")): comparison
        for comparison in followup_comparisons
    }
    comparisons = []
    for baseline_comparison, followup_comparison in zip(
        baseline_comparisons,
        followup_comparisons,
    ):
        baseline_key = (
            baseline_comparison.get("quantity"),
            baseline_comparison.get("component"),
        )
        followup_key = (
            followup_comparison.get("quantity"),
            followup_comparison.get("component"),
        )
        if baseline_key != followup_key:
            raise RuntimeError(
                f"{followup_result.fixture_id}: comparison mismatch "
                f"{baseline_key!r} != {followup_key!r}."
            )
        if baseline_key == ("wireframe_Bnormal", "WireframeField"):
            field_key = ("wireframe_field_B", "WireframeField")
            normal_key = ("surface_unit_normal", "wireframe_surface")
            if field_key not in baseline_by_key or field_key not in followup_by_key:
                raise RuntimeError(
                    f"{followup_result.fixture_id}: wireframe_Bnormal follow-up "
                    "comparison requires wireframe_field_B source comparisons."
                )
            if normal_key not in baseline_by_key or normal_key not in followup_by_key:
                raise RuntimeError(
                    f"{followup_result.fixture_id}: wireframe_Bnormal follow-up "
                    "comparison requires surface_unit_normal source comparisons."
                )
            derived_entry = _compare_derived_normal_projection(
                cpu_field=_right_value_for_comparison(
                    baseline_by_key[field_key],
                    lane_name=LANE_JAX_CPU,
                ),
                jax_field=_right_value_for_comparison(
                    followup_by_key[field_key],
                    lane_name=followup_lane_name,
                ),
                cpu_normal=_right_value_for_comparison(
                    baseline_by_key[normal_key],
                    lane_name=LANE_JAX_CPU,
                ),
                jax_normal=_right_value_for_comparison(
                    followup_by_key[normal_key],
                    lane_name=followup_lane_name,
                ),
                cpu_projection=_right_value_for_comparison(
                    baseline_comparison,
                    lane_name=LANE_JAX_CPU,
                ),
                jax_projection=_right_value_for_comparison(
                    followup_comparison,
                    lane_name=followup_lane_name,
                ),
                quantity=str(followup_comparison["quantity"]),
                component=str(followup_comparison["component"]),
                active_dof_names=(),
            )
            comparisons.append(
                _retarget_comparison_entry(
                    derived_entry,
                    left_lane=LANE_JAX_CPU,
                    right_lane=followup_lane_name,
                )
            )
            continue
        comparisons.append(
            _compare_json_values(
                left_value=_right_value_for_comparison(
                    baseline_comparison,
                    lane_name=LANE_JAX_CPU,
                ),
                right_value=_right_value_for_comparison(
                    followup_comparison,
                    lane_name=followup_lane_name,
                ),
                quantity=str(followup_comparison["quantity"]),
                component=str(followup_comparison["component"]),
                left_lane=LANE_JAX_CPU,
                right_lane=followup_lane_name,
            )
        )
    return tuple(comparisons)


def _jax_cpu_vs_jax_gpu_comparisons(
    *,
    baseline_entry: Mapping[str, Any],
    gpu_result: FixtureResult,
) -> Sequence[Mapping[str, Any]]:
    return _jax_cpu_vs_followup_comparisons(
        baseline_entry=baseline_entry,
        followup_result=gpu_result,
        followup_lane_name=LANE_JAX_GPU,
    )


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
    followup_lanes = tuple(lane for lane in lane_set if lane in FOLLOWUP_JAX_LANES)
    if len(followup_lanes) > 1:
        raise RuntimeError(
            "Select one accelerator follow-up lane per process and pass "
            "--baseline-json for JAX CPU vs accelerator comparisons."
        )
    followup_lane_name = followup_lanes[0] if followup_lanes else None
    if LANE_JAX_CPU in lane_set and followup_lane_name is not None:
        raise RuntimeError(
            f"Select {followup_lane_name} in a separate accelerator process and "
            "pass --baseline-json for JAX CPU vs accelerator comparisons."
        )
    if followup_lane_name is not None and baseline_json is None:
        raise RuntimeError(f"{followup_lane_name} lane requires --baseline-json.")

    baseline_by_id = None
    if baseline_json is not None:
        baseline_by_id = _baseline_fixture_by_id(_load_baseline_payload(baseline_json))

    metadata = build_run_metadata(git_sha_override=git_sha_override, lanes=lane_set)
    metadata = dict(metadata)
    metadata["selected_lanes"] = list(lane_set)
    if baseline_json is not None:
        metadata["baseline_json"] = str(baseline_json)
    fixtures = []
    jax_lane_name = followup_lane_name or LANE_JAX_CPU
    for fid in fixture_ids_to_run:
        record = get_fixture(fid)
        if record.spec.classification == SUPPORTED:
            try:
                result = _evaluate_supported_fixture(
                    record,
                    jax_lane_name=jax_lane_name,
                )
                if baseline_by_id is not None and followup_lane_name is not None:
                    baseline_entry = baseline_by_id.get(result.fixture_id)
                    if baseline_entry is None:
                        raise RuntimeError(
                            f"{result.fixture_id}: missing from baseline artifact."
                        )
                    jax_followup_comparisons = _jax_cpu_vs_followup_comparisons(
                        baseline_entry=baseline_entry,
                        followup_result=result,
                        followup_lane_name=followup_lane_name,
                    )
                    followup_comparison_key = f"{LANE_JAX_CPU}_vs_{followup_lane_name}"
                    result = replace(
                        result,
                        comparisons={
                            **dict(result.comparisons),
                            followup_comparison_key: list(jax_followup_comparisons),
                        },
                        failures=(
                            *result.failures,
                            *(
                                f"{entry['quantity']}/{entry['component']} "
                                f"{followup_comparison_key}: "
                                f"max_abs_diff={entry['max_abs_diff']!r} "
                                f"rtol={entry['tolerance_rtol']:.2e}"
                                for entry in jax_followup_comparisons
                                if comparison_failure_gates_verdict(entry)
                            ),
                        ),
                    )
                    if result.failures:
                        result = replace(
                            result,
                            verdict="fail",
                            passed=False,
                        )
                    elif any(
                        comparison_failure_is_diagnostic(entry)
                        for entry in jax_followup_comparisons
                    ):
                        result = replace(result, verdict="partial", passed=True)
                    followup_lane = dict(result.lanes[followup_lane_name])
                    if followup_lane_name == LANE_JAX_GPU:
                        followup_lane["gpu_readiness"] = dict(
                            gpu_readiness_metadata(proven=result.passed)
                        )
                    else:
                        followup_lane["mps_readiness"] = dict(
                            _mps_readiness_metadata(proven=result.passed)
                        )
                    result = replace(
                        result,
                        lanes={
                            **dict(result.lanes),
                            followup_lane_name: followup_lane,
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
                    comparisons={f"{LANE_CPU_CPP}_vs_{jax_lane_name}": []},
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
                    comparisons={f"{LANE_CPU_CPP}_vs_{jax_lane_name}": []},
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
            "Accelerator follow-up runs use cpu_cpp,jax_gpu or cpu_cpp,jax_mps "
            "with --baseline-json."
        ),
    )
    parser.add_argument(
        "--baseline-json",
        default=None,
        help=(
            "CPU artifact JSON used to compare jax_cpu against the selected "
            "accelerator follow-up lane."
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
