"""Non-banana example CPU C++/JAX parity benchmark (Phase 0–2 implementation).

Implements the harness described by
``docs/non_banana_example_cpp_jax_cpu_parity_plan_2026-05-12.md``:

* Phase 0 — baseline metadata + fixture-contract gates (x64, no-GPU,
  native-spec).
* Phase 1 — P0 ``minimal_stage2_flux_length_gap`` fixed-state parity
  (SquaredFlux / SquaredFluxJAX, B, B·n, surface geometry, gradient,
  deterministic perturbation diagnostics).
* Phase 2 — P1 ``cws_saved_local_flux_nfp{2,3}`` fixed-state parity
  using the saved BiotSavart artifacts.

Remaining fixtures (``full_stage2_composite``, ``planar_stage2_composite``,
``position_orientation_flux_support_gate``, ``boozer_surface_basic``,
``boozer_qa_wrappers``, ``finite_beta_target_flux``,
``finitebuild_multifilament_support_gate``, ``qfm_surface``) are registered
as explicit ``unsupported`` or ``support_gate`` records. Follow-up phases add
native comparisons; this harness does not silently loosen unsupported gates.

CPU only — the benchmark refuses to run if any CUDA device is visible to
JAX after configuration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

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

# ``jax`` is configured at import time so subsequent imports see the
# canonical CPU + x64 runtime, even if the parent shell exported a
# GPU-capable platform list.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import jax  # noqa: E402  (after env-var setup)

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

from benchmarks.non_banana_example_parity_fixtures import (  # noqa: E402
    FixtureBuild,
    FixtureNotSupportedError,
    FixtureRecord,
    LaneArtifact,
    SCHEMA_VERSION,
    SUPPORTED,
    fixture_ids,
    get_fixture,
    supported_fixture_ids,
)
from benchmarks.validation_ladder_contract import (  # noqa: E402
    parity_ladder_tolerances,
)


# ---------------------------------------------------------------------------
# Tolerance mapping (mirrors plan §"Default Tolerance Mapping").


_TOLERANCE_BUCKETS = {
    "field_B": "direct_kernel",
    "surface_gamma": "direct_kernel",
    "surface_unit_normal": "direct_kernel",
    "Bdotn": "direct_kernel",
    "objective_native_subtotal": "ls_wrapper_gradient",
    "SquaredFlux": "ls_wrapper_gradient",
    "SquaredFluxJAX": "ls_wrapper_gradient",
    "gradient": "ls_wrapper_gradient",
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
            "tolerance_bucket": bucket,
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
    argmax_index = np.unravel_index(argmax_flat, cpu.shape) if cpu.ndim else (0,)
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
        "tolerance_bucket": bucket,
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
        "tolerance_bucket": bucket,
        "tolerance_rtol": rtol,
        "tolerance_atol": atol,
        "max_abs_diff": float(abs_diff),
        "max_rel_diff": float(rel),
        "argmax_index": None,
        "argmax_dof_name": None,
        "verdict": "pass" if passed else "fail",
    }


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


def _lane_to_jsonable(lane: LaneArtifact) -> Mapping[str, Any]:
    return {
        "lane": lane.lane,
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
        "timing": dict(lane.timing),
    }


def _supported_comparisons(build: FixtureBuild) -> Sequence[Mapping[str, Any]]:
    """Compute comparison entries for every native-supported quantity."""
    cpu = build.cpu_lane
    jax_lane = build.jax_lane
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


def _evaluate_supported_fixture(record: FixtureRecord) -> FixtureResult:
    build = record.builder()
    spec = build.spec

    comparisons = _supported_comparisons(build)
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
            "cpu_cpp": _lane_to_jsonable(build.cpu_lane),
            "jax_cpu": _lane_to_jsonable(build.jax_lane),
        },
        comparisons={"cpu_cpp_vs_jax_cpu": list(comparisons)},
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


def _simsopt_version() -> str:
    import simsopt

    return getattr(simsopt, "__version__", "editable")


def _jax_devices_metadata() -> Sequence[Mapping[str, Any]]:
    return [
        {"platform": d.platform, "device_kind": getattr(d, "device_kind", "")}
        for d in jax.devices()
    ]


def _assert_no_gpu_devices() -> None:
    bad = [d for d in jax.devices() if d.platform != "cpu"]
    if bad:
        raise RuntimeError(
            "Non-banana parity harness is CPU-only; refusing to run with "
            f"non-CPU JAX devices visible: {bad!r}"
        )


def _assert_x64() -> None:
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "Non-banana parity harness requires JAX_ENABLE_X64=1; jax x64 "
            "is currently disabled."
        )


def build_run_metadata(*, git_sha_override: Optional[str]) -> Mapping[str, Any]:
    _assert_no_gpu_devices()
    _assert_x64()
    jax_devices = list(_jax_devices_metadata())
    return {
        "git_head": git_sha_override or _git_head(),
        "git_branch": _git_branch(),
        "dirty_tree_summary": _dirty_tree_summary(),
        "jax_platform": jax_devices[0]["platform"],
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "jax_backend": jax_devices[0]["platform"],
        "jax_devices": jax_devices,
        "python_version": _python_version(),
        "jax_version": _jax_version(),
        "simsopt_version": _simsopt_version(),
        "platform": platform.platform(),
        "host_machine": platform.machine(),
        "executable": sys.executable,
    }


# ---------------------------------------------------------------------------
# Run loop.


def run_fixtures(
    fixture_ids_to_run: Sequence[str],
    *,
    git_sha_override: Optional[str] = None,
) -> Mapping[str, Any]:
    metadata = build_run_metadata(git_sha_override=git_sha_override)
    fixtures = []
    for fid in fixture_ids_to_run:
        record = get_fixture(fid)
        if record.spec.classification == SUPPORTED:
            try:
                result = _evaluate_supported_fixture(record)
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

        fixtures.append(result.__dict__)

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
        description=(
            "Run the non-banana example CPU C++/JAX parity harness (CPU-only)."
        ),
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
