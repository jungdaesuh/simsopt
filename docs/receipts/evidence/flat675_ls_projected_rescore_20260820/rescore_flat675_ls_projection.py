#!/usr/bin/env python
"""Re-score F3 B37 before and after frozen-coil C++ LS projection.

Diagnostic only. This driver evaluates the canonical native eight-term outer
objective at two candidates that share the same F3 coils and vessel: the
published flat-675 endpoint and its archived 661-DOF LS-projected surface.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Final

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "16")

PRODUCTION_ROOT: Final[Path] = Path(__file__).resolve().parents[4]
INSTRUMENT_ROOT: Final[Path] = Path(
    "/home/jungdaesuh/code/columbia/simsopt-genuine675-fairbar"
)
NATIVE_BUILD_SOURCE_ROOT: Final[Path] = Path(
    "/home/jungdaesuh/code/columbia/simopt-jax-mixed-exploratory-62a262b09c"
)
V0C_RUNTIME_ROOT: Final[Path] = Path(
    "/home/jungdaesuh/simsopt_mixed_artifacts/v0c_62a262b09c_20260715T2150Z"
)
NATIVE_EXTENSION_ATTESTATION: Final[Path] = (
    V0C_RUNTIME_ROOT / "native_extension_sha256.txt"
)
EXPECTED_NATIVE_BUILD_EXTENSION: Final[Path] = (
    NATIVE_BUILD_SOURCE_ROOT
    / "build"
    / "cp311-cp311-linux_x86_64"
    / "simsoptpp.cpython-311-x86_64-linux-gnu.so"
)
EXPECTED_INSTRUMENT_COMMIT: Final[str] = "1c23f6c5f8964c74cc60f63d81b7f93f2db852f3"
EXPECTED_NATIVE_BUILD_SOURCE_COMMIT: Final[str] = (
    "62a262b09c27c8dc755cd844ccc452b7a9ceb74d"
)
EXPECTED_NATIVE_CPP_TREE: Final[str] = "50a56cdb44c294fdf86a1226ff353abfcbad59b4"
EXPECTED_NATIVE_EXTENSION_SHA256: Final[str] = (
    "0f9e597d1d4d40047c332f610c39a2871ed8def8bbcf8fdf2349bc78eca2d698"
)
EXPECTED_NATIVE_EXTENSION_ATTESTATION_SHA256: Final[str] = (
    "e627e45a55192a33d87b25f612b6933e8a9d77b1495ee5967fab60e38b8cf39b"
)
EXPECTED_F3_PRODUCTION_COMMIT: Final[str] = "580217e0cadc427f93721f12aa0f4d8e8915b4e9"
EXPECTED_BUNDLE_MANIFEST_SHA256: Final[str] = (
    "84febc05d195d84c0802205b2b4c85ea1fa38faa7ff856efca7c12d980647c0c"
)
EXPECTED_F3_LANE_SHA256: Final[str] = (
    "bde32ab9987d4f2116cf7c7410753c83a9d74ca7836031c25db0e12603155d64"
)
EXPECTED_F3_MANIFEST_SHA256: Final[str] = (
    "9a55d45f13fbcdad104b6cc7fefdb6b4c9c2a906f593dcab757993f38d29e788"
)
EXPECTED_PURPOSE_SHA256: Final[str] = (
    "2fbd444d0e59fb19222f1c2cb42791288ac93e365c71a080dc7b5838acbbbb07"
)
EXPECTED_RECONSTRUCT_SHA256: Final[str] = (
    "1b68f7f6ba54784982e5dd0aa669537cd1ea159c1172b8776b8b2bbcc0fd7846"
)
F3_POINT_NAME: Final[str] = "f3_b37_pair2_l1_endpoint"
OUTER_OBJECTIVE_TERM_COUNT: Final[int] = 8

sys.path.insert(0, str(INSTRUMENT_ROOT))
sys.path.insert(0, str(INSTRUMENT_ROOT / "src"))

from benchmarks.validation_ladder_common import (  # noqa: E402
    apply_requested_platform,
    bootstrap_local_simsopt,
    require_requested_platform_runtime,
    require_x64_runtime,
)

apply_requested_platform("cpu")

import jax  # noqa: E402

require_x64_runtime(jax, context="F3 LS-projected outer-objective rescore")
require_requested_platform_runtime(
    jax,
    requested_platform="cpu",
    context="F3 LS-projected outer-objective rescore",
)
bootstrap_local_simsopt()

import numpy as np  # noqa: E402
import simsoptpp  # noqa: E402
from benchmarks.fixed_state_genuine_675_input_manifest import (  # noqa: E402
    validate_frozen_genuine_675_input_bundle,
)
from simsopt_jax.runtime.fixed_state_genuine_675_parity_diagnostic import (  # noqa: E402
    FixedStateGenuine675Provenance,
)
from simsopt_jax.runtime.fixed_state_genuine_675_input_manifest import (  # noqa: E402
    FrozenGenuine675InputManifest,
)
from simsopt_jax.runtime.single_stage_fullspace_675 import (  # noqa: E402
    Fullspace675Candidate,
    Fullspace675ComparisonLane,
)
from simsopt_jax_adapters.geo.fixed_state_genuine_675_native import (  # noqa: E402
    Fullspace675FixedStateNativeEvaluator,
    Fullspace675FixedStateNativeMaterial,
)
from simsopt_jax_adapters.geo.single_stage_fullspace_675 import (  # noqa: E402
    Fullspace675NativeBoozerMaterial,
)


@dataclass(frozen=True, slots=True)
class Arguments:
    bundle_manifest: Path
    f3_lane: Path
    f3_manifest: Path
    purpose_json: Path
    reconstruct_json: Path
    output_json: Path


def _parse_args() -> Arguments:
    evidence_root = PRODUCTION_ROOT / "docs" / "receipts" / "evidence"
    reconstruct_root = evidence_root / "boozer_unnest_newton_reconstruct_20260820"
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--bundle-manifest",
        type=Path,
        default=(
            Path.home()
            / "simsopt_mixed_artifacts"
            / "genuine675-r3-input-1c23f6c5-20260721-r1"
            / "manifest.json"
        ),
    )
    parser.add_argument(
        "--f3-lane",
        type=Path,
        default=(
            Path.home()
            / "simsopt_mixed_artifacts"
            / "flat675_fused_campaign"
            / "20260819T163816Z-pairs-b37-2751085"
            / "pair2-l1"
            / "lane.json"
        ),
    )
    parser.add_argument(
        "--f3-manifest",
        type=Path,
        default=(
            evidence_root
            / "flat675_fused_campaign"
            / "20260819T163816Z-pairs-b37-2751085"
            / "manifest.json"
        ),
    )
    parser.add_argument(
        "--purpose-json",
        type=Path,
        default=reconstruct_root / "boozer_ls_exact_purpose.json",
    )
    parser.add_argument(
        "--reconstruct-json",
        type=Path,
        default=reconstruct_root / "newton_reconstruct_flat675.json",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parsed = parser.parse_args()
    return Arguments(
        bundle_manifest=parsed.bundle_manifest,
        f3_lane=parsed.f3_lane,
        f3_manifest=parsed.f3_manifest,
        purpose_json=parsed.purpose_json,
        reconstruct_json=parsed.reconstruct_json,
        output_json=parsed.output_json,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str) -> str:
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"{path} has SHA-256 {observed}; expected frozen bytes {expected}."
        )
    return observed


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *arguments),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _instrument_identity() -> dict[str, object]:
    commit = _git(INSTRUMENT_ROOT, "rev-parse", "HEAD")
    if commit != EXPECTED_INSTRUMENT_COMMIT:
        raise ValueError(
            f"instrument is at {commit}; expected {EXPECTED_INSTRUMENT_COMMIT}."
        )
    dirty_paths = tuple(
        line
        for line in _git(INSTRUMENT_ROOT, "status", "--porcelain").splitlines()
        if line
    )
    if dirty_paths:
        raise ValueError("the pinned native instrument tree is dirty.")
    return {"root": str(INSTRUMENT_ROOT), "commit": commit, "dirty_paths": []}


def _native_source_pairing(simsoptpp_path: Path) -> dict[str, object]:
    runtime_extension_sha256 = _sha256_file(simsoptpp_path)
    if runtime_extension_sha256 != EXPECTED_NATIVE_EXTENSION_SHA256:
        raise ValueError("loaded simsoptpp differs from the attested v0c extension.")
    _require_sha256(
        NATIVE_EXTENSION_ATTESTATION,
        EXPECTED_NATIVE_EXTENSION_ATTESTATION_SHA256,
    )
    attested_sha256, attested_path_text = (
        NATIVE_EXTENSION_ATTESTATION.read_text().split()
    )
    attested_path = Path(attested_path_text).resolve()
    if (
        attested_sha256 != EXPECTED_NATIVE_EXTENSION_SHA256
        or attested_path != EXPECTED_NATIVE_BUILD_EXTENSION.resolve()
        or _sha256_file(attested_path) != EXPECTED_NATIVE_EXTENSION_SHA256
    ):
        raise ValueError("the v0c native-extension attestation is inconsistent.")
    build_commit = _git(NATIVE_BUILD_SOURCE_ROOT, "rev-parse", "HEAD")
    if build_commit != EXPECTED_NATIVE_BUILD_SOURCE_COMMIT:
        raise ValueError("the native build-source worktree moved.")
    if _git(NATIVE_BUILD_SOURCE_ROOT, "status", "--porcelain"):
        raise ValueError("the native build-source worktree is dirty.")
    build_native_tree = _git(
        NATIVE_BUILD_SOURCE_ROOT, "rev-parse", "HEAD:src/simsoptpp"
    )
    instrument_native_tree = _git(INSTRUMENT_ROOT, "rev-parse", "HEAD:src/simsoptpp")
    if (
        build_native_tree != EXPECTED_NATIVE_CPP_TREE
        or instrument_native_tree != EXPECTED_NATIVE_CPP_TREE
    ):
        raise ValueError(
            "native C++ source differs between build and instrument trees."
        )
    return {
        "build_source_root": str(NATIVE_BUILD_SOURCE_ROOT),
        "build_source_commit": build_commit,
        "build_source_dirty_paths": [],
        "build_native_cpp_tree": build_native_tree,
        "instrument_native_cpp_tree": instrument_native_tree,
        "native_cpp_trees_identical": True,
        "build_extension_path": str(attested_path),
        "runtime_extension_path": str(simsoptpp_path),
        "extension_sha256": runtime_extension_sha256,
        "attestation_path": str(NATIVE_EXTENSION_ATTESTATION),
        "attestation_sha256": EXPECTED_NATIVE_EXTENSION_ATTESTATION_SHA256,
    }


def _mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{where} must be an object with string keys.")
    return value


def _float(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{where} must be a real number.")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{where} must be finite.")
    return scalar


def _float_tuple(value: object, where: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{where} must be a sequence of real numbers.")
    return tuple(
        _float(entry, f"{where}[{index}]") for index, entry in enumerate(value)
    )


def _point(payload: Mapping[str, object]) -> Mapping[str, object]:
    points = payload.get("points")
    if not isinstance(points, list):
        raise TypeError("diagnostic payload points must be a list.")
    matches = [
        point
        for point in points
        if isinstance(point, Mapping) and point.get("name") == F3_POINT_NAME
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {F3_POINT_NAME} diagnostic point.")
    return matches[0]


def _candidate(payload: Mapping[str, object]) -> Fullspace675Candidate:
    return Fullspace675Candidate(
        coil_coordinates=_float_tuple(
            payload.get("coil_coordinates"), "candidate.coil_coordinates"
        ),
        vessel_coordinates=_float_tuple(
            payload.get("vessel_coordinates"), "candidate.vessel_coordinates"
        ),
        surface_coordinates=_float_tuple(
            payload.get("surface_coordinates"), "candidate.surface_coordinates"
        ),
    )


def _candidate_with_surface(
    candidate: Fullspace675Candidate,
    surface_coordinates: tuple[float, ...],
) -> Fullspace675Candidate:
    return Fullspace675Candidate(
        coil_coordinates=candidate.coil_coordinates,
        vessel_coordinates=candidate.vessel_coordinates,
        surface_coordinates=surface_coordinates,
    )


def _provenance(
    candidate: Fullspace675Candidate,
    manifest: FrozenGenuine675InputManifest,
    material: Fullspace675FixedStateNativeMaterial,
) -> FixedStateGenuine675Provenance:
    return FixedStateGenuine675Provenance(
        candidate_semantic_sha256=candidate.semantic_sha256,
        source=manifest.launch_source,
        material=manifest.material_identity,
        objective_policy_semantic_sha256=manifest.objective_policy.semantic_sha256,
        boozer_construction_policy_semantic_sha256=(
            manifest.boozer_construction_policy.semantic_sha256
        ),
        boozer_physical_material_semantic_sha256=(
            material.boozer.physical_material_semantic_sha256
        ),
        evaluator_material_semantic_sha256=(
            material.evaluator_material_semantic_sha256
        ),
        boozer_system_material_semantic_sha256=material.boozer.semantic_sha256,
    )


def _score(
    evaluator: Fullspace675FixedStateNativeEvaluator,
    candidate: Fullspace675Candidate,
    manifest: FrozenGenuine675InputManifest,
    material: Fullspace675FixedStateNativeMaterial,
) -> dict[str, object]:
    result = evaluator.evaluate(
        candidate,
        Fullspace675ComparisonLane.NATIVE_CPP_CPU,
        _provenance(candidate, manifest, material),
    )
    term_payloads = [term.as_payload() for term in result.objective_terms]
    if len(term_payloads) != OUTER_OBJECTIVE_TERM_COUNT:
        raise ValueError("native result did not contain the canonical eight terms.")
    y_certificate = result.y_certificate.as_payload()
    return {
        "candidate_semantic_sha256": candidate.semantic_sha256,
        "objective_value": result.objective_value,
        "objective_term_order": [term.term.value for term in result.objective_terms],
        "objective_terms": term_payloads,
        "objective_term_sum": result.objective_term_sum,
        "objective_assembly": result.objective_assembly.as_payload(),
        "gradient_inf_norm": max(abs(value) for value in result.gradient.full),
        "inner_state": y_certificate["solution"],
        "inner_certificate": {
            "accepted": y_certificate["accepted"],
            "numerical_rank": y_certificate["numerical_rank"],
            "residual_l2_norm": y_certificate["residual_l2_norm"],
            "relative_fit_residual": y_certificate["relative_fit_residual"],
            "normal_residual_l2_norm": y_certificate["normal_residual_l2_norm"],
        },
        "hardware_metrics": result.hardware_metrics.as_payload(),
        "result_semantic_sha256": result.semantic_sha256,
        "provenance_semantic_sha256": result.provenance.semantic_sha256,
    }


def _canonical_vector_sha256(values: tuple[float, ...]) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _openmp_runtime() -> dict[str, object]:
    libgomp_path = next(
        Path(line.rsplit(" ", 1)[-1].strip())
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "libgomp" in line
    )
    library = ctypes.CDLL(str(libgomp_path))
    library.omp_get_max_threads.restype = ctypes.c_int
    return {
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "libgomp_path": str(libgomp_path),
        "libgomp_sha256": _sha256_file(libgomp_path),
        "omp_get_max_threads": int(library.omp_get_max_threads()),
    }


def _shared_object_dependency_closure(extension_path: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        ("ldd", str(extension_path)),
        capture_output=True,
        text=True,
        check=True,
    )
    if "not found" in completed.stdout:
        raise ValueError("simsoptpp has an unresolved shared-object dependency.")
    dependencies: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    for line in completed.stdout.splitlines():
        tokens = line.strip().split()
        if not tokens or tokens[0].startswith("linux-vdso"):
            continue
        soname = tokens[0]
        dependency_path_text = tokens[2] if "=>" in tokens else tokens[0]
        dependency_path = Path(dependency_path_text).resolve()
        if dependency_path in seen_paths:
            continue
        seen_paths.add(dependency_path)
        dependencies.append(
            {
                "soname": soname,
                "path": str(dependency_path),
                "sha256": _sha256_file(dependency_path),
            }
        )
    if not any(entry["soname"].startswith("libgomp.so") for entry in dependencies):
        raise ValueError("simsoptpp dependency closure contains no libgomp.")
    return dependencies


def main() -> int:
    args = _parse_args()
    input_hashes = {
        "bundle_manifest": _require_sha256(
            args.bundle_manifest, EXPECTED_BUNDLE_MANIFEST_SHA256
        ),
        "f3_lane": _require_sha256(args.f3_lane, EXPECTED_F3_LANE_SHA256),
        "f3_manifest": _require_sha256(args.f3_manifest, EXPECTED_F3_MANIFEST_SHA256),
        "purpose_json": _require_sha256(args.purpose_json, EXPECTED_PURPOSE_SHA256),
        "reconstruct_json": _require_sha256(
            args.reconstruct_json, EXPECTED_RECONSTRUCT_SHA256
        ),
    }
    lane = _mapping(json.loads(args.f3_lane.read_text()), "F3 lane")
    f3_manifest = _mapping(
        json.loads(args.f3_manifest.read_text()), "F3 campaign manifest"
    )
    purpose = _mapping(json.loads(args.purpose_json.read_text()), "purpose receipt")
    reconstruct = _mapping(
        json.loads(args.reconstruct_json.read_text()), "reconstruct receipt"
    )
    purpose_point = _point(purpose)
    reconstruct_point = _point(reconstruct)
    lane_result = _mapping(lane.get("result"), "F3 lane result")
    endpoint_payload = _mapping(
        lane_result.get("endpoint_candidate"), "F3 endpoint candidate"
    )
    original_candidate = _candidate(endpoint_payload)
    ls_payload = _mapping(purpose_point.get("ls"), "F3 LS purpose result")
    projected_surface = _float_tuple(
        ls_payload.get("surface_dofs"), "F3 LS projected surface"
    )
    projected_candidate = _candidate_with_surface(
        original_candidate,
        projected_surface,
    )

    validated = validate_frozen_genuine_675_input_bundle(
        args.bundle_manifest,
        source_repo=INSTRUMENT_ROOT,
    )
    manifest = validated.manifest
    boozer_material = Fullspace675NativeBoozerMaterial.from_runtime_spec(
        validated.runtime_spec,
        native_biot_savart_payload=validated.native_biot_savart_payload,
    )
    material = Fullspace675FixedStateNativeMaterial(
        boozer=boozer_material,
        vessel_template=validated.vessel_template,
        tf_current_magnitude_A=abs(float(validated.runtime_spec.seed.tf_current_A)),
    )
    evaluator = Fullspace675FixedStateNativeEvaluator(
        material=material,
        objective_policy=manifest.objective_policy,
        boozer_policy=manifest.boozer_construction_policy,
    )
    original_score = _score(
        evaluator,
        original_candidate,
        manifest,
        material,
    )
    projected_score = _score(
        evaluator,
        projected_candidate,
        manifest,
        material,
    )

    published_objective = _float(
        lane_result.get("objective_value"), "F3 published objective"
    )
    original_rescore = _float(
        original_score.get("objective_value"), "F3 native rescore"
    )
    original_relative_error = abs(original_rescore - published_objective) / abs(
        published_objective
    )
    if original_relative_error >= 1.0e-10:
        raise ValueError("native rescore did not reproduce the published F3 objective.")
    projected_objective = _float(
        projected_score.get("objective_value"), "LS-projected native rescore"
    )
    objective_delta = projected_objective - original_rescore
    reconstruct_ls = _mapping(reconstruct_point.get("ls_newton"), "F3 LS Newton result")
    reconstruct_after = _mapping(
        reconstruct_ls.get("after"), "F3 LS Newton final state"
    )
    f3_git = _mapping(f3_manifest.get("git"), "F3 campaign git identity")
    f3_production = _mapping(
        f3_git.get("production"), "F3 campaign production identity"
    )
    f3_instrument = _mapping(
        f3_git.get("instrument"), "F3 campaign instrument identity"
    )
    if (
        f3_production.get("commit") != EXPECTED_F3_PRODUCTION_COMMIT
        or f3_production.get("dirty_file_count") != 0
    ):
        raise ValueError(
            "F3 campaign manifest does not bind the clean production tree."
        )
    if (
        f3_instrument.get("commit") != EXPECTED_INSTRUMENT_COMMIT
        or f3_instrument.get("dirty_file_count") != 0
    ):
        raise ValueError(
            "F3 campaign manifest does not bind the clean instrument tree."
        )

    simsoptpp_path = Path(str(simsoptpp.__file__)).resolve()
    native_source_pairing = _native_source_pairing(simsoptpp_path)
    payload = {
        "schema": "flat675-ls-projected-outer-rescore.v1",
        "status": "diagnostic_nonpromoting",
        "claim_scope": (
            "Frozen F3 coils and vessel; native C++/CPU eight-term outer-objective "
            "comparison of the published flat-675 surface and archived C++ "
            "LS-projected surface. No optimization and no exact polish."
        ),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "simsoptpp_file": str(simsoptpp_path),
            "simsoptpp_sha256": _sha256_file(simsoptpp_path),
            "simsoptpp_dependency_closure": _shared_object_dependency_closure(
                simsoptpp_path
            ),
            "openmp": _openmp_runtime(),
        },
        "source_identities": {
            "instrument": _instrument_identity(),
            "native_source_pairing": native_source_pairing,
            "f3_campaign_production": dict(f3_production),
            "f3_campaign_instrument": dict(f3_instrument),
            "driver_path": str(Path(__file__).resolve()),
            "driver_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "inputs": {
            "paths": {
                "bundle_manifest": str(args.bundle_manifest.resolve()),
                "f3_lane": str(args.f3_lane.resolve()),
                "f3_manifest": str(args.f3_manifest.resolve()),
                "purpose_json": str(args.purpose_json.resolve()),
                "reconstruct_json": str(args.reconstruct_json.resolve()),
            },
            "sha256": input_hashes,
        },
        "physics_gate": {
            "required_abs_delta_iota_lt": 1.0e-3,
            "passed": (
                abs(_float(ls_payload.get("delta_iota"), "LS delta iota")) < 1.0e-3
            ),
            "delta_iota": _float(ls_payload.get("delta_iota"), "LS delta iota"),
            "delta_surface_l2": _float(
                ls_payload.get("delta_surface_l2"), "LS surface delta L2"
            ),
            "delta_surface_inf": _float(
                ls_payload.get("delta_surface_inf"), "LS surface delta infinity"
            ),
            "gradient_l2_after": _float(
                reconstruct_after.get("ls_grad_l2"), "LS final gradient L2"
            ),
            "gradient_inf_after": _float(
                ls_payload.get("grad_inf_after"), "LS final gradient infinity"
            ),
            "coil_x_delta_inf": _float(
                ls_payload.get("coil_x_delta_inf"), "LS coil delta infinity"
            ),
        },
        "candidate_binding": {
            "coil_dof_count": len(original_candidate.coil_coordinates),
            "vessel_dof_count": len(original_candidate.vessel_coordinates),
            "surface_dof_count": len(projected_surface),
            "coils_frozen": (
                original_candidate.coil_coordinates
                == projected_candidate.coil_coordinates
            ),
            "vessel_frozen": (
                original_candidate.vessel_coordinates
                == projected_candidate.vessel_coordinates
            ),
            "original_surface_vector_sha256": _canonical_vector_sha256(
                original_candidate.surface_coordinates
            ),
            "projected_surface_vector_sha256": _canonical_vector_sha256(
                projected_surface
            ),
            "surface_delta_l2": float(
                np.linalg.norm(
                    np.asarray(projected_surface, dtype=np.float64)
                    - np.asarray(
                        original_candidate.surface_coordinates, dtype=np.float64
                    )
                )
            ),
            "surface_delta_inf": float(
                np.linalg.norm(
                    np.asarray(projected_surface, dtype=np.float64)
                    - np.asarray(
                        original_candidate.surface_coordinates, dtype=np.float64
                    ),
                    ord=np.inf,
                )
            ),
        },
        "scores": {
            "published_f3_outer_objective": published_objective,
            "published_f3_native_rescore": original_score,
            "ls_projected_native_rescore": projected_score,
            "published_rescore_relative_error": original_relative_error,
            "projected_minus_published": objective_delta,
            "projected_over_published": projected_objective / original_rescore,
            "projected_relative_change": objective_delta / original_rescore,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "published": original_rescore,
                "projected": projected_objective,
                "relative_change": objective_delta / original_rescore,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
