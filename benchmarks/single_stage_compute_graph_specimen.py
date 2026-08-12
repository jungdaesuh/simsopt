"""Freeze the native-default changed-state specimen for Phase 0 canaries.

This module performs construction and serialization only.  It deliberately
does not prepare or evaluate the JAX objective.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from examples.jax.parity.artifacts import write_array, write_bytes_exclusive
from examples.jax.parity.cases.native_boozerqa import (
    _prepare_jax_variant_runtime,
    create_variant_input,
)
from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
from examples.jax.parity.input_bundle import read_input_bundle

from benchmarks.single_stage_compute_graph_c0_evaluator import (
    EXPECTED_PARAMETER_COUNT,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    COIL_DOF_COUNT,
    STATE_DIMENSION,
    canonical_json_bytes,
    canonical_sha256,
)

SCHEMA_ID: Final = "single-stage-compute-graph-frozen-specimen-v1"
SPECIMEN_ID: Final = "native-single-stage-changed-state-c0-v1"
CANDIDATE_PATH: Final = "changed_state_candidate.npy"
INPUT_BUNDLE_PATH: Final = "input_bundle"
DOCUMENT_PATH: Final = "specimen.json"


class SpecimenError(ValueError):
    """The frozen specimen cannot be constructed without contract drift."""


@dataclass(frozen=True, slots=True)
class EffectivePolicies:
    """Exact policies shared by every measurement of this specimen."""

    dense_batch_width: int = 8
    point_chunk_size: int | None = None
    coil_chunk_size: int | None = None
    quadrature_block_sizes: tuple[int, ...] = (128, 122)

    def __post_init__(self) -> None:
        if self.dense_batch_width < 1:
            raise SpecimenError("dense_batch_width must be positive")
        for name, value in (
            ("point_chunk_size", self.point_chunk_size),
            ("coil_chunk_size", self.coil_chunk_size),
        ):
            if value is not None and value < 1:
                raise SpecimenError(f"{name} must be positive or None")
        if not self.quadrature_block_sizes or any(
            size < 1 for size in self.quadrature_block_sizes
        ):
            raise SpecimenError("quadrature_block_sizes must be positive")


DEFAULT_EFFECTIVE_POLICIES: Final = EffectivePolicies()


@dataclass(frozen=True, slots=True)
class FrozenSpecimen:
    """Paths and identities of one immutable-on-publication specimen."""

    root: Path
    document_path: Path
    input_bundle_path: Path
    candidate_path: Path
    specimen_sha256: str
    parameter_sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parameter_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    return _sha256_bytes(canonical.tobytes(order="C"))


def _changed_candidate(baseline: np.ndarray) -> np.ndarray:
    if baseline.dtype != np.dtype(np.float64) or baseline.shape != (
        EXPECTED_PARAMETER_COUNT,
    ):
        raise SpecimenError("baseline coil_dofs must have shape (461,) and float64")
    if not bool(np.all(np.isfinite(baseline))):
        raise SpecimenError("baseline coil_dofs must be finite")
    index = np.arange(EXPECTED_PARAMETER_COUNT, dtype=np.int64)
    pattern = ((index % 17) - 8).astype(np.float64) / 8.0
    magnitude = np.maximum(1.0, np.abs(baseline))
    candidate = np.ascontiguousarray(
        baseline + np.ldexp(magnitude, -20) * pattern,
        dtype=np.dtype("<f8"),
    )
    if not bool(np.all(np.isfinite(candidate))):
        raise SpecimenError("changed-state construction produced non-finite values")
    if np.array_equal(candidate, baseline):
        raise SpecimenError("changed-state candidate equals the baseline")
    candidate.setflags(write=False)
    return candidate


def _required_integer(configuration: Mapping[str, object], name: str) -> int:
    value = configuration.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecimenError(f"configuration {name} must be an integer")
    return value


def _required_number(configuration: Mapping[str, object], name: str) -> float:
    value = configuration.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecimenError(f"configuration {name} must be numeric")
    checked = float(value)
    if not np.isfinite(checked):
        raise SpecimenError(f"configuration {name} must be finite")
    return checked


def build_frozen_changed_state_specimen(
    root: Path,
    *,
    policies: EffectivePolicies = DEFAULT_EFFECTIVE_POLICIES,
) -> FrozenSpecimen:
    """Construct and persist one exact native-default changed-state specimen."""

    if root.exists():
        raise SpecimenError("specimen root must not already exist")
    input_root = root / INPUT_BUNDLE_PATH
    create_variant_input(input_root, "native_default", SPEC)
    bundle, arrays = read_input_bundle(input_root)
    if bundle.case_id != SPEC.case_id or bundle.scale != "native_default":
        raise SpecimenError("input bundle does not match current native-default SPEC")
    baseline = arrays.get("coil_dofs")
    if baseline is None:
        raise SpecimenError("input bundle is missing coil_dofs")
    candidate = _changed_candidate(baseline)
    if COIL_DOF_COUNT != EXPECTED_PARAMETER_COUNT:
        raise SpecimenError("Phase 0 receipt and C0 evaluator coil counts disagree")

    configuration = bundle.configuration
    mpol = _required_integer(configuration, "mpol")
    ntor = _required_integer(configuration, "ntor")
    non_qs_sdim = _required_integer(configuration, "non_qs_sdim")
    grids = {
        "inner_surface_points": (2 * mpol + 1) * (2 * ntor + 1),
        "non_qs_surface_points": (2 * non_qs_sdim) ** 2,
        "physical_coil_contributions": 18,
        "quadrature_nodes": sum(policies.quadrature_block_sizes),
    }
    if grids != {
        "inner_surface_points": 169,
        "non_qs_surface_points": 1600,
        "physical_coil_contributions": 18,
        "quadrature_nodes": 250,
    }:
        raise SpecimenError("native-default production grid contract drifted")

    solver_graph = {
        "owner": (
            f"{_prepare_jax_variant_runtime.__module__}."
            f"{_prepare_jax_variant_runtime.__qualname__}"
        ),
        "variant": "C0",
        "newton_linearization": "matrix_free_jvp_incremental_gmres",
        "step_control": "current_jax_backtracking",
        "workflow_stages": list(SPEC.workflow_stages),
        "policies": dataclasses.asdict(policies),
    }
    parameter_sha256 = _parameter_sha256(candidate)
    baseline_sha256 = _parameter_sha256(baseline)
    candidate_reference = write_array(root, CANDIDATE_PATH, candidate)
    input_bundle_bytes = (input_root / "input_bundle.json").read_bytes()
    specimen = {
        "specimen_id": SPECIMEN_ID,
        "input_bundle_sha256": _sha256_bytes(input_bundle_bytes),
        "parameter_sha256": parameter_sha256,
        "state_dimension": STATE_DIMENSION,
        "coil_dof_count": COIL_DOF_COUNT,
        "grids": grids,
        "weights": {
            "iota": 1.0,
            "length": 1.0,
            "major_radius": 1.0,
            "non_qs": 1.0,
            "residual": _required_number(configuration, "residual_weight"),
        },
        "tolerances": {
            "inner": _required_number(configuration, "inner_tolerance"),
            "outer_atol": _required_number(configuration, "outer_atol"),
            "outer_rtol": _required_number(configuration, "outer_rtol"),
        },
        "solver_graph_id": "c0-current-jvp-incremental-gmres",
        "solver_graph_sha256": canonical_sha256(solver_graph),
    }
    document = {
        "schema_id": SCHEMA_ID,
        "specimen": specimen,
        "specimen_sha256": canonical_sha256(specimen),
        "input_bundle": {
            "relative_path": INPUT_BUNDLE_PATH,
            "input_fingerprint": bundle.input_fingerprint,
            "configuration_fingerprint": bundle.configuration_fingerprint,
        },
        "candidate": {
            "relative_path": candidate_reference.path,
            "file_sha256": candidate_reference.sha256,
            "dtype": candidate_reference.dtype,
            "shape": list(candidate_reference.shape),
            "parameter_sha256": parameter_sha256,
            "baseline_parameter_sha256": baseline_sha256,
            "differs_from_baseline": True,
            "generator": "relative-binary-rational-period-17-exp-minus-20-v1",
        },
        "solver_graph": solver_graph,
        "effective_policies": dataclasses.asdict(policies),
    }
    write_bytes_exclusive(root, DOCUMENT_PATH, canonical_json_bytes(document))
    return FrozenSpecimen(
        root=root,
        document_path=root / DOCUMENT_PATH,
        input_bundle_path=input_root,
        candidate_path=root / CANDIDATE_PATH,
        specimen_sha256=str(document["specimen_sha256"]),
        parameter_sha256=parameter_sha256,
    )
