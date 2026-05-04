"""Build the release-gate proof for the legacy ``JF.x`` to target ``bs.x`` map."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.validation_ladder_common import (  # noqa: E402
    bootstrap_local_simsopt,
    max_relative_error,
    parity_ladder_tolerances,
    write_json,
)

bootstrap_local_simsopt()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from simsopt.field import BiotSavart, Coil, Current  # noqa: E402
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX  # noqa: E402
from simsopt.geo import CurveXYZFourier  # noqa: E402
from simsopt.jax_core.field import coil_specs_from_dof_extraction_spec  # noqa: E402


SCHEMA_VERSION = 1
LEGACY_LANE = "cpp_cpu"
TARGET_LANE = "jax_cpu"
TARGET_GPU_LANE = "jax_gpu"
DERIVATIVE_TOLERANCES = parity_ladder_tolerances("derivative_heavy")
GRADIENT_RTOL = float(DERIVATIVE_TOLERANCES["first_derivative_rtol"])
GRADIENT_ATOL = float(DERIVATIVE_TOLERANCES["first_derivative_atol"])


@dataclass(frozen=True)
class PhysicalDof:
    full_index: int
    coil_index: int
    component: str
    local_index: int
    label: str
    free: bool
    value: float


@dataclass(frozen=True)
class MappingEntry:
    legacy_x_index: int
    target_bs_index: int
    legacy_full_index: int
    label: str
    component: str
    coil_index: int
    local_index: int


@dataclass(frozen=True)
class CoordinateMappingFixture:
    legacy_objective: DeterministicLegacyJF
    legacy_bs: BiotSavart
    target_bs: BiotSavartJAX
    descriptors: tuple[PhysicalDof, ...]


class DeterministicLegacyJF:
    """Small CPU/reference objective with the ``JF`` vector contract."""

    def __init__(
        self,
        *,
        full_x: np.ndarray,
        dofs_free_status: np.ndarray,
        reference_full_x: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        self._full_x = np.asarray(full_x, dtype=np.float64).copy()
        self.dofs_free_status = np.asarray(dofs_free_status, dtype=bool).copy()
        self._reference_full_x = np.asarray(reference_full_x, dtype=np.float64).copy()
        self._weights = np.asarray(weights, dtype=np.float64).copy()

    @property
    def x(self) -> np.ndarray:
        return self._full_x[self.dofs_free_status].copy()

    @x.setter
    def x(self, values: np.ndarray) -> None:
        updated = self._full_x.copy()
        updated[self.dofs_free_status] = np.asarray(values, dtype=np.float64)
        self._full_x = updated

    @property
    def full_x(self) -> np.ndarray:
        return self._full_x.copy()

    def J(self) -> float:
        residual = self._full_x - self._reference_full_x
        return float(0.5 * np.sum(self._weights * residual * residual))

    def dJ(self) -> np.ndarray:
        full_gradient = self._weights * (self._full_x - self._reference_full_x)
        return full_gradient[self.dofs_free_status].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the release-gate coordinate-mapping proof from legacy JF.x "
            "indices into target-lane bs.x indices."
        )
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write the coordinate-mapping proof artifact.",
    )
    return parser.parse_args()


def _make_curve(offset: float) -> CurveXYZFourier:
    curve = CurveXYZFourier(16, 1)
    curve.local_full_x = offset + np.linspace(
        0.1,
        0.9,
        curve.local_full_dof_size,
        dtype=np.float64,
    )
    return curve


def _make_fixture_coils() -> tuple[Coil, Coil]:
    tf_curve = _make_curve(0.0)
    tf_current = Current(80_000.0)
    banana_curve = _make_curve(1.0)
    banana_current = Current(12_500.0)

    tf_curve.fix_all()
    tf_current.fix_all()
    banana_curve.unfix_all()
    banana_current.unfix_all()

    return (
        Coil(tf_curve, tf_current),
        Coil(banana_curve, banana_current),
    )


def _append_dofs(
    descriptors: list[PhysicalDof],
    *,
    coil_index: int,
    component: str,
    values: np.ndarray,
    free_status: np.ndarray,
) -> None:
    for local_index, (value, free) in enumerate(
        zip(values, free_status, strict=True),
    ):
        descriptors.append(
            PhysicalDof(
                full_index=len(descriptors),
                coil_index=coil_index,
                component=component,
                local_index=int(local_index),
                label=f"coil[{coil_index}].{component}[{local_index}]",
                free=bool(free),
                value=float(value),
            )
        )


def _descriptor_payload(descriptor: PhysicalDof) -> dict[str, object]:
    return {
        "full_index": int(descriptor.full_index),
        "coil_index": int(descriptor.coil_index),
        "component": descriptor.component,
        "local_index": int(descriptor.local_index),
        "label": descriptor.label,
        "free": bool(descriptor.free),
        "value": float(descriptor.value),
    }


def _mapping_payload(entry: MappingEntry) -> dict[str, object]:
    return {
        "legacy_x_index": int(entry.legacy_x_index),
        "target_bs_index": int(entry.target_bs_index),
        "legacy_full_index": int(entry.legacy_full_index),
        "label": entry.label,
        "component": entry.component,
        "coil_index": int(entry.coil_index),
        "local_index": int(entry.local_index),
    }


def build_deterministic_coordinate_mapping_fixture() -> CoordinateMappingFixture:
    """Build a deterministic two-group coil graph for the mapping proof."""
    target_coils = _make_fixture_coils()
    legacy_coils = _make_fixture_coils()
    descriptors: list[PhysicalDof] = []
    for coil_index, coil in enumerate(target_coils):
        _append_dofs(
            descriptors,
            coil_index=coil_index,
            component="current",
            values=np.asarray(coil.current.local_full_x, dtype=np.float64),
            free_status=np.asarray(coil.current.local_dofs_free_status, dtype=bool),
        )
        _append_dofs(
            descriptors,
            coil_index=coil_index,
            component="curve",
            values=np.asarray(coil.curve.local_full_x, dtype=np.float64),
            free_status=np.asarray(coil.curve.local_dofs_free_status, dtype=bool),
        )

    full_x = np.asarray([descriptor.value for descriptor in descriptors], dtype=np.float64)
    free_status = np.asarray(
        [descriptor.free for descriptor in descriptors],
        dtype=bool,
    )
    reference_offsets = np.linspace(
        0.015,
        0.015 + 0.004 * (full_x.size - 1),
        full_x.size,
        dtype=np.float64,
    )
    reference_full_x = full_x - reference_offsets
    weights = 1.0 + 0.125 * np.arange(full_x.size, dtype=np.float64)
    legacy_objective = DeterministicLegacyJF(
        full_x=full_x,
        dofs_free_status=free_status,
        reference_full_x=reference_full_x,
        weights=weights,
    )
    return CoordinateMappingFixture(
        legacy_objective=legacy_objective,
        legacy_bs=BiotSavart(list(legacy_coils)),
        target_bs=BiotSavartJAX(list(target_coils)),
        descriptors=tuple(descriptors),
    )


def _coordinate_lookup(
    descriptors: tuple[PhysicalDof, ...],
) -> dict[tuple[int, str, int], PhysicalDof]:
    return {
        (descriptor.coil_index, descriptor.component, descriptor.local_index): descriptor
        for descriptor in descriptors
    }


def _full_to_legacy_x_index(descriptors: tuple[PhysicalDof, ...]) -> dict[int, int]:
    active_full_indices = [
        descriptor.full_index for descriptor in descriptors if descriptor.free
    ]
    return {
        int(full_index): int(legacy_index)
        for legacy_index, full_index in enumerate(active_full_indices)
    }


def _mapping_entries_from_target_spec(
    fixture: CoordinateMappingFixture,
) -> tuple[MappingEntry, ...]:
    spec = fixture.target_bs.coil_dof_extraction_spec()
    coordinate_lookup = _coordinate_lookup(fixture.descriptors)
    full_to_legacy_index = _full_to_legacy_x_index(fixture.descriptors)
    entries: list[MappingEntry] = []

    for coil_index, coil_spec in enumerate(spec.coils):
        component_maps = (
            ("current", coil_spec.current_map),
            ("curve", coil_spec.curve_map),
        )
        for component, map_spec in component_maps:
            for owner_start, owner_end, target_start, _target_end in (
                map_spec.owner_segments
            ):
                for segment_offset, target_bs_index in enumerate(
                    range(owner_start, owner_end),
                ):
                    local_index = int(target_start + segment_offset)
                    descriptor = coordinate_lookup[
                        (int(coil_index), component, local_index)
                    ]
                    entries.append(
                        MappingEntry(
                            legacy_x_index=full_to_legacy_index[
                                descriptor.full_index
                            ],
                            target_bs_index=int(target_bs_index),
                            legacy_full_index=int(descriptor.full_index),
                            label=descriptor.label,
                            component=component,
                            coil_index=int(coil_index),
                            local_index=local_index,
                        )
                    )
    return tuple(sorted(entries, key=lambda entry: entry.target_bs_index))


def _status_from_sections(sections: list[dict[str, object]]) -> str:
    statuses = [str(section.get("status", "blocked")) for section in sections]
    if "blocked" in statuses:
        return "blocked"
    if any(status != "pass" for status in statuses):
        return "drift"
    return "pass"


def _section_failure(section_name: str, section: dict[str, object]) -> list[str]:
    status = str(section.get("status", "blocked"))
    if status == "pass":
        return []
    reason = section.get("reason", status)
    return [f"{section_name}: {reason}"]


def _values_from_entries(values: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.asarray([values[index] for index in indices], dtype=np.float64)


def _target_values_from_specs(
    fixture: CoordinateMappingFixture,
    target_dofs: jax.Array,
) -> jax.Array:
    coil_specs = coil_specs_from_dof_extraction_spec(
        fixture.target_bs.coil_dof_extraction_spec(),
        target_dofs,
    )
    values = []
    for descriptor in fixture.descriptors:
        coil_spec = coil_specs[descriptor.coil_index]
        if descriptor.component == "current":
            values.append(coil_spec.current.value[0])
        else:
            values.append(coil_spec.curve.dofs[descriptor.local_index])
    return jnp.stack(values)


def _target_quadratic_objective(
    fixture: CoordinateMappingFixture,
    target_dofs: jax.Array,
    reference_full_x: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    target_full_x = _target_values_from_specs(fixture, target_dofs)
    residual = target_full_x - reference_full_x
    return 0.5 * jnp.sum(weights * residual * residual)


def _build_mapping_section(
    fixture: CoordinateMappingFixture,
    entries: tuple[MappingEntry, ...],
) -> dict[str, object]:
    legacy_x = fixture.legacy_objective.x
    target_x = np.asarray(fixture.target_bs.x, dtype=np.float64)
    legacy_indices = [entry.legacy_x_index for entry in entries]
    target_indices = [entry.target_bs_index for entry in entries]
    mapped_legacy_values = _values_from_entries(legacy_x, legacy_indices)
    mapped_target_values = _values_from_entries(target_x, target_indices)
    unique_legacy_count = len(set(legacy_indices))
    unique_target_count = len(set(target_indices))
    value_delta = np.abs(mapped_legacy_values - mapped_target_values)

    failures = []
    if len(entries) != legacy_x.size:
        failures.append(
            "legacy cpp_cpu JF.x to jax_cpu bs.x mapping size mismatch: "
            f"entries={len(entries)} legacy_x={legacy_x.size}"
        )
    if len(entries) != target_x.size:
        failures.append(
            "target jax_cpu bs.x mapping size mismatch: "
            f"entries={len(entries)} target_x={target_x.size}"
        )
    if unique_legacy_count != len(entries):
        failures.append("legacy cpp_cpu JF.x indices are not one-to-one")
    if unique_target_count != len(entries):
        failures.append("target jax_cpu bs.x indices are not one-to-one")
    if value_delta.size and float(np.max(value_delta)) > 0.0:
        failures.append("mapped legacy and target initial coordinate values differ")

    return {
        "status": "pass" if not failures else "blocked",
        "entries": [_mapping_payload(entry) for entry in entries],
        "legacy_to_target": {
            str(entry.legacy_x_index): int(entry.target_bs_index)
            for entry in entries
        },
        "target_to_legacy": {
            str(entry.target_bs_index): int(entry.legacy_x_index)
            for entry in entries
        },
        "entry_count": int(len(entries)),
        "legacy_x_size": int(legacy_x.size),
        "target_bs_x_size": int(target_x.size),
        "max_abs_initial_value_delta": (
            float(np.max(value_delta)) if value_delta.size else 0.0
        ),
        "failures": failures,
    }


def _build_active_indices_section(
    fixture: CoordinateMappingFixture,
    entries: tuple[MappingEntry, ...],
) -> dict[str, object]:
    active_full_indices = [
        descriptor.full_index for descriptor in fixture.descriptors if descriptor.free
    ]
    mapped_full_indices = [entry.legacy_full_index for entry in entries]
    mapped_legacy_indices = [entry.legacy_x_index for entry in entries]
    mapped_target_indices = [entry.target_bs_index for entry in entries]
    failures = []
    if sorted(active_full_indices) != sorted(mapped_full_indices):
        failures.append("active legacy full indices do not match mapping entries")
    if sorted(mapped_legacy_indices) != list(range(len(mapped_legacy_indices))):
        failures.append("legacy cpp_cpu JF.x active indices are not contiguous")
    if sorted(mapped_target_indices) != list(range(len(mapped_target_indices))):
        failures.append("target jax_cpu bs.x active indices are not contiguous")
    return {
        "status": "pass" if not failures else "blocked",
        "legacy_full": [int(index) for index in active_full_indices],
        "legacy_x": [int(index) for index in mapped_legacy_indices],
        "target_bs_x": [int(index) for index in mapped_target_indices],
        "failures": failures,
    }


def _build_frozen_indices_section(
    fixture: CoordinateMappingFixture,
    entries: tuple[MappingEntry, ...],
) -> dict[str, object]:
    frozen_full_indices = [
        descriptor.full_index for descriptor in fixture.descriptors if not descriptor.free
    ]
    mapped_frozen = sorted(
        set(frozen_full_indices).intersection(
            {entry.legacy_full_index for entry in entries}
        )
    )
    failures = [
        f"frozen legacy full index leaked into target bs.x mapping: {index}"
        for index in mapped_frozen
    ]
    return {
        "status": "pass" if not failures else "blocked",
        "legacy_full": [int(index) for index in frozen_full_indices],
        "target_hits": [int(index) for index in mapped_frozen],
        "frozen_labels": [
            descriptor.label for descriptor in fixture.descriptors if not descriptor.free
        ],
        "failures": failures,
    }


def _apply_target_perturbation_to_legacy_full_x(
    baseline_full_x: np.ndarray,
    entries: tuple[MappingEntry, ...],
    target_perturbation: np.ndarray,
) -> np.ndarray:
    updated = baseline_full_x.copy()
    for entry in entries:
        updated[entry.legacy_full_index] += target_perturbation[
            entry.target_bs_index
        ]
    return updated


def _state_reconstruction_section(
    fixture: CoordinateMappingFixture,
    entries: tuple[MappingEntry, ...],
) -> dict[str, object]:
    baseline_target_x = np.asarray(fixture.target_bs.x, dtype=np.float64)
    perturbation = np.linspace(
        -2.5e-4,
        2.5e-4,
        baseline_target_x.size,
        dtype=np.float64,
    )
    perturbed_target_x = baseline_target_x + perturbation
    legacy_full_x = _apply_target_perturbation_to_legacy_full_x(
        fixture.legacy_objective.full_x,
        entries,
        perturbation,
    )
    target_values = np.asarray(
        _target_values_from_specs(
            fixture,
            jnp.asarray(perturbed_target_x, dtype=jnp.float64),
        ),
        dtype=np.float64,
    )
    full_delta = np.abs(legacy_full_x - target_values)

    gamma_deltas = []
    fixture.target_bs.x = perturbed_target_x
    for coil_index, coil in enumerate(fixture.legacy_bs.coils):
        curve_indices = [
            descriptor.full_index
            for descriptor in fixture.descriptors
            if descriptor.coil_index == coil_index and descriptor.component == "curve"
        ]
        coil.curve.local_full_x = _values_from_entries(legacy_full_x, curve_indices)
        target_coil = fixture.target_bs.coils[coil_index]
        gamma_deltas.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(coil.curve.gamma(), dtype=np.float64)
                        - np.asarray(target_coil.curve.gamma(), dtype=np.float64)
                    )
                )
            )
        )
    fixture.target_bs.x = baseline_target_x
    max_full_delta = float(np.max(full_delta)) if full_delta.size else 0.0
    max_gamma_delta = float(np.max(gamma_deltas)) if gamma_deltas else 0.0
    failures = []
    if max_full_delta > GRADIENT_ATOL:
        failures.append("legacy and target reconstructed DOF state differ")
    if max_gamma_delta > GRADIENT_ATOL:
        failures.append("legacy and target reconstructed curve geometry differ")

    return {
        "status": "pass" if not failures else "drift",
        "perturbation": {
            "size": int(perturbation.size),
            "inf_norm": float(np.max(np.abs(perturbation))) if perturbation.size else 0.0,
        },
        "max_abs_full_state_delta": max_full_delta,
        "max_abs_curve_gamma_delta": max_gamma_delta,
        "failures": failures,
    }


def _project_legacy_gradient(
    legacy_gradient: np.ndarray,
    entries: tuple[MappingEntry, ...],
) -> np.ndarray:
    projected = np.zeros(len(entries), dtype=np.float64)
    for entry in entries:
        projected[entry.target_bs_index] = legacy_gradient[entry.legacy_x_index]
    return projected


def _gradient_projection_section(
    fixture: CoordinateMappingFixture,
    entries: tuple[MappingEntry, ...],
    *,
    target_gradient_override: np.ndarray | None,
) -> dict[str, object]:
    reference_full_x = jnp.asarray(
        fixture.legacy_objective._reference_full_x,
        dtype=jnp.float64,
    )
    weights = jnp.asarray(fixture.legacy_objective._weights, dtype=jnp.float64)
    target_x = jnp.asarray(fixture.target_bs.x, dtype=jnp.float64)
    target_gradient = np.asarray(
        jax.grad(
            lambda x: _target_quadratic_objective(
                fixture,
                x,
                reference_full_x,
                weights,
            )
        )(target_x),
        dtype=np.float64,
    )
    if target_gradient_override is not None:
        target_gradient = np.asarray(target_gradient_override, dtype=np.float64)
    legacy_gradient = fixture.legacy_objective.dJ()
    projected = _project_legacy_gradient(legacy_gradient, entries)
    failures = []
    if target_gradient.shape != projected.shape:
        failures.append(
            "gradient shape mismatch for cpp_cpu JF.x -> jax_cpu bs.x projection: "
            f"projected={projected.size} target={target_gradient.size}"
        )
        return {
            "status": "blocked",
            "reason": failures[0],
            "legacy_gradient_size": int(legacy_gradient.size),
            "projected_gradient_size": int(projected.size),
            "target_gradient_size": int(target_gradient.size),
            "failures": failures,
        }

    abs_delta = np.abs(projected - target_gradient)
    max_abs_delta = float(np.max(abs_delta)) if abs_delta.size else 0.0
    max_rel_delta = max_relative_error(projected, target_gradient)
    if not np.allclose(
        projected,
        target_gradient,
        rtol=GRADIENT_RTOL,
        atol=GRADIENT_ATOL,
    ):
        failures.append("projected cpp_cpu gradient differs from target jax_cpu gradient")
    return {
        "status": "pass" if not failures else "drift",
        "legacy_gradient": legacy_gradient.tolist(),
        "projected_legacy_gradient": projected.tolist(),
        "target_gradient": target_gradient.tolist(),
        "max_abs_delta": max_abs_delta,
        "max_relative_delta": max_rel_delta,
        "tolerances": {
            "rtol": GRADIENT_RTOL,
            "atol": GRADIENT_ATOL,
            "lane": "derivative_heavy",
        },
        "failures": failures,
    }


def _finite_difference_checks_section(
    fixture: CoordinateMappingFixture,
    gradient_projection: dict[str, object],
) -> dict[str, object]:
    if gradient_projection.get("status") == "blocked":
        return {
            "status": "blocked",
            "reason": "gradient projection is blocked",
            "directions": [],
            "failures": ["gradient projection is blocked"],
        }
    target_gradient = np.asarray(
        gradient_projection["target_gradient"],
        dtype=np.float64,
    )
    target_x = np.asarray(fixture.target_bs.x, dtype=np.float64)
    direction_indices = sorted({0, target_x.size // 2, target_x.size - 1})
    step = 1.0e-4
    reference_full_x = jnp.asarray(
        fixture.legacy_objective._reference_full_x,
        dtype=jnp.float64,
    )
    weights = jnp.asarray(fixture.legacy_objective._weights, dtype=jnp.float64)

    def evaluate(x: np.ndarray) -> float:
        return float(
            _target_quadratic_objective(
                fixture,
                jnp.asarray(x, dtype=jnp.float64),
                reference_full_x,
                weights,
            )
        )

    directions = []
    failures = []
    for target_index in direction_indices:
        basis = np.zeros_like(target_x)
        basis[target_index] = 1.0
        fd = (evaluate(target_x + step * basis) - evaluate(target_x - step * basis)) / (
            2.0 * step
        )
        analytic = float(target_gradient[target_index])
        abs_delta = float(abs(fd - analytic))
        status = (
            "pass"
            if np.isclose(fd, analytic, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL)
            else "drift"
        )
        if status != "pass":
            failures.append(f"finite difference drift at target bs.x index {target_index}")
        directions.append(
            {
                "status": status,
                "target_bs_index": int(target_index),
                "finite_difference": float(fd),
                "analytic": analytic,
                "abs_delta": abs_delta,
            }
        )

    return {
        "status": "pass" if not failures else "drift",
        "step_size": float(step),
        "directions": directions,
        "failures": failures,
    }


def _aggregate_failures(sections: dict[str, dict[str, object]]) -> list[str]:
    failures: list[str] = []
    for section_name, section in sections.items():
        failures.extend(_section_failure(section_name, section))
        section_failures = section.get("failures", [])
        if isinstance(section_failures, list):
            failures.extend(str(failure) for failure in section_failures)
    return failures


def build_coordinate_mapping_proof(
    *,
    target_gradient_override: np.ndarray | None = None,
) -> dict[str, object]:
    fixture = build_deterministic_coordinate_mapping_fixture()
    entries = _mapping_entries_from_target_spec(fixture)
    descriptors_payload = [
        _descriptor_payload(descriptor) for descriptor in fixture.descriptors
    ]
    inputs = {
        "status": "pass",
        "fixture": "deterministic_two_group_single_stage_coordinate_contract",
        "legacy_lane": LEGACY_LANE,
        "target_lane": TARGET_LANE,
        "target_gpu_lane": TARGET_GPU_LANE,
        "legacy_optimizer_space": "JF.x",
        "target_optimizer_space": "bs.x",
        "legacy_x_size": int(fixture.legacy_objective.x.size),
        "target_bs_x_size": int(np.asarray(fixture.target_bs.x).size),
        "legacy_full_size": int(fixture.legacy_objective.full_x.size),
        "num_tf_groups": 1,
        "num_active_banana_groups": 1,
        "dofs": descriptors_payload,
    }
    sections = {
        "mapping": _build_mapping_section(fixture, entries),
        "active_indices": _build_active_indices_section(fixture, entries),
        "frozen_indices": _build_frozen_indices_section(fixture, entries),
        "state_reconstruction": _state_reconstruction_section(fixture, entries),
        "gradient_projection": _gradient_projection_section(
            fixture,
            entries,
            target_gradient_override=target_gradient_override,
        ),
    }
    sections["finite_difference_checks"] = _finite_difference_checks_section(
        fixture,
        sections["gradient_projection"],
    )
    status = _status_from_sections([inputs, *sections.values()])
    failures = _aggregate_failures(sections)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "inputs": inputs,
        **sections,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    artifact = build_coordinate_mapping_proof()
    write_json(args.output_json, artifact)
    return 0 if artifact["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
