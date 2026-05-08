"""Bit-identity census schema and helpers for Boozer derivative inputs.

The strict CPU/JAX parity gate at
``benchmarks/single_stage_init_parity.py:1905`` reports a divergent
``boozer_solve.pre_newton_state`` whose magnitude (~4.5e-9) is the BFGS
amplification of a tiny first-step gradient mismatch (~1.6e-15). This module
captures the *raw derivative inputs* both backends feed into the Boozer
residual, hashes the float64 bytes, and reports the first array where they
differ — the ladder traced in
``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`` Phase 1.

Diagnostic-only — not on the production import path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from simsopt.geo.boozersurface import BoozerSurface
    from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX


# Canonical names for the boundary arrays. Producers MUST use these names so
# diff records can be paired one-to-one across backends. The order is
# significant for first-divergence reporting and matches the ladder the plan
# uses ("gamma is the first kernel that ULP-drifts" → §1).
CENSUS_BOUNDARY_ARRAY_ORDER: tuple[str, ...] = (
    "gamma",
    "xphi",
    "xtheta",
    "dx_ds",
    "dxphi_ds",
    "dxtheta_ds",
    "B",
    "dB_dX",
)

CENSUS_BOUNDARY_SCALAR_ORDER: tuple[str, ...] = (
    "G_value",
    "iota",
    "weight_inv_modB",
)

CENSUS_STAGE_DEFAULT = "boozer_ls_callback_input"


@dataclasses.dataclass(frozen=True)
class CensusArrayRecord:
    """Single producer's snapshot of one boundary array."""

    array_name: str
    producer: str  # "cpu" | "jax"
    stage: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    contiguity: str  # "C" | "F" | "non-contiguous"
    sha256_float64_bytes: str
    norm_l2: float
    norm_linf: float

    def to_json_record(self) -> dict[str, Any]:
        return {
            "kind": "array",
            "array_name": self.array_name,
            "producer": self.producer,
            "stage": self.stage,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "strides": list(self.strides),
            "contiguity": self.contiguity,
            "sha256_float64_bytes": self.sha256_float64_bytes,
            "norm_l2": self.norm_l2,
            "norm_linf": self.norm_linf,
        }


@dataclasses.dataclass(frozen=True)
class CensusScalarRecord:
    """Producer's snapshot of one boundary scalar (e.g. G, iota).

    Scalars get a dedicated record type because their byte digest is over a
    single float64 and the whole concept of "shape" / "strides" is degenerate;
    we keep the float and a dtype tag so the diff helper can still flag bit
    drift.
    """

    name: str
    producer: str
    stage: str
    value: float
    dtype: str
    sha256_float64_bytes: str

    def to_json_record(self) -> dict[str, Any]:
        return {
            "kind": "scalar",
            "name": self.name,
            "producer": self.producer,
            "stage": self.stage,
            "value": self.value,
            "dtype": self.dtype,
            "sha256_float64_bytes": self.sha256_float64_bytes,
        }


@dataclasses.dataclass(frozen=True)
class CensusArrayDiff:
    """Paired diff for one boundary array (CPU vs JAX)."""

    array_name: str
    stage: str
    max_abs_diff: float
    argmax_abs_diff: tuple[int, ...] | None
    first_unequal_byte_index: int | None
    first_unequal_numeric_index: tuple[int, ...] | None
    n_bit_different_entries: int
    byte_identical: bool
    cpu_sha256_float64_bytes: str | None
    jax_sha256_float64_bytes: str | None
    shape_match: bool
    dtype_match: bool

    def to_json_record(self) -> dict[str, Any]:
        return {
            "kind": "array_diff",
            "array_name": self.array_name,
            "stage": self.stage,
            "max_abs_diff": self.max_abs_diff,
            "argmax_abs_diff": (
                list(self.argmax_abs_diff) if self.argmax_abs_diff is not None else None
            ),
            "first_unequal_byte_index": self.first_unequal_byte_index,
            "first_unequal_numeric_index": (
                list(self.first_unequal_numeric_index)
                if self.first_unequal_numeric_index is not None
                else None
            ),
            "n_bit_different_entries": self.n_bit_different_entries,
            "byte_identical": self.byte_identical,
            "cpu_sha256_float64_bytes": self.cpu_sha256_float64_bytes,
            "jax_sha256_float64_bytes": self.jax_sha256_float64_bytes,
            "shape_match": self.shape_match,
            "dtype_match": self.dtype_match,
        }


@dataclasses.dataclass(frozen=True)
class CensusScalarDiff:
    """Paired diff for one scalar boundary input."""

    name: str
    stage: str
    cpu_value: float
    jax_value: float
    abs_diff: float
    byte_identical: bool

    def to_json_record(self) -> dict[str, Any]:
        return {
            "kind": "scalar_diff",
            "name": self.name,
            "stage": self.stage,
            "cpu_value": self.cpu_value,
            "jax_value": self.jax_value,
            "abs_diff": self.abs_diff,
            "byte_identical": self.byte_identical,
        }


# ---------------------------------------------------------------------------
# Producers


class CensusLayoutError(RuntimeError):
    """Raised when an array does not satisfy the float64 layout invariant.

    Per Phase 1 of the implementation plan, the byte digest must be over the
    *original* float64 representation; silently casting away the evidence is
    forbidden. Producers should hit this only for clear bugs (e.g. accidental
    float32 leak from a JAX kernel without x64).
    """


def _array_contiguity_tag(arr: np.ndarray) -> str:
    if arr.flags["C_CONTIGUOUS"]:
        return "C"
    if arr.flags["F_CONTIGUOUS"]:
        return "F"
    return "non-contiguous"


def _sha256_float64_bytes(arr: np.ndarray) -> str:
    if arr.dtype != np.float64:
        raise CensusLayoutError(
            f"census array {arr.shape} dtype={arr.dtype}; expected float64"
        )
    canonical = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def build_array_record(
    *,
    array_name: str,
    producer: str,
    stage: str,
    array: np.ndarray,
) -> CensusArrayRecord:
    if array.dtype != np.float64:
        raise CensusLayoutError(
            f"{producer}::{array_name} dtype={array.dtype}; expected float64"
        )
    digest = _sha256_float64_bytes(array)
    flat = array.ravel()
    return CensusArrayRecord(
        array_name=array_name,
        producer=producer,
        stage=stage,
        dtype=str(array.dtype),
        shape=tuple(int(s) for s in array.shape),
        strides=tuple(int(s) for s in array.strides),
        contiguity=_array_contiguity_tag(array),
        sha256_float64_bytes=digest,
        norm_l2=float(np.linalg.norm(flat)),
        norm_linf=float(np.max(np.abs(flat)) if flat.size else 0.0),
    )


def build_scalar_record(
    *,
    name: str,
    producer: str,
    stage: str,
    value: float,
) -> CensusScalarRecord:
    arr = np.asarray(value, dtype=np.float64).reshape(())
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return CensusScalarRecord(
        name=name,
        producer=producer,
        stage=stage,
        value=float(arr),
        dtype=str(arr.dtype),
        sha256_float64_bytes=digest,
    )


# ---------------------------------------------------------------------------
# Diffs


def _first_unequal_byte_index(cpu: np.ndarray, jax_: np.ndarray) -> int | None:
    """Return the first byte index where the float64 representations differ.

    Returns ``None`` when the byte sequences are identical OR when shapes
    differ (the latter is reported separately by ``shape_match``); using
    ``0`` as a sentinel for shape mismatches would conflict with a real
    first-byte mismatch at index 0.
    """
    cpu_bytes = np.ascontiguousarray(cpu, dtype=np.float64).view(np.uint8)
    jax_bytes = np.ascontiguousarray(jax_, dtype=np.float64).view(np.uint8)
    if cpu_bytes.shape != jax_bytes.shape:
        return None
    diff = cpu_bytes != jax_bytes
    if not diff.any():
        return None
    return int(np.argmax(diff))


def _bytewise_unequal_double_count(cpu: np.ndarray, jax_: np.ndarray) -> int:
    """Count float64 lanes whose 8-byte representation differs.

    Distinct from ``cpu != jax_`` (IEEE numeric equality), which treats
    ``+0.0`` and ``-0.0`` as equal even though their byte representations
    differ. This count reflects the bit-identity contract.
    """
    if cpu.shape != jax_.shape:
        return -1
    cpu_view = np.ascontiguousarray(cpu, dtype=np.float64).view(np.uint64)
    jax_view = np.ascontiguousarray(jax_, dtype=np.float64).view(np.uint64)
    return int(np.count_nonzero(cpu_view != jax_view))


def _first_unequal_numeric_index(
    cpu: np.ndarray, jax_: np.ndarray
) -> tuple[int, ...] | None:
    if cpu.shape != jax_.shape:
        return None
    diff_mask = cpu != jax_
    if not diff_mask.any():
        return None
    flat_index = int(np.argmax(diff_mask))
    return tuple(int(i) for i in np.unravel_index(flat_index, cpu.shape))


def compare_array(
    *,
    array_name: str,
    stage: str,
    cpu: np.ndarray,
    jax_: np.ndarray,
    cpu_record: CensusArrayRecord | None = None,
    jax_record: CensusArrayRecord | None = None,
) -> CensusArrayDiff:
    shape_match = cpu.shape == jax_.shape
    dtype_match = cpu.dtype == jax_.dtype == np.float64
    if shape_match and cpu.dtype == np.float64 and jax_.dtype == np.float64:
        diff_abs = np.abs(cpu - jax_)
        max_abs = float(diff_abs.max()) if diff_abs.size else 0.0
        if max_abs > 0.0:
            argmax = tuple(
                int(i) for i in np.unravel_index(int(np.argmax(diff_abs)), cpu.shape)
            )
        else:
            argmax = None
        first_byte = _first_unequal_byte_index(cpu, jax_)
        first_numeric = _first_unequal_numeric_index(cpu, jax_)
        # Count over the float64 byte representation rather than IEEE ``!=``
        # so ``+0.0``/``-0.0`` divergences (Mistake Book Pattern around bit
        # identity) get counted faithfully.
        n_diff = _bytewise_unequal_double_count(cpu, jax_)
    else:
        max_abs = float("nan")
        argmax = None
        first_byte = None
        first_numeric = None
        n_diff = -1
    cpu_digest = cpu_record.sha256_float64_bytes if cpu_record is not None else None
    jax_digest = jax_record.sha256_float64_bytes if jax_record is not None else None
    byte_identical = (
        shape_match
        and dtype_match
        and cpu_digest is not None
        and jax_digest is not None
        and cpu_digest == jax_digest
    )
    return CensusArrayDiff(
        array_name=array_name,
        stage=stage,
        max_abs_diff=max_abs,
        argmax_abs_diff=argmax,
        first_unequal_byte_index=first_byte,
        first_unequal_numeric_index=first_numeric,
        n_bit_different_entries=n_diff,
        byte_identical=byte_identical,
        cpu_sha256_float64_bytes=cpu_digest,
        jax_sha256_float64_bytes=jax_digest,
        shape_match=shape_match,
        dtype_match=dtype_match,
    )


def compare_scalar(
    *,
    name: str,
    stage: str,
    cpu_record: CensusScalarRecord,
    jax_record: CensusScalarRecord,
) -> CensusScalarDiff:
    abs_diff = abs(jax_record.value - cpu_record.value)
    return CensusScalarDiff(
        name=name,
        stage=stage,
        cpu_value=cpu_record.value,
        jax_value=jax_record.value,
        abs_diff=abs_diff,
        byte_identical=(
            cpu_record.sha256_float64_bytes == jax_record.sha256_float64_bytes
        ),
    )


# ---------------------------------------------------------------------------
# NDJSON I/O


def write_ndjson(
    path: Path,
    records: Sequence[
        CensusArrayRecord
        | CensusArrayDiff
        | CensusScalarRecord
        | CensusScalarDiff
        | dict[str, Any]
    ],
) -> Path:
    """Persist census records as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for rec in records:
            payload = rec.to_json_record() if hasattr(rec, "to_json_record") else rec
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def _records_from_cpu_inputs(
    inputs: dict[str, Any],
    *,
    G: float,
    iota: float,
    weight_inv_modB: bool,
    stage: str,
) -> tuple[list[CensusArrayRecord], list[CensusScalarRecord]]:
    """Translate the CPU boundary-input dict into census records."""
    array_records = [
        build_array_record(
            array_name="gamma",
            producer="cpu",
            stage=stage,
            array=np.asarray(inputs["gamma"]),
        ),
        build_array_record(
            array_name="xphi",
            producer="cpu",
            stage=stage,
            array=np.asarray(inputs["xphi"]),
        ),
        build_array_record(
            array_name="xtheta",
            producer="cpu",
            stage=stage,
            array=np.asarray(inputs["xtheta"]),
        ),
        build_array_record(
            array_name="dx_ds",
            producer="cpu",
            stage=stage,
            array=np.asarray(inputs["dx_dc"]),
        ),
        build_array_record(
            array_name="dxphi_ds",
            producer="cpu",
            stage=stage,
            array=np.asarray(inputs["dxphi_dc"]),
        ),
        build_array_record(
            array_name="dxtheta_ds",
            producer="cpu",
            stage=stage,
            array=np.asarray(inputs["dxtheta_dc"]),
        ),
        build_array_record(
            array_name="B",
            producer="cpu",
            stage=stage,
            array=np.asarray(inputs["B"]),
        ),
        build_array_record(
            array_name="dB_dX",
            producer="cpu",
            stage=stage,
            array=np.asarray(inputs["dB_dx"]),
        ),
    ]
    scalar_records = [
        build_scalar_record(
            name="G_value", producer="cpu", stage=stage, value=float(G)
        ),
        build_scalar_record(
            name="iota", producer="cpu", stage=stage, value=float(iota)
        ),
        build_scalar_record(
            name="weight_inv_modB",
            producer="cpu",
            stage=stage,
            value=1.0 if weight_inv_modB else 0.0,
        ),
    ]
    return array_records, scalar_records


def capture_cpu_boozer_inputs(
    boozer_surface: "BoozerSurface",
    *,
    sdofs: np.ndarray,
    iota: float,
    G: float,
    weight_inv_modB: bool,
    stage: str = CENSUS_STAGE_DEFAULT,
) -> tuple[list[CensusArrayRecord], list[CensusScalarRecord]]:
    """Capture the CPU-side Boozer LS callback boundary arrays.

    Calls
    :meth:`simsopt.geo.boozersurface.BoozerSurface._boozer_penalty_vectorized_inputs`
    at the requested DOFs (``derivatives=1`` matches what the LS callback
    consumes for value+gradient) and translates the materialized numpy
    arrays into census records.

    Side effects: ``boozer_surface.surface.set_dofs(sdofs)`` and
    ``boozer_surface.biotsavart.compute(1)`` are invoked.
    """
    inputs = boozer_surface._boozer_penalty_vectorized_inputs(np.asarray(sdofs), 1)
    return _records_from_cpu_inputs(
        inputs,
        G=float(G),
        iota=float(iota),
        weight_inv_modB=bool(weight_inv_modB),
        stage=stage,
    )


def capture_jax_boozer_inputs(
    boozer_surface_jax: "BoozerSurfaceJAX",
    *,
    sdofs: np.ndarray,
    iota: float,
    G: float | None,
    weight_inv_modB: bool,
    optimize_G: bool,
    stage: str = CENSUS_STAGE_DEFAULT,
    parity_policy: str = "production",
) -> tuple[list[CensusArrayRecord], list[CensusScalarRecord]]:
    """Capture the JAX-side Boozer LS callback boundary arrays.

    Builds the decision vector ``x = [sdofs, iota, (G)]`` and invokes the
    private boundary helper
    :func:`simsopt.geo.boozersurface_jax._boozer_penalty_value_and_grad_inputs_cpu_ordered`
    using the same kwargs the production
    ``_make_penalty_value_and_grad_cpu_ordered_with`` factory uses. The JAX
    arrays are materialized to host numpy with ``jax.device_get`` so the
    bytes compared by the census reflect what SciPy / the Boozer residual
    routines actually see.

    Args:
        boozer_surface_jax: a ``BoozerSurfaceJAX`` configured with the same
            surface, coils, and label as the CPU counterpart.
        sdofs: surface DOF vector (without iota / G appended).
        iota: rotational transform value.
        G: poloidal current; required when ``optimize_G`` is True. Pass
            ``None`` when ``optimize_G`` is False — the helper will compute
            G from coil currents via :func:`compute_G_from_currents`.
        weight_inv_modB: weight by 1/|B| at the residual.
        optimize_G: whether G is part of the decision vector.
        stage: census ``stage`` tag (default
            ``"boozer_ls_callback_input"``).
    """
    import jax
    import jax.numpy as jnp

    from simsopt.geo.boozersurface_jax import (  # noqa: PLC0415 - diagnostic
        _boozer_penalty_value_and_grad_inputs_cpu_ordered,
        _hostify_tree,
        _resolved_coil_set_spec,
    )

    if optimize_G and G is None:
        raise ValueError("G must be provided when optimize_G=True")
    sdofs_arr = np.asarray(sdofs, dtype=np.float64)
    pieces = [sdofs_arr, np.asarray([float(iota)], dtype=np.float64)]
    if optimize_G:
        pieces.append(np.asarray([float(G)], dtype=np.float64))
    x = jnp.asarray(np.concatenate(pieces), dtype=jnp.float64)

    instance = boozer_surface_jax
    coil_set_spec = _hostify_tree(
        _resolved_coil_set_spec(
            instance.coil_set_spec, coil_arrays=None, coil_set_spec=None
        )
    )
    _, geometry, _, inputs = _boozer_penalty_value_and_grad_inputs_cpu_ordered(
        x,
        coil_arrays=None,
        coil_set_spec=coil_set_spec,
        quadpoints_phi=_hostify_tree(instance.quadpoints_phi),
        quadpoints_theta=_hostify_tree(instance.quadpoints_theta),
        mpol=instance.mpol,
        ntor=instance.ntor,
        nfp=instance.nfp,
        stellsym=instance.stellsym,
        scatter_indices=_hostify_tree(instance.scatter_indices),
        surface_kind=instance._surface_geometry_kind,
        optimize_G=optimize_G,
        parity_policy=parity_policy,
    )

    def _to_np(value: Any) -> np.ndarray:
        host = jax.device_get(value)
        return np.asarray(host)

    gamma_host = _to_np(geometry.gamma)
    array_records = [
        build_array_record(
            array_name="gamma", producer="jax", stage=stage, array=gamma_host
        ),
        build_array_record(
            array_name="xphi",
            producer="jax",
            stage=stage,
            array=_to_np(inputs.xphi),
        ),
        build_array_record(
            array_name="xtheta",
            producer="jax",
            stage=stage,
            array=_to_np(inputs.xtheta),
        ),
        build_array_record(
            array_name="dx_ds",
            producer="jax",
            stage=stage,
            array=_to_np(inputs.dx_ds),
        ),
        build_array_record(
            array_name="dxphi_ds",
            producer="jax",
            stage=stage,
            array=_to_np(inputs.dxphi_ds),
        ),
        build_array_record(
            array_name="dxtheta_ds",
            producer="jax",
            stage=stage,
            array=_to_np(inputs.dxtheta_ds),
        ),
        build_array_record(
            array_name="B",
            producer="jax",
            stage=stage,
            array=_to_np(inputs.B),
        ),
        build_array_record(
            array_name="dB_dX",
            producer="jax",
            stage=stage,
            array=_to_np(inputs.dB_dX),
        ),
    ]
    G_value_host = float(_to_np(inputs.G_value))
    iota_host = float(_to_np(inputs.iota))
    scalar_records = [
        build_scalar_record(
            name="G_value", producer="jax", stage=stage, value=G_value_host
        ),
        build_scalar_record(name="iota", producer="jax", stage=stage, value=iota_host),
        build_scalar_record(
            name="weight_inv_modB",
            producer="jax",
            stage=stage,
            value=1.0 if weight_inv_modB else 0.0,
        ),
    ]
    return array_records, scalar_records


def compare_boundary_inputs(
    *,
    cpu_array_records: Sequence[CensusArrayRecord],
    cpu_scalar_records: Sequence[CensusScalarRecord],
    jax_array_records: Sequence[CensusArrayRecord],
    jax_scalar_records: Sequence[CensusScalarRecord],
    cpu_arrays: dict[str, np.ndarray],
    jax_arrays: dict[str, np.ndarray],
) -> list[CensusArrayDiff | CensusScalarDiff]:
    """Pair CPU/JAX records by name and emit diffs.

    Args:
        cpu_array_records: producer="cpu" array records (one per name).
        jax_array_records: producer="jax" array records (one per name).
        cpu_scalar_records: producer="cpu" scalar records.
        jax_scalar_records: producer="jax" scalar records.
        cpu_arrays: numpy arrays keyed by canonical array name (CPU side).
            Must include every name in :data:`CENSUS_BOUNDARY_ARRAY_ORDER`
            that the producer captured.
        jax_arrays: numpy arrays keyed by canonical array name (JAX side).

    Returns:
        Ordered list of :class:`CensusArrayDiff` and
        :class:`CensusScalarDiff`. Array diffs come first in the canonical
        ladder order; scalar diffs come last.
    """
    cpu_array_index = {rec.array_name: rec for rec in cpu_array_records}
    jax_array_index = {rec.array_name: rec for rec in jax_array_records}
    diffs: list[CensusArrayDiff | CensusScalarDiff] = []
    for name in CENSUS_BOUNDARY_ARRAY_ORDER:
        if name not in cpu_array_index or name not in jax_array_index:
            continue
        diffs.append(
            compare_array(
                array_name=name,
                stage=cpu_array_index[name].stage,
                cpu=cpu_arrays[name],
                jax_=jax_arrays[name],
                cpu_record=cpu_array_index[name],
                jax_record=jax_array_index[name],
            )
        )
    cpu_scalar_index = {rec.name: rec for rec in cpu_scalar_records}
    jax_scalar_index = {rec.name: rec for rec in jax_scalar_records}
    for name in CENSUS_BOUNDARY_SCALAR_ORDER:
        cpu_rec = cpu_scalar_index.get(name)
        jax_rec = jax_scalar_index.get(name)
        if cpu_rec is None or jax_rec is None:
            continue
        diffs.append(
            compare_scalar(
                name=name,
                stage=cpu_rec.stage,
                cpu_record=cpu_rec,
                jax_record=jax_rec,
            )
        )
    return diffs


def first_divergence(
    diffs: Sequence[CensusArrayDiff | CensusScalarDiff],
    *,
    array_order: Sequence[str] = CENSUS_BOUNDARY_ARRAY_ORDER,
) -> CensusArrayDiff | CensusScalarDiff | None:
    """Return the first non-byte-identical diff in the canonical ladder order.

    Array diffs sort according to ``array_order``; scalars come last to match
    the dependency ladder (scalars are inputs to the array kernels, but in
    practice ``G`` and ``iota`` are constants per outer step).
    """
    array_diffs: dict[str, CensusArrayDiff] = {}
    scalar_diffs: list[CensusScalarDiff] = []
    for diff in diffs:
        if isinstance(diff, CensusArrayDiff):
            array_diffs[diff.array_name] = diff
        elif isinstance(diff, CensusScalarDiff):
            scalar_diffs.append(diff)
    for name in array_order:
        diff = array_diffs.get(name)
        if diff is not None and not diff.byte_identical:
            return diff
    for diff in scalar_diffs:
        if not diff.byte_identical:
            return diff
    return None
