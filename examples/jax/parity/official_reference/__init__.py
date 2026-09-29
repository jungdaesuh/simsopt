"""Tracked official-reference numbers for the 25 upstream example mirrors.

The JSON files under ``9e027eac3/`` are the ONE tracked home for numbers measured on the official simsopt build at
upstream commit ``9e027eac38028d57aa23777be52a781aa860e347``.  Tests and parity cases read them through this loader
instead of pasting literals or reading the git-ignored campaign capture tree, so the comparison against upstream
runs on a clean checkout.

Reading a fixture imports nothing but the standard library and numpy: never simsopt, never jax.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np

#: Upstream commit every fixture file was measured at.
UPSTREAM_COMMIT: Final[str] = "9e027eac38028d57aa23777be52a781aa860e347"

#: Directory holding the per-case JSON files, named by the short upstream commit.
REFERENCE_ROOT: Final[Path] = Path(__file__).resolve().parent / UPSTREAM_COMMIT[:9]

#: Directory holding the per-case SENSITIVITY records (upstream's own one-ulp start-perturbation samples).
SENSITIVITY_ROOT: Final[Path] = REFERENCE_ROOT / "sensitivity"

#: An array with at most this many elements is stored element-exact; a larger one is stored as a digest only.
INLINE_ELEMENT_LIMIT: Final[int] = 1024

#: Schema version of the per-case JSON files.
SCHEMA_VERSION: Final[int] = 1

#: Name of the record stored at the top level of a fixture file: the official run at the script's shipped scale.
#: Every other captured run of the same official script lives under the optional ``variants`` key.
CANONICAL_VARIANT: Final[str] = "canonical"

#: Single-key object that stands in for a non-finite float, which JSON cannot spell portably.
NONFINITE_KEY: Final[str] = "$nonfinite"

_NONFINITE_TO_FLOAT: Final[dict[str, float]] = {
    "inf": math.inf,
    "-inf": -math.inf,
    "nan": math.nan,
}

#: A JSON scalar as it appears in a fixture file after decoding.
ScalarValue = float | int | bool | str | None

#: Any decoded JSON value; ``Mapping``/``Sequence`` members are themselves ``JsonValue``.
JsonValue = ScalarValue | Mapping[str, object] | Sequence[object]


def encode_nonfinite(value: object) -> object:
    """Replace every non-finite float in ``value`` by the sentinel object, recursively."""
    if isinstance(value, Mapping):
        return {str(key): encode_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_nonfinite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {NONFINITE_KEY: "nan"}
        return {NONFINITE_KEY: "inf" if value > 0.0 else "-inf"}
    return value


def decode_nonfinite(value: object) -> JsonValue:
    """Invert :func:`encode_nonfinite`."""
    if isinstance(value, Mapping):
        if len(value) == 1 and NONFINITE_KEY in value:
            return _NONFINITE_TO_FLOAT[str(value[NONFINITE_KEY])]
        return {str(key): decode_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_nonfinite(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    raise TypeError(f"{type(value).__name__} is not a JSON value")


def decode_scalar(value: object) -> ScalarValue:
    """Decode one JSON scalar, undoing the non-finite sentinel."""
    decoded = decode_nonfinite(value)
    if isinstance(decoded, (bool, int, float, str)) or decoded is None:
        return decoded
    raise TypeError(f"expected a JSON scalar, got {type(decoded).__name__}")


def decode_mapping(value: object) -> dict[str, JsonValue]:
    """Decode one JSON object, undoing the non-finite sentinel in every leaf."""
    decoded = decode_nonfinite(value)
    if not isinstance(decoded, Mapping):
        raise TypeError(f"expected a JSON object, got {type(decoded).__name__}")
    return dict(decoded)


def decode_scalar_mapping(value: object) -> dict[str, ScalarValue]:
    """Decode one JSON object whose values are all scalars."""
    return {key: decode_scalar(item) for key, item in decode_mapping(value).items()}


def _as_float(value: object) -> float:
    decoded = decode_scalar(value)
    if isinstance(decoded, (bool, str)) or decoded is None:
        raise TypeError(f"expected a number, got {type(decoded).__name__}")
    return float(decoded)


def array_sha256(values: np.ndarray) -> str:
    """sha256 of the C-contiguous little-endian float64 bytes of ``values``."""
    return hashlib.sha256(
        np.ascontiguousarray(values, dtype="<f8").tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class ArrayDigest:
    """Shape-and-content summary of one array observable, carried whether or not the array is stored inline."""

    shape: tuple[int, ...]
    dtype: str
    count: int
    sha256: str
    min: float
    max: float
    sum: float
    l2: float

    @classmethod
    def of(cls, values: np.ndarray) -> ArrayDigest:
        """Digest of ``values``. An empty array has no extremum, so ``min``/``max`` are NaN there."""
        flat = np.ascontiguousarray(values, dtype="<f8").reshape(-1)
        empty = flat.size == 0
        return cls(
            shape=tuple(int(extent) for extent in values.shape),
            dtype=str(values.dtype),
            count=int(values.size),
            sha256=array_sha256(values),
            min=math.nan if empty else float(np.min(flat)),
            max=math.nan if empty else float(np.max(flat)),
            sum=float(np.sum(flat)),
            l2=float(np.sqrt(np.sum(flat * flat))),
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "count": self.count,
            "sha256": self.sha256,
            "min": encode_nonfinite(self.min),
            "max": encode_nonfinite(self.max),
            "sum": encode_nonfinite(self.sum),
            "l2": encode_nonfinite(self.l2),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ArrayDigest:
        return cls(
            shape=tuple(int(extent) for extent in payload["shape"]),
            dtype=str(payload["dtype"]),
            count=int(payload["count"]),
            sha256=str(payload["sha256"]),
            min=_as_float(payload["min"]),
            max=_as_float(payload["max"]),
            sum=_as_float(payload["sum"]),
            l2=_as_float(payload["l2"]),
        )


@dataclass(frozen=True)
class ProviderCall:
    """One optimizer call the official run made, with the terminal facts its result reported."""

    index: int
    function: str
    method: str | None
    options: Mapping[str, JsonValue]
    tol: float | None
    result: Mapping[str, ScalarValue]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ProviderCall:
        method = payload["method"]
        tol = payload["tol"]
        return cls(
            index=int(payload["index"]),
            function=str(payload["function"]),
            method=None if method is None else str(method),
            options=decode_mapping(payload["options"]),
            tol=None if tol is None else _as_float(tol),
            result=decode_scalar_mapping(payload["result"]),
        )


@dataclass(frozen=True)
class CaptureProvenance:
    """Where the numbers came from, so a stale fixture is visible without rerunning anything."""

    path: str
    capture_json_sha256: str
    array_file_sha256: str | None
    threads: Mapping[str, str]
    python: str | None
    numpy: str | None
    scipy: str | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CaptureProvenance:
        array_file_sha256 = payload["array_file_sha256"]
        python = payload["python"]
        numpy_version = payload["numpy"]
        scipy_version = payload["scipy"]
        return cls(
            path=str(payload["path"]),
            capture_json_sha256=str(payload["capture_json_sha256"]),
            array_file_sha256=None
            if array_file_sha256 is None
            else str(array_file_sha256),
            threads={
                str(name): str(value)
                for name, value in dict(payload["threads"]).items()
            },
            python=None if python is None else str(python),
            numpy=None if numpy_version is None else str(numpy_version),
            scipy=None if scipy_version is None else str(scipy_version),
        )


class MissingObservableError(KeyError):
    """Raised when a fixture has no observable under the requested key."""


class ObservableKindError(TypeError):
    """Raised when an observable exists but is not the kind the accessor returns."""


class MissingVariantError(KeyError):
    """Raised when a fixture has no record for the requested variant."""


@dataclass(frozen=True)
class OfficialReference:
    """The official numbers of one case and one captured run of it, loaded from its tracked JSON file."""

    case_id: str
    variant: str
    upstream_commit: str
    official_script: str
    official_script_sha256: str
    capture: CaptureProvenance
    provider_calls: tuple[ProviderCall, ...]
    _observables: Mapping[str, Mapping[str, object]]

    def keys(self) -> tuple[str, ...]:
        """Every observable key, sorted."""
        return tuple(sorted(self._observables))

    def has(self, key: str) -> bool:
        return key in self._observables

    def kind(self, key: str) -> str:
        """``"scalar"``, ``"array"`` or ``"structure"``."""
        return str(self._entry(key)["kind"])

    def is_inline(self, key: str) -> bool:
        """True when the stored value is element-exact rather than a digest only."""
        entry = self._entry(key)
        return entry["kind"] != "array" or "values" in entry

    def scalar(self, key: str) -> ScalarValue:
        entry = self._require(key, "scalar")
        return decode_scalar(entry["value"])

    def array(self, key: str) -> np.ndarray:
        """The exact array. Raises when the fixture kept only a digest for this key."""
        return _inline_array(
            self._require(key, "array"),
            f"{key!r} of case {self.case_id!r} ({self.variant})",
        )

    def digest(self, key: str) -> ArrayDigest:
        return ArrayDigest.from_payload(self._require(key, "array"))

    def structure(self, key: str) -> JsonValue:
        entry = self._require(key, "structure")
        return decode_nonfinite(entry["value"])

    def _entry(self, key: str) -> Mapping[str, object]:
        if key not in self._observables:
            raise MissingObservableError(
                f"case {self.case_id!r} ({self.variant}) has no observable {key!r}; "
                f"available keys: {list(self.keys())}"
            )
        return self._observables[key]

    def _require(self, key: str, kind: str) -> Mapping[str, object]:
        entry = self._entry(key)
        if entry["kind"] != kind:
            raise ObservableKindError(
                f"observable {key!r} of case {self.case_id!r} ({self.variant}) is a {entry['kind']}, not a {kind}"
            )
        return entry


def _inline_array(entry: Mapping[str, object], label: str) -> np.ndarray:
    """Decode one element-exact array entry; a digest-only entry is refused."""
    if "values" not in entry:
        raise ObservableKindError(
            f"observable {label} has {entry['count']} elements, above the inline limit "
            f"{INLINE_ELEMENT_LIMIT}; only a digest is stored"
        )
    values = np.asarray(decode_nonfinite(entry["values"]), dtype=str(entry["dtype"]))
    return values.reshape(tuple(int(extent) for extent in entry["shape"]))


def reference_path(case_id: str) -> Path:
    """Path of the tracked JSON file for ``case_id`` (it need not exist)."""
    return REFERENCE_ROOT / f"{case_id}.json"


def official_case_ids() -> tuple[str, ...]:
    """Sorted case ids the fixture set covers."""
    return tuple(sorted(path.stem for path in REFERENCE_ROOT.glob("*.json")))


def _payload(case_id: str) -> Mapping[str, object]:
    path = reference_path(case_id)
    if not path.is_file():
        raise MissingObservableError(
            f"no official reference for case {case_id!r} at {path}; available cases: {list(official_case_ids())}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _variants(payload: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    stored = payload.get("variants", {})
    if not isinstance(stored, Mapping):
        raise TypeError("the 'variants' key must be a JSON object")
    return {str(name): record for name, record in stored.items()}


def official_variants(case_id: str) -> tuple[str, ...]:
    """Captured runs this case has, canonical first, the rest sorted."""
    return (CANONICAL_VARIANT, *sorted(_variants(_payload(case_id))))


def load_official_reference(
    case_id: str, *, variant: str = CANONICAL_VARIANT
) -> OfficialReference:
    """Load one captured run of ``case_id``; the default is the canonical, shipped-scale official run."""
    payload = _payload(case_id)
    if variant == CANONICAL_VARIANT:
        record: Mapping[str, object] = payload
    else:
        records = _variants(payload)
        if variant not in records:
            raise MissingVariantError(
                f"case {case_id!r} has no {variant!r} record; available variants: "
                f"{list(official_variants(case_id))}"
            )
        record = records[variant]
    return OfficialReference(
        case_id=str(payload["case_id"]),
        variant=variant,
        upstream_commit=str(payload["upstream_commit"]),
        official_script=str(payload["official_script"]),
        official_script_sha256=str(payload["official_script_sha256"]),
        capture=CaptureProvenance.from_payload(record["capture"]),
        provider_calls=tuple(
            ProviderCall.from_payload(call) for call in record["provider_calls"]
        ),
        _observables=record["observables"],
    )


@dataclass(frozen=True)
class SensitivityProtocol:
    """The pre-registered protocol the sensitivity runs followed. Fixed before any sample existed."""

    text: str
    perturbation_rule: str
    seed_rule: str
    perturbed_call_index: int
    unperturbed_k: int
    pre_registered_in: str
    threads: Mapping[str, str]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> SensitivityProtocol:
        return cls(
            text=str(payload["text"]),
            perturbation_rule=str(payload["perturbation_rule"]),
            seed_rule=str(payload["seed_rule"]),
            perturbed_call_index=int(payload["perturbed_call_index"]),
            unperturbed_k=int(payload["unperturbed_k"]),
            pre_registered_in=str(payload["pre_registered_in"]),
            threads={
                str(name): str(value)
                for name, value in dict(payload["threads"]).items()
            },
        )


@dataclass(frozen=True)
class ProviderOutcome:
    """What one provider call of one sensitivity run reported."""

    index: int
    method: str | None
    status: int
    success: bool
    nit: int
    nfev: int
    njev: int
    message: str
    fun: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ProviderOutcome:
        method = payload["method"]
        return cls(
            index=int(payload["index"]),
            method=None if method is None else str(method),
            status=int(payload["status"]),
            success=bool(payload["success"]),
            nit=int(payload["nit"]),
            nfev=int(payload["nfev"]),
            njev=int(payload["njev"]),
            message=str(payload["message"]),
            fun=_as_float(payload["fun"]),
        )


@dataclass(frozen=True)
class SensitivityRun:
    """One run of the unmodified official script: ``k = 0`` unperturbed, ``k = 1..8`` one ulp off the start."""

    k: int
    end_value: float
    provider_calls: tuple[ProviderOutcome, ...]
    capture_sha256: str
    perturbation_sha256: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> SensitivityRun:
        return cls(
            k=int(payload["k"]),
            end_value=_as_float(payload["end_value"]),
            provider_calls=tuple(
                ProviderOutcome.from_payload(call) for call in payload["provider_calls"]
            ),
            capture_sha256=str(payload["capture_sha256"]),
            perturbation_sha256=str(payload["perturbation_sha256"]),
        )


@dataclass(frozen=True)
class OfficialSensitivity:
    """Upstream's own end-point scatter for one case. Raw numbers only: no band, no ceiling, no derived bound."""

    case_id: str
    upstream_commit: str
    official_script: str
    official_script_sha256: str
    observable: str
    """Upstream's own key of the sampled end value in its capture (e.g. ``stage2:final:objective``)."""
    lane_observable: str
    """The port lane's key of the same quantity, which a quality band judges (e.g. ``final:objective``)."""
    protocol: SensitivityProtocol
    runs: tuple[SensitivityRun, ...]

    @property
    def end_values(self) -> tuple[float, ...]:
        """The end value of every run, in run order (k ascending)."""
        return tuple(run.end_value for run in self.runs)

    def run(self, k: int) -> SensitivityRun:
        for run in self.runs:
            if run.k == k:
                return run
        raise MissingObservableError(
            f"case {self.case_id!r} has no sensitivity run k={k}; available: {[r.k for r in self.runs]}"
        )


def sensitivity_path(case_id: str) -> Path:
    """Path of the tracked sensitivity file for ``case_id`` (it need not exist)."""
    return SENSITIVITY_ROOT / f"{case_id}.json"


def official_sensitivity_case_ids() -> tuple[str, ...]:
    """Sorted case ids that have a tracked sensitivity record."""
    return tuple(sorted(path.stem for path in SENSITIVITY_ROOT.glob("*.json")))


def load_official_sensitivity(case_id: str) -> OfficialSensitivity:
    """Load the sensitivity record of ``case_id``."""
    path = sensitivity_path(case_id)
    if not path.is_file():
        raise MissingObservableError(
            f"no official sensitivity record for case {case_id!r} at {path}; "
            f"available cases: {list(official_sensitivity_case_ids())}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OfficialSensitivity(
        case_id=str(payload["case_id"]),
        upstream_commit=str(payload["upstream_commit"]),
        official_script=str(payload["official_script"]),
        official_script_sha256=str(payload["official_script_sha256"]),
        observable=str(payload["observable"]),
        lane_observable=str(payload["lane_observable"]),
        protocol=SensitivityProtocol.from_payload(payload["protocol"]),
        runs=tuple(SensitivityRun.from_payload(run) for run in payload["runs"]),
    )


#: Directory holding the per-case TRACING scatter records: upstream's own one-ulp start-perturbation samples of the
#: three tracing mirrors, whose observable is a whole set of trajectories rather than one optimizer end value.
TRACING_ROOT: Final[Path] = REFERENCE_ROOT / "tracing"

#: The scatter quantities EVERY tracing sampling run recorded against the unperturbed run, in the order the
#: investigation report lists them: (a) status changes, (b) hit-count differences, (c) final-time difference,
#: (d) final-state distances, (e) hit-to-section geometry.
TRACING_SCATTER_QUANTITIES: Final[tuple[str, ...]] = (
    "status_changes",
    "hit_count_difference_per_line_max_abs",
    "hit_count_difference_per_plane_max_abs",
    "final_time_difference_max",
    "final_state_distance_max",
    "final_position_distance_max",
    "geometry_max_over_groups",
    "geometry_max_of_group_medians",
)

#: Scatter quantities a case records only when the object it traces HAS the quantity: (f) the guiding-centre case's
#: final parallel-speed fraction, which a field line does not have.  A record carries such a key when, and only when,
#: its case measured it, so the RECORD is the one place that knows which case has which quantity: nothing downstream
#: keeps a second per-case table.
TRACING_SCATTER_CASE_QUANTITIES: Final[tuple[str, ...]] = (
    "final_parallel_speed_fraction_difference_max",
)


def _as_int(value: object) -> int:
    decoded = decode_scalar(value)
    if isinstance(decoded, bool) or not isinstance(decoded, int):
        raise TypeError(f"expected an integer, got {type(decoded).__name__}")
    return decoded


@dataclass(frozen=True)
class TracingScatterProtocol:
    """The pre-registered tracing sampling protocol. Fixed before any tracing sample existed."""

    text: str
    perturbation_rule: str
    seed_rule: str
    perturbed_arguments: tuple[str, ...]
    unperturbed_k: int
    pre_registered_in: str
    threads: Mapping[str, str]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> TracingScatterProtocol:
        return cls(
            text=str(payload["text"]),
            perturbation_rule=str(payload["perturbation_rule"]),
            seed_rule=str(payload["seed_rule"]),
            perturbed_arguments=tuple(
                str(name) for name in list(payload["perturbed_arguments"])
            ),
            unperturbed_k=_as_int(payload["unperturbed_k"]),
            pre_registered_in=str(payload["pre_registered_in"]),
            threads={
                str(name): str(value)
                for name, value in dict(payload["threads"]).items()
            },
        )


@dataclass(frozen=True)
class TracingScatter:
    """What one perturbed tracing run moved, measured against the unperturbed run of the same case."""

    status_changes: int
    hit_count_difference_per_line_max_abs: int
    hit_count_difference_per_plane_max_abs: int
    final_time_difference_max: float
    final_state_distance_max: float
    final_position_distance_max: float
    geometry_max_over_groups: float
    geometry_max_of_group_medians: float
    final_parallel_speed_fraction_difference_max: float | None = None

    def as_mapping(self) -> dict[str, float]:
        """Every quantity this record holds, in :data:`TRACING_SCATTER_QUANTITIES` order.

        A member of :data:`TRACING_SCATTER_CASE_QUANTITIES` follows the universal ones and appears only when the
        case measured it, so ``tuple(as_mapping())`` is the record's own quantity list.
        """
        values = {
            "status_changes": self.status_changes,
            "hit_count_difference_per_line_max_abs": self.hit_count_difference_per_line_max_abs,
            "hit_count_difference_per_plane_max_abs": self.hit_count_difference_per_plane_max_abs,
            "final_time_difference_max": self.final_time_difference_max,
            "final_state_distance_max": self.final_state_distance_max,
            "final_position_distance_max": self.final_position_distance_max,
            "geometry_max_over_groups": self.geometry_max_over_groups,
            "geometry_max_of_group_medians": self.geometry_max_of_group_medians,
        }
        if self.final_parallel_speed_fraction_difference_max is not None:
            values["final_parallel_speed_fraction_difference_max"] = (
                self.final_parallel_speed_fraction_difference_max
            )
        return values

    def quantity(self, name: str) -> float:
        values = self.as_mapping()
        if name not in values:
            raise MissingObservableError(
                f"no tracing scatter quantity {name!r}; available: {list(values)}"
            )
        return values[name]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> TracingScatter:
        return cls(
            status_changes=_as_int(payload["status_changes"]),
            hit_count_difference_per_line_max_abs=_as_int(
                payload["hit_count_difference_per_line_max_abs"]
            ),
            hit_count_difference_per_plane_max_abs=_as_int(
                payload["hit_count_difference_per_plane_max_abs"]
            ),
            final_time_difference_max=_as_float(payload["final_time_difference_max"]),
            final_state_distance_max=_as_float(payload["final_state_distance_max"]),
            final_position_distance_max=_as_float(
                payload["final_position_distance_max"]
            ),
            geometry_max_over_groups=_as_float(payload["geometry_max_over_groups"]),
            geometry_max_of_group_medians=_as_float(
                payload["geometry_max_of_group_medians"]
            ),
            final_parallel_speed_fraction_difference_max=(
                _as_float(payload["final_parallel_speed_fraction_difference_max"])
                if "final_parallel_speed_fraction_difference_max" in payload
                else None
            ),
        )


@dataclass(frozen=True)
class TracingPerturbation:
    """What the sampling actually moved in one run; every field is ``None`` for the unperturbed run."""

    seed: int
    perturbed_arguments: tuple[str, ...] | None
    entries: int | None
    entries_changed: int | None
    max_abs_delta: float | None
    max_rel_delta: float | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> TracingPerturbation:
        arguments = payload["perturbed_arguments"]
        entries = payload["entries"]
        entries_changed = payload["entries_changed"]
        max_abs_delta = payload["max_abs_delta"]
        max_rel_delta = payload["max_rel_delta"]
        return cls(
            seed=_as_int(payload["seed"]),
            perturbed_arguments=None
            if arguments is None
            else tuple(str(name) for name in list(arguments)),
            entries=None if entries is None else _as_int(entries),
            entries_changed=None
            if entries_changed is None
            else _as_int(entries_changed),
            max_abs_delta=None if max_abs_delta is None else _as_float(max_abs_delta),
            max_rel_delta=None if max_rel_delta is None else _as_float(max_rel_delta),
        )


@dataclass(frozen=True)
class TracingScatterRun:
    """One run of the official tracing script: ``k = 0`` unperturbed, ``k = 1..8`` one ulp off the start data."""

    k: int
    capture_sha256: str
    perturbation_sha256: str
    perturbation: TracingPerturbation
    scatter: TracingScatter | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> TracingScatterRun:
        scatter = payload["scatter"]
        return cls(
            k=_as_int(payload["k"]),
            capture_sha256=str(payload["capture_sha256"]),
            perturbation_sha256=str(payload["perturbation_sha256"]),
            perturbation=TracingPerturbation.from_payload(
                decode_mapping(payload["perturbation"])
            ),
            scatter=None
            if scatter is None
            else TracingScatter.from_payload(decode_mapping(scatter)),
        )


@dataclass(frozen=True)
class UnperturbedTrace:
    """The absolute facts of the unperturbed run, which reproduces the canonical capture bitwise."""

    final_status_histogram: Mapping[str, int]
    poincare_hits_total: int
    observable_sha256: Mapping[str, str]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> UnperturbedTrace:
        return cls(
            final_status_histogram={
                str(status): _as_int(count)
                for status, count in dict(payload["final_status_histogram"]).items()
            },
            poincare_hits_total=_as_int(payload["poincare_hits_total"]),
            observable_sha256={
                str(key): str(value)
                for key, value in dict(payload["observable_sha256"]).items()
            },
        )


@dataclass(frozen=True)
class OfficialTracingScatter:
    """Upstream's own trajectory scatter for one tracing case. Raw numbers only: no ceiling, no derived bound."""

    case_id: str
    upstream_commit: str
    official_script: str
    official_script_sha256: str
    lines: int
    protocol: TracingScatterProtocol
    unperturbed: UnperturbedTrace
    runs: tuple[TracingScatterRun, ...]
    maxima: TracingScatter

    def run(self, k: int) -> TracingScatterRun:
        for run in self.runs:
            if run.k == k:
                return run
        raise MissingObservableError(
            f"case {self.case_id!r} has no tracing run k={k}; available: {[r.k for r in self.runs]}"
        )

    @property
    def perturbed_runs(self) -> tuple[TracingScatterRun, ...]:
        """Every run but the unperturbed one, in run order."""
        return tuple(run for run in self.runs if run.k != self.protocol.unperturbed_k)


def tracing_scatter_path(case_id: str) -> Path:
    """Path of the tracked tracing scatter file for ``case_id`` (it need not exist)."""
    return TRACING_ROOT / f"{case_id}.json"


def official_tracing_scatter_case_ids() -> tuple[str, ...]:
    """Sorted case ids that have a tracked tracing scatter record."""
    return tuple(sorted(path.stem for path in TRACING_ROOT.glob("*.json")))


def load_official_tracing_scatter(case_id: str) -> OfficialTracingScatter:
    """Load the tracing scatter record of ``case_id``."""
    path = tracing_scatter_path(case_id)
    if not path.is_file():
        raise MissingObservableError(
            f"no official tracing scatter record for case {case_id!r} at {path}; "
            f"available cases: {list(official_tracing_scatter_case_ids())}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OfficialTracingScatter(
        case_id=str(payload["case_id"]),
        upstream_commit=str(payload["upstream_commit"]),
        official_script=str(payload["official_script"]),
        official_script_sha256=str(payload["official_script_sha256"]),
        lines=_as_int(payload["lines"]),
        protocol=TracingScatterProtocol.from_payload(payload["protocol"]),
        unperturbed=UnperturbedTrace.from_payload(payload["unperturbed"]),
        runs=tuple(TracingScatterRun.from_payload(run) for run in payload["runs"]),
        maxima=TracingScatter.from_payload(payload["maxima_over_k"]),
    )


#: Directory holding UPSTREAM's own end-state scatter at the parity harness's own scales: one file per case and scale,
#: ``<case_id>.<scale>.json``. Unlike ``sensitivity/`` (one end value per run of the unmodified official script), a
#: scatter record keeps every end-state observable a contract judges, and at a reduced scale its runner is DERIVED:
#: the official body with only the scale lines changed, the diff stored in the record.
UPSTREAM_SCATTER_ROOT: Final[Path] = REFERENCE_ROOT / "scatter"


@dataclass(frozen=True)
class UpstreamScatterRunner:
    """Which bytes ran: the official body verbatim, or the official body with the recorded scale edits."""

    kind: str
    body_sha256: str
    body_diff: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> UpstreamScatterRunner:
        kind = str(payload["kind"])
        if kind not in ("verbatim", "derived"):
            raise ValueError(f"unknown upstream scatter runner kind {kind!r}")
        body_diff = str(payload["body_diff"])
        if (kind == "verbatim") != (body_diff == ""):
            raise ValueError(
                "a verbatim runner has no body diff and a derived runner has one"
            )
        return cls(
            kind=kind, body_sha256=str(payload["body_sha256"]), body_diff=body_diff
        )


@dataclass(frozen=True)
class UpstreamScatterRun:
    """One upstream run at one start: its provider outcomes and its end-state observables under LANE keys."""

    k: int
    workflow_success: bool
    """Upstream's own success flags of every stage the record names were all true."""
    provider_calls: tuple[ProviderOutcome, ...]
    capture_sha256: str
    perturbation_sha256: str
    _values: Mapping[str, Mapping[str, object]]

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._values))

    def value(self, lane_key: str) -> np.ndarray:
        """The end-state observable as an FP64 array (0-d for a scalar), element-exact."""
        if lane_key not in self._values:
            raise MissingObservableError(
                f"upstream scatter run k={self.k} has no observable {lane_key!r}; available: {list(self.keys())}"
            )
        entry = self._values[lane_key]
        if entry["kind"] == "scalar":
            return np.asarray(_as_float(entry["value"]), dtype=np.float64)
        if entry["kind"] == "array":
            return np.asarray(
                _inline_array(entry, f"{lane_key!r} of run k={self.k}"),
                dtype=np.float64,
            )
        raise ObservableKindError(
            f"observable {lane_key!r} of run k={self.k} is a {entry['kind']}"
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> UpstreamScatterRun:
        return cls(
            k=_as_int(payload["k"]),
            workflow_success=bool(payload["workflow_success"]),
            provider_calls=tuple(
                ProviderOutcome.from_payload(call) for call in payload["provider_calls"]
            ),
            capture_sha256=str(payload["capture_sha256"]),
            perturbation_sha256=str(payload["perturbation_sha256"]),
            _values=MappingProxyType(
                {
                    str(key): dict(entry)
                    for key, entry in dict(payload["values"]).items()
                }
            ),
        )


@dataclass(frozen=True)
class OfficialUpstreamScatter:
    """Upstream's own end states at one of the harness's scales. Raw numbers only: no band, no set, no bound."""

    case_id: str
    scale: str
    upstream_commit: str
    official_script: str
    official_script_sha256: str
    runner: UpstreamScatterRunner
    protocol: SensitivityProtocol
    capture_keys: Mapping[str, str]
    """Lane key -> upstream's capture key of the same quantity."""
    runs: tuple[UpstreamScatterRun, ...]

    def run(self, k: int) -> UpstreamScatterRun:
        for run in self.runs:
            if run.k == k:
                return run
        raise MissingObservableError(
            f"case {self.case_id!r} ({self.scale}) has no upstream scatter run k={k}; "
            f"available: {[r.k for r in self.runs]}"
        )


def upstream_scatter_path(case_id: str, scale: str) -> Path:
    """Path of the tracked upstream scatter file of ``case_id`` at ``scale`` (it need not exist)."""
    return UPSTREAM_SCATTER_ROOT / f"{case_id}.{scale}.json"


def upstream_scatter_records() -> tuple[tuple[str, str], ...]:
    """Sorted ``(case_id, scale)`` pairs that have a tracked upstream scatter record."""
    return tuple(
        sorted(
            tuple(path.name.removesuffix(".json").rsplit(".", maxsplit=1))
            for path in UPSTREAM_SCATTER_ROOT.glob("*.json")
        )
    )


def load_upstream_scatter(case_id: str, scale: str) -> OfficialUpstreamScatter:
    """Load upstream's own end-state scatter of ``case_id`` at ``scale``."""
    path = upstream_scatter_path(case_id, scale)
    if not path.is_file():
        raise MissingObservableError(
            f"no upstream scatter record for case {case_id!r} at scale {scale!r} ({path}); "
            f"available: {list(upstream_scatter_records())}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["case_id"] != case_id or payload["scale"] != scale:
        raise ValueError(f"{path} names {payload['case_id']!r} at {payload['scale']!r}")
    return OfficialUpstreamScatter(
        case_id=str(payload["case_id"]),
        scale=str(payload["scale"]),
        upstream_commit=str(payload["upstream_commit"]),
        official_script=str(payload["official_script"]),
        official_script_sha256=str(payload["official_script_sha256"]),
        runner=UpstreamScatterRunner.from_payload(payload["runner"]),
        protocol=SensitivityProtocol.from_payload(payload["protocol"]),
        capture_keys=MappingProxyType(
            {
                str(key): str(value)
                for key, value in dict(payload["capture_keys"]).items()
            }
        ),
        runs=tuple(UpstreamScatterRun.from_payload(run) for run in payload["runs"]),
    )
