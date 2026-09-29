"""Frozen contracts shared by parity cases, lanes, and the arbiter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, get_args

import numpy as np
from simsopt_contracts.examples_runtime import ExecutionScale


def _is_hex_digest(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _is_lane_key(key: object) -> bool:
    """Whether ``key`` spells one published lane value key, ``phase:name``."""
    if not isinstance(key, str):
        return False
    phase, separator, name = key.partition(":")
    return bool(phase and separator and name)


@dataclass(frozen=True)
class ArrayReference:
    path: str
    dtype: str
    shape: tuple[int, ...]
    order: str
    sha256: str


@dataclass(frozen=True)
class ParityInputMetadata:
    case_id: str
    random_seed: int
    input_fingerprint: str
    configuration_fingerprint: str


@dataclass(frozen=True)
class InitialStateResult:
    objective_sum_squares: float | None
    solver_cost: float | None
    applicability: Mapping[str, bool]
    arrays: Mapping[str, ArrayReference]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "applicability", MappingProxyType(dict(self.applicability))
        )
        object.__setattr__(self, "arrays", MappingProxyType(dict(self.arrays)))

    @property
    def objective_gradient_factor(self) -> float:
        """Multiplier converting ``J.T @ r`` to the public objective gradient."""
        return 2.0


@dataclass(frozen=True)
class FinalStateResult:
    objective_sum_squares: float | None
    solver_cost: float | None
    feasibility: float | None
    applicability: Mapping[str, bool]
    arrays: Mapping[str, ArrayReference]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "applicability", MappingProxyType(dict(self.applicability))
        )
        object.__setattr__(self, "arrays", MappingProxyType(dict(self.arrays)))


@dataclass(frozen=True)
class LaneResult:
    lane: str
    input_metadata: ParityInputMetadata
    initial: InitialStateResult
    final: FinalStateResult | None
    normalized_status: str
    raw_status: str
    success: bool
    nit: int | None
    nfev: int | None
    njev: int | None


@dataclass(frozen=True)
class ComparisonResult:
    phase: str
    observable: str
    lane_pair: str
    passed: bool
    tolerance_bucket: str
    diagnostic: str


@dataclass(frozen=True)
class QualityBand:
    """One case-owned endpoint quality floor, certifiable at its declared ``scale`` only.

    The 2026-08-15 certification-gate ruling (rule 3) admits a continuous
    optimizer whose lanes fork by rejection sentinel or line search only as an
    endpoint quality band -- "every compared lane reaches ``observable`` <=
    ``max_value`` at the matched budget" -- and never as final-value
    equivalence. ``derivation`` records the measured evidence the band was read
    off, so the floor can never become a free parameter; ``scale`` is the one
    execution scale that evidence was measured at, and the arbiter refuses the
    band at any other. When a case also declares an ``AdmittedTerminalOutcome``,
    that one lane is exempt from the matched budget (it stopped early by
    definition); the matched-budget clause then binds the remaining
    ``budget_exhausted`` lanes, of which at least one must exist.
    """

    observable: str
    max_value: float
    derivation: str
    scale: ExecutionScale = field(kw_only=True)

    def __post_init__(self) -> None:
        if not _is_lane_key(self.observable):
            raise ValueError("quality band observable must be 'phase:name'")
        if not isinstance(self.max_value, float) or not math.isfinite(self.max_value):
            raise ValueError("quality band max_value must be a finite float")
        if not self.derivation:
            raise ValueError("quality band requires a recorded derivation")
        if self.scale not in get_args(ExecutionScale):
            raise ValueError(
                f"quality band scale is not an execution scale: {self.scale!r}"
            )


@dataclass(frozen=True)
class UpstreamEndState:
    """One end state upstream's own workflow reached from one start of the one-ulp protocol.

    ``k`` is the draw index of that start (k = 0 is the unperturbed start), and
    ``values`` maps each published lane key (``phase:name``) to upstream's
    finite FP64 value there, 0-d for a scalar. The arrays are copied and made
    read-only, so a recorded draw can never drift after it is declared.
    """

    k: int
    values: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 0:
            raise ValueError("upstream end state k must be a non-negative int")
        frozen: dict[str, np.ndarray] = {}
        for key, value in self.values.items():
            if not _is_lane_key(key):
                raise ValueError("upstream end-state key must be 'phase:name'")
            # Held in the published receipt form: a lane receipt stores a scalar as a
            # one-element array (artifacts.write_array), so upstream's does too.
            array = np.array(value, copy=True, ndmin=1)
            if array.dtype != np.dtype(np.float64):
                raise ValueError(f"upstream end-state value must be FP64: {key}")
            if array.size == 0 or not bool(np.all(np.isfinite(array))):
                raise ValueError(
                    f"upstream end-state value must be non-empty and finite: {key}"
                )
            array.setflags(write=False)
            frozen[key] = array
        if not frozen:
            raise ValueError("upstream end state requires at least one value")
        object.__setattr__(self, "values", MappingProxyType(frozen))


@dataclass(frozen=True)
class UpstreamEndStates:
    """Upstream's own end states at ONE scale for a workflow whose end state its input does not determine.

    Where upstream's own script, started from one-ulp perturbed copies of the
    same input, lands on several distinct end states (Boozer surface branches,
    say), no lane-versus-lane equality can be required of the port. A lane is
    then accepted when its end state MATCHES at least one of upstream's own
    draws: one draw whose value, for EVERY key in ``observables``, passes the
    comparator and tolerance the case's own route matrix declares for that key.
    This is an engineering acceptance against upstream's own scatter, never an
    equivalence proof, so the verdict it yields is ``quality-band`` at most.
    ``derivation`` records how the draws were produced and which tracked test
    proves the lanes compute upstream's function at upstream's states.
    """

    case_id: str
    scale: ExecutionScale
    observables: tuple[str, ...]
    states: tuple[UpstreamEndState, ...]
    derivation: str

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("upstream end states require an owning case_id")
        if self.scale not in get_args(ExecutionScale):
            raise ValueError(
                f"upstream end-state scale is not an execution scale: {self.scale!r}"
            )
        observables = tuple(self.observables)
        if (
            not observables
            or len(set(observables)) != len(observables)
            or not all(_is_lane_key(key) for key in observables)
        ):
            raise ValueError(
                "upstream end-state observables must be unique non-empty 'phase:name' keys"
            )
        states = tuple(self.states)
        if len(states) < 2:
            raise ValueError("an upstream end-state set needs at least two draws")
        if len({state.k for state in states}) != len(states):
            raise ValueError("upstream end-state draws must have unique k")
        missing = sorted(
            {key for state in states for key in observables if key not in state.values}
        )
        if missing:
            raise ValueError(f"upstream end-state draws lack judged keys: {missing}")
        if not self.derivation:
            raise ValueError("upstream end states require a recorded derivation")
        object.__setattr__(self, "observables", observables)
        object.__setattr__(self, "states", states)


@dataclass(frozen=True)
class AdmittedTerminalOutcome:
    """One case-owned terminal lane outcome authorized at ``native_default``.

    A case declares this where upstream's own official script produced exactly
    this composite provider outcome under the one-ulp start protocol;
    ``upstream_evidence`` names the draws. The admission is then bound by four
    properties. It never relabels the lane: ``raw_status`` and
    ``normalized_status`` are the receipt's published strings, matched exactly
    and never parsed, and the lane keeps ``success`` false. Every finite, FP64
    and physical check and the upstream-only endpoint band stay in force, so
    the band still decides the verdict -- an engineering endpoint acceptance
    rule, never an equivalence or convergence proof. The verdict can therefore
    only ever be ``quality-band``, never ``pass``. And the admission is
    published in the arbitration result and in the receipt, so it can never be
    a silent pass. ``case_id`` and ``lane`` bind the authorization to ONE case
    and ONE lane (a case that admits an outcome on several lanes declares one
    entry per lane): the arbiter refuses an admission whose ``case_id`` is not
    the case it arbitrates, and any undeclared lane publishing the same raw
    status stays a rejected failure.
    """

    case_id: str
    lane: str
    raw_status: str
    normalized_status: str
    upstream_evidence: str

    def __post_init__(self) -> None:
        if self.normalized_status != "failed":
            raise ValueError(
                "admitted terminal outcome normalized status must be 'failed'"
            )
        if not self.case_id:
            raise ValueError("admitted terminal outcome requires an owning case_id")
        if not self.lane:
            raise ValueError("admitted terminal outcome requires the admitted lane")
        if not self.raw_status:
            raise ValueError("admitted terminal outcome requires a raw status")
        if not self.upstream_evidence:
            raise ValueError("admitted terminal outcome requires upstream evidence")


@dataclass(frozen=True)
class QualityBandResult:
    lane: str
    observable: str
    max_value: float
    observed_value: float
    passed: bool


@dataclass(frozen=True)
class EndStateResult:
    """Which of upstream's own end states one lane's end state matches."""

    lane: str
    matched_draws: tuple[int, ...]
    passed: bool

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.matched_draws))) != self.matched_draws:
            raise ValueError("matched draws must be unique and ascending")
        if self.passed != bool(self.matched_draws):
            raise ValueError("an end-state result passes iff it matches a draw")


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    authoritative: bool
    repository_commit: str
    repository_dirty: bool
    lane_results: tuple[LaneResult, ...]
    comparisons: tuple[ComparisonResult, ...]
    verdict: str


def validate_authoritative_source(
    *,
    authoritative: bool,
    repository_dirty: bool,
    repository_commit: str,
    executed_source_hashes: Mapping[str, str],
    simsoptpp_path: str | None,
    simsoptpp_sha256: str | None,
) -> None:
    """Reject provenance that cannot support an authoritative parity claim."""
    if not authoritative:
        return
    if repository_dirty:
        raise ValueError("authoritative evidence requires a clean repository")
    if not _is_hex_digest(repository_commit, 40):
        raise ValueError(
            "authoritative repository commit must contain 40 hexadecimal characters"
        )
    if not executed_source_hashes or any(
        not _is_hex_digest(digest, 64) for digest in executed_source_hashes.values()
    ):
        raise ValueError(
            "authoritative source hashes must contain 64 hexadecimal characters"
        )
    if (simsoptpp_path is None) != (simsoptpp_sha256 is None):
        raise ValueError("simsoptpp path and SHA-256 must be recorded together")
    if simsoptpp_sha256 is not None and not _is_hex_digest(simsoptpp_sha256, 64):
        raise ValueError("simsoptpp SHA-256 must contain 64 hexadecimal characters")
