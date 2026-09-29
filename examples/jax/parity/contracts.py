"""Frozen contracts shared by parity cases, lanes, and the arbiter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


def _is_hex_digest(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


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
    """One case-owned endpoint quality floor certifiable at ``native_default``.

    The 2026-08-15 certification-gate ruling (rule 3) admits a continuous
    optimizer whose lanes fork by rejection sentinel or line search at
    ``native_default`` only as an endpoint quality band -- "every compared lane
    reaches ``observable`` <= ``max_value`` at the matched budget" -- and never
    as final-value equivalence. ``derivation`` records the measured evidence
    the band was read off, so the floor can never become a free parameter.
    When a case also declares an ``AdmittedTerminalOutcome``, that one lane is
    exempt from the matched budget (it stopped early by definition); the
    matched-budget clause then binds the remaining ``budget_exhausted`` lanes,
    of which at least one must exist.
    """

    observable: str
    max_value: float
    derivation: str

    def __post_init__(self) -> None:
        phase, separator, name = self.observable.partition(":")
        if not phase or not separator or not name:
            raise ValueError("quality band observable must be 'phase:name'")
        if not isinstance(self.max_value, float) or not math.isfinite(self.max_value):
            raise ValueError("quality band max_value must be a finite float")
        if not self.derivation:
            raise ValueError("quality band requires a recorded derivation")


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
