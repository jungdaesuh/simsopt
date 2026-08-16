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
