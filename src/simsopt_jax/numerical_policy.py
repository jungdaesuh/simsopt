"""JAX-free numerical policy shared by runtime code and artifact validators."""

from __future__ import annotations

import math
import secrets
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Final, Literal

CertificateDType = Literal["float64"]
CertificateProbePrngImpl = Literal["threefry2x32"]
CertificateProbeSamplingModel = Literal["jax_threefry2x32_finite_pseudorandom_normal"]
CertificateProbabilityModel = Literal["independent_ideal_standard_gaussian"]
CertificateProbeSource = Literal["fresh_runtime", "supplied_replay"]
PRODUCTION_HYBRID_FINAL_DENSE_IR_BACKEND_CODE: Final[int] = 4
MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS: Final[int] = sys.float_info.mant_dig
DENSE_IR_HISTORY_RESIDUAL_RELATIVE_CAPACITY: Final[int] = (
    MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS + 1
)
DENSE_IR_HISTORY_CONTRACTION_RATIO_CAPACITY: Final[int] = (
    MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS
)
NEWTON_ARMIJO_C1: Final[float] = 1.0e-4


class DenseIrHistorySource(IntEnum):
    """Factor origin for one selected dense-IR correction history."""

    NONE = 0
    FP32_PROPOSAL_FACTORS = 1
    FP64_CERTIFICATE_FACTORS = 2
    FP64_REFACTOR_RETRY = 3


MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_BITS: Final[int] = 32
MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_COUNT: Final[int] = 2
MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX: Final[int] = (
    1 << MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_BITS
) - 1


@dataclass(frozen=True, slots=True)
class CertificateProbeKeyData:
    """Complete typed Threefry key data for one certificate challenge."""

    word0: int
    word1: int

    def __post_init__(self) -> None:
        for word in self.words:
            if (
                isinstance(word, bool)
                or not isinstance(word, int)
                or not 0 <= word <= MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX
            ):
                raise ValueError("Certificate probe key words must fit in uint32.")

    @property
    def words(self) -> tuple[int, int]:
        """Return the canonical Threefry word order."""
        return self.word0, self.word1

    def as_json(self) -> list[int]:
        """Return the canonical JSON representation."""
        return list(self.words)

    @classmethod
    def from_json(cls, value: object) -> CertificateProbeKeyData:
        """Parse the exact two-word artifact representation."""
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_COUNT
        ):
            raise ValueError("Certificate probe key data must contain two words.")
        word0, word1 = value
        if (
            isinstance(word0, bool)
            or not isinstance(word0, int)
            or isinstance(word1, bool)
            or not isinstance(word1, int)
        ):
            raise ValueError(  # noqa: TRY004 - malformed serialized value
                "Certificate probe key words must be integers."
            )
        return cls(word0=word0, word1=word1)


@dataclass(frozen=True, slots=True)
class CertificateProbeAuthority:
    """Replayable host authority for one runtime certificate challenge."""

    source: CertificateProbeSource
    key_data: CertificateProbeKeyData

    def __post_init__(self) -> None:
        if self.source not in ("fresh_runtime", "supplied_replay"):
            raise ValueError("Certificate probe authority source is invalid.")
        if not isinstance(self.key_data, CertificateProbeKeyData):
            raise ValueError("Certificate probe authority key has the wrong type.")

    def as_json(self) -> dict[str, object]:
        """Return the exact challenge and the two distinct probability models."""
        return {
            "source": self.source,
            "key_data": self.key_data.as_json(),
            "prng_impl": MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL,
            "sampling_model": MIXED_DENSE_IR_CERTIFICATE_PROBE_SAMPLING_MODEL,
            "probability_model": MIXED_DENSE_IR_CERTIFICATE_PROBABILITY_MODEL,
        }

    @classmethod
    def from_json(cls, value: object) -> CertificateProbeAuthority:
        """Parse the canonical fresh-or-replay authority representation."""
        expected_keys = {
            "source",
            "key_data",
            "prng_impl",
            "sampling_model",
            "probability_model",
        }
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ValueError(
                "Certificate probe authority fields differ from the canonical schema."
            )
        if value["prng_impl"] != MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL:
            raise ValueError("Certificate probe PRNG implementation differs.")
        if value["sampling_model"] != MIXED_DENSE_IR_CERTIFICATE_PROBE_SAMPLING_MODEL:
            raise ValueError("Certificate probe sampling model differs.")
        if value["probability_model"] != MIXED_DENSE_IR_CERTIFICATE_PROBABILITY_MODEL:
            raise ValueError("Certificate probe probability model differs.")
        source_value = value["source"]
        if source_value == "fresh_runtime":
            source: CertificateProbeSource = "fresh_runtime"
        elif source_value == "supplied_replay":
            source = "supplied_replay"
        else:
            raise ValueError("Certificate probe authority source is invalid.")
        return cls(
            source=source,
            key_data=CertificateProbeKeyData.from_json(value["key_data"]),
        )

    @property
    def claim_eligible(self) -> bool:
        """Return whether fresh runtime entropy authorized this challenge."""
        return self.source == "fresh_runtime"


@dataclass(frozen=True, slots=True)
class CertificateProbeEvidence:
    """Host-normalized trust decision bound to one exact probe challenge."""

    authority: CertificateProbeAuthority
    observed_key_data: CertificateProbeKeyData
    active: bool
    proposal_trusted: bool
    fp64_rebuild_count: int
    fallback_attempted: bool
    fallback_success: bool

    def __post_init__(self) -> None:
        if not isinstance(self.authority, CertificateProbeAuthority):
            raise ValueError("Certificate probe authority has the wrong type.")
        if not isinstance(self.observed_key_data, CertificateProbeKeyData):
            raise ValueError("Observed certificate probe key has the wrong type.")
        flags = (
            self.active,
            self.proposal_trusted,
            self.fallback_attempted,
            self.fallback_success,
        )
        if any(not isinstance(flag, bool) for flag in flags):
            raise ValueError("Certificate probe decision flags must be boolean.")
        if (
            isinstance(self.fp64_rebuild_count, bool)
            or not isinstance(self.fp64_rebuild_count, int)
            or self.fp64_rebuild_count not in (0, 1)
        ):
            raise ValueError(
                "Certificate probe FP64 rebuild count must be zero or one."
            )

    def require_valid_for_mixed(self) -> None:
        """Reject any mismatch between the challenge, trust, and FP64 fallback."""
        if not self.active:
            raise ValueError("Mixed certificate evidence must be active.")
        if self.observed_key_data != self.authority.key_data:
            raise ValueError(
                "Mixed certificate observed key differs from its challenge."
            )
        expected_rebuild_count = int(not self.proposal_trusted)
        if self.fp64_rebuild_count != expected_rebuild_count:
            raise ValueError(
                "Mixed certificate trust and FP64 rebuild decisions disagree."
            )
        if self.fallback_attempted != bool(self.fp64_rebuild_count):
            raise ValueError(
                "Mixed certificate rebuild count and fallback evidence disagree."
            )
        if self.fallback_success and not self.fallback_attempted:
            raise ValueError(
                "Mixed certificate fallback reports success without an attempt."
            )
        if self.fp64_rebuild_count and not self.fallback_success:
            raise ValueError(
                "Mixed certificate FP64 fallback did not certify the solve."
            )

    def as_json(self) -> dict[str, object]:
        """Return the strict replayable challenge and observed decision."""
        return {
            "authority": self.authority.as_json(),
            "observed_key_data": self.observed_key_data.as_json(),
            "active": self.active,
            "proposal_trusted": self.proposal_trusted,
            "fp64_rebuild_count": self.fp64_rebuild_count,
            "fallback_attempted": self.fallback_attempted,
            "fallback_success": self.fallback_success,
            "claim_eligible": self.authority.claim_eligible,
        }

    @classmethod
    def from_json(
        cls,
        value: object,
        *,
        require_claim_eligible: bool = False,
    ) -> CertificateProbeEvidence:
        """Parse and validate one strict mixed-certificate decision."""
        expected_keys = {
            "authority",
            "observed_key_data",
            "active",
            "proposal_trusted",
            "fp64_rebuild_count",
            "fallback_attempted",
            "fallback_success",
            "claim_eligible",
        }
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise ValueError("Certificate probe evidence fields differ.")
        flags = (
            value["active"],
            value["proposal_trusted"],
            value["fallback_attempted"],
            value["fallback_success"],
            value["claim_eligible"],
        )
        if any(not isinstance(flag, bool) for flag in flags):
            raise ValueError("Certificate probe evidence flags must be boolean.")
        rebuild_count = value["fp64_rebuild_count"]
        if isinstance(rebuild_count, bool) or not isinstance(rebuild_count, int):
            raise ValueError("Certificate probe FP64 rebuild count must be an integer.")
        evidence = cls(
            authority=CertificateProbeAuthority.from_json(value["authority"]),
            observed_key_data=CertificateProbeKeyData.from_json(
                value["observed_key_data"]
            ),
            active=value["active"],
            proposal_trusted=value["proposal_trusted"],
            fp64_rebuild_count=rebuild_count,
            fallback_attempted=value["fallback_attempted"],
            fallback_success=value["fallback_success"],
        )
        evidence.require_valid_for_mixed()
        if value["claim_eligible"] != evidence.authority.claim_eligible:
            raise ValueError("Certificate probe claim eligibility differs from source.")
        if require_claim_eligible and not evidence.authority.claim_eligible:
            raise ValueError(
                "Replay certificate probes cannot authorize a fresh claim."
            )
        return evidence


def fresh_certificate_probe_key_data() -> CertificateProbeKeyData:
    """Mint one host-side challenge after operator inputs freeze, outside tracing."""
    return CertificateProbeKeyData(
        word0=secrets.randbits(MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_BITS),
        word1=secrets.randbits(MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_BITS),
    )


def resolve_certificate_probe_authority(
    replay_key_data: CertificateProbeKeyData | None,
) -> CertificateProbeAuthority:
    """Mint fresh authority or preserve an explicitly supplied replay key."""
    if replay_key_data is None:
        return CertificateProbeAuthority(
            source="fresh_runtime",
            key_data=fresh_certificate_probe_key_data(),
        )
    return CertificateProbeAuthority(
        source="supplied_replay",
        key_data=replay_key_data,
    )


def dense_ir_factorization_precision_evidence_is_complete(
    *,
    factorization_event_count: int,
    last_factorization_dtype_bits: int | None,
    factor_application_dtype_bits: int | None,
) -> bool:
    """Match factor/apply dtype evidence exactly to factorization activity."""
    dtype_evidence_absent = (
        last_factorization_dtype_bits is None and factor_application_dtype_bits is None
    )
    dtype_evidence_complete = (
        last_factorization_dtype_bits is not None
        and factor_application_dtype_bits is not None
    )
    return (factorization_event_count == 0 and dtype_evidence_absent) or (
        factorization_event_count > 0 and dtype_evidence_complete
    )


@dataclass(frozen=True, slots=True)
class MixedDenseIrAccuracyPolicy:
    """Host-side SSOT for certified FP32-factor/FP64-residual accuracy."""

    certificate_dtype: CertificateDType
    linear_solve_tolerance_floor: float
    linear_solve_tolerance_cap: float
    forward_error_tolerance_multiplier: float

    def forward_error_tolerance(self, linear_solve_tolerance: float) -> float:
        """Derive the relative forward-error limit from a certified solve input."""
        tolerance = float(linear_solve_tolerance)
        if (
            not math.isfinite(tolerance)
            or not self.linear_solve_tolerance_floor
            <= tolerance
            <= self.linear_solve_tolerance_cap
        ):
            raise ValueError(
                "Mixed dense-IR linear-solve tolerance is outside the FP64 policy."
            )
        return max(
            math.sqrt(sys.float_info.epsilon),
            self.forward_error_tolerance_multiplier * tolerance,
        )


MIXED_DENSE_IR_ACCURACY_POLICY: Final[MixedDenseIrAccuracyPolicy] = (
    MixedDenseIrAccuracyPolicy(
        certificate_dtype="float64",
        linear_solve_tolerance_floor=1e-14,
        linear_solve_tolerance_cap=1e-10,
        forward_error_tolerance_multiplier=10.0,
    )
)
# Halko–Martinsson–Tropp contraction-probe constants for mixed dense-IR
# certificates. Owned here so production kernels and validators share one SSOT.
MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT: Final[int] = 64
MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA: Final[float] = 2.0
MIXED_DENSE_IR_CONTRACTION_NORM_UPPER_LIMIT: Final[float] = 0.9
MIXED_DENSE_IR_CONTRACTION_IDEAL_GAUSSIAN_FAILURE_PROBABILITY_BOUND: Final[float] = (
    MIXED_DENSE_IR_CONTRACTION_PROBE_ALPHA**-MIXED_DENSE_IR_CONTRACTION_PROBE_COUNT
)
MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL: Final[CertificateProbePrngImpl] = "threefry2x32"
MIXED_DENSE_IR_CERTIFICATE_PROBE_SAMPLING_MODEL: Final[
    CertificateProbeSamplingModel
] = "jax_threefry2x32_finite_pseudorandom_normal"
MIXED_DENSE_IR_CERTIFICATE_PROBABILITY_MODEL: Final[CertificateProbabilityModel] = (
    "independent_ideal_standard_gaussian"
)


def mixed_dense_ir_certificate_dtype_name() -> CertificateDType:
    """Return the policy certificate dtype name (numpy/JAX-compatible string)."""
    return MIXED_DENSE_IR_ACCURACY_POLICY.certificate_dtype
