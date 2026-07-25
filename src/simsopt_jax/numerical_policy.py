"""JAX-free numerical policy shared by runtime code and artifact validators."""

from __future__ import annotations

import math
import secrets
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

CertificateDType = Literal["float64"]
CertificateProbePrngImpl = Literal["threefry2x32"]
CertificateProbeSamplingModel = Literal["jax_threefry2x32_finite_pseudorandom_normal"]
CertificateProbabilityModel = Literal["independent_ideal_standard_gaussian"]
PRODUCTION_HYBRID_FINAL_DENSE_IR_BACKEND_CODE: Final[int] = 4
MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS: Final[int] = sys.float_info.mant_dig


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


def fresh_certificate_probe_key_data() -> CertificateProbeKeyData:
    """Mint one full-entropy Threefry challenge before JAX import or tracing."""
    return CertificateProbeKeyData(
        word0=secrets.randbits(MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_BITS),
        word1=secrets.randbits(MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_BITS),
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
MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL: Final[CertificateProbePrngImpl] = "threefry2x32"
MIXED_DENSE_IR_CERTIFICATE_PROBE_SAMPLING_MODEL: Final[
    CertificateProbeSamplingModel
] = "jax_threefry2x32_finite_pseudorandom_normal"
MIXED_DENSE_IR_CERTIFICATE_PROBABILITY_MODEL: Final[CertificateProbabilityModel] = (
    "independent_ideal_standard_gaussian"
)
