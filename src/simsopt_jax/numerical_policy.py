"""JAX-free numerical policy shared by runtime code and artifact validators."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Final

MIXED_DENSE_IR_MAX_REFINEMENT_CORRECTIONS: Final[int] = sys.float_info.mant_dig
NEWTON_ARMIJO_C1: Final[float] = 1.0e-4


@dataclass(frozen=True, slots=True)
class MixedDenseIrAccuracyPolicy:
    """Host-side SSOT for FP64 dense linear-solve tolerance and forward-error limits."""

    linear_solve_tolerance_floor: float
    linear_solve_tolerance_cap: float
    forward_error_tolerance_multiplier: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.linear_solve_tolerance_floor)
            or self.linear_solve_tolerance_floor <= 0.0
        ):
            raise ValueError(
                "Mixed dense-IR accuracy policy requires a finite positive "
                "linear-solve tolerance floor."
            )
        if (
            not math.isfinite(self.linear_solve_tolerance_cap)
            or self.linear_solve_tolerance_cap < self.linear_solve_tolerance_floor
        ):
            raise ValueError(
                "Mixed dense-IR accuracy policy requires a finite tolerance cap "
                "at least as large as its floor."
            )
        if (
            not math.isfinite(self.forward_error_tolerance_multiplier)
            or self.forward_error_tolerance_multiplier <= 0.0
        ):
            raise ValueError(
                "Mixed dense-IR accuracy policy requires a finite positive "
                "forward-error tolerance multiplier."
            )

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
        linear_solve_tolerance_floor=1e-14,
        linear_solve_tolerance_cap=1e-10,
        forward_error_tolerance_multiplier=10.0,
    )
)


def mixed_dense_ir_accuracy_policy() -> MixedDenseIrAccuracyPolicy:
    """Return the immutable dense-IR accuracy policy owner."""
    return MIXED_DENSE_IR_ACCURACY_POLICY
