"""Shared host reporting contract for exact Boozer outer problems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BoozerEndpoint:
    """Objective, gradient and physics left by one evaluated coil state.

    On inner-solve failure, value and gradient carry the problem's sentinel
    policy; the physics fields describe the Boozer state left in place.
    """

    value: float
    gradient: NDArray[np.float64]
    inner_success: bool
    iota: float
    volume: float
    non_qs_ratio: float
    boozer_residual: float
    major_radius_penalty: float
    length_penalty: float

    @property
    def boozer_residual_rms(self) -> float:
        """Root mean square of the Boozer residual vector.

        ``boozer_residual`` is half the mean square of that vector, which is
        what native's ``BoozerResidual.J()`` returns.
        """
        return float(np.sqrt(2.0 * self.boozer_residual))
