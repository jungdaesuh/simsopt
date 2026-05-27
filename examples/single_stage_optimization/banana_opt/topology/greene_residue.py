from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


GREENE_RESIDUE_ELLIPTIC_O = "elliptic_o_point"
GREENE_RESIDUE_HYPERBOLIC_X = "hyperbolic_x_point"
GREENE_RESIDUE_PARABOLIC = "parabolic_rational_surface"
GREENE_RESIDUE_PERIOD_DOUBLING = "period_doubling_boundary"


@dataclass(frozen=True, slots=True)
class GreeneResidueDiagnostic:
    trace_m: float
    residue: float
    classification: str


def greene_residue_from_trace(trace_m: float) -> float:
    trace_value = float(trace_m)
    if not isfinite(trace_value):
        raise ValueError("Greene residue trace must be finite")
    return float((2.0 - trace_value) / 4.0)


def classify_greene_residue(residue: float) -> str:
    residue_value = float(residue)
    if not isfinite(residue_value):
        raise ValueError("Greene residue must be finite")
    if residue_value == 0.0:
        return GREENE_RESIDUE_PARABOLIC
    if residue_value == 1.0:
        return GREENE_RESIDUE_PERIOD_DOUBLING
    if 0.0 < residue_value < 1.0:
        return GREENE_RESIDUE_ELLIPTIC_O
    return GREENE_RESIDUE_HYPERBOLIC_X


def greene_residue_diagnostic_from_matrix(
    monodromy: np.ndarray,
) -> GreeneResidueDiagnostic:
    matrix = np.asarray(monodromy, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(
            f"Greene monodromy matrix must have shape (2, 2), got {matrix.shape}"
        )
    trace_m = float(np.trace(matrix))
    residue = greene_residue_from_trace(trace_m)
    return GreeneResidueDiagnostic(
        trace_m=trace_m,
        residue=residue,
        classification=classify_greene_residue(residue),
    )
