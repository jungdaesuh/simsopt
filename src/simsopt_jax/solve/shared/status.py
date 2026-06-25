"""Shared public status enums for ``simsopt_jax.solve`` callbacks."""

from __future__ import annotations

from enum import StrEnum


class LineSearchStatus(StrEnum):
    SUCCESS = "success"
    MAXITER = "maxiter"
    FAILED = "failed"


class InvalidStepReason(StrEnum):
    NONFINITE = "nonfinite"
    STALLED = "stalled"
    LINE_SEARCH_FAILED = "line_search_failed"
