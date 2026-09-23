"""Case-owned permission to compare honest fixed-budget optimizer endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from simsopt_jax.examples import ExecutionScale


@dataclass(frozen=True, slots=True)
class WorkBudgetContract:
    """Allow budget exits only at explicit scales under a documented upstream policy.

    Admission compares aggregate terminal category and configured budgets;
    per-stage reasons and actual counts remain in ``raw_status`` and ``nit``.
    Numerical comparisons remain decisive.
    """

    scales: tuple[ExecutionScale, ...]
    derivation: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scales, tuple)
            or not self.scales
            or len(self.scales) != len(set(self.scales))
            or not set(self.scales) <= {"bounded", "native_default"}
        ):
            raise ValueError("work-budget scales must be a unique explicit tuple")
        if not isinstance(self.derivation, str) or not self.derivation.strip():
            raise ValueError("work-budget derivation must name the upstream policy")
