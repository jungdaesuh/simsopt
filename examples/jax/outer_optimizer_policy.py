"""The two approved host SciPy outer optimizers over JAX Boozer physics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OuterOptimizerPolicyId = Literal[
    "scipy-bfgs-over-jax-exact-boozer",
    "scipy-lbfgsb-over-jax-reduced-boozer",
]

SHIPPED_SINGLE_STAGE_SCIPY_DRIVER_ID = (
    "simsopt_jax_scipy_bfgs_with_exact_analytic_boozer_newton"
)


class OuterOptimizerPolicyError(ValueError):
    """An example record does not carry its approved outer optimizer policy.

    A ``ValueError`` so the manifest readers that already treat validation
    failures as one class keep working, and its own type so a caller that
    must react to exactly this rejection -- a legacy manifest that declares no
    policy for a ready host SciPy example -- never catches an unrelated one.
    """


@dataclass(frozen=True, slots=True)
class OuterOptimizerPolicy:
    """An approved owner and exact driver, independent of scientific authority."""

    policy_id: OuterOptimizerPolicyId
    example_id: str
    example_path: str
    case_id: str | None
    expected_driver: str


_APPROVED_POLICIES = (
    OuterOptimizerPolicy(
        "scipy-bfgs-over-jax-exact-boozer",
        "native-single-stage-boozer-vacuum-optimization",
        "3_Advanced/single_stage_boozer_vacuum_optimization.py",
        "native-single-stage-boozer-vacuum-optimization",
        SHIPPED_SINGLE_STAGE_SCIPY_DRIVER_ID,
    ),
    OuterOptimizerPolicy(
        "scipy-lbfgsb-over-jax-reduced-boozer",
        "native-boozerqa-ls",
        "2_Intermediate/boozerQA_ls.py",
        None,
        "scipy.optimize.minimize:L-BFGS-B",
    ),
)


def parse_outer_optimizer_policy(
    value: object, *, example_id: str, example_path: str, ready: bool = False
) -> OuterOptimizerPolicy | None:
    """Resolve an optional declaration only for its approved example and path."""
    if value is None:
        if ready and any(
            policy.example_id == example_id or policy.example_path == example_path
            for policy in _APPROVED_POLICIES
        ):
            raise OuterOptimizerPolicyError(
                "ready host SciPy example requires its outer optimizer policy declaration"
            )
        return None
    for policy in _APPROVED_POLICIES:
        if value == policy.policy_id:
            if (example_id, example_path) != (policy.example_id, policy.example_path):
                raise OuterOptimizerPolicyError(
                    "outer optimizer policy belongs to a different example"
                )
            return policy
    raise OuterOptimizerPolicyError("unknown outer optimizer policy")


def policy_owns_parity_case(
    policy: OuterOptimizerPolicy, *, case_id: str | None, example_id: str | None
) -> bool:
    """Reject forged policy records and tutorial policies without a parity case."""
    return (
        policy in _APPROVED_POLICIES
        and policy.case_id is not None
        and (case_id, example_id) == (policy.case_id, policy.example_id)
    )


def validate_ready_example_policy(
    policy: OuterOptimizerPolicy | None, *, example_id: str, example_path: str
) -> None:
    """Guard execution of ready canonical or legacy records before child launch."""
    canonical = parse_outer_optimizer_policy(
        None if policy is None else policy.policy_id,
        example_id=example_id,
        example_path=example_path,
        ready=True,
    )
    if policy != canonical:
        raise OuterOptimizerPolicyError(
            "outer optimizer policy differs from the approved declaration"
        )
