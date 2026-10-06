"""Example-bound host SciPy outer optimizers over JAX physics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OuterOptimizerPolicyId = Literal[
    "scipy-lbfgsb-over-jax-standard-stage-two",
    "scipy-lbfgsb-over-jax-planar-stage-two",
    "scipy-lbfgsb-over-jax-stochastic-stage-two",
    "scipy-lbfgsb-over-jax-minimal-stage-two",
    "scipy-lbfgsb-over-jax-finite-build-stage-two",
    "scipy-lbfgsb-over-jax-coil-forces",
    "scipy-lbfgsb-slsqp-over-jax-qfm",
    "scipy-trf-over-jax-quadratic",
    "scipy-trf-over-jax-curve-length",
    "scipy-trf-over-jax-surf-vol-area",
]


class OuterOptimizerPolicyError(ValueError):
    """An example record does not carry its approved outer optimizer policy.

    A ``ValueError`` so the manifest readers that already treat validation
    failures as one class keep working, and its own type so a caller that
    must react to exactly this rejection -- a manifest record that declares no
    policy for a ready host SciPy example -- never catches an unrelated one.
    """


@dataclass(frozen=True, slots=True)
class OuterOptimizerPolicy:
    """An approved owner and exact driver."""

    policy_id: OuterOptimizerPolicyId
    example_id: str
    example_path: str
    expected_driver: str


_APPROVED_POLICIES = (
    OuterOptimizerPolicy(
        "scipy-trf-over-jax-quadratic",
        "native-just-a-quadratic",
        "1_Simple/just_a_quadratic.py",
        "scipy_least_squares_trf_jax_quadratic",
    ),
    OuterOptimizerPolicy(
        "scipy-trf-over-jax-curve-length",
        "native-minimize-curve-length",
        "1_Simple/minimize_curve_length.py",
        "scipy_least_squares_trf_jax_curve_length",
    ),
    OuterOptimizerPolicy(
        "scipy-trf-over-jax-surf-vol-area",
        "native-surf-vol-area",
        "1_Simple/surf_vol_area.py",
        "scipy_least_squares_trf_jax_surf_vol_area",
    ),
    OuterOptimizerPolicy(
        "scipy-lbfgsb-slsqp-over-jax-qfm",
        "native-qfm",
        "1_Simple/qfm.py",
        "scipy_lbfgsb_slsqp_qfm_sequence",
    ),
    # The official native scripts of standard, planar and stochastic stage two,
    # minimal stage two, finite-build stage two and coil forces call
    # scipy.optimize.minimize with method='L-BFGS-B'; their shipped JAX mirrors
    # select Driver.SCIPY_LBFGSB. The rows are named rather than counted: the
    # count said "five" over six rows after two were inserted here.
    OuterOptimizerPolicy(
        "scipy-lbfgsb-over-jax-standard-stage-two",
        "native-stage-two-optimization",
        "2_Intermediate/stage_two_optimization.py",
        "scipy_lbfgsb",
    ),
    OuterOptimizerPolicy(
        "scipy-lbfgsb-over-jax-planar-stage-two",
        "native-stage-two-optimization-planar-coils",
        "2_Intermediate/stage_two_optimization_planar_coils.py",
        "scipy_lbfgsb",
    ),
    OuterOptimizerPolicy(
        "scipy-lbfgsb-over-jax-stochastic-stage-two",
        "native-stage-two-optimization-stochastic",
        "2_Intermediate/stage_two_optimization_stochastic.py",
        "scipy_lbfgsb",
    ),
    OuterOptimizerPolicy(
        "scipy-lbfgsb-over-jax-minimal-stage-two",
        "native-stage-two-optimization-minimal",
        "1_Simple/stage_two_optimization_minimal.py",
        "scipy_lbfgsb",
    ),
    OuterOptimizerPolicy(
        "scipy-lbfgsb-over-jax-finite-build-stage-two",
        "native-stage-two-optimization-finitebuild",
        "3_Advanced/stage_two_optimization_finitebuild.py",
        "scipy_lbfgsb",
    ),
    OuterOptimizerPolicy(
        "scipy-lbfgsb-over-jax-coil-forces",
        "native-coil-forces",
        "3_Advanced/coil_forces.py",
        "scipy_lbfgsb",
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


def validate_ready_example_policy(
    policy: OuterOptimizerPolicy | None, *, example_id: str, example_path: str
) -> None:
    """Guard execution of ready manifest records before child launch."""
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
