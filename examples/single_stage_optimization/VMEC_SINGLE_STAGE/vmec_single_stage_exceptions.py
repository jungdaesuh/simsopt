"""Typed exceptions for the VMEC true single-stage objective.

These classes implement the transactional failure-classification contract from
``program_cw_poloidal_legacy_vmec_true_single_stage_plan.md`` section "Failure
Taxonomy" and the Crucible-addendum (post-2026-05-14 plan) gradient/promotion
identities. Each exception carries the fields the autoresearch ledger and the
driver step-clamp need to classify a candidate without further string parsing:

- ``failure_class``: stable identifier matching the autoresearch failure
  taxonomy (``vmec_internal_error``, ``iota_surface_target_miss``, etc.).
- ``retryable``: whether the autoresearch agent loop is permitted to spawn a
  continuation row for this candidate. Resume from an accepted checkpoint is
  *not* a retry; see plan "Checkpoint and resume".

These exceptions are intentionally narrow. They never mask VMEC errors with
``try``/``except`` higher up the call stack; the driver propagates them to the
autoresearch ledger and the optimizer treats the candidate as rejected.
"""

from __future__ import annotations

from typing import Optional


class VmecCallFailure(Exception):
    """Base class for classified VMEC-related failures inside the objective.

    Used by the driver and the autoresearch ledger to classify a candidate
    without parsing exception messages. Subclasses pin the canonical
    ``failure_class`` string from the plan failure taxonomy.

    Args:
        message: Human-readable failure description.
        ier_flag: VMEC ``ier_flag`` value when available, else ``None``.
        failure_class: Stable autoresearch-taxonomy class (see plan section
            "Failure Taxonomy > Minimum failure classes").
        retryable: ``True`` if the autoresearch agent loop should spawn a
            continuation row with adjusted settings.
    """

    failure_class: str = "vmec_internal_error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        ier_flag: Optional[int] = None,
        failure_class: Optional[str] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        super().__init__(message)
        self.ier_flag = ier_flag
        if failure_class is not None:
            self.failure_class = failure_class
        if retryable is not None:
            self.retryable = retryable


class IotaSurfaceTargetMiss(VmecCallFailure):
    """Candidate ``wout`` violates the iota-at-promotion-surface tolerance.

    Raised by ``J_and_grad`` (and therefore the ``J_scalar`` /
    ``compute_grad`` views) after a successful VMEC solve when

    .. code-block:: text

        abs(iota_at_promotion_surface(wout) - iota_target) > iota_tol

    Per plan section "Objective Design > HBT objective extensions", this
    rejects the candidate regardless of objective improvement. Both the
    in-objective tolerance and the looser seed-quarantine tolerance are
    placeholder thresholds pending first-campaign calibration.
    """

    failure_class: str = "iota_surface_target_miss"
    retryable: bool = False


class StepClampBarrierTriggered(VmecCallFailure):
    """Diagnostic marker that the driver step-clamp barrier was active.

    The plan ("Optimizer entrypoint" / "Driver step-clamp and rejection")
    requires the barrier to return a *finite* quadratic value and gradient,
    not to raise. This class exists only for diagnostic taxonomy so a
    candidate whose step-clamp barrier dominated its objective can be tagged
    without misclassifying it as a VMEC failure.

    The clamp barrier itself is implemented in the driver layer
    (``banana_drivers/04_vmec_singlestage_driver.py``), which wraps the
    simsopt-surrogate objective bundle; the bundle's ``J_scalar`` /
    ``compute_grad`` / ``J_and_grad`` always run VMEC. Instances of this
    exception are never thrown out of the objective and are emitted on
    the diagnostic record only.
    """

    failure_class: str = "step_clamp_barrier_triggered"
    retryable: bool = True


class SeedZeroGradientTrap(Exception):
    """First-FD gradient norms at seed fall below the configured floors.

    Per plan section "Runtime, Concurrency, FD Schedule, and Checkpointing >
    Finite-difference step schedule", two independent conditions can trip
    this trap:

    1. Boundary-DOF FD gradient norm of ``J1 + w_coils * J2`` below
       ``GRADIENT_NORM_FLOOR_BOUNDARY``.
    2. When ``w_coils > 0``, the *analytic* J2-only coil-DOF gradient norm
       below ``GRADIENT_NORM_FLOOR_COILS`` (no VMEC call required).

    Mechanism is distinct from the Boozer-surface chain-rule trap of
    ``feedback_zero_gradient_seeds.md`` (which does not apply to the VMEC
    gradient) and from the paper page-7 zero-current trap (classified
    separately as ``current_dof_pinned_to_zero``).
    """

    failure_class: str = "seed_zero_gradient_trap"

    def __init__(
        self,
        message: str,
        *,
        boundary_norm: float,
        coil_norm: float,
    ) -> None:
        super().__init__(message)
        self.boundary_norm = float(boundary_norm)
        self.coil_norm = float(coil_norm)
