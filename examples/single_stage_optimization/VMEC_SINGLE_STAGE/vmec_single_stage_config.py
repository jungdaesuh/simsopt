"""Frozen configuration for the VMEC true single-stage objective.

Pins every parameter that participates in run-identity hashing per plan
section "Artifact Manifest". All weights, thresholds, FD-step values, and
step-clamp values are first-class config keys, not magic numbers in the
objective body.

The values exposed here are *placeholders pending first-campaign calibration*
unless explicitly marked hardware (e.g., ``TF_CURRENT_HARD_LIMIT_A``). The
plan calls out the calibration requirement in section "Driver step-clamp and
rejection". Each campaign must record the active config in
``artifact_manifest.json`` under the ``runtime_config`` block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# First-FD gradient-trap floors. Plan section "Finite-difference step schedule"
# names these as the two independent conditions for ``seed_zero_gradient_trap``.
# Both are placeholders pending the first-campaign noise-floor measurement.
# ---------------------------------------------------------------------------
GRADIENT_NORM_FLOOR_BOUNDARY: float = 1.0e-8
GRADIENT_NORM_FLOOR_COILS: float = 1.0e-8


@dataclass(frozen=True)
class VmecSingleStageConfig:
    """Frozen configuration for one VMEC true single-stage run.

    The dataclass is frozen so the objective body can take a config reference
    without defensive copying. Adding a field requires bumping the
    ``schema_version`` recorded in the artifact manifest.

    Attributes:
        ns_array, ftol_array, niter_array, delt: VMEC radial schedule per
            plan "VMEC convergence". Hot-loop runs use ``ns <= 51``.
        mpol, ntor: VMEC boundary mode resolution.
        nphi_vmec, ntheta_vmec: VMEC boundary sampling for SquaredFlux and
            normal-vector evaluation.
        w_coils: ``coils_objective_weight`` -- combines J1 + w_coils*J2.
        omega_L, omega_kappa_max, omega_kappa_msc, omega_d, omega_ell:
            Paper Eq. 26 coil penalty weights.
        aspect_weight, iota_weight, qs_weight, volume_weight: J1 component
            weights for the HBT extension block (paper Eq. 13 baseline plus
            volume target).
        aspect_target: Plasma aspect-ratio target.
        iota_target: Iota target at ``iota_promotion_surface``.
        volume_target: Plasma-volume target (m^3); references
            ``Vmec.volume()`` via the materialized ``phiedge``, not the
            Boozer-surface volume.
        length_target: Coil-length target (m). Plan keeps 1.778 m as the
            port-fit reference; the hardware ceiling stays at 2.0 m.
        L_threshold, cc_threshold, cs_threshold, kappa_max_threshold,
            kappa_msc_threshold: J2 penalty thresholds. ``cs_threshold`` is
            consumed by ``CurveSurfaceDistance`` for diagnostic purposes;
            the plan keeps coil-surface distance as a hard gate, not a
            full-gradient J2 term, until SIMSOPT exposes the surface-side
            derivative.
        eps_fd_xsurface_abs, eps_fd_xsurface_rel: ``MPIFiniteDifference``
            steps for J1 boundary-DOF gradient.
        eps_fd_xcoils_abs, eps_fd_xcoils_rel: Coil-DOF FD steps used only by
            the Taylor-test harness; production coil gradient is analytic.
        iota_tol: In-objective tolerance for ``IotaSurfaceTargetMiss``.
        iota_seed_quarantine_tol: Looser intake tolerance for
            ``seed_iota_surface_mismatch``.
        iota_promotion_surface_s: Normalized toroidal-flux label
            ``s`` at which iota is evaluated for both the J1 target and the
            promotion gate. Plan: "Pick one iota surface as SSOT".
        R0: Major-radius pin for the boundary mode ``RBC(0,0)``. Must equal
            ``VACUUM_VESSEL_MAJOR_RADIUS_M = 0.976 m`` for promotable runs.
        tf_current_pin_A: Pinned TF coil current (A). Negative sign per CW
            toroidal-field convention; absolute value must be at or below
            ``TF_CURRENT_HARD_LIMIT_A = 8.0e4 A``.
        pinned_current_coil_index: Index into the base-current vector of the
            coil whose current is fixed (removed from optimization DOFs).
            Recorded in ``dof_schema.fixed_dofs`` and avoids the paper
            page-7 zero-current quadratic-flux trap.
        max_curvature_inv_m: Hardware-gate ceiling. Per
            ``feedback_hardware_constraints_2026-04-13.md`` this is
            ``100 m^-1``, not ``40 m^-1``.
        qs_helicity_m, qs_helicity_n: ``QuasisymmetryRatioResidual``
            helicity. Stage A pins quasi-axisymmetry (``m=1, n=0``).
        qs_surfaces: Normalized-flux surfaces for the QS residual.
    """

    # VMEC radial schedule
    ns_array: Tuple[int, ...] = (13, 25, 51, 75, 101)
    ftol_array: Tuple[float, ...] = (1.0e-10, 1.0e-10, 1.0e-12, 1.0e-12, 1.0e-14)
    niter_array: Tuple[int, ...] = (3000, 3000, 5000, 5000, 10000)
    delt: float = 0.9

    # VMEC boundary mode resolution
    mpol: int = 6
    ntor: int = 6

    # SquaredFlux / normal sampling on VMEC boundary
    nphi_vmec: int = 34
    ntheta_vmec: int = 34

    # Combined-objective weight
    w_coils: float = 1.0e3

    # Paper Eq. 26 coil-penalty weights
    omega_L: float = 1.0e-1
    omega_kappa_max: float = 1.0e-3
    omega_kappa_msc: float = 1.0e-3
    omega_d: float = 1.0e0
    omega_ell: float = 1.0e-9

    # J1 / HBT-extension component weights
    aspect_weight: float = 1.0
    iota_weight: float = 1.0
    qs_weight: float = 1.0
    volume_weight: float = 1.0

    # Physics targets
    aspect_target: float = 7.0
    iota_target: float = 0.22
    volume_target: float = 0.10
    length_target: float = 1.778

    # J2 penalty thresholds
    L_threshold: float = 1.778
    cc_threshold: float = 0.05
    cs_threshold: float = 0.015
    kappa_max_threshold: float = 100.0
    kappa_msc_threshold: float = 10.0

    # Finite-difference step schedule
    eps_fd_xsurface_abs: float = 1.0e-7
    eps_fd_xsurface_rel: float = 0.0
    eps_fd_xcoils_abs: float = 0.0
    eps_fd_xcoils_rel: float = 1.0e-4

    # Iota promotion / quarantine thresholds
    iota_tol: float = 0.02
    iota_seed_quarantine_tol: float = 0.05
    iota_promotion_surface_s: float = 0.24

    # Hardware pins
    R0: float = 0.976
    tf_current_pin_A: float = -8.0e4
    pinned_current_coil_index: int = 0
    max_curvature_inv_m: float = 100.0

    # Quasisymmetry helicity (Stage A: quasi-axisymmetry)
    qs_helicity_m: int = 1
    qs_helicity_n: int = 0
    qs_surfaces: Tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    )
