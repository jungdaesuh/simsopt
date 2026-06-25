"""Legacy VMEC object adapters for JAX diagnostic entrypoints."""

from __future__ import annotations

from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_splines as _host_vmec_splines
from simsopt_jax.mhd.vmec_diagnostics import (
    VmecFrozenSplineState,
    VmecGeometryResultsJAX,
    vmec_compute_geometry_jax as _vmec_compute_geometry_jax,
    vmec_fieldlines_jax as _vmec_fieldlines_jax,
    vmec_freeze_splines as _vmec_freeze_splines,
)

__all__ = (
    "VmecFrozenSplineState",
    "VmecGeometryResultsJAX",
    "vmec_compute_geometry_jax",
    "vmec_fieldlines_jax",
    "vmec_freeze_splines",
)


def _spline_state(vs):
    return _host_vmec_splines(vs) if isinstance(vs, Vmec) else vs


def vmec_freeze_splines(vs) -> VmecFrozenSplineState:
    """Freeze a VMEC object or host spline state into a JAX pytree."""
    return _vmec_freeze_splines(_spline_state(vs))


def vmec_compute_geometry_jax(vs, s, theta, phi, phi_center: float = 0.0):
    """Evaluate VMEC geometry diagnostics from legacy or frozen VMEC state."""
    return _vmec_compute_geometry_jax(
        _spline_state(vs),
        s,
        theta,
        phi,
        phi_center,
    )


def vmec_fieldlines_jax(
    vs,
    s,
    alpha,
    theta1d=None,
    phi1d=None,
    phi_center: float = 0.0,
):
    """Evaluate VMEC fieldline diagnostics from legacy or frozen VMEC state."""
    return _vmec_fieldlines_jax(
        _spline_state(vs),
        s,
        alpha,
        theta1d=theta1d,
        phi1d=phi1d,
        phi_center=phi_center,
    )
