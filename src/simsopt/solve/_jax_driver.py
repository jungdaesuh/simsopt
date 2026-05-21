"""Shared solver-driver enum for JAX optimizer contracts.

This module is intentionally importable by legacy Python-supported modules.
The Python-3.11-only ``simsopt.solve.jax`` package re-exports the same enum.
"""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - compatibility for Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


class Driver(StrEnum):
    SCIPY_LBFGSB = "scipy_lbfgsb"
    SCIPY_LM = "scipy_lm"
    SCIPY_BFGS = "scipy_bfgs"
    OPTAX_LBFGS = "optax_lbfgs"
    OPTAX_ADAM = "optax_adam"
    OPTIMISTIX_LBFGS = "optimistix_lbfgs"
    OPTIMISTIX_LM = "optimistix_lm"
    SIMSOPT_LBFGSB = "simsopt_lbfgsb"
    SIMSOPT_BFGS = "simsopt_bfgs"
    SIMSOPT_TRACE_LBFGS = "simsopt_trace_lbfgs"
    SIMSOPT_ADAM_HOST = "simsopt_adam_host"
    SIMSOPT_ADAM = "simsopt_adam"
    SIMSOPT_LM_GMRES_HOST = "simsopt_lm_gmres_host"
    SIMSOPT_LM_GMRES = "simsopt_lm_gmres"
    SIMSOPT_LM_QR = "simsopt_lm_qr"


_LEGACY_REFERENCE_MINIMIZE_METHODS = {
    Driver.SCIPY_BFGS: "bfgs",
    Driver.SCIPY_LBFGSB: "lbfgs",
    Driver.SIMSOPT_TRACE_LBFGS: "lbfgs-trace",
    Driver.SIMSOPT_ADAM_HOST: "adam",
}
_LEGACY_REFERENCE_LEAST_SQUARES_METHODS = {
    Driver.SCIPY_BFGS: "bfgs",
    Driver.SCIPY_LBFGSB: "lbfgs",
    Driver.SIMSOPT_LM_GMRES_HOST: "lm",
}
_LEGACY_TARGET_MINIMIZE_METHODS = {
    Driver.SIMSOPT_BFGS: "bfgs-ondevice",
    Driver.SIMSOPT_LBFGSB: "lbfgs-ondevice",
    Driver.SIMSOPT_ADAM: "adam-ondevice",
    Driver.OPTAX_LBFGS: "optax-lbfgs-ondevice",
    Driver.OPTIMISTIX_LBFGS: "optimistix-lbfgs-ondevice",
}
_LEGACY_TARGET_LEAST_SQUARES_METHODS = {
    Driver.SIMSOPT_LM_GMRES: "lm-ondevice",
    Driver.SIMSOPT_LM_QR: "lm-minpack-ondevice",
    Driver.OPTIMISTIX_LM: "optimistix-lm-ondevice",
}
_LEGACY_TARGET_METHODS = {
    **_LEGACY_TARGET_MINIMIZE_METHODS,
    **_LEGACY_TARGET_LEAST_SQUARES_METHODS,
}
_LEGACY_TARGET_SCIPY_CONTROL_METHODS = {
    "scipy_jax": "lbfgs-scipy-jax",
    "scipy_jax_fullgraph": "lbfgs-scipy-jax-fullgraph",
}


def _legacy_method(mapping: dict[Driver, str], driver: Driver, *, role: str) -> str:
    try:
        return mapping[driver]
    except KeyError as exc:
        allowed = ", ".join(sorted(item.value for item in mapping))
        raise ValueError(
            f"{role} driver must be one of: {allowed}. Got {driver!r}."
        ) from exc


def legacy_reference_minimize_method(driver: Driver) -> str:
    """Translate a reference minimize driver at the legacy optimizer boundary."""
    return _legacy_method(
        _LEGACY_REFERENCE_MINIMIZE_METHODS,
        driver,
        role="reference minimize",
    )


def legacy_reference_least_squares_method(driver: Driver) -> str:
    """Translate a reference least-squares driver at the legacy boundary."""
    return _legacy_method(
        _LEGACY_REFERENCE_LEAST_SQUARES_METHODS,
        driver,
        role="reference least-squares",
    )


def legacy_target_minimize_method(driver: Driver) -> str:
    """Translate a target minimize driver at the legacy optimizer boundary."""
    return _legacy_method(
        _LEGACY_TARGET_MINIMIZE_METHODS,
        driver,
        role="target minimize",
    )


def legacy_target_least_squares_method(driver: Driver) -> str:
    """Translate a target least-squares driver at the legacy boundary."""
    return _legacy_method(
        _LEGACY_TARGET_LEAST_SQUARES_METHODS,
        driver,
        role="target least-squares",
    )


def legacy_target_method(driver: Driver) -> str:
    """Translate any target driver at the legacy optimizer boundary."""
    return _legacy_method(_LEGACY_TARGET_METHODS, driver, role="target optimizer")


def legacy_target_scipy_control_method(objective_route: str) -> str:
    """Translate a target SciPy-control route at the legacy optimizer boundary."""
    try:
        return _LEGACY_TARGET_SCIPY_CONTROL_METHODS[objective_route]
    except KeyError as exc:
        allowed = ", ".join(sorted(_LEGACY_TARGET_SCIPY_CONTROL_METHODS))
        raise ValueError(
            "Target Driver.SCIPY_LBFGSB requires objective_route to be one "
            f"of: {allowed}."
        ) from exc


__all__ = [
    "Driver",
    "legacy_reference_least_squares_method",
    "legacy_reference_minimize_method",
    "legacy_target_least_squares_method",
    "legacy_target_method",
    "legacy_target_minimize_method",
    "legacy_target_scipy_control_method",
]
