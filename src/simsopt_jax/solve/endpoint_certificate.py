"""Compatibility exports for the base optimization endpoint contract."""

from simsopt.optimization_endpoint import (
    OptimizationEndpointCertificate,
    StatusConvention,
    StoppingReason,
    _stopping_reason,
    certify_optimization_endpoint,
    status_convention_for,
)

__all__ = (
    "OptimizationEndpointCertificate",
    "StatusConvention",
    "StoppingReason",
    "_stopping_reason",
    "certify_optimization_endpoint",
    "status_convention_for",
)
