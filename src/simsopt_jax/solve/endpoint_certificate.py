"""Compatibility exports for the base optimization endpoint contract.

The contract itself is owned by :mod:`simsopt_contracts.optimization_endpoint`,
a package neither lane owns; this module only keeps the historical
``simsopt_jax.solve`` import path alive for the benchmark record writers.
"""

from simsopt_contracts.optimization_endpoint import (
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
