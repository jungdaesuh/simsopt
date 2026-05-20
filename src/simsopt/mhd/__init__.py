# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

import importlib.util

from .._lazy_exports import build_lazy_export_map, resolve_lazy_export

try:
    _jax_spec = importlib.util.find_spec("jax")
except ModuleNotFoundError:
    _jax_spec = None

_has_jax = _jax_spec is not None

_CPU_MHD_MODULES = (
    "vmec",
    "virtual_casing",
    "vmec_diagnostics",
    "profiles",
    "bootstrap",
    "boozer",
    "spec",
)
_JAX_MHD_MODULES = ("bootstrap_jax", "profiles_jax", "vmec_diagnostics_jax")

_MHD_MODULES = _CPU_MHD_MODULES + (_JAX_MHD_MODULES if _has_jax else ())

_EXPORT_TO_MODULE, __all__ = build_lazy_export_map(__file__, _MHD_MODULES)


def __getattr__(name):
    value = resolve_lazy_export(__name__, _EXPORT_TO_MODULE, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
