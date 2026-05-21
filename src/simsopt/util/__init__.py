import os

from .._lazy_exports import build_lazy_export_map, resolve_lazy_export

_UTIL_MODULES = (
    "mpi",
    "logger",
    "famus_helpers",
    "polarization_project",
    "permanent_magnet_helper_functions",
    "coil_optimization_helper_functions",
)

_EXPORT_TO_MODULE, _lazy_all = build_lazy_export_map(__file__, _UTIL_MODULES)

"""Boolean indicating if we are in the GitHub actions CI"""
in_github_actions = "CI" in os.environ and os.environ["CI"].lower() in ["1", "true"]

__all__ = list(_lazy_all) + ["in_github_actions"]


def __getattr__(name):
    value = resolve_lazy_export(__name__, _EXPORT_TO_MODULE, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
