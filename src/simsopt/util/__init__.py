import os
import sys

from .mpi import *
from .logger import *
from .famus_helpers import *
from .polarization_project import *
from .permanent_magnet_helper_functions import *
from .coil_optimization_helper_functions import *

"""Boolean indicating if we are in the GitHub actions CI"""
in_github_actions = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']


def _module_all(name):
    return list(sys.modules[f"{__name__}.{name}"].__all__)


__all__ = (
    _module_all("mpi")
    + _module_all("logger")
    + _module_all("famus_helpers")
    + _module_all("polarization_project")
    + _module_all("permanent_magnet_helper_functions")
    + _module_all("coil_optimization_helper_functions")
    + ['in_github_actions']
)
