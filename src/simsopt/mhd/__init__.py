import sys

# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

from .vmec import *
from .virtual_casing import *
from .vmec_diagnostics import *
from .profiles import *
from .bootstrap import *
from .boozer import *
from .spec import *


def _module_all(name):
    return list(sys.modules[f"{__name__}.{name}"].__all__)


__all__ = (
    _module_all("vmec")
    + _module_all("virtual_casing")
    + _module_all("vmec_diagnostics")
    + _module_all("profiles")
    + _module_all("bootstrap")
    + _module_all("boozer")
    + _module_all("spec")
)
