"""Back-compat re-export shim for the per-class JAX MagneticField wrappers.

Each JAX wrapper now lives in its own module (one class per file). Import
sites that still use the historical ``magneticfieldclasses_jax`` path keep
working through this shim; new code should import the wrappers directly
from their dedicated modules below.
"""

from __future__ import annotations

from .circular_coil import CircularCoilJAX
from .dommaschk import DommaschkJAX
from .mirror_model import MirrorModelJAX
from .poloidal_field import PoloidalFieldJAX
from .reiman import ReimanJAX
from .toroidal_field import ToroidalFieldJAX


__all__ = [
    "CircularCoilJAX",
    "DommaschkJAX",
    "MirrorModelJAX",
    "PoloidalFieldJAX",
    "ReimanJAX",
    "ToroidalFieldJAX",
]
