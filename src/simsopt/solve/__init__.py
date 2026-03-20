from . import serial as serial
from . import mpi as mpi
from . import permanent_magnet_optimization as permanent_magnet_optimization
from . import wireframe_optimization as wireframe_optimization

from .serial import *
from .mpi import *
from .permanent_magnet_optimization import *
from .wireframe_optimization import *

__all__ = (serial.__all__ + mpi.__all__ + permanent_magnet_optimization.__all__
           + wireframe_optimization.__all__)
