"""Legacy geometry-object adapters for ``simsopt_jax``."""

from simsopt_jax_adapters.geo.curvecwsfourier import (
    CurveCWSFourier,
    CurveCWSFourierCPP,
)
from simsopt_jax_adapters.geo.stochastic_perturbations import (
    materialize_stochastic_coil_perturbations,
)

__all__ = (
    "CurveCWSFourier",
    "CurveCWSFourierCPP",
    "materialize_stochastic_coil_perturbations",
)
