"""Dependency-light copy of the one physical constant simsopt_jax needs.

Kept separate from :mod:`simsopt.util.constants` so importing it does not pull
in the ``simsopt.util`` package's heavier import surface.
"""

ELEMENTARY_CHARGE = 1.602176634e-19  # C
