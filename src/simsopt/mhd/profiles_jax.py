# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""JAX-backed radial profile classes."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import splrep

from .._core.optimizable import Optimizable
from .._core.descriptor import PositiveInteger
from ..jax_core._math_utils import as_jax_float64
from ..jax_core.profiles import (
    profile_polynomial_dfds,
    profile_polynomial_value,
    profile_pressure_dfds,
    profile_pressure_value,
    profile_scaled_dfds,
    profile_scaled_value,
    profile_spline_dfds,
    profile_spline_value,
)
from .profiles import Profile

__all__ = [
    "ProfilePolynomialJAX",
    "ProfilePressureJAX",
    "ProfileScaledJAX",
    "ProfileSplineJAX",
]


def _set_profile_spline_jax_state(profile, dofs) -> None:
    profile._set_spline_state(dofs)


class ProfilePolynomialJAX(Profile):
    """JAX-native equivalent of :class:`ProfilePolynomial`."""

    def __init__(self, data):
        super().__init__(x0=np.asarray(data, dtype=np.float64))
        self.local_fix_all()

    def f(self, s):
        """Return the value of the profile at specified points in ``s``."""
        return self.f_from_dofs(self.local_full_x, s)

    def dfds(self, s):
        """Return the d/ds derivative of the profile at specified points in ``s``."""
        return self.dfds_from_dofs(self.local_full_x, s)

    def f_from_dofs(self, dofs, s):
        """Evaluate the profile from explicit DOFs for JIT-safe callers."""
        return profile_polynomial_value(dofs, s)

    def dfds_from_dofs(self, dofs, s):
        """Evaluate d/ds from explicit DOFs for JIT-safe callers."""
        return profile_polynomial_dfds(dofs, s)


class ProfileScaledJAX(Profile):
    """JAX-native equivalent of :class:`ProfileScaled`."""

    def __init__(self, base: Optimizable, scalefac):
        self.base = base
        super().__init__(
            x0=np.asarray([scalefac], dtype=np.float64),
            names=["scalefac"],
            depends_on=[base],
        )
        self.local_fix_all()

    def f(self, s):
        """Return the value of the profile at specified points in ``s``."""
        return self.f_from_scale(self.local_full_x[0], self.base.f(s))

    def dfds(self, s):
        """Return the d/ds derivative of the profile at specified points in ``s``."""
        return self.dfds_from_scale(self.local_full_x[0], self.base.dfds(s))

    def f_from_scale(self, scale, base_value):
        """Scale an explicit base value for JIT-safe callers."""
        return profile_scaled_value(scale, base_value)

    def dfds_from_scale(self, scale, base_dfds):
        """Scale an explicit base derivative for JIT-safe callers."""
        return profile_scaled_dfds(scale, base_dfds)


class ProfileSplineJAX(Profile):
    """JAX-native evaluator for a host-fitted FITPACK profile spline."""

    degree = PositiveInteger()

    def __init__(self, s, f, degree=3):
        self.s = np.asarray(s, dtype=np.float64)
        self.degree = degree
        if int(self.degree) < 1 or int(self.degree) > 5:
            raise ValueError("ProfileSplineJAX degree must be in [1, 5].")
        self._spline_knots = np.array([], dtype=np.float64)
        self._spline_coeffs = np.array([], dtype=np.float64)
        self._spline_degree = int(self.degree)
        super().__init__(
            x0=np.asarray(f, dtype=np.float64),
            external_dof_setter=_set_profile_spline_jax_state,
        )
        self._set_spline_state(self.local_full_x)
        self.local_fix_all()

    def _set_spline_state(self, dofs) -> None:
        knots, coeffs, degree = splrep(
            self.s,
            np.asarray(dofs, dtype=np.float64),
            k=int(self.degree),
            s=0,
        )
        self._spline_knots = as_jax_float64(np.asarray(knots, dtype=np.float64))
        self._spline_coeffs = as_jax_float64(np.asarray(coeffs, dtype=np.float64))
        self._spline_degree = int(degree)

    def f(self, s):
        """Return the value of the profile at specified points in ``s``."""
        return self.f_from_state(self._spline_knots, self._spline_coeffs, s)

    def dfds(self, s):
        """Return the d/ds derivative of the profile at specified points in ``s``."""
        return self.dfds_from_state(self._spline_knots, self._spline_coeffs, s)

    @property
    def spline_state(self):
        """Return the explicit FITPACK coefficient state for JIT-safe callers."""
        return self._spline_knots, self._spline_coeffs

    def f_from_state(self, knots, coeffs, s):
        """Evaluate the profile from explicit FITPACK state."""
        return profile_spline_value(knots, coeffs, self._spline_degree, s)

    def dfds_from_state(self, knots, coeffs, s):
        """Evaluate d/ds from explicit FITPACK state."""
        return profile_spline_dfds(knots, coeffs, self._spline_degree, s)

    def resample(self, new_s, degree=None):
        """Return a ``ProfileSplineJAX`` object on a different spline grid."""
        new_degree = self.degree
        if degree is not None:
            new_degree = degree
        return ProfileSplineJAX(new_s, self.f(new_s), degree=new_degree)


class ProfilePressureJAX(Profile):
    """JAX-native equivalent of :class:`ProfilePressure`."""

    def __init__(self, *args: Optimizable):
        if len(args) == 0:
            raise ValueError(
                "At least one density and temperature profile must be provided."
            )
        if len(args) % 2 == 1:
            raise ValueError(
                "The number of input profiles for a ProfilePressureJAX object must be even"
            )
        super().__init__(depends_on=args)

    def f(self, s):
        """Return the value of the profile at specified points in ``s``."""
        values = tuple(parent.f(s) for parent in self.parents)
        return self.f_from_values(values)

    def dfds(self, s):
        """Return the d/ds derivative of the profile at specified points in ``s``."""
        values = tuple(parent.f(s) for parent in self.parents)
        derivatives = tuple(parent.dfds(s) for parent in self.parents)
        return self.dfds_from_values(values, derivatives)

    def f_from_values(self, values):
        """Evaluate pressure from explicit profile values for JIT-safe callers."""
        return profile_pressure_value(tuple(values))

    def dfds_from_values(self, values, derivatives):
        """Evaluate pressure d/ds from explicit values for JIT-safe callers."""
        return profile_pressure_dfds(tuple(values), tuple(derivatives))
