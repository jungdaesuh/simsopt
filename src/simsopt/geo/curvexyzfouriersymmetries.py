import numpy as np
from .curve import JaxCurve
from math import gcd

__all__ = ["CurveXYZFourierSymmetries"]


def _jaxXYZFourierSymmetriescurve_pure(dofs, points, order, nfp, stellsym, ntor):
    from ..jax_core.curve_xyz_fourier_symmetries import (
        jaxXYZFourierSymmetriescurve_pure,
    )

    return jaxXYZFourierSymmetriescurve_pure(dofs, points, order, nfp, stellsym, ntor)


class CurveXYZFourierSymmetries(JaxCurve):
    r"""A curve representation that allows for stellarator and discrete rotational symmetries.  This class can be used to
    represent a helical coil that does not lie on a torus.  The coordinates of the curve are given by:

    .. math::
        x(\theta) &= \hat x(\theta)  \cos(2 \pi \theta n_{\text{tor}}) - \hat y(\theta)  \sin(2 \pi \theta n_{\text{tor}})\\
        y(\theta) &= \hat x(\theta)  \sin(2 \pi \theta n_{\text{tor}}) + \hat y(\theta)  \cos(2 \pi \theta n_{\text{tor}})\\
        z(\theta) &= \sum_{m=1}^{\text{order}} z_{s,m} \sin(2 \pi n_{\text{fp}} m \theta)

    where

    .. math::
        \hat x(\theta) &= x_{c, 0} + \sum_{m=1}^{\text{order}} x_{c,m} \cos(2 \pi n_{\text{fp}} m \theta)\\
        \hat y(\theta) &=            \sum_{m=1}^{\text{order}} y_{s,m} \sin(2 \pi n_{\text{fp}} m \theta)\\


    if the coil is stellarator symmetric.  When the coil is not stellarator symmetric, the formulas above
    become

    .. math::
        x(\theta) &= \hat x(\theta)  \cos(2 \pi \theta n_{\text{tor}}) - \hat y(\theta)  \sin(2 \pi \theta n_{\text{tor}})\\
        y(\theta) &= \hat x(\theta)  \sin(2 \pi \theta n_{\text{tor}}) + \hat y(\theta)  \cos(2 \pi \theta n_{\text{tor}})\\
        z(\theta) &= z_{c, 0} + \sum_{m=1}^{\text{order}} \left[ z_{c, m} \cos(2 \pi n_{\text{fp}} m \theta) + z_{s, m} \sin(2 \pi n_{\text{fp}} m \theta) \right]

    where

    .. math::
        \hat x(\theta) &= x_{c, 0} + \sum_{m=1}^{\text{order}} \left[ x_{c, m} \cos(2 \pi n_{\text{fp}} m \theta) +  x_{s, m} \sin(2 \pi n_{\text{fp}} m \theta) \right] \\
        \hat y(\theta) &= y_{c, 0} + \sum_{m=1}^{\text{order}} \left[ y_{c, m} \cos(2 \pi n_{\text{fp}} m \theta) +  y_{s, m} \sin(2 \pi n_{\text{fp}} m \theta) \right] \\

    Args:
        quadpoints: number of grid points/resolution along the curve,
        order:  how many Fourier harmonics to include in the Fourier representation,
        nfp: discrete rotational symmetry number, 
        stellsym: stellaratory symmetry if True, not stellarator symmetric otherwise,
        ntor: the number of times the curve wraps toroidally before biting its tail. Note,
              it is assumed that nfp and ntor are coprime.  If they are not coprime,
              then then the curve actually has nfp_new:=nfp // gcd(nfp, ntor),
              and ntor_new:=ntor // gcd(nfp, ntor).  The operator `//` is integer division.
              To avoid confusion, we assert that ntor and nfp are coprime at instantiation.
    """

    def __init__(self, quadpoints, order, nfp, stellsym, ntor=1, **kwargs):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        def pure(dofs, points):
            return _jaxXYZFourierSymmetriescurve_pure(
                dofs, points, order, nfp, stellsym, ntor
            )

        if gcd(ntor, nfp) != 1:
            raise ValueError("nfp and ntor must be coprime")

        self.order = order
        self.nfp = nfp
        self.stellsym = stellsym
        self.ntor = ntor
        self.coefficients = np.zeros(self.num_dofs())
        if "dofs" not in kwargs:
            if "x0" not in kwargs:
                kwargs["x0"] = self.coefficients
            else:
                self.set_dofs_impl(kwargs["x0"])

        super().__init__(quadpoints, pure, names=self._make_names(order), **kwargs)

    def _make_names(self, order):
        if self.stellsym:
            x_cos_names = [f"xc({i})" for i in range(0, order + 1)]
            x_names = x_cos_names
            y_sin_names = [f"ys({i})" for i in range(1, order + 1)]
            y_names = y_sin_names
            z_sin_names = [f"zs({i})" for i in range(1, order + 1)]
            z_names = z_sin_names
        else:
            x_names = ["xc(0)"]
            x_cos_names = [f"xc({i})" for i in range(1, order + 1)]
            x_sin_names = [f"xs({i})" for i in range(1, order + 1)]
            x_names += x_cos_names + x_sin_names
            y_names = ["yc(0)"]
            y_cos_names = [f"yc({i})" for i in range(1, order + 1)]
            y_sin_names = [f"ys({i})" for i in range(1, order + 1)]
            y_names += y_cos_names + y_sin_names
            z_names = ["zc(0)"]
            z_cos_names = [f"zc({i})" for i in range(1, order + 1)]
            z_sin_names = [f"zs({i})" for i in range(1, order + 1)]
            z_names += z_cos_names + z_sin_names

        return x_names + y_names + z_names

    def num_dofs(self):
        return (
            (self.order + 1) + self.order + self.order
            if self.stellsym
            else 3 * (2 * self.order + 1)
        )

    def get_dofs(self):
        return self.coefficients

    def set_dofs_impl(self, dofs):
        self.coefficients[:] = dofs[:]

    def to_spec(self):
        """Return an immutable JAX ``CurveXYZFourierSymmetriesSpec``.

        The spec captures the host-side ``(dofs, quadpoints, order, nfp,
        stellsym, ntor)`` tuple so downstream JAX routes can consume the
        curve without holding a reference to this mutable wrapper.
        """
        from ..jax_core import make_curve_xyzfouriersymmetries_spec

        return make_curve_xyzfouriersymmetries_spec(
            dofs=self.get_dofs(),
            quadpoints=self.quadpoints,
            order=self.order,
            nfp=self.nfp,
            stellsym=self.stellsym,
            ntor=self.ntor,
        )
