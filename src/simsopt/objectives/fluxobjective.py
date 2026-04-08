import numpy as np

import simsoptpp as sopp
from .._core.util import ObjectiveFailure
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec


__all__ = ["SquaredFlux"]


def _surface_normal_geometry(normal):
    absn = np.linalg.norm(normal, axis=2)
    has_normal = absn > 0.0
    safe_absn = np.where(has_normal, absn, 1.0)
    unitn = np.where(has_normal[..., None], normal / safe_absn[..., None], 0.0)
    return absn, has_normal, unitn


class SquaredFlux(Optimizable):
    r"""
    Objective representing quadratic-flux-like quantities, useful for stage-2
    coil optimization. Several variations are available, which can be selected
    using the ``definition`` argument. For ``definition="quadratic flux"``
    (the default), the objective is defined as

    .. math::
        J = \frac12 \int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds,

    where :math:`\mathbf{n}` is the surface unit normal vector and
    :math:`B_T` is an optional (zero by default) target value for the
    magnetic field. Also :math:`\int_{S} ds` indicates a surface integral.
    For ``definition="normalized"``, the objective is defined as

    .. math::
        J = \frac12 \frac{\int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds}
                         {\int_{S} |\mathbf{B}|^2 ds}.

    For ``definition="local"``, the objective is defined as

    .. math::
        J = \frac12 \int_{S} \frac{(\mathbf{B}\cdot \mathbf{n} - B_T)^2}{|\mathbf{B}|^2} ds.

    Zero-area surface elements contribute zero to all three definitions.
    For ``definition="normalized"``, a nonpositive global denominator
    :math:`\int_S |\mathbf{B}|^2 ds` is treated as invalid and the
    objective returns ``inf``. For ``definition="local"``, any
    positive-area quadrature point with :math:`|\mathbf{B}|^2 = 0`
    is likewise treated as invalid and yields ``inf``.

    The definition ``"quadratic flux"`` has the advantage of simplicity, and it
    is used in other contexts such as REGCOIL. However for stage-2 optimization,
    the optimizer can "cheat", lowering this objective by reducing the magnitude
    of the field. The definitions ``"normalized"`` and ``"local"`` close this loophole.

    Args:
        surface: A :obj:`simsopt.geo.surface.Surface` object on which to compute the flux
        field: A :obj:`simsopt.field.magneticfield.MagneticField` for which to compute the flux.
        target: A ``nphi x ntheta`` numpy array containing target values for the flux. Here
          ``nphi`` and ``ntheta`` correspond to the number of quadrature points on `surface`
          in ``phi`` and ``theta`` direction.
        definition: A string to select among the definitions above. The
          available options are ``"quadratic flux"``, ``"normalized"``, and ``"local"``.
    """

    def __init__(self, surface, field, target=None, definition="quadratic flux"):
        self.surface = surface
        if target is not None:
            self.target = np.ascontiguousarray(target)
        else:
            self.target = np.zeros(self.surface.normal().shape[:2])
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        return sopp.integral_BdotN(Bcoil, self.target, n, self.definition)

    @derivative_dec
    def dJ(self):
        n = self.surface.normal()
        absn, has_normal, unitn = _surface_normal_geometry(n)
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil * unitn, axis=2)
        if self.target is not None:
            B_n = Bcoil_n - self.target
        else:
            B_n = Bcoil_n

        if self.definition == "quadratic flux":
            dJdB = (B_n[..., None] * unitn * absn[..., None]) / absn.size
            dJdB = dJdB.reshape((-1, 3))

        elif self.definition == "local":
            B2 = np.sum(Bcoil * Bcoil, axis=2)
            if np.any(has_normal & (B2 <= 0.0)):
                raise ObjectiveFailure(
                    "SquaredFlux local gradient is singular because |B|^2 "
                    "vanishes on positive-area surface samples."
                )
            safe_B2 = np.where(B2 > 0.0, B2, 1.0)
            dJdB = (
                (
                    (B_n / safe_B2)[..., None]
                    * (unitn - (B_n / safe_B2)[..., None] * Bcoil)
                )
                * absn[..., None]
            ) / absn.size

        elif self.definition == "normalized":
            num = np.mean(B_n**2 * absn)
            B2 = np.sum(Bcoil * Bcoil, axis=2)
            denom = np.mean(B2 * absn)
            if denom <= 0.0:
                raise ObjectiveFailure(
                    "SquaredFlux normalized gradient is singular because the "
                    "surface integral of |B|^2 is zero."
                )
            dnum = 2 * (B_n[..., None] * unitn * absn[..., None]) / absn.size
            ddenom = 2 * (Bcoil * absn[..., None]) / absn.size
            dJdB = 0.5 * (dnum / denom - num * ddenom / denom**2)

        else:
            raise ValueError("Should never get here")

        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB)
