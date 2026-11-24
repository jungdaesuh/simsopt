import numpy as np
import jax
import jax.numpy as jnp
import simsoptpp as sopp
from .curve import Curve
from .curveplanarfourier import CurvePlanarFourier, JaxCurvePlanarFourier
from .curvesuperellipse import *
from .curveplanarellipticalcylindrical import CurvePlanarEllipticalCylindrical
from .._core.derivative import Derivative

__all__ = ['CurvePlanarOnSurface']


PLACEMENT_DOF_NAMES = {
    CurvePlanarFourier: ['q0', 'qi', 'qj', 'qk', 'X', 'Y', 'Z'],
    JaxCurvePlanarFourier: ['q0', 'qi', 'qj', 'qk', 'X', 'Y', 'Z'],
    CurveSuperEllipse: ['q0', 'qi', 'qj', 'qk', 'X', 'Y', 'Z'],
    CurvePlanarEllipticalCylindrical: ['R0', 'phi', 'Z0', 'r_rotation',
                                       'phi_rotation', 'z_rotation'],
    ScaledCurveSuperEllipse: [],
}


def surface_frame_pure(unitnormal, tangent):
    e3 = unitnormal / jnp.linalg.norm(unitnormal)
    t = tangent - jnp.dot(tangent, e3) * e3
    e1 = t / jnp.linalg.norm(t)
    e2 = jnp.cross(e3, e1)
    return jnp.stack((e1, e2, e3), axis=-1)


def place_points_pure(points, center, unitnormal, tangent):
    return center[None, :] + points @ surface_frame_pure(unitnormal, tangent).T


def rotate_vectors_pure(vectors, unitnormal, tangent):
    return vectors @ surface_frame_pure(unitnormal, tangent).T

_PLACE_POINTS_VJP = jax.jit(
    lambda p, c, n, t, v: jax.vjp(place_points_pure, p, c, n, t)[1](v))
_ROTATE_VECTORS_VJP = jax.jit(
    lambda p, n, t, v: jax.vjp(rotate_vectors_pure, p, n, t)[1](v))


class CurvePlanarOnSurface(sopp.Curve, Curve):
    r"""
    ``CurvePlanarOnSurface`` pins a planar curve to a surface. The curve is
    placed with its centre on the surface quadrature point ``(iphi, itheta)``
    and its plane tangent to the surface there, with the curve's local
    :math:`x` direction aligned to :math:`\partial\Gamma/\partial\theta` or
    :math:`\partial\Gamma/\partial\phi` according to ``align``. For a
    :class:`simsopt.geo.CurveSuperEllipse` base this puts the ``a`` semi-axis
    along the chosen tangent and ``b`` perpendicular to it.

    The curve has no placement degrees of freedom of its own. Its position and
    orientation are functions of the surface dofs, so reshaping the surface
    carries the curve rigidly with it and gradients flow through to the surface
    analytically. The base curve's plane, centroid and reference direction are
    found once at construction, so whatever pose it arrived in is factored out.

    .. warning::
        The base curve's placement dofs (``q0``, ``qi``, ``qj``, ``qk``, ``X``,
        ``Y``, ``Z``, or all of its dofs for types that do not expose those
        names) are **fixed** on construction. The base frame is computed once,
        so a later change to the base curve's placement would leave that frame
        stale and slide the curve off the surface.

    Args:
        curve: Planar base curve supplying the in-plane shape. Must be one of
            ``CurvePlanarFourier``, ``JaxCurvePlanarFourier``,
            ``CurveSuperEllipse`` or ``CurvePlanarEllipticalCylindrical``.
        surface: Surface the curve is pinned to.
        iphi (int): Toroidal index into ``surface.quadpoints_phi``.
        itheta (int): Poloidal index into ``surface.quadpoints_theta``.
        align (str): ``"theta"`` (default) or ``"phi"``; which surface tangent
            the curve's local :math:`x` direction is aligned with.
    """
    def __init__(self, curve, surface, iphi, itheta, align="theta"):
        nphi = len(surface.quadpoints_phi)
        ntheta = len(surface.quadpoints_theta)
        if not 0 <= iphi < nphi or not 0 <= itheta < ntheta:
            raise ValueError(
                f"(iphi, itheta) = ({iphi}, {itheta}) is out of range for a "
                f"surface with {nphi} x {ntheta} quadpoints.")
        if align not in ("theta", "phi"):
            raise ValueError(
                f"align must be 'theta' or 'phi'; got {align!r}.")
        self.curve = curve
        self.surface = surface
        self.iphi = iphi
        self.itheta = itheta
        self.align = align
        self.base_center, self.base_rotation = self._base_frame()
        self._fix_placement()
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve, surface])
        self._place_vjp = _PLACE_POINTS_VJP
        self._rotate_vjp = _ROTATE_VECTORS_VJP

    def _base_frame(self):
        if type(self.curve) not in PLACEMENT_DOF_NAMES:
            raise TypeError(
                f"CurvePlanarOnSurface requires a planar curve, one of "
                f"{[t.__name__ for t in PLACEMENT_DOF_NAMES]}; got "
                f"{type(self.curve).__name__}.")
        gamma = self.curve.gamma()
        center = gamma.mean(axis=0)
        d = gamma - center
        normal = np.linalg.svd(d, full_matrices=False)[2][2]
        e1 = d[0] - (d[0] @ normal) * normal
        e1 /= np.linalg.norm(e1)
        return center, np.stack((e1, np.cross(normal, e1), normal), axis=-1)

    def _fix_placement(self):
        for name in PLACEMENT_DOF_NAMES[type(self.curve)]:
            self.curve.fix(name)

    def _local_points(self, points):
        return (points - self.base_center) @ self.base_rotation

    def _local_vectors(self, vectors):
        return vectors @ self.base_rotation

    def _anchor(self):
        i, j = self.iphi, self.itheta
        c = self.surface.gamma()[i, j]
        n = self.surface.unitnormal()[i, j]
        t = (self.surface.gammadash2()[i, j] if self.align == "theta"
             else self.surface.gammadash1()[i, j])
        return jnp.array(c), jnp.array(n), jnp.array(t)

    def _danchor(self):
        i, j = self.iphi, self.itheta
        dn = self.surface.dunitnormal_by_dcoeff()[i, j]
        dt = (self.surface.dgammadash2_by_dcoeff()[i, j] if self.align == "theta"
              else self.surface.dgammadash1_by_dcoeff()[i, j])
        return self.surface.dgamma_by_dcoeff()[i, j], dn, dt

    def rotation_matrix(self):
        _, n, t = self._anchor()
        return np.asarray(surface_frame_pure(n, t))

    def center(self):
        return self.surface.gamma()[self.iphi, self.itheta]

    def unitnormal(self):
        return self.surface.unitnormal()[self.iphi, self.itheta]

    def gamma_impl(self, gamma, quadpoints):
        c, n, t = self._anchor()
        if len(quadpoints) == len(self.curve.quadpoints) \
                and np.sum((quadpoints - self.curve.quadpoints)**2) < 1e-15:
            points = self.curve.gamma()
        else:
            points = np.zeros((len(quadpoints), 3))
            self.curve.gamma_impl(points, quadpoints)
        points = self._local_points(points)
        gamma[:] = np.asarray(place_points_pure(jnp.array(points), c, n, t))

    def gammadash_impl(self, gammadash):
        gammadash[:] = self._local_vectors(self.curve.gammadash()) @ self.rotation_matrix().T

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self._local_vectors(self.curve.gammadashdash()) @ self.rotation_matrix().T

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self._local_vectors(self.curve.gammadashdashdash()) @ self.rotation_matrix().T

    def _rotate_surface_vjp(self, v, vectors):
        _, n, t = self._anchor()
        _, gn, gt = self._rotate_vjp(jnp.array(vectors), n, t, jnp.array(v))
        _, dn, dt = self._danchor()
        return Derivative({self.surface: np.asarray(gn) @ dn + np.asarray(gt) @ dt}) # pyright: ignore[reportArgumentType]

    def dgamma_by_dcoeff_vjp(self, v):
        c, n, t = self._anchor()
        _, gc, gn, gt = self._place_vjp(
            jnp.array(self._local_points(self.curve.gamma())), c, n, t, jnp.array(v))
        dc, dn, dt = self._danchor()
        surface_term = Derivative({self.surface:
                                   np.asarray(gc) @ dc + np.asarray(gn) @ dn
                                   + np.asarray(gt) @ dt}) # pyright: ignore[reportArgumentType]
        w = (v @ self.rotation_matrix()) @ self.base_rotation.T
        return self.curve.dgamma_by_dcoeff_vjp(w) + surface_term

    def dgammadash_by_dcoeff_vjp(self, v):
        w = (v @ self.rotation_matrix()) @ self.base_rotation.T
        base = self.curve.dgammadash_by_dcoeff_vjp(w)
        return base + self._rotate_surface_vjp(
            v, self._local_vectors(self.curve.gammadash()))

    def dgammadashdash_by_dcoeff_vjp(self, v):
        w = (v @ self.rotation_matrix()) @ self.base_rotation.T
        base = self.curve.dgammadashdash_by_dcoeff_vjp(w)
        return base + self._rotate_surface_vjp(v, self._local_vectors(self.curve.gammadashdash()))

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        w = (v @ self.rotation_matrix()) @ self.base_rotation.T
        base = self.curve.dgammadashdashdash_by_dcoeff_vjp(w)
        return base + self._rotate_surface_vjp(v, self._local_vectors(self.curve.gammadashdashdash()))

    def dkappa_by_dcoeff_vjp(self, v):
        return self.curve.dkappa_by_dcoeff_vjp(v)
    
    def num_dofs(self):
        """
        This curve has no local dofs. All the dofs are on the winding surface
        and the base curve.
        """
        return 0

    def _not_implemented(self, name):
        raise NotImplementedError(
            f"CurvePlanarOnSurface does not implement {name} because it has no "
            f"local dofs. All the dofs are on the winding surface and the base "
            f"curve.")

    def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
        self._not_implemented("dgamma_by_dcoeff")

    def dgammadash_by_dcoeff_impl(self, dgammadash_by_dcoeff):
        self._not_implemented("dgammadash_by_dcoeff")

    def dgammadashdash_by_dcoeff_impl(self, dgammadashdash_by_dcoeff):
        self._not_implemented("dgammadashdash_by_dcoeff")

    def dgammadashdashdash_by_dcoeff_impl(self, dgammadashdashdash_by_dcoeff):
        self._not_implemented("dgammadashdashdash_by_dcoeff")
