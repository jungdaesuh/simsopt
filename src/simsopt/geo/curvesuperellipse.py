import numpy as np
import jax.numpy as jnp

from jax import vjp
from scipy.special import gamma as gamma_fn

import simsoptpp as sopp
from .._core.derivative import Derivative
from .curve import Curve, JaxCurve
from .jit import jit

__all__ = ['CurveSuperEllipse', 'ScaledCurveSuperEllipse']

TINY = float(np.finfo(float).tiny)


def safe_kappa_pure(d1gamma, d2gamma):
    r"""
    Curvature with a finite derivative where the curvature vanishes.

    ``kappa_pure`` uses :math:`\|\gamma' \times \gamma''\|`, whose derivative
    is undefined when the cross product is exactly zero. That happens on the
    semi-axes for :math:`n > 2`, giving a ``nan`` gradient even though a
    one-sided curvature penalty contributes nothing at those points. Flooring
    the norm returns zero for the derivative there and leaves the value
    unchanged to sqrt(TINY) = 1e-154.
    """
    w = jnp.cross(d1gamma, d2gamma)
    return (jnp.sqrt(jnp.sum(w * w, axis=1) + TINY)
            / jnp.linalg.norm(d1gamma, axis=1) ** 3)

def superellipse_rho(theta, n):
    r"""
    This function returns the radial profile of the unit superellipse,

    .. math::
        \rho(\theta; n) = \left(|\cos\theta|^n + |\sin\theta|^n\right)^{-1/n},

    which depends on the exponent :math:`n` alone. The semi-axes are applied
    afterwards in :func:`curve_superellipse_pure`, which keeps them linear.

    The bases are floored at the smallest positive double. On the semi-axes one
    of them is exactly zero, and since :math:`n` is an optimizable dof,
    differentiating there evaluates
    :math:`\partial u^n / \partial n = u^n \log u` as ``0 * -inf = nan``.
    A floor is used instead of `where` because a branch would also flatten the
    derivatives with respect to :math:`\theta`.

    Args:
        theta (array, shape (N,)): Angles in radians.
        n (float): Superellipse exponent.

    Returns:
        Array of radii, shape (N,)
    """
    cos_term = (jnp.abs(jnp.cos(theta)) + TINY) ** n
    sin_term = (jnp.abs(jnp.sin(theta)) + TINY) ** n
    return (cos_term + sin_term) ** (-1 / n)


def curve_superellipse_pure(dofs, quadpoints):
    r"""
    This pure function returns the curve coordinates (X, Y, Z).

    The in-plane curve is the affine image of the unit superellipse,

    .. math::
        x(\theta) = a \rho(\theta; n) \cos\theta, \qquad
        y(\theta) = b \rho(\theta; n) \sin\theta,

    which satisfies :math:`|x/a|^n + |y/b|^n = 1` exactly, since
    :math:`\rho^n \left(|\cos\theta|^n + |\sin\theta|^n\right) = 1`. Note that
    :math:`\theta` is not the polar angle of the curve point unless
    :math:`a = b`. The plane is then rotated by the normalized quaternion and
    translated by the center.

    Args:
        dofs (array, shape (10,)): Array of dofs.
        quadpoints (array, shape (N,)): Array of quadrature points.

    Returns:
        Array of curve points, shape (N, 3)
    """
    a = dofs[0]
    b = dofs[1]
    n = dofs[2]
    q = dofs[3:7]
    center = dofs[7:]

    norm_q = jnp.linalg.norm(q)
    q_norm = jnp.where(norm_q < 1e-8,
                       q / (norm_q + 1e-8),  # safe division when norm is small
                       q / norm_q)  # this shouldn't happen if the quaternion dofs are properly initialized

    theta = 2 * np.pi * quadpoints  # quadpoints is an angle in [0, 1)
    rho = superellipse_rho(theta, n)
    x_curve_in_plane = a * rho * jnp.cos(theta)
    y_curve_in_plane = b * rho * jnp.sin(theta)

    gamma_x = (1.0 - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3])) * x_curve_in_plane \
        + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * y_curve_in_plane \
        + center[0]
    gamma_y = (1.0 - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3])) * y_curve_in_plane \
        + 2 * (q_norm[0] * q_norm[3] + q_norm[1] * q_norm[2]) * x_curve_in_plane \
        + center[1]
    gamma_z = 2 * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * x_curve_in_plane \
        + 2 * (q_norm[0] * q_norm[1] + q_norm[2] * q_norm[3]) * y_curve_in_plane \
        + center[2]
    # apply the quaternion rotation
    return jnp.stack((gamma_x, gamma_y, gamma_z), axis=-1)


class CurveSuperEllipse(JaxCurve):
    r"""
    ``CurveSuperEllipse`` is a curve that is restricted to lie in a plane. The
    shape of the curve within the plane is a superellipse,

    .. math::
        \left|\frac{x}{a}\right|^n + \left|\frac{y}{b}\right|^n = 1,

    with semi-axis :math:`a` along the plane's local :math:`x` direction and
    :math:`b` along its local :math:`y`. It is evaluated pointwise with no
    intermediate Fourier representation, so the semi-axes are reproduced
    exactly. The planar curve is then rotated by a quaternion and translated by
    the center point (X, Y, Z), following the same convention as
    :class:`simsopt.geo.CurvePlanarFourier`.

    The exponent :math:`n` is an optimizable degree of freedom like any other,
    and the shape is continuous in it: :math:`n = 2` is an ellipse, values
    between 3 and 5 give progressively squarer coils, and :math:`n = 1` is a
    diamond whose corners have unbounded curvature. Because the enclosed area
    :math:`4ab\,\Gamma(1+1/n)^2/\Gamma(1+2/n)` increases monotonically with
    :math:`n`, an optimizer will drive it to its upper bound unless the
    objective contains a counteracting curvature or coil-coil distance
    penalty. Use ``set_lower_bound('n', ...)`` and ``set_upper_bound(...)`` to
    restrict the range.

    The dofs are stored in the order

    .. math::
       [a, b, n, q0, qi, qj, qk, X, Y, Z]

    Args:
        quadpoints (array): Array of quadrature points, or the number thereof.
        a (float): Initial semi-axis along the plane's local :math:`x`.
        b (float): Initial semi-axis along the plane's local :math:`y`.
        n (float): Initial superellipse exponent. Must exceed 1.
        dofs (array, optional): Array of dofs. When given, the shared dofs
            take precedence and ``a``, ``b``, ``n`` seed only the local copy.
    """
    def __init__(self, quadpoints, a, b, n, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1., quadpoints, endpoint=False)
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        if a <= 0 or b <= 0:
            raise ValueError(f"The semi-axes must be positive; got a={a}, b={b}.")
        if n <= 1:
            raise ValueError(
                f"The exponent must satisfy n > 1 to keep the curve convex "
                f"with bounded curvature; got n={n}.")
        self._a = a
        self._b = b
        self._n = n
        self.coefficients = np.array([a, b, n, 1., 0., 0., 0., 0., 0., 0.])
        if dofs is None:
            super().__init__(quadpoints, curve_superellipse_pure,
                             x0=self.coefficients,
                             external_dof_setter=CurveSuperEllipse.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, curve_superellipse_pure,
                             dofs=dofs,
                             external_dof_setter=CurveSuperEllipse.set_dofs_impl,
                             names=self._make_names())
        self.set_lower_bound('n', 1.0) # pyright: ignore[reportArgumentType]
        self.set_lower_bound('a', 1e-12) # pyright: ignore[reportArgumentType]
        self.set_lower_bound('b', 1e-12) # pyright: ignore[reportArgumentType]
        self.dkappa_by_dcoeff_vjp_jax = jit(
            lambda x, v: vjp(lambda d: safe_kappa_pure(self.gammadash_jax(d),
                                                       self.gammadashdash_jax(d)),
                             x)[1](v)[0])

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 10

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return self.coefficients

    def set_dofs_impl(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.coefficients[:] = dofs

    def _make_names(self):
        """
        This function returns the names of the dofs associated to this object.

        Returns:
            List of dof names.
        """
        shape_names = ['a', 'b', 'n']
        quaternion_names = ['q0', 'qi', 'qj', 'qk']
        center_names = ['X', 'Y', 'Z']
        return shape_names + quaternion_names + center_names

    def area(self):
        r"""
        This function returns the area enclosed by the curve,
        :math:`4ab\,\Gamma(1+1/n)^2/\Gamma(1+2/n)`.
        """
        a, b, n = self.coefficients[0], self.coefficients[1], self.coefficients[2]
        return 4 * a * b * gamma_fn(1 + 1 / n)**2 / gamma_fn(1 + 2 / n)


ALL_DOF_NAMES = ['a', 'b', 'n', 'q0', 'qi', 'qj', 'qk', 'X', 'Y', 'Z']
PLACEMENT_DOF_NAMES = ['q0', 'qi', 'qj', 'qk', 'X', 'Y', 'Z']
SHAPE_DOF_NAMES = ['a', 'b', 'n']


class ScaledCurveSuperEllipse(sopp.Curve, Curve):
    r"""
    ``ScaledCurveSuperEllipse`` is a :class:`CurveSuperEllipse` that shares 
    dofs in `shared_dofs` with `curve_to_scale`.

    Used, for example, for an array of dipole coils on an axisymmetric winding
    surface, where every coil shares one poloidal semi-axis and one exponent
    but the toroidal semi-axis grows with the local major radius.

    Args:
        curve_to_scale (CurveSuperEllipse): The base curve that supplies the global dofs.
        shared_dofs (tuple, optional): Tuple of dofs to share with `curve_to_scale`.
        a (float, optional): Initial semi-axis along the plane's local :math:`x`.
        b (float, optional): Initial semi-axis along the plane's local :math:`y`.
        n (float, optional): Initial superellipse exponent. Must exceed 1.
        dofs (array, optional): Array of dofs.
    """

    def __init__(self, curve_to_scale, shared_dofs=(), a=0.1, b=0.1, n=3.0, dofs=None):
        bad = [n for n in shared_dofs if n not in SHAPE_DOF_NAMES]
        if bad:
            raise ValueError(f"shared dofs must be among {SHAPE_DOF_NAMES}; got {bad}.")
        self._shared_dofs = tuple(n for n in SHAPE_DOF_NAMES if n in shared_dofs)
        self._local_names = tuple(n for n in SHAPE_DOF_NAMES if n not in shared_dofs)
        self._local_idx = [ALL_DOF_NAMES.index(n) for n in self._local_names]
        self._shared_idx = [ALL_DOF_NAMES.index(name) for name in self._shared_dofs]
        
        self.curve_to_scale = curve_to_scale
        for name in PLACEMENT_DOF_NAMES + list(self._local_names):
            self.curve_to_scale.fix(name)

        given = {"a": a, "b": b, "n": n}
        values = {name: (given[name] if name in self._local_names
                         else curve_to_scale.get(name))
                  for name in SHAPE_DOF_NAMES}
        x0 = np.array([values[name] for name in self._local_names])
        self.curve = CurveSuperEllipse(
            curve_to_scale.quadpoints, values['a'], values['b'], values['n'])
        sopp.Curve.__init__(self, self.curve.quadpoints)
        kwargs = ({"x0": x0} if dofs is None else {"dofs": dofs})
        Curve.__init__(self, names=self._make_names(), depends_on=[curve_to_scale],
                       external_dof_setter=ScaledCurveSuperEllipse.set_dofs_impl, **kwargs)
        
        if 'n' in self._local_names:
            self.set_lower_bound('n', 1.0) # pyright: ignore[reportArgumentType]
        if 'a' in self._local_names:
            self.set_lower_bound('a', 1e-12) # pyright: ignore[reportArgumentType]
        if 'b' in self._local_names:
            self.set_lower_bound('b', 1e-12) # pyright: ignore[reportArgumentType]

    @property
    def a(self):
        self._update_curve()
        return float(self.curve.get('a'))

    @property
    def b(self):
        self._update_curve()
        return float(self.curve.get('b'))

    @property
    def n(self):
        self._update_curve()
        return float(self.curve.get('n'))

    def num_dofs(self):
        return len(self._local_names)

    def get_dofs(self):
        return np.array([self.curve.get(n) for n in self._local_names])

    def set_dofs_impl(self, dofs):
        self._update_curve(**dict(zip(self._local_names, map(float, dofs))))

    def _make_names(self):
        return list(self._local_names)

    def _update_curve(self, **local):
        values = {}
        for name in SHAPE_DOF_NAMES:
            if name in local:
                values[name] = local[name]
            elif name in self._local_names:
                values[name] = self.curve.get(name)
            else:
                values[name] = self.curve_to_scale.get(name)
        self.curve.x = np.array([values['a'], values['b'], values['n'],
                                 1., 0., 0., 0., 0., 0., 0.])

    def recompute_bell(self, parent=None):
        """
        This function keeps the scaled curve in step with the parent curve.
        """
        self._update_curve()
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        """
        This function returns the x, y, z coordinates of the curve.
        """
        self._update_curve()
        gamma[:] = self.curve.gamma()

    def gammadash_impl(self, gammadash):
        self._update_curve()
        gammadash[:] = self.curve.gammadash()

    def gammadashdash_impl(self, gammadashdash):
        self._update_curve()
        gammadashdash[:] = self.curve.gammadashdash()

    def gammadashdashdash_impl(self, gammadashdashdash):
        self._update_curve()
        gammadashdashdash[:] = self.curve.gammadashdashdash()

    def _split_vjp(self, dofs_vjp):
        dofs_vjp = np.asarray(dofs_vjp)
        parent_vjp = np.zeros_like(dofs_vjp)
        parent_vjp[self._shared_idx] = dofs_vjp[self._shared_idx]
        return (Derivative({self: dofs_vjp[self._local_idx]}) # pyright: ignore[reportArgumentType]
                + Derivative({self.curve_to_scale: parent_vjp})) # pyright: ignore[reportArgumentType]

    def dgamma_by_dcoeff_vjp(self, v):
        self._update_curve()
        return self._split_vjp(
            self.curve.dgamma_by_dcoeff_vjp_jax(self.curve.get_dofs(), v))

    def dgammadash_by_dcoeff_vjp(self, v):
        self._update_curve()
        return self._split_vjp(
            self.curve.dgammadash_by_dcoeff_vjp_jax(self.curve.get_dofs(), v))

    def dgammadashdash_by_dcoeff_vjp(self, v):
        self._update_curve()
        return self._split_vjp(
            self.curve.dgammadashdash_by_dcoeff_vjp_jax(self.curve.get_dofs(), v))

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        self._update_curve()
        return self._split_vjp(
            self.curve.dgammadashdashdash_by_dcoeff_vjp_jax(self.curve.get_dofs(), v))

    def dkappa_by_dcoeff_vjp(self, v):
        self._update_curve()
        return self._split_vjp(
            self.curve.dkappa_by_dcoeff_vjp_jax(self.curve.get_dofs(), v))
