import numpy as np
import jax
import jax.numpy as jnp
import simsoptpp as sopp
from .curve import Curve, JaxCurve, _install_curve_jax_contract

__all__ = ['CurvePlanarFourier', 'JaxCurvePlanarFourier']


def _normalized_quaternion(quaternion):
    norm_sq = jnp.sum(quaternion * quaternion)
    inv_norm = jnp.where(norm_sq > 0.0, jax.lax.rsqrt(norm_sq), 1.0)
    return quaternion * inv_norm


def _quaternion_rotation_matrix(quaternion):
    q0, q1, q2, q3 = quaternion
    return jnp.array(
        [
            [
                1.0 - 2.0 * (q2 * q2 + q3 * q3),
                2.0 * (q1 * q2 - q3 * q0),
                2.0 * (q1 * q3 + q2 * q0),
            ],
            [
                2.0 * (q1 * q2 + q3 * q0),
                1.0 - 2.0 * (q1 * q1 + q3 * q3),
                2.0 * (q2 * q3 - q1 * q0),
            ],
            [
                2.0 * (q1 * q3 - q2 * q0),
                2.0 * (q2 * q3 + q1 * q0),
                1.0 - 2.0 * (q1 * q1 + q2 * q2),
            ],
        ],
        dtype=jnp.float64,
    )


def curveplanarfourier_pure(dofs, quadpoints, order):
    rc_end = order + 1
    rs_end = rc_end + order

    rc = dofs[:rc_end]
    rs = dofs[rc_end:rs_end]
    quaternion = _normalized_quaternion(dofs[rs_end : rs_end + 4])
    center = dofs[rs_end + 4 :]

    phi = 2.0 * jnp.pi * quadpoints
    cosphi = jnp.cos(phi)
    sinphi = jnp.sin(phi)

    radius = rc[0] * jnp.ones_like(phi)
    if order > 0:
        modes = jnp.arange(1, order + 1, dtype=jnp.float64)
        phase = phi[:, None] * modes[None, :]
        radius = radius + jnp.sum(
            rc[1:][None, :] * jnp.cos(phase) + rs[None, :] * jnp.sin(phase),
            axis=1,
        )

    base_curve = jnp.column_stack(
        (
            radius * cosphi,
            radius * sinphi,
            jnp.zeros_like(phi),
        )
    )
    rotation = _quaternion_rotation_matrix(quaternion)
    return base_curve @ rotation.T + center[None, :]


def jaxplanarcurve_pure(dofs, quadpoints, order):
    return curveplanarfourier_pure(dofs, quadpoints, order)


class CurvePlanarFourier(sopp.CurvePlanarFourier, Curve):
    r"""
    ``CurvePlanarFourier`` is a curve that is restricted to lie in a plane. The
    shape of the curve within the plane is represented by a Fourier series in
    polar coordinates centered at the center of curve. 
    The resulting planar curve is then rotated in three
    dimensions using a quaternion, and finally a translation is applied by the center point
    (X, Y, Z). The Fourier series in polar coordinates is

    .. math::

       r(\phi) = \sum_{m=0}^{\text{order}} r_{c,m}\cos(m \phi) + \sum_{m=1}^{\text{order}} r_{s,m}\sin(m \phi).

    The rotation quaternion is

    .. math::

       \bf{q} &= [q0,qi,qj,qk]

       &= [\cos(\theta / 2), \hat{x}\sin(\theta / 2), \hat{y}\sin(\theta / 2), \hat{z}\sin(\theta / 2)]

    where :math:`\theta` is the counterclockwise rotation angle about a unit axis
    :math:`(\hat{x},\hat{y},\hat{z})`. Details of the quaternion rotation can be
    found for example in pages 575-576 of
    https://www.cis.upenn.edu/~cis5150/ws-book-Ib.pdf.


    A quaternion is used for rotation rather than other methods for rotation to
    prevent gimbal locking during optimization. The quaternion is normalized
    before being applied to prevent scaling of the curve. The dofs themselves are not normalized. This
    results in a redundancy in the optimization, where several different sets of 
    dofs may correspond to the same normalized quaternion. Normalizing the dofs 
    directly would create a dependence between the quaternion dofs, which may cause 
    issues during optimization.

    The dofs are stored in the order

    .. math::
       [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, q0, qi, qj, qk, X, Y, Z]

    Args:
        quadpoints (array): Array of quadrature points.
        order (int): Order of the Fourier series.
        dofs (array, optional): Array of dofs.
    """

    def __init__(self, quadpoints, order, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1., quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.CurvePlanarFourier.__init__(self, quadpoints, order)
        if dofs is None:
            Curve.__init__(self, external_dof_setter=CurvePlanarFourier.set_dofs_impl,
                           names=self._make_names(order),
                           x0=self.get_dofs())
        else:
            Curve.__init__(self, external_dof_setter=CurvePlanarFourier.set_dofs_impl,
                           dofs=dofs,
                           names=self._make_names(order))
        _install_curve_jax_contract(
            self,
            lambda dofs, points: curveplanarfourier_pure(dofs, points, order),
        )

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.asarray(sopp.CurvePlanarFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.local_x = dofs
        sopp.CurvePlanarFourier.set_dofs(self, dofs)

    def _make_names(self, order):
        """
        This function returns the names of the dofs associated to this object.

        Args:
            order (int): Order of the Fourier series.

        Returns:
            List of dof names.
        """
        x_names = ['rc(0)']
        x_cos_names = [f'rc({i})' for i in range(1, order + 1)]
        x_sin_names = [f'rs({i})' for i in range(1, order + 1)]

        x_names += x_cos_names + x_sin_names
        y_names = ['q0', 'qi', 'qj', 'qk']
        z_names = ['X', 'Y', 'Z']
        return x_names + y_names + z_names


class JaxCurvePlanarFourier(JaxCurve):
    r"""
    A Python+Jax implementation of the CurvePlanarFourier class.
    """

    def __init__(self, quadpoints, order, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        def pure(local_dofs, points):
            return jaxplanarcurve_pure(local_dofs, points, order)

        self.order = order
        self.dof_list = np.zeros(2 * order + 1 + 4 + 3)
        if dofs is None:
            super().__init__(
                quadpoints,
                pure,
                x0=self.dof_list,
                names=self._make_names(order),
                external_dof_setter=JaxCurvePlanarFourier.set_dofs_impl,
            )
        else:
            super().__init__(
                quadpoints,
                pure,
                dofs=dofs,
                names=self._make_names(order),
                external_dof_setter=JaxCurvePlanarFourier.set_dofs_impl,
            )

    def num_dofs(self):
        return 2 * self.order + 1 + 4 + 3

    def get_dofs(self):
        return np.array(self.dof_list)

    def set_dofs_impl(self, dofs):
        self.dof_list = np.array(dofs)

    def _make_names(self, order):
        x_names = ['rc(0)']
        x_cos_names = [f'rc({i})' for i in range(1, order + 1)]
        x_sin_names = [f'rs({i})' for i in range(1, order + 1)]
        x_names += x_cos_names + x_sin_names
        y_names = ['q0', 'qi', 'qj', 'qk']
        z_names = ['X', 'Y', 'Z']
        return x_names + y_names + z_names
