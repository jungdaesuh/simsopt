import numpy as np

import simsoptpp as sopp
from .curve import (
    Curve,
    _as_runtime_float64_ref,
    _install_curve_jax_contract,
    _two_pi_like,
    jax,
    jnp,
)

__all__ = ["CurveRZFourier"]


def curverzfourier_pure(dofs, quadpoints, order, nfp, stellsym):
    quadpoints = _as_runtime_float64_ref(quadpoints, reference=dofs)
    phi = _two_pi_like(quadpoints) * quadpoints
    cosphi = jnp.cos(phi)
    sinphi = jnp.sin(phi)

    rc = jax.lax.slice_in_dim(dofs, 0, order + 1, axis=0)
    if stellsym:
        rs = None
        zc = None
        zs = jax.lax.slice_in_dim(dofs, order + 1, dofs.shape[0], axis=0)
    else:
        rs = jax.lax.slice_in_dim(dofs, order + 1, 2 * order + 1, axis=0)
        zc = jax.lax.slice_in_dim(dofs, 2 * order + 1, 3 * order + 2, axis=0)
        zs = jax.lax.slice_in_dim(dofs, 3 * order + 2, dofs.shape[0], axis=0)

    cos_modes = _as_runtime_float64_ref(
        np.arange(order + 1, dtype=np.float64),
        reference=phi,
    )
    nfp_scale = _as_runtime_float64_ref(float(nfp), reference=phi)
    cos_phase = phi[:, None] * (nfp_scale * cos_modes)[None, :]
    radius = jnp.sum(rc[None, :] * jnp.cos(cos_phase), axis=1)

    sin_modes = _as_runtime_float64_ref(
        np.arange(1, order + 1, dtype=np.float64),
        reference=phi,
    )
    if order > 0:
        sin_phase = phi[:, None] * (nfp_scale * sin_modes)[None, :]
        z = jnp.sum(zs[None, :] * jnp.sin(sin_phase), axis=1)
        if not stellsym:
            radius = radius + jnp.sum(rs[None, :] * jnp.sin(sin_phase), axis=1)
    else:
        z = jnp.zeros_like(phi)

    if not stellsym:
        z = z + jnp.sum(zc[None, :] * jnp.cos(cos_phase), axis=1)

    return jnp.column_stack((radius * cosphi, radius * sinphi, z))


class CurveRZFourier(sopp.CurveRZFourier, Curve):
    r"""
    ``CurveRZFourier`` is a curve that is represented in cylindrical
    coordinates using the following Fourier series:

    .. math::
       r(\phi) &= \sum_{m=0}^{\text{order}} r_{c,m}\cos(n_{\text{fp}} m \phi) + \sum_{m=1}^{\text{order}} r_{s,m}\sin(n_{\text{fp}} m \phi) \\
       z(\phi) &= \sum_{m=0}^{\text{order}} z_{c,m}\cos(n_{\text{fp}} m \phi) + \sum_{m=1}^{\text{order}} z_{s,m}\sin(n_{\text{fp}} m \phi)

    If ``stellsym = True``, then the :math:`\sin` terms for :math:`r` and the :math:`\cos` terms for :math:`z` are zero.

    For the ``stellsym = False`` case, the dofs are stored in the order

    .. math::
       [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, z_{c,0},....]

    or in the ``stellsym = True`` case they are stored

    .. math::
       [r_{c,0},...,r_{c,order},z_{s,1},...,z_{s,order}]
    """

    def __init__(self, quadpoints, order, nfp, stellsym, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1.0 / nfp, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.CurveRZFourier.__init__(self, quadpoints, order, nfp, stellsym)
        if dofs is None:
            Curve.__init__(
                self,
                x0=self.get_dofs(),
                names=self._make_names(order, stellsym),
                external_dof_setter=CurveRZFourier.set_dofs_impl,
            )
        else:
            Curve.__init__(
                self, external_dof_setter=CurveRZFourier.set_dofs_impl, dofs=dofs
            )
        _install_curve_jax_contract(
            self,
            lambda dofs, points: curverzfourier_pure(
                dofs,
                points,
                order,
                nfp,
                stellsym,
            ),
        )

    def _make_names(self, order, stellsym):
        r_names = [f"rc({i})" for i in range(0, order + 1)]
        z_names = [f"zs({i})" for i in range(1, order + 1)]
        if not stellsym:
            r_names += [f"rs({i})" for i in range(1, order + 1)]
            z_names = [f"zc({i})" for i in range(0, order + 1)] + z_names
        return r_names + z_names

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.asarray(sopp.CurveRZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.local_x = dofs
        sopp.CurveRZFourier.set_dofs(self, dofs)

    def to_spec(self):
        """Build an immutable JAX geometry spec from the current curve state."""
        from ..jax_core import make_curve_rzfourier_spec

        return make_curve_rzfourier_spec(
            dofs=self.get_dofs(),
            quadpoints=self.quadpoints,
            order=self.order,
            nfp=self.nfp,
            stellsym=self.stellsym,
        )
