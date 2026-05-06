import numpy as np

import simsoptpp as sopp
from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier

__all__ = ['SurfaceXYZFourier']


def _surface_copy_quadpoints_kwargs(surface, kwargs):
    otherntheta = surface.quadpoints_theta.size
    othernphi = surface.quadpoints_phi.size

    ntheta = kwargs.pop("ntheta", otherntheta)
    nphi = kwargs.pop("nphi", othernphi)
    grid_range = kwargs.pop("range", None)
    quadpoints_theta = kwargs.pop("quadpoints_theta", None)
    quadpoints_phi = kwargs.pop("quadpoints_phi", None)

    if quadpoints_theta is None and quadpoints_phi is None:
        if (
            ntheta != otherntheta
            or nphi != othernphi
            or grid_range is not None
        ):
            kwargs["quadpoints_phi"], kwargs["quadpoints_theta"] = (
                Surface.get_quadpoints(
                    ntheta=ntheta,
                    nphi=nphi,
                    nfp=surface.nfp,
                    range=grid_range,
                )
            )
        else:
            kwargs["quadpoints_phi"] = surface.quadpoints_phi
            kwargs["quadpoints_theta"] = surface.quadpoints_theta
    else:
        if quadpoints_theta is None:
            if ntheta != otherntheta or grid_range is not None:
                kwargs["quadpoints_theta"] = Surface.get_theta_quadpoints(ntheta)
            else:
                kwargs["quadpoints_theta"] = surface.quadpoints_theta
        else:
            kwargs["quadpoints_theta"] = quadpoints_theta
        if quadpoints_phi is None:
            if nphi != othernphi or grid_range is not None:
                kwargs["quadpoints_phi"] = Surface.get_phi_quadpoints(
                    nphi,
                    range=grid_range,
                    nfp=surface.nfp,
                )
            else:
                kwargs["quadpoints_phi"] = surface.quadpoints_phi
        else:
            kwargs["quadpoints_phi"] = quadpoints_phi
    return kwargs


def _copy_xyz_fourier_coefficients(source, target, coefficient_names):
    max_m = min(source.mpol, target.mpol)
    max_abs_n = min(source.ntor, target.ntor)
    for coefficient_name in coefficient_names:
        source_coefficients = getattr(source, coefficient_name)
        target_coefficients = getattr(target, coefficient_name)
        for m in range(max_m + 1):
            for n in range(-max_abs_n, max_abs_n + 1):
                target_coefficients[m, n + target.ntor] = (
                    source_coefficients[m, n + source.ntor]
                )


class SurfaceXYZFourier(sopp.SurfaceXYZFourier, Surface):
    r"""`SurfaceXYZFourier` is a surface that is represented in Cartesian
    coordinates using the following Fourier series:

    .. math::
        \hat x(\phi,\theta) &= \sum_{m=0}^{m_\text{pol}} \sum_{n=-n_{\text{tor}}}^{n_{tor}} [
              x_{c,m,n} \cos(m \theta - n_\text{ fp } n \phi)
            + x_{s,m,n} \sin(m \theta - n_\text{ fp } n \phi)]\\
        \hat y(\phi,\theta) &= \sum_{m=0}^{m_\text{pol}} \sum_{n=-n_\text{tor}}^{n_\text{tor}} [
              y_{c,m,n} \cos(m \theta - n_\text{fp} n \phi)
            + y_{s,m,n} \sin(m \theta - n_\text{fp} n \phi)]\\
        z(\phi,\theta) &= \sum_{m=0}^{m_\text{pol}} \sum_{n=-n_\text{tor}}^{n_\text{tor}} [
              z_{c,m,n} \cos(m \theta - n_\text{fp}n \phi)
            + z_{s,m,n} \sin(m \theta - n_\text{fp}n \phi)]

    where

    .. math::
        x &= \hat x \cos(\phi) - \hat y \sin(\phi)\\
        y &= \hat x \sin(\phi) + \hat y \cos(\phi)

    Note that for :math:`m=0` we skip the :math:`n<0` term for the cos
    terms, and the :math:`n \leq 0` for the sin terms.

    When enforcing stellarator symmetry, we set the

    .. math::
        x_{s,*,*}, ~y_{c,*,*}, \text{and} ~z_{c,*,*}

    terms to zero.

    For more information about the arguments `quadpoints_phi``, and
    ``quadpoints_theta``, see the general documentation on :ref:`surfaces`.
    Instead of supplying the quadrature point arrays along :math:`\phi` and
    :math:`\theta` directions, one could also specify the number of
    quadrature points for :math:`\phi` and :math:`\theta` using the
    class method :py:meth:`~simsopt.geo.surface.Surface.from_nphi_ntheta`.

    Args:
        nfp: The number of field periods.
        stellsym: Whether the surface is stellarator-symmetric, i.e.
          symmetry under rotation by :math:`\pi` about the x-axis.
        mpol: Maximum poloidal mode number included.
        ntor: Maximum toroidal mode number included, divided by ``nfp``.
        quadpoints_phi: Set this to a list or 1D array to set the :math:`\phi_j` grid points directly.
        quadpoints_theta: Set this to a list or 1D array to set the :math:`\theta_j` grid points directly.
    """

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=0,
                 quadpoints_phi=None, quadpoints_theta=None,
                 dofs=None):

        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints(nfp=nfp)

        sopp.SurfaceXYZFourier.__init__(self, mpol, ntor, nfp, stellsym,
                                        quadpoints_phi, quadpoints_theta)
        self.xc[0, ntor] = 1.0
        self.xc[1, ntor] = 0.1
        self.zs[1, ntor] = 0.1
        if dofs is None:
            Surface.__init__(self, x0=self.get_dofs(), names=self._make_names(),
                             external_dof_setter=SurfaceXYZFourier.set_dofs_impl)
        else:
            Surface.__init__(self, dofs=dofs,
                             external_dof_setter=SurfaceXYZFourier.set_dofs_impl)

    def _make_names(self):
        """
        Form a list of names of the ``xc``, ``ys``, ``zs``, ``xs``,
        ``yc``, or ``zc`` array elements. The order of these four arrays
        here must match the order in ``set_dofs_impl()`` and ``get_dofs()``
        in ``src/simsoptpp/surfacexyzfourier.h``.
        """
        if self.stellsym:
            names = self._make_names_helper('xc', True) \
                + self._make_names_helper('ys', False) \
                + self._make_names_helper('zs', False)
        else:
            names = self._make_names_helper('xc', True) \
                + self._make_names_helper('xs', False) \
                + self._make_names_helper('yc', True) \
                + self._make_names_helper('ys', False) \
                + self._make_names_helper('zc', True) \
                + self._make_names_helper('zs', False)
        return names

    def _make_names_helper(self, prefix, include0):
        """
        Helper function for `_make_names` method. Forms array of coefficients
        for :math:'m = [0, m_{pol}]' and :math:'n = [-n_{tor}, n_{tor}]'. If :math:'m = 0', only
        positive values of :math:'n' are used. If it is a cosine term, the :math:'(0,0)' term is included.

        Args:
            prefix: The prefix for the name of the coefficients.
            include0: Whether to include the (0,0) term.
        """
        if include0:
            names = [prefix + "(0,0)"]
        else:
            names = []

        names += [prefix + '(0,' + str(n) + ')' for n in range(1, self.ntor + 1)]
        for m in range(1, self.mpol + 1):
            names += [prefix + '(' + str(m) + ',' + str(n) + ')' for n in range(-self.ntor, self.ntor + 1)]
        return names

    def get_dofs(self):
        """
        Return the dofs associated to this surface.
        """
        return np.asarray(sopp.SurfaceXYZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        Set the dofs associated to this surface.
        """
        self.local_full_x = dofs

    def surface_spec(self):
        """Build an immutable JAX geometry spec from the current surface state."""
        from ..jax_core import make_surface_xyz_fourier_spec

        return make_surface_xyz_fourier_spec(
            dofs=self.get_dofs(),
            quadpoints_phi=self.quadpoints_phi,
            quadpoints_theta=self.quadpoints_theta,
            nfp=self.nfp,
            stellsym=self.stellsym,
            mpol=self.mpol,
            ntor=self.ntor,
        )

    def to_spec(self):
        """Alias for :meth:`surface_spec`."""
        return self.surface_spec()

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def copy(self, **kwargs):
        """
        Return a copy of the ``SurfaceXYZFourier`` object, but with the specified
        attributes changed.
        """
        mpol = kwargs.pop("mpol", self.mpol)
        ntor = kwargs.pop("ntor", self.ntor)
        nfp = kwargs.pop("nfp", self.nfp)
        stellsym = kwargs.pop("stellsym", self.stellsym)
        kwargs = _surface_copy_quadpoints_kwargs(self, kwargs)

        surf = SurfaceXYZFourier(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            **kwargs,
        )
        surf.x[:] = 0
        _copy_xyz_fourier_coefficients(self, surf, ("xc", "ys", "zs"))
        if not surf.stellsym and not self.stellsym:
            _copy_xyz_fourier_coefficients(self, surf, ("xs", "yc", "zc"))
        surf.local_full_x = surf.get_dofs()
        return surf

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        copied = self.copy()
        memo[id(self)] = copied
        return copied

    def to_RZFourier(self):
        """
        Return a SurfaceRZFourier instance corresponding to the shape of this
        surface.
        """
        ntor = self.ntor
        mpol = self.mpol
        surf = SurfaceRZFourier(nfp=self.nfp,
                                stellsym=self.stellsym,
                                mpol=mpol,
                                ntor=ntor,
                                quadpoints_phi=self.quadpoints_phi,
                                quadpoints_theta=self.quadpoints_theta)

        gamma = np.zeros((surf.quadpoints_phi.size, surf.quadpoints_theta.size, 3))
        for idx in range(gamma.shape[0]):
            gamma[idx, :, :] = self.cross_section(surf.quadpoints_phi[idx])

        surf.least_squares_fit(gamma)
        return surf

    def extend_via_normal(self, distance):
        """
        Extend the surface in the normal direction by a uniform distance.

        Args:
            distance: The distance to extend the surface.
        """
        self._extend_via_normal_for_nonuniform_phi(distance)


    return_fn_map = {'area': sopp.SurfaceXYZFourier.area,
                     'volume': sopp.SurfaceXYZFourier.volume,
                     'aspect-ratio': Surface.aspect_ratio}
