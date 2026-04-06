import numpy as np
from .curve import JaxCurve, _as_runtime_float64_ref, jax, jnp

__all__ = ["CurveHelical"]


def curve_helical_pure(dofs, quadpoints, order, m, ell, R0, r):
    """Pure function for the position vector used by CurveHelical."""
    dofs = _as_runtime_float64_ref(dofs, reference=quadpoints)
    quadpoints = _as_runtime_float64_ref(quadpoints, reference=dofs)
    A = jax.lax.slice_in_dim(dofs, 0, order + 1, axis=0)
    B = jnp.concatenate(
        (
            _as_runtime_float64_ref(np.zeros(1, dtype=np.float64), reference=dofs),
            jax.lax.slice_in_dim(dofs, order + 1, dofs.shape[0], axis=0),
        )
    )  # Add 0 at the start for k = 0
    two_pi = _as_runtime_float64_ref(2.0 * np.pi, reference=quadpoints)
    ell_scale = _as_runtime_float64_ref(float(ell), reference=quadpoints)
    m_scale = _as_runtime_float64_ref(float(m), reference=quadpoints)
    phi = quadpoints * two_pi * ell_scale
    mode_numbers = _as_runtime_float64_ref(
        np.arange(order + 1, dtype=np.float64),
        reference=phi,
    )
    k, phi_2d = jnp.meshgrid(mode_numbers, phi)
    phase = k * phi_2d * m_scale / ell_scale
    eta = m_scale * phi / ell_scale + jnp.sum(
        A * jnp.cos(phase) + B * jnp.sin(phase), axis=1
    )
    R0_scale = _as_runtime_float64_ref(float(R0), reference=eta)
    r_scale = _as_runtime_float64_ref(float(r), reference=eta)
    R = R0_scale + r_scale * jnp.cos(eta)
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    z = -r_scale * jnp.sin(eta)
    gamma = jnp.column_stack((x, y, z))
    return gamma


class CurveHelical(JaxCurve):
    r"""A helical curve lying on a circular-cross-section axisymmetric torus.

    This curve type can describe the coils from the
    `Hanson & Cary (1984) <https://doi.org/10.1063/1.864692>`__
    and
    `Cary & Hanson (1986) <https://doi.org/10.1063/1.865539>`__
    papers on optimization for reduced stochasticity, as well as the Compact
    Toroidal Hybrid (CTH) device at Auburn University.

    The helical curve lies on a circular-cross-section axisymmetric torus with
    major radius :math:`R_0` and minor radius :math:`r`. Following Hanson and
    Cary, the position along the curve is written in terms of a poloidal angle
    on the winding surface :math:`\eta` and the standard toroidal angle :math:`\phi`:

    .. math::
        \begin{align*}
        x &= [R_0 + r \cos(\eta)] \cos(\phi), \\
        y &= [R_0 + r \cos(\eta)] \sin(\phi), \\
        z &= -r \sin(\eta).
        \end{align*}

    Along the curve, the poloidal angle depends on the toroidal angle both
    through a secular linear term and periodic Fourier terms:

    .. math:: 
        \eta = \frac{m \phi}{l}
        + A_0 + \sum_{k=1}^N \left[ A_k \cos(k \phi m / l) + B_k \sin(k \phi m / l) \right]

    When the toroidal angle :math:`\phi` increases by :math:`2 \pi l`, the
    poloidal angle :math:`\eta` increases by :math:`2 \pi m`, so the "rotational
    transform" of the curve is :math:`m / l`. The shape of the helical curve on
    the surface can be adjusted through the :math:`A_k` and :math:`B_k` Fourier
    coefficients, which are the optimizable degrees of freedom for this class.
    Although :math:`\phi` in the above formulas is periodic in the range
    :math:`[0, 2 \pi)`, the ``quadpoints`` argument should be in the range
    :math:`[0, 1)` as usual for simsopt curves.

    The 1986 Cary-Hanson paper has a more general parameterization than the 1984
    Hanson-Cary paper, relaxing the constraint that the curves lie on a circle
    in the R-z plane, although no results in the paper use this freedom. We do
    not consider this more general parameterization here in this class. For a
    more general parameterization of helical curves and coils in simsopt, see
    :class:`simsopt.geo.CurveXYZFourierSymmetries`.

    The order in which the degrees of freedom are stored in the state vector
    ``x`` is

    .. math::
        A_0, A_1, \ldots, A_N, B_1, \ldots, B_N.

    The default values of ``m``, ``ell``, ``R0``, and ``r`` correspond to the
    coils in the Hanson-Cary and Cary-Hanson papers.

    Args:
        quadpoints: (int or array-like) Grid points (or number thereof) along the curve.
        order:  Maximum (inclusive) Fourier mode number :math:`N` for :math:`A_k` and :math:`B_k`.
        m:  Integer :math:`m` describing helicity of the coil.
        ell:  Integer :math:`l` describing helicity of the coil.
        R0:  Major radius of the toroidal surface on which the coil lies.
        r:  Minor radius of the toroidal surface on which the coil lies.
    """

    def __init__(self, quadpoints, order, m=5, ell=2, R0=1.0, r=0.3, **kwargs):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        def pure(dofs, points):
            return curve_helical_pure(dofs, points, order, m, ell, R0, r)

        self.order = order
        self.m = m
        self.ell = ell
        self.R0 = R0
        self.r = r
        self.coefficients = np.zeros(self.num_dofs())
        dof_names = [f"A_{i}" for i in range(order + 1)] + [
            f"B_{i}" for i in range(1, order + 1)
        ]

        if "dofs" not in kwargs:
            if "x0" not in kwargs:
                kwargs["x0"] = self.coefficients
            else:
                self.set_dofs_impl(kwargs["x0"])

        super().__init__(quadpoints, pure, names=dof_names, **kwargs)

    def num_dofs(self):
        return 1 + 2 * self.order

    def get_dofs(self):
        return self.coefficients

    def set_dofs_impl(self, dofs):
        self.coefficients[:] = dofs

    def to_spec(self):
        """Build an immutable JAX geometry spec from the current curve state."""
        from ..jax_core import make_curve_helical_spec

        return make_curve_helical_spec(
            dofs=self.get_dofs(),
            quadpoints=self.quadpoints,
            order=self.order,
            m=self.m,
            ell=self.ell,
            R0=self.R0,
            r=self.r,
        )
