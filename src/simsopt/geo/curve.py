from math import sin, cos

import numpy as np

try:
    import jax
    from jax import grad as _jax_grad
    from jax import hessian as _jax_hessian
    from jax import jacfwd as _jax_jacfwd
    from jax import jvp as _jax_jvp
    from jax import vjp as _jax_vjp
    import jax.numpy as jnp
except ImportError:
    jax = None
    _jax_grad = None
    _jax_hessian = None
    _jax_jacfwd = None
    _jax_jvp = None
    _jax_vjp = None
    jnp = np

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import Derivative
from .surfacerzfourier import SurfaceRZFourier
from .surfacexyztensorfourier import SurfaceXYZTensorFourier

from .jit import jit
from .plotting import fix_matplotlib_3d

__all__ = [
    "Curve",
    "JaxCurve",
    "RotatedCurve",
    "curves_to_vtk",
    "create_equally_spaced_curves",
    "create_equally_spaced_oriented_curves",
    "create_equally_spaced_planar_curves",
    "create_planar_curves_between_two_toroidal_surfaces",
]


_HAS_JAX = _jax_vjp is not None


def _as_jax_float64(value):
    if not _HAS_JAX:
        return np.asarray(value, dtype=np.float64)
    if isinstance(value, jax.Array):
        return jnp.asarray(value, dtype=jnp.float64)
    return jax.device_put(np.asarray(value, dtype=np.float64))


def _as_numpy_float64(value):
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    if not _HAS_JAX:
        return np.asarray(value, dtype=np.float64)
    return np.asarray(jax.device_get(value), dtype=np.float64)


def _require_jax(feature_name):
    if not _HAS_JAX:
        raise ImportError(
            f"JAX is required for {feature_name}. Install simsopt[JAX] or simsopt[JAX_GPU]."
        )


def _missing_jax_callable(feature_name):
    def _missing(*args, **kwargs):
        del args, kwargs
        _require_jax(feature_name)

    return _missing


vjp = _jax_vjp or _missing_jax_callable("curve reverse-mode derivatives")
jacfwd = _jax_jacfwd or _missing_jax_callable("curve forward-mode derivatives")
jvp = _jax_jvp or _missing_jax_callable("curve JVP geometry helpers")
hessian = _jax_hessian or _missing_jax_callable("curve Hessian helpers")
grad = _jax_grad or _missing_jax_callable("curve gradient helpers")

if _HAS_JAX:
    __all__.append("CurveCWSFourier")


@jit
def centroid_pure(gamma, gammadash):
    r"""
    This pure function is used in a Python+Jax implementation of formula for centroid.

    .. math::
        \mathbf{c} = \frac{1}{L} \int_0^L \mathbf{\gamma}(l) dl

    where :math:`\gamma` is the position vector on the curve.
    """
    arclength = jnp.linalg.norm(gammadash, axis=-1)
    centroid = jnp.sum(gamma * arclength[:, None], axis=0) / jnp.sum(arclength)
    return centroid


@jit
def incremental_arclength_pure(d1gamma):
    """
    This function is used in a Python+Jax implementation of the curve arc length formula.
    """

    return jnp.linalg.norm(d1gamma, axis=1)


incremental_arclength_vjp = jit(
    lambda d1gamma, v: vjp(lambda d1g: incremental_arclength_pure(d1g), d1gamma)[1](v)[
        0
    ]
)


@jit
def kappa_pure(d1gamma, d2gamma):
    """
    This function is used in a Python+Jax implementation of formula for curvature.
    """

    return (
        jnp.linalg.norm(jnp.cross(d1gamma, d2gamma), axis=1)
        / jnp.linalg.norm(d1gamma, axis=1) ** 3
    )


kappavjp0 = jit(
    lambda d1gamma, d2gamma, v: vjp(lambda d1g: kappa_pure(d1g, d2gamma), d1gamma)[1](
        v
    )[0]
)
kappavjp1 = jit(
    lambda d1gamma, d2gamma, v: vjp(lambda d2g: kappa_pure(d1gamma, d2g), d2gamma)[1](
        v
    )[0]
)
kappagrad0 = jit(
    lambda d1gamma, d2gamma: jacfwd(lambda d1g: kappa_pure(d1g, d2gamma))(d1gamma)
)
kappagrad1 = jit(
    lambda d1gamma, d2gamma: jacfwd(lambda d2g: kappa_pure(d1gamma, d2g))(d2gamma)
)


@jit
def torsion_pure(d1gamma, d2gamma, d3gamma):
    """
    This function is used in a Python+Jax implementation of formula for torsion.
    """

    return jnp.sum(jnp.cross(d1gamma, d2gamma, axis=1) * d3gamma, axis=1) / jnp.sum(
        jnp.cross(d1gamma, d2gamma, axis=1) ** 2, axis=1
    )


torsionvjp0 = jit(
    lambda d1gamma, d2gamma, d3gamma, v: vjp(
        lambda d1g: torsion_pure(d1g, d2gamma, d3gamma), d1gamma
    )[1](v)[0]
)
torsionvjp1 = jit(
    lambda d1gamma, d2gamma, d3gamma, v: vjp(
        lambda d2g: torsion_pure(d1gamma, d2g, d3gamma), d2gamma
    )[1](v)[0]
)
torsionvjp2 = jit(
    lambda d1gamma, d2gamma, d3gamma, v: vjp(
        lambda d3g: torsion_pure(d1gamma, d2gamma, d3g), d3gamma
    )[1](v)[0]
)


@jit
def frenet_frame_pure(gammadash, gammadashdash, incremental_arclength):
    r"""
    This function returns the Frenet frame, :math:`(\mathbf{t}, \mathbf{n}, \mathbf{b})`,
    associated to the curve.
    """

    def norm(a):
        return jnp.linalg.norm(a, axis=1)

    def inner(a, b):
        return jnp.sum(a * b, axis=1)

    tangent = (1.0 / incremental_arclength[:, None]) * gammadash
    tdash = (1.0 / incremental_arclength[:, None]) ** 2 * (
        incremental_arclength[:, None] * gammadashdash
        - (inner(gammadash, gammadashdash) / incremental_arclength)[:, None] * gammadash
    )
    normal = (1.0 / norm(tdash))[:, None] * tdash
    binormal = jnp.cross(tangent, normal, axis=1)
    return tangent, normal, binormal


def _quadpoint_jvp(geometry_fn):
    return lambda dofs, quadpoints: jvp(
        lambda qpts: geometry_fn(dofs, qpts),
        (quadpoints,),
        (jnp.ones_like(quadpoints),),
    )[1]


def _curve_uses_full_dofs(curve):
    return getattr(curve, "_jax_curve_dof_mode", "local") == "full"


def _install_curve_jax_contract(curve, gamma_pure, *, dof_mode="local"):
    """Attach the standard JAX geometry/VJP contract to a curve instance."""
    if not _HAS_JAX:
        return

    points = np.asarray(curve.quadpoints)

    curve._jax_curve_dof_mode = dof_mode
    curve.gamma_pure = gamma_pure
    curve.gamma_jax = jit(lambda dofs: gamma_pure(dofs, points))
    curve.gamma_impl_jax = jit(lambda dofs, quadpoints: gamma_pure(dofs, quadpoints))

    curve.gammadash_pure = _quadpoint_jvp(gamma_pure)
    curve.gammadash_jax = jit(lambda dofs: curve.gammadash_pure(dofs, points))

    curve.gammadashdash_pure = _quadpoint_jvp(curve.gammadash_pure)
    curve.gammadashdash_jax = jit(lambda dofs: curve.gammadashdash_pure(dofs, points))

    curve.gammadashdashdash_pure = _quadpoint_jvp(curve.gammadashdash_pure)
    curve.gammadashdashdash_jax = jit(
        lambda dofs: curve.gammadashdashdash_pure(dofs, points)
    )

    curve.dgamma_by_dcoeff_jax = jit(jacfwd(curve.gamma_jax))
    curve.dgamma_by_dcoeff_vjp_jax = jit(
        lambda dofs, v: vjp(curve.gamma_jax, dofs)[1](v)[0]
    )
    curve.dgammadash_by_dcoeff_jax = jit(jacfwd(curve.gammadash_jax))
    curve.dgammadash_by_dcoeff_vjp_jax = jit(
        lambda dofs, v: vjp(curve.gammadash_jax, dofs)[1](v)[0]
    )
    curve.dgammadashdash_by_dcoeff_jax = jit(jacfwd(curve.gammadashdash_jax))
    curve.dgammadashdash_by_dcoeff_vjp_jax = jit(
        lambda dofs, v: vjp(curve.gammadashdash_jax, dofs)[1](v)[0]
    )
    curve.dgammadashdashdash_by_dcoeff_jax = jit(jacfwd(curve.gammadashdashdash_jax))
    curve.dgammadashdashdash_by_dcoeff_vjp_jax = jit(
        lambda dofs, v: vjp(curve.gammadashdashdash_jax, dofs)[1](v)[0]
    )


def _extract_full_subgraph_dofs(owner, subcurve, owner_dofs):
    """Rebuild ``subcurve.full_x`` from an owning wrapper's flat full-DOF vector."""
    owner_dofs = jnp.asarray(owner_dofs, dtype=jnp.float64)
    if owner is subcurve:
        return owner_dofs

    subcurve_full_dofs = jnp.asarray(subcurve.full_x, dtype=jnp.float64)
    for opt, (sub_start, sub_end) in subcurve._full_dof_indices.items():
        owner_start, owner_end = owner._full_dof_indices[opt]
        subcurve_full_dofs = subcurve_full_dofs.at[sub_start:sub_end].set(
            owner_dofs[owner_start:owner_end]
        )
    return subcurve_full_dofs


def _curve_jax_arg_from_full_dofs(owner, subcurve, owner_dofs):
    """Return the DOF block expected by ``subcurve.gamma_jax`` / ``gammadash_jax``."""
    subcurve_full_dofs = _extract_full_subgraph_dofs(owner, subcurve, owner_dofs)
    if _curve_uses_full_dofs(subcurve):
        return subcurve_full_dofs

    start, end = subcurve._full_dof_indices[subcurve]
    return subcurve_full_dofs[start:end]


def _optimizable_local_full_dofs_from_full_dofs(owner, opt, owner_dofs):
    """Extract one Optimizable's local full DOFs from an owner's full-DOF vector."""
    owner_dofs = jnp.asarray(owner_dofs, dtype=jnp.float64)
    start, end = owner._full_dof_indices[opt]
    return owner_dofs[start:end]


def _optimizable_dof_map_components(owner, opt):
    """Return immutable mapping metadata from an owner's full DOFs into ``opt``."""
    template_full_dofs = jnp.asarray(opt.full_x, dtype=jnp.float64)
    owner_segments = tuple(
        (
            int(owner._full_dof_indices[dep_opt][0]),
            int(owner._full_dof_indices[dep_opt][1]),
            int(sub_start),
            int(sub_end),
        )
        for dep_opt, (sub_start, sub_end) in opt._full_dof_indices.items()
    )
    if _curve_uses_full_dofs(opt):
        input_mode = "full"
        input_start = 0
        input_end = int(template_full_dofs.shape[0])
    else:
        input_mode = "local"
        input_start, input_end = opt._full_dof_indices[opt]
    return (
        template_full_dofs,
        owner_segments,
        input_mode,
        int(input_start),
        int(input_end),
    )


def _optimizable_dof_map_spec(owner, opt):
    """Build the immutable owner-to-optimizable DOF map spec for ``opt``."""
    from ..jax_core import make_optimizable_dof_map_spec

    (
        template_full_dofs,
        owner_segments,
        input_mode,
        input_start,
        input_end,
    ) = _optimizable_dof_map_components(owner, opt)
    return make_optimizable_dof_map_spec(
        template_full_dofs=template_full_dofs,
        owner_segments=owner_segments,
        input_mode=input_mode,
        input_start=input_start,
        input_end=input_end,
    )


def _curve_jax_eval_from_arg(curve, method_name, curve_dofs, surf_dofs=None):
    """Evaluate a curve JAX method for either local-DOF or surface-coupled curves."""
    curve_dofs = jnp.asarray(curve_dofs, dtype=jnp.float64)
    method = getattr(curve, method_name)
    if _curve_uses_full_dofs(curve):
        return method(curve_dofs)

    surf = getattr(curve, "surf", None)
    if surf is not None and surf.dof_size > 0:
        if surf_dofs is None:
            surf_dofs = surf.get_dofs()
        surf_dofs = jnp.asarray(surf_dofs, dtype=jnp.float64)
        return method(curve_dofs, surf_dofs)
    return method(curve_dofs)


class Curve(Optimizable):
    """
    Curve  is a base class for various representations of curves in SIMSOPT
    using the graph based Optimizable framework with external handling of DOFS
    as well.
    """

    def __init__(self, **kwargs):
        Optimizable.__init__(self, **kwargs)

    def recompute_bell(self, parent=None):
        """
        For derivative classes of Curve, all of which also subclass
        from C++ Curve class, call invalidate_cache which is implemented
        in C++ side.
        """
        self.invalidate_cache()

    def plot(
        self,
        engine="matplotlib",
        ax=None,
        show=True,
        plot_derivative=False,
        close=False,
        axis_equal=True,
        **kwargs,
    ):
        """
        Plot the curve in 3D using ``matplotlib.pyplot``, ``mayavi``, or ``plotly``.

        Args:
            engine: The graphics engine to use. Available settings are ``"matplotlib"``, ``"mayavi"``, and ``"plotly"``.
            ax: The axis object on which to plot. This argument is useful when plotting multiple
              objects on the same axes. If equal to the default ``None``, a new axis will be created.
            show: Whether to call the ``show()`` function of the graphics engine. Should be set to
              ``False`` if more objects will be plotted on the same axes.
            plot_derivative: Whether to plot the tangent of the curve too. Not implemented for plotly.
            close: Whether to connect the first and last point on the
              curve. Can lead to surprising results when only quadrature points
              on a part of the curve are considered, e.g. when exploting rotational symmetry.
            axis_equal: For matplotlib, whether all three dimensions should be scaled equally.
            kwargs: Any additional arguments to pass to the plotting function, like ``color='r'``.

        Returns:
            An axis which could be passed to a further call to the graphics engine
            so multiple objects are shown together.
        """

        def rep(data):
            if close:
                return np.concatenate((data, [data[0]]))
            else:
                return data

        x = rep(self.gamma()[:, 0])
        y = rep(self.gamma()[:, 1])
        z = rep(self.gamma()[:, 2])
        if plot_derivative:
            xt = rep(self.gammadash()[:, 0])
            yt = rep(self.gammadash()[:, 1])
            zt = rep(self.gammadash()[:, 2])

        if engine == "matplotlib":
            # plot in matplotlib.pyplot
            import matplotlib.pyplot as plt

            if ax is None or ax.name != "3d":
                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
            ax.plot(x, y, z, **kwargs)
            if plot_derivative:
                ax.quiver(
                    x,
                    y,
                    z,
                    0.1 * xt,
                    0.1 * yt,
                    0.1 * zt,
                    arrow_length_ratio=0.1,
                    color="r",
                )
            if axis_equal:
                fix_matplotlib_3d(ax)
            if show:
                plt.show()

        elif engine == "mayavi":
            # plot 3D curve in mayavi.mlab
            from mayavi import mlab

            mlab.plot3d(x, y, z, **kwargs)
            if plot_derivative:
                mlab.quiver3d(x, y, z, 0.1 * xt, 0.1 * yt, 0.1 * zt)
            if show:
                mlab.show()

        elif engine == "plotly":
            import plotly.graph_objects as go

            if "color" in list(kwargs.keys()):
                color = kwargs["color"]
                del kwargs["color"]
            else:
                color = "blue"
            kwargs.setdefault("line", go.scatter3d.Line(color=color, width=4))
            if ax is None:
                ax = go.Figure()
            ax.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", **kwargs))
            ax.update_layout(scene_aspectmode="data")
            if show:
                ax.show()
        else:
            raise ValueError(
                "Invalid engine option! Please use one of {matplotlib, mayavi, plotly}."
            )
        return ax

    def dgamma_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dgamma_by_dcoeff_vjp_impl(v)})

    def dgammadash_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dgammadash_by_dcoeff_vjp_impl(v)})

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dgammadashdash_by_dcoeff_vjp_impl(v)})

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return Derivative({self: self.dgammadashdashdash_by_dcoeff_vjp_impl(v)})

    def dincremental_arclength_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \|\Gamma'\|}{\partial \mathbf{c}}

        where :math:`\|\Gamma'\|` is the incremental arclength, :math:`\Gamma'` is the tangent
        to the curve and :math:`\mathbf{c}` are the curve dofs.
        """

        return self.dgammadash_by_dcoeff_vjp(
            _as_numpy_float64(
                incremental_arclength_vjp(
                    _as_jax_float64(self.gammadash()),
                    _as_jax_float64(v),
                )
            )
        )

    def kappa_impl(self, kappa):
        r"""
        This function implements the curvature, :math:`\kappa(\varphi)`.
        """
        kappa[:] = _as_numpy_float64(
            kappa_pure(
                _as_jax_float64(self.gammadash()),
                _as_jax_float64(self.gammadashdash()),
            )
        )

    def dkappa_by_dcoeff_impl(self, dkappa_by_dcoeff):
        r"""
        This function computes the derivative of the curvature with respect to the curve coefficients.

        .. math::
            \frac{\partial \kappa}{\partial \mathbf c}

        where :math:`\mathbf c` are the curve dofs, :math:`\kappa` is the curvature.
        """

        dgamma_by_dphi = self.gammadash()
        dgamma_by_dphidphi = self.gammadashdash()
        dgamma_by_dphidcoeff = self.dgammadash_by_dcoeff()
        dgamma_by_dphidphidcoeff = self.dgammadashdash_by_dcoeff()

        def norm(a):
            return np.linalg.norm(a, axis=1)

        numerator = np.cross(dgamma_by_dphi, dgamma_by_dphidphi)
        denominator = self.incremental_arclength()
        dkappa_by_dcoeff[:, :] = (1 / (denominator**3 * norm(numerator)))[
            :, None
        ] * np.sum(
            numerator[:, :, None]
            * (
                np.cross(
                    dgamma_by_dphidcoeff[:, :, :],
                    dgamma_by_dphidphi[:, :, None],
                    axis=1,
                )
                + np.cross(
                    dgamma_by_dphi[:, :, None],
                    dgamma_by_dphidphidcoeff[:, :, :],
                    axis=1,
                )
            ),
            axis=1,
        ) - (norm(numerator) * 3 / denominator**5)[:, None] * np.sum(
            dgamma_by_dphi[:, :, None] * dgamma_by_dphidcoeff[:, :, :], axis=1
        )

    def torsion_impl(self, torsion):
        r"""
        This function returns the torsion, :math:`\tau`, of a curve.
        """
        torsion[:] = _as_numpy_float64(
            torsion_pure(
                _as_jax_float64(self.gammadash()),
                _as_jax_float64(self.gammadashdash()),
                _as_jax_float64(self.gammadashdashdash()),
            )
        )

    def dtorsion_by_dcoeff_impl(self, dtorsion_by_dcoeff):
        r"""
        This function returns the derivative of torsion with respect to the curve dofs.

        .. math::
            \frac{\partial \tau}{\partial \mathbf c}

        where :math:`\mathbf c` are the curve dofs, and :math:`\tau` is the torsion.
        """
        d1gamma = self.gammadash()
        d2gamma = self.gammadashdash()
        d3gamma = self.gammadashdashdash()
        d1gammadcoeff = self.dgammadash_by_dcoeff()
        d2gammadcoeff = self.dgammadashdash_by_dcoeff()
        d3gammadcoeff = self.dgammadashdashdash_by_dcoeff()
        dtorsion_by_dcoeff[:, :] = (
            np.sum(
                np.cross(d1gamma, d2gamma, axis=1)[:, :, None] * d3gammadcoeff, axis=1
            )
            + np.sum(
                (
                    np.cross(d1gammadcoeff, d2gamma[:, :, None], axis=1)
                    + np.cross(d1gamma[:, :, None], d2gammadcoeff, axis=1)
                )
                * d3gamma[:, :, None],
                axis=1,
            )
        ) / np.sum(np.cross(d1gamma, d2gamma, axis=1) ** 2, axis=1)[:, None]
        dtorsion_by_dcoeff[:, :] -= (
            np.sum(np.cross(d1gamma, d2gamma, axis=1) * d3gamma, axis=1)[:, None]
            * np.sum(
                2
                * np.cross(d1gamma, d2gamma, axis=1)[:, :, None]
                * (
                    np.cross(d1gammadcoeff, d2gamma[:, :, None], axis=1)
                    + np.cross(d1gamma[:, :, None], d2gammadcoeff, axis=1)
                ),
                axis=1,
            )
            / np.sum(np.cross(d1gamma, d2gamma, axis=1) ** 2, axis=1)[:, None] ** 2
        )

    def dkappa_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \kappa}{\partial \mathbf{c}}

        where :math:`\mathbf c` are the curve dofs and :math:`\kappa` is the curvature.
        """

        return self.dgammadash_by_dcoeff_vjp(
            _as_numpy_float64(
                kappavjp0(
                    _as_jax_float64(self.gammadash()),
                    _as_jax_float64(self.gammadashdash()),
                    _as_jax_float64(v),
                )
            )
        ) + self.dgammadashdash_by_dcoeff_vjp(
            _as_numpy_float64(
                kappavjp1(
                    _as_jax_float64(self.gammadash()),
                    _as_jax_float64(self.gammadashdash()),
                    _as_jax_float64(v),
                )
            )
        )

    def dtorsion_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \tau}{\partial \mathbf{c}}

        where :math:`\mathbf c` are the curve dofs, and :math:`\tau` is the torsion.
        """

        return (
            self.dgammadash_by_dcoeff_vjp(
                _as_numpy_float64(
                    torsionvjp0(
                        _as_jax_float64(self.gammadash()),
                        _as_jax_float64(self.gammadashdash()),
                        _as_jax_float64(self.gammadashdashdash()),
                        _as_jax_float64(v),
                    )
                )
            )
            + self.dgammadashdash_by_dcoeff_vjp(
                _as_numpy_float64(
                    torsionvjp1(
                        _as_jax_float64(self.gammadash()),
                        _as_jax_float64(self.gammadashdash()),
                        _as_jax_float64(self.gammadashdashdash()),
                        _as_jax_float64(v),
                    )
                )
            )
            + self.dgammadashdashdash_by_dcoeff_vjp(
                _as_numpy_float64(
                    torsionvjp2(
                        _as_jax_float64(self.gammadash()),
                        _as_jax_float64(self.gammadashdash()),
                        _as_jax_float64(self.gammadashdashdash()),
                        _as_jax_float64(v),
                    )
                )
            )
        )

    def frenet_frame(self):
        r"""
        This function returns the Frenet frame, :math:`(\mathbf{t}, \mathbf{n}, \mathbf{b})`,
        associated to the curve.
        """
        return frenet_frame_pure(
            self.gammadash(), self.gammadashdash(), self.incremental_arclength()
        )

    def kappadash(self):
        r"""
        This function returns :math:`\kappa'(\phi)`, where :math:`\kappa` is the curvature.
        """
        dkappa_by_dphi = np.zeros((len(self.quadpoints),))
        dgamma = self.gammadash()
        d2gamma = self.gammadashdash()
        d3gamma = self.gammadashdashdash()

        def norm(a):
            return np.linalg.norm(a, axis=1)

        def inner(a, b):
            return np.sum(a * b, axis=1)

        def cross(a, b):
            return np.cross(a, b, axis=1)

        dkappa_by_dphi[:] = (
            inner(cross(dgamma, d2gamma), cross(dgamma, d3gamma))
            / (norm(cross(dgamma, d2gamma)) * norm(dgamma) ** 3)
            - 3
            * inner(dgamma, d2gamma)
            * norm(cross(dgamma, d2gamma))
            / norm(dgamma) ** 5
        )
        return dkappa_by_dphi

    def dfrenet_frame_by_dcoeff(self):
        r"""
        This function returns the derivative of the curve's Frenet frame,

        .. math::
            \left(\frac{\partial \mathbf{t}}{\partial \mathbf{c}}, \frac{\partial \mathbf{n}}{\partial \mathbf{c}}, \frac{\partial \mathbf{b}}{\partial \mathbf{c}}\right),

        with respect to the curve dofs, where :math:`(\mathbf t, \mathbf n, \mathbf b)` correspond to the Frenet frame, and :math:`\mathbf c` are the curve dofs.
        """
        dgamma_by_dphi = self.gammadash()
        d2gamma_by_dphidphi = self.gammadashdash()
        d2gamma_by_dphidcoeff = self.dgammadash_by_dcoeff()
        d3gamma_by_dphidphidcoeff = self.dgammadashdash_by_dcoeff()

        l = self.incremental_arclength()
        dl_by_dcoeff = self.dincremental_arclength_by_dcoeff()

        def norm(a):
            return np.linalg.norm(a, axis=1)

        def inner(a, b):
            return np.sum(a * b, axis=1)

        N = len(self.quadpoints)
        dt_by_dcoeff, dn_by_dcoeff, db_by_dcoeff = (
            np.zeros((N, 3, self.num_dofs())),
            np.zeros((N, 3, self.num_dofs())),
            np.zeros((N, 3, self.num_dofs())),
        )
        t, n, b = self.frenet_frame()

        dt_by_dcoeff[:, :, :] = (
            -(dl_by_dcoeff[:, None, :] / l[:, None, None] ** 2)
            * dgamma_by_dphi[:, :, None]
            + d2gamma_by_dphidcoeff / l[:, None, None]
        )

        tdash = (1.0 / l[:, None]) ** 2 * (
            l[:, None] * d2gamma_by_dphidphi
            - (inner(dgamma_by_dphi, d2gamma_by_dphidphi) / l)[:, None] * dgamma_by_dphi
        )

        dtdash_by_dcoeff = (-2 * dl_by_dcoeff[:, None, :] / l[:, None, None] ** 3) * (
            l[:, None] * d2gamma_by_dphidphi
            - (inner(dgamma_by_dphi, d2gamma_by_dphidphi) / l)[:, None] * dgamma_by_dphi
        )[:, :, None] + (1.0 / l[:, None, None]) ** 2 * (
            dl_by_dcoeff[:, None, :] * d2gamma_by_dphidphi[:, :, None]
            + l[:, None, None] * d3gamma_by_dphidphidcoeff
            - (
                inner(d2gamma_by_dphidcoeff, d2gamma_by_dphidphi[:, :, None])[
                    :, None, :
                ]
                / l[:, None, None]
            )
            * dgamma_by_dphi[:, :, None]
            - (
                inner(dgamma_by_dphi[:, :, None], d3gamma_by_dphidphidcoeff)[:, None, :]
                / l[:, None, None]
            )
            * dgamma_by_dphi[:, :, None]
            + (
                inner(dgamma_by_dphi, d2gamma_by_dphidphi)[:, None, None]
                * dl_by_dcoeff[:, None, :]
                / l[:, None, None] ** 2
            )
            * dgamma_by_dphi[:, :, None]
            - (inner(dgamma_by_dphi, d2gamma_by_dphidphi) / l)[:, None, None]
            * d2gamma_by_dphidcoeff
        )
        dn_by_dcoeff[:, :, :] = (1.0 / norm(tdash))[
            :, None, None
        ] * dtdash_by_dcoeff - (
            inner(tdash[:, :, None], dtdash_by_dcoeff)[:, None, :]
            / inner(tdash, tdash)[:, None, None] ** 1.5
        ) * tdash[:, :, None]

        db_by_dcoeff[:, :, :] = np.cross(
            dt_by_dcoeff, n[:, :, None], axis=1
        ) + np.cross(t[:, :, None], dn_by_dcoeff, axis=1)
        return dt_by_dcoeff, dn_by_dcoeff, db_by_dcoeff

    def dkappadash_by_dcoeff(self):
        r"""
        This function returns

        .. math::
            \frac{\partial \kappa'(\phi)}{\partial \mathbf{c}}.

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\kappa` is the curvature.
        """

        dkappadash_by_dcoeff = np.zeros((len(self.quadpoints), self.num_dofs()))
        dgamma = self.gammadash()
        d2gamma = self.gammadashdash()
        d3gamma = self.gammadashdashdash()

        def norm(a):
            return np.linalg.norm(a, axis=1)

        def inner(a, b):
            return np.sum(a * b, axis=1)

        def cross(a, b):
            return np.cross(a, b, axis=1)

        d1_dot_d2 = inner(dgamma, d2gamma)
        d1_x_d2 = cross(dgamma, d2gamma)
        d1_x_d3 = cross(dgamma, d3gamma)
        normdgamma = norm(dgamma)
        norm_d1_x_d2 = norm(d1_x_d2)
        dgamma_dcoeff_ = self.dgammadash_by_dcoeff()
        d2gamma_dcoeff_ = self.dgammadashdash_by_dcoeff()
        d3gamma_dcoeff_ = self.dgammadashdashdash_by_dcoeff()
        for i in range(self.num_dofs()):
            dgamma_dcoeff = dgamma_dcoeff_[:, :, i]
            d2gamma_dcoeff = d2gamma_dcoeff_[:, :, i]
            d3gamma_dcoeff = d3gamma_dcoeff_[:, :, i]

            d1coeff_x_d2 = cross(dgamma_dcoeff, d2gamma)
            d1coeff_dot_d2 = inner(dgamma_dcoeff, d2gamma)
            d1coeff_x_d3 = cross(dgamma_dcoeff, d3gamma)
            d1_x_d2coeff = cross(dgamma, d2gamma_dcoeff)
            d1_dot_d2coeff = inner(dgamma, d2gamma_dcoeff)
            d1_dot_d1coeff = inner(dgamma, dgamma_dcoeff)
            d1_x_d3coeff = cross(dgamma, d3gamma_dcoeff)

            dkappadash_by_dcoeff[:, i] = (
                (
                    +inner(d1coeff_x_d2 + d1_x_d2coeff, d1_x_d3)
                    + inner(d1_x_d2, d1coeff_x_d3 + d1_x_d3coeff)
                )
                / (norm_d1_x_d2 * normdgamma**3)
                - inner(d1_x_d2, d1_x_d3)
                * (
                    inner(d1coeff_x_d2 + d1_x_d2coeff, d1_x_d2)
                    / (norm_d1_x_d2**3 * normdgamma**3)
                    + 3 * inner(dgamma, dgamma_dcoeff) / (norm_d1_x_d2 * normdgamma**5)
                )
                - 3
                * (
                    +(d1coeff_dot_d2 + d1_dot_d2coeff) * norm_d1_x_d2 / normdgamma**5
                    + d1_dot_d2
                    * inner(d1coeff_x_d2 + d1_x_d2coeff, d1_x_d2)
                    / (norm_d1_x_d2 * normdgamma**5)
                    - 5 * d1_dot_d2 * norm_d1_x_d2 * d1_dot_d1coeff / normdgamma**7
                )
            )
        return dkappadash_by_dcoeff

    def centroid(self):
        r"""
        Compute the centroid of the curve.
        """
        return centroid_pure(self.gamma(), self.gammadash())


class JaxCurve(sopp.Curve, Curve):
    def __init__(self, quadpoints, gamma_pure, **kwargs):
        _require_jax(type(self).__name__)

        if isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.Curve.__init__(self, quadpoints)
        if "external_dof_setter" not in kwargs:
            kwargs["external_dof_setter"] = sopp.Curve.set_dofs_impl
        # We are not doing the same search for x0
        Curve.__init__(self, **kwargs)
        self.gamma_pure = gamma_pure
        points = np.asarray(self.quadpoints)
        ones = jnp.ones_like(points)

        self.gamma_jax = jit(lambda dofs: self.gamma_pure(dofs, points))
        self.gamma_impl_jax = jit(lambda dofs, p: self.gamma_pure(dofs, p))
        self.dgamma_by_dcoeff_jax = jit(jacfwd(self.gamma_jax))
        self.dgamma_by_dcoeff_vjp_jax = jit(
            lambda x, v: vjp(self.gamma_jax, x)[1](v)[0]
        )

        self.gammadash_pure = lambda x, q: jvp(
            lambda p: self.gamma_pure(x, p), (q,), (ones,)
        )[1]
        self.gammadash_jax = jit(lambda x: self.gammadash_pure(x, points))
        self.dgammadash_by_dcoeff_jax = jit(jacfwd(self.gammadash_jax))
        self.dgammadash_by_dcoeff_vjp_jax = jit(
            lambda x, v: vjp(self.gammadash_jax, x)[1](v)[0]
        )

        self.gammadashdash_pure = lambda x, q: jvp(
            lambda p: self.gammadash_pure(x, p), (q,), (ones,)
        )[1]
        self.gammadashdash_jax = jit(lambda x: self.gammadashdash_pure(x, points))
        self.dgammadashdash_by_dcoeff_jax = jit(jacfwd(self.gammadashdash_jax))
        self.dgammadashdash_by_dcoeff_vjp_jax = jit(
            lambda x, v: vjp(self.gammadashdash_jax, x)[1](v)[0]
        )

        self.gammadashdashdash_pure = lambda x, q: jvp(
            lambda p: self.gammadashdash_pure(x, p), (q,), (ones,)
        )[1]
        self.gammadashdashdash_jax = jit(
            lambda x: self.gammadashdashdash_pure(x, points)
        )
        self.dgammadashdashdash_by_dcoeff_jax = jit(jacfwd(self.gammadashdashdash_jax))
        self.dgammadashdashdash_by_dcoeff_vjp_jax = jit(
            lambda x, v: vjp(self.gammadashdashdash_jax, x)[1](v)[0]
        )

        self.dkappa_by_dcoeff_vjp_jax = jit(
            lambda x, v: vjp(
                lambda d: kappa_pure(self.gammadash_jax(d), self.gammadashdash_jax(d)),
                x,
            )[1](v)[0]
        )

        self.dtorsion_by_dcoeff_vjp_jax = jit(
            lambda x, v: vjp(
                lambda d: torsion_pure(
                    self.gammadash_jax(d),
                    self.gammadashdash_jax(d),
                    self.gammadashdashdash_jax(d),
                ),
                x,
            )[1](v)[0]
        )

    def set_dofs(self, dofs):
        self.local_x = dofs
        sopp.Curve.set_dofs(self, dofs)

    def gamma_impl(self, gamma, quadpoints):
        r"""
        This function returns the x,y,z coordinates of the curve :math:`\Gamma`.
        """

        gamma[:, :] = self.gamma_impl_jax(self.get_dofs(), quadpoints)

    def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        dgamma_by_dcoeff[:, :, :] = self.dgamma_by_dcoeff_jax(self.get_dofs())

    def dgamma_by_dcoeff_vjp_impl(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        return self.dgamma_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def gammadash_impl(self, gammadash):
        r"""
        This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        gammadash[:, :] = self.gammadash_jax(self.get_dofs())

    def dgammadash_by_dcoeff_impl(self, dgammadash_by_dcoeff):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        dgammadash_by_dcoeff[:, :, :] = self.dgammadash_by_dcoeff_jax(self.get_dofs())

    def dgammadash_by_dcoeff_vjp_impl(self, v):
        r"""
        This function returns

        .. math::
            \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        return self.dgammadash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def gammadashdash_impl(self, gammadashdash):
        r"""
        This function returns :math:`\Gamma''(\varphi)`, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        gammadashdash[:, :] = self.gammadashdash_jax(self.get_dofs())

    def dgammadashdash_by_dcoeff_impl(self, dgammadashdash_by_dcoeff):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        dgammadashdash_by_dcoeff[:, :, :] = self.dgammadashdash_by_dcoeff_jax(
            self.get_dofs()
        )

    def dgammadashdash_by_dcoeff_vjp_impl(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.

        """

        return self.dgammadashdash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def gammadashdashdash_impl(self, gammadashdashdash):
        r"""
        This function returns :math:`\Gamma'''(\varphi)`, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        gammadashdashdash[:, :] = self.gammadashdashdash_jax(self.get_dofs())

    def dgammadashdashdash_by_dcoeff_impl(self, dgammadashdashdash_by_dcoeff):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma'''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        dgammadashdashdash_by_dcoeff[:, :, :] = self.dgammadashdashdash_by_dcoeff_jax(
            self.get_dofs()
        )

    def dgammadashdashdash_by_dcoeff_vjp_impl(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma'''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.

        """

        return self.dgammadashdashdash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def dkappa_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \kappa}{\partial \mathbf{c}}

        where :math:`\mathbf{c}` are the curve dofs and :math:`\kappa` is the curvature.

        """
        return Derivative({self: self.dkappa_by_dcoeff_vjp_jax(self.get_dofs(), v)})

    def dtorsion_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \tau}{\partial \mathbf{c}}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\tau` is the torsion.

        """

        return Derivative({self: self.dtorsion_by_dcoeff_vjp_jax(self.get_dofs(), v)})


class RotatedCurve(sopp.Curve, Curve):
    """
    RotatedCurve inherits from the Curve base class.  It takes an
    input a Curve, rotates it about the ``z`` axis by a toroidal angle
    ``phi``, and optionally completes a reflection when ``flip=True``.
    """

    def __init__(self, curve, phi, flip):
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve])
        self._phi = phi
        self.rotmat = np.asarray(
            [[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]
        ).T
        if flip:
            self.rotmat = self.rotmat @ np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.rotmatT = self.rotmat.T.copy()

    def change_curve(self, new_curve):
        if isinstance(self.curve, RotatedCurve):
            self.curve.change_curve(new_curve)
        else:
            self.curve = new_curve

    def get_dofs(self):
        """
        RotatedCurve does not have any dofs of its own.
        This function returns null array
        """
        return np.array([])

    def set_dofs_impl(self, d):
        """
        RotatedCurve does not have any dofs of its own.
        This function does nothing.
        """
        pass

    def num_dofs(self):
        """
        This function returns the number of dofs associated to the curve.
        """
        return self.curve.num_dofs()

    def gamma_impl(self, gamma, quadpoints):
        r"""
        This function returns the x,y,z coordinates of the curve, :math:`\Gamma`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        if (
            len(quadpoints) == len(self.curve.quadpoints)
            and np.sum((quadpoints - self.curve.quadpoints) ** 2) < 1e-15
        ):
            gamma[:] = self.curve.gamma() @ self.rotmat
        else:
            self.curve.gamma_impl(gamma, quadpoints)
            gamma[:] = gamma @ self.rotmat

    def gammadash_impl(self, gammadash):
        r"""
        This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        gammadash[:] = self.curve.gammadash() @ self.rotmat

    def gammadashdash_impl(self, gammadashdash):
        r"""
        This function returns :math:`\Gamma''(\varphi)`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        gammadashdash[:] = self.curve.gammadashdash() @ self.rotmat

    def gammadashdashdash_impl(self, gammadashdashdash):
        r"""
        This function returns :math:`\Gamma'''(\varphi)`, where :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        gammadashdashdash[:] = self.curve.gammadashdashdash() @ self.rotmat

    def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        dgamma_by_dcoeff[:] = self.rotmatT @ self.curve.dgamma_by_dcoeff()

    def dgammadash_by_dcoeff_impl(self, dgammadash_by_dcoeff):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.
        """

        dgammadash_by_dcoeff[:] = self.rotmatT @ self.curve.dgammadash_by_dcoeff()

    def dgammadashdash_by_dcoeff_impl(self, dgammadashdash_by_dcoeff):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        dgammadashdash_by_dcoeff[:] = (
            self.rotmatT @ self.curve.dgammadashdash_by_dcoeff()
        )

    def dgammadashdashdash_by_dcoeff_impl(self, dgammadashdashdash_by_dcoeff):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma'''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        dgammadashdashdash_by_dcoeff[:] = (
            self.rotmatT @ self.curve.dgammadashdashdash_by_dcoeff()
        )

    def dgamma_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """
        v = sopp.matmult(v, self.rotmatT)  # v = v @ self.rotmatT
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """
        v = sopp.matmult(v, self.rotmatT)  # v = v @ self.rotmatT
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        v = sopp.matmult(v, self.rotmatT)  # v = v @ self.rotmatT
        return self.curve.dgammadashdash_by_dcoeff_vjp(v)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \Gamma'''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z
        coordinates of the curve.

        """

        v = sopp.matmult(v, self.rotmatT)  # v = v @ self.rotmatT
        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v)

    @property
    def flip(self):
        return True if self.rotmat[2][2] == -1 else False


def curves_to_vtk(curves, filename, close=False, extra_data=None):
    """
    Export a list of Curve objects in VTK format, so they can be
    viewed using Paraview. This function requires the python package ``pyevtk``,
    which can be installed using ``pip install pyevtk``.

    Args:
        curves: A python list of Curve objects.
        filename: Name of the file to write.
        close: Whether to draw the segment from the last quadrature point back to the first.
    """
    from pyevtk.hl import polyLinesToVTK

    def wrap(data):
        return np.concatenate([data, [data[0]]])

    if close:
        x = np.concatenate([wrap(c.gamma()[:, 0]) for c in curves])
        y = np.concatenate([wrap(c.gamma()[:, 1]) for c in curves])
        z = np.concatenate([wrap(c.gamma()[:, 2]) for c in curves])
        ppl = np.asarray([c.gamma().shape[0] + 1 for c in curves])
    else:
        x = np.concatenate([c.gamma()[:, 0] for c in curves])
        y = np.concatenate([c.gamma()[:, 1] for c in curves])
        z = np.concatenate([c.gamma()[:, 2] for c in curves])
        ppl = np.asarray([c.gamma().shape[0] for c in curves])
    data = np.concatenate([i * np.ones((ppl[i],)) for i in range(len(curves))])
    pointData = {"idx": data}

    if extra_data is not None:
        pointData = {**pointData, **extra_data}

    polyLinesToVTK(str(filename), x, y, z, pointsPerLine=ppl, pointData=pointData)


def _setup_uniform_grid_in_bounding_box(s_outer, Nx, Ny, Nz, Nmin_factor=2.01):
    """
    Generate a uniform 3D grid of candidate planar coil centers.
    """
    import warnings

    if Nmin_factor <= 2.0:
        warnings.warn("Nmin_factor should be greater than 2.0 to avoid coil overlap.")

    nfp = s_outer.nfp
    xyz_outer = s_outer.gamma().reshape(-1, 3)
    x_outer = xyz_outer[:, 0]
    y_outer = xyz_outer[:, 1]
    z_outer = xyz_outer[:, 2]
    x_max = np.max(x_outer)
    x_min = np.min(x_outer)
    y_max = np.max(y_outer)
    y_min = 0
    z_max = np.max(z_outer)
    z_min = np.min(z_outer)
    z_max = min(z_max, abs(z_min))

    if nfp != 1:
        x_min = 0.0

    dx = (x_max - x_min) / (Nx - 1)
    dy = y_max / (Ny - 1)
    X = np.linspace(dx / 2.0 + x_min, x_max - dx / 2.0, Nx, endpoint=True)
    Y = np.linspace(dy / 2.0 + y_min, y_max - dy / 2.0, Ny, endpoint=True)
    Z = np.linspace(-z_max, z_max, Nz, endpoint=True)

    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    dz = Z[1] - Z[0]
    R = min(dx, min(dy, dz)) / Nmin_factor

    X, Y, Z = np.meshgrid(X, Y, Z, indexing="ij")
    xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(
        Nx * Ny * Nz, 3
    )

    phi0 = np.pi / nfp if s_outer.stellsym else 2 * np.pi / nfp
    nt = 100
    t = np.linspace(0, 2 * np.pi, nt)
    circle_xy = np.zeros((nt, Nx * Ny * Nz, 2))
    circle_xy[:, :, 0] = R * np.outer(np.cos(t), np.ones(Nx * Ny * Nz)) + np.outer(
        np.ones(nt), xyz_uniform[:, 0]
    )
    circle_xy[:, :, 1] = R * np.outer(np.sin(t), np.ones(Nx * Ny * Nz)) + np.outer(
        np.ones(nt), xyz_uniform[:, 1]
    )

    phi = np.arctan2(circle_xy[:, :, 1], circle_xy[:, :, 0])
    remove_inds = np.logical_or(phi >= phi0, phi <= 0)
    xyz_uniform = xyz_uniform[~np.any(remove_inds, axis=0), :]
    return xyz_uniform, R


def create_planar_curves_between_two_toroidal_surfaces(
    s,
    s_inner,
    s_outer,
    Nx=10,
    Ny=10,
    Nz=10,
    order=1,
    use_jax_curve=False,
    numquadpoints=None,
    Nmin_factor=2.01,
):
    """
    Create planar curves initialized on a filtered Cartesian grid between two toroidal surfaces.
    """
    from simsopt.geo import CurvePlanarFourier, JaxCurvePlanarFourier
    from simsopt.field import apply_symmetries_to_curves

    nfp = s.nfp
    stellsym = s.stellsym
    normal_inner = s_inner.unitnormal().reshape(-1, 3)
    xyz_inner = s_inner.gamma().reshape(-1, 3)
    normal_outer = s_outer.unitnormal().reshape(-1, 3)
    xyz_outer = s_outer.gamma().reshape(-1, 3)

    xyz_uniform, R = _setup_uniform_grid_in_bounding_box(
        s_outer, Nx, Ny, Nz, Nmin_factor=Nmin_factor
    )
    contig = np.ascontiguousarray
    grid_xyz = sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
        contig(normal_inner),
        contig(normal_outer),
        contig(xyz_uniform),
        contig(xyz_inner),
        contig(xyz_outer),
    )
    inds = np.ravel(np.logical_not(np.all(grid_xyz == 0.0, axis=-1)))
    grid_xyz = np.array(grid_xyz[inds, :], dtype=float)
    ncoils = grid_xyz.shape[0]
    nquad = (order + 1) * 40 if numquadpoints is None else numquadpoints

    curve_cls = JaxCurvePlanarFourier if use_jax_curve else CurvePlanarFourier
    curves = [curve_cls(nquad, order) for _ in range(ncoils)]
    for ic in range(ncoils):
        alpha2 = (np.random.rand(1) * np.pi - np.pi / 2.0)[0]
        delta2 = (np.random.rand(1) * np.pi)[0]
        calpha2 = np.cos(alpha2)
        salpha2 = np.sin(alpha2)
        cdelta2 = np.cos(delta2)
        sdelta2 = np.sin(delta2)
        dofs = np.zeros(2 * order + 8)
        dofs[0] = R
        dofs[2 * order + 1] = calpha2 * cdelta2
        dofs[2 * order + 2] = salpha2 * cdelta2
        dofs[2 * order + 3] = calpha2 * sdelta2
        dofs[2 * order + 4] = -salpha2 * sdelta2
        dofs[2 * order + 5 : 2 * order + 8] = grid_xyz[ic, :]
        curves[ic].set_dofs(dofs)
        curves[ic].x = curves[ic].x

    all_curves = apply_symmetries_to_curves(curves, nfp, stellsym)
    return curves, all_curves


def create_equally_spaced_curves(
    ncurves,
    nfp,
    stellsym,
    R0=1.0,
    R1=0.5,
    order=6,
    numquadpoints=None,
    use_jax_curve=False,
):
    """
    Create ``ncurves`` curves of type
    :obj:`~simsopt.geo.curvexyzfourier.CurveXYZFourier` of order
    ``order`` that will result in circular equally spaced coils (major
    radius ``R0`` and minor radius ``R1``) after applying
    :obj:`~simsopt.field.coil.coils_via_symmetries`.

    Usage example: create 4 base curves, which are then rotated 3 times and
    flipped for stellarator symmetry:

    .. code-block::

        base_curves = create_equally_spaced_curves(4, 3, stellsym=True)
        base_currents = [Current(1e5) for c in base_curves]
        coils = coils_via_symmetries(base_curves, base_currents, 3, stellsym=True)
    """
    if numquadpoints is None:
        numquadpoints = 15 * order
    curves = []
    from simsopt.geo.curvexyzfourier import CurveXYZFourier, JaxCurveXYZFourier

    curvefunc = JaxCurveXYZFourier if use_jax_curve else CurveXYZFourier
    for i in range(ncurves):
        curve = curvefunc(numquadpoints, order)
        angle = (i + 0.5) * (2 * np.pi) / ((1 + int(stellsym)) * nfp * ncurves)
        curve.set("xc(0)", cos(angle) * R0)
        curve.set("xc(1)", cos(angle) * R1)
        curve.set("yc(0)", sin(angle) * R0)
        curve.set("yc(1)", sin(angle) * R1)
        # The the next line, the minus sign is for consistency with
        # Vmec.external_current(), so the coils create a toroidal field of the
        # proper sign and free-boundary equilibrium works following stage-2 optimization.
        curve.set("zs(1)", -R1)
        curve.x = curve.x  # need to do this to transfer data to C++
        curves.append(curve)
    return curves


def create_equally_spaced_oriented_curves(
    ncurves, nfp, R0, R1, Z0, order, numquadpoints=None
):
    if numquadpoints is None:
        numquadpoints = 15 * order

    curves = []
    from .orientedcurve import OrientedCurveXYZFourier

    phi = np.linspace(0, np.pi / nfp, ncurves, endpoint=False)
    dphi = np.pi / nfp * 1 / ncurves
    phi = phi + dphi / 2
    for ii in range(ncurves):
        c = OrientedCurveXYZFourier(numquadpoints, order)
        c.set("xc(1)", R1)
        c.set("zs(1)", R1)
        c.set("x0", R0 * np.cos(phi[ii]))
        c.set("y0", R0 * np.sin(phi[ii]))
        c.set("z0", Z0)
        c.set("yaw", np.pi / 2 - phi[ii])
        curves.append(c)

    return curves


def create_equally_spaced_planar_curves(
    ncurves,
    nfp,
    stellsym,
    R0=1.0,
    R1=0.5,
    order=6,
    numquadpoints=None,
    use_jax_curve=False,
):
    """
    Create ``ncurves`` curves of type
    :obj:`~simsopt.geo.curveplanarfourier.CurvePlanarFourier` of order
    ``order`` that will result in circular equally spaced coils (major
    radius ``R0`` and minor radius ``R1``) after applying
    :obj:`~simsopt.field.coil.coils_via_symmetries`.
    """

    if numquadpoints is None:
        numquadpoints = 15 * order
    curves = []
    from simsopt.geo.curveplanarfourier import CurvePlanarFourier, JaxCurvePlanarFourier

    curvefunc = JaxCurvePlanarFourier if use_jax_curve else CurvePlanarFourier
    for k in range(ncurves):
        angle = (k + 0.5) * (2 * np.pi) / ((1 + int(stellsym)) * nfp * ncurves)
        curve = curvefunc(numquadpoints, order)

        rcCoeffs = np.zeros(order + 1)
        rcCoeffs[0] = R1
        rsCoeffs = np.zeros(order)
        center = [R0 * cos(angle), R0 * sin(angle), 0]
        rotation = [1, -cos(angle), -sin(angle), 0]
        dofs = np.zeros(len(curve.get_dofs()))

        j = 0
        for i in rcCoeffs:
            dofs[j] = i
            j += 1
        for i in rsCoeffs:
            dofs[j] = i
            j += 1
        for i in rotation:
            dofs[j] = i
            j += 1
        for i in center:
            dofs[j] = i
            j += 1

        curve.set_dofs(dofs)
        curves.append(curve)
    return curves


def gamma_2d(modes, qpts, order, G: int = 0, H: int = 0):
    """Given some dofs, return curve position in 2D cartesian coordinate

    Args:
     - modes: Input dofs. Array of size 2*(2*order+1)
     - qpts: quadrature points. Array of floats from 0 to 1, of size N.
     - order: Maximum Fourier series order.

    Returns:
     - phi: Array of size N x 1.
     - theta: Array of size N x 1.
    """
    # Unpack dofs
    phic = modes[: order + 1]
    phis = modes[order + 1 : 2 * order + 1]
    thetac = modes[2 * order + 1 : 3 * order + 2]
    thetas = modes[3 * order + 2 :]

    # Construct theta and phi arrays
    theta = jnp.zeros((qpts.size,))
    phi = jnp.zeros((qpts.size,))

    ll = qpts * 2.0 * jnp.pi
    for ii in range(order + 1):
        theta = theta + thetac[ii] * jnp.cos(ii * ll)
        phi = phi + phic[ii] * jnp.cos(ii * ll)

    for ii in range(order):
        theta = theta + thetas[ii] * jnp.sin((ii + 1) * ll)
        phi = phi + phis[ii] * jnp.sin((ii + 1) * ll)

    # Add secular terms
    theta = theta + G * qpts
    phi = phi + H * qpts

    return phi, theta


def gamma_curve_on_surface(
    curve_dofs, qpts, order, G, H, surf_dofs, surf_type, mpol, ntor, nfp, stellsym=True
):
    """Returns position in 3D space of a curve lying on a surface

    Args:
     - gamma2d: Curve position in 2D space.
     - surf_dofs: Surface dofs. The surface is assumed to be a surfaceRZFourier object.
     - qpts: Quadrature points. Array of floats of size N, values should be between 0 and 1.
     - mpol: Max poloidal mode number of surface
     - ntor: Max toroidal mode number of surface
     - nfp: Number of field periods.

    Returns:
     - gamma: Position in 3D space. Array of size N x 3.
    """
    phi, theta = gamma_2d(curve_dofs, qpts, order, G, H)

    if surf_type == "RZ_Fourier":
        gamma = surfrz_gamma_lin(phi, theta, mpol, ntor, surf_dofs, nfp, stellsym)
    elif surf_type == "XYZ_Tensor_Fourier":
        gamma = surfxyztensor_gamma_lin(
            phi, theta, mpol, ntor, surf_dofs, nfp, stellsym
        )
    elif surf_type is None:
        return phi, theta

    return gamma


def surfrz_gamma_lin(
    quadpoints_phi, quadpoints_theta, mpol, ntor, surf_dofs, nfp, stellsym
):
    npts = quadpoints_phi.size
    th = quadpoints_theta * 2.0 * jnp.pi
    ph = quadpoints_phi * 2.0 * jnp.pi

    # Construct curve on surface
    r = jnp.zeros((npts,))
    z = jnp.zeros((npts,))

    nmn_cos = ntor + 1 + mpol * (2 * ntor + 1)
    nmn_sin = ntor + mpol * (2 * ntor + 1)

    rc_offset = 0
    rs_offset = nmn_cos
    zc_offset = rs_offset + (0 if stellsym else nmn_sin)
    zs_offset = zc_offset + (0 if stellsym else nmn_cos)

    cos_counter = -1
    sin_counter = -1
    for mm in range(mpol + 1):
        for nn in range(-ntor, ntor + 1):
            angle = mm * th - nn * ph * nfp
            if not (mm == 0 and nn < 0):
                cos_counter += 1
                r = r + surf_dofs[rc_offset + cos_counter] * jnp.cos(angle)
                if not stellsym:
                    z = z + surf_dofs[zc_offset + cos_counter] * jnp.cos(angle)

            if mm == 0 and nn <= 0:
                continue

            sin_counter += 1
            if not stellsym:
                r = r + surf_dofs[rs_offset + sin_counter] * jnp.sin(angle)
            z = z + surf_dofs[zs_offset + sin_counter] * jnp.sin(angle)

    gamma = jnp.zeros((quadpoints_phi.size, 3))
    gamma = gamma.at[:, 0].set(r * jnp.cos(ph))
    gamma = gamma.at[:, 1].set(r * jnp.sin(ph))
    gamma = gamma.at[:, 2].set(z)

    return gamma


def skip(ii, m, n, stellsym, mpol, ntor):
    if not stellsym:
        pass
    if ii == 0:
        return (n <= ntor and m > mpol) or (n > ntor and m <= mpol)
    if ii == 1:
        return (n <= ntor and m <= mpol) or (n > ntor and m > mpol)
    if ii == 2:
        return (n <= ntor and m <= mpol) or (n > ntor and m > mpol)


def read_dofs(dofs, mpol, ntor, stellsym):
    xcs = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))
    ycs = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))
    zcs = jnp.zeros((2 * mpol + 1, 2 * ntor + 1))

    counter = 0
    for m in range(2 * mpol + 1):
        for n in range(2 * ntor + 1):
            if skip(0, m, n, stellsym, mpol, ntor):
                continue
            xcs = xcs.at[m, n].set(dofs[counter])
            counter += 1
    for m in range(2 * mpol + 1):
        for n in range(2 * ntor + 1):
            if skip(1, m, n, stellsym, mpol, ntor):
                continue
            ycs = ycs.at[m, n].set(dofs[counter])
            counter += 1
    for m in range(2 * mpol + 1):
        for n in range(2 * ntor + 1):
            if skip(2, m, n, stellsym, mpol, ntor):
                continue
            zcs = zcs.at[m, n].set(dofs[counter])
            counter += 1

    return xcs, ycs, zcs


def get_coeff(ii, m, n, stellsym, mpol, ntor, xcs, ycs, zcs):
    if skip(ii, m, n, stellsym, mpol, ntor):
        return 0
    if ii == 0:
        return xcs[m, n]
    if ii == 1:
        return ycs[m, n]
    if ii == 2:
        return zcs[m, n]


def basis_fun(n, phi, m, theta, mpol, ntor, nfp):
    if n <= ntor:
        a = jnp.cos(nfp * n * phi)
    else:
        a = jnp.sin(nfp * (n - ntor) * phi)
    if m <= mpol:
        b = jnp.cos(m * theta)
    else:
        b = jnp.sin((m - mpol) * theta)

    return a * b


def surfxyztensor_gamma_lin(qpts_phi, qpts_theta, mpol, ntor, dofs, nfp, stellsym):
    numqpts = qpts_phi.size
    if numqpts != qpts_theta.size:
        raise ValueError("quadpoint_theta and phi should have the same size")

    xcs, ycs, zcs = read_dofs(dofs, mpol, ntor, stellsym)

    data = jnp.zeros((numqpts, 3))
    theta = jnp.pi * 2 * qpts_theta
    phi = jnp.pi * 2 * qpts_phi
    xhat = jnp.zeros((numqpts,))
    yhat = jnp.zeros((numqpts,))
    z = jnp.zeros((numqpts,))
    for m in range(2 * mpol + 1):
        for n in range(2 * ntor + 1):
            xhat += get_coeff(0, m, n, stellsym, mpol, ntor, xcs, ycs, zcs) * basis_fun(
                n, phi, m, theta, mpol, ntor, nfp
            )
            yhat += get_coeff(1, m, n, stellsym, mpol, ntor, xcs, ycs, zcs) * basis_fun(
                n, phi, m, theta, mpol, ntor, nfp
            )
            z += get_coeff(2, m, n, stellsym, mpol, ntor, xcs, ycs, zcs) * basis_fun(
                n, phi, m, theta, mpol, ntor, nfp
            )

    data = data.at[:, 0].set(xhat * jnp.cos(phi) - yhat * jnp.sin(phi))
    data = data.at[:, 1].set(xhat * jnp.sin(phi) + yhat * jnp.cos(phi))
    data = data.at[:, 2].set(z)

    return data


def normal(curve_dofs, qpts, order, G, H, surf_dofs, surf_type, mpol, ntor, nfp):
    """Returns the unitary vector normal to the surface on a curve that lies on the surface

    Args:
     - gamma2d: Curve position in 2D space.
     - surf_dofs: Surface dofs. The surface is assumed to be a surfaceRZFourier object.
     - qpts: Quadrature points. Array of floats of size N, values should be between 0 and 1.
     - mpol: Max poloidal mode number of surface
     - ntor: Max toroidal mode number of surface
     - nfp: Number of field periods.

    Returns:
     - n: Nx3 array; unitary normal vector.
    """
    if not surf_type == "RZ_Fourier":
        raise NotImplementedError("Normal only implemented for SurfaceRZFourier")
    phi, theta = gamma_2d(curve_dofs, qpts, order, G, H)

    # Construct normal on surface
    r = jnp.zeros((qpts.size,))
    drdt = jnp.zeros((qpts.size,))
    drdp = jnp.zeros((qpts.size,))
    dzdt = jnp.zeros((qpts.size,))
    dzdp = jnp.zeros((qpts.size,))

    nmn = ntor + 1 + mpol * (2 * ntor + 1)
    rc = surf_dofs[:nmn]
    zs = surf_dofs[nmn:]

    th = theta * 2.0 * jnp.pi
    ph = phi * 2.0 * jnp.pi * nfp

    counter = -1
    for mm in range(mpol + 1):
        for nn in range(-ntor, ntor + 1):
            if mm == 0 and nn < 0:
                continue
            counter = counter + 1
            r = r + rc[counter] * jnp.cos(mm * th - nn * ph)
            drdt = drdt - mm * rc[counter] * jnp.sin(mm * th - nn * ph)
            drdp = drdp + nn * nfp * rc[counter] * jnp.sin(mm * th - nn * ph)

    counter = -1
    for mm in range(mpol + 1):
        for nn in range(-ntor, ntor + 1):
            if mm == 0 and nn <= 0:
                continue
            counter = counter + 1
            dzdt = dzdt + mm * zs[counter] * jnp.cos(mm * th - nn * ph)
            dzdp = dzdp - nn * nfp * zs[counter] * jnp.cos(mm * th - nn * ph)

    n = jnp.zeros((qpts.size, 3))
    nnorm = jnp.sqrt(r**2 * (drdt**2 + dzdt**2) + (drdp * dzdt - drdt * dzdp) ** 2)

    n = n.at[:, 0].set(r * dzdt / nnorm)
    n = n.at[:, 1].set(-(drdp * dzdt - drdt * dzdp) / nnorm)
    n = n.at[:, 2].set(-r * drdt / nnorm)

    return n


def nfactor(
    curve_dofs,
    qpts,
    order,
    G,
    H,
    sdofs,
    surf_type,
    mpol,
    ntor,
    nfp,
    stellsym,
    direction="z",
):
    """Compute the scalar product between the unitary vector normal to the surface and some direction.

    Args:
     - gamma2d: Curve position in 2D space.
     - surf_dofs: Surface dofs. The surface is assumed to be a surfaceRZFourier object.
     - qpts: Quadrature points. Array of floats of size N, values should be between 0 and 1.
     - mpol: Max poloidal mode number of surface
     - ntor: Max toroidal mode number of surface
     - nfp: Number of field periods.
     - direction: Access direction. For now, only 'r' and 'z' are supported.

    Returns:
     - Scalar product between the unitary normal vector and the access direction.
    """
    if direction == "z":
        # return normal(curve_dofs, qpts, order, G, H, mpol, ntor, sdofs, nfp, stellsym, dgamma_dtheta, dgamma_dphi, surf_type)[:,2]
        return normal(curve_dofs, qpts, order, G, H, sdofs, surf_type, mpol, ntor, nfp)[
            :, 2
        ]
    elif direction == "r":
        # return normal(curve_dofs, qpts, order, G, H, mpol, ntor, sdofs, nfp, stellsym, dgamma_dtheta, dgamma_dphi, surf_type)[:,0]
        return -normal(
            curve_dofs, qpts, order, G, H, sdofs, surf_type, mpol, ntor, nfp
        )[:, 0]


class CurveCWSFourier(Curve, sopp.Curve):
    """Curve that lies on a surface

    This class describes a closed curve constrained to remain on a given surface. Derivatives are provided using JAX, and their implementation is heavily inspired from the class JaxCurve.

    Args:
     - curve2d: Instance of Curve2D
     - surf: Instance of SurfaceRZFourier
    """

    def __init__(self, quadpoints, order, surf, G=0, H=0, **kwargs):
        if isinstance(quadpoints, int):
            quadpoints = jnp.linspace(0, 1, quadpoints, endpoint=False)

        # Curve order. Number of Fourier harmonics for phi and theta
        self.order = order
        self.G = G
        self.H = H

        # Modes are order as phic, phis, thetac, thetas
        self.modes = [
            np.zeros((order + 1,)),
            np.zeros((order,)),
            np.zeros((order + 1,)),
            np.zeros((order,)),
        ]

        # self.quadpoints = quadpoints
        self.surf = surf

        if isinstance(surf, SurfaceRZFourier):
            self.surf_type = "RZ_Fourier"
        elif isinstance(surf, SurfaceXYZTensorFourier):
            self.surf_type = "XYZ_Tensor_Fourier"
        else:
            raise NotImplementedError(
                "CurveCWSFourier is only implemented for SurfaceRZFourier and SurfaceXYZTensorFourier classes."
            )

        # We are not doing the same search for x0
        sopp.Curve.__init__(self, quadpoints)

        Curve.__init__(
            self,
            x0=self.get_dofs(),
            depends_on=[],
            names=self._make_names(),
            external_dof_setter=CurveCWSFourier.set_dofs_impl,
            **kwargs,
        )

        self.gamma_2d_pure = jit(
            lambda cdofs, sdofs, pts: gamma_curve_on_surface(
                cdofs,
                pts,
                self.order,
                self.G,
                self.H,
                sdofs,
                self.surf_type,
                self.surf.mpol,
                self.surf.ntor,
                self.surf.nfp,
                self.surf.stellsym,
            )
        )
        self.gamma_pure = jit(
            lambda dofs, surf_dofs, points: gamma_curve_on_surface(
                dofs,
                points,
                self.order,
                self.G,
                self.H,
                surf_dofs,
                self.surf_type,
                self.surf.mpol,
                self.surf.ntor,
                self.surf.nfp,
                self.surf.stellsym,
            )
        )

        # GAMMA
        points = np.asarray(self.quadpoints)
        ones = jnp.ones_like(points)
        self.gamma_jax = jit(
            lambda cdofs, sdofs: self.gamma_pure(cdofs, sdofs, self.quadpoints)
        )
        self.gamma_impl_jax = jit(
            lambda cdofs, sdofs, p: self.gamma_pure(cdofs, sdofs, p)
        )
        self.gammac_jax = jit(
            lambda dofs: self.gamma_pure(dofs, self.surf.get_dofs(), self.quadpoints)
        )
        self.gammac_impl_jax = jit(
            lambda dofs, p: self.gamma_pure(dofs, self.surf.get_dofs(), p)
        )
        self.gammas_jax = jit(
            lambda sdofs: self.gamma_pure(self.get_dofs(), sdofs, self.quadpoints)
        )
        self.gammas_impl_jax = jit(
            lambda sdofs, p: self.gamma_pure(self.get_dofs(), sdofs, p)
        )

        self.dgamma_by_dcoeff_jax = jit(jacfwd(self.gammac_jax))
        self.dgamma_by_dcoeff_vjp_jax = jit(
            lambda cdofs, v: vjp(self.gammac_jax, cdofs)[1](v)[0]
        )  # derivative w.r.t to curve2d dofs
        self.dgamma_by_dsurf_jax = jit(jacfwd(self.gammas_jax))
        self.dgamma_by_dsurf_vjp_jax = jit(
            lambda sdofs, v: vjp(self.gammas_jax, sdofs)[1](v)[0]
        )  # derivative w.r.t to surface dofs

        self.gammadash_pure = jit(
            lambda cdofs, sdofs, q: jvp(
                lambda p: self.gamma_pure(cdofs, sdofs, p), (q,), (ones,)
            )[1]
        )
        self.gammadash_jax = jit(
            lambda cdofs, sdofs: self.gammadash_pure(cdofs, sdofs, points)
        )
        self.gammacdash_jax = jit(
            lambda cdofs: self.gammadash_pure(cdofs, self.surf.get_dofs(), points)
        )
        self.gammasdash_jax = jit(
            lambda sdofs: self.gammadash_pure(self.get_dofs(), sdofs, points)
        )
        self.dgammadash_by_dcoeff_jax = jit(jacfwd(self.gammacdash_jax))
        self.dgammadash_by_dcoeff_vjp_jax = jit(
            lambda cdofs, v: vjp(self.gammacdash_jax, cdofs)[1](v)[0]
        )
        self.dgammadash_by_dsurf_jax = jit(jacfwd(self.gammasdash_jax))
        self.dgammadash_by_dsurf_vjp_jax = jit(
            lambda sdofs, v: vjp(self.gammasdash_jax, sdofs)[1](v)[0]
        )

        self.gammadashdash_pure = jit(
            lambda cdofs, sdofs, q: jvp(
                lambda p: self.gammadash_pure(cdofs, sdofs, p), (q,), (ones,)
            )[1]
        )
        self.gammadashdash_jax = jit(
            lambda cdofs, sdofs: self.gammadashdash_pure(cdofs, sdofs, points)
        )
        self.gammacdashdash_jax = jit(
            lambda cdofs: self.gammadashdash_pure(cdofs, self.surf.get_dofs(), points)
        )
        self.gammasdashdash_jax = jit(
            lambda sdofs: self.gammadashdash_pure(self.get_dofs(), sdofs, points)
        )
        self.dgammadashdash_by_dcoeff_jax = jit(jacfwd(self.gammacdashdash_jax))
        self.dgammadashdash_by_dcoeff_vjp_jax = jit(
            lambda cdofs, v: vjp(self.gammacdashdash_jax, cdofs)[1](v)[0]
        )
        self.dgammadashdash_by_dsurf_jax = jit(jacfwd(self.gammasdashdash_jax))
        self.dgammadashdash_by_dsurf_vjp_jax = jit(
            lambda sdofs, v: vjp(self.gammasdashdash_jax, sdofs)[1](v)[0]
        )

        self.gammadashdashdash_pure = jit(
            lambda cdofs, sdofs, q: jvp(
                lambda p: self.gammadashdash_pure(cdofs, sdofs, p), (q,), (ones,)
            )[1]
        )
        self.gammadashdashdash_jax = jit(
            lambda cdofs, sdofs: self.gammadashdashdash_pure(cdofs, sdofs, points)
        )
        self.gammacdashdashdash_jax = jit(
            lambda cdofs: self.gammadashdashdash_pure(
                cdofs, self.surf.get_dofs(), points
            )
        )
        self.gammasdashdashdash_jax = jit(
            lambda sdofs: self.gammadashdashdash_pure(self.get_dofs(), sdofs, points)
        )
        self.dgammadashdashdash_by_dcoeff_jax = jit(jacfwd(self.gammacdashdashdash_jax))
        self.dgammadashdashdash_by_dcoeff_vjp_jax = jit(
            lambda x, v: vjp(self.gammacdashdashdash_jax, x)[1](v)[0]
        )
        self.dgammadashdashdash_by_dsurf_jax = jit(jacfwd(self.gammasdashdashdash_jax))
        self.dgammadashdashdash_by_dsurf_vjp_jax = jit(
            lambda x, v: vjp(self.gammasdashdashdash_jax, x)[1](v)[0]
        )

        self.dkappa_by_dcoeff_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(
                lambda x: kappa_pure(
                    self.gammadash_jax(x, sdofs), self.gammadashdash_jax(x, sdofs)
                ),
                cdofs,
            )[1](v)[0]
        )
        self.dkappa_by_dsurf_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(
                lambda x: kappa_pure(
                    self.gammadash_jax(cdofs, x), self.gammadashdash_jax(cdofs, x)
                ),
                sdofs,
            )[1](v)[0]
        )

        self.dtorsion_by_dcoeff_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(
                lambda x: torsion_pure(
                    self.gammadash_jax(x, sdofs),
                    self.gammadashdash_jax(x, sdofs),
                    self.gammadashdashdash_jax(x, sdofs),
                ),
                cdofs,
            )[1](v)[0]
        )
        self.dtorsion_by_dsurf_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(
                lambda x: torsion_pure(
                    self.gammadash_jax(cdofs, x),
                    self.gammadashdash_jax(cdofs, x),
                    self.gammadashdashdash_jax(cdofs, x),
                ),
                sdofs,
            )[1](v)[0]
        )

        # NORMAL
        if self.surf_type == "RZ_Fourier":
            fun = lambda qphi, qtheta: surfrz_gamma_lin(
                qphi,
                qtheta,
                self.surf.mpol,
                self.surf.ntor,
                self.surf.get_dofs(),
                self.surf.nfp,
                self.surf.stellsym,
            )
        elif self.surf_type == "XYZ_Tensor_Fourier":
            fun = lambda qphi, qtheta: surfxyztensor_gamma_lin(
                qphi,
                qtheta,
                self.surf.mpol,
                self.surf.ntor,
                self.surf.get_dofs(),
                self.surf.nfp,
                self.surf.stellsym,
            )

        # quadpoints_phi, quadpoints_theta, mpol, ntor, surf_dofs, nfp, stellsym
        self.dgammalin_by_dtheta = jit(
            lambda quadpoints_phi, quadpoints_theta: jacfwd(fun, argnums=1)(
                quadpoints_phi, quadpoints_theta
            )
        )
        self.dgammalin_by_dphi = jit(
            lambda quadpoints_phi, quadpoints_theta: jacfwd(fun, argnums=0)(
                quadpoints_phi, quadpoints_theta
            )
        )

        # curve_dofs, qpts, order, G, H, sdofs, surf_type, mpol, ntor, nfp, stellsym, dgamma_dtheta, dgamma_dphi, direction
        self.snz = jit(
            lambda cdofs, sdofs: nfactor(
                cdofs,
                quadpoints,
                order,
                G,
                H,
                sdofs,
                self.surf_type,
                self.surf.mpol,
                self.surf.ntor,
                self.surf.nfp,
                self.surf.stellsym,
                direction="z",
            )
        )
        self.snr = jit(
            lambda cdofs, sdofs: nfactor(
                cdofs,
                quadpoints,
                order,
                G,
                H,
                sdofs,
                self.surf_type,
                self.surf.mpol,
                self.surf.ntor,
                self.surf.nfp,
                self.surf.stellsym,
                direction="r",
            )
        )

        self.dsnz_by_dcoeff_jax = lambda cdofs, sdofs: jacfwd(self.snz)(cdofs, sdofs)
        self.dsnr_by_dcoeff_jax = lambda cdofs, sdofs: jacfwd(self.snr)(cdofs, sdofs)
        self.snr_hessian_jax = lambda cdofs, sdofs: hessian(self.snr)(cdofs, sdofs)
        self.snz_hessian_jax = lambda cdofs, sdofs: hessian(self.snz)(cdofs, sdofs)
        self.dsnz_by_dcoeff_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda x: self.snz(x, sdofs), cdofs)[1](v)[0]
        )
        self.dsnr_by_dcoeff_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda x: self.snr(x, sdofs), cdofs)[1](v)[0]
        )

    def set_dofs(self, dofs):
        self.local_x = dofs
        sopp.Curve.set_dofs(self, dofs)

    def num_dofs(self):
        return 2 * (self.order + 1) + 2 * self.order

    def get_dofs(self):
        return np.concatenate(self.modes)

    def set_dofs_impl(self, dofs):
        self.modes[0] = dofs[0 : self.order + 1]
        self.modes[1] = dofs[self.order + 1 : 2 * self.order + 1]
        self.modes[2] = dofs[2 * self.order + 1 : 3 * self.order + 2]
        self.modes[3] = dofs[3 * self.order + 2 : 4 * self.order + 2]

    def _make_names(self):
        dofs_name = []
        for mode in ["phic", "phis", "thetac", "thetas"]:
            for ii in range(self.order + 1):
                if mode == "phis" and ii == 0:
                    continue

                if mode == "thetas" and ii == 0:
                    continue

                dofs_name.append(f"{mode}({ii})")

        return dofs_name

    # GAMMA
    # =====
    def gamma_2d(self):
        return self.gamma_2d_pure(
            self.get_dofs(), self.surf.get_dofs(), self.quadpoints
        )

    def gamma_hessian(self):
        cdofs = self.get_dofs()
        return hessian(self.gammac_jax)(cdofs)

    def gammadash_hessian(self):
        cdofs = self.get_dofs()
        return hessian(self.gammacdash_jax)(cdofs)

    def gammadashdash_hessian(self):
        cdofs = self.get_dofs()
        return hessian(self.gammacdashdash_jax)(cdofs)

    def gamma_impl(self, gamma, quadpoints):
        r"""
        This function returns the x,y,z coordinates of the curve :math:`\Gamma`.
        """
        sdofs = self.surf.get_dofs()
        gamma[:, :] = self.gamma_impl_jax(self.get_dofs(), sdofs, quadpoints)

    def _curve_and_surface_derivative(self, curve_term, surface_term):
        return Derivative({self: curve_term, self.surf: surface_term})

    def dgamma_by_dcoeff_vjp(self, v):
        return self._curve_and_surface_derivative(
            self.dgamma_by_dcoeff_vjp_impl(v),
            self.dgamma_by_dsurf_vjp_impl(v),
        )

    def dgamma_by_dcoeff_impl(self, v):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        v[:, :, :] = self.dgamma_by_dcoeff_jax(self.get_dofs())

    def dgamma_by_dsurf_impl(self, v):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the surface dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """
        v[:, :, :] = self.dgamma_by_dsurf_jax(self.surf.get_dofs())

    def dgamma_by_dcoeff_vjp_impl(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        return self.dgamma_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def dgamma_by_dsurf_vjp_impl(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the surface dofs, and :math:`\Gamma` are the x, y, z coordinates of the curve.
        """

        return self.dgamma_by_dsurf_vjp_jax(self.surf.get_dofs(), v)

    # GAMMA DASH
    # ==========
    def dgammadash_by_dcoeff_vjp(self, v):
        return self._curve_and_surface_derivative(
            self.dgammadash_by_dcoeff_vjp_impl(v),
            self.dgammadash_by_dsurf_vjp_impl(v),
        )

    def gammadash_impl(self, v):
        r"""
        This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        v[:, :] = self.gammadash_jax(self.get_dofs(), self.surf.get_dofs())

    def dgammadash_by_dcoeff_impl(self, v):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        v[:, :, :] = self.dgammadash_by_dcoeff_jax(self.get_dofs())

    def dgammadash_by_dsurf_impl(self, v):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        v[:, :, :] = self.dgammadash_by_dsurf_jax(self.surf.get_dofs())

    def dgammadash_by_dcoeff_vjp_impl(self, v):
        r"""
        This function returns

        .. math::
            \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        return self.dgammadash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def dgammadash_by_dsurf_vjp_impl(self, v):
        r"""
        This function returns

        .. math::
            \mathbf v^T \frac{\partial \Gamma'}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        return self.dgammadash_by_dsurf_vjp_jax(self.surf.get_dofs(), v)

    # GAMMA DASH DASH
    # ===============
    def dgammadashdash_by_dcoeff_vjp(self, v):
        return self._curve_and_surface_derivative(
            self.dgammadashdash_by_dcoeff_vjp_impl(v),
            self.dgammadashdash_by_dsurf_vjp_impl(v),
        )

    def gammadashdash_impl(self, v):
        r"""
        This function returns :math:`\Gamma''(\varphi)`, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        v[:, :] = self.gammadashdash_jax(self.get_dofs(), self.surf.get_dofs())

    def dgammadashdash_by_dcoeff_impl(self, v):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        v[:, :, :] = self.dgammadashdash_by_dcoeff_jax(self.get_dofs())

    def dgammadashdash_by_dsurf_impl(self, v):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the surface dofs, and :math:`\Gamma` are the x, y, z coordinates of the curve.
        """

        v[:, :, :] = self.dgammadashdash_by_dsurf_jax(self.surf.get_dofs())

    def dgammadashdash_by_dcoeff_vjp_impl(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.

        """

        return self.dgammadashdash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def dgammadashdash_by_dsurf_vjp_impl(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the surface dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.

        """

        return self.dgammadashdash_by_dsurf_vjp_jax(self.surf.get_dofs(), v)

    # GAMMA DASH DASH DASH
    # ====================
    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return self._curve_and_surface_derivative(
            self.dgammadashdashdash_by_dcoeff_vjp_impl(v),
            self.dgammadashdashdash_by_dsurf_vjp_impl(v),
        )

    def gammadashdashdash_impl(self, v):
        r"""
        This function returns :math:`\Gamma'''(\varphi)`, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        v[:, :] = self.gammadashdashdash_jax(self.get_dofs(), self.surf.get_dofs())

    def dgammadashdashdash_by_dcoeff_impl(self, v):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma'''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        v[:, :, :] = self.dgammadashdashdash_by_dcoeff_jax(self.get_dofs())

    def dgammadashdashdash_by_dsurf_impl(self, v):
        r"""
        This function returns

        .. math::
            \frac{\partial \Gamma'''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the surface dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        v[:, :, :] = self.dgammadashdashdash_by_dsurf_jax(self.surf.get_dofs())

    def dgammadashdashdash_by_dcoeff_vjp_impl(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma'''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.

        """

        return self.dgammadashdashdash_by_dcoeff_vjp_jax(self.get_dofs(), v)

    def dgammadashdashdash_by_dsurf_vjp_impl(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T  \frac{\partial \Gamma'''}{\partial \mathbf c}

        where :math:`\mathbf{c}` are the surface dofs, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.

        """

        return self.dgammadashdashdash_by_dsurf_vjp_jax(self.surf.get_dofs(), v)

    # KAPPA
    # =====
    def dkappa_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \kappa}{\partial \mathbf{c}}

        where :math:`\mathbf{c}` are the curve dofs and :math:`\kappa` is the curvature.

        """
        return self._curve_and_surface_derivative(
            self.dkappa_by_dcoeff_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v),
            self.dkappa_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v),
        )
        # return Derivative({
        #     self: self.dkappa_by_dcoeff_vjp_jax(self.get_dofs(), v),
        #     self.surf: self.dkappa_by_dsurf_vjp_jax(self.surf.get_dofs(), v)
        #     })

    # TORSION
    # =======
    def dtorsion_by_dcoeff_vjp(self, v):
        r"""
        This function returns the vector Jacobian product

        .. math::
            v^T \frac{\partial \tau}{\partial \mathbf{c}}

        where :math:`\mathbf{c}` are the curve dofs, and :math:`\tau` is the torsion.

        """

        return self._curve_and_surface_derivative(
            self.dtorsion_by_dcoeff_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v),
            self.dtorsion_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v),
        )

    # NORMAL COMPONENTS
    def zfactor(self):
        cdofs = self.get_dofs()
        sdofs = self.surf.get_dofs()
        return self.snz(cdofs, sdofs)

    def dzfactor_by_dcoeff(self):
        cdofs = self.get_dofs()
        sdofs = self.surf.get_dofs()
        return self.dsnz_by_dcoeff_jax(cdofs, sdofs)

    def dzfactor_by_dcoeff_vjp(self, v):
        cdofs = self.get_dofs()
        sdofs = self.surf.get_dofs()
        return Derivative({self: self.dsnz_by_dcoeff_vjp_jax(cdofs, sdofs, v)})

    def zfactor_hessian(self):
        cdofs = self.get_dofs()
        sdofs = self.surf.get_dofs()
        return self.snz_hessian_jax(cdofs, sdofs)

    def rfactor(self):
        cdofs = self.get_dofs()
        sdofs = self.surf.get_dofs()
        return self.snr(cdofs, sdofs)

    def drfactor_by_dcoeff(self):
        cdofs = self.get_dofs()
        sdofs = self.surf.get_dofs()
        return self.dsnr_by_dcoeff_jax(cdofs, sdofs)

    def drfactor_by_dcoeff_vjp(self, v):
        cdofs = self.get_dofs()
        sdofs = self.surf.get_dofs()
        return Derivative({self: self.dsnr_by_dcoeff_vjp_jax(cdofs, sdofs, v)})

    def rfactor_hessian(self):
        cdofs = self.get_dofs()
        sdofs = self.surf.get_dofs()
        return self.snr_hessian_jax(cdofs, sdofs)
