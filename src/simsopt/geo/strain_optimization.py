from functools import partial

import jax.numpy as jnp
from jax import grad, vjp
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.curveobjectives import Lp_torsion_pure
from simsopt.geo.jit import jit

__all__ = [
    "LPBinormalCurvatureStrainPenalty",
    "LPTorsionalStrainPenalty",
    "CoilStrain",
]


@partial(jit, static_argnames=("p", "threshold"))
def _lp_strain_penalty_value(strain_like, gammadash, *, p, threshold):
    return Lp_torsion_pure(strain_like, gammadash, p, threshold)


@partial(jit, static_argnames=("p", "threshold"))
def _lp_strain_penalty_grad(strain_like, gammadash, *, p, threshold):
    return grad(_lp_strain_penalty_value, argnums=(0, 1))(
        strain_like,
        gammadash,
        p=p,
        threshold=threshold,
    )


@partial(jit, static_argnames=("width",))
def _torstrain_eval(torsion, *, width):
    return torstrain_pure(torsion, width)


@partial(jit, static_argnames=("width",))
def _binormstrain_eval(binorm, *, width):
    return binormstrain_pure(binorm, width)


@partial(jit, static_argnames=("width",))
def _torstrain_vjp(torsion, v, *, width):
    return vjp(lambda g: torstrain_pure(g, width), torsion)[1](v)[0]


@partial(jit, static_argnames=("width",))
def _binormstrain_vjp(binorm, v, *, width):
    return vjp(lambda g: binormstrain_pure(g, width), binorm)[1](v)[0]


class LPBinormalCurvatureStrainPenalty(Optimizable):
    r"""
    This class computes a penalty term based on the :math:`L_p` norm
    of the binormal curvature strain, and penalizes where the local strain exceeds a threshold

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \text{max}(\epsilon_{\text{bend}} - \epsilon_0, 0)^p ~dl,

    where

    .. math::
        \epsilon_{\text{bend}} = \frac{w |\hat{\textbf{b}} \cdot \boldsymbol{\kappa}|}{2},

    :math:`w` is the width of the tape, :math:`\hat{\textbf{b}}` is the
    frame binormal vector, :math:`\boldsymbol{\kappa}` is the curvature vector of the
    filamentary coil, and :math:`\epsilon_0` is a threshold strain, given by the argument ``threshold``.
    """

    def __init__(self, framedcurve, width=1e-3, p=2, threshold=0):
        self.framedcurve = framedcurve
        self.strain = CoilStrain(framedcurve, width)
        self.width = width
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[framedcurve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        strain_like = self.strain.binormal_curvature_strain()
        gammadash = self.framedcurve.curve.gammadash()
        return _lp_strain_penalty_value(
            strain_like,
            gammadash,
            p=self.p,
            threshold=self.threshold,
        )

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve and rotation dofs.
        """
        strain_like = self.strain.binormal_curvature_strain()
        gammadash = self.framedcurve.curve.gammadash()
        grad0, grad1 = _lp_strain_penalty_grad(
            strain_like,
            gammadash,
            p=self.p,
            threshold=self.threshold,
        )
        vjp0 = _binormstrain_vjp(
            self.framedcurve.frame_binormal_curvature(),
            grad0,
            width=self.width,
        )
        return self.framedcurve.dframe_binormal_curvature_by_dcoeff_vjp(
            vjp0
        ) + self.framedcurve.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {"J": J, "dJ": dJ}


class LPTorsionalStrainPenalty(Optimizable):
    r"""
    This class computes a penalty term based on the :math:`L_p` norm
    of the torsional strain, and penalizes where the local strain exceeds a threshold

    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \text{max}(\epsilon_{\text{tor}} - \epsilon_0, 0)^p ~dl

    where

    .. math::
        \epsilon_{\text{tor}} = \frac{\tau^2 w^2}{12},

    :math:`\tau` is the torsion of the tape frame, :math:`w` is the width of the tape,
    and :math:`\epsilon_0` is a threshold strain, given by the argument ``threshold``.
    """

    def __init__(self, framedcurve, width=1e-3, p=2, threshold=0):
        self.framedcurve = framedcurve
        self.strain = CoilStrain(framedcurve, width)
        self.width = width
        self.p = p
        self.threshold = threshold
        super().__init__(depends_on=[framedcurve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        strain_like = self.strain.torsional_strain()
        gammadash = self.framedcurve.curve.gammadash()
        return _lp_strain_penalty_value(
            strain_like,
            gammadash,
            p=self.p,
            threshold=self.threshold,
        )

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve and rotation dofs.
        """
        strain_like = self.strain.torsional_strain()
        gammadash = self.framedcurve.curve.gammadash()
        grad0, grad1 = _lp_strain_penalty_grad(
            strain_like,
            gammadash,
            p=self.p,
            threshold=self.threshold,
        )
        vjp0 = _torstrain_vjp(
            self.framedcurve.frame_torsion(),
            grad0,
            width=self.width,
        )
        return self.framedcurve.dframe_torsion_by_dcoeff_vjp(
            vjp0
        ) + self.framedcurve.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {"J": J, "dJ": dJ}


class CoilStrain(Optimizable):
    r"""
    This class evaluates the torsional and binormal curvature strains on HTS, based on
    a filamentary model of the coil and the orientation of the HTS tape.

    As defined in,

    Paz Soldan, "Non-planar coil winding angle optimization for compatibility with
    non-insulated high-temperature superconducting magnets", Journal of Plasma Physics
    86 (2020), doi:10.1017/S0022377820001208,

    the expressions for the strains are:

    .. math::
        \epsilon_{\text{tor}} = \frac{\tau^2 w^2}{12}

        \epsilon_{\text{bend}} = \frac{w |\hat{\textbf{b}} \cdot \boldsymbol{\kappa}|}{2},

    where :math:`\tau` is the torsion of the tape frame, :math:`\hat{\textbf{b}}` is the
    frame binormal vector, :math:`\boldsymbol{\kappa}` is the curvature vector of the
    filamentary coil, and :math:`w` is the width of the tape.

    This class is not intended to be used as an objective function inside
    optimization. For that purpose you should instead use
    :obj:`LPBinormalCurvatureStrainPenalty` or :obj:`LPTorsionalStrainPenalty`.
    Those classes also compute gradients whereas this class does not.
    """

    def __init__(self, framedcurve, width=1e-3):
        self.framedcurve = framedcurve
        self.width = width
        super().__init__(depends_on=[framedcurve])

    def torsional_strain(self):
        r"""
        Returns the value of the torsional strain, :math:`\epsilon_{\text{tor}}`, along
        the quadpoints defining the filamentary coil.
        """
        return _torstrain_eval(self.framedcurve.frame_torsion(), width=self.width)

    def binormal_curvature_strain(self):
        r"""
        Returns the value of the torsional strain, :math:`\epsilon_{\text{bend}}`, along
        the quadpoints defining the filamentary coil.
        """
        return _binormstrain_eval(
            self.framedcurve.frame_binormal_curvature(),
            width=self.width,
        )


@jit
def torstrain_pure(torsion, width):
    """
    This function is used in a Python+Jax implementation of the LPTorsionalStrainPenalty objective.
    """
    return torsion**2 * width**2 / 12


@jit
def binormstrain_pure(binorm, width):
    """
    This function is used in a Python+Jax implementation of the LPBinormalCurvatureStrainPenalty
    objective.
    """
    return (width / 2) * jnp.abs(binorm)
