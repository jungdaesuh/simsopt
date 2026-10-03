"""Native and JAX planar-coils objectives agree within a DERIVED rounding bound.

At the bounded scale the planar-coils end objective is informational: the
lanes' paths fork at round-off.  This file asks whether the lanes compute the
same function where both are evaluated at one state: at every state of a live
native run of the bounded two-stage workflow -- each of three one-ulp draws'
start, stage-1 end and stage-2 end -- do the native objective (``_native_evaluator``)
and the JAX lane's objective program (``standard_stage_two_state``, the entry
the JAX lane's own solve evaluates) take the same VALUE and the same GRADIENT,
each at the stage's own length weight?

"The same" means: within the DERIVED rounding bound of the formula both lanes
evaluate (:func:`_planar_objective_bound`), doubled for two implementations --
worst-order sums, per-rounding charges, the chain rule's path sums for the
gradient, per-element geometry counts taken from the longer of the two lanes'
forms.  No tolerance here is fitted to a measured gap.  The bound is an
ENGINEERING ERROR ENVELOPE: its xsimd ``sincos`` constant is assumed, not
established for the pinned build.  The JAX evaluator here runs on the CPU
device only, so this test covers the jax-cpu lane and not jax-gpu.

The states are produced live by a native SIMSOPT run of the same workflow from
upstream's one-ulp protocol starts; no recorded state is read.

That bound replaces a scalar relative budget whose scale was the objective's
value and the gradient's largest entry, which fails at these states for a
reason the scale cannot see: the length penalty ``0.5 w (L - L0)^2`` sits at
``L - L0 ~ 4e-3`` on ``L ~ 10.4``, so one ulp of ``L`` moves its value by
``2.5e3`` ulps of itself, and at the stage-2 ends the flux and length gradients
cancel to a total gradient far below either term's.

The bound (``_planar_objective_bound`` and its helpers below)
--------------------------------------------------------------
The native lane (C++ ``CurvePlanarFourier``, ``BiotSavart``, ``integral_BdotN``
and the NumPy objective layer) and the JAX lane (``fused_stage_two_values``
under ``jax.value_and_grad``) evaluate one formula -- upstream's objective --
in two ways.  :func:`_planar_objective_bound` returns that formula at one state
as a :class:`forward_roundoff_bound.Bounded`, whose bookkeeping bounds how far
ANY implementation that differs only in summation order, product association,
derivative mode, and the per-element forms counted below may round away from
the exact value and gradient.

It assumes library accuracies that are not all established for the pinned
builds: ``sin``/``cos`` within one ulp in both lanes (covered for glibc,
documented at about 0.55 ulp, and for XLA CPU where it equals glibc) and the
C++ ``SurfaceRZFourier`` xsimd ``sincos`` within two ulp (not established).

Per-element quantities are injected with derived bounds rather than traced op
by op, because the two lanes write them differently.  With ``u = 2**-53``, all
counts in units of ``u`` and first order in ``u``:

* Curve geometry ``gamma``, ``gamma'``, ``gamma''`` and their dof Jacobians.
  Expanded into monomials ``coefficient * dof * trig * trig * q_hat * q_hat``,
  every lane's form -- C++ closed forms that distribute the rotation, JAX's
  ``R @ p`` of a jvp-differentiated planar curve -- is a rearrangement of the
  same monomials, so the sum of their absolute values ``A`` is form-invariant.
  A computed element is within ``K A + T`` of exact, where ``T`` carries the
  absolute errors of the trig factors (``2 n |phi| + 2`` for ``cos(n phi)``:
  ``phi = 2 pi t`` and ``n phi`` rounded once each, ``cos``/``sin`` within one
  ulp) and ``K`` counts, for the longer lane form: 10 multiplications per
  monomial (mode factor, trig product, dof, ``(2 pi)^o`` as up to two roundings,
  ``2 q_a q_b``, ``R p``, and the jvp tangent products), ``2 order + 7``
  additions (planar accumulation, ``R`` entry, ``R p``, centre) and two
  normalized-quaternion factors of relative error 5 each (``sum q^2`` 4,
  ``sqrt`` 1, reciprocal 1, product 1, against JAX's 4).  ``K = 2 order + 27``.
  The quaternion Jacobian is written by C++ as a hand-simplified expression
  (``curveplanarfourier.cpp``, ``d*_by_dcoeff_impl``) and by JAX as the chain
  rule through ``q / |q|``; ``A`` is the larger of the two expansions and
  ``K_q = 2 order + 39`` (up to 11 multiplications, 3 normalized-quaternion
  factors, ``1/|q|`` with 4, ``2 order + 9`` additions).
* Surface points and normals feeding the flux, which each lane computes itself
  (C++ ``SurfaceRZFourier`` with an angle recurrence, JAX a separable
  evaluation).  ``A`` uses the separable expansion (the larger), trig factors
  carry the larger of the two lanes' argument, library and recurrence errors
  (:func:`_surface_trig_error`), and ``K = 5 + (modes - 1) + 2``.

Everything downstream -- rotations to the symmetric copies, Biot-Savart,
``B . n``, the lengths, distances, curvatures and penalties -- is traced by
:mod:`forward_roundoff_bound` in upstream's form, with :func:`envelope` where
the lanes' per-contribution forms differ (``1/r^3`` and the flux weight).
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import forward_roundoff_bound as fb
import jax
import numpy as np
import pytest
from scipy.optimize import minimize
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurvePlanarFourier,
    CurveSurfaceDistance,
    LinkingNumber,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    create_equally_spaced_planar_curves,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt_jax.examples.stage_two_standard import standard_stage_two_state
from simsopt_jax.objectives import StageTwoObjectiveConfig
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

SURFACE_INPUT = (
    Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
)

#: The reduced (bounded) scale of ``stage_two_optimization_planar_coils.py``.
CONFIGURATION: Mapping[str, float | int] = {
    "surface_resolution": 4,
    "curve_order": 2,
    "curve_quadrature": 32,
    "num_base_curves": 4,
    "major_radius": 1.0,
    "minor_radius": 0.5,
    "initial_current": 1.0e5,
    "length_target": 10.4,
    "first_length_weight": 10.0,
    "second_length_weight": 1.0,
    "curve_curve_threshold": 0.08,
    "curve_curve_weight": 1000.0,
    "curve_surface_threshold": 0.12,
    "curve_surface_weight": 10.0,
    "curvature_threshold": 10.0,
    "curvature_weight": 1.0e-6,
    "mean_squared_curvature_threshold": 10.0,
    "mean_squared_curvature_weight": 1.0e-6,
    "linking_number_weight": 1.0,
    "max_steps": 50,
    "rtol": 1.0e-15,
    "atol": 1.0e-15,
}
LBFGS_HISTORY = 300
#: Upstream's one-ulp protocol: draw 0 is the unperturbed start.  Three draws
#: of upstream's nine: each live native run costs ~20 s at this scale.
DRAWS = tuple(range(3))
#: (published state, stage whose length weight the objective carries there)
RUN_STATES = (
    ("initial:parameters", "first_length_weight"),
    ("first:parameters", "first_length_weight"),
    ("final:parameters", "second_length_weight"),
)
CASES = tuple(
    (k, parameter_key, weight_name)
    for k in DRAWS
    for parameter_key, weight_name in RUN_STATES
)


def _geometry() -> tuple[SurfaceRZFourier, list[CurvePlanarFourier], list[Coil]]:
    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        range="half period",
        nphi=int(CONFIGURATION["surface_resolution"]),
        ntheta=int(CONFIGURATION["surface_resolution"]),
    )
    surface.fix_all()
    base_curves = create_equally_spaced_planar_curves(
        int(CONFIGURATION["num_base_curves"]),
        surface.nfp,
        stellsym=True,
        R0=float(CONFIGURATION["major_radius"]),
        R1=float(CONFIGURATION["minor_radius"]),
        order=int(CONFIGURATION["curve_order"]),
        numquadpoints=int(CONFIGURATION["curve_quadrature"]),
    )
    base_currents = [
        Current(float(CONFIGURATION["initial_current"])) for _ in base_curves
    ]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    return surface, base_curves, coils


@dataclass(frozen=True)
class _NativePlanarEvaluator:
    """The one native assembly of the Stage-II objective, in the script's order.

    The record freezes its references only: evaluating any term mutates the
    shared simsopt graph through ``objective.x``, so one evaluator serves one
    consumer at a time.
    """

    configuration: Mapping[str, float | int]
    surface: SurfaceRZFourier
    base_curves: tuple[CurvePlanarFourier, ...]
    coils: tuple[Coil, ...]
    flux: SquaredFlux
    length_penalty: Optimizable
    curve_curve: CurveCurveDistance
    curve_surface: CurveSurfaceDistance
    curvature: Optimizable
    mean_squared_curvature: Optimizable
    linking_number: LinkingNumber

    def weighted(self, length_weight: float) -> Optimizable:
        """The script's objective at one stage's length weight."""
        return (
            self.flux
            + length_weight * self.length_penalty
            + float(self.configuration["curve_curve_weight"]) * self.curve_curve
            + float(self.configuration["curve_surface_weight"]) * self.curve_surface
            + float(self.configuration["curvature_weight"]) * self.curvature
            + float(self.configuration["mean_squared_curvature_weight"])
            * self.mean_squared_curvature
            + float(self.configuration["linking_number_weight"]) * self.linking_number
        )


def _native_evaluator() -> _NativePlanarEvaluator:
    surface, base_curves, coils = _geometry()
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    curves = [coil.curve for coil in coils]
    return _NativePlanarEvaluator(
        configuration=CONFIGURATION,
        surface=surface,
        base_curves=tuple(base_curves),
        coils=tuple(coils),
        flux=SquaredFlux(surface, field),
        length_penalty=QuadraticPenalty(
            sum(CurveLength(curve) for curve in base_curves),
            float(CONFIGURATION["length_target"]),
        ),
        curve_curve=CurveCurveDistance(
            curves,
            float(CONFIGURATION["curve_curve_threshold"]),
            num_basecurves=int(CONFIGURATION["num_base_curves"]),
        ),
        curve_surface=CurveSurfaceDistance(
            curves, surface, float(CONFIGURATION["curve_surface_threshold"])
        ),
        curvature=sum(
            LpCurveCurvature(curve, 2, float(CONFIGURATION["curvature_threshold"]))
            for curve in base_curves
        ),
        mean_squared_curvature=sum(
            QuadraticPenalty(
                MeanSquaredCurvature(curve),
                float(CONFIGURATION["mean_squared_curvature_threshold"]),
            )
            for curve in base_curves
        ),
        linking_number=LinkingNumber(curves),
    )


def _one_ulp_start(parameters: np.ndarray, k: int) -> np.ndarray:
    """Upstream's one-ulp protocol draw ``k`` of a start vector (``k = 0``: itself)."""
    if k == 0:
        return parameters
    signs = np.random.RandomState(20260920 + k).choice(
        [-1.0, 1.0], size=parameters.size
    )
    return np.nextafter(parameters, signs * np.inf)


def _native_run_states(start: np.ndarray) -> dict[str, np.ndarray]:
    """The bounded two-stage native workflow from ``start``: its three states."""
    evaluator = _native_evaluator()

    def solve(length_weight_name: str, initial: np.ndarray) -> np.ndarray:
        objective = evaluator.weighted(float(CONFIGURATION[length_weight_name]))

        def value_and_gradient(parameters: np.ndarray):
            objective.x = parameters
            return float(objective.J()), np.asarray(objective.dJ(), dtype=np.float64)

        result = minimize(
            value_and_gradient,
            initial,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": int(CONFIGURATION["max_steps"]),
                "maxcor": LBFGS_HISTORY,
                "ftol": float(CONFIGURATION["rtol"]),
                "gtol": float(CONFIGURATION["atol"]),
            },
        )
        return np.asarray(result.x, dtype=np.float64)

    first = solve("first_length_weight", start)
    return {
        "initial:parameters": np.asarray(start, dtype=np.float64),
        "first:parameters": first,
        "final:parameters": solve("second_length_weight", first),
    }


@pytest.fixture(scope="module")
def evaluator() -> _NativePlanarEvaluator:
    """The native objective assembly at the bounded scale."""
    return _native_evaluator()


@pytest.fixture(scope="module")
def native_states() -> dict[int, dict[str, np.ndarray]]:
    """Every published state of the live native runs, by draw."""
    start = np.asarray(BiotSavart(_geometry()[2]).x, dtype=np.float64)
    return {k: _native_run_states(_one_ulp_start(start, k)) for k in DRAWS}


def _regularization_config() -> StageTwoObjectiveConfig:
    """The regularization the JAX lane builds (length weight passed explicitly)."""
    return StageTwoObjectiveConfig(
        num_base_curves=int(CONFIGURATION["num_base_curves"]),
        length_target=float(CONFIGURATION["length_target"]),
        length_target_mode="identity",
        curve_curve_minimum_distance=float(CONFIGURATION["curve_curve_threshold"]),
        curve_curve_weight=float(CONFIGURATION["curve_curve_weight"]),
        curve_surface_minimum_distance=float(CONFIGURATION["curve_surface_threshold"]),
        curve_surface_weight=float(CONFIGURATION["curve_surface_weight"]),
        curvature_threshold=float(CONFIGURATION["curvature_threshold"]),
        curvature_weight=float(CONFIGURATION["curvature_weight"]),
        mean_squared_curvature_threshold=float(
            CONFIGURATION["mean_squared_curvature_threshold"]
        ),
        mean_squared_curvature_target_mode="identity",
        mean_squared_curvature_weight=float(
            CONFIGURATION["mean_squared_curvature_weight"]
        ),
        linking_number_weight=float(CONFIGURATION["linking_number_weight"]),
    )


TWO_PI = 2.0 * math.pi

#: Rotation matrix of a unit quaternion as monomials in ``q_hat``:
#: ``R[a][b] = sum(coefficient * prod(q_hat[l] for l in factors))``.
_ROTATION: tuple[tuple[tuple[tuple[float, tuple[int, ...]], ...], ...], ...] = (
    (
        ((1.0, ()), (-2.0, (2, 2)), (-2.0, (3, 3))),
        ((2.0, (1, 2)), (-2.0, (3, 0))),
        ((2.0, (1, 3)), (2.0, (2, 0))),
    ),
    (
        ((2.0, (1, 2)), (2.0, (3, 0))),
        ((1.0, ()), (-2.0, (1, 1)), (-2.0, (3, 3))),
        ((2.0, (2, 3)), (-2.0, (1, 0))),
    ),
    (
        ((2.0, (1, 3)), (-2.0, (2, 0))),
        ((2.0, (2, 3)), (2.0, (1, 0))),
        ((1.0, ()), (-2.0, (1, 1)), (-2.0, (2, 2))),
    ),
)


def _monomials_value(monomials, q_hat: np.ndarray, absolute: bool) -> float:
    total = 0.0
    for coefficient, factors in monomials:
        term = abs(coefficient) if absolute else coefficient
        for factor in factors:
            term *= abs(q_hat[factor]) if absolute else q_hat[factor]
        total += term
    return total


def _monomials_derivative(
    monomials, q_hat: np.ndarray, index: int, absolute: bool
) -> float:
    total = 0.0
    for coefficient, factors in monomials:
        count = factors.count(index)
        if count == 0:
            continue
        rest = list(factors)
        rest.remove(index)
        term = count * (abs(coefficient) if absolute else coefficient)
        for factor in rest:
            term *= abs(q_hat[factor]) if absolute else q_hat[factor]
        total += term
    return total


def _rotation(q_hat: np.ndarray, absolute: bool) -> np.ndarray:
    return np.array(
        [
            [_monomials_value(_ROTATION[a][b], q_hat, absolute) for b in range(3)]
            for a in range(3)
        ]
    )


def _rotation_derivative(q_hat: np.ndarray, absolute: bool) -> np.ndarray:
    """``[a, b, l] = d R[a][b] / d q_hat[l]``."""
    return np.array(
        [
            [
                [
                    _monomials_derivative(_ROTATION[a][b], q_hat, l, absolute)
                    for l in range(4)
                ]
                for b in range(3)
            ]
            for a in range(3)
        ]
    )


@dataclass(frozen=True)
class _PlanarBasis:
    """Planar (pre-rotation) x, y parts of one derivative order, per Fourier dof.

    Arrays are ``[dof, quadrature point]``: the value of the basis function,
    the sum of its monomials' absolute values, and its trig-error charge.
    """

    value: np.ndarray  # [2, dofs, points]
    absolute: np.ndarray
    trig_error: np.ndarray


def _planar_basis(phi: np.ndarray, order: int, derivative: int) -> _PlanarBasis:
    """Monomials of ``d^o/dt^o`` of ``(r cos phi, r sin phi)`` per dof, ``(2 pi)^o`` included."""
    modes = np.arange(order + 1)
    cos_n = np.cos(np.outer(modes, phi))
    sin_n = np.sin(np.outer(modes, phi))
    trig = 2.0 * np.outer(modes, np.abs(phi)) + 2.0  # |error| of cos(n phi), sin(n phi)
    trig[0] = 0.0  # n = 0: exact 1 and 0

    def c(n):
        return cos_n[n], trig[n]

    def s(n):
        return sin_n[n], trig[n]

    def monomial(coefficient, first, second=None):
        (a, ea) = first
        (b, eb) = (
            second if second is not None else (np.ones_like(phi), np.zeros_like(phi))
        )
        return (
            coefficient * a * b,
            abs(coefficient) * np.abs(a * b),
            abs(coefficient) * (ea * np.abs(b) + np.abs(a) * eb),
        )

    # Per dof: list of monomials for x and for y (upstream C++ forms, the JAX
    # jvp expansions differ only by splitting (n^2 + 1) into n^2 and 1).
    tables = []
    if derivative == 0:
        tables.append(([monomial(1, c(1))], [monomial(1, s(1))]))
        for n in range(1, order + 1):
            tables.append(([monomial(1, c(n), c(1))], [monomial(1, c(n), s(1))]))
        for n in range(1, order + 1):
            tables.append(([monomial(1, s(n), c(1))], [monomial(1, s(n), s(1))]))
    elif derivative == 1:
        tables.append(([monomial(-1, s(1))], [monomial(1, c(1))]))
        for n in range(1, order + 1):
            tables.append(
                (
                    [monomial(-n, s(n), c(1)), monomial(-1, c(n), s(1))],
                    [monomial(-n, s(n), s(1)), monomial(1, c(n), c(1))],
                )
            )
        for n in range(1, order + 1):
            tables.append(
                (
                    [monomial(n, c(n), c(1)), monomial(-1, s(n), s(1))],
                    [monomial(n, c(n), s(1)), monomial(1, s(n), c(1))],
                )
            )
    else:
        tables.append(([monomial(-1, c(1))], [monomial(-1, s(1))]))
        for n in range(1, order + 1):
            tables.append(
                (
                    [monomial(2 * n, s(n), s(1)), monomial(-(n * n + 1), c(n), c(1))],
                    [monomial(-2 * n, s(n), c(1)), monomial(-(n * n + 1), c(n), s(1))],
                )
            )
        for n in range(1, order + 1):
            tables.append(
                (
                    [monomial(-(n * n + 1), s(n), c(1)), monomial(-2 * n, c(n), s(1))],
                    [monomial(-(n * n + 1), s(n), s(1)), monomial(2 * n, c(n), c(1))],
                )
            )
    factor = TWO_PI**derivative
    value = (
        np.array(
            [[sum(m[0] for m in table[axis]) for table in tables] for axis in range(2)]
        )
        * factor
    )
    absolute = (
        np.array(
            [[sum(m[1] for m in table[axis]) for table in tables] for axis in range(2)]
        )
        * factor
    )
    trig_error = (
        np.array(
            [[sum(m[2] for m in table[axis]) for table in tables] for axis in range(2)]
        )
        * factor
    )
    return _PlanarBasis(value, absolute, trig_error)


def _cpp_quaternion_jacobian(i, j, q, inverse_norm, absolute: bool) -> np.ndarray:
    """C++'s simplified ``d (R(q/|q|) p) / d q_m`` (``curveplanarfourier.cpp``).

    Returns ``[component, m, point]``; with ``absolute`` the same expression is
    evaluated on absolute values with every subtraction turned into an addition.
    """
    if absolute:
        i, j, q = np.abs(i), np.abs(j), np.abs(q)

    def sub(a, b):
        return a + b if absolute else a - b

    def neg(a):
        return a if absolute else -a

    q0, q1, q2, q3 = q
    rows = []
    for m, qm in enumerate((q0, q1, q2, q3)):
        one_x = 1.0 if m in (2, 3) else 0.0
        one_y = 1.0 if m in (1, 3) else 0.0
        x_first = 4 * i * sub(q2 * q2 + q3 * q3, one_x) * qm
        y_first = 4 * j * sub(q1 * q1 + q3 * q3, one_y) * qm
        half = {
            0: (0.5 * q3, -0.5 * q3, 0.5 * q2, -0.5 * q1),
            1: (-0.5 * q2, -0.5 * q2, -0.5 * q3, -0.5 * q0),
            2: (-0.5 * q1, -0.5 * q1, 0.5 * q0, -0.5 * q3),
            3: (0.5 * q0, -0.5 * q0, -0.5 * q1, -0.5 * q2),
        }[m]
        if absolute:
            half = tuple(abs(h) for h in half)
        x = sub(x_first, 4 * j * ((sub(q1 * q2, q0 * q3)) * qm + half[0]))
        y = sub(y_first, 4 * i * ((q1 * q2 + q0 * q3) * qm + half[1]))
        z = sub(
            neg(4 * i * ((sub(q1 * q3, q0 * q2)) * qm + half[2])),
            4 * j * ((q2 * q3 + q0 * q1) * qm + half[3]),
        )
        rows.append((x * inverse_norm, y * inverse_norm, z * inverse_norm))
    return np.array(rows).transpose(1, 0, 2)


def _curve_elements(
    dofs: np.ndarray, quadpoints: np.ndarray, order: int, derivative: int
):
    """Value, error, Jacobian, Jacobian |paths| and Jacobian error of one curve array.

    Shapes: value/error ``[points, 3]``; Jacobian arrays ``[points, 3, dofs]``.
    """
    phi = TWO_PI * np.asarray(quadpoints, dtype=np.float64)
    fourier = 2 * order + 1
    fourier_dofs = dofs[:fourier]
    q = dofs[fourier : fourier + 4]
    centre = dofs[fourier + 4 :]
    norm_q = float(np.sqrt(np.sum(q * q)))
    q_hat = q / norm_q
    basis = _planar_basis(phi, order, derivative)
    planar = np.einsum("adp,d->ap", basis.value, fourier_dofs)  # [2, points]
    planar_abs = np.einsum("adp,d->ap", basis.absolute, np.abs(fourier_dofs))
    planar_trig = np.einsum("adp,d->ap", basis.trig_error, np.abs(fourier_dofs))
    rotation = _rotation(q_hat, absolute=False)[:, :2]
    rotation_abs = _rotation(q_hat, absolute=True)[:, :2]
    count = 2 * order + 27
    count_q = 2 * order + 39

    value = (rotation @ planar).T + (centre if derivative == 0 else 0.0)
    absolute = (rotation_abs @ planar_abs).T + (
        np.abs(centre) if derivative == 0 else 0.0
    )
    error = count * absolute + (rotation_abs @ planar_trig).T

    points = phi.size
    size = dofs.size
    jac = np.zeros((points, 3, size))
    jac_abs = np.zeros((points, 3, size))
    jac_err = np.zeros((points, 3, size))
    # Fourier dofs: J = R B, one monomial set per dof.
    jac[:, :, :fourier] = np.einsum("ab,bdp->pad", rotation, basis.value)
    fourier_abs = np.einsum("ab,bdp->pad", rotation_abs, basis.absolute)
    jac_abs[:, :, :fourier] = fourier_abs
    jac_err[:, :, :fourier] = count * fourier_abs + np.einsum(
        "ab,bdp->pad", rotation_abs, basis.trig_error
    )
    # Quaternion dofs: chain rule through q_hat = q / |q|, and C++'s closed form.
    d_rotation = _rotation_derivative(q_hat, absolute=False)[:, :2, :]  # [a, b, l]
    d_rotation_abs = _rotation_derivative(q_hat, absolute=True)[:, :2, :]
    normalization = (np.eye(4) - np.outer(q_hat, q_hat)) / norm_q  # [l, m]
    normalization_abs = (np.eye(4) + np.abs(np.outer(q_hat, q_hat))) / norm_q
    jac[:, :, fourier : fourier + 4] = np.einsum(
        "abl,bp,lm->pam", d_rotation, planar, normalization
    )
    chain_abs = np.einsum(
        "abl,bp,lm->pam", d_rotation_abs, planar_abs, normalization_abs
    )
    chain_trig = np.einsum(
        "abl,bp,lm->pam", d_rotation_abs, planar_trig, normalization_abs
    )
    cpp_value = _cpp_quaternion_jacobian(
        planar[0], planar[1], q_hat, 1.0 / norm_q, absolute=False
    ).transpose(2, 0, 1)
    cpp_abs = _cpp_quaternion_jacobian(
        planar_abs[0], planar_abs[1], q_hat, 1.0 / norm_q, absolute=True
    ).transpose(2, 0, 1)
    cpp_trig = _cpp_quaternion_jacobian(
        planar_trig[0], planar_trig[1], q_hat, 1.0 / norm_q, absolute=True
    ).transpose(2, 0, 1)
    scale = max(1.0, float(np.max(np.abs(cpp_value))))
    if not np.allclose(
        cpp_value, jac[:, :, fourier : fourier + 4], rtol=0.0, atol=1e-12 * scale
    ):
        raise AssertionError(
            "transcribed C++ quaternion Jacobian disagrees with the chain rule"
        )
    quaternion_abs = np.maximum(chain_abs, cpp_abs)
    jac_abs[:, :, fourier : fourier + 4] = quaternion_abs
    jac_err[:, :, fourier : fourier + 4] = count_q * quaternion_abs + np.maximum(
        chain_trig, cpp_trig
    )
    # Centre dofs: gamma only, exact identity.
    if derivative == 0:
        for axis in range(3):
            jac[:, axis, fourier + 4 + axis] = 1.0
            jac_abs[:, axis, fourier + 4 + axis] = 1.0
    return value, error, jac, jac_abs, jac_err


def _surface_trig_error(
    m_theta: np.ndarray, n_phi: np.ndarray, field_period_phi: float
) -> np.ndarray:
    """Absolute error of the trig factors of ``cos(m theta - n nfp phi)`` in either lane.

    C++ (``surfacerzfourier.cpp``): argument ``m theta - (n nfp) phi`` with
    ``theta``, ``phi`` each rounded once (3 relative roundings on each part),
    xsimd ``sincos`` taken within two ulp, then up to four recurrence steps
    ``ANGLE_RECOMPUTE = 5`` by the rotation through ``-nfp phi`` (each step 3
    roundings of magnitude at most ``sqrt 2`` plus the step angle's own error,
    and a componentwise-to-Euclidean factor ``sqrt 2``).  JAX: separable
    ``cos(m theta) cos(n nfp phi) + sin(m theta) sin(n nfp phi)``, two factors
    each with argument error (2 and 3 roundings) and one ulp.
    """
    step = 2.0 * field_period_phi + 2.0
    cpp = math.sqrt(2.0) * (3.0 * (m_theta + n_phi) + 4.0) + 4.0 * (
        3.0 * math.sqrt(2.0) + 2.0 * step
    )
    separable = 2.0 * (2.0 * m_theta + 2.0) + 2.0 * (3.0 * n_phi + 2.0)
    return np.maximum(cpp, separable)


def _surface_elements(surface) -> tuple[fb.Bounded, fb.Bounded]:
    """Lane-computed surface points and normals as bounded constants."""
    nfp = int(surface.nfp)
    mpol, ntor = int(surface.mpol), int(surface.ntor)
    phi = TWO_PI * np.asarray(surface.quadpoints_phi, dtype=np.float64)
    theta = TWO_PI * np.asarray(surface.quadpoints_theta, dtype=np.float64)
    rc = np.asarray(surface.rc, dtype=np.float64)
    zs = np.asarray(surface.zs, dtype=np.float64)
    m = np.arange(mpol + 1)[:, None, None, None]
    n = (np.arange(2 * ntor + 1) - ntor)[None, :, None, None]
    ph = phi[None, None, :, None]
    th = theta[None, None, None, :]
    m_theta = np.abs(m * th)
    n_phi = np.abs(n * nfp * ph)
    separable_abs = np.abs(np.cos(m * th) * np.cos(n * nfp * ph)) + np.abs(
        np.sin(m * th) * np.sin(n * nfp * ph)
    )
    trig = _surface_trig_error(m_theta, n_phi, float(nfp * np.max(np.abs(phi))))
    modes = (mpol + 1) * (2 * ntor + 1)
    count = 5 + (modes - 1) + 2

    def series(coefficients, factor):
        weights = np.abs(coefficients)[:, :, None, None] * np.abs(factor)
        return np.sum(weights * separable_abs, axis=(0, 1)), np.sum(
            weights * trig, axis=(0, 1)
        )

    r_abs, r_trig = series(rc, 1.0)
    z_abs, z_trig = series(zs, 1.0)
    dr_dphi_abs, dr_dphi_trig = series(rc, n * nfp * TWO_PI)
    dz_dphi_abs, dz_dphi_trig = series(zs, n * nfp * TWO_PI)
    dr_dtheta_abs, dr_dtheta_trig = series(rc, m * TWO_PI)
    dz_dtheta_abs, dz_dtheta_trig = series(zs, m * TWO_PI)
    cos_phi = np.abs(np.cos(phi))[:, None]
    sin_phi = np.abs(np.sin(phi))[:, None]
    phi_trig = (np.abs(phi) + 2.0)[:, None]

    def planar_error(radial_abs, radial_trig, extra_abs=None, extra_trig=None):
        """Error of (x, y) = radial * (cos phi, sin phi) [+ extra * (-sin phi, cos phi)]."""
        out = []
        for own, other in ((cos_phi, sin_phi), (sin_phi, cos_phi)):
            absolute = radial_abs * own + (
                0.0 if extra_abs is None else extra_abs * other
            )
            trig_part = radial_trig * own + radial_abs * phi_trig
            if extra_abs is not None:
                trig_part = trig_part + extra_trig * other + extra_abs * phi_trig
            out.append(count * absolute + trig_part)
        return out

    x_err, y_err = planar_error(r_abs, r_trig)
    points_error = np.stack([x_err, y_err, count * z_abs + z_trig], axis=-1)
    d1x, d1y = planar_error(dr_dphi_abs, dr_dphi_trig, TWO_PI * r_abs, TWO_PI * r_trig)
    d2x, d2y = planar_error(dr_dtheta_abs, dr_dtheta_trig)
    gammadash1_error = np.stack([d1x, d1y, count * dz_dphi_abs + dz_dphi_trig], axis=-1)
    gammadash2_error = np.stack(
        [d2x, d2y, count * dz_dtheta_abs + dz_dtheta_trig], axis=-1
    )
    components = 0
    points = fb.constant(
        np.asarray(surface.gamma(), dtype=np.float64), components, points_error
    )
    gammadash1 = fb.constant(
        np.asarray(surface.gammadash1(), dtype=np.float64), components, gammadash1_error
    )
    gammadash2 = fb.constant(
        np.asarray(surface.gammadash2(), dtype=np.float64), components, gammadash2_error
    )
    normals = fb.cross(gammadash1, gammadash2)
    return points, normals


@dataclass(frozen=True)
class _PlanarObjectiveBound:
    """The objective at one state with its rounding bookkeeping, and its terms."""

    total: fb.Bounded
    terms: Mapping[str, fb.Bounded]


def _with_components(quantity: fb.Bounded, components: int) -> fb.Bounded:
    """Give a derivative-free constant ``components`` zero derivative columns."""
    zeros = np.zeros(quantity.v.shape + (components,))
    return fb.Bounded(quantity.v, quantity.e, zeros, zeros, zeros, zeros)


def _planar_objective_bound(
    evaluator, parameters: np.ndarray, length_weight: float
) -> _PlanarObjectiveBound:
    """Upstream's weighted planar objective at ``parameters`` as a :class:`Bounded`.

    ``evaluator`` is the native lane's ``_NativePlanarEvaluator``; only its
    structure (dof layout, coils, symmetries, surface, configuration) is read,
    and its objective state is set to ``parameters`` so the dof names resolve.
    Derivative components follow ``evaluator.weighted(length_weight).x``.
    """
    configuration = evaluator.configuration
    objective = evaluator.weighted(length_weight)
    objective.x = parameters
    names = list(objective.dof_names)
    x = np.asarray(parameters, dtype=np.float64)
    size = x.size
    base_curves = list(evaluator.base_curves)
    order = int(base_curves[0].order)
    quadpoints = np.asarray(base_curves[0].quadpoints, dtype=np.float64)
    count_q = quadpoints.size

    geometry = []  # per base curve: (gamma, gammadash, gammadashdash) as Bounded
    for curve in base_curves:
        index = [
            names.index(f"{curve.name}:{local}") for local in curve.local_full_dof_names
        ]
        arrays = []
        for derivative in range(3):
            value, error, jac, jac_abs, jac_err = _curve_elements(
                x[index], quadpoints, order, derivative
            )
            full = np.zeros(value.shape + (size,))
            full_abs = np.zeros_like(full)
            full_err = np.zeros_like(full)
            full[..., index] = jac
            full_abs[..., index] = jac_abs
            full_err[..., index] = jac_err
            arrays.append(fb.element(value, error, full, full_abs, full_err))
        geometry.append(tuple(arrays))

    seeds = fb.seed(x)
    coil_gamma, coil_gammadash, currents = [], [], []
    for coil in evaluator.coils:
        curve = coil.curve
        base = base_curves.index(curve.curve if hasattr(curve, "curve") else curve)
        rotation = np.asarray(getattr(curve, "rotmat", np.eye(3)), dtype=np.float64)
        gamma, gammadash, _ = geometry[base]
        coil_gamma.append(fb.matmul_constant(gamma, rotation))
        coil_gammadash.append(fb.matmul_constant(gammadash, rotation))
        current = coil.current
        sign = float(getattr(current, "scale", 1.0))
        owner = getattr(current, "current_to_scale", current)
        name = f"{owner.name}:x0"
        if name in names:
            currents.append(fb.scale(seeds[names.index(name)], sign))
        else:
            currents.append(fb.constant(sign * float(owner.get_value()), size))
    gammas = fb.stack(coil_gamma)  # [coils, points, 3]
    gammadashes = fb.stack(coil_gammadash)
    current_values = fb.stack(currents)  # [coils]

    terms: dict[str, fb.Bounded] = {}
    terms["squared_flux"] = _squared_flux(
        evaluator, gammas, gammadashes, current_values, count_q, size
    )

    base_speed = fb.norm(fb.stack([g[1] for g in geometry]))  # [base, points]
    total_length = fb.total(fb.mean(base_speed, 1), 0)
    excess = fb.sub(
        total_length, fb.constant(float(configuration["length_target"]), size)
    )
    terms["length_penalty"] = fb.scale(
        fb.square(excess), 0.5 * length_weight, relative_error=1.0
    )

    terms["curve_curve_distance"] = fb.scale(
        _curve_curve(
            gammas,
            gammadashes,
            float(configuration["curve_curve_threshold"]),
            int(configuration["num_base_curves"]),
            size,
        ),
        float(configuration["curve_curve_weight"]),
    )
    surface_points = fb.constant(
        np.asarray(evaluator.surface.gamma(), dtype=np.float64).reshape((-1, 3)), size
    )
    surface_normals = fb.constant(
        np.asarray(evaluator.surface.normal(), dtype=np.float64).reshape((-1, 3)), size
    )
    terms["curve_surface_distance"] = fb.scale(
        _curve_surface(
            gammas,
            gammadashes,
            surface_points,
            surface_normals,
            float(configuration["curve_surface_threshold"]),
        ),
        float(configuration["curve_surface_weight"]),
    )
    kappa = _curvature(
        fb.stack([g[1] for g in geometry]), fb.stack([g[2] for g in geometry])
    )
    terms["lp_curvature"] = fb.scale(
        _lp_curvature(kappa, base_speed, float(configuration["curvature_threshold"])),
        float(configuration["curvature_weight"]),
    )
    terms["mean_squared_curvature_penalty"] = fb.scale(
        _mean_squared_curvature_penalty(
            kappa, base_speed, float(configuration["mean_squared_curvature_threshold"])
        ),
        float(configuration["mean_squared_curvature_weight"]),
    )
    # The linking number is an integer in both lanes (rounded Gauss sums), with
    # a zero gradient; it enters the total exactly.
    terms["linking_number"] = fb.constant(
        float(configuration["linking_number_weight"])
        * float(evaluator.linking_number.J()),
        size,
    )
    total = fb.total(fb.stack(list(terms.values())), 0)
    return _PlanarObjectiveBound(total=total, terms=terms)


def _inverse_cube(distance_squared: fb.Bounded) -> fb.Bounded:
    """``1/r^3`` as C++ (``(1/sqrt(r2))^3``) and JAX (``(1/sqrt(r2)) (1/r2)``) form it."""
    one = fb.constant(np.ones_like(distance_squared.v), distance_squared.components)
    inverse = fb.div(one, fb.sqrt(distance_squared))
    native = fb.mul(fb.mul(inverse, inverse), inverse)
    jax_form = fb.mul(inverse, fb.div(one, distance_squared))
    return fb.envelope(native, jax_form)


def _squared_flux(
    evaluator, gammas, gammadashes, currents, quadrature: int, size: int
) -> fb.Bounded:
    surface = evaluator.surface
    points, normals = _surface_elements(surface)
    points = _with_components(points, size)
    normals = _with_components(normals, size)
    flat_points = points.v.reshape((-1, 3))
    point_count = flat_points.shape[0]
    points = fb.Bounded(
        flat_points,
        points.e.reshape((-1, 3)),
        points.d.reshape((-1, 3, size)),
        points.D.reshape((-1, 3, size)),
        points.Ed.reshape((-1, 3, size)),
        points.paths.reshape((-1, 3, size)),
    )
    normals = fb.Bounded(
        normals.v.reshape((-1, 3)),
        normals.e.reshape((-1, 3)),
        normals.d.reshape((-1, 3, size)),
        normals.D.reshape((-1, 3, size)),
        normals.Ed.reshape((-1, 3, size)),
        normals.paths.reshape((-1, 3, size)),
    )
    fields = []
    for i in range(point_count):
        difference = fb.sub(points[i], gammas)  # [coils, points, 3]
        inverse_cube = _inverse_cube(fb.dot(difference, difference))
        integrand = fb.mul(
            fb.cross(gammadashes, difference),
            fb.Bounded(
                inverse_cube.v[..., None],
                inverse_cube.e[..., None],
                inverse_cube.d[..., None, :],
                inverse_cube.D[..., None, :],
                inverse_cube.Ed[..., None, :],
                inverse_cube.paths[..., None, :],
            ),
        )
        weighted = fb.mul(
            integrand,
            fb.Bounded(
                currents.v[:, None, None],
                currents.e[:, None, None],
                currents.d[:, None, None, :],
                currents.D[:, None, None, :],
                currents.Ed[:, None, None, :],
                currents.paths[:, None, None, :],
            ),
        )
        # mu0 / (4 pi) / quadrature: each lane may round this constant twice.
        fields.append(
            fb.scale(
                fb.total(weighted, (0, 1)), 1.0e-7 / quadrature, relative_error=2.0
            )
        )
    field = fb.stack(fields)  # [surface points, 3]
    normal_norm = fb.norm(normals)
    unit_normal = fb.div(
        normals,
        fb.Bounded(
            normal_norm.v[:, None],
            normal_norm.e[:, None],
            normal_norm.d[:, None, :],
            normal_norm.D[:, None, :],
            normal_norm.Ed[:, None, :],
            normal_norm.paths[:, None, :],
        ),
    )
    normal_field = fb.dot(field, unit_normal)
    # Native: 0.5 sum((B.n)^2 |n|) / N.  JAX: 0.5 sum((B.n sqrt(|n| / N))^2).
    native = fb.scale(
        fb.total(fb.mul(fb.square(normal_field), normal_norm), 0),
        0.5 / point_count,
        relative_error=1.0,
    )
    residual = fb.mul(
        normal_field,
        fb.sqrt(fb.scale(normal_norm, 1.0 / point_count, relative_error=1.0)),
    )
    jax_form = fb.scale(fb.total(fb.square(residual), 0), 0.5)
    return fb.envelope(native, jax_form)


def _pairwise_distance(first: fb.Bounded, second: fb.Bounded) -> fb.Bounded:
    """``|first[a] - second[b]|`` over ``[a, b]`` for ``[a, 3]`` and ``[b, 3]``."""

    def expand(quantity, axis):
        index = (slice(None), None) if axis == 0 else (None, slice(None))
        return fb.Bounded(
            quantity.v[index],
            quantity.e[index],
            quantity.d[index],
            quantity.D[index],
            quantity.Ed[index],
            quantity.paths[index],
        )

    return fb.norm(fb.sub(expand(first, 0), expand(second, 1)))


def _outer(first: fb.Bounded, second: fb.Bounded) -> fb.Bounded:
    return fb.mul(
        fb.Bounded(
            first.v[:, None],
            first.e[:, None],
            first.d[:, None],
            first.D[:, None],
            first.Ed[:, None],
            first.paths[:, None],
        ),
        fb.Bounded(
            second.v[None],
            second.e[None],
            second.d[None],
            second.D[None],
            second.Ed[None],
            second.paths[None],
        ),
    )


def _curve_curve(
    gammas, gammadashes, threshold: float, base_count: int, size: int
) -> fb.Bounded:
    """Upstream ``CurveCurveDistance``: pairs ``(i, j)``, ``j < min(i, num_basecurves)``."""
    speeds = fb.norm(gammadashes)
    minimum = fb.constant(threshold, size)
    pairs = []
    for i in range(gammas.v.shape[0]):
        for j in range(min(i, base_count)):
            excess = fb.positive_part(
                fb.sub(minimum, _pairwise_distance(gammas[i], gammas[j]))
            )
            integrand = fb.mul(_outer(speeds[i], speeds[j]), fb.square(excess))
            pairs.append(fb.mean(integrand, (0, 1)))
    return fb.total(fb.stack(pairs), 0)


def _curve_surface(
    gammas, gammadashes, surface_points, surface_normals, threshold: float
) -> fb.Bounded:
    """Upstream ``CurveSurfaceDistance`` over every coil (both lanes share the surface arrays)."""
    speeds = fb.norm(gammadashes)
    normal_norms = fb.norm(surface_normals)
    minimum = fb.constant(threshold, gammas.components)
    per_coil = []
    for c in range(gammas.v.shape[0]):
        excess = fb.positive_part(
            fb.sub(minimum, _pairwise_distance(gammas[c], surface_points))
        )
        integrand = fb.mul(_outer(speeds[c], normal_norms), fb.square(excess))
        per_coil.append(fb.mean(integrand, (0, 1)))
    return fb.total(fb.stack(per_coil), 0)


def _curvature(gammadash: fb.Bounded, gammadashdash: fb.Bounded) -> fb.Bounded:
    """``|gamma' x gamma''| / |gamma'|^3`` (``**3`` as two products)."""
    speed = fb.norm(gammadash)
    return fb.div(
        fb.norm(fb.cross(gammadash, gammadashdash)), fb.mul(fb.mul(speed, speed), speed)
    )


def _lp_curvature(kappa: fb.Bounded, speed: fb.Bounded, threshold: float) -> fb.Bounded:
    """Upstream ``Lp_curvature_pure`` with ``p = 2``; ``excess ** 2.0`` is a ``pow``."""
    excess = fb.positive_part(fb.sub(kappa, fb.constant(threshold, kappa.components)))
    powered = fb.extra_roundings(fb.square(excess), 1.0)
    per_curve = fb.scale(fb.mean(fb.mul(powered, speed), 1), 0.5, relative_error=1.0)
    return fb.total(per_curve, 0)


def _mean_squared_curvature_penalty(
    kappa: fb.Bounded, speed: fb.Bounded, threshold: float
) -> fb.Bounded:
    """Upstream ``curve_msc_pure`` per curve, ``QuadraticPenalty`` summed."""
    msc = fb.div(fb.mean(fb.mul(fb.square(kappa), speed), 1), fb.mean(speed, 1))
    excess = fb.sub(msc, fb.constant(threshold, kappa.components))
    return fb.scale(fb.total(fb.square(excess), 0), 0.5)


@pytest.mark.parametrize(
    ("k", "parameter_key", "weight_name"),
    CASES,
    ids=[f"k{k}-{key.split(':')[0]}" for k, key, _weight in CASES],
)
def test_native_and_jax_objective_and_gradient_agree_at_native_bounded_states(
    evaluator: _NativePlanarEvaluator,
    native_states: dict[int, dict[str, np.ndarray]],
    k: int,
    parameter_key: str,
    weight_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    parameters = native_states[k][parameter_key]
    length_weight = float(CONFIGURATION[weight_name])

    objective = evaluator.weighted(length_weight)
    objective.x = parameters
    native_value = float(objective.J())
    native_gradient = np.asarray(objective.dJ(), dtype=np.float64)

    surface, _base_curves, coils = _geometry()
    field = BiotSavartJAX(coils)
    state = standard_stage_two_state(
        field=field,
        flux_spec=SquaredFluxJAX(surface, field).fixed_surface_flux_spec(),
        surface_gamma=np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)),
        surface_normal=np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)),
        parameters=parameters,
        regularization_config=_regularization_config(),
        length_weight=np.asarray(length_weight, dtype=np.float64),
    )
    jax_value = float(np.asarray(jax.device_get(state.objective)))
    jax_gradient = np.asarray(
        jax.device_get(state.objective_gradient), dtype=np.float64
    )

    model = _planar_objective_bound(evaluator, parameters, length_weight).total
    value_bound, gradient_bound = fb.cross_implementation_bounds(model)
    value_bound = float(value_bound)

    # The bound's own float64 evaluation of the formula is a third
    # implementation: if it strayed from the native lane, the bound would be a
    # bound on some other formula.
    assert abs(float(model.v) - native_value) <= value_bound
    assert np.all(np.abs(model.d - native_gradient) <= gradient_bound)

    value_difference = abs(jax_value - native_value)
    assert value_difference <= value_bound, (
        f"k={k} {parameter_key}: JAX objective {jax_value!r} against native "
        f"{native_value!r}: {value_difference:.3e} over the derived bound "
        f"{value_bound:.3e}"
    )
    assert jax_gradient.shape == native_gradient.shape
    gradient_difference = np.abs(jax_gradient - native_gradient)
    worst = int(np.argmax(gradient_difference / gradient_bound))
    assert np.all(gradient_difference <= gradient_bound), (
        f"k={k} {parameter_key}: JAX gradient component {worst} differs from "
        f"native by {gradient_difference[worst]:.3e}, over the derived bound "
        f"{gradient_bound[worst]:.3e}"
    )
