#!/usr/bin/env python
r"""Stage-2 coil optimization with the ALM solver: coils for a fixed target
surface, with coil-regularity requirements as constraints instead of weights.

Copy this file to ``alm_problem.py`` next to ``run_alm.py`` and edit the parts
marked ``SETUP``. As shipped it is the problem of the simsopt-alm package's
``examples/stage_two_optimization_alm.py`` (the QA target of arXiv:2108.03711,
four base coils), with f divided by the size of its initial value and every
row divided by its threshold, so that all of them are O(1):

    minimize    f(x) / |f(x0)|,  f = (1/2) \int |B.n|^2 ds + LENGTH_WEIGHT * sum_i CurveLength_i
    subject to  (CC_MIN_DISTANCE - min coil-coil distance) / CC_MIN_DISTANCE    <= 0
                (CS_MIN_DISTANCE - min coil-surface distance) / CS_MIN_DISTANCE <= 0
                (max curvature_i - MAX_CURVATURE) / MAX_CURVATURE                <= 0  (each base coil)
                (MeanSquaredCurvature_i - MAX_MEAN_SQUARED_CURVATURE)
                    / MAX_MEAN_SQUARED_CURVATURE                                 <= 0  (each base coil)
                (coil length - MAX_LENGTH) / MAX_LENGTH                          <= 0  (if set)

where the coil length is, by ``LENGTH_SCOPE``, each base coil's length, the
sum over the NCOILS base coils, or the sum over all 2 * nfp * NCOILS physical
coils after the stellarator symmetry (2 * nfp times the base-coil sum).

The distance and maximum-curvature rows are the smooth signed constraints of
``simsopt_alm.signed_constraints``: log-sum-exp values never looser than the
extremum over the sampled quadrature points (the "hard" value), so a point
feasible for the smooth row is feasible over those samples. That is not a
bound on the continuous coils, which can come closer or bend more between
samples: re-evaluate clearance and curvature at a higher quadrature resolution
before accepting a physical bound. The physics depends on the coil dofs alone,
so the solver gets ``cached_alm_evaluator(physics)``. With ``HYBRID_QUARTET =
True`` the evaluator also returns the hard row values: the smooth values still
define the augmented Lagrangian, while the hard ones decide feasibility and
drive the multiplier update; ``converged`` needs both within
``feasibility_tol`` (read the skill's ``references/pitfalls.md`` first).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, NamedTuple, Tuple

import numpy as np
from scipy.spatial.distance import cdist

import simsopt_alm
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (CurveLength, MeanSquaredCurvature, SurfaceRZFourier,
                         create_equally_spaced_curves, curves_to_vtk)
from simsopt.objectives import SquaredFlux
from simsopt_alm import (ALMPhysics, ALMResult, ALMSettings, alm_problem_physics,
                         cached_alm_evaluator, signed_upper_bound)
from simsopt_alm.signed_constraints import (smooth_max_curvature_signed_constraint,
                                            smooth_min_curve_curve_signed_constraint,
                                            smooth_min_curve_surface_signed_constraint)

# SETUP: the target surface (a VMEC input file). The default is the QA target
# of arXiv:2108.03711 that simsopt-alm ships (simsopt's
# tests/test_files/input.LandremanPaul2021_QA).
SURFACE_FILE = Path(simsopt_alm.__file__).parent / "data" / "input.LandremanPaul2021_QA"

# SETUP: the surface sampling, the initial coils and the objective.
QUADRATURE_POINTS = 32  # surface quadrature points in each direction (nphi = ntheta)
ORDER = 5               # Fourier order of each coil
NCOILS = 4              # base coils per half field period
R0 = 1.0                # m, major radius of the initial circular coils
R1 = 0.5                # m, minor radius of the initial circular coils
CURRENT = 1e5           # A, initial current of every base coil (the first stays fixed)
LENGTH_WEIGHT = 1e-6    # weight of the base-coil length sum in f (a regularizer, not a constraint)
# f is divided by |f(x0)|, a positive scale, so its sign and so the direction
# of minimization are kept; an f(x0) of 0 has no size, and f is divided by
# this positive reference (in f's units) instead.
ZERO_OBJECTIVE_SCALE = 1.0

# What the coil-length bound applies to:
PER_BASE_COIL = "per_base_coil"
SUM_OF_BASE_COILS = "sum_of_base_coils"
SUM_OF_ALL_COILS = "sum_of_all_coils"

# SETUP: the constraint thresholds; None drops that row.
CC_MIN_DISTANCE = 0.1              # m, coil to coil
CS_MIN_DISTANCE = 0.3              # m, coil to surface
MAX_CURVATURE = 5.0                # 1/m, each base coil
MAX_MEAN_SQUARED_CURVATURE = 5.0   # 1/m^2, each base coil
# SETUP: the coil-length bound in m (None drops it) and its scope:
#   PER_BASE_COIL      each of the NCOILS base coils: NCOILS rows "length_of_base_coil_<i>";
#   SUM_OF_BASE_COILS  the sum over the NCOILS base coils: one row "length_of_base_coils";
#   SUM_OF_ALL_COILS   the sum over all physical coils after the symmetry, 2 * nfp * NCOILS
#                      of them (16 for the shipped QA target), so 2 * nfp times the base-coil
#                      sum: one row "length_of_all_coils".
MAX_LENGTH = None
LENGTH_SCOPE = SUM_OF_BASE_COILS

# SETUP: smoothing temperatures of the smooth rows, in the constrained
# quantity's units. Smaller is closer to the sampled extremum but less smooth.
DISTANCE_TEMPERATURE = 0.005   # m
CURVATURE_TEMPERATURE = 0.05   # 1/m

# SETUP: True returns the hybrid quartet (sampled extrema for feasibility and
# the multiplier update; converged needs the smooth rows feasible too); False
# uses the smooth values for everything.
HYBRID_QUARTET = False

OUT_DIR = Path("output_alm")

# Row senses: quantity <= bound (UPPER_BOUND) or quantity >= bound (LOWER_BOUND).
UPPER_BOUND = 1.0
LOWER_BOUND = -1.0


class SignProbe(NamedTuple):
    """A point where the listed rows are known to be violated (g > 0) or
    satisfied (g <= 0), independently of the row code."""

    label: str
    x: np.ndarray
    violated: Tuple[str, ...]
    satisfied: Tuple[str, ...]


class RowSpec(NamedTuple):
    """One constraint row. ``row(base_objective)`` returns the scaled
    ``(signed value, gradient[, hard signed value])``; ``measure()`` is the
    constrained quantity at the current dofs, which the sign probes compare
    with ``bound`` from the side ``sense``, ignoring points within ``margin``
    of it. ``independent`` says whether ``measure`` is computed apart from the
    row's own code (False: it reads the same source, so the probes check the
    row's sign and bound, not its value)."""

    name: str
    row: Callable
    measure: Callable[[], float]
    sense: float
    bound: float
    margin: float
    independent: bool = True


def require_positive(name: str, value) -> float:
    """``value`` if it is a finite positive number (a threshold, scale or
    temperature: each divides or smooths a row); else ``ValueError``."""
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return float(value)


def scaled_row(row, scale, base_objective):
    """``row(base_objective)`` with every item divided by ``scale`` (checked
    positive when the row is built)."""
    return tuple(item / scale for item in row(base_objective))


def min_curve_curve_distance(curves) -> float:
    return float(min(np.min(cdist(curves[i].gamma(), curves[j].gamma()))
                     for i in range(len(curves)) for j in range(i)))


def min_curve_surface_distance(curves, surface) -> float:
    points = surface.gamma().reshape((-1, 3))
    return float(min(np.min(cdist(curve.gamma(), points)) for curve in curves))


def mean_squared_curvature(curve) -> float:
    arclength = np.linalg.norm(curve.gammadash(), axis=1)
    return float(np.mean(curve.kappa() ** 2 * arclength) / np.mean(arclength))


def total_length(curves) -> float:
    return float(sum(np.mean(np.linalg.norm(curve.gammadash(), axis=1)) for curve in curves))


class Stage2Problem:
    """The problem-module contract of the simsopt-alm-setup skill (see its
    ``references/api.md``) for Stage-2 coils."""

    name = "stage2"

    def __init__(self, smoke: bool):
        # Every threshold and temperature divides or smooths a row.
        for name, value in (("CC_MIN_DISTANCE", CC_MIN_DISTANCE), ("CS_MIN_DISTANCE", CS_MIN_DISTANCE),
                            ("MAX_CURVATURE", MAX_CURVATURE),
                            ("MAX_MEAN_SQUARED_CURVATURE", MAX_MEAN_SQUARED_CURVATURE),
                            ("MAX_LENGTH", MAX_LENGTH), ("DISTANCE_TEMPERATURE", DISTANCE_TEMPERATURE),
                            ("CURVATURE_TEMPERATURE", CURVATURE_TEMPERATURE),
                            ("ZERO_OBJECTIVE_SCALE", ZERO_OBJECTIVE_SCALE)):
            if value is not None:
                require_positive(name, value)
        # Smoke runs a coarse surface and low-order coils.
        quadrature_points = 12 if smoke else QUADRATURE_POINTS
        self.order = 2 if smoke else ORDER
        self.surface = SurfaceRZFourier.from_vmec_input(
            SURFACE_FILE, range="half period", nphi=quadrature_points, ntheta=quadrature_points)
        self.base_curves = create_equally_spaced_curves(
            NCOILS, self.surface.nfp, stellsym=True, R0=R0, R1=R1, order=self.order)
        base_currents = [Current(CURRENT) for _ in range(NCOILS)]
        # The target field is zero, so fix one current to rule out zero currents.
        base_currents[0].fix_all()
        coils = coils_via_symmetries(self.base_curves, base_currents, self.surface.nfp, True)
        self.biot_savart = BiotSavart(coils)
        self.biot_savart.set_points(self.surface.gamma().reshape((-1, 3)))
        self.curves = [coil.curve for coil in coils]
        self.coil_lengths = [CurveLength(curve) for curve in self.base_curves]
        self.squared_flux = SquaredFlux(self.surface, self.biot_savart)
        # SETUP: the objective f, any simsopt Optimizable of the coil dofs.
        unscaled = self.squared_flux + LENGTH_WEIGHT * sum(self.coil_lengths)
        # f / |f(x0)|: the stationarity tolerance becomes relative to the start.
        self.objective_scale = abs(float(unscaled.J())) or ZERO_OBJECTIVE_SCALE
        self.objective = (1.0 / self.objective_scale) * unscaled
        self.x0 = self.objective.x.copy()
        self.row_specs = self._row_specs()
        self.constraint_names = tuple(spec.name for spec in self.row_specs)
        self.rows = tuple(spec.row for spec in self.row_specs)
        # Every measure here is computed apart from its row's code.
        self.shared_source_rows = tuple(spec.name for spec in self.row_specs if not spec.independent)
        self.evaluator = cached_alm_evaluator(self.physics)
        self.settings = ALMSettings(
            max_outer_iterations=3 if smoke else 10,  # each outer iteration is one multiplier update
            # Rows are divided by their thresholds: 1e-4 is a violation of
            # 0.01% of a threshold at the sampled points (see the docstring
            # for the continuous coils).
            feasibility_tol=1e-4,
            # f is divided by |f(x0)|: 1e-4 asks for a 1e-4 relative gradient.
            stationarity_tol=1e-4,
        )
        # maxiter is the L-BFGS-B budget of the whole minimize_alm call (all
        # subproblems; never exceeded), not a per-subproblem limit; maxcor as in
        # stage_two_optimization.py.
        self.inner_options = {"maxiter": 40 if smoke else 4000, "maxcor": 300}

    def _row_specs(self) -> Tuple[RowSpec, ...]:
        # SETUP: one RowSpec per row; a new row needs a scaled row callable
        # (e.g. partial(scaled_row, partial(signed_upper_bound, objective, bound), bound))
        # and an independent measure of the same quantity for the sign probes.
        specs = []
        if CC_MIN_DISTANCE is not None:
            specs.append(RowSpec(
                name="coil_coil_distance",
                row=partial(scaled_row, partial(smooth_min_curve_curve_signed_constraint, self.curves,
                                                CC_MIN_DISTANCE, DISTANCE_TEMPERATURE), CC_MIN_DISTANCE),
                measure=partial(min_curve_curve_distance, self.curves),
                sense=LOWER_BOUND, bound=CC_MIN_DISTANCE, margin=10 * DISTANCE_TEMPERATURE))
        if CS_MIN_DISTANCE is not None:
            specs.append(RowSpec(
                name="coil_surface_distance",
                row=partial(scaled_row, partial(smooth_min_curve_surface_signed_constraint, self.curves,
                                                self.surface, CS_MIN_DISTANCE, DISTANCE_TEMPERATURE),
                            CS_MIN_DISTANCE),
                measure=partial(min_curve_surface_distance, self.curves, self.surface),
                sense=LOWER_BOUND, bound=CS_MIN_DISTANCE, margin=10 * DISTANCE_TEMPERATURE))
        for i, curve in enumerate(self.base_curves):
            if MAX_CURVATURE is not None:
                specs.append(RowSpec(
                    name=f"max_curvature_{i}",
                    row=partial(scaled_row, partial(smooth_max_curvature_signed_constraint, curve,
                                                    MAX_CURVATURE, CURVATURE_TEMPERATURE), MAX_CURVATURE),
                    measure=lambda curve=curve: float(np.max(curve.kappa())),
                    sense=UPPER_BOUND, bound=MAX_CURVATURE, margin=10 * CURVATURE_TEMPERATURE))
        for i, curve in enumerate(self.base_curves):
            if MAX_MEAN_SQUARED_CURVATURE is not None:
                specs.append(RowSpec(
                    name=f"mean_squared_curvature_{i}",
                    row=partial(scaled_row, partial(signed_upper_bound, MeanSquaredCurvature(curve),
                                                    MAX_MEAN_SQUARED_CURVATURE), MAX_MEAN_SQUARED_CURVATURE),
                    measure=partial(mean_squared_curvature, curve),
                    sense=UPPER_BOUND, bound=MAX_MEAN_SQUARED_CURVATURE, margin=1e-6))
        if MAX_LENGTH is not None:
            specs.extend(self._length_specs(MAX_LENGTH))
        return tuple(specs)

    def _length_specs(self, max_length: float) -> list:
        """The coil-length rows of ``LENGTH_SCOPE``: length <= ``max_length``
        (m) for each base coil, for the sum over the base coils, or for the
        sum over all physical coils (the base-coil sum times the number of
        symmetry copies per base coil, 2 * nfp with stellarator symmetry).
        Each ``measure`` sums the lengths of the physical curves the row
        covers, independently of that multiplicity."""
        copies_per_base_coil = len(self.curves) // len(self.base_curves)
        scoped = {
            PER_BASE_COIL: [(f"length_of_base_coil_{i}", length, [curve])
                            for i, (curve, length) in enumerate(zip(self.base_curves, self.coil_lengths))],
            SUM_OF_BASE_COILS: [("length_of_base_coils", sum(self.coil_lengths), self.base_curves)],
            SUM_OF_ALL_COILS: [("length_of_all_coils", copies_per_base_coil * sum(self.coil_lengths),
                                self.curves)],
        }[LENGTH_SCOPE]
        return [RowSpec(name=name,
                        row=partial(scaled_row, partial(signed_upper_bound, length, max_length), max_length),
                        measure=partial(total_length, curves),
                        sense=UPPER_BOUND, bound=max_length, margin=1e-6)
                for name, length, curves in scoped]

    def physics(self, x) -> ALMPhysics:
        if not HYBRID_QUARTET:
            return alm_problem_physics(x, base_objective=self.objective, inequalities=self.rows)
        self.objective.x = x
        values = [row(self.objective) for row in self.rows]
        surrogate = np.array([value[0] for value in values])
        # A row without a hard value (signed_upper_bound) is its own hard value.
        hard = np.array([value[2] if len(value) > 2 else value[0] for value in values])
        hard_violation = np.maximum(hard, 0.0)
        return ALMPhysics(
            base_value=float(self.objective.J()),
            base_grad=np.asarray(self.objective.dJ(), dtype=float),
            constraint_values=surrogate,
            constraint_grads=tuple(value[1] for value in values),
            extras={
                # Feasibility and the multiplier update read the hard values;
                # ``converged`` also needs the smooth rows L uses (surrogate)
                # within feasibility_tol, which the solver checks itself.
                "dual_update_values": hard,
                "feasibility_values": hard_violation,
                "max_feasibility_violation": float(np.max(hard_violation)),
                # The hybrid quartet: all four keys or none.
                "hard_signed_constraint_values": hard,
                "hard_violation_values": hard_violation,
                "surrogate_signed_constraint_values": surrogate,
                "hard_dual_update_values": hard,
            },
        )

    def solver_callbacks(self) -> dict:
        """Extra ``minimize_alm`` arguments; a stateless physics needs none."""
        return {}

    def _circular_coils_x(self, minor_radius: float) -> np.ndarray:
        """x0 with every base coil a circle of ``minor_radius`` (currents kept)."""
        circles = create_equally_spaced_curves(
            NCOILS, self.surface.nfp, stellsym=True, R0=R0, R1=minor_radius, order=self.order)
        for curve, circle in zip(self.base_curves, circles):
            curve.x = circle.x
        x = self.objective.x.copy()
        self.objective.x = self.x0
        return x

    def sign_probes(self) -> Tuple[SignProbe, ...]:
        """Circular coils of three minor radii; each row's status comes from its
        independent ``measure``, and rows within ``margin`` of the bound are
        not asserted."""
        probes = []
        for minor_radius in (R1, 0.9, 0.15):
            x = self._circular_coils_x(minor_radius)
            self.objective.x = x
            expected = [spec.sense * (spec.measure() - spec.bound) / spec.margin for spec in self.row_specs]
            probes.append(SignProbe(
                label=f"circular coils of minor radius {minor_radius} m",
                x=x,
                violated=tuple(spec.name for spec, e in zip(self.row_specs, expected) if e > 1.0),
                satisfied=tuple(spec.name for spec, e in zip(self.row_specs, expected) if e < -1.0),
            ))
        self.objective.x = self.x0
        return tuple(probes)

    def finish(self, result: ALMResult) -> dict:
        """Set the coils to the returned x (possibly a restored best-feasible
        iterate) and write them out."""
        self.objective.x = result.x
        OUT_DIR.mkdir(exist_ok=True)
        curves_to_vtk(self.curves, str(OUT_DIR / "curves_opt_alm"))
        self.biot_savart.save(str(OUT_DIR / "biot_savart_opt_alm.json"))
        return {
            # f without the 1 / |f(x0)| scale: the squared flux plus LENGTH_WEIGHT x the base-coil lengths.
            "objective": float(result.objective) * self.objective_scale,
            "squared_flux": float(self.squared_flux.J()),
            "min_coil_coil_distance": min_curve_curve_distance(self.curves),
            "min_coil_surface_distance": min_curve_surface_distance(self.curves, self.surface),
            "max_curvature": float(max(np.max(curve.kappa()) for curve in self.base_curves)),
            "output_dir": str(OUT_DIR.resolve()),
        }


def build_problem(smoke: bool = False) -> Stage2Problem:
    return Stage2Problem(smoke)
