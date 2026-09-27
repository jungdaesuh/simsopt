#!/usr/bin/env python
r"""Boozer single-stage optimization with the ALM solver: coils optimized for
quasi-symmetry on a Boozer surface that is re-solved at every evaluation.

Copy this file to ``alm_problem.py`` next to ``run_alm.py`` and edit the parts
marked ``SETUP``. As shipped it is the problem of simsopt's
``examples/2_Intermediate/boozerQA_alm.py`` (NCSX coils), with f divided by
its initial value and every row divided by the size of its bound:

    minimize    J(x) / J(x0),  J = (\int_S B_nonQA^2 dS) / (\int_S B_QA dS)
    subject to  iota within IOTA_TARGET +- IOTA_HALF_WIDTH                  (two rows)
                major radius within MAJOR_RADIUS_TARGET +- its half width   (two rows)
                sum_i CurveLength_i <= LENGTH_MAX
                min coil-coil distance >= CC_MIN_DISTANCE                     (smooth row)
                max curvature_i <= MAX_CURVATURE                              (smooth row, each base coil)
                MeanSquaredCurvature_i <= MAX_MEAN_SQUARED_CURVATURE          (each base coil)

A target of None takes the initial configuration's value; a zero half width
makes the pair of rows an equality.

The evaluator is stateful: every evaluation re-solves the Boozer surface by
Newton's method, warm-started from a solution the solver accepted (never from
a rejected line-search trial), so a second evaluation at the same coils can
differ from the first. It is therefore NOT a ``cached_alm_evaluator``, and it
hands ``minimize_alm`` the accepted solution through
``snapshot_accepted_state_fn`` / ``restore_incumbent_state_fn``, so a returned
best-feasible iterate or a resumed run re-solves from the solution it had.
A failed Newton solve returns a non-finite evaluation, and the line search
backtracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import (BoozerSurface, CurveLength, Iotas, MajorRadius, MeanSquaredCurvature,
                         NonQuasiSymmetricRatio, SurfaceXYZTensorFourier, Volume)
from simsopt.geo.signed_constraints import (smooth_max_curvature_signed_constraint,
                                            smooth_min_curve_curve_signed_constraint)
from simsopt.solve.alm import (ALMPhysics, ALMResult, ALMSettings, signed_lower_bound,
                               signed_upper_bound)

# SETUP: the coils (a simsopt.configs name) and the initial surface guess.
COIL_CONFIG = "ncsx"
INITIAL_IOTA = -0.406          # Newton's starting iota
SURFACE_MINOR_RADIUS = 0.1     # m, the initial surface is fitted around the magnetic axis

# SETUP: targets and bounds; None takes the initial configuration's value.
IOTA_TARGET: Optional[float] = None
IOTA_HALF_WIDTH = 0.0                          # 0: iota is held at the target
MAJOR_RADIUS_TARGET: Optional[float] = None    # m
MAJOR_RADIUS_HALF_WIDTH = 0.0                  # m
LENGTH_MAX: Optional[float] = None             # m, sum over the base coils
CC_MIN_DISTANCE = 0.15                         # m, coil to coil
MAX_CURVATURE = 15.0                           # 1/m, each base coil
MAX_MEAN_SQUARED_CURVATURE = 15.0              # 1/m^2, each base coil

# SETUP: smoothing temperatures of the smooth rows, in the constrained
# quantity's units.
DISTANCE_TEMPERATURE = 0.005   # m
CURVATURE_TEMPERATURE = 0.05   # 1/m

UPPER_BOUND = 1.0    # quantity <= bound
LOWER_BOUND = -1.0   # quantity >= bound


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
    constrained quantity from an independent computation at the current
    solve, which the sign probes compare with ``bound`` from the side
    ``sense``, ignoring points within ``margin`` of it."""

    name: str
    row: Callable
    measure: Callable[[], float]
    sense: float
    bound: float
    margin: float


@dataclass(frozen=True, eq=False)
class BoozerSolution:
    """A solved Boozer surface and the coil dofs it was solved at."""

    coil_dofs: np.ndarray
    surface_dofs: np.ndarray
    iota: float
    G: float


def scaled_row(row, scale, base_objective):
    """``row(base_objective)`` with every item divided by ``scale`` > 0."""
    return tuple(item / scale for item in row(base_objective))


def read_only_copy(array) -> np.ndarray:
    array = np.array(array, dtype=float)
    array.setflags(write=False)
    return array


def min_curve_curve_distance(curves) -> float:
    return float(min(np.min(cdist(curves[i].gamma(), curves[j].gamma()))
                     for i in range(len(curves)) for j in range(i)))


def mean_squared_curvature(curve) -> float:
    arclength = np.linalg.norm(curve.gammadash(), axis=1)
    return float(np.mean(curve.kappa() ** 2 * arclength) / np.mean(arclength))


def total_length(curves) -> float:
    return float(sum(np.mean(np.linalg.norm(curve.gammadash(), axis=1)) for curve in curves))


class BoozerSingleStageProblem:
    """The problem-module contract of the simsopt-alm-setup skill (see its
    ``references/api.md``) for a stateful, Boozer-surface physics."""

    name = "boozer_single_stage"
    # Steps small enough for the smooth rows and for Newton's warm start.
    taylor_epsilons = (1e-5, 5e-6, 2.5e-6)

    def __init__(self, smoke: bool):
        base_curves, base_currents, magnetic_axis, nfp, biot_savart = get_data(COIL_CONFIG)
        self.base_curves = base_curves
        self.curves = [coil.curve for coil in biot_savart.coils]
        current_sum = sum(abs(coil.current.get_value()) for coil in biot_savart.coils)
        G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

        # SETUP: surface resolution; smoke runs a coarse one.
        mpol = ntor = 3 if smoke else 6
        surface = SurfaceXYZTensorFourier(
            mpol=mpol, ntor=ntor, stellsym=True, nfp=nfp,
            quadpoints_phi=np.linspace(0, 1 / nfp, 2 * ntor + 1, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 2 * mpol + 1, endpoint=False))
        surface.fit_to_curve(magnetic_axis, SURFACE_MINOR_RADIUS, flip_theta=True)
        volume = Volume(surface)
        self.boozer_surface = BoozerSurface(biot_savart, surface, volume, volume.J())
        res = self.boozer_surface.solve_residual_equation_exactly_newton(
            tol=1e-13, maxiter=20, iota=INITIAL_IOTA, G=G0)
        if not res["success"]:
            raise RuntimeError("the initial Boozer surface did not converge; adjust INITIAL_IOTA "
                               "or SURFACE_MINOR_RADIUS")

        self.iotas = Iotas(self.boozer_surface)
        self.major_radius = MajorRadius(self.boozer_surface)
        self.coil_lengths = [CurveLength(curve) for curve in base_curves]
        # SETUP: the objective f, any simsopt Optimizable of the coil dofs.
        unscaled = NonQuasiSymmetricRatio(self.boozer_surface, BiotSavart(biot_savart.coils))
        # Fix one current so the problem cannot scale all currents together.
        base_currents[0].fix_all()
        # J / J(x0): the stationarity tolerance becomes relative to the start.
        self.objective_scale = float(unscaled.J())
        self.objective = (1.0 / self.objective_scale) * unscaled
        self.x0 = self.objective.x.copy()
        self.row_specs = self._row_specs(
            iota_target=float(res["iota"]) if IOTA_TARGET is None else IOTA_TARGET,
            major_radius_target=(float(self.major_radius.J()) if MAJOR_RADIUS_TARGET is None
                                 else MAJOR_RADIUS_TARGET),
            length_max=float(sum(self.coil_lengths).J()) if LENGTH_MAX is None else LENGTH_MAX,
        )
        self.constraint_names = tuple(spec.name for spec in self.row_specs)
        self.rows = tuple(spec.row for spec in self.row_specs)
        # Accepted by minimize_alm, the current L-BFGS-B iterate, and the last successful solve.
        self.accepted = self.iterate = self.solved = self._solution_at(self.x0)
        # Not a cached_alm_evaluator: a warm-started re-solve at the same coils can differ.
        self.evaluator = self.evaluate
        self.settings = ALMSettings(
            max_outer_iterations=2 if smoke else 10,  # each outer iteration is one multiplier update
            # Rows are divided by the size of their bounds: 1e-4 is a
            # violation of 0.01% of a bound.
            feasibility_tol=1e-4,
            # J is divided by J(x0): 1e-4 asks for a 1e-4 relative gradient.
            stationarity_tol=1e-4,
        )
        # maxiter is the L-BFGS-B budget of the whole minimize_alm call (all
        # subproblems), not a per-subproblem limit.
        self.inner_options = {"maxiter": 10 if smoke else 5000}

    def _row_specs(self, *, iota_target, major_radius_target, length_max) -> Tuple[RowSpec, ...]:
        # SETUP: one RowSpec per row; a new row needs a scaled row callable
        # and an independent measure of the same quantity for the sign probes.
        surface = self.boozer_surface.surface
        total_coil_length = sum(self.coil_lengths)
        band_rows = (
            ("iota", self.iotas, lambda: float(self.boozer_surface.res["iota"]),
             iota_target, IOTA_HALF_WIDTH, abs(iota_target), 1e-6),
            ("major_radius", self.major_radius, lambda: float(surface.major_radius()),
             major_radius_target, MAJOR_RADIUS_HALF_WIDTH, major_radius_target, 1e-6),
        )
        specs = []
        for name, quantity, measure, target, half_width, scale, margin in band_rows:
            specs.append(RowSpec(
                name=f"{name}_min",
                row=partial(scaled_row, partial(signed_lower_bound, quantity, target - half_width), scale),
                measure=measure, sense=LOWER_BOUND, bound=target - half_width, margin=margin))
            specs.append(RowSpec(
                name=f"{name}_max",
                row=partial(scaled_row, partial(signed_upper_bound, quantity, target + half_width), scale),
                measure=measure, sense=UPPER_BOUND, bound=target + half_width, margin=margin))
        specs.append(RowSpec(
            name="total_coil_length",
            row=partial(scaled_row, partial(signed_upper_bound, total_coil_length, length_max), length_max),
            measure=partial(total_length, self.base_curves),
            sense=UPPER_BOUND, bound=length_max, margin=1e-6))
        specs.append(RowSpec(
            name="coil_coil_distance",
            row=partial(scaled_row, partial(smooth_min_curve_curve_signed_constraint, self.curves,
                                            CC_MIN_DISTANCE, DISTANCE_TEMPERATURE), CC_MIN_DISTANCE),
            measure=partial(min_curve_curve_distance, self.curves),
            sense=LOWER_BOUND, bound=CC_MIN_DISTANCE, margin=10 * DISTANCE_TEMPERATURE))
        for i, curve in enumerate(self.base_curves):
            specs.append(RowSpec(
                name=f"max_curvature_{i}",
                row=partial(scaled_row, partial(smooth_max_curvature_signed_constraint, curve,
                                                MAX_CURVATURE, CURVATURE_TEMPERATURE), MAX_CURVATURE),
                measure=lambda curve=curve: float(np.max(curve.kappa())),
                sense=UPPER_BOUND, bound=MAX_CURVATURE, margin=10 * CURVATURE_TEMPERATURE))
        for i, curve in enumerate(self.base_curves):
            specs.append(RowSpec(
                name=f"mean_squared_curvature_{i}",
                row=partial(scaled_row, partial(signed_upper_bound, MeanSquaredCurvature(curve),
                                                MAX_MEAN_SQUARED_CURVATURE), MAX_MEAN_SQUARED_CURVATURE),
                measure=partial(mean_squared_curvature, curve),
                sense=UPPER_BOUND, bound=MAX_MEAN_SQUARED_CURVATURE, margin=1e-6))
        return tuple(specs)

    def _solution_at(self, coil_dofs) -> BoozerSolution:
        return BoozerSolution(
            coil_dofs=read_only_copy(coil_dofs),
            surface_dofs=read_only_copy(self.boozer_surface.surface.x),
            iota=self.boozer_surface.res["iota"],
            G=self.boozer_surface.res["G"],
        )

    def _restore_surface(self, solution: BoozerSolution) -> None:
        self.boozer_surface.surface.x = solution.surface_dofs

    def solve(self, coil_dofs) -> dict:
        """Solve the Boozer surface at ``coil_dofs`` from the current warm start."""
        if np.array_equal(coil_dofs, self.accepted.coil_dofs):
            self.iterate = self.accepted
        start = self.iterate
        self._restore_surface(start)
        # Setting the coils marks the surface for a re-solve; run_code seeds
        # Newton with the start's iota and G.
        self.objective.x = coil_dofs
        res = self.boozer_surface.run_code(start.iota, G=start.G)
        if res["success"]:
            self.solved = self._solution_at(coil_dofs)
        else:
            self._restore_surface(start)
        return res

    def physics(self, x) -> ALMPhysics:
        x = np.asarray(x, dtype=float)
        # Solve first: every objective below differentiates through the solve.
        if not self.solve(x)["success"]:
            # Non-finite: minimize_alm rejects the trial and backtracks.
            return ALMPhysics(
                base_value=np.nan,
                base_grad=np.full(x.size, np.nan),
                constraint_values=np.full(len(self.rows), np.nan),
                constraint_grads=tuple(np.full(x.size, np.nan) for _ in self.rows),
            )
        values = [row(self.objective)[:2] for row in self.rows]
        return ALMPhysics(
            base_value=float(self.objective.J()),
            base_grad=np.asarray(self.objective.dJ(), dtype=float),
            constraint_values=np.array([value for value, _grad in values]),
            constraint_grads=tuple(grad for _value, grad in values),
        )

    def evaluate(self, x, multipliers, penalty):
        """``evaluate_problem`` of ``minimize_alm``: one fresh solve per call."""
        return self.physics(x).evaluation(multipliers, penalty)

    def accept_inner_iterate(self, coil_dofs) -> None:
        """``inner_callback``: L-BFGS-B moved its iterate to ``coil_dofs``."""
        if not np.array_equal(coil_dofs, self.solved.coil_dofs):
            raise RuntimeError("inner_callback at coils other than the last successful solve")
        self.iterate = self.solved

    def accept_outer_iterate(self, coil_dofs) -> None:
        """``accepted_callback``: ``minimize_alm`` accepted the subproblem's result."""
        if not np.array_equal(coil_dofs, self.iterate.coil_dofs):
            raise RuntimeError("accepted_callback at coils other than the current L-BFGS-B iterate")
        self.accepted = self.iterate

    def snapshot_accepted(self) -> BoozerSolution:
        """``snapshot_accepted_state_fn``."""
        return self.accepted

    def restore_incumbent(self, solution: BoozerSolution) -> None:
        """``restore_incumbent_state_fn``: back to an earlier accepted solution."""
        self.accepted = self.iterate = solution
        self._restore_surface(solution)

    def solver_callbacks(self) -> dict:
        """The warm-start bookkeeping ``minimize_alm`` drives."""
        return {
            "inner_callback": self.accept_inner_iterate,
            "accepted_callback": self.accept_outer_iterate,
            "snapshot_accepted_state_fn": self.snapshot_accepted,
            "restore_incumbent_state_fn": self.restore_incumbent,
        }

    def _moved_x(self, quantity, step: float) -> np.ndarray:
        """x0 moved by ``step`` along the gradient of ``quantity`` (at x0's solve)."""
        self.solve(self.x0)
        grad = np.asarray(quantity.dJ(partials=True)(self.objective), dtype=float)
        return self.x0 + step * grad / np.linalg.norm(grad)

    def sign_probes(self) -> Tuple[SignProbe, ...]:
        """x0 and small steps that raise or lower iota, the major radius and
        the total length; each row's status comes from its independent
        ``measure`` at the probe's solve, and rows within ``margin`` of the
        bound are not asserted. The steps stay small so Newton's warm start
        converges, so rows far from their bounds are probed on one side only."""
        points = [("initial coils", self.x0)]
        for label, quantity in (("iota", self.iotas), ("major radius", self.major_radius),
                                ("total length", sum(self.coil_lengths))):
            points.append((f"{label} raised", self._moved_x(quantity, 1e-3)))
            points.append((f"{label} lowered", self._moved_x(quantity, -1e-3)))
        probes = []
        for label, x in points:
            if not self.solve(x)["success"]:
                raise RuntimeError(f"the Boozer solve failed at the sign probe {label!r}")
            expected = [spec.sense * (spec.measure() - spec.bound) / spec.margin for spec in self.row_specs]
            probes.append(SignProbe(
                label=label,
                x=x,
                violated=tuple(spec.name for spec, e in zip(self.row_specs, expected) if e > 1.0),
                satisfied=tuple(spec.name for spec, e in zip(self.row_specs, expected) if e < -1.0),
            ))
        self.solve(self.x0)
        return tuple(probes)

    def finish(self, result: ALMResult) -> dict:
        """Re-solve at the returned x (possibly a restored best-feasible
        iterate, whose accepted solution the solver restored) and report."""
        res = self.solve(result.x)
        return {
            "surface_solved": bool(res["success"]),
            "nonqs_ratio": float(result.objective) * self.objective_scale,
            "iota": float(self.boozer_surface.res["iota"]),
            "major_radius": float(self.boozer_surface.surface.major_radius()),
            "total_coil_length": total_length(self.base_curves),
            "min_coil_coil_distance": min_curve_curve_distance(self.curves),
        }


def build_problem(smoke: bool = False) -> BoozerSingleStageProblem:
    return BoozerSingleStageProblem(smoke)
