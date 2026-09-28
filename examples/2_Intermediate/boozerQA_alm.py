#!/usr/bin/env python3
r"""
Single-stage quasi-axisymmetry (QA) optimization of the NCSX coils with the
augmented Lagrangian (ALM) solver.

This is the problem of ``boozerQA.py`` (same coils, same Boozer surface), with
its penalty terms replaced by constraints ``g_i(x) <= 0``:

    minimize    ( \int_S B_nonQA**2 dS )/(\int_S B_QA dS)
    subject to  IOTA_TARGET - iota                           <= 0
                iota - IOTA_TARGET                           <= 0
                MAJOR_RADIUS_TARGET - (major radius)         <= 0
                (major radius) - MAJOR_RADIUS_TARGET         <= 0
                (sum CurveLength) - LENGTH_MAX               <= 0
                MIN_DIST_THRESHOLD - (min coil-to-coil distance) <= 0
                max(curvature) - KAPPA_THRESHOLD             <= 0   (each base coil)
                MeanSquaredCurvature - MSC_THRESHOLD         <= 0   (each base coil)

As in ``boozerQA.py``, the iota and major radius targets and the length cap are
the values of the initial configuration; ``boozerQA.py`` targets iota and the
major radius exactly, so each is an equality written as two rows. The coil-coil
distance, curvature, and mean squared curvature thresholds are taken from
``boozerQA_ls_mpi.py``.

Every evaluation re-solves the Boozer surface by Newton's method before any
objective reads the solve's adjoint (its LU factorization). Newton warm-starts
only from solutions the solver kept, never from a rejected line-search trial,
which may have converged to a different surface (the idea of
``boozerQA_ls_mpi.py``):

- An evaluation at the accepted coils (each outer-iteration evaluation, and the
  first evaluation of each L-BFGS-B subproblem) starts from their solution, so
  Newton converges at once.
- A line-search trial starts from the solution at the subproblem's current
  iterate. ``inner_callback`` records each iterate with the coils it was solved
  at, and ``accepted_callback`` promotes the subproblem's last iterate to the
  accepted solution once ``minimize_alm`` accepts it; a rejected subproblem
  leaves the accepted solution unchanged.

When Newton fails at a trial, the evaluation returns NaN and the line search
backtracks. ``snapshot_accepted_state_fn`` / ``restore_incumbent_state_fn`` hand
``minimize_alm`` the accepted solution, so when it returns an earlier (best
feasible) iterate, the final surface is re-solved from the solution it had there.

More details on this work can be found at doi:10.1017/S0022377822000563 or arxiv:2203.03753.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial

import numpy as np

from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import (BoozerSurface, CurveLength, Iotas, MajorRadius, MeanSquaredCurvature,
                         NonQuasiSymmetricRatio, SurfaceXYZTensorFourier, Volume, boozer_surface_residual,
                         curves_to_vtk)
from simsopt.geo.signed_constraints import (smooth_max_curvature_signed_constraint,
                                            smooth_min_curve_curve_signed_constraint)
from simsopt.solve.alm import (ALMSettings, augmented_inequality_objective, minimize_alm,
                               signed_lower_bound, signed_upper_bound)
from simsopt.util import in_github_actions

# Thresholds taken from boozerQA_ls_mpi.py. The distance and curvature rows are
# smooth (log-sum-exp) bounds on the true extremum, so they are conservative:
# a point feasible for the smooth row keeps the true distance above, or the true
# curvature below, the threshold, with some extra margin.
MIN_DIST_THRESHOLD = 0.15
KAPPA_THRESHOLD = 15.
MSC_THRESHOLD = 15.

# Smoothing temperatures of the signed constraints, in the units of the
# constrained quantity (meters for distances, 1/meters for curvature):
DISTANCE_TEMPERATURE = 0.005
CURVATURE_TEMPERATURE = 0.05

# Surface resolution: Fourier modes, which also set the quadrature grid
# (2*mpol+1 by 2*ntor+1 points per field period).
MPOL = NTOR = 4 if in_github_actions else 6

# The inner L-BFGS-B options (maxiter: iterations for the whole minimize_alm
# call, one budget shared by all subproblems) and the outer settings. In CI the
# run's cost is bounded by counting evaluations, not only iterations: a path
# whose line-search trials fail Newton spends up to maxls + 1 evaluations per
# iteration and can re-solve a subproblem max_subproblem_continuations times,
# so maxfun, maxls and the continuations cap every subproblem call.
if in_github_actions:
    INNER_OPTIONS = {"maxiter": 10, "maxfun": 15, "maxls": 5}
    SETTINGS = ALMSettings(max_outer_iterations=2, max_subproblem_continuations=2)
else:
    INNER_OPTIONS = {"maxiter": 1000}
    SETTINGS = ALMSettings(max_outer_iterations=10)

# Directory for output
OUT_DIR = "./output/"


@dataclass(frozen=True, eq=False)
class BoozerSolution:
    """A solved Boozer surface and the coil dofs it was solved at."""
    coil_dofs: np.ndarray
    surface_dofs: np.ndarray
    iota: float
    G: float


def _read_only_copy(array):
    array = np.array(array, dtype=float)
    array.setflags(write=False)
    return array


class BoozerQAProblem:
    """The constrained problem, and the Boozer solutions that warm-start Newton."""

    def __init__(self, *, boozer_surface, J_nonQSRatio, iotas, iota_target, major_radius,
                 major_radius_target, coil_lengths, curves, constraint_names, inequalities):
        self.boozer_surface = boozer_surface
        self.J_nonQSRatio = J_nonQSRatio
        self.iotas = iotas
        self.iota_target = iota_target
        self.major_radius = major_radius
        self.major_radius_target = major_radius_target
        self.coil_lengths = coil_lengths
        self.curves = curves
        self.constraint_names = constraint_names
        self.inequalities = inequalities
        # Accepted by minimize_alm, the current L-BFGS-B iterate, and the last
        # successful solve:
        self.accepted = self.iterate = self.solved = self._solution_at(J_nonQSRatio.x)

    def _solution_at(self, coil_dofs):
        return BoozerSolution(
            coil_dofs=_read_only_copy(coil_dofs),
            surface_dofs=_read_only_copy(self.boozer_surface.surface.x),
            iota=self.boozer_surface.res['iota'],
            G=self.boozer_surface.res['G'],
        )

    def _restore_surface(self, solution):
        self.boozer_surface.surface.x = solution.surface_dofs

    def solve(self, coil_dofs):
        """Solve the Boozer surface at ``coil_dofs``; return the solver's result."""
        if np.array_equal(coil_dofs, self.accepted.coil_dofs):
            self.iterate = self.accepted
        start = self.iterate
        self._restore_surface(start)
        # Setting the coils always marks the surface for a re-solve, and run_code
        # seeds Newton with the start's iota and G.
        self.J_nonQSRatio.x = coil_dofs
        res = self.boozer_surface.run_code(start.iota, G=start.G)
        if res['success']:
            self.solved = self._solution_at(coil_dofs)
        else:
            self._restore_surface(start)
        return res

    def evaluate(self, coil_dofs, multipliers, penalty):
        """``evaluate_problem`` of ``minimize_alm``."""
        # Not a cached_alm_evaluator: warm-started Newton can differ on a second solve at the same coils.
        # Solve first: every objective below differentiates through the solve's
        # LU factorization, which a failed solve may not have (res['PLU'] is None).
        if not self.solve(coil_dofs)['success']:
            # A non-finite evaluation: minimize_alm treats the point as unusable
            # and backtracks from its current iterate.
            print("Boozer Newton solve failed; keeping the last good surface")
            return augmented_inequality_objective(
                base_value=np.nan,
                base_grad=np.full(coil_dofs.size, np.nan),
                constraint_values=np.full(len(self.inequalities), np.nan),
                constraint_grads=[np.full(coil_dofs.size, np.nan) for _ in self.inequalities],
                multipliers=multipliers,
                penalty=penalty,
            )
        # The coils stay set, so the objectives below reuse this solve.
        rows = [row(self.J_nonQSRatio)[:2] for row in self.inequalities]
        return augmented_inequality_objective(
            base_value=float(self.J_nonQSRatio.J()),
            base_grad=np.asarray(self.J_nonQSRatio.dJ(), dtype=float),
            constraint_values=[signed_value for signed_value, _grad in rows],
            constraint_grads=[grad for _signed_value, grad in rows],
            multipliers=multipliers,
            penalty=penalty,
        )

    def accept_inner_iterate(self, coil_dofs):
        """``inner_callback``: L-BFGS-B moved its iterate to ``coil_dofs``."""
        if not np.array_equal(coil_dofs, self.solved.coil_dofs):
            raise RuntimeError("inner_callback at coils other than the last successful solve")
        self.iterate = self.solved

    def accept_outer_iterate(self, coil_dofs):
        """``accepted_callback``: ``minimize_alm`` accepted the subproblem's result."""
        if not np.array_equal(coil_dofs, self.iterate.coil_dofs):
            raise RuntimeError("accepted_callback at coils other than the current L-BFGS-B iterate")
        self.accepted = self.iterate

    def snapshot_accepted(self):
        """``snapshot_accepted_state_fn``."""
        return self.accepted

    def restore_incumbent(self, solution):
        """``restore_incumbent_state_fn``: ``minimize_alm`` went back to an earlier iterate."""
        # minimize_alm evaluates nothing after this, so the coils stay whatever the caller sets.
        self.accepted = self.iterate = solution
        self._restore_surface(solution)

    def report(self, label, objective, max_violation, constraint_values):
        print(f"{label}: J_nonQSRatio={objective:.3e}, max violation={max_violation:.3e}")
        for name, value in zip(self.constraint_names, constraint_values):
            print(f"    g[{name}] = {value:+.3e}")


def build_problem():
    """Solve the initial Boozer surface of the NCSX coils; build the constrained problem."""
    base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
    # bs.coils includes all coils after symmetry expansion (not just the base coils).
    coils = bs.coils
    curves = [c.curve for c in coils]
    current_sum = sum(abs(c.current.get_value()) for c in coils)
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    ## COMPUTE THE INITIAL SURFACE ON WHICH WE WANT TO OPTIMIZE FOR QA ##
    mpol = MPOL
    ntor = NTOR
    stellsym = True

    phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    s = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.fit_to_curve(ma, 0.1, flip_theta=True)
    iota = -0.406

    # Use a volume surface label
    vol = Volume(s)
    vol_target = vol.J()

    boozer_surface = BoozerSurface(bs, s, vol, vol_target)
    res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota, G=G0)
    out_res = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)[0]
    print(f"NEWTON {res['success']}: iter={res['iter']}, iota={res['iota']:.3f}, vol={s.volume():.3f}, "
          f"||residual||={np.linalg.norm(out_res):.3e}")

    ## SET UP THE CONSTRAINED PROBLEM ##
    # boozerQA.py targets the initial iota and major radius exactly, so each band
    # has zero width (an equality written as two rows).
    iota_target = res['iota']
    major_radius = MajorRadius(boozer_surface)
    major_radius_target = float(major_radius.J())

    J_nonQSRatio = NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils))
    coil_lengths = [CurveLength(c) for c in base_curves]
    total_length = sum(coil_lengths)
    length_max = float(total_length.J())
    iotas = Iotas(boozer_surface)

    # let's fix the coil current
    base_currents[0].fix_all()

    # One row per constraint: row(J_nonQSRatio) -> (signed value, gradient, ...).
    constraint_names = ["iota_min", "iota_max", "major_radius_min", "major_radius_max",
                        "total_coil_length", "coil_coil_distance"]
    inequalities = [
        partial(signed_lower_bound, iotas, iota_target),
        partial(signed_upper_bound, iotas, iota_target),
        partial(signed_lower_bound, major_radius, major_radius_target),
        partial(signed_upper_bound, major_radius, major_radius_target),
        partial(signed_upper_bound, total_length, length_max),
        partial(smooth_min_curve_curve_signed_constraint, curves, MIN_DIST_THRESHOLD, DISTANCE_TEMPERATURE),
    ]
    for i, c in enumerate(base_curves):
        constraint_names.append(f"max_curvature_{i}")
        inequalities.append(partial(smooth_max_curvature_signed_constraint, c, KAPPA_THRESHOLD, CURVATURE_TEMPERATURE))
    for i, c in enumerate(base_curves):
        constraint_names.append(f"mean_squared_curvature_{i}")
        inequalities.append(partial(signed_upper_bound, MeanSquaredCurvature(c), MSC_THRESHOLD))

    return BoozerQAProblem(
        boozer_surface=boozer_surface, J_nonQSRatio=J_nonQSRatio, iotas=iotas, iota_target=iota_target,
        major_radius=major_radius, major_radius_target=major_radius_target,
        coil_lengths=coil_lengths, curves=curves, constraint_names=constraint_names, inequalities=inequalities,
    )


def main():
    """Run the optimization; return its initial and final objective and
    feasibility, and the final surface's iota and major radius."""
    print("Running 2_Intermediate/boozerQA_alm.py")
    print("======================================")
    problem = build_problem()
    os.makedirs(OUT_DIR, exist_ok=True)
    curves_to_vtk(problem.curves, OUT_DIR + "curves_init")
    problem.boozer_surface.surface.to_vtk(OUT_DIR + "surf_init")

    print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""")
    dofs = problem.J_nonQSRatio.x.copy()
    initial_evaluation = problem.evaluate(dofs, np.zeros(len(problem.inequalities)), 1.0)
    problem.report("initial", initial_evaluation["base_value"], initial_evaluation["max_violation"],
                   initial_evaluation["constraint_values"])
    result = minimize_alm(
        dofs,
        problem.constraint_names,
        problem.evaluate,
        SETTINGS,
        INNER_OPTIONS,
        inner_callback=problem.accept_inner_iterate,
        accepted_callback=problem.accept_outer_iterate,
        snapshot_accepted_state_fn=problem.snapshot_accepted,
        restore_incumbent_state_fn=problem.restore_incumbent,
    )
    print(result.message)
    accepted_coils_are_result_x = bool(np.array_equal(problem.snapshot_accepted().coil_dofs, result.x))
    # minimize_alm may return an earlier (best feasible) iterate, so solve at its dofs:
    final_boozer_res = problem.solve(result.x)
    print(f"Boozer surface of the returned coils solved: {final_boozer_res['success']}")
    problem.report("final", result.objective, result.max_violation, result.constraint_values)
    final_iota = float(problem.iotas.J())
    final_major_radius = float(problem.major_radius.J())
    total_length = sum(J.J() for J in problem.coil_lengths)
    print(f"iota={final_iota:.4f}, major radius={final_major_radius:.4f}, "
          f"Len=sum([{', '.join(f'{J.J():.1f}' for J in problem.coil_lengths)}])={total_length:.1f}")

    curves_to_vtk(problem.curves, OUT_DIR + "curves_opt_alm")
    problem.boozer_surface.surface.to_vtk(OUT_DIR + "surf_opt_alm")

    print("End of 2_Intermediate/boozerQA_alm.py")
    print("======================================")
    return {
        "initial_objective": float(initial_evaluation["base_value"]),
        "initial_max_violation": float(initial_evaluation["max_violation"]),
        "final_objective": result.objective,
        "final_max_violation": result.max_violation,
        "outer_iterations": result.outer_iterations,
        "termination_reason": result.termination_reason,
        "final_surface_solved": bool(final_boozer_res["success"]),
        "accepted_coils_are_result_x": accepted_coils_are_result_x,
        "final_iota": final_iota,
        "iota_target": float(problem.iota_target),
        "final_major_radius": final_major_radius,
        "major_radius_target": float(problem.major_radius_target),
        "final_constraint_values": {
            name: float(value)
            for name, value in zip(problem.constraint_names, result.constraint_values)
        },
    }


if __name__ == "__main__":
    main()
