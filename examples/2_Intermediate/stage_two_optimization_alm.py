#!/usr/bin/env python
r"""
Stage-II coil optimization with the augmented Lagrangian (ALM) solver.

This is the problem of ``stage_two_optimization.py`` (same target equilibrium,
same initial coils), but the coil-regularity terms are constraints
``g_i(x) <= 0`` instead of weighted penalties:

    minimize    (1/2) \int |B dot n|^2 ds + LENGTH_WEIGHT * (sum CurveLength)
    subject to  CC_THRESHOLD - (min coil-to-coil distance)      <= 0
                CS_THRESHOLD - (min coil-to-surface distance)   <= 0
                max(curvature) - CURVATURE_THRESHOLD            <= 0   (each base coil)
                MeanSquaredCurvature - MSC_THRESHOLD            <= 0   (each base coil)

The distance and curvature rows come from ``simsopt.geo.signed_constraints``:
smooth (log-sum-exp) signed values that keep their slack when inactive, unlike
the hinge objectives ``CurveCurveDistance`` or ``LpCurveCurvature``. They are
conservative (the smoothed minimum distance is at most the true one, the
smoothed maximum curvature at least the true one), so a feasible point meets the
thresholds with some margin to spare. The mean
squared curvature row wraps the stock objective with ``signed_upper_bound``.
``minimize_alm`` then needs no weights for these terms.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from functools import partial
from pathlib import Path

import numpy as np

from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, MeanSquaredCurvature)
from simsopt.geo.signed_constraints import (smooth_max_curvature_signed_constraint,
                                            smooth_min_curve_curve_signed_constraint,
                                            smooth_min_curve_surface_signed_constraint)
from simsopt.objectives import SquaredFlux
from simsopt.solve.alm import (ALMSettings, alm_problem_physics, cached_alm_evaluator,
                               minimize_alm, run_directional_taylor_test, signed_upper_bound)
from simsopt.util import in_github_actions

# Coils and weights as in stage_two_optimization.py:
ncoils = 4
R0 = 1.0
R1 = 0.5
order = 5
LENGTH_WEIGHT = 1e-6

# Thresholds of stage_two_optimization.py. The distance and curvature rows are
# smooth (log-sum-exp) bounds over every point pair of every coil, symmetry
# copies included, and never looser than the true extremum: a point feasible
# for the smooth row keeps the true minimum distance above, or the true maximum
# curvature below, the threshold, so the effective margin is larger than the
# threshold alone.
CC_THRESHOLD = 0.1
CS_THRESHOLD = 0.3
CURVATURE_THRESHOLD = 5.
MSC_THRESHOLD = 5

# Smoothing temperatures of the signed constraints, in the units of the
# constrained quantity (meters for distances, 1/meters for curvature):
DISTANCE_TEMPERATURE = 0.005
CURVATURE_TEMPERATURE = 0.05

# L-BFGS-B iterations per ALM subproblem, and number of multiplier updates:
MAXITER = 50 if in_github_actions else 400
MAX_OUTER_ITERATIONS = 3 if in_github_actions else 10

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"

#######################################################
# End of input parameters.
#######################################################

nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
# The target field is zero, so fix one current to rule out the zero-current solution:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]

# The objective f. Its free dofs (coil shapes and currents) are the optimization
# variables, and every constraint gradient is taken with respect to them.
Jf = SquaredFlux(s, bs)
Jls = [CurveLength(c) for c in base_curves]
JF = Jf + LENGTH_WEIGHT * sum(Jls)

# One row per constraint: row(JF) -> (signed value, gradient, ...).
constraint_names = ["coil_coil_distance", "coil_surface_distance"]
inequalities = [
    partial(smooth_min_curve_curve_signed_constraint, curves, CC_THRESHOLD, DISTANCE_TEMPERATURE),
    partial(smooth_min_curve_surface_signed_constraint, curves, s, CS_THRESHOLD, DISTANCE_TEMPERATURE),
]
for i, c in enumerate(base_curves):
    constraint_names.append(f"max_curvature_{i}")
    inequalities.append(partial(smooth_max_curvature_signed_constraint, c, CURVATURE_THRESHOLD, CURVATURE_TEMPERATURE))
for i, c in enumerate(base_curves):
    constraint_names.append(f"mean_squared_curvature_{i}")
    inequalities.append(partial(signed_upper_bound, MeanSquaredCurvature(c), MSC_THRESHOLD))


# minimize_alm evaluates some coils more than once, with new multipliers or a new
# penalty. The physics (Jf, the constraint rows and their gradients) depends only
# on the coils, so the evaluator reuses it and recomputes only the augmented terms,
# bit for bit. It reads nothing else that changes, so its cache never needs clearing.
evaluate_problem = cached_alm_evaluator(partial(alm_problem_physics, base_objective=JF,
                                                inequalities=inequalities))


def report(label, objective, max_violation, constraint_values):
    print(f"{label}: Jf={objective:.3e}, max violation={max_violation:.3e}")
    for name, value in zip(constraint_names, constraint_values):
        print(f"    g[{name}] = {value:+.3e}")


def main():
    """Run the Taylor test and the optimization; return the test verdict and the
    initial and final objective and feasibility."""
    os.makedirs(OUT_DIR, exist_ok=True)
    curves_to_vtk(curves, OUT_DIR + "curves_init")

    print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
    # The signed constraints keep only points near the extremum, and a point that
    # enters or leaves that selection makes the rows non-smooth. At the largest
    # step here (1e-5 along a unit direction) the curvature moves by about 5e-4
    # 1/m and the distances by about 1e-5 m, about 1% of the smoothing
    # temperatures and far below the selection window, so the differences stay
    # in the smooth regime; at 1e-4 the first error ratio already shows a jump.
    # A row enters the augmented Lagrangian through max(0, multiplier + penalty * g),
    # so with zero multipliers every inactive row (g < 0) drops out unchecked. These
    # multipliers make that shift at least 1 for every row at the initial coils, so
    # every row's gradient is tested.
    dofs = JF.x.copy()
    initial_evaluation = evaluate_problem(dofs, np.zeros(len(inequalities)), 1.0)
    taylor_multipliers = 1.0 + np.maximum(0.0, -np.asarray(initial_evaluation["constraint_values"]))
    taylor = run_directional_taylor_test(evaluate_problem, dofs, taylor_multipliers, 1.0,
                                         epsilons=[1e-5, 5e-6, 2.5e-6])
    print(f"Taylor test passed: {taylor['passed']}")

    print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
    report("initial", initial_evaluation["base_value"], initial_evaluation["max_violation"],
           initial_evaluation["constraint_values"])
    result = minimize_alm(
        dofs,
        constraint_names,
        evaluate_problem,
        ALMSettings(max_outer_iterations=MAX_OUTER_ITERATIONS),
        {"maxiter": MAXITER, "maxcor": 300},
    )
    print(result.message)
    # minimize_alm may return an earlier (best feasible) iterate, so set its dofs:
    JF.x = result.x
    report("final", result.objective, result.max_violation, result.constraint_values)

    curves_to_vtk(curves, OUT_DIR + "curves_opt_alm")
    pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
    s.to_vtk(OUT_DIR + "surf_opt_alm", extra_data=pointData)
    bs.save(OUT_DIR + "biot_savart_opt_alm.json")

    return {
        "initial_objective": float(initial_evaluation["base_value"]),
        "initial_max_violation": float(initial_evaluation["max_violation"]),
        "final_objective": result.objective,
        "final_max_violation": result.max_violation,
        "outer_iterations": result.outer_iterations,
        "termination_reason": result.termination_reason,
        "taylor_passed": bool(taylor["passed"]),
    }


if __name__ == "__main__":
    main()
