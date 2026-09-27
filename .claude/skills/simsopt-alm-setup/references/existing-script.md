# Adapting an existing script

When the user's current optimization script should use the generated files
(step 3.5 of `SKILL.md`), change it by this pattern, show the unified diff,
and apply it only after the user approves.

## Pattern

1. Put `<dir>` first on `sys.path` and import `build_problem` from the
   generated `alm_problem.py` and `run` from the generated `run_alm.py`. Call
   `run(build_problem())` directly; do not start `run_alm.py` as a
   subprocess. `run(problem, history=None, checkpoints=None, resume=None)`
   returns the summary dict (`termination_reason`, `max_violation`,
   `multipliers`, ..., `finish`) and leaves the problem's objects at the
   returned x.
2. Move the script's setup values into the `SETUP` constants of
   `alm_problem.py` (the table below), then delete what the constraints
   replace: the penalty weights, the penalty objects, the objective sum, the
   `fun` wrapper, the hand-written Taylor test (`gradient_check.py` replaces
   it) and the optimizer calls. The setup code that built the surface, coils
   and field goes too: `alm_problem.py` builds them.
3. Output code that used the script's objects reads them from the problem:
   `problem.surface`, `problem.base_curves`, `problem.curves`,
   `problem.biot_savart` (Stage 2; the Boozer template has
   `problem.boozer_surface`).
4. Keep the rest of the script (its output directory, its plots, its saves).

## Stage 2: from upstream's stage_two_optimization.py

The penalty script below is simsopt's
`examples/2_Intermediate/stage_two_optimization.py` shortened (its Taylor
test, its second run with a smaller length weight and its progress printout
removed). Its values go to these `alm_problem.py` constants of the Stage-2
template, and each penalty term becomes these rows:

| Penalty script | alm_problem.py | Row, and what changes |
|---|---|---|
| `ncoils` | `NCOILS` | |
| `R0` | `R0` | |
| `R1` | `R1` | |
| `order` | `ORDER` | |
| `nphi`, `ntheta` | `QUADRATURE_POINTS` | One number for both directions. |
| `filename` | `SURFACE_FILE` | |
| `LENGTH_WEIGHT * sum(Jls)`, with `LENGTH_WEIGHT = Weight(1e-6)` | `LENGTH_WEIGHT` | Stays a term of f: it has no threshold, so it is not a constraint. The template's is the plain float `1e-6` (upstream's `Weight` only lets a script change it between runs). For a length limit set `MAX_LENGTH` with `LENGTH_SCOPE = SUM_OF_BASE_COILS`, the same sum over the base coils. |
| `CC_WEIGHT * CurveCurveDistance(curves, CC_THRESHOLD)` | `CC_MIN_DISTANCE` | Row `coil_coil_distance` (= `CC_THRESHOLD`): a smooth minimum over every pair of physical coils. |
| `CS_WEIGHT * CurveSurfaceDistance(curves, s, CS_THRESHOLD)` | `CS_MIN_DISTANCE` | Row `coil_surface_distance` (= `CS_THRESHOLD`). |
| `CURVATURE_WEIGHT * sum(LpCurveCurvature(c, 2, CURVATURE_THRESHOLD))` | `MAX_CURVATURE` | Rows `max_curvature_<i>`, one per base coil (= `CURVATURE_THRESHOLD`): the row bounds the pointwise maximum, where the penalty weighed the L2 norm of the excess. |
| `MSC_WEIGHT * QuadraticPenalty(MeanSquaredCurvature(c), MSC_THRESHOLD, "max")` | `MAX_MEAN_SQUARED_CURVATURE` | Rows `mean_squared_curvature_<i>`, one per base coil (= `MSC_THRESHOLD`). |
| `MAXITER` | `inner_options["maxiter"]` | The budget of the whole ALM run (all subproblems), not of one `minimize` call: give it several times the penalty run's value. |

Deleted without a counterpart: the weights `CC_WEIGHT`, `CS_WEIGHT`,
`CURVATURE_WEIGHT`, `MSC_WEIGHT`; the objects `Jf`, `Jls`, `Jccdist`,
`Jcsdist`, `Jcs`, `Jmscs`, `JF`; `fun` and the `minimize` call; the setup
objects `s`, `base_curves`, `base_currents`, `coils`, `bs`, `curves`. A
weight continuation (upstream reruns with `LENGTH_WEIGHT *= 0.1`) is not
needed: to trade coil length for B.n, change `LENGTH_WEIGHT` or `MAX_LENGTH`
and rerun.

The change, with `<dir>` = `alm_stage2/` next to the script (the generated
`alm_problem.py` has the constants above; in CI, `in_github_actions` selects
the smoke size as upstream's `MAXITER` did):

```diff
--- a/my_stage2.py
+++ b/my_stage2.py
@@ -1,67 +1,23 @@
 #!/usr/bin/env python
-"""Stage-II coils with penalty weights (simsopt's
-examples/2_Intermediate/stage_two_optimization.py, shortened)."""
+"""Stage-II coils with the ALM solver: the coil-regularity terms are the
+constraint rows of alm_stage2/alm_problem.py instead of penalty weights."""
 
 import os
+import sys
 from pathlib import Path
-from scipy.optimize import minimize
-from simsopt.field import BiotSavart, Current, coils_via_symmetries
-from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
-                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
-                         LpCurveCurvature, CurveSurfaceDistance)
-from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
+from simsopt.geo import curves_to_vtk
 from simsopt.util import in_github_actions
 
-ncoils = 4
-R0 = 1.0
-R1 = 0.5
-order = 5
-LENGTH_WEIGHT = Weight(1e-6)
-CC_THRESHOLD = 0.1
-CC_WEIGHT = 1000
-CS_THRESHOLD = 0.3
-CS_WEIGHT = 10
-CURVATURE_THRESHOLD = 5.
-CURVATURE_WEIGHT = 1e-6
-MSC_THRESHOLD = 5
-MSC_WEIGHT = 1e-6
-MAXITER = 50 if in_github_actions else 400
-TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
-filename = TEST_DIR / 'input.LandremanPaul2021_QA'
+# The generated files, alm_stage2/alm_problem.py and alm_stage2/run_alm.py:
+sys.path.insert(0, str(Path(__file__).resolve().parent / "alm_stage2"))
+from alm_problem import build_problem  # noqa: E402
+from run_alm import run  # noqa: E402
+
 OUT_DIR = "./output/"
 os.makedirs(OUT_DIR, exist_ok=True)
 
-nphi = 32
-ntheta = 32
-s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
-base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
-base_currents = [Current(1e5) for i in range(ncoils)]
-base_currents[0].fix_all()
-coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
-bs = BiotSavart(coils)
-bs.set_points(s.gamma().reshape((-1, 3)))
-curves = [c.curve for c in coils]
-
-Jf = SquaredFlux(s, bs)
-Jls = [CurveLength(c) for c in base_curves]
-Jccdist = CurveCurveDistance(curves, CC_THRESHOLD, num_basecurves=ncoils)
-Jcsdist = CurveSurfaceDistance(curves, s, CS_THRESHOLD)
-Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
-Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
-JF = Jf \
-    + LENGTH_WEIGHT * sum(Jls) \
-    + CC_WEIGHT * Jccdist \
-    + CS_WEIGHT * Jcsdist \
-    + CURVATURE_WEIGHT * sum(Jcs) \
-    + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
-
-
-def fun(dofs):
-    JF.x = dofs
-    return JF.J(), JF.dJ()
-
-
-res = minimize(fun, JF.x, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300}, tol=1e-15)
-print(res.message)
-curves_to_vtk(curves, OUT_DIR + "curves_opt")
-bs.save(OUT_DIR + "biot_savart_opt.json")
+problem = build_problem(smoke=in_github_actions)
+summary = run(problem)
+print(summary["termination_reason"], summary["message"])
+curves_to_vtk(problem.curves, OUT_DIR + "curves_opt")
+problem.biot_savart.save(OUT_DIR + "biot_savart_opt.json")
```

## Other scripts

The same four steps apply to any script: its thresholds become `SETUP`
constants, its penalty terms rows, and its optimizer call `run(problem)`.
For a stateful problem (the Boozer template) `run` passes the warm-start
callbacks from `problem.solver_callbacks()` itself; the script adds nothing.
