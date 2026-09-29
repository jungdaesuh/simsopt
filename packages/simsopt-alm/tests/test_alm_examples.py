"""Tests for the ALM stock demos in ``examples``.

The smoke tests run each demo in a fresh interpreter with ``CI=true``, so
``in_github_actions`` shrinks it to its CI size, and in a scratch directory,
since the demos write ``./output/``. The probe imports the demo, records the
type of each ``minimize_alm`` result, calls the ``main()`` its
``if __name__ == "__main__"`` block runs, and prints what it returns.

Importing a demo builds nothing that runs Newton: ``boozerQA_alm`` builds its
problem in ``build_problem()``, which the evaluator tests below call per test.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import simsopt_alm

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import boozerQA_alm  # noqa: E402  (module scope builds no problem)

PROBE_PREFIX = "ALM_DEMO_PROBE "
_PROBE = (
    "import json, os\n"
    "import simsopt_alm\n"
    "import {module}\n"
    "result_types = []\n"
    "library_minimize_alm = {module}.minimize_alm\n"
    "def recording_minimize_alm(*args, **kwargs):\n"
    "    result = library_minimize_alm(*args, **kwargs)\n"
    "    result_types.append(type(result).__module__ + '.' + type(result).__name__)\n"
    "    return result\n"
    "{module}.minimize_alm = recording_minimize_alm\n"
    "summary = {module}.main()\n"
    "print({prefix!r} + json.dumps({{\n"
    "    **summary,\n"
    "    'result_types': result_types,\n"
    "    'simsopt_alm_file': os.path.realpath(simsopt_alm.__file__),\n"
    "}}))\n"
)


def _child_env():
    """``CI=true``; the demo directory first on ``PYTHONPATH``."""
    python_path = [str(EXAMPLES_DIR)]
    if os.environ.get("PYTHONPATH"):
        python_path.append(os.environ["PYTHONPATH"])
    return {**os.environ, "CI": "true", "PYTHONPATH": os.pathsep.join(python_path)}


def _run_demo(module, timeout_seconds):
    with tempfile.TemporaryDirectory() as scratch_dir:
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE.format(module=module, prefix=PROBE_PREFIX)],
            cwd=scratch_dir,
            env=_child_env(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed_seconds = time.perf_counter() - started
    probe_lines = [
        line for line in completed.stdout.splitlines()
        if line.startswith(PROBE_PREFIX)
    ]
    return completed, probe_lines, elapsed_seconds


class AlmDemoSmokeTests(unittest.TestCase):
    def _run_demo_and_check_feasibility(self, module, timeout_seconds):
        completed, probe_lines, elapsed_seconds = _run_demo(module, timeout_seconds)

        self.assertEqual(
            completed.returncode,
            0,
            f"{module} exited {completed.returncode}:\n"
            f"{completed.stdout[-4000:]}\n{completed.stderr[-4000:]}",
        )
        self.assertEqual(len(probe_lines), 1, completed.stdout[-4000:])
        summary = json.loads(probe_lines[0][len(PROBE_PREFIX):])
        print(f"{module}: {summary}, {elapsed_seconds:.1f} s")
        self.assertEqual(
            summary["simsopt_alm_file"],
            os.path.realpath(simsopt_alm.__file__),
            f"{module} imported another simsopt_alm than this test process",
        )
        self.assertEqual(
            summary["result_types"],
            ["simsopt_alm.control.ALMResult"],
            f"{module} must read the library's lean result",
        )
        self.assertLessEqual(
            summary["final_max_violation"],
            summary["initial_max_violation"],
            f"{module} ended less feasible than it started: {summary}",
        )
        return summary

    def test_stage_two_demo(self):
        summary = self._run_demo_and_check_feasibility(
            "stage_two_optimization_alm", timeout_seconds=300
        )
        self.assertIs(
            summary["taylor_passed"],
            True,
            "the demo's Taylor test of the augmented Lagrangian gradient failed",
        )

    def test_boozer_qa_demo(self):
        summary = self._run_demo_and_check_feasibility(
            "boozerQA_alm", timeout_seconds=600
        )
        self.assertIs(
            summary["final_surface_solved"],
            True,
            "the Boozer surface of the returned coils did not solve",
        )
        self.assertIs(
            summary["accepted_coils_are_result_x"],
            True,
            "the problem's accepted solution does not belong to the returned coils",
        )
        # The final surface must be the one minimize_alm evaluated at result.x
        # (result.constraint_values), not another solution of the same coils.
        constraint_values = summary["final_constraint_values"]
        self.assertAlmostEqual(
            summary["final_iota"] - summary["iota_target"],
            constraint_values["iota_max"],
            delta=1e-9,
            msg=f"final iota differs from the returned evaluation's: {summary}",
        )
        self.assertAlmostEqual(
            summary["final_major_radius"] - summary["major_radius_target"],
            constraint_values["major_radius_max"],
            delta=1e-9,
            msg=f"final major radius differs from the returned evaluation's: {summary}",
        )


def _record_newton_start_surfaces(boozer_surface):
    """Patch ``boozer_surface``'s Newton solve to record the surface it starts from."""
    real_solve = boozer_surface.solve_residual_equation_exactly_newton
    start_surfaces = []

    def record_start_surface(*args, **kwargs):
        start_surfaces.append(boozer_surface.surface.x.copy())
        return real_solve(*args, **kwargs)

    patch = mock.patch.object(
        boozer_surface, "solve_residual_equation_exactly_newton", side_effect=record_start_surface
    )
    return patch, start_surfaces


class BoozerQADemoEvaluatorTests(unittest.TestCase):
    """The evaluator and warm-start bookkeeping of ``boozerQA_alm.py``.

    Each test builds its own problem, so no warm-start state carries between tests.
    """

    def setUp(self):
        self.problem = boozerQA_alm.build_problem()
        self.boozer_surface = self.problem.boozer_surface
        self.initial_dofs = self.problem.J_nonQSRatio.x.copy()
        rng = np.random.default_rng(0)
        self.step_1 = 1e-4 * rng.standard_normal(self.initial_dofs.size)
        self.step_2 = 1e-4 * rng.standard_normal(self.initial_dofs.size)
        self.multipliers = np.zeros(len(self.problem.inequalities))

    def _evaluate(self, coil_dofs):
        return self.problem.evaluate(coil_dofs, self.multipliers, 1.0)

    def _promote_inner_iterate(self):
        """Evaluate at new coils and make them the current L-BFGS-B iterate."""
        inner_iterate = self.initial_dofs + self.step_1
        self._evaluate(inner_iterate)
        self.problem.accept_inner_iterate(inner_iterate)
        self.assertFalse(
            np.array_equal(self.problem.iterate.surface_dofs, self.problem.snapshot_accepted().surface_dofs),
            "the iterate's surface must differ from the accepted one for this test to discriminate",
        )
        return inner_iterate

    def test_failed_newton_solve_returns_nan_and_keeps_the_iterate_surface(self):
        inner_iterate = self._promote_inner_iterate()
        iterate_surface = self.problem.iterate.surface_dofs
        real_solve = self.boozer_surface.solve_residual_equation_exactly_newton
        trial_surfaces = []

        def solve_with_nonfinite_jacobian(*args, **kwargs):
            # BoozerSurface's result when Newton's Jacobian goes non-finite:
            # no factorization, and no success.
            res = {**real_solve(*args, **kwargs), "success": False, "PLU": None}
            trial_surfaces.append(self.boozer_surface.surface.x.copy())
            self.boozer_surface.res = res
            return res

        with mock.patch.object(
            self.boozer_surface,
            "solve_residual_equation_exactly_newton",
            side_effect=solve_with_nonfinite_jacobian,
        ):
            evaluation = self._evaluate(inner_iterate + self.step_2)

        self.assertEqual(len(trial_surfaces), 1, "the trial coils were not solved exactly once")
        self.assertFalse(
            np.array_equal(trial_surfaces[0], iterate_surface),
            "the failed trial must move the surface for this test to discriminate",
        )
        self.assertTrue(
            np.isnan(evaluation["total"]),
            "a failed Newton solve must give a non-finite total so minimize_alm backtracks",
        )
        np.testing.assert_array_equal(
            self.boozer_surface.surface.x,
            iterate_surface,
            err_msg="a failed solve must leave the current iterate's surface in place",
        )

    def test_successful_evaluation_solves_the_boozer_surface_once(self):
        patch, start_surfaces = _record_newton_start_surfaces(self.boozer_surface)
        with patch:
            evaluation = self._evaluate(self.initial_dofs + self.step_1)

        self.assertEqual(
            len(start_surfaces),
            1,
            "the objective and every constraint row must reuse one Newton solve per evaluation",
        )
        self.assertTrue(self.boozer_surface.res["success"])
        self.assertTrue(np.isfinite(evaluation["total"]))
        self.assertTrue(np.all(np.isfinite(evaluation["grad"])))

    def test_outer_evaluation_after_a_rejected_subproblem_starts_from_the_accepted_surface(self):
        accepted = self.problem.snapshot_accepted()
        self._promote_inner_iterate()
        # minimize_alm rejects the subproblem's candidate: no accepted_callback,
        # and the next outer-iteration evaluation is at the accepted coils.
        patch, start_surfaces = _record_newton_start_surfaces(self.boozer_surface)
        with patch:
            self._evaluate(self.initial_dofs)

        np.testing.assert_array_equal(
            start_surfaces[0],
            accepted.surface_dofs,
            err_msg="the outer evaluation warm-started from a rejected subproblem's iterate",
        )
        np.testing.assert_array_equal(self.problem.snapshot_accepted().coil_dofs, self.initial_dofs)

    def test_callbacks_promote_only_matching_coils_and_raise_otherwise(self):
        with self.assertRaisesRegex(RuntimeError, "inner_callback"):
            self.problem.accept_inner_iterate(self.initial_dofs + self.step_2)
        inner_iterate = self._promote_inner_iterate()
        iterate_surface = self.problem.iterate.surface_dofs

        with self.assertRaisesRegex(RuntimeError, "accepted_callback"):
            self.problem.accept_outer_iterate(self.initial_dofs + self.step_2)
        np.testing.assert_array_equal(
            self.problem.snapshot_accepted().coil_dofs,
            self.initial_dofs,
            err_msg="coils that are not the current iterate must not become accepted",
        )

        self.problem.accept_outer_iterate(inner_iterate)
        accepted = self.problem.snapshot_accepted()
        np.testing.assert_array_equal(accepted.coil_dofs, inner_iterate)
        np.testing.assert_array_equal(accepted.surface_dofs, iterate_surface)

    def test_line_search_trials_start_from_the_current_iterate_not_the_last_trial(self):
        inner_iterate = self._promote_inner_iterate()
        iterate_surface = self.problem.iterate.surface_dofs
        self._evaluate(inner_iterate + self.step_2)  # a trial the line search rejects
        self.assertFalse(
            np.array_equal(self.boozer_surface.surface.x, iterate_surface),
            "the trial coils must move the surface for this test to discriminate",
        )

        patch, start_surfaces = _record_newton_start_surfaces(self.boozer_surface)
        with patch:
            self._evaluate(inner_iterate + 0.5 * self.step_2)

        np.testing.assert_array_equal(
            start_surfaces[0],
            iterate_surface,
            err_msg="Newton warm-started from a rejected trial's surface",
        )

    def test_iota_and_major_radius_rows_are_equalities_at_the_initial_values(self):
        evaluation = self._evaluate(self.initial_dofs)

        values = dict(zip(self.problem.constraint_names, evaluation["constraint_values"]))
        for name in ("iota_min", "iota_max", "major_radius_min", "major_radius_max"):
            self.assertAlmostEqual(
                values[name], 0.0, delta=1e-10,
                msg=f"{name} must be active at the initial coils, as boozerQA.py targets that value",
            )


if __name__ == "__main__":
    unittest.main()
