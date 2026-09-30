"""Golden ALM trajectories: ``minimize_alm`` must replay bit for bit.

Each scenario in ``tests/alm_golden/alm_golden_scenarios.py`` drives
the library solver with a deterministic synthetic evaluator. Its golden holds
every ``ALMResult`` field, the history entries, every call the solver makes into
caller code in order, and every outer-boundary checkpoint.

These are characterization tests: they fail on any drift in values, order, keys
or types. L-BFGS-B iterates are bitwise only for the numpy, SciPy, machine and
OpenBLAS kernels the goldens were recorded with, so the bitwise replay runs only
there: with the recorded numpy and SciPy on x86_64, set the manifest's
recording environment (``OPENBLAS_CORETYPE=Haswell``, one thread per library)
before the test process starts. Under any
supported numpy and SciPy each scenario replays its coarse outcomes (actions,
termination and restore reasons, flags) and its numeric boundary values (x,
objective, max violation, multipliers, penalty at every outer step, outer
boundary and result) within tolerances set by its measured sensitivity to
last-bit noise (``sensitivity.json``). A scenario whose outcomes or path change
under that noise is label-only: it replays its exact outcomes only in the
recording environment, and elsewhere must finish with every outcome in the set
observed under noise and a result that keeps ``result_invariant_violations``
empty. Regenerate only for a
reviewed, intended behavior change (command in ``manifest.json``); a change that
leaves every fixture byte-identical refreshes the manifest's provenance with
``--provenance-only``.
"""

import ast
import copy
import functools
import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

GOLDEN_DIR = Path(__file__).resolve().parent / "alm_golden"
if str(GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GOLDEN_DIR))
import alm_golden_scenarios as golden  # noqa: E402
from generate_alm_golden import INTENDED_OUTCOMES, alm_source_blob_ids  # noqa: E402

MANIFEST = golden.load_manifest()
RECORDED_ENVIRONMENT = golden.recorded_environment()
_golden = golden.load_golden

# Per-scenario sensitivity to last-bit noise, written by
# measure_alm_golden_sensitivity.py (its docstring states the noise model: the
# evaluator's x moved by 1-4 ulp of max(||x||_inf, 1), every returned float
# scaled by 1 + 1-4 ulp, x0 moved by 1 ulp; deterministic in the evaluation's
# inputs, as another CPU or BLAS build would be). 144 calibration samples per
# scenario set the observed outcomes and spreads; 96 held-out samples must stay
# within the numeric tolerances and reach no unobserved outcome, so every
# recorded outcome set is closed under them.
# A scenario whose outcomes or path changed under that noise, or whose values
# moved too far for a tolerance to catch a NUMERIC_REPLAY_CEILING regression,
# is label-only (reason recorded); the others get, per quantity,
# tolerance = tolerance_factor x max(spread, eps). Measured on x86_64, numpy
# 2.4.6, SciPy 1.17.1, OpenBLAS Haswell kernels.
#
#   label-only scenario (first change seen)          intended outcome not
#                                                    reached in every sample
#   penalty_ramp_to_cap, resume_penalty_ramp_mid_run termination:penalty_cap_reached
#       (1 ulp flips the final-vs-best-feasible restore)
#   hybrid_mismatch_penalty_increase (same flip)    termination:max_outer_after_
#                                                    signal_mismatch_penalty_increase
#   plateau_restore_best_feasible,
#   resume_plateau_best_feasible (an inner retry)    -
#   frozen_warm_start_restore (an infeasible stall)  action:infeasible_stall_penalty_increase
#   cached_physics_smoothing (outer-step count)      -
#   constraints_inactive_stall (whether L-BFGS-B
#       counts an iteration at the frozen start)     -
#
#   numeric replay, largest tolerance over scenarios:
#   x              1.5e-13  (multiplier_cap_process_budget)
#   objective      2.2e-13  (inner_iteration_budget)
#   max_violation  2.5e-13  (dual_update_penalty_cap)
#   multipliers    3.1e-12  (toy_convex)
#   penalty        2.2e-15  (every scenario: eps)
SENSITIVITY = golden.load_sensitivity()
NUMERIC_REPLAY_CEILING = SENSITIVITY["noise"]["numeric_replay_ceiling"]
LABEL_ONLY_SCENARIOS = MappingProxyType({
    name: measured["label_only"]
    for name, measured in SENSITIVITY["scenarios"].items()
    if measured["label_only"] is not None
})


def required_outcomes(name: str) -> frozenset:
    """The intended outcomes a run of ``name`` reached under every perturbation;
    every replay must reach them."""
    return frozenset(INTENDED_OUTCOMES[name]) & frozenset(
        SENSITIVITY["scenarios"][name]["always_observed"]
    )


BOUNDARY_TOLERANCES = MappingProxyType({
    name: MappingProxyType({
        quantity: SENSITIVITY["noise"]["tolerance_factor"] * max(spread, golden.EPS)
        for quantity, spread in measured["spread"].items()
    })
    for name, measured in SENSITIVITY["scenarios"].items()
    if measured["label_only"] is None
})


@functools.lru_cache(maxsize=None)
def _fresh_run(name: str) -> dict:
    """One run of scenario ``name`` per test process (runs are deterministic)."""
    return golden.SCENARIOS_BY_NAME[name].run()


class AlmGoldenFixtureSetTests(unittest.TestCase):
    def test_every_scenario_has_one_unedited_golden(self):
        recorded = [entry["scenario"] for entry in MANIFEST["scenarios"]]
        self.assertEqual(
            recorded,
            [scenario.name for scenario in golden.SCENARIOS],
            "scenario catalog and manifest disagree; regenerate: "
            + MANIFEST["regenerate"],
        )
        for entry in MANIFEST["scenarios"]:
            data = (GOLDEN_DIR / entry["fixture"]).read_bytes()
            self.assertEqual(
                hashlib.sha256(data).hexdigest(),
                entry["sha256"],
                f"{entry['fixture']} differs from the recorded golden; goldens "
                "are written only by generate_alm_golden.py",
            )

    def test_the_sensitivity_was_measured_on_these_sources_and_scenarios(self):
        self.assertEqual(list(SENSITIVITY["scenarios"]), [s.name for s in golden.SCENARIOS])
        self.assertEqual(
            SENSITIVITY["source_blob_ids"],
            alm_source_blob_ids(),
            "the ALM sources changed since the sensitivity was measured; rerun "
            + SENSITIVITY["measure"],
        )

    def test_the_manifest_names_the_sources_that_produced_the_goldens(self):
        self.assertEqual(
            MANIFEST["source_blob_ids"],
            alm_source_blob_ids(),
            "the ALM sources changed since the goldens were recorded; if every "
            "fixture still replays byte-identically, refresh the provenance: "
            + MANIFEST["regenerate"] + " --provenance-only",
        )

    def test_the_recorded_source_ids_cover_the_rules_that_judge_a_replay(self):
        # The generator's INTENDED_OUTCOMES and the measurement's noise and
        # tolerance rules decide what a replay requires, like the scenarios.
        ids = alm_source_blob_ids()
        for script in (
            "alm_golden_scenarios.py",
            "generate_alm_golden.py",
            "measure_alm_golden_sensitivity.py",
        ):
            with self.subTest(script=script):
                data = (GOLDEN_DIR / script).read_bytes()
                self.assertEqual(
                    ids.get(script),
                    hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
                )

    def test_every_golden_covers_its_intended_outcomes(self):
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                missed = set(INTENDED_OUTCOMES[scenario.name]) - set(
                    _golden(scenario.name)["outcomes"]
                )
                self.assertEqual(sorted(missed), [])


class AlmGoldenEnvironmentTests(unittest.TestCase):
    """The bitwise replay runs only in the recorded environment, OpenBLAS
    kernels included: the kernel family decides the last bits of the BLAS
    calls inside L-BFGS-B and the evaluators."""

    def test_the_manifest_records_the_openblas_kernels(self):
        self.assertTrue(MANIFEST["environment"].get("openblas_coretype"))

    def test_the_sensitivity_was_measured_in_the_recording_environment(self):
        self.assertEqual(SENSITIVITY["environment"], MANIFEST["environment"])

    def test_the_kernels_are_read_from_the_loaded_openblas(self):
        if sys.platform != "linux":
            self.skipTest("OPENBLAS_CORETYPE=Haswell names a kernel of the Linux x86-64 wheels")
        # numpy's and SciPy's OpenBLAS may detect a host differently (None);
        # pinned before they load, both run and report the pinned kernels.
        pinned = subprocess.run(
            [sys.executable, "-c",
             "import alm_golden_scenarios as g; print(g.openblas_coretype())"],
            cwd=GOLDEN_DIR, env={**os.environ, "OPENBLAS_CORETYPE": "Haswell"},
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(pinned.stdout.strip(), "Haswell")
        self.assertEqual(golden.current_environment()[-1], golden.openblas_coretype())

    def test_the_kernel_probe_loads_no_library(self):
        """R16-08: the probe asks threadpoolctl about the OpenBLAS numpy and
        SciPy already loaded; it opens no library itself (no ctypes)."""
        tree = ast.parse((GOLDEN_DIR / "alm_golden_scenarios.py").read_text(encoding="utf-8"))
        roots = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                 for alias in node.names}
        roots |= {node.module.split(".")[0] for node in ast.walk(tree)
                  if isinstance(node, ast.ImportFrom) and node.module}
        self.assertNotIn("ctypes", roots)
        self.assertNotIn("importlib", roots)
        self.assertIn("threadpoolctl", roots)

    def test_the_kernel_probe_fails_closed(self):
        """No OpenBLAS, or OpenBLAS builds that report different kernels, give
        no kernel family, so the bitwise gate stays shut."""
        cases = {
            "no OpenBLAS": [{"internal_api": "mkl", "architecture": None}],
            "disagreeing kernels": [{"internal_api": "openblas", "architecture": "Haswell"},
                                    {"internal_api": "openblas", "architecture": "SkylakeX"}],
        }
        golden.openblas_coretype.cache_clear()
        self.addCleanup(golden.openblas_coretype.cache_clear)
        for label, info in cases.items():
            with self.subTest(case=label):
                with patch.object(golden.threadpoolctl, "threadpool_info", return_value=info):
                    self.assertIsNone(golden.openblas_coretype())
                    self.assertFalse(golden.bitwise_environment())
                golden.openblas_coretype.cache_clear()

    def test_the_bitwise_gate_requires_the_recorded_kernels(self):
        numpy_version, scipy_version, machine, coretype = golden.recorded_environment()
        with patch.object(golden.np, "__version__", numpy_version), \
                patch.object(golden.scipy, "__version__", scipy_version), \
                patch.object(golden.platform, "machine", return_value=machine):
            with patch.object(golden, "openblas_coretype", return_value=coretype):
                self.assertTrue(golden.bitwise_environment())
            for other in (None, f"not {coretype}"):
                with patch.object(golden, "openblas_coretype", return_value=other):
                    self.assertFalse(golden.bitwise_environment())


def outcome_replay_failure(name: str, outcomes, *, exact: bool):
    """Why ``outcomes`` of a fresh run of scenario ``name`` fail its outcome
    replay, or ``None``. Exact replay requires the golden's outcomes; otherwise
    a label-only scenario must reach its ``required_outcomes`` and nothing
    outside the (closed) set observed under last-bit noise, and every other
    scenario must reach exactly the golden's."""
    recorded = _golden(name)["outcomes"]
    if exact or name not in LABEL_ONLY_SCENARIOS:
        if sorted(outcomes) != recorded:
            return f"outcomes {sorted(outcomes)} != golden {recorded}"
        return None
    missing = sorted(required_outcomes(name) - set(outcomes))
    if missing:
        return f"intended outcomes {missing} were not reached"
    unobserved = sorted(set(outcomes) - set(SENSITIVITY["scenarios"][name]["observed_outcomes"]))
    if unobserved:
        return f"outcomes {unobserved} were never observed under last-bit noise"
    return None


class AlmGoldenOutcomeReplayTests(unittest.TestCase):
    """Each scenario reaches its recorded outcomes: exactly in the recording
    environment; elsewhere exactly too, except that a label-only scenario may
    reach any outcome observed under last-bit noise. Every fresh run keeps the
    result invariants."""

    def test_every_scenario_replays_its_outcomes(self):
        exact = golden.bitwise_environment()
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                failure = outcome_replay_failure(
                    scenario.name,
                    golden.scenario_outcomes(_fresh_run(scenario.name)),
                    exact=exact,
                )
                self.assertIsNone(failure, f"ALM golden '{scenario.name}': {failure}")

    def test_every_run_keeps_the_result_invariants(self):
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                self.assertEqual(
                    golden.result_invariant_violations(_fresh_run(scenario.name)), []
                )

    def test_observed_outcomes_contain_the_golden_outcomes(self):
        for name, measured in SENSITIVITY["scenarios"].items():
            with self.subTest(scenario=name):
                self.assertLessEqual(
                    set(_golden(name)["outcomes"]), set(measured["observed_outcomes"])
                )
                if name not in LABEL_ONLY_SCENARIOS:
                    self.assertEqual(measured["observed_outcomes"], _golden(name)["outcomes"])

    def test_an_unobserved_label_fails_a_label_only_replay(self):
        for name in LABEL_ONLY_SCENARIOS:
            with self.subTest(scenario=name):
                observed = SENSITIVITY["scenarios"][name]["observed_outcomes"]
                self.assertIsNone(outcome_replay_failure(name, observed, exact=False))
                for extra in ("termination:converged", "action:constraints_inactive_stall"):
                    if extra in observed:
                        continue
                    self.assertIn(
                        extra,
                        outcome_replay_failure(name, [*observed, extra], exact=False),
                    )
                if observed != _golden(name)["outcomes"]:
                    # The recording environment still demands the golden's outcomes.
                    self.assertIsNotNone(outcome_replay_failure(name, observed, exact=True))

    def test_an_invented_action_or_termination_fails_every_label_only_replay(self):
        for name in LABEL_ONLY_SCENARIOS:
            observed = SENSITIVITY["scenarios"][name]["observed_outcomes"]
            for invented in ("action:invented_regression", "termination:invented_regression"):
                with self.subTest(scenario=name, outcome=invented):
                    self.assertIn(
                        invented,
                        outcome_replay_failure(name, [*observed, invented], exact=False) or "",
                    )

    def test_a_missing_intended_outcome_fails_a_label_only_replay(self):
        for name in LABEL_ONLY_SCENARIOS:
            with self.subTest(scenario=name):
                required = required_outcomes(name)
                self.assertTrue(required, "a label-only replay must require something")
                observed = SENSITIVITY["scenarios"][name]["observed_outcomes"]
                for outcome in sorted(required):
                    dropped = [o for o in observed if o != outcome]
                    self.assertIn(outcome, outcome_replay_failure(name, dropped, exact=False))
                self.assertIsNotNone(
                    outcome_replay_failure(
                        name, set(observed) - set(INTENDED_OUTCOMES[name]), exact=False
                    )
                )

    def test_a_broken_invariant_is_reported(self):
        trajectory = _fresh_run("toy_convex")
        broken = golden.RecordedTrajectory(copy.deepcopy(dict(trajectory)), trajectory.settings)
        broken["result"]["fields"]["restored_best_feasible"] = True
        broken["result"]["fields"]["multipliers"]["data"][0] = golden.float_token(-1.0)
        violations = golden.result_invariant_violations(broken)
        self.assertEqual(len(violations), 2, violations)


class AlmGoldenNumericReplayTests(unittest.TestCase):
    """Any numpy and SciPy: each scenario's boundary values stay within its
    ``BOUNDARY_TOLERANCES`` of its golden; label-only scenarios are skipped."""

    def test_every_scenario_replays_its_boundary_values(self):
        for name, tolerances in BOUNDARY_TOLERANCES.items():
            with self.subTest(scenario=name):
                structural, deviations = golden.boundary_deviations(
                    golden.scenario_boundary_values(_golden(name)["trajectory"]),
                    golden.scenario_boundary_values(_fresh_run(name)),
                )
                self.assertIsNone(structural, f"ALM golden '{name}' took another path")
                for quantity, tolerance in tolerances.items():
                    self.assertLessEqual(
                        deviations[quantity],
                        tolerance,
                        f"ALM golden '{name}' {quantity} drifted",
                    )

    def test_every_scenario_is_numeric_or_label_only_with_a_reason(self):
        self.assertEqual(
            set(BOUNDARY_TOLERANCES) | set(LABEL_ONLY_SCENARIOS), set(golden.SCENARIOS_BY_NAME)
        )
        self.assertFalse(set(BOUNDARY_TOLERANCES) & set(LABEL_ONLY_SCENARIOS))
        for reason in LABEL_ONLY_SCENARIOS.values():
            self.assertTrue(reason.strip())
        for tolerances in BOUNDARY_TOLERANCES.values():
            self.assertEqual(tuple(tolerances), golden.BOUNDARY_QUANTITIES)

    def test_every_scenario_records_boundary_values(self):
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                values = golden.scenario_boundary_values(_golden(scenario.name)["trajectory"])
                self.assertTrue(all(values[quantity] for quantity in golden.BOUNDARY_QUANTITIES))

    def test_a_ceiling_sized_regression_is_caught_in_every_quantity(self):
        # A relative drift of NUMERIC_REPLAY_CEILING in the last recorded value
        # of any quantity of any numeric scenario exceeds its tolerance.
        for name, tolerances in BOUNDARY_TOLERANCES.items():
            expected = golden.scenario_boundary_values(_golden(name)["trajectory"])
            for quantity in golden.BOUNDARY_QUANTITIES:
                with self.subTest(scenario=name, quantity=quantity):
                    index = max(
                        i for i, (_, floats) in enumerate(expected[quantity]) if floats
                    )
                    location, floats = expected[quantity][index]
                    shifted = (floats[0] + NUMERIC_REPLAY_CEILING * max(1.0, abs(floats[0])),)
                    drifted = dict(expected)
                    drifted[quantity] = (
                        expected[quantity][:index]
                        + [(location, shifted + floats[1:])]
                        + expected[quantity][index + 1:]
                    )
                    structural, deviations = golden.boundary_deviations(expected, drifted)
                    self.assertIsNone(structural)
                    self.assertGreater(deviations[quantity], tolerances[quantity])


@unittest.skipUnless(
    golden.bitwise_environment(),
    f"the goldens were recorded with numpy {RECORDED_ENVIRONMENT[0]} and SciPy "
    f"{RECORDED_ENVIRONMENT[1]} on {RECORDED_ENVIRONMENT[2]} with OpenBLAS's "
    f"{RECORDED_ENVIRONMENT[3]} kernels (this process: {golden.current_environment()}); "
    "L-BFGS-B iterates are bitwise only there",
)
class AlmGoldenReplayTests(unittest.TestCase):
    """One test per scenario; the scenario descriptions say what each covers."""

    def assert_replays_bitwise(self, name: str) -> None:
        golden_trajectory = _golden(name)["trajectory"]
        difference = golden.first_difference(golden_trajectory, _fresh_run(name))
        self.assertIsNone(
            difference,
            f"ALM golden '{name}' drifted:\n{difference}\n"
            "Results, histories, callback order and checkpoints must stay "
            "bitwise. Regenerate only for an intended behavior change: "
            f"{MANIFEST['regenerate']}",
        )

    def test_toy_convex(self):
        self.assert_replays_bitwise("toy_convex")

    def test_penalty_ramp_to_cap(self):
        self.assert_replays_bitwise("penalty_ramp_to_cap")

    def test_resume_penalty_ramp_mid_run(self):
        self.assert_replays_bitwise("resume_penalty_ramp_mid_run")

    def test_hybrid_mismatch_repair(self):
        self.assert_replays_bitwise("hybrid_mismatch_repair")

    def test_hybrid_mismatch_penalty_increase(self):
        self.assert_replays_bitwise("hybrid_mismatch_penalty_increase")

    def test_hybrid_zero_shift_split(self):
        self.assert_replays_bitwise("hybrid_zero_shift_split")

    def test_constraints_inactive_stall(self):
        self.assert_replays_bitwise("constraints_inactive_stall")

    def test_plateau_restore_best_feasible(self):
        self.assert_replays_bitwise("plateau_restore_best_feasible")

    def test_resume_plateau_best_feasible(self):
        self.assert_replays_bitwise("resume_plateau_best_feasible")

    def test_trust_radius_retries(self):
        self.assert_replays_bitwise("trust_radius_retries")

    def test_frozen_warm_start_restore(self):
        self.assert_replays_bitwise("frozen_warm_start_restore")

    def test_cached_physics_smoothing(self):
        self.assert_replays_bitwise("cached_physics_smoothing")

    def test_multiplier_cap_process_budget(self):
        self.assert_replays_bitwise("multiplier_cap_process_budget")

    def test_inner_iteration_budget(self):
        self.assert_replays_bitwise("inner_iteration_budget")

    def test_dual_update_penalty_cap(self):
        self.assert_replays_bitwise("dual_update_penalty_cap")

    def test_preinner_converged(self):
        self.assert_replays_bitwise("preinner_converged")

    def test_zero_inner_budget(self):
        self.assert_replays_bitwise("zero_inner_budget")

    def test_every_scenario_has_a_replay_test(self):
        tested = {
            name[len("test_"):]
            for name in dir(self)
            if name.startswith("test_") and name != "test_every_scenario_has_a_replay_test"
        }
        self.assertEqual(tested, set(golden.SCENARIOS_BY_NAME))


class AlmGoldenResumeContractTests(unittest.TestCase):
    """A resumed run ends where the uninterrupted run ends.

    Exceptions, each a documented resume property rather than drift: the inner
    ``maxiter`` budget is per process, so resumed history entries report the
    resumed process's budget; a best-feasible incumbent restored from a
    checkpoint carries no constraint Jacobians and no inner ``OptimizeResult``.
    """

    PER_PROCESS_HISTORY_KEYS = frozenset(("inner_maxiter", "inner_maxfun"))
    CHECKPOINT_RESTORED_RESULT_FIELDS = frozenset(("evaluation", "inner_result"))

    def assert_resume_matches(
        self, resumed_name: str, uninterrupted_name: str, *, excluded_fields=frozenset()
    ) -> None:
        resumed = _golden(resumed_name)["trajectory"]["resumed"]
        full = _golden(uninterrupted_name)["trajectory"]
        for field in full["result"]["fields"]:
            if field in excluded_fields:
                continue
            self.assertIsNone(
                golden.first_difference(
                    full["result"]["fields"][field], resumed["result"]["fields"][field]
                ),
                f"resumed {field} differs from the uninterrupted run",
            )
        tail = full["history"][len(full["history"]) - len(resumed["history"]):]
        for index, (expected, actual) in enumerate(zip(tail, resumed["history"])):
            self.assertEqual(list(expected), list(actual))
            for key in expected:
                if key in self.PER_PROCESS_HISTORY_KEYS:
                    continue
                self.assertIsNone(
                    golden.first_difference(expected[key], actual[key], key),
                    f"resumed history entry {index} differs at {key}",
                )
        checkpoint_tail = full["checkpoints"][
            len(full["checkpoints"]) - len(resumed["checkpoints"]):
        ]
        self.assertIsNone(
            golden.first_difference(checkpoint_tail, resumed["checkpoints"]),
            "resumed checkpoints differ from the uninterrupted run's",
        )

    def test_mid_run_resume_matches_uninterrupted_penalty_ramp(self):
        self.assert_resume_matches("resume_penalty_ramp_mid_run", "penalty_ramp_to_cap")

    def test_best_feasible_resume_matches_uninterrupted_plateau(self):
        self.assert_resume_matches(
            "resume_plateau_best_feasible",
            "plateau_restore_best_feasible",
            excluded_fields=self.CHECKPOINT_RESTORED_RESULT_FIELDS,
        )


if __name__ == "__main__":
    unittest.main()
