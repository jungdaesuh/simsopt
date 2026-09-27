"""Golden ALM trajectories: ``minimize_alm`` must replay bit for bit.

Each scenario in ``tests/test_files/alm_golden/alm_golden_scenarios.py`` drives
the library solver with a deterministic synthetic evaluator. Its golden holds
every ``ALMResult`` field, the history entries, every call the solver makes into
caller code in order, and every outer-boundary checkpoint.

These are characterization tests: they fail on any drift in values, order, keys
or types. L-BFGS-B iterates are bitwise only for the numpy, SciPy and machine
the goldens were recorded with, so the bitwise replay runs only there. Under any
supported numpy and SciPy each scenario replays its coarse outcomes (actions,
termination and restore reasons, flags) and its numeric boundary values (x,
objective, max violation, multipliers, penalty at every outer step, outer
boundary and result) within measured tolerances. Regenerate only for a
reviewed, intended behavior change (command in ``manifest.json``); a change that
leaves every fixture byte-identical refreshes the manifest's provenance with
``--provenance-only``.
"""

import functools
import hashlib
import sys
import unittest
from pathlib import Path
from types import MappingProxyType

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "test_files" / "alm_golden"
if str(GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GOLDEN_DIR))
import alm_golden_scenarios as golden  # noqa: E402
from generate_alm_golden import INTENDED_OUTCOMES, alm_source_blob_ids  # noqa: E402

MANIFEST = golden.load_manifest()
RECORDED_ENVIRONMENT = golden.recorded_environment()
_golden = golden.load_golden

# Largest boundary-value deviation, |value - golden| / max(1, |golden|), that
# the numeric replay accepts. Measured on x86_64 across Python 3.11 (numpy
# 2.4.6, SciPy 1.17.1; the recording environment, deviation 0), Python 3.9
# (numpy 2.0.2, SciPy 1.13.1) and Python 3.8 (numpy 1.24.4, SciPy 1.10.1),
# single-threaded, over every scenario; the two non-recording environments
# gave identical maxima. Tolerance = 10 x max(observed, machine epsilon); the
# epsilon floor keeps an exactly reproduced quantity (penalty) from demanding
# bit equality.
#
#   quantity        max observed   (scenario)                      tolerance
#   x               7.8e-16        toy_convex                      7.8e-15
#   objective       1.4e-15        multiplier_cap_process_budget   1.4e-14
#   max_violation   1.3e-15        multiplier_cap_process_budget   1.3e-14
#   multipliers     4.4e-15        toy_convex, inner_iteration_... 4.4e-14
#   penalty         0              (all)                           2.2e-15
BOUNDARY_TOLERANCES = MappingProxyType({
    "x": 7.8e-15,
    "objective": 1.4e-14,
    "max_violation": 1.3e-14,
    "multipliers": 4.4e-14,
    "penalty": 2.2e-15,
})
# Scenarios whose trajectory legitimately branches across supported numpy and
# SciPy versions, with the reason; they replay their outcomes only. None
# branched in the measurement above.
OUTCOME_ONLY_SCENARIOS = MappingProxyType({})


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

    def test_the_manifest_names_the_sources_that_produced_the_goldens(self):
        self.assertEqual(
            MANIFEST["source_blob_ids"],
            alm_source_blob_ids(),
            "the ALM sources changed since the goldens were recorded; if every "
            "fixture still replays byte-identically, refresh the provenance: "
            "OMP_NUM_THREADS=1 python tests/test_files/alm_golden/"
            "generate_alm_golden.py --provenance-only",
        )

    def test_every_golden_covers_its_intended_outcomes(self):
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                missed = set(INTENDED_OUTCOMES[scenario.name]) - set(
                    _golden(scenario.name)["outcomes"]
                )
                self.assertEqual(sorted(missed), [])


class AlmGoldenOutcomeReplayTests(unittest.TestCase):
    """Any numpy and SciPy: each scenario reaches its recorded outcomes."""

    def test_every_scenario_replays_its_outcomes(self):
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                outcomes = golden.scenario_outcomes(_fresh_run(scenario.name))
                self.assertEqual(
                    sorted(outcomes),
                    _golden(scenario.name)["outcomes"],
                    f"ALM golden '{scenario.name}' reached other outcomes",
                )


class AlmGoldenNumericReplayTests(unittest.TestCase):
    """Any numpy and SciPy: each scenario's boundary values stay within
    ``BOUNDARY_TOLERANCES`` of its golden."""

    def test_every_scenario_replays_its_boundary_values(self):
        for scenario in golden.SCENARIOS:
            if scenario.name in OUTCOME_ONLY_SCENARIOS:
                continue
            with self.subTest(scenario=scenario.name):
                structural, deviations = golden.boundary_deviations(
                    golden.scenario_boundary_values(_golden(scenario.name)["trajectory"]),
                    golden.scenario_boundary_values(_fresh_run(scenario.name)),
                )
                self.assertIsNone(
                    structural,
                    f"ALM golden '{scenario.name}' took another path; name it in "
                    "OUTCOME_ONLY_SCENARIOS with the reason only if that is legitimate",
                )
                for quantity, tolerance in BOUNDARY_TOLERANCES.items():
                    self.assertLessEqual(
                        deviations[quantity],
                        tolerance,
                        f"ALM golden '{scenario.name}' {quantity} drifted",
                    )

    def test_the_tolerances_cover_every_quantity(self):
        self.assertEqual(tuple(BOUNDARY_TOLERANCES), golden.BOUNDARY_QUANTITIES)

    def test_outcome_only_scenarios_exist_and_give_a_reason(self):
        for name, reason in OUTCOME_ONLY_SCENARIOS.items():
            self.assertIn(name, golden.SCENARIOS_BY_NAME)
            self.assertTrue(reason.strip())

    def test_every_scenario_records_boundary_values(self):
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                values = golden.scenario_boundary_values(_golden(scenario.name)["trajectory"])
                self.assertTrue(all(values[quantity] for quantity in golden.BOUNDARY_QUANTITIES))

    def test_a_drift_beyond_tolerance_is_detected(self):
        expected = golden.scenario_boundary_values(_golden("penalty_ramp_to_cap")["trajectory"])
        location, floats = expected["x"][-1]
        drifted = dict(expected)
        drifted["x"] = expected["x"][:-1] + [(location, tuple(value + 1e-6 for value in floats))]
        structural, deviations = golden.boundary_deviations(expected, drifted)
        self.assertIsNone(structural)
        self.assertGreater(deviations["x"], BOUNDARY_TOLERANCES["x"])
        self.assertEqual(deviations["penalty"], 0.0)


@unittest.skipUnless(
    golden.bitwise_environment(),
    f"the goldens were recorded with numpy {RECORDED_ENVIRONMENT[0]} and SciPy "
    f"{RECORDED_ENVIRONMENT[1]} on {RECORDED_ENVIRONMENT[2]}; L-BFGS-B iterates "
    "are bitwise only there",
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

    def test_hybrid_mismatch_stall(self):
        self.assert_replays_bitwise("hybrid_mismatch_stall")

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
