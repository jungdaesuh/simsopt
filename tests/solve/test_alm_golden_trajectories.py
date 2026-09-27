"""Golden ALM trajectories: ``minimize_alm`` must replay bit for bit.

Each scenario in ``tests/test_files/alm_golden/alm_golden_scenarios.py`` drives
the library solver with a deterministic synthetic evaluator. Its golden holds
every ``ALMResult`` field, the history entries, every call the solver makes into
caller code in order, and every outer-boundary checkpoint.

These are characterization tests: they fail on any drift in values, order, keys
or types. L-BFGS-B iterates are bitwise only for the numpy and SciPy the goldens
were recorded with, so the bitwise replay runs only there; the coarse outcomes
(actions, termination and restore reasons, flags) replay everywhere. Regenerate
only for a reviewed, intended behavior change (command in ``manifest.json``).
"""

import hashlib
import sys
import unittest
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "test_files" / "alm_golden"
if str(GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GOLDEN_DIR))
import alm_golden_scenarios as golden  # noqa: E402
from generate_alm_golden import INTENDED_OUTCOMES  # noqa: E402

MANIFEST = golden.load_manifest()
RECORDED_ENVIRONMENT = golden.recorded_environment()
_golden = golden.load_golden


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
                outcomes = golden.scenario_outcomes(scenario.run())
                self.assertEqual(
                    sorted(outcomes),
                    _golden(scenario.name)["outcomes"],
                    f"ALM golden '{scenario.name}' reached other outcomes",
                )


@unittest.skipUnless(
    golden.bitwise_environment(),
    f"the goldens were recorded with numpy {RECORDED_ENVIRONMENT[0]} and SciPy "
    f"{RECORDED_ENVIRONMENT[1]}; L-BFGS-B iterates are bitwise only there",
)
class AlmGoldenReplayTests(unittest.TestCase):
    """One test per scenario; the scenario descriptions say what each covers."""

    def assert_replays_bitwise(self, name: str) -> None:
        golden_trajectory = _golden(name)["trajectory"]
        trajectory = golden.SCENARIOS_BY_NAME[name].run()
        difference = golden.first_difference(golden_trajectory, trajectory)
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
