"""``ALMHistoryRecorder``: the ALM history is built from outer-step events only.

The golden scenarios attach a recorder to the library ``minimize_alm``. Here
every library call is intercepted: its events and its result are captured, and
a fresh recorder that sees nothing but those events rebuilds the history the run
recorded, bit for bit. The library loop owns no history code.
"""

import ast
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np

import simsopt_alm as alm
from simsopt_alm import control as alm_control
from simsopt_alm.history import ALMHistoryRecorder

GOLDEN_DIR = Path(__file__).resolve().parent / "alm_golden"
if str(GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GOLDEN_DIR))
import alm_golden_scenarios as golden  # noqa: E402

PACKAGE_ROOT = Path(alm.__file__).resolve().parent


@contextmanager
def captured_library_runs():
    """Each library ``minimize_alm`` call in this block, as
    ``(settings, events, lean result)``."""
    runs: list = []
    library_minimize_alm = alm_control.minimize_alm

    def observed_minimize_alm(*args, on_outer_step=None, **kwargs):
        settings = args[3] if len(args) > 3 else kwargs["settings"]
        events: list = []
        runs.append((settings, events))

        def observe(event):
            events.append(event)
            if on_outer_step is not None:
                on_outer_step(event)

        result = library_minimize_alm(*args, on_outer_step=observe, **kwargs)
        runs[-1] = (settings, events, result)
        return result

    with patch.object(alm_control, "minimize_alm", observed_minimize_alm):
        yield runs


def _runs(trajectory: dict) -> List[dict]:
    if "resumed" in trajectory:
        return [trajectory["interrupted"], trajectory["resumed"]]
    return [trajectory]


class AlmHistoryRecorderTests(unittest.TestCase):
    def test_recorder_fed_only_events_rebuilds_every_recorded_history(self):
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                with captured_library_runs() as runs:
                    trajectory = scenario.run()
                recorded_runs = _runs(trajectory)
                self.assertEqual(len(runs), len(recorded_runs))
                for run, recorded in zip(runs, recorded_runs):
                    if recorded["result"] is None:
                        # Interrupted at a checkpoint boundary: no result.
                        self.assertEqual(len(run), 2)
                    else:
                        self.assertIs(type(run[2]), alm.ALMResult)
                    if "history" not in recorded:
                        continue
                    settings, events = run[0], run[1]
                    history = ALMHistoryRecorder(settings.history_max_entries)
                    for event in events:
                        history.record(event)
                    self.assertIsNone(
                        golden.first_difference(
                            recorded["history"], golden.encode(history.history())
                        ),
                        f"events alone do not rebuild '{scenario.name}' history",
                    )

    def test_history_callback_sees_each_entry_as_it_is_recorded(self):
        calls = []

        def history_callback(history, latest_entry, multipliers, penalty):
            calls.append((len(history), latest_entry["action"], float(penalty)))

        with captured_library_runs() as runs:
            golden.SCENARIOS_BY_NAME["penalty_ramp_to_cap"].run()
        ((settings, events, library_result),) = runs
        recorder = ALMHistoryRecorder(
            settings.history_max_entries, history_callback=history_callback
        )
        for event in events:
            recorder.record(event)
        history = recorder.history()
        self.assertEqual(
            calls,
            [
                (index + 1, entry["action"], float(event.after.penalty))
                for index, (entry, event) in enumerate(zip(history, events))
            ],
        )


class AlmHistoryRecorderContractTests(unittest.TestCase):
    def test_max_entries_must_be_positive_when_provided(self):
        for max_entries in (0, -1):
            with self.subTest(max_entries=max_entries):
                with self.assertRaisesRegex(
                    ValueError,
                    r"ALMHistoryRecorder\.max_entries must be positive when provided",
                ):
                    ALMHistoryRecorder(max_entries)

    def test_returned_history_is_not_changed_by_later_records(self):
        with captured_library_runs() as runs:
            golden.SCENARIOS_BY_NAME["penalty_ramp_to_cap"].run()
        ((settings, events, _library_result),) = runs
        recorder = ALMHistoryRecorder(settings.history_max_entries)
        recorder.record(events[0])
        early = recorder.history()
        early_history = golden.encode(early)
        for event in events[1:]:
            recorder.record(event)
        later = recorder.history()
        later[0]["action"] = "overwritten_in_a_later_history"
        self.assertEqual(len(early), 1)
        self.assertIsNone(golden.first_difference(early_history, golden.encode(early)))


def _penalty_ramp_events():
    with captured_library_runs() as runs:
        golden.SCENARIOS_BY_NAME["penalty_ramp_to_cap"].run()
    ((settings, events, _library_result),) = runs
    return settings, events


class AlmHistoryOwnershipTests(unittest.TestCase):
    """What ``history()`` and ``history_callback`` hand out is the caller's,
    nested lists and dicts included: writing it never reaches the recorder."""

    def test_history_rows_are_owned_all_the_way_down(self):
        settings, events = _penalty_ramp_events()
        recorder = ALMHistoryRecorder(settings.history_max_entries)
        for event in events:
            recorder.record(event)
        expected = golden.encode(recorder.history())

        handed_out = recorder.history()
        handed_out[0]["constraint_names"][0] = "changed"
        handed_out[-1]["penalty_values"].append(-1.0)
        handed_out[-1]["block_max_normalized_violation"] = {"edited": 1.0}

        self.assertIsNone(golden.first_difference(expected, golden.encode(recorder.history())))

    def test_callback_rows_are_owned_all_the_way_down(self):
        settings, events = _penalty_ramp_events()
        self.assertGreaterEqual(len(events), 3, "the fixture needs earlier rows")

        def vandal(history, latest_entry, multipliers, penalty):
            # An earlier row, the latest row and the latest-entry argument.
            history[0]["constraint_names"][0] = "changed"
            history[-1]["penalty_values"].append(-1.0)
            latest_entry["multipliers"].append(-1.0)

        reference = ALMHistoryRecorder(settings.history_max_entries)
        recorder = ALMHistoryRecorder(settings.history_max_entries, history_callback=vandal)
        seen = []
        observer = ALMHistoryRecorder(
            settings.history_max_entries,
            history_callback=lambda history, *_: seen.append(golden.encode(history)),
        )
        for event in events:
            reference.record(event)
            recorder.record(event)
            observer.record(event)

        expected = golden.encode(reference.history())
        self.assertIsNone(golden.first_difference(expected, golden.encode(recorder.history())))
        # A callback that only reads sees the same rows as history() at that point.
        self.assertIsNone(golden.first_difference(expected, seen[-1]))

    def test_from_settings_keeps_the_settings_retention(self):
        _settings, events = _penalty_ramp_events()
        for max_entries in (2, None):
            with self.subTest(max_entries=max_entries):
                recorder = ALMHistoryRecorder.from_settings(
                    alm.ALMSettings(history_max_entries=max_entries)
                )
                for event in events:
                    recorder.record(event)
                kept = len(events) if max_entries is None else max_entries
                self.assertEqual(len(recorder.history()), kept)
                self.assertEqual(recorder.truncated_count, len(events) - kept)

    def test_from_settings_passes_the_callback(self):
        settings, events = _penalty_ramp_events()
        calls = []
        recorder = ALMHistoryRecorder.from_settings(
            settings, lambda history, *_: calls.append(len(history))
        )
        recorder.record(events[0])
        self.assertEqual(calls, [1])


class AlmLibraryWithoutRecorderTests(unittest.TestCase):
    def test_library_runs_without_a_recorder(self):
        def evaluate_problem(x, multipliers, penalty):
            return alm.augmented_inequality_objective(
                base_value=float(x @ x),
                base_grad=2.0 * x,
                constraint_values=np.array([1.0 - x[0]]),
                constraint_grads=[np.array([-1.0, 0.0])],
                multipliers=multipliers,
                penalty=penalty,
            )

        result = alm.minimize_alm(
            np.array([3.0, 2.0]),
            ["x0_at_least_one"],
            evaluate_problem,
            alm.ALMSettings(),
            {"maxiter": 200},
        )

        self.assertTrue(result.success, result.message)
        self.assertFalse(hasattr(result, "history"))
        self.assertFalse(hasattr(result, "alm_summary"))

    def test_package_import_does_not_load_the_history_plugin(self):
        probe = (
            "import sys\n"
            "import simsopt_alm\n"
            "sys.exit(1 if 'simsopt_alm.history' in sys.modules else 0)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class AlmLoopOwnsNoHistoryTests(unittest.TestCase):
    HISTORY_SYMBOLS = frozenset(
        (
            "_build_alm_history_entry",
            "_build_skipped_inner_history_entry",
            "_refresh_alm_history_for_penalty_update",
            "_constraint_history_diagnostics",
            "_constraint_history_diagnostics_source",
            "_constraint_history_diagnostics_from_source",
            "_alm_summary",
            "_alm_summary_diagnostics",
            "_history_entry_snapshot",
        )
    )

    def test_control_defines_no_history_builder(self):
        tree = ast.parse((PACKAGE_ROOT / "control.py").read_text(encoding="utf-8"))
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        self.assertEqual(sorted(defined & self.HISTORY_SYMBOLS), [])

    def test_loop_state_and_entry_point_carry_no_history(self):
        self.assertNotIn(
            "history", alm_control.ALMRunState.__dataclass_fields__
        )
        self.assertNotIn(
            "history_callback",
            alm_control.minimize_alm.__code__.co_varnames,
        )


if __name__ == "__main__":
    unittest.main()
