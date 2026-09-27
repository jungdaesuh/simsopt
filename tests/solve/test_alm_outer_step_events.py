"""``minimize_alm(on_outer_step=...)``: one frozen event per outer-loop decision.

Every golden scenario is replayed with an ``on_outer_step`` observer. Each
continuation step makes exactly one decision, so the events must line up one to
one with the decisions the golden recorded and with its history entries (same
outer and continuation iteration, same action), and every event is a read-only
snapshot.
"""

import dataclasses
import sys
import unittest
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

import simsopt.solve.alm as alm
from simsopt.solve.alm import control as alm_control

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "test_files" / "alm_golden"
if str(GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GOLDEN_DIR))
import alm_golden_scenarios as golden  # noqa: E402


@contextmanager
def captured_outer_step_events():
    """Observe every ``on_outer_step`` event of the library runs the golden
    scenarios start in this block (their own recorders still see them)."""
    events: list = []
    library_minimize_alm = alm_control.minimize_alm

    def minimize_alm_with_observer(*args, on_outer_step=None, **kwargs):
        def observe(event):
            events.append(event)
            if on_outer_step is not None:
                on_outer_step(event)

        return library_minimize_alm(*args, on_outer_step=observe, **kwargs)

    with patch.object(alm_control, "minimize_alm", minimize_alm_with_observer):
        yield events


def _mutable_parts(value, path: str, seen: set) -> list[str]:
    """Paths of everything an event reaches that could be written: writable
    or object-dtype arrays, dicts, lists and sets. The caller's accepted
    states (``incumbent_state``) are the caller's own objects."""
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, np.ndarray):
        found = []
        if value.flags.writeable:
            found.append(f"{path}: writable array")
        if value.dtype == object:
            found.append(f"{path}: object-dtype array")
        return found
    if isinstance(value, (dict, list, set)):
        return [f"{path}: {type(value).__name__}"]
    if isinstance(value, Mapping):
        return [
            part
            for key, item in value.items()
            for part in _mutable_parts(item, f"{path}[{key!r}]", seen)
        ]
    if isinstance(value, tuple):
        return [
            part
            for index, item in enumerate(value)
            for part in _mutable_parts(item, f"{path}[{index}]", seen)
        ]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            part
            for field in dataclasses.fields(value)
            if field.name != "incumbent_state"
            for part in _mutable_parts(
                getattr(value, field.name), f"{path}.{field.name}", seen
            )
        ]
    return []


def _golden_runs(trajectory: dict) -> list[dict]:
    """The library runs of one golden trajectory, in execution order."""
    if "resumed" in trajectory:
        return [trajectory["interrupted"], trajectory["resumed"]]
    return [trajectory]


def _decision_key(outer_iteration, continuation_iteration, action) -> tuple:
    return (int(outer_iteration), int(continuation_iteration), str(action))


class AlmOuterStepEventTests(unittest.TestCase):
    def test_event_types_are_public_and_frozen(self):
        for name in (
            "ALMOuterStepEvent",
            "ALMIterateMeasurement",
            "ALMInnerSolveOutcome",
            "ALMDualUpdate",
            "ALMLoopState",
        ):
            with self.subTest(name=name):
                self.assertIn(name, alm.__all__)
                event_type = getattr(alm, name)
                self.assertTrue(dataclasses.is_dataclass(event_type))
                self.assertTrue(event_type.__dataclass_params__.frozen)

    def test_every_event_is_a_read_only_snapshot(self):
        # Nothing an observer reaches can be written: every array is a
        # read-only copy and no dict, list or set remains.
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                with captured_outer_step_events() as events:
                    scenario.run()
                self.assertGreater(len(events), 0)
                for index, event in enumerate(events):
                    self.assertEqual(
                        _mutable_parts(event, f"event[{index}]", set()), []
                    )

    def test_events_line_up_with_every_golden_decision(self):
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                runs = _golden_runs(golden.load_golden(scenario.name)["trajectory"])
                with captured_outer_step_events() as events:
                    scenario.run()
                self.assertGreater(len(events), 0)
                for event in events:
                    self.assertIsInstance(event, alm.ALMOuterStepEvent)
                    with self.assertRaises(dataclasses.FrozenInstanceError):
                        event.action = "mutated"
                event_keys = [
                    _decision_key(
                        event.outer_iteration, event.continuation_iteration, event.action
                    )
                    for event in events
                ]

                records = [
                    record
                    for run in runs
                    for record in run["events"]
                    if record["event"] == "on_outer_step"
                ]
                if not records:
                    continue
                self.assertEqual(
                    event_keys,
                    [
                        _decision_key(
                            record["outer_iteration"],
                            record["continuation_iteration"],
                            record["action"],
                        )
                        for record in records
                    ],
                )
                for run in runs:
                    # One history entry per event (the latest ones when
                    # history_max_entries truncates), for the same decision.
                    run_keys = [
                        _decision_key(
                            record["outer_iteration"],
                            record["continuation_iteration"],
                            record["action"],
                        )
                        for record in run["events"]
                        if record["event"] == "on_outer_step"
                    ]
                    history_keys = [
                        _decision_key(
                            entry["outer_iteration"],
                            entry["continuation_iteration"],
                            entry["action"],
                        )
                        for entry in run["history"]
                    ]
                    self.assertEqual(
                        run_keys[len(run_keys) - len(history_keys):], history_keys
                    )
                if golden.bitwise_environment():
                    self.assertEqual(
                        [golden.encoded_digest(event) for event in events],
                        [record["event_sha256"] for record in records],
                    )

    def test_skipped_inner_solve_measures_the_start_iterate(self):
        with captured_outer_step_events() as events:
            golden.SCENARIOS_BY_NAME["preinner_converged"].run()
        (event,) = events
        self.assertEqual(event.action, "converged")
        self.assertIsNone(event.inner)
        self.assertIs(event.measured, event.start)
        np.testing.assert_array_equal(event.after.x, event.start_x)
        # The step's own incumbent, not the one before it.
        incumbent = event.after.best_feasible
        self.assertIsNotNone(incumbent)
        np.testing.assert_array_equal(incumbent.x, event.start_x)
        self.assertEqual(
            incumbent.evaluation["total"], event.start.evaluation["total"]
        )

    def test_penalty_raise_carries_the_new_penalty_measurement(self):
        with captured_outer_step_events() as events:
            golden.SCENARIOS_BY_NAME["penalty_ramp_to_cap"].run()
        raises = [event for event in events if event.penalty_update is not None]
        self.assertGreater(len(raises), 0)
        for event in raises:
            self.assertIsNotNone(event.inner)
            self.assertEqual(event.penalty_update.penalty, event.after.penalty)
            self.assertGreaterEqual(event.penalty_update.penalty, event.measured.penalty)
            np.testing.assert_array_equal(
                event.penalty_update.multipliers, event.after.multipliers
            )


if __name__ == "__main__":
    unittest.main()
