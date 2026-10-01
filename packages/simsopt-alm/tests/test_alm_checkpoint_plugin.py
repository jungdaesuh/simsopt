"""``simsopt_alm.checkpoint``: checkpoint/resume is a plugin of the loop.

The library ``minimize_alm`` publishes an :class:`ALMOuterBoundary` after each
completed outer iteration and once when it returns (``on_outer_boundary``), and
accepts one back as ``resume_from``. The checkpoint plugin turns boundaries into
``ALMTransitionSnapshot`` checkpoints (schema ``alm_transition_checkpoint_v1``)
and snapshots back into resume boundaries. Here every library call of the
golden scenarios is intercepted: its boundaries alone rebuild every checkpoint
the plugin wrote, bit for bit, and a snapshot round-trips through its boundary.
Both sides of that rebuild come from ``transition_snapshot``, so the rebuilt
checkpoints are also held to the stored golden: bit for bit in the recording
environment, field path by field path elsewhere for a scenario whose path is
closed under last-bit noise. The golden resume scenarios resume through the
plugin.
"""

import ast
import dataclasses
import inspect
import pickle
import subprocess
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import List, Optional
from unittest.mock import patch

import numpy as np

import simsopt_alm as alm
from simsopt_alm import checkpoint as alm_checkpoint
from simsopt_alm import control as alm_control

GOLDEN_DIR = Path(__file__).resolve().parent / "alm_golden"
if str(GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GOLDEN_DIR))
import alm_golden_scenarios as golden  # noqa: E402
from test_alm_history_recorder import CLOSED_FORM_TOLERANCE, penalty_ramp_closed_form  # noqa: E402

PACKAGE_ROOT = Path(alm.__file__).resolve().parent
# Scenarios whose outcomes or path change under last-bit noise
# (sensitivity.json): outside the recording environment their checkpoints may
# hold other states.
LABEL_ONLY_SCENARIOS = frozenset(
    name
    for name, measured in golden.load_sensitivity()["scenarios"].items()
    if measured["label_only"] is not None
)


def _runs(trajectory: dict) -> List[dict]:
    if "resumed" in trajectory:
        return [trajectory["interrupted"], trajectory["resumed"]]
    return [trajectory]


def _field_paths(encoded: object, path: str = "$") -> List[str]:
    """Every dict key path of an encoded value, in order (list items as
    ``[i]``), a None marked ``=None``: which fields are present and which are
    None or a value."""
    if isinstance(encoded, dict):
        return [
            found
            for key, item in encoded.items()
            for found in [f"{path}.{key}", *_field_paths(item, f"{path}.{key}")]
        ]
    if isinstance(encoded, list):
        return [
            found for index, item in enumerate(encoded) for found in _field_paths(item, f"{path}[{index}]")
        ]
    return [f"{path}=None"] if encoded is None else []


def frozen_checkpoints_difference(name: str, run_index: int, checkpoints: list) -> Optional[str]:
    """How ``checkpoints`` (encoded) differ from run ``run_index`` of scenario
    ``name``'s stored golden: bit for bit in the recording environment;
    elsewhere their field paths, unless the scenario is label-only (None:
    nothing to compare)."""
    frozen = _runs(golden.load_golden(name)["trajectory"])[run_index]["checkpoints"]
    if golden.bitwise_environment():
        return golden.first_difference(frozen, checkpoints)
    if name in LABEL_ONLY_SCENARIOS:
        return None
    return golden.first_difference(_field_paths(frozen), _field_paths(checkpoints))


@contextmanager
def captured_library_boundaries():
    """Each library ``minimize_alm`` call in this block, as
    ``(inner options the loop ran with, published boundaries)``."""
    runs: list = []
    library_minimize_alm = alm_control.minimize_alm

    def observed_minimize_alm(*args, on_outer_boundary=None, **kwargs):
        inner_options = args[4] if len(args) > 4 else kwargs["inner_options"]
        boundaries: list = []
        runs.append((inner_options, boundaries))
        observe = None
        if on_outer_boundary is not None:

            def observe(boundary):
                boundaries.append(boundary)
                on_outer_boundary(boundary)

        return library_minimize_alm(*args, on_outer_boundary=observe, **kwargs)

    with patch.object(alm_control, "minimize_alm", observed_minimize_alm):
        yield runs


class AlmCheckpointProjectionValueTests(unittest.TestCase):
    def test_rebuilt_checkpoints_hold_the_closed_form_penalty_ramp(self):
        with captured_library_boundaries() as runs:
            golden.SCENARIOS_BY_NAME["penalty_ramp_to_cap"].run()
        ((inner_options, boundaries),) = runs
        snapshots = [alm_checkpoint.transition_snapshot(boundary, inner_options) for boundary in boundaries]
        expected = penalty_ramp_closed_form()
        self.assertEqual(
            [snapshot.completed_outer_iterations for snapshot in snapshots], expected["outer_iteration"]
        )
        self.assertEqual([snapshot.penalty for snapshot in snapshots], expected["penalty"])
        # Infeasible until the last boundary, which holds its iterate as the
        # best-feasible incumbent.
        self.assertEqual(
            [snapshot.best_feasible is None for snapshot in snapshots], [True] * 7 + [False]
        )
        for index, snapshot in enumerate(snapshots):
            with self.subTest(boundary=index):
                self.assertEqual(snapshot.constraint_names, ("x0_cap",))
                np.testing.assert_allclose(
                    snapshot.multipliers, [expected["post_update_multipliers"][index]],
                    rtol=0.0, atol=CLOSED_FORM_TOLERANCE,
                )
                x = np.asarray(snapshot.x, dtype=float)
                self.assertEqual(x.shape, (2,))
                # x1 = 1 is inside the inner solve's tolerance, not exact.
                self.assertAlmostEqual(float(x[1]), 1.0, delta=1e-6)
                if expected["x0"][index] is not None:
                    self.assertAlmostEqual(float(x[0]), expected["x0"][index], delta=CLOSED_FORM_TOLERANCE)
        self.assertLess(float(snapshots[-1].x[0]), 1.0)
        incumbent = snapshots[-1].best_feasible
        np.testing.assert_array_equal(incumbent.x, snapshots[-1].x)


class AlmCheckpointFromBoundariesTests(unittest.TestCase):
    def test_boundaries_alone_rebuild_every_checkpoint(self):
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                with captured_library_boundaries() as runs:
                    trajectory = scenario.run()
                recorded_runs = _runs(trajectory)
                self.assertEqual(len(runs), len(recorded_runs))
                for run_index, ((inner_options, boundaries), recorded) in enumerate(
                    zip(runs, recorded_runs)
                ):
                    rebuilt = [
                        golden.encode(
                            alm_checkpoint.transition_snapshot(boundary, inner_options)
                        )
                        for boundary in boundaries
                    ]
                    difference = golden.first_difference(recorded["checkpoints"], rebuilt)
                    self.assertIsNone(
                        difference,
                        f"boundaries do not rebuild '{scenario.name}' "
                        f"checkpoints:\n{difference}",
                    )
                    difference = frozen_checkpoints_difference(scenario.name, run_index, rebuilt)
                    self.assertIsNone(
                        difference,
                        f"'{scenario.name}' checkpoints differ from the stored golden:\n{difference}",
                    )

    def test_published_and_resume_boundaries_are_read_only(self):
        with captured_library_boundaries() as runs:
            golden.SCENARIOS_BY_NAME["plateau_restore_best_feasible"].run()
        ((inner_options, published),) = runs
        resumed = [
            alm_checkpoint.resume_boundary(
                alm_checkpoint.transition_snapshot(boundary, inner_options)
            )
            for boundary in published
        ]
        boundaries = published + resumed
        self.assertTrue(
            any(boundary.state.best_feasible is not None for boundary in resumed)
        )
        for boundary in boundaries:
            state = boundary.state
            arrays = [state.x, state.multipliers]
            if state.best_feasible is not None:
                arrays += [state.best_feasible.x, state.best_feasible.multipliers]
                self.assertIsInstance(
                    state.best_feasible.evaluation, MappingProxyType
                )
            for array in arrays:
                self.assertFalse(array.flags.writeable)

    def test_a_resumable_snapshot_round_trips_through_its_boundary(self):
        with captured_library_boundaries() as runs:
            golden.SCENARIOS_BY_NAME["penalty_ramp_to_cap"].run()
        ((inner_options, boundaries),) = runs
        resumable = [
            boundary for boundary in boundaries if boundary.termination_reason is None
        ]
        self.assertGreater(len(resumable), 0)
        for boundary in resumable:
            snapshot = alm_checkpoint.transition_snapshot(boundary, inner_options)
            round_trip = alm_checkpoint.transition_snapshot(
                alm_checkpoint.resume_boundary(snapshot), inner_options
            )
            self.assertIsNone(
                golden.first_difference(
                    golden.encode(snapshot), golden.encode(round_trip)
                )
            )


def _diagnostic_run(diagnostic, completed_outer_callback=None):
    """min (x - 1)^2 s.t. x <= 0 from x = 0 (a feasible start, so every
    boundary carries a best-feasible incumbent) with ``diagnostic`` in every
    evaluation; checkpointed through ``alm_checkpointing``. Returns the
    published boundaries and the plugin's inner options."""

    def physics(x):
        return alm.ALMPhysics(
            (x[0] - 1.0) ** 2, [2.0 * (x[0] - 1.0)], [x[0]], ([1.0],),
            extras={"diagnostic": diagnostic},
        )

    plugin = alm_checkpoint.alm_checkpointing(
        {"maxiter": 20}, completed_outer_callback=completed_outer_callback
    )
    boundaries = []

    def observe(boundary):
        boundaries.append(boundary)
        if plugin.on_outer_boundary is not None:
            plugin.on_outer_boundary(boundary)

    alm.minimize_alm(
        [0.0], ["g"], alm.cached_alm_evaluator(physics),
        alm.ALMSettings(max_outer_iterations=3), plugin.inner_options,
        on_outer_boundary=observe,
    )
    return boundaries, plugin.inner_options


# Diagnostics the checkpoint must carry unchanged (arrays and lists come
# back as tuples, the documented sequence form): a pair of arrays, and raw
# values shaped like the codec's own tagged tuples.
RAW_DIAGNOSTICS = {
    "array_pair": (
        (np.array([1.0, 2.0]), np.array([3.0, 4.0])),
        ((1.0, 2.0), (3.0, 4.0)),
    ),
    "sequence_marker_tuple": (
        ("__alm_sequence__", ("a", "b")),
        ("__alm_sequence__", ("a", "b")),
    ),
    "mapping_marker_tuple": (
        ("__alm_mapping__", (("k", 1.0),)),
        ("__alm_mapping__", (("k", 1.0),)),
    ),
    "marker_list": (["__alm_sequence__", ["a"]], ("__alm_sequence__", ("a",))),
    "mapping_with_marker_key": ({"__alm_mapping__": 1.0}, {"__alm_mapping__": 1.0}),
}


class AlmCheckpointRawValueTests(unittest.TestCase):
    """SR-01: a checkpoint encodes every raw value; nothing raw is mistaken
    for an encoded one (an array-valued tuple crashed the codec, and a raw
    marker-shaped tuple lost its marker)."""

    def test_raw_diagnostics_round_trip_through_snapshot_and_resume_boundary(self):
        for name, (diagnostic, expected) in RAW_DIAGNOSTICS.items():
            with self.subTest(diagnostic=name):
                boundaries, inner_options = _diagnostic_run(diagnostic)
                snapshot = alm_checkpoint.transition_snapshot(boundaries[0], inner_options)
                restored = alm_checkpoint.resume_boundary(snapshot)
                self.assertEqual(
                    restored.state.best_feasible.evaluation["diagnostic"], expected
                )

    def test_raw_diagnostics_round_trip_through_alm_checkpointing(self):
        for name, (diagnostic, expected) in RAW_DIAGNOSTICS.items():
            with self.subTest(diagnostic=name):
                snapshots = []
                boundaries, inner_options = _diagnostic_run(diagnostic, snapshots.append)
                self.assertEqual(len(snapshots), len(boundaries))
                reloaded = pickle.loads(pickle.dumps(snapshots[0]))
                plugin = alm_checkpoint.alm_checkpointing(
                    {"maxiter": 20}, resume_state=reloaded
                )
                self.assertEqual(
                    plugin.resume_from.state.best_feasible.evaluation["diagnostic"],
                    expected,
                )
                resumed = alm.minimize_alm(
                    np.asarray(reloaded.x, dtype=float), ["g"],
                    alm.cached_alm_evaluator(
                        lambda x: alm.ALMPhysics(
                            (x[0] - 1.0) ** 2, [2.0 * (x[0] - 1.0)], [x[0]], ([1.0],),
                            extras={"diagnostic": diagnostic},
                        )
                    ),
                    alm.ALMSettings(max_outer_iterations=3), plugin.inner_options,
                    resume_from=plugin.resume_from,
                )
                self.assertEqual(resumed.evaluation["diagnostic"], expected)

    def test_raw_inner_options_round_trip(self):
        boundaries, _inner_options = _diagnostic_run(1.0)
        for name, (value, expected) in RAW_DIAGNOSTICS.items():
            with self.subTest(option=name):
                snapshot = alm_checkpoint.transition_snapshot(
                    boundaries[0], {"maxiter": 20, "note": value}
                )
                resumed_options = alm_checkpoint.alm_checkpointing(
                    {"maxiter": 7}, resume_state=snapshot
                ).inner_options
                self.assertEqual(resumed_options, {"maxiter": 7, "note": expected})

    def test_snapshot_constructors_take_encoded_values_only(self):
        # transition_snapshot encodes; a constructor only checks, so a raw
        # value is rejected instead of guessed at.
        for name, (raw, _expected) in RAW_DIAGNOSTICS.items():
            if name in ("sequence_marker_tuple", "mapping_marker_tuple"):
                continue  # also well-formed encoded values (of ("a", "b") and {"k": 1.0})
            with self.subTest(value=name):
                with self.assertRaisesRegex(TypeError, "encoded"):
                    alm_checkpoint.ALMFeasibleIncumbentSnapshot(
                        x=(0.0,), evaluation=(("diagnostic", raw),), multipliers=(0.0,),
                        penalty=1.0, accepted_state=None,
                    )


class AlmTerminalSnapshotTests(unittest.TestCase):
    def test_terminal_snapshot_cannot_resume(self):
        with captured_library_boundaries() as runs:
            golden.SCENARIOS_BY_NAME["penalty_ramp_to_cap"].run()
        ((inner_options, boundaries),) = runs
        terminal = alm_checkpoint.transition_snapshot(boundaries[-1], inner_options)
        self.assertFalse(terminal.resume_eligible)
        with self.assertRaisesRegex(ValueError, "terminal and cannot be resumed"):
            alm.minimize_alm(
                np.asarray(terminal.x, dtype=float),
                list(terminal.constraint_names),
                golden.far_target_evaluate,
                alm.ALMSettings(max_outer_iterations=20),
                dict(inner_options),
                resume_from=alm_checkpoint.resume_boundary(terminal),
            )


def _penalty_ramp_resume_boundary():
    """The first resumable boundary of ``penalty_ramp_to_cap`` and its inner options."""
    with captured_library_boundaries() as runs:
        golden.SCENARIOS_BY_NAME["penalty_ramp_to_cap"].run()
    ((inner_options, boundaries),) = runs
    return boundaries[0], dict(inner_options)


class AlmResumeBoundaryValidationTests(unittest.TestCase):
    """``resume_from`` is an integrity boundary: a malformed boundary fails at
    loop entry with the rule ``ALMTransitionSnapshot`` states, before any
    restore or evaluation."""

    def _assert_rejected(self, boundary, inner_options, message):
        calls = []

        def evaluate_problem(*args):
            calls.append("evaluate_problem")
            raise AssertionError("a malformed resume boundary reached evaluation")

        def restore_incumbent_state(state):
            calls.append("restore_incumbent_state")

        with self.assertRaisesRegex(ValueError, message):
            alm.minimize_alm(
                np.asarray(boundary.state.x, dtype=float).copy(),
                list(boundary.constraint_names),
                evaluate_problem,
                alm.ALMSettings(max_outer_iterations=20, penalty_max=500.0),
                inner_options,
                snapshot_accepted_state_fn=lambda: None,
                restore_incumbent_state_fn=restore_incumbent_state,
                resume_from=boundary,
            )
        self.assertEqual(calls, [])

    def test_malformed_resume_boundaries_are_rejected_loudly(self):
        boundary, inner_options = _penalty_ramp_resume_boundary()
        state = boundary.state
        bad_states = (
            (
                "negative multipliers",
                dataclasses.replace(state, multipliers=np.array([-1.0])),
                "multipliers must be nonnegative",
            ),
            (
                "zero update_feasibility_tol",
                dataclasses.replace(state, update_feasibility_tol=0.0),
                "update_feasibility_tol must be finite and positive",
            ),
            (
                "negative trust radius",
                dataclasses.replace(state, trust_radius=-1.0),
                "trust_radius must be finite and positive",
            ),
            (
                "zero trust radius (None is no box)",
                dataclasses.replace(state, trust_radius=0.0),
                "trust_radius must be finite and positive",
            ),
            (
                "multiplier length differs from the constraint names",
                dataclasses.replace(state, multipliers=np.array([0.0, 0.0])),
                "constraint names and multipliers must have equal dimensions",
            ),
            (
                "non-finite x",
                dataclasses.replace(state, x=np.array([np.nan, 0.0])),
                r"x\[0\] must be a finite real number",
            ),
        )
        for label, bad_state, message in bad_states:
            with self.subTest(case=label):
                self._assert_rejected(
                    dataclasses.replace(boundary, state=bad_state),
                    inner_options,
                    message,
                )
        with self.subTest(case="geometry identity without an accepted state"):
            self._assert_rejected(
                dataclasses.replace(
                    boundary, accepted_state=None, geometry_identity="unpaired"
                ),
                inner_options,
                "geometry_identity requires accepted_state",
            )


class AlmLoopOwnsNoCheckpointTests(unittest.TestCase):
    CHECKPOINT_SYMBOLS = frozenset(
        (
            "ALMTransitionSnapshot",
            "ALMFeasibleIncumbentSnapshot",
            "_make_alm_transition_snapshot",
            "_transition_json_value",
            "_snapshot_transition_evaluation",
            "_restore_transition_json_value",
            "_restore_transition_evaluation",
        )
    )

    def test_control_defines_no_checkpoint_type_or_codec(self):
        tree = ast.parse((PACKAGE_ROOT / "control.py").read_text(encoding="utf-8"))
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        self.assertEqual(sorted(defined & self.CHECKPOINT_SYMBOLS), [])

    def test_loop_entry_point_takes_a_generic_resume_boundary(self):
        parameters = inspect.signature(alm_control.minimize_alm).parameters
        self.assertIn("resume_from", parameters)
        self.assertIn("on_outer_boundary", parameters)
        self.assertNotIn("resume_state", parameters)
        self.assertNotIn("completed_outer_callback", parameters)

    def test_checkpoint_module_owns_the_snapshot_types(self):
        for name in ("ALMTransitionSnapshot", "ALMFeasibleIncumbentSnapshot"):
            with self.subTest(name=name):
                self.assertEqual(
                    getattr(alm_checkpoint, name).__module__,
                    "simsopt_alm.checkpoint",
                )

    def test_package_import_does_not_load_the_checkpoint_plugin(self):
        probe = (
            "import sys\n"
            "import simsopt_alm\n"
            "sys.exit(1 if 'simsopt_alm.checkpoint' in sys.modules else 0)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=False
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
