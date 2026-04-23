import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def load_replay_module():
    return importlib.import_module("banana_opt.banana_current_replay")


def load_incumbents_module():
    return importlib.import_module("banana_opt.incumbents")


def build_incumbent_state(*, offset=0.0):
    incumbents_module = load_incumbents_module()
    return incumbents_module.SingleStageIncumbentState(
        x=np.asarray([1.0 + offset, 2.0 + offset, 3.0 + offset], dtype=float),
        surface_state={"surface_state": [offset]},
        objective_total=float(4.0 + offset),
        objective_grad=np.asarray([5.0 + offset, 6.0 + offset, 7.0 + offset]),
        search_eval={"total": float(4.0 + offset)},
        surface_status={"success": True, "offset": offset},
        search_surface_status={"success": True, "offset": offset},
        accepted_hardware_status={"success": True},
        topology_gate_status={"state": "ok"},
    )


def build_replay_contract(
    *,
    seed_currents_A=(15000.0, -14000.0),
    configured_seed_currents_A=(15438.0, -15335.0),
):
    return {
        "mode": "independent",
        "num_control_currents": 2,
        "coordinate_dof_names": ["Current31:x0", "Current32:x0"],
        "current_coordinate_scale_factors_A": [15438.0, 15335.0],
        "seed_currents_A": list(seed_currents_A),
        "configured_seed_currents_A": list(configured_seed_currents_A),
    }


def set_replay_contract(replay_module, context_state, **overrides):
    contract = build_replay_contract(**overrides)
    replay_module.set_banana_current_replay_context_contract(
        context_state,
        mode=contract["mode"],
        num_control_currents=contract["num_control_currents"],
        coordinate_dof_names=contract["coordinate_dof_names"],
        current_coordinate_scale_factors_A=contract[
            "current_coordinate_scale_factors_A"
        ],
        seed_currents_A=contract["seed_currents_A"],
        configured_seed_currents_A=contract["configured_seed_currents_A"],
    )


class BananaCurrentReplayTests(unittest.TestCase):
    def test_replay_context_round_trips_multiple_incumbents(self):
        replay_module = load_replay_module()
        context_state = replay_module.build_banana_current_replay_context_state()
        set_replay_contract(
            replay_module,
            context_state,
            seed_currents_A=(15000.0, -14000.0),
            configured_seed_currents_A=(15438.0, -15335.0),
        )
        seed_incumbent = build_incumbent_state(offset=0.0)
        accepted_incumbent = build_incumbent_state(offset=10.0)

        replay_module.record_banana_current_replay_context_snapshot(
            context_state,
            accepted_iteration=0,
            accepted_boozer_stage="seed",
            incumbent=seed_incumbent,
        )
        replay_module.record_banana_current_replay_context_snapshot(
            context_state,
            accepted_iteration=2,
            accepted_boozer_stage="refined",
            incumbent=accepted_incumbent,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = replay_module.write_banana_current_replay_context_artifact(
                temp_dir,
                context_state,
            )
            loaded_state = replay_module.load_banana_current_replay_context(
                artifact_path
            )

        seed_stage, restored_seed = replay_module.restore_banana_current_replay_incumbent(
            loaded_state,
            0,
        )
        accepted_stage, restored_accepted = (
            replay_module.restore_banana_current_replay_incumbent(
                loaded_state,
                2,
            )
        )

        self.assertEqual(seed_stage, "seed")
        self.assertEqual(accepted_stage, "refined")
        self.assertTrue(np.allclose(restored_seed.x, seed_incumbent.x))
        self.assertTrue(np.allclose(restored_accepted.x, accepted_incumbent.x))
        self.assertEqual(
            restored_accepted.surface_status["offset"],
            accepted_incumbent.surface_status["offset"],
        )
        self.assertEqual(
            loaded_state["replay_contract"]["coordinate_dof_names"],
            ["Current31:x0", "Current32:x0"],
        )

    def test_build_replayed_candidate_x_replaces_only_current_block(self):
        replay_module = load_replay_module()

        replayed_x = replay_module.build_replayed_candidate_x(
            np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float),
            coordinate_indices=(1, 3),
            optimizer_coordinate_values=(20.0, 40.0),
        )

        self.assertTrue(np.allclose(replayed_x, np.asarray([1.0, 20.0, 3.0, 40.0])))

    def test_validate_coordinate_contract_accepts_matching_live_contract(self):
        replay_module = load_replay_module()

        replay_module.validate_banana_current_replay_coordinate_contract(
            {
                "coordinate_dof_names": ["Current31:x0", "Current32:x0"],
                "current_coordinate_scale_factors_A": [15438.0, 15335.0],
            },
            live_dof_names=("Current31:x0", "Current32:x0"),
            live_scale_factors_A=(15438.0, 15335.0),
        )

    def test_validate_coordinate_contract_rejects_mismatched_live_contract(self):
        replay_module = load_replay_module()

        with self.assertRaisesRegex(ValueError, "DOF names do not match"):
            replay_module.validate_banana_current_replay_coordinate_contract(
                {
                    "coordinate_dof_names": ["Current31:x0", "Current32:x0"],
                    "current_coordinate_scale_factors_A": [15438.0, 15335.0],
                },
                live_dof_names=("Current99:x0", "Current32:x0"),
                live_scale_factors_A=(15438.0, 15335.0),
            )

    def test_validate_replay_context_contract_rejects_mismatched_seed_currents(self):
        replay_module = load_replay_module()
        context_state = replay_module.build_banana_current_replay_context_state()
        set_replay_contract(
            replay_module,
            context_state,
            seed_currents_A=(15438.0, -15335.0),
        )

        with self.assertRaisesRegex(ValueError, "seed currents do not match"):
            replay_module.validate_banana_current_replay_context_contract(
                context_state,
                {
                    **build_replay_contract(
                        seed_currents_A=(15438.0, -9999.0),
                        configured_seed_currents_A=(15438.0, -15335.0),
                    ),
                    "seed_report": {
                        "coordinate_dof_names": ["Current31:x0", "Current32:x0"],
                        "current_coordinate_scale_factors_A": [15438.0, 15335.0],
                    },
                },
            )

    def test_validate_replay_context_contract_accepts_distinct_runtime_and_configured_seed_currents(
        self,
    ):
        replay_module = load_replay_module()
        context_state = replay_module.build_banana_current_replay_context_state()
        set_replay_contract(
            replay_module,
            context_state,
            seed_currents_A=(15000.0, -14000.0),
            configured_seed_currents_A=(15438.0, -15335.0),
        )

        replay_module.validate_banana_current_replay_context_contract(
            context_state,
            {
                **build_replay_contract(
                    seed_currents_A=(15000.0, -14000.0),
                    configured_seed_currents_A=(15438.0, -15335.0),
                ),
                "seed_report": {
                    "coordinate_dof_names": ["Current31:x0", "Current32:x0"],
                    "current_coordinate_scale_factors_A": [15438.0, 15335.0],
                },
            },
        )

    def test_load_replay_context_can_require_contract_metadata(self):
        replay_module = load_replay_module()
        context_state = replay_module.build_banana_current_replay_context_state()
        replay_module.record_banana_current_replay_context_snapshot(
            context_state,
            accepted_iteration=0,
            accepted_boozer_stage="seed",
            incumbent=build_incumbent_state(offset=0.0),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = replay_module.write_banana_current_replay_context_artifact(
                temp_dir,
                context_state,
            )
            with self.assertRaisesRegex(ValueError, "missing replay_contract metadata"):
                replay_module.load_banana_current_replay_context(
                    artifact_path,
                    require_replay_contract=True,
                )


if __name__ == "__main__":
    unittest.main()
