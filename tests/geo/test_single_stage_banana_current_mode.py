import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

from simsopt.field import BiotSavart
from simsopt.field.coil import Coil, Current, ScaledCurrent
from simsopt.geo import CurveXYZFourier


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def load_current_mode_module():
    return importlib.import_module("banana_opt.single_stage_banana_current_mode")


def load_handoff_module():
    return importlib.import_module("banana_opt.stage2_single_stage_handoff")


def _build_stage2_seed_fixture(
    *,
    shared_current_A: float | None = None,
    banana_currents_a: tuple[float, float] | None = None,
):
    handoff_module = load_handoff_module()
    tf_coil = Coil(CurveXYZFourier(10, 1), Current(8.0e4))
    if banana_currents_a is None:
        assert shared_current_A is not None
        shared_current = Current(shared_current_A)
        banana_coil_a = Coil(CurveXYZFourier(10, 1), shared_current)
        banana_coil_b = Coil(
            CurveXYZFourier(10, 1),
            ScaledCurrent(shared_current, -1.0),
        )
    else:
        banana_coil_a = Coil(CurveXYZFourier(10, 1), Current(banana_currents_a[0]))
        banana_coil_b = Coil(CurveXYZFourier(10, 1), Current(banana_currents_a[1]))
    biot_savart = BiotSavart([tf_coil, banana_coil_a, banana_coil_b])
    coil_partitions = handoff_module.Stage2CoilPartitions(
        tf_coils=(tf_coil,),
        banana_coils=(banana_coil_a, banana_coil_b),
        proxy_coils=(),
        vf_coils=(),
        num_tf_coils=1,
        num_banana_coils=2,
        num_proxy_coils=0,
        num_vf_coils=0,
        finite_current_mode="wataru_proxy_field",
    )
    return biot_savart, coil_partitions


def build_stage2_seed_fixture():
    return _build_stage2_seed_fixture(shared_current_A=1.0e4)


def build_asymmetric_stage2_seed_fixture():
    return _build_stage2_seed_fixture(
        banana_currents_a=(1.2e4, -9.5e3),
    )


class SingleStageBananaCurrentModeTests(unittest.TestCase):
    def test_resolve_shared_mode_preserves_loaded_state(self):
        module = load_current_mode_module()
        biot_savart, coil_partitions = build_stage2_seed_fixture()

        resolved_bs, resolved_partitions, state = (
            module.resolve_single_stage_banana_current_state(
                biot_savart,
                coil_partitions,
                mode="shared",
            )
        )

        self.assertIs(resolved_bs, biot_savart)
        self.assertIs(resolved_partitions, coil_partitions)
        self.assertEqual(state.mode, "shared")
        self.assertEqual(state.current_values_A(), (1.0e4, -1.0e4))
        self.assertEqual(state.seed_currents_A, (1.0e4, -1.0e4))
        self.assertEqual(state.control_current_A(), 1.0e4)
        self.assertEqual(state.representative_current_A(), 1.0e4)
        self.assertEqual(state.compatibility_current_A(), 1.0e4)
        self.assertEqual(state.num_control_currents(), 1)

    def test_resolve_independent_mode_rebuilds_per_coil_currents(self):
        module = load_current_mode_module()
        biot_savart, coil_partitions = build_stage2_seed_fixture()

        resolved_bs, resolved_partitions, state = (
            module.resolve_single_stage_banana_current_state(
                biot_savart,
                coil_partitions,
                mode="independent",
            )
        )

        self.assertIsNot(resolved_bs, biot_savart)
        self.assertIsNot(resolved_partitions.banana_coils[0], coil_partitions.banana_coils[0])
        self.assertIsNot(resolved_partitions.banana_coils[1], coil_partitions.banana_coils[1])
        self.assertEqual(state.current_values_A(), (1.0e4, -1.0e4))
        self.assertEqual(state.seed_currents_A, (1.0e4, -1.0e4))
        self.assertEqual(state.num_control_currents(), 2)

        first_current = resolved_partitions.banana_coils[0].current
        second_current = resolved_partitions.banana_coils[1].current
        self.assertIsNot(first_current, second_current)

        first_current.set_dofs(np.asarray([1.2e4], dtype=float))
        self.assertEqual(first_current.get_value(), 1.2e4)
        self.assertEqual(second_current.get_value(), -1.0e4)
        self.assertEqual(state.current_values_A(), (1.2e4, -1.0e4))
        self.assertEqual(state.control_current_A(), 1.2e4)
        self.assertEqual(state.compatibility_current_A(), 1.2e4)

    def test_resolve_banana_current_coordinate_spec_deduplicates_shared_mode(self):
        module = load_current_mode_module()
        biot_savart, coil_partitions = build_stage2_seed_fixture()

        _, _, state = module.resolve_single_stage_banana_current_state(
            biot_savart,
            coil_partitions,
            mode="shared",
        )
        coordinate_spec = module.resolve_banana_current_coordinate_spec(
            biot_savart,
            state,
        )

        self.assertEqual(
            coordinate_spec.dof_names,
            tuple(state.currents[0].dof_names),
        )
        self.assertEqual(coordinate_spec.indices, (1,))

    def test_resolve_banana_current_coordinate_spec_tracks_independent_currents(self):
        module = load_current_mode_module()
        biot_savart, coil_partitions = build_stage2_seed_fixture()

        resolved_bs, _, state = module.resolve_single_stage_banana_current_state(
            biot_savart,
            coil_partitions,
            mode="independent",
        )
        coordinate_spec = module.resolve_banana_current_coordinate_spec(
            resolved_bs,
            state,
        )

        self.assertEqual(
            coordinate_spec.dof_names,
            (
                *state.currents[0].dof_names,
                *state.currents[1].dof_names,
            ),
        )
        expected_indices = tuple(
            list(resolved_bs.dof_names).index(dof_name)
            for dof_name in coordinate_spec.dof_names
        )
        self.assertEqual(coordinate_spec.indices, expected_indices)

    def test_apply_penalty_bounds_applies_per_independent_current(self):
        module = load_current_mode_module()
        state = module.SingleStageBananaCurrentState(
            mode="independent",
            currents=(Current(1.0e4), Current(-1.0e4)),
            seed_currents_A=(1.0e4, -1.0e4),
        )

        module.apply_single_stage_penalty_banana_current_bounds(
            state,
            banana_current_max_A=1.6e4,
            validate_seed=True,
            seed_context="Loaded Stage 2 banana current",
        )

        for current in state.currents:
            self.assertTrue(
                np.allclose(current.local_lower_bounds, np.asarray([-1.6e4]))
            )
            self.assertTrue(
                np.allclose(current.local_upper_bounds, np.asarray([1.6e4]))
            )

    def test_build_payload_fields_reports_mode_vector_and_control_metric(self):
        module = load_current_mode_module()
        state = module.SingleStageBananaCurrentState(
            mode="independent",
            currents=(Current(1.2e4), Current(-1.5e4)),
            seed_currents_A=(1.0e4, -1.0e4),
        )

        payload = module.build_single_stage_banana_current_payload_fields(
            state,
            prefix="BEST_FEASIBLE_",
        )

        self.assertEqual(payload["BEST_FEASIBLE_BANANA_CURRENT_MODE"], "independent")
        self.assertEqual(
            payload["BEST_FEASIBLE_BANANA_CURRENTS_A"],
            [1.2e4, -1.5e4],
        )
        self.assertEqual(payload["BEST_FEASIBLE_BANANA_CURRENT_MAX_ABS_A"], 1.5e4)
        self.assertEqual(
            payload["BEST_FEASIBLE_BANANA_CURRENT_CONTROL_METRIC"],
            "max_abs",
        )
        self.assertEqual(payload["BEST_FEASIBLE_BANANA_NUM_CURRENT_CONTROLS"], 2)
        self.assertEqual(payload["BEST_FEASIBLE_BANANA_CURRENT_A"], 1.5e4)

    def test_resolve_independent_mode_preserves_loaded_asymmetric_seed_vector(self):
        module = load_current_mode_module()
        biot_savart, coil_partitions = build_asymmetric_stage2_seed_fixture()

        resolved_bs, resolved_partitions, state = (
            module.resolve_single_stage_banana_current_state(
                biot_savart,
                coil_partitions,
                mode="independent",
            )
        )

        self.assertIsNot(resolved_bs, biot_savart)
        self.assertEqual(state.seed_currents_A, (1.2e4, -9.5e3))
        self.assertEqual(state.current_values_A(), (1.2e4, -9.5e3))
        self.assertEqual(state.compatibility_current_A(), 1.2e4)
        self.assertEqual(state.num_control_currents(), 2)
        self.assertIsNot(
            resolved_partitions.banana_coils[0].current,
            resolved_partitions.banana_coils[1].current,
        )


if __name__ == "__main__":
    unittest.main()
