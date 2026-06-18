"""Backward-compat derivation of lcfs edge metrics for the stage-2 seed contract.

Pre-2026 stage-2 seeds record FINAL_LCFS_MAJOR_RADIUS_M + FINAL_LCFS_MINOR_RADIUS_M
but not the lcfs_outboard_edge / lcfs_inboard_edge metrics the recovery contract
checks. The loader derives them from the recorded LCFS geometry via the canonical
edge formula (R0 +/- a) so such seeds load without per-seed metadata stamping, and
must never clobber an explicitly recorded edge.
"""

import importlib
import unittest

from geo._frontier_test_helpers import ensure_examples_import_path

ensure_examples_import_path()
handoff = importlib.import_module("banana_opt.stage2_single_stage_handoff")


class LcfsEdgeBackfillTest(unittest.TestCase):
    def test_edges_derived_from_recorded_lcfs_geometry(self):
        seed = {
            "FINAL_LCFS_MAJOR_RADIUS_M": 0.9148,
            "FINAL_LCFS_MINOR_RADIUS_M": 0.0780,
        }
        measured = handoff._stage2_seed_measured_values(seed)
        self.assertAlmostEqual(measured["lcfs_outboard_edge"], 0.9148 + 0.0780)
        self.assertAlmostEqual(measured["lcfs_inboard_edge"], 0.9148 - 0.0780)

    def test_no_derivation_without_recorded_geometry(self):
        # No LCFS geometry recorded -> cannot derive -> stays missing (fail-closed
        # upstream), never fabricated.
        measured = handoff._stage2_seed_measured_values({})
        self.assertIsNone(measured["lcfs_outboard_edge"])
        self.assertIsNone(measured["lcfs_inboard_edge"])

    def test_explicit_edge_is_not_overwritten(self):
        # A seed that recorded the edge explicitly must keep that value, not the
        # derived one (only None entries are filled).
        field = handoff.hardware_constraint_artifact_value_field_names(
            "lcfs_outboard_edge"
        )[0]
        seed = {
            "FINAL_LCFS_MAJOR_RADIUS_M": 0.90,
            "FINAL_LCFS_MINOR_RADIUS_M": 0.08,
            field: 1.234,
        }
        measured = handoff._stage2_seed_measured_values(seed)
        self.assertEqual(measured["lcfs_outboard_edge"], 1.234)


if __name__ == "__main__":
    unittest.main()
