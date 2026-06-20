"""Behavior of the target-mode invariant-torus (KAM) floor incumbent ratchet.

target_kam_floor_satisfied gates incumbent promotion in target mode: a config is
promotable only when its in-loop invariant-torus fraction clears the floor. The floor
is fail-closed (an unevaluated/missing fraction blocks promotion) so an unconfined
config whose field lines escape before classification can never slip through.
"""

import unittest

from geo._frontier_test_helpers import load_frontier_constraints_module

target_kam_floor_satisfied = (
    load_frontier_constraints_module().target_kam_floor_satisfied
)


class TargetKamFloorTest(unittest.TestCase):
    def test_disabled_floor_promotes_any_entry(self):
        # floor=None must be a no-op so default runs stay byte-identical.
        self.assertTrue(target_kam_floor_satisfied(None, floor=None))
        self.assertTrue(
            target_kam_floor_satisfied(
                {"invariant_torus_fraction": 0.0}, floor=None
            )
        )

    def test_fraction_at_or_above_floor_is_promotable(self):
        self.assertTrue(
            target_kam_floor_satisfied(
                {"invariant_torus_fraction": 0.70}, floor=0.60
            )
        )
        # Boundary: fraction exactly at the floor passes (>=).
        self.assertTrue(
            target_kam_floor_satisfied(
                {"invariant_torus_fraction": 0.60}, floor=0.60
            )
        )

    def test_fraction_below_floor_blocks_promotion(self):
        self.assertFalse(
            target_kam_floor_satisfied(
                {"invariant_torus_fraction": 0.59}, floor=0.60
            )
        )

    def test_missing_topology_entry_fails_closed(self):
        self.assertFalse(target_kam_floor_satisfied(None, floor=0.60))

    def test_unevaluated_fraction_fails_closed(self):
        # An unconfined config escapes before classification -> fraction None ->
        # must NOT be promoted under an active floor.
        self.assertFalse(
            target_kam_floor_satisfied(
                {"invariant_torus_fraction": None}, floor=0.60
            )
        )
        self.assertFalse(target_kam_floor_satisfied({}, floor=0.60))

    def test_current_geometry_with_sufficient_fraction_is_promotable(self):
        # Entry scored for the geometry under test, fraction >= floor -> promotable.
        self.assertTrue(
            target_kam_floor_satisfied(
                {"geometry_key": "abc123", "invariant_torus_fraction": 0.72},
                floor=0.60,
                current_geometry_key="abc123",
            )
        )

    def test_stale_geometry_fails_closed(self):
        # Entry scored for a DIFFERENT geometry must not promote the current config,
        # even though its (other config's) fraction clears the floor.
        self.assertFalse(
            target_kam_floor_satisfied(
                {"geometry_key": "abc123", "invariant_torus_fraction": 0.90},
                floor=0.60,
                current_geometry_key="xyz789",
            )
        )

    def test_entry_without_geometry_key_fails_closed_when_key_checked(self):
        # No geometry_key on the entry -> cannot confirm currency -> block.
        self.assertFalse(
            target_kam_floor_satisfied(
                {"invariant_torus_fraction": 0.90},
                floor=0.60,
                current_geometry_key="xyz789",
            )
        )

    def test_floor_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            target_kam_floor_satisfied(
                {"invariant_torus_fraction": 0.5}, floor=1.5
            )
        with self.assertRaises(ValueError):
            target_kam_floor_satisfied(
                {"invariant_torus_fraction": 0.5}, floor=-0.1
            )


if __name__ == "__main__":
    unittest.main()
