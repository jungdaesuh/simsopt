"""Mock-solver unit tests for ``build_boozer_surface_family``.

These tests inject a FAKE Boozer solver (canned ``BoozerInitializationResult``
objects) and a fake nesting predicate so no real BoozerSurface solve runs -- the
suite exercises the family helper's CONTROL FLOW (accept/stop/raise) and its
provenance/ordering contract, fast and deterministically. The real adjoint solve
is covered elsewhere; here we only prove the marcher's policy.

Observable behaviors under test:
  * strict all-success returns every shell, re-ordered inner-to-outer;
  * strict mode re-raises the driver's ``RuntimeError`` on the first failure;
  * truncation stops at the first shell that fails to solve, fails to nest, or
    breaks volume ordering, keeping the outer band;
  * ``min_surfaces`` raises when too few shells are accepted;
  * provenance (accepted/requested/last_good_name/truncated) is correct.
"""

import sys
import unittest
from pathlib import Path

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt import boozer_surface_family as family  # noqa: E402
from banana_opt.boozer_surface_family import (  # noqa: E402
    BoozerFamilyOuterSeed,
    build_boozer_surface_family,
)
from banana_opt.stage2_single_stage_handoff import (  # noqa: E402
    BoozerInitializationResult,
)


class _FakeSurface:
    """Minimal stand-in for a SurfaceXYZTensorFourier in the family marcher.

    Exposes only what the helper touches: ``volume()`` (used for ordering and by
    the contract callable) and a ``nested`` flag read by the fake nesting
    predicate to decide adjacent-pair nesting without any geometry.
    """

    def __init__(self, volume, nested=True):
        self._volume = float(volume)
        self.nested = nested

    def volume(self):
        return self._volume


class _FakeBoozerSurface:
    """Stand-in for a solved BoozerSurface: a ``.surface`` and a ``.res`` dict."""

    def __init__(self, surface, iota=0.3, G=1.0):
        self.surface = surface
        self.res = {"iota": float(iota), "G": float(G)}


def _success(volume, *, nested=True, iota=0.3, G=1.0):
    """A canned solved-and-clean BoozerInitializationResult for one shell."""
    surface = _FakeSurface(volume, nested=nested)
    return BoozerInitializationResult(
        boozer_surface=_FakeBoozerSurface(surface, iota=iota, G=G),
        solve_success=True,
        self_intersecting=False,
        success=True,
        solved_iota=iota,
        solved_G=G,
        volume=float(volume),
    )


def _failure(volume):
    """A canned solver-failure BoozerInitializationResult (``success=False``)."""
    surface = _FakeSurface(volume, nested=True)
    return BoozerInitializationResult(
        boozer_surface=_FakeBoozerSurface(surface),
        solve_success=False,
        self_intersecting=None,
        success=False,
        solved_iota=None,
        solved_G=None,
        volume=float(volume),
        error_type=None,
        error_message=None,
    )


def _configs(names_volumes):
    """Inner-to-outer config dicts from ``[(name, target_volume), ...]``."""
    return [
        {"name": name, "seed_label": "fake", "target_volume": float(volume)}
        for name, volume in names_volumes
    ]


def _fake_make_entry(config, boozer_surface, provenance):
    """Stand-in for ``_surface_data_entry`` (no name-keyed assertions here)."""
    return {
        "name": config["name"],
        "seed_label": config["seed_label"],
        "target_volume": config["target_volume"],
        "boozer_surface": boozer_surface,
        "initialization_provenance": provenance,
    }


def _fake_contract(previous_surface, target_volume):
    """Stand-in for ``contract_surface_to_target_volume``.

    Returns a surface at exactly ``target_volume`` (the marcher only feeds this
    as the next guess; the canned solver decides the solved geometry), preserving
    the previous surface's ``nested`` flag so a non-nested seed stays non-nested.
    """
    return _FakeSurface(target_volume, nested=getattr(previous_surface, "nested", True))


def _fake_nested(inner_surface, outer_surface, *args, **kwargs):
    """Stand-in for ``cross_sections_are_nested``: read the inner ``nested`` flag.

    Returns the (bool, bad_phis) tuple shape the real predicate returns.
    """
    nested = bool(getattr(inner_surface, "nested", True))
    return nested, [] if nested else [0.0]


class BuildBoozerSurfaceFamilyTests(unittest.TestCase):
    def setUp(self):
        # Inject the fake solver + nesting predicate at module scope; the marcher
        # resolves both through this module's globals, so patching here suffices.
        self._real_attempt = family.attempt_initialize_boozer_surface
        self._real_nested = family.cross_sections_are_nested
        family.cross_sections_are_nested = _fake_nested
        self.addCleanup(self._restore)
        # An outer seed whose surface volume is the largest in the stack.
        self.outer_seed = BoozerFamilyOuterSeed(
            surface=_FakeSurface(5.0), iota=0.3, G=1.0
        )

    def _restore(self):
        family.attempt_initialize_boozer_surface = self._real_attempt
        family.cross_sections_are_nested = self._real_nested

    def _install_solver(self, results_in_solve_order):
        """Queue canned results consumed in solve order: outer, then inner shells."""
        queue = list(results_in_solve_order)

        def fake_attempt(*args, **kwargs):
            return queue.pop(0)

        family.attempt_initialize_boozer_surface = fake_attempt
        return queue

    def _build(self, ordered_configs, **kwargs):
        return build_boozer_surface_family(
            ordered_configs,
            mpol=4,
            ntor=4,
            bs=object(),
            constraint_weight=1.0,
            boozer_I=0.0,
            nfp=5,
            outer_seed=self.outer_seed,
            contract=_fake_contract,
            make_entry=_fake_make_entry,
            **kwargs,
        )

    def test_strict_all_success_returns_inner_to_outer(self):
        """Strict mode keeps every shell and returns them inner-to-outer."""
        configs = _configs(
            [("inner0", 1.0), ("inner1", 2.0), ("inner2", 3.0), ("outer", 4.0)]
        )
        # Solve order is outer, inner2, inner1, inner0 with strictly shrinking
        # volumes inner-to-outer (4 > 3 > 2 > 1).
        self._install_solver(
            [
                _success(4.0),  # outer
                _success(3.0),  # inner2
                _success(2.0),  # inner1
                _success(1.0),  # inner0
            ]
        )
        data, provenance = self._build(configs, allow_truncation=False, min_surfaces=4)

        self.assertEqual(
            [entry["name"] for entry in data],
            ["inner0", "inner1", "inner2", "outer"],
        )
        self.assertFalse(provenance.truncated)
        self.assertEqual(provenance.accepted, 4)
        self.assertEqual(provenance.requested, 4)

    def test_strict_raises_on_first_failure(self):
        """Strict mode re-raises the driver's RuntimeError when a shell fails."""
        configs = _configs(
            [("inner0", 1.0), ("inner1", 2.0), ("inner2", 3.0), ("outer", 4.0)]
        )
        # outer ok, inner2 ok, inner1 FAILS -> strict must raise (never reaches inner0).
        self._install_solver(
            [
                _success(4.0),  # outer
                _success(3.0),  # inner2
                _failure(2.0),  # inner1 -> fails
                _success(1.0),  # inner0 (never consumed)
            ]
        )
        with self.assertRaises(RuntimeError) as ctx:
            self._build(configs, allow_truncation=False, min_surfaces=4)
        self.assertIn("Something went wrong with the Boozer solve", str(ctx.exception))

    def test_truncation_stops_at_first_solver_failure(self):
        """Truncation keeps the outer band and stops at the first failed solve."""
        configs = _configs(
            [("inner0", 1.0), ("inner1", 2.0), ("inner2", 3.0), ("outer", 4.0)]
        )
        # outer ok, inner2 ok, inner1 FAILS -> accepted band is {inner2, outer}.
        self._install_solver(
            [
                _success(4.0),  # outer
                _success(3.0),  # inner2
                _failure(2.0),  # inner1 -> stop here
                _success(1.0),  # inner0 (never consumed)
            ]
        )
        data, provenance = self._build(configs, allow_truncation=True, min_surfaces=2)

        self.assertEqual([entry["name"] for entry in data], ["inner2", "outer"])
        self.assertTrue(provenance.truncated)
        self.assertEqual(provenance.accepted, 2)
        self.assertEqual(provenance.requested, 4)
        self.assertEqual(provenance.last_good_name, "inner2")

    def test_truncation_stops_at_first_non_nested_shell(self):
        """Truncation stops when a solved shell does not nest inside the previous."""
        configs = _configs(
            [("inner0", 1.0), ("inner1", 2.0), ("inner2", 3.0), ("outer", 4.0)]
        )
        # inner2 solves but is flagged non-nested -> rejected; band is {outer}.
        self._install_solver(
            [
                _success(4.0),  # outer
                _success(3.0, nested=False),  # inner2 -> not nested, stop
                _success(2.0),  # inner1 (never consumed)
                _success(1.0),  # inner0 (never consumed)
            ]
        )
        data, provenance = self._build(configs, allow_truncation=True, min_surfaces=1)

        self.assertEqual([entry["name"] for entry in data], ["outer"])
        self.assertTrue(provenance.truncated)
        self.assertEqual(provenance.accepted, 1)
        self.assertEqual(provenance.last_good_name, "outer")

    def test_truncation_stops_when_volume_not_ordered(self):
        """Truncation stops when a solved shell is not strictly smaller than prev."""
        configs = _configs(
            [("inner0", 1.0), ("inner1", 2.0), ("inner2", 3.0), ("outer", 4.0)]
        )
        # inner2 solves but at volume 4.5 (>= outer 4.0) -> ordering broken, stop.
        self._install_solver(
            [
                _success(4.0),  # outer
                _success(4.5),  # inner2 -> not smaller than outer, stop
                _success(2.0),  # inner1 (never consumed)
                _success(1.0),  # inner0 (never consumed)
            ]
        )
        data, provenance = self._build(configs, allow_truncation=True, min_surfaces=1)

        self.assertEqual([entry["name"] for entry in data], ["outer"])
        self.assertTrue(provenance.truncated)
        self.assertEqual(provenance.accepted, 1)

    def test_min_surfaces_raises_when_accepted_below_floor(self):
        """Truncation raises when fewer than min_surfaces shells are accepted."""
        configs = _configs(
            [("inner0", 1.0), ("inner1", 2.0), ("inner2", 3.0), ("outer", 4.0)]
        )
        # Only outer + inner2 accepted (inner1 fails), but floor is 3 -> raise.
        self._install_solver(
            [
                _success(4.0),  # outer
                _success(3.0),  # inner2
                _failure(2.0),  # inner1 -> stop, accepted=2
                _success(1.0),  # inner0 (never consumed)
            ]
        )
        with self.assertRaises(RuntimeError) as ctx:
            self._build(configs, allow_truncation=True, min_surfaces=3)
        self.assertIn("requires at least 3", str(ctx.exception))

    def test_provenance_on_full_truncation_acceptance(self):
        """A truncation run that accepts every shell reports truncated=False."""
        configs = _configs([("inner0", 1.0), ("inner1", 2.0), ("outer", 3.0)])
        self._install_solver(
            [
                _success(3.0),  # outer
                _success(2.0),  # inner1
                _success(1.0),  # inner0
            ]
        )
        data, provenance = self._build(configs, allow_truncation=True, min_surfaces=3)

        self.assertEqual(
            [entry["name"] for entry in data], ["inner0", "inner1", "outer"]
        )
        self.assertFalse(provenance.truncated)
        self.assertEqual(provenance.accepted, 3)
        self.assertEqual(provenance.requested, 3)
        self.assertEqual(provenance.last_good_name, "inner0")

    def test_outer_provenance_label(self):
        """The outermost accepted entry carries the stage2 outer-seed provenance."""
        configs = _configs([("inner0", 1.0), ("inner1", 2.0), ("outer", 3.0)])
        self._install_solver(
            [_success(3.0), _success(2.0), _success(1.0)]
        )
        data, _ = self._build(configs, allow_truncation=False, min_surfaces=3)

        by_name = {entry["name"]: entry for entry in data}
        self.assertEqual(
            by_name["outer"]["initialization_provenance"], "stage2_outer_seed"
        )
        # Inner shells carry the "<prev>_continuation_<name>" chain label.
        self.assertEqual(
            by_name["inner1"]["initialization_provenance"],
            "outer_continuation_inner1",
        )
        self.assertEqual(
            by_name["inner0"]["initialization_provenance"],
            "inner1_continuation_inner0",
        )


class RelaxedBandPostconditionTests(unittest.TestCase):
    """Direct unit tests of ``_require_relaxed_family_postconditions`` -- the doc's
    Phase-2 Validation headline (every adjacent pair nests; volumes strictly
    increase inner-to-outer). The per-step march guard normally filters a bad band
    before it reaches this whole-band check, so these exercise the raise paths
    directly. ``cross_sections_are_nested`` is monkeypatched (the real predicate
    needs true geometry the fake surfaces lack); the volume-order branch fires
    before nesting, so it needs no patch.
    """

    @staticmethod
    def _band(volumes):
        return [
            {"name": f"s{i}", "boozer_surface": _FakeBoozerSurface(_FakeSurface(v))}
            for i, v in enumerate(volumes)
        ]

    def test_raises_when_band_volumes_not_strictly_increasing(self):
        with self.assertRaises(RuntimeError) as ctx:
            family._require_relaxed_family_postconditions(self._band([2.0, 1.0]))
        self.assertIn("strictly", str(ctx.exception))

    def test_raises_when_adjacent_pair_not_nested(self):
        original = family.cross_sections_are_nested
        family.cross_sections_are_nested = lambda inner, outer: (False, [0.0])
        try:
            with self.assertRaises(RuntimeError) as ctx:
                family._require_relaxed_family_postconditions(self._band([1.0, 2.0]))
            self.assertIn("must nest", str(ctx.exception))
        finally:
            family.cross_sections_are_nested = original

    def test_accepts_strictly_increasing_nested_band(self):
        original = family.cross_sections_are_nested
        family.cross_sections_are_nested = lambda inner, outer: (True, [])
        try:
            family._require_relaxed_family_postconditions(self._band([1.0, 2.0, 3.0]))
        finally:
            family.cross_sections_are_nested = original


if __name__ == "__main__":
    unittest.main()
