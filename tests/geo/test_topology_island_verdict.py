"""Strict-tier island / classifiability certification verdict tests.

Phase-2 audit-1 (spec-143, 2026-06-11) wires the WBA island verdict into the
strict topology certification verdict and adds a hard ``min_returns``
classifiability floor. These tests pin:

* default-OFF byte-identity of ``topology_tier_passed`` (the legacy
  survival-only verdict is preserved for every payload, including payloads with
  no WBA data),
* the opt-in island gate rejects a payload with one ``island_chain`` line when
  ``n_island_max=0`` and admits it when the cap is raised,
* the opt-in classifiability floor fails when too few surviving lines clear
  ``min_returns`` (and fails closed when no line survives),
* requesting either opt-in gate on a result with no per-line classifications
  raises loudly (never a silent pass),
* the per-line tallies are surfaced so a report can quote
  "43/50 survived, 0 island-chain, N/43 classifiable".
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt.topology.kam_birkhoff import (  # noqa: E402
    DEFAULT_WBA_MIN_RETURNS,
    KAM_CLASS_CHAOTIC,
    KAM_CLASS_INVARIANT_TORUS,
    KAM_CLASS_ISLAND_CHAIN,
    KAM_CLASS_LOST,
    island_verdict_counts,
)
from banana_opt.topology_fidelity_ladder import (  # noqa: E402
    DEFAULT_CLASSIFIABLE_FRACTION_MIN,
    DEFAULT_N_ISLAND_MAX,
    WBA_SEED_CLASSIFICATIONS_KEY,
    topology_tier_passed,
    topology_tier_verdict,
)


def _legacy_topology_tier_passed(result, *, survival_threshold):
    """Reference copy of the pre-audit survival-only verdict."""

    if bool(result.get("broken", False)):
        return False
    return float(result.get("survival_fraction", 0.0)) >= float(survival_threshold)


def _seed(classification, *, return_count, seed_index=0):
    return {
        "seed_index": int(seed_index),
        "classification": classification,
        "return_count": int(return_count),
    }


def _build_classifications(
    *,
    survived,
    lost,
    island,
    below_floor,
    min_returns=DEFAULT_WBA_MIN_RETURNS,
):
    """Build a per-line WBA payload.

    ``survived`` surviving lines, of which ``island`` are island_chain and
    ``below_floor`` have too few returns; remaining survivors are clean tori.
    ``lost`` additional lost lines round out the seed set.
    """

    entries = []
    idx = 0
    for _ in range(island):
        entries.append(
            _seed(KAM_CLASS_ISLAND_CHAIN, return_count=min_returns + 50, seed_index=idx)
        )
        idx += 1
    for _ in range(below_floor):
        entries.append(
            _seed(KAM_CLASS_CHAOTIC, return_count=min_returns - 1, seed_index=idx)
        )
        idx += 1
    clean = survived - island - below_floor
    if clean < 0:
        raise ValueError("island + below_floor exceeds survived")
    for _ in range(clean):
        entries.append(
            _seed(
                KAM_CLASS_INVARIANT_TORUS,
                return_count=min_returns + 100,
                seed_index=idx,
            )
        )
        idx += 1
    for _ in range(lost):
        entries.append(_seed(KAM_CLASS_LOST, return_count=0, seed_index=idx))
        idx += 1
    return entries


class DefaultOffByteIdentityTests(unittest.TestCase):
    def test_default_off_matches_legacy_across_payloads(self):
        payloads = [
            {"survival_fraction": 1.0, "broken": False},
            {"survival_fraction": 0.86, "broken": False},
            {"survival_fraction": 1.0, "broken": True},
            {"survival_fraction": 0.0, "broken": False},
            {},  # missing keys -> legacy defaults (0.0, False)
            {"broken": True},
            {"survival_fraction": 0.5},
        ]
        for payload in payloads:
            for threshold in (0.0, 0.25, 0.5, 0.86, 1.0):
                with self.subTest(payload=payload, threshold=threshold):
                    self.assertEqual(
                        topology_tier_passed(
                            payload, survival_threshold=threshold
                        ),
                        _legacy_topology_tier_passed(
                            payload, survival_threshold=threshold
                        ),
                    )

    def test_default_off_does_not_touch_wba_data(self):
        # No classification payload present at all: default-OFF must not raise.
        result = {"survival_fraction": 1.0, "broken": False}
        self.assertTrue(topology_tier_passed(result, survival_threshold=1.0))
        verdict = topology_tier_verdict(result, survival_threshold=1.0)
        self.assertTrue(verdict["passed"])
        self.assertFalse(verdict["island_gate_enabled"])
        self.assertFalse(verdict["classifiability_floor_enabled"])
        self.assertIsNone(verdict["island_chain_count"])
        self.assertIsNone(verdict["classifiable_fraction"])

    def test_returns_are_bool(self):
        self.assertIs(
            topology_tier_passed(
                {"survival_fraction": 1.0, "broken": False},
                survival_threshold=1.0,
            ),
            True,
        )


class IslandGateTests(unittest.TestCase):
    def _result(self, **kwargs):
        classifications = _build_classifications(**kwargs)
        return {
            "survival_fraction": kwargs["survived"]
            / float(kwargs["survived"] + kwargs["lost"]),
            "broken": False,
            WBA_SEED_CLASSIFICATIONS_KEY: classifications,
        }

    def test_one_island_fails_with_default_cap(self):
        result = self._result(survived=50, lost=0, island=1, below_floor=0)
        self.assertEqual(DEFAULT_N_ISLAND_MAX, 0)
        self.assertFalse(
            topology_tier_passed(
                result,
                survival_threshold=1.0,
                require_island_gate=True,
            )
        )
        verdict = topology_tier_verdict(
            result, survival_threshold=1.0, require_island_gate=True
        )
        self.assertTrue(verdict["survival_gate_passed"])
        self.assertFalse(verdict["island_gate_passed"])
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["island_chain_count"], 1)
        self.assertEqual(len(verdict["island_chain_seed_indices"]), 1)

    def test_island_within_cap_passes(self):
        result = self._result(survived=50, lost=0, island=1, below_floor=0)
        self.assertTrue(
            topology_tier_passed(
                result,
                survival_threshold=1.0,
                require_island_gate=True,
                n_island_max=1,
            )
        )

    def test_zero_islands_passes_island_gate(self):
        result = self._result(survived=50, lost=0, island=0, below_floor=0)
        verdict = topology_tier_verdict(
            result, survival_threshold=1.0, require_island_gate=True
        )
        self.assertTrue(verdict["passed"])
        self.assertEqual(verdict["island_chain_count"], 0)

    def test_island_gate_cannot_rescue_failed_survival(self):
        # 0 islands but survival below threshold -> still fails (tightening only).
        result = self._result(survived=43, lost=7, island=0, below_floor=0)
        self.assertFalse(
            topology_tier_passed(
                result,
                survival_threshold=1.0,
                require_island_gate=True,
            )
        )


class ClassifiabilityFloorTests(unittest.TestCase):
    def _result(self, **kwargs):
        classifications = _build_classifications(**kwargs)
        survived = kwargs["survived"]
        lost = kwargs["lost"]
        return {
            "survival_fraction": survived / float(survived + lost) if (survived + lost) else 0.0,
            "broken": False,
            WBA_SEED_CLASSIFICATIONS_KEY: classifications,
        }

    def test_floor_fails_when_too_few_classifiable(self):
        # 10 survived, 2 below the returns floor -> fraction 0.8 < 0.90 default.
        result = self._result(survived=10, lost=0, island=0, below_floor=2)
        self.assertFalse(
            topology_tier_passed(
                result,
                survival_threshold=1.0,
                require_classifiability_floor=True,
            )
        )
        verdict = topology_tier_verdict(
            result, survival_threshold=1.0, require_classifiability_floor=True
        )
        self.assertAlmostEqual(verdict["classifiable_fraction"], 0.8)
        self.assertFalse(verdict["classifiability_floor_passed"])
        self.assertFalse(verdict["passed"])

    def test_floor_passes_with_relaxed_fraction(self):
        result = self._result(survived=10, lost=0, island=0, below_floor=2)
        self.assertTrue(
            topology_tier_passed(
                result,
                survival_threshold=1.0,
                require_classifiability_floor=True,
                classifiable_fraction_min=0.75,
            )
        )

    def test_floor_fails_closed_when_no_survivors(self):
        result = self._result(survived=0, lost=10, island=0, below_floor=0)
        verdict = topology_tier_verdict(
            result,
            survival_threshold=0.0,
            require_classifiability_floor=True,
        )
        self.assertEqual(verdict["surviving_seed_count"], 0)
        self.assertIsNone(verdict["classifiable_fraction"])
        self.assertFalse(verdict["classifiability_floor_passed"])
        self.assertFalse(verdict["passed"])

    def test_return_count_exactly_at_floor_is_classifiable(self):
        result = {
            "survival_fraction": 1.0,
            "broken": False,
            WBA_SEED_CLASSIFICATIONS_KEY: [
                _seed(
                    KAM_CLASS_CHAOTIC,
                    return_count=DEFAULT_WBA_MIN_RETURNS,
                    seed_index=0,
                ),
                _seed(
                    KAM_CLASS_CHAOTIC,
                    return_count=DEFAULT_WBA_MIN_RETURNS - 1,
                    seed_index=1,
                ),
            ],
        }
        verdict = topology_tier_verdict(
            result, survival_threshold=1.0, require_classifiability_floor=True
        )
        self.assertEqual(verdict["surviving_seed_count"], 2)
        self.assertEqual(verdict["classifiable_seed_count"], 1)


class LoudRaiseTests(unittest.TestCase):
    def test_island_gate_without_classifications_raises(self):
        result = {"survival_fraction": 1.0, "broken": False}
        with self.assertRaises(ValueError):
            topology_tier_passed(
                result, survival_threshold=1.0, require_island_gate=True
            )

    def test_classifiability_floor_without_classifications_raises(self):
        result = {"survival_fraction": 1.0, "broken": False}
        with self.assertRaises(ValueError):
            topology_tier_passed(
                result,
                survival_threshold=1.0,
                require_classifiability_floor=True,
            )

    def test_none_classifications_payload_raises(self):
        result = {
            "survival_fraction": 1.0,
            "broken": False,
            WBA_SEED_CLASSIFICATIONS_KEY: None,
        }
        with self.assertRaises(ValueError):
            topology_tier_verdict(
                result, survival_threshold=1.0, require_island_gate=True
            )

    def test_missing_classification_field_raises(self):
        result = {
            "survival_fraction": 1.0,
            "broken": False,
            WBA_SEED_CLASSIFICATIONS_KEY: [{"seed_index": 0, "return_count": 300}],
        }
        with self.assertRaises(ValueError):
            topology_tier_verdict(
                result, survival_threshold=1.0, require_island_gate=True
            )

    def test_missing_return_count_field_raises_under_floor(self):
        result = {
            "survival_fraction": 1.0,
            "broken": False,
            WBA_SEED_CLASSIFICATIONS_KEY: [
                {"seed_index": 0, "classification": KAM_CLASS_CHAOTIC}
            ],
        }
        with self.assertRaises(ValueError):
            topology_tier_verdict(
                result,
                survival_threshold=1.0,
                require_classifiability_floor=True,
            )


class PayloadSurfacingTests(unittest.TestCase):
    def test_43_survived_0_island_classifiable_surfaced(self):
        # 50 seeds: 43 survived (7 lost), 0 island, 3 survivors below the floor
        # -> 40/43 classifiable. Report line: "43/50 survived, 0 island-chain,
        # 40/43 classifiable".
        result = {
            "survival_fraction": 0.86,
            "broken": False,
            WBA_SEED_CLASSIFICATIONS_KEY: _build_classifications(
                survived=43, lost=7, island=0, below_floor=3
            ),
        }
        verdict = topology_tier_verdict(
            result,
            survival_threshold=0.80,
            require_island_gate=True,
            require_classifiability_floor=True,
            classifiable_fraction_min=0.90,
        )
        self.assertEqual(verdict["seed_count"], 50)
        self.assertEqual(verdict["surviving_seed_count"], 43)
        self.assertEqual(verdict["island_chain_count"], 0)
        self.assertEqual(verdict["classifiable_seed_count"], 40)
        self.assertAlmostEqual(verdict["classifiable_fraction"], 40.0 / 43.0)
        self.assertTrue(verdict["island_gate_passed"])
        # 40/43 = 0.930 >= 0.90 -> floor passes; survival 0.86 >= 0.80 -> pass.
        self.assertTrue(verdict["classifiability_floor_passed"])
        self.assertTrue(verdict["passed"])

    def test_island_verdict_counts_direct(self):
        counts = island_verdict_counts(
            _build_classifications(survived=43, lost=7, island=2, below_floor=3),
            min_returns=DEFAULT_WBA_MIN_RETURNS,
        )
        self.assertEqual(counts["seed_count"], 50)
        self.assertEqual(counts["surviving_seed_count"], 43)
        self.assertEqual(counts["island_chain_count"], 2)
        # island lines clear the returns floor here -> classifiable = 43 - 3.
        self.assertEqual(counts["classifiable_seed_count"], 40)


if __name__ == "__main__":
    unittest.main()
