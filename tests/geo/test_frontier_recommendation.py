import unittest

from geo._frontier_test_helpers import (
    load_frontier_archive_module,
    load_frontier_recommendation_module,
    make_frontier_archive_member,
)


class FrontierRecommendationTests(unittest.TestCase):
    def setUp(self):
        self.archive_module = load_frontier_archive_module()
        self.recommendation_module = load_frontier_recommendation_module()

    def _assert_recommended(
        self,
        recommendation,
        *,
        member_id: str,
        policy_name: str | None = None,
    ) -> None:
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation["recommended_member"].member_id, member_id)
        if policy_name is not None:
            self.assertEqual(recommendation["policy_name"], policy_name)

    def _member(self, *, member_id="campaign:lane_01", **overrides):
        values = {
            "iota": 0.17,
            "volume": 0.11,
            "qa_error": 0.010,
            "boozer_residual": 0.007,
        }
        values.update(overrides)
        return make_frontier_archive_member(
            self.archive_module,
            member_id=member_id,
            **values,
        )

    def test_recommend_empty_archive_returns_none(self):
        self.assertIsNone(self.recommendation_module.recommend_frontier_member([]))

    def test_recommend_single_member_archive_returns_that_member(self):
        member = self._member()

        recommendation = self.recommendation_module.recommend_frontier_member([member])

        self._assert_recommended(recommendation, member_id="campaign:lane_01")
        self.assertFalse(recommendation["gate_fallback"])

    def test_recommend_tiebreaker_is_member_id_lex_ordered(self):
        member_b = self._member(member_id="campaign:lane_02")
        member_a = self._member()

        recommendation = self.recommendation_module.recommend_frontier_member(
            [member_b, member_a],
            policy_name="balanced",
        )

        self._assert_recommended(recommendation, member_id="campaign:lane_01")

    def test_recommend_with_no_safe_members_returns_none_for_strict_gate(self):
        lower_volume = self._member(
            iota=0.18,
            hardware_constraints_ok=False,
        )
        higher_volume = self._member(
            member_id="campaign:lane_02",
            volume=0.12,
            qa_error=0.011,
            boozer_residual=0.0075,
            hardware_constraints_ok=False,
        )

        recommendation = self.recommendation_module.recommend_frontier_member(
            [lower_volume, higher_volume],
            policy_name="max_volume_under_safe_hardware",
        )

        # Hardware-safe gate is strict (missing_is_eligible=False); no member
        # passes, so recommendation must be None; no silent bypass to unsafe
        # members.
        self.assertIsNone(recommendation)

    def test_recommend_with_all_explicitly_unsafe_boozer_returns_none(self):
        unsafe_member_a = self._member(
            iota=0.20,
            qa_error=0.011,
            frontier_trust_ok=False,
        )
        unsafe_member_b = self._member(
            member_id="campaign:lane_02",
            iota=0.18,
            qa_error=0.012,
            frontier_trust_ok=False,
        )

        recommendation = self.recommendation_module.recommend_frontier_member(
            [unsafe_member_a, unsafe_member_b],
            policy_name="max_iota_under_safe_boozer",
        )

        # The permissive Boozer gate only rescues missing legacy metadata.
        # Explicit False is a certified unsafe signal, not a fallback path.
        self.assertIsNone(recommendation)

    def test_recommend_with_missing_boozer_metadata_uses_permissive_gate(self):
        missing_metadata_member = self._member(
            iota=0.20,
            qa_error=0.011,
            frontier_trust_ok=None,
        )
        unsafe_member = self._member(
            member_id="campaign:lane_02",
            iota=0.18,
            qa_error=0.012,
            frontier_trust_ok=False,
        )

        recommendation = self.recommendation_module.recommend_frontier_member(
            [unsafe_member, missing_metadata_member],
            policy_name="max_iota_under_safe_boozer",
        )

        self._assert_recommended(recommendation, member_id="campaign:lane_01")
        self.assertFalse(recommendation["gate_fallback"])
        self.assertEqual(
            recommendation["policy_inputs"]["eligible_member_ids"],
            ["campaign:lane_01"],
        )

    def test_recommend_frontier_member_policy_cases(self):
        cases = [
            (
                "balanced",
                "campaign:lane_01",
                [
                    self._member(member_id="campaign:lane_02", iota=0.19, volume=0.095, qa_error=0.015, boozer_residual=0.011),
                    self._member(iota=0.165, volume=0.108, qa_error=0.011, boozer_residual=0.0075),
                ],
                None,
            ),
            (
                "max_iota_under_safe_boozer",
                "campaign:lane_01",
                [
                    self._member(iota=0.17, boozer_residual=0.0065),
                    self._member(member_id="campaign:lane_01", iota=0.19, qa_error=0.012),
                ],
                True,
            ),
            (
                "max_volume_under_safe_hardware",
                "campaign:lane_02",
                [
                    self._member(iota=0.19, volume=0.10, qa_error=0.011),
                    self._member(member_id="campaign:lane_02", volume=0.12, qa_error=0.012),
                ],
                False,
            ),
            (
                "closest_to_seed",
                "campaign:lane_02",
                [
                    self._member(iota=0.18, qa_error=0.011),
                    self._member(member_id="campaign:lane_02", iota=0.16, volume=0.102),
                ],
                None,
            ),
            (
                "max_iota_under_safe_boozer",
                "campaign:lane_01",
                [
                    self._member(iota=0.18, qa_error=0.011, frontier_trust_ok=True),
                    self._member(member_id="campaign:lane_01", iota=0.20, frontier_trust_ok=None),
                ],
                True,
            ),
            (
                "max_volume_under_safe_hardware",
                "campaign:lane_02",
                [
                    self._member(iota=0.19, volume=0.13, hardware_constraints_ok=None),
                    self._member(member_id="campaign:lane_02", hardware_constraints_ok=True),
                ],
                False,
            ),
        ]
        for policy_name, member_id, members, gate_missing_is_eligible in cases:
            with self.subTest(policy_name=policy_name, member_id=member_id):
                recommendation = self.recommendation_module.recommend_frontier_member(
                    members,
                    policy_name=policy_name,
                )
                self._assert_recommended(
                    recommendation,
                    member_id=member_id,
                    policy_name=policy_name,
                )
                if gate_missing_is_eligible is not None:
                    self.assertEqual(
                        recommendation["policy_inputs"]["gate_missing_is_eligible"],
                        gate_missing_is_eligible,
                    )

    def test_recommend_frontier_member_balanced_uses_fixed_ideal_nadir_normalization(self):
        aggressive_iota_tradeoff = self._member(
            iota=0.17,
            volume=0.10,
            qa_error=0.012,
            boozer_residual=0.012,
            reference_metrics={
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
        )
        near_seed_member = self._member(
            member_id="campaign:lane_02",
            iota=0.15,
            volume=0.10,
            qa_error=0.012,
            boozer_residual=0.008,
            reference_metrics={
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
        )
        fixed_ideal_nadir_normalization = {
            "kind": "fixed_ideal_nadir_span_with_floor",
            "reference_metrics": {
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
            "ideal_metrics": {
                "iota": 0.30,
                "volume": 0.12,
                "qa_error": 0.008,
                "boozer_residual": 0.0,
            },
            "nadir_metrics": {
                "iota": 0.10,
                "volume": 0.08,
                "qa_error": 0.020,
                "boozer_residual": 0.20,
            },
        }

        default_recommendation = self.recommendation_module.recommend_frontier_member(
            [aggressive_iota_tradeoff, near_seed_member],
            policy_name="balanced",
        )
        normalized_recommendation = self.recommendation_module.recommend_frontier_member(
            [aggressive_iota_tradeoff, near_seed_member],
            policy_name="balanced",
            pareto_objective_normalization=fixed_ideal_nadir_normalization,
        )

        self._assert_recommended(default_recommendation, member_id="campaign:lane_02")
        self._assert_recommended(normalized_recommendation, member_id="campaign:lane_01")
