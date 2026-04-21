import importlib
import sys
import unittest
from pathlib import Path


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def load_frontier_archive_module():
    return importlib.import_module("banana_opt.frontier_archive")


def load_frontier_recommendation_module():
    return importlib.import_module("banana_opt.frontier_recommendation")


class FrontierRecommendationTests(unittest.TestCase):
    def _make_member(
        self,
        archive_module,
        *,
        member_id: str,
        iota: float,
        volume: float,
        qa_error: float,
        boozer_residual: float,
        distance_from_seed: float,
        frontier_trust_ok: bool = True,
        hardware_constraints_ok: bool = True,
    ):
        return archive_module.FrontierArchiveMember(
            member_id=member_id,
            lane_id=member_id.split(":")[-1],
            campaign_id="campaign",
            archive_state="certified",
            dominance_signature={},
            objective_metrics={
                "iota": iota,
                "volume": volume,
                "qa_error": qa_error,
                "boozer_residual": boozer_residual,
            },
            reference_metrics={
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
            constraint_metrics={
                "frontier_trust_ok": frontier_trust_ok,
                "hardware_constraints_ok": hardware_constraints_ok,
            },
            hard_certification_ok=True,
            soft_search_score=-1.0,
            distance_from_seed=distance_from_seed,
            hypervolume_contribution=None,
            recommendation_flags={},
            rerun_contract={},
            result_source="final",
            results_path=f"/tmp/{member_id}.json",
            termination_reason="ok",
            success=True,
        )

    def test_recommend_frontier_member_max_iota_under_safe_boozer_prefers_highest_iota(self):
        archive_module = load_frontier_archive_module()
        recommendation_module = load_frontier_recommendation_module()

        safe_high_iota = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.19,
            volume=0.10,
            qa_error=0.012,
            boozer_residual=0.0078,
            distance_from_seed=0.40,
        )
        safer_lower_iota = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.17,
            volume=0.11,
            qa_error=0.010,
            boozer_residual=0.0065,
            distance_from_seed=0.20,
        )

        recommendation = recommendation_module.recommend_frontier_member(
            [safer_lower_iota, safe_high_iota],
            policy_name="max_iota_under_safe_boozer",
        )

        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        self.assertEqual(
            recommendation["recommended_member"].member_id,
            "campaign:lane_01",
        )
        self.assertEqual(
            recommendation["policy_name"],
            "max_iota_under_safe_boozer",
        )

    def test_recommend_frontier_member_max_volume_under_safe_hardware_prefers_highest_volume(self):
        archive_module = load_frontier_archive_module()
        recommendation_module = load_frontier_recommendation_module()

        lower_volume = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.19,
            volume=0.10,
            qa_error=0.011,
            boozer_residual=0.007,
            distance_from_seed=0.25,
        )
        higher_volume = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.17,
            volume=0.12,
            qa_error=0.012,
            boozer_residual=0.0075,
            distance_from_seed=0.30,
        )

        recommendation = recommendation_module.recommend_frontier_member(
            [lower_volume, higher_volume],
            policy_name="max_volume_under_safe_hardware",
        )

        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        self.assertEqual(
            recommendation["recommended_member"].member_id,
            "campaign:lane_02",
        )
        self.assertEqual(
            recommendation["policy_name"],
            "max_volume_under_safe_hardware",
        )

    def test_recommend_frontier_member_closest_to_seed_prefers_smallest_distance(self):
        archive_module = load_frontier_archive_module()
        recommendation_module = load_frontier_recommendation_module()

        farther = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.18,
            volume=0.11,
            qa_error=0.011,
            boozer_residual=0.007,
            distance_from_seed=0.30,
        )
        closer = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.16,
            volume=0.102,
            qa_error=0.0115,
            boozer_residual=0.0078,
            distance_from_seed=0.08,
        )

        recommendation = recommendation_module.recommend_frontier_member(
            [farther, closer],
            policy_name="closest_to_seed",
        )

        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        self.assertEqual(
            recommendation["recommended_member"].member_id,
            "campaign:lane_02",
        )
        self.assertEqual(recommendation["policy_name"], "closest_to_seed")
