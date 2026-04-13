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


def load_frontier_dominance_module():
    return importlib.import_module("banana_opt.frontier_dominance")


def load_frontier_recommendation_module():
    return importlib.import_module("banana_opt.frontier_recommendation")


class FrontierArchiveTests(unittest.TestCase):
    def _make_member(
        self,
        archive_module,
        *,
        member_id: str,
        iota: float,
        volume: float,
        qa_error: float,
        boozer_residual: float,
        soft_search_score: float,
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
            constraint_metrics={},
            hard_certification_ok=True,
            soft_search_score=soft_search_score,
            distance_from_seed=0.0,
            hypervolume_contribution=None,
            recommendation_flags={},
            rerun_contract={},
            result_source="final",
            results_path=f"/tmp/{member_id}.json",
            termination_reason="ok",
            success=True,
        )

    def test_dominates_respects_objective_directions(self):
        module = load_frontier_dominance_module()

        better = {
            "iota": 0.18,
            "volume": 0.11,
            "qa_error": 0.010,
            "boozer_residual": 0.007,
        }
        worse = {
            "iota": 0.17,
            "volume": 0.10,
            "qa_error": 0.011,
            "boozer_residual": 0.008,
        }

        self.assertTrue(module.dominates(better, worse))
        self.assertFalse(module.dominates(worse, better))

    def test_update_frontier_archive_removes_dominated_member(self):
        archive_module = load_frontier_archive_module()

        incumbent = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.17,
            volume=0.10,
            qa_error=0.011,
            boozer_residual=0.008,
            soft_search_score=-1.0,
        )
        candidate = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.18,
            volume=0.11,
            qa_error=0.010,
            boozer_residual=0.007,
            soft_search_score=-1.5,
        )

        updated_members, update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
        )

        self.assertEqual(update["action"], "inserted")
        self.assertEqual(update["dominated_members"], ["campaign:lane_01"])
        self.assertEqual([member.member_id for member in updated_members], ["campaign:lane_02"])

    def test_update_frontier_archive_skips_dominated_candidate(self):
        archive_module = load_frontier_archive_module()

        incumbent = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.18,
            volume=0.11,
            qa_error=0.010,
            boozer_residual=0.007,
            soft_search_score=-1.0,
        )
        candidate = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.17,
            volume=0.10,
            qa_error=0.011,
            boozer_residual=0.008,
            soft_search_score=-1.5,
        )

        updated_members, update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
        )

        self.assertEqual(update["action"], "dominated")
        self.assertEqual(update["dominated_by"], ["campaign:lane_01"])
        self.assertEqual([member.member_id for member in updated_members], ["campaign:lane_01"])

    def test_update_frontier_archive_prefers_lower_search_score_for_duplicate(self):
        archive_module = load_frontier_archive_module()

        incumbent = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.18,
            volume=0.11,
            qa_error=0.010,
            boozer_residual=0.007,
            soft_search_score=-1.0,
        )
        candidate = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.1800001,
            volume=0.1100001,
            qa_error=0.0100001,
            boozer_residual=0.0070000001,
            soft_search_score=-1.2,
        )

        updated_members, update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
        )

        self.assertEqual(update["action"], "duplicate_replaced")
        self.assertEqual(update["replaced_member_id"], "campaign:lane_01")
        self.assertEqual([member.member_id for member in updated_members], ["campaign:lane_02"])

    def test_update_frontier_archive_keeps_dominating_candidate_even_if_duplicate_score_is_worse(self):
        archive_module = load_frontier_archive_module()

        incumbent = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.18,
            volume=0.11,
            qa_error=0.0100,
            boozer_residual=0.0070,
            soft_search_score=-1.0,
        )
        candidate = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.1800002,
            volume=0.1100002,
            qa_error=0.0099999,
            boozer_residual=0.0069999,
            soft_search_score=-0.9,
        )

        updated_members, update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
        )

        self.assertEqual(update["action"], "duplicate_replaced")
        self.assertEqual(update["replaced_member_id"], "campaign:lane_01")
        self.assertEqual(update["dominated_members"], ["campaign:lane_01"])
        self.assertEqual([member.member_id for member in updated_members], ["campaign:lane_02"])

    def test_recommend_frontier_member_balanced_prefers_better_overall_tradeoff(self):
        archive_module = load_frontier_archive_module()
        recommendation_module = load_frontier_recommendation_module()

        balanced = self._make_member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.165,
            volume=0.108,
            qa_error=0.011,
            boozer_residual=0.0075,
            soft_search_score=-1.1,
        )
        iota_only = self._make_member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.19,
            volume=0.095,
            qa_error=0.015,
            boozer_residual=0.011,
            soft_search_score=-1.0,
        )

        recommendation = recommendation_module.recommend_frontier_member(
            [iota_only, balanced],
            policy_name="balanced",
        )

        self.assertIsNotNone(recommendation)
        assert recommendation is not None
        self.assertEqual(
            recommendation["recommended_member"].member_id,
            "campaign:lane_01",
        )
        self.assertEqual(recommendation["policy_name"], "balanced")
