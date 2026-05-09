import math
import unittest
import warnings
from unittest.mock import patch

from geo._frontier_test_helpers import (
    load_frontier_archive_module,
    load_frontier_dominance_module,
    load_frontier_progress_state_module,
    load_frontier_reporting_module,
    make_frontier_archive_member,
)


class FrontierArchiveTests(unittest.TestCase):
    HYPERVOLUME_REFERENCE = {
        "iota": 0.15,
        "volume": 0.10,
        "qa_error": 0.012,
        "boozer_residual": 0.008,
    }

    def _make_completed_payload(
        self,
        *,
        qa_error: float = 0.011,
        boozer_residual: float = 0.007,
    ) -> dict[str, object]:
        return {
            "result_source": "final",
            "results_path": "/tmp/lane_safe_iota/results.json",
            "results": {
                "FINAL_IOTA": 0.18,
                "FINAL_VOLUME": 0.11,
                "NONQS_RATIO": qa_error,
                "BOOZER_RESIDUAL": boozer_residual,
                "FINAL_FEASIBILITY_OK": True,
                "HARDWARE_CONSTRAINTS_OK": True,
                "FINAL_TOPOLOGY_GATE_SUCCESS": True,
                "FRONTIER_TRUST_OK": True,
                "FRONTIER_REFERENCE_IOTA": 0.15,
                "FRONTIER_REFERENCE_VOLUME": 0.10,
                "FRONTIER_REFERENCE_QA": 0.012,
                "FRONTIER_REFERENCE_BOOZER": 0.008,
                "FRONTIER_RANK_OBJECTIVE_J": -1.0,
                "OPTIMIZER_SUCCESS": True,
                "TERMINATION_MESSAGE": "ok",
            },
        }

    def _make_hypervolume_members(self, archive_module):
        return [
            self._member(
                archive_module,
                member_id="campaign:lane_01",
                iota=0.20,
                volume=0.11,
                qa_error=0.010,
                boozer_residual=0.006,
                soft_search_score=-1.0,
            ),
            self._member(
                archive_module,
                member_id="campaign:lane_02",
                iota=0.17,
                volume=0.13,
                qa_error=0.010,
                boozer_residual=0.006,
                soft_search_score=-1.1,
            ),
        ]

    def _member(
        self,
        archive_module,
        *,
        member_id="campaign:lane_01",
        iota=0.18,
        volume=0.11,
        qa_error=0.010,
        boozer_residual=0.007,
        soft_search_score=-1.0,
        reference_metrics=None,
    ):
        return make_frontier_archive_member(
            archive_module,
            member_id=member_id,
            iota=iota,
            volume=volume,
            qa_error=qa_error,
            boozer_residual=boozer_residual,
            soft_search_score=soft_search_score,
            reference_metrics=reference_metrics,
        )

    def _lane_contract(self, progress_state_module, lane_id):
        return progress_state_module.build_frontier_lane_contract(
            campaign_id="campaign",
            lane_id=lane_id,
            engine="multilane_local",
            scalarization_type="weight_schedule_v1",
            scalarization_params={"iota_share": 0.5, "volume_share": 0.5},
            constraint_mode="frontier_v2_single_lane_contract",
            warm_start_source="seed.json",
            optimizer_budget=10,
            rng_seed=0,
            rerun_contract={},
        )

    def _lane_record(self, progress_state_module, lane_id, member, **overrides):
        values = {
            "command": ["python"],
            "weights": {"iotas_weight": 1.0, "frontier_volume_weight": 1.0, "res_weight": 1.0},
            "lane_budget": 10,
            "status": "completed",
            "result_source": "final",
            "success": True,
            "archive_state": member.archive_state,
            "archive_member": member,
            "results_path": f"/tmp/{lane_id}.json",
            "results": {},
        }
        values.update(overrides)
        return progress_state_module.build_frontier_lane_record(
            self._lane_contract(progress_state_module, lane_id),
            **values,
        )

    def test_dominates_respects_objective_directions_and_invariants(self):
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

        self.assertFalse(module.dominates(better, better))
        self.assertTrue(module.dominates(better, worse))
        self.assertFalse(module.dominates(worse, better))
        self.assertFalse(module.dominates(better, worse) and module.dominates(worse, better))

    def test_archive_best_by_metric_uses_objective_directions(self):
        archive_module = load_frontier_archive_module()
        lane_02 = self._member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.20,
            volume=0.11,
            qa_error=0.010,
            boozer_residual=0.008,
            soft_search_score=-1.0,
        )
        lane_01 = self._member(
            archive_module,
            member_id="campaign:lane_01",
            iota=0.20,
            volume=0.12,
            qa_error=0.011,
            boozer_residual=0.007,
            soft_search_score=-1.1,
        )

        best_by_metric = archive_module.archive_best_by_metric([lane_02, lane_01])

        self.assertEqual(best_by_metric["iota"]["member_id"], "campaign:lane_01")
        self.assertEqual(best_by_metric["volume"]["member_id"], "campaign:lane_01")
        self.assertEqual(best_by_metric["qa_error"]["member_id"], "campaign:lane_02")
        self.assertEqual(best_by_metric["boozer_residual"]["member_id"], "campaign:lane_01")

    def test_dominates_rejects_nonfinite_metric_values(self):
        module = load_frontier_dominance_module()
        valid = {
            "iota": 0.18,
            "volume": 0.11,
            "qa_error": 0.010,
            "boozer_residual": 0.007,
        }
        invalid = dict(valid)
        invalid["qa_error"] = math.nan

        with self.assertRaisesRegex(ValueError, "Non-finite Pareto metric"):
            module.dominates(invalid, valid)
        with self.assertRaisesRegex(ValueError, "Non-finite Pareto metric"):
            module.dominates(valid, invalid)

    def test_archive_member_from_json_dict_filters_nonfinite_metrics(self):
        archive_module = load_frontier_archive_module()
        payload = {
            "member_id": "campaign:lane_01",
            "lane_id": "lane_01",
            "campaign_id": "campaign",
            "archive_state": archive_module.FRONTIER_ARCHIVE_STATE_CERTIFIED,
            "objective_metrics": {
                "iota": 0.18,
                "volume": float("nan"),
                "qa_error": float("inf"),
                "boozer_residual": True,
            },
            "reference_metrics": {
                "iota": 0.17,
                "volume": float("nan"),
                "qa_error": 0.011,
                "boozer_residual": False,
            },
        }
        member = archive_module.frontier_archive_member_from_json_dict(payload)
        self.assertEqual(member.objective_metrics["iota"], 0.18)
        self.assertIsNone(member.objective_metrics["volume"])
        self.assertIsNone(member.objective_metrics["qa_error"])
        self.assertIsNone(member.objective_metrics["boozer_residual"])
        self.assertIsNone(member.reference_metrics["volume"])
        self.assertIsNone(member.reference_metrics["boozer_residual"])

    def test_objective_metric_scale_handles_degenerate_ideal_nadir_axis(self):
        module = load_frontier_dominance_module()

        scale = module.objective_metric_scale(
            "iota",
            None,
            pareto_objective_normalization={
                "kind": module.PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR,
                "ideal_metrics": {"iota": 0.17},
                "nadir_metrics": {"iota": 0.17},
            },
        )

        self.assertEqual(scale, 0.05)

    def test_update_frontier_archive_actions(self):
        archive_module = load_frontier_archive_module()
        cases = [
            (
                "removes_dominated_member",
                {"iota": 0.17, "volume": 0.10, "qa_error": 0.011, "boozer_residual": 0.008},
                {"soft_search_score": -1.5},
                {"action": "inserted", "dominated_members": ["campaign:lane_01"]},
                ["campaign:lane_02"],
            ),
            (
                "skips_dominated_candidate",
                {},
                {
                    "iota": 0.17,
                    "volume": 0.10,
                    "qa_error": 0.011,
                    "boozer_residual": 0.008,
                    "soft_search_score": -1.5,
                },
                {"action": "dominated", "dominated_by": ["campaign:lane_01"]},
                ["campaign:lane_01"],
            ),
            (
                "prefers_lower_search_score_for_duplicate",
                {},
                {
                    "iota": 0.1800001,
                    "volume": 0.1100001,
                    "qa_error": 0.0100001,
                    "boozer_residual": 0.0070000001,
                    "soft_search_score": -1.2,
                },
                {"action": "duplicate_replaced", "replaced_member_id": "campaign:lane_01"},
                ["campaign:lane_02"],
            ),
            (
                "keeps_dominating_duplicate_candidate",
                {},
                {
                    "iota": 0.1800002,
                    "volume": 0.1100002,
                    "qa_error": 0.0099999,
                    "boozer_residual": 0.0069999,
                    "soft_search_score": -0.9,
                },
                {
                    "action": "duplicate_replaced",
                    "replaced_member_id": "campaign:lane_01",
                    "dominated_members": ["campaign:lane_01"],
                },
                ["campaign:lane_02"],
            ),
        ]
        for name, incumbent_kwargs, candidate_kwargs, expected_update, expected_ids in cases:
            with self.subTest(name=name):
                updated_members, update = archive_module.update_frontier_archive(
                    [self._member(archive_module, **incumbent_kwargs)],
                    self._member(
                        archive_module,
                        member_id="campaign:lane_02",
                        **candidate_kwargs,
                    ),
                )
                for key, expected in expected_update.items():
                    self.assertEqual(update[key], expected)
                self.assertEqual([member.member_id for member in updated_members], expected_ids)

    def test_update_frontier_archive_uses_fixed_ideal_nadir_normalization_for_duplicates(self):
        archive_module = load_frontier_archive_module()
        dominance_module = load_frontier_dominance_module()
        reference_metrics = {"iota": 0.15, "volume": 0.10, "qa_error": 0.012, "boozer_residual": 0.008}

        incumbent = self._member(
            archive_module,
            iota=0.15,
            volume=0.10,
            qa_error=0.012,
            boozer_residual=0.008,
            reference_metrics=reference_metrics,
        )
        candidate = self._member(
            archive_module,
            member_id="campaign:lane_02",
            iota=0.156,
            volume=0.09995,
            qa_error=0.012,
            boozer_residual=0.008,
            soft_search_score=-1.5,
            reference_metrics=reference_metrics,
        )
        fixed_ideal_nadir_normalization = {
            "schema_version": dominance_module.PARETO_OBJECTIVE_NORMALIZATION_SCHEMA_VERSION,
            "kind": dominance_module.PARETO_OBJECTIVE_NORMALIZATION_KIND_IDEAL_NADIR,
            "distance_metric": "euclidean",
            "reference_metrics": reference_metrics,
            "ideal_metrics": {"iota": 0.30, "volume": 0.12, "qa_error": 0.008, "boozer_residual": 0.004},
            "nadir_metrics": {"iota": 0.10, "volume": 0.08, "qa_error": 0.020, "boozer_residual": 0.012},
            "metric_rules": dict(dominance_module.PARETO_OBJECTIVE_NORMALIZATION_IDEAL_NADIR_RULES),
        }

        default_members, default_update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
        )
        normalized_members, normalized_update = archive_module.update_frontier_archive(
            [incumbent],
            candidate,
            pareto_objective_normalization=fixed_ideal_nadir_normalization,
        )

        self.assertEqual(default_update["action"], "inserted")
        self.assertEqual(len(default_members), 2)
        self.assertEqual(normalized_update["action"], "duplicate_replaced")
        self.assertEqual(normalized_update["replaced_member_id"], incumbent.member_id)
        self.assertEqual(len(normalized_members), 1)

    def test_replay_archive_from_lane_records_reapplies_dominance_updates(self):
        archive_module = load_frontier_archive_module()
        progress_state_module = load_frontier_progress_state_module()

        incumbent = self._member(
            archive_module,
            iota=0.17,
            volume=0.10,
            qa_error=0.011,
            boozer_residual=0.008,
        )
        candidate = self._member(
            archive_module,
            member_id="campaign:lane_02",
            soft_search_score=-1.5,
        )

        archive_members, incumbent_update = archive_module.update_frontier_archive(
            [],
            incumbent,
        )
        archive_members, candidate_update = archive_module.update_frontier_archive(
            archive_members,
            candidate,
        )

        lane_records = []
        for lane_id, member, archive_update in (
            ("lane_01", incumbent, incumbent_update),
            ("lane_02", candidate, candidate_update),
        ):
            lane_records.append(
                self._lane_record(
                    progress_state_module,
                    lane_id,
                    member,
                    archive_update=archive_update,
                )
            )

        replayed_members = progress_state_module.replay_archive_from_lane_records(lane_records)

        self.assertEqual([member.member_id for member in replayed_members], ["campaign:lane_02"])

    def test_finalize_archive_member_converts_provisional_member_to_canonical_final_member(self):
        archive_module = load_frontier_archive_module()

        provisional_member = archive_module.build_archive_member_from_results(
            campaign_id="campaign",
            lane_id="lane_01",
            payload=self._make_completed_payload(),
            rerun_contract={},
            archive_state=archive_module.FRONTIER_ARCHIVE_STATE_PROVISIONAL,
        )
        final_member = archive_module.finalize_archive_member(provisional_member)

        self.assertEqual(provisional_member.member_id, "campaign:lane_01:provisional")
        self.assertEqual(provisional_member.archive_state, archive_module.FRONTIER_ARCHIVE_STATE_PROVISIONAL)
        self.assertEqual(final_member.member_id, "campaign:lane_01")
        self.assertEqual(final_member.archive_state, archive_module.FRONTIER_ARCHIVE_STATE_CERTIFIED)

    def test_build_frontier_lane_record_tracks_provisional_and_certified_ids(self):
        archive_module = load_frontier_archive_module()
        progress_state_module = load_frontier_progress_state_module()
        provisional_member = archive_module.build_archive_member_from_results(
            campaign_id="campaign",
            lane_id="lane_01",
            payload=self._make_completed_payload(),
            rerun_contract={},
            archive_state=archive_module.FRONTIER_ARCHIVE_STATE_PROVISIONAL,
        )
        certified_member = archive_module.finalize_archive_member(provisional_member)

        lane_record = self._lane_record(
            progress_state_module,
            "lane_01",
            certified_member,
            provisional_archive_member=provisional_member,
            archive_update={"action": "inserted", "member_id": certified_member.member_id},
        )

        self.assertEqual(lane_record.provisional_member_ids, ["campaign:lane_01:provisional"])
        self.assertEqual(lane_record.certified_member_ids, ["campaign:lane_01"])
        self.assertTrue(lane_record.final_certified)

    def test_hypervolume_contributions_are_computed_from_reference_point(self):
        archive_module = load_frontier_archive_module()

        member_a, member_b = self._make_hypervolume_members(archive_module)

        annotated_members = archive_module.annotate_hypervolume_contributions(
            [member_a, member_b],
            hypervolume_reference=self.HYPERVOLUME_REFERENCE,
        )
        hypervolume_total = archive_module.frontier_archive_hypervolume(
            annotated_members,
            hypervolume_reference=self.HYPERVOLUME_REFERENCE,
        )

        self.assertIsNotNone(hypervolume_total)
        self.assertAlmostEqual(hypervolume_total, 3.6e-9)
        contributions = {member.member_id: member.hypervolume_contribution for member in annotated_members}
        self.assertAlmostEqual(contributions["campaign:lane_01"], 1.2e-9)
        self.assertAlmostEqual(contributions["campaign:lane_02"], 1.6e-9)

        serialized = archive_module.serialize_frontier_archive(
            [member_a, member_b],
            hypervolume_reference=self.HYPERVOLUME_REFERENCE,
        )
        self.assertAlmostEqual(serialized["hypervolume_total"], 3.6e-9)
        self.assertEqual(serialized["hypervolume_reference"], self.HYPERVOLUME_REFERENCE)

    def test_hypervolume_permutation_invariance_and_cache_key(self):
        archive_module = load_frontier_archive_module()
        members = self._make_hypervolume_members(archive_module)
        archive_module._hypervolume_cached.cache_clear()

        original = archive_module.frontier_archive_hypervolume(
            members,
            hypervolume_reference=self.HYPERVOLUME_REFERENCE,
        )
        reversed_order = archive_module.frontier_archive_hypervolume(
            list(reversed(members)),
            hypervolume_reference=self.HYPERVOLUME_REFERENCE,
        )

        self.assertEqual(original, reversed_order)
        cache_info = archive_module._hypervolume_cached.cache_info()
        self.assertEqual(cache_info.misses, 1)
        self.assertEqual(cache_info.hits, 1)

    def test_hypervolume_cache_reuses_final_archive_across_reporting_paths(self):
        archive_module = load_frontier_archive_module()
        reporting_module = load_frontier_reporting_module()
        members = self._make_hypervolume_members(archive_module)
        archive_module._hypervolume_cached.cache_clear()
        uncached_inputs = []

        def counted_uncached(boxes, reference_tuple):
            uncached_inputs.append((boxes, reference_tuple))
            return archive_module._union_hypervolume(list(boxes))

        final_archive_key = archive_module._hypervolume_cache_key(
            members,
            hypervolume_reference=self.HYPERVOLUME_REFERENCE,
        )
        lane_records = [{"lane_id": member.lane_id, "status": "completed", "archive_member": member.to_json_dict()} for member in members]
        with patch.object(
            archive_module,
            "_hypervolume_uncached",
            side_effect=counted_uncached,
        ):
            archive_module.annotate_hypervolume_contributions(
                members,
                hypervolume_reference=self.HYPERVOLUME_REFERENCE,
            )
            archive_module.serialize_frontier_archive(
                members,
                hypervolume_reference=self.HYPERVOLUME_REFERENCE,
            )
            archive_module.frontier_archive_hypervolume(
                members,
                hypervolume_reference=self.HYPERVOLUME_REFERENCE,
            )
            reporting_module.build_frontier_hypervolume_history(
                lane_records,
                hypervolume_reference=self.HYPERVOLUME_REFERENCE,
            )

        self.assertEqual(len(uncached_inputs), len(set(uncached_inputs)))
        self.assertEqual(uncached_inputs.count(final_archive_key), 1)

    def test_hypervolume_monotone_under_dominated_member(self):
        archive_module = load_frontier_archive_module()
        dominant = self._member(
            archive_module,
            iota=0.20,
            volume=0.12,
            boozer_residual=0.006,
        )
        dominated = self._member(
            archive_module,
            member_id="campaign:lane_02",
            qa_error=0.011,
            soft_search_score=-1.1,
        )

        base = archive_module.frontier_archive_hypervolume(
            [dominant],
            hypervolume_reference=self.HYPERVOLUME_REFERENCE,
        )
        with_dominated = archive_module.frontier_archive_hypervolume(
            [dominant, dominated],
            hypervolume_reference=self.HYPERVOLUME_REFERENCE,
        )

        self.assertEqual(base, with_dominated)

    def test_hypervolume_reductions(self):
        archive_module = load_frontier_archive_module()

        self.assertEqual(archive_module._union_hypervolume([(2.0,), (1.0,)]), 2.0)
        self.assertEqual(archive_module._union_hypervolume([(2.0, 1.0), (1.0, 3.0)]), 4.0)

    def test_hypervolume_reference_warns_when_not_nadir(self):
        archive_module = load_frontier_archive_module()
        member = self._member(
            archive_module,
            iota=0.20,
            boozer_residual=0.006,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            archive_module.resolve_hypervolume_reference(
                reference_spec=(
                    "iota=0.21,volume=0.10,qa_error=0.012,boozer_residual=0.008"
                ),
                members=[member],
            )

        self.assertEqual(len(caught), 1)
        self.assertIn("not a nadir", str(caught[0].message))

    def test_build_archive_member_from_results_applies_epsilon_certification_contract(self):
        archive_module = load_frontier_archive_module()

        member = archive_module.build_archive_member_from_results(
            campaign_id="campaign",
            lane_id="lane_safe_iota",
            payload=self._make_completed_payload(qa_error=0.013),
            rerun_contract={
                "scalarization_type": "epsilon_constraint_sweep_v1",
                "scalarization_params": {
                    "epsilon_constraint_qa_max": 0.012,
                    "epsilon_constraint_boozer_max": 0.008,
                },
            },
        )

        self.assertFalse(member.hard_certification_ok)
        self.assertFalse(member.constraint_metrics["epsilon_constraints_ok"])
        self.assertAlmostEqual(member.constraint_metrics["epsilon_constraint_violations"]["qa_error"], 0.001)

    def test_epsilon_certifier_allows_single_metric_threshold_contracts(self):
        archive_module = load_frontier_archive_module()

        cases = [
            ({"epsilon_constraint_qa_max": 0.012}, self._make_completed_payload(qa_error=0.013), "qa_error", 0.001),
            ({"epsilon_constraint_boozer_max": 0.008}, self._make_completed_payload(boozer_residual=0.009), "boozer_residual", 0.001),
        ]

        for scalarization_params, payload, metric_name, expected_excess in cases:
            with self.subTest(metric_name=metric_name):
                member = archive_module.build_archive_member_from_results(
                    campaign_id="campaign",
                    lane_id="lane_safe_iota",
                    payload=payload,
                    rerun_contract={
                        "scalarization_type": "epsilon_constraint_sweep_v1",
                        "scalarization_params": scalarization_params,
                    },
                )

                self.assertFalse(member.hard_certification_ok)
                self.assertFalse(member.constraint_metrics["epsilon_constraints_ok"])
                self.assertEqual(set(member.constraint_metrics["epsilon_constraint_violations"]), {metric_name})
                self.assertAlmostEqual(member.constraint_metrics["epsilon_constraint_violations"][metric_name], expected_excess)

    def test_epsilon_certifier_silent_pass_on_unknown_scalarization_type(self):
        archive_module = load_frontier_archive_module()

        member = archive_module.build_archive_member_from_results(
            campaign_id="campaign",
            lane_id="lane_weighted",
            payload=self._make_completed_payload(qa_error=0.9, boozer_residual=0.9),
            rerun_contract={"scalarization_type": "weight_schedule_v1", "scalarization_params": {}},
        )

        self.assertTrue(member.constraint_metrics["epsilon_constraints_ok"])
        self.assertEqual(member.constraint_metrics["epsilon_constraint_violations"], {})

    def test_epsilon_certifier_raises_on_missing_threshold_keys(self):
        archive_module = load_frontier_archive_module()

        with self.assertRaisesRegex(ValueError, "missing threshold key"):
            archive_module.build_archive_member_from_results(
                campaign_id="campaign",
                lane_id="lane_safe_iota",
                payload=self._make_completed_payload(),
                rerun_contract={
                    "scalarization_type": "epsilon_constraint_sweep_v1",
                    "scalarization_params": {},
                },
            )

    def test_epsilon_certifier_uses_boundary_slack(self):
        archive_module = load_frontier_archive_module()
        rerun_contract = {
            "scalarization_type": "epsilon_constraint_sweep_v1",
            "scalarization_params": {
                "epsilon_constraint_qa_max": 0.012,
                "epsilon_constraint_boozer_max": 0.008,
            },
        }

        boundary_member = archive_module.build_archive_member_from_results(
            campaign_id="campaign",
            lane_id="lane_safe_iota",
            payload=self._make_completed_payload(qa_error=0.012 + 5.0e-13),
            rerun_contract=rerun_contract,
        )
        outside_member = archive_module.build_archive_member_from_results(
            campaign_id="campaign",
            lane_id="lane_safe_iota",
            payload=self._make_completed_payload(qa_error=0.012 + 2.0e-12),
            rerun_contract=rerun_contract,
        )

        self.assertTrue(boundary_member.hard_certification_ok)
        self.assertFalse(outside_member.hard_certification_ok)
