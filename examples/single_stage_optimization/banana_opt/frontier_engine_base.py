from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping

from .frontier_contracts import (
    FRONTIER_ARCHIVE_STATE_PROVISIONAL,
    FRONTIER_CAMPAIGN_PROGRESS_SCHEMA_VERSION,
    FRONTIER_LANE_CONTRACT_SCHEMA_VERSION,
    FRONTIER_LANE_RECORD_SCHEMA_VERSION,
    validate_frontier_campaign_progress_payload,
    validate_frontier_lane_contract_payload,
    validate_frontier_lane_record_payload,
)
from .frontier_archive import (
    FrontierArchiveMember,
    build_archive_member_from_results,
    frontier_archive_member_from_json_dict,
    update_frontier_archive,
)


@dataclass(frozen=True)
class FrontierLaneContract:
    schema_version: str
    lane_id: str
    campaign_id: str
    engine: str
    reference_point: dict[str, float] | None
    scalarization_type: str
    scalarization_params: dict[str, float]
    constraint_mode: str
    warm_start_source: str | None
    optimizer_budget: int
    rng_seed: int
    rerun_contract: dict[str, object]

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        validate_frontier_lane_contract_payload(payload)
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> FrontierLaneContract:
        reference_point_payload = payload.get("reference_point")
        reference_point = None
        if isinstance(reference_point_payload, Mapping):
            reference_point = {
                str(key): float(value)
                for key, value in reference_point_payload.items()
            }
        scalarization_params_payload = payload.get("scalarization_params", {})
        rerun_contract_payload = payload.get("rerun_contract", {})
        return cls(
            schema_version=str(
                payload.get(
                    "schema_version",
                    FRONTIER_LANE_CONTRACT_SCHEMA_VERSION,
                )
            ),
            lane_id=str(payload["lane_id"]),
            campaign_id=str(payload["campaign_id"]),
            engine=str(payload["engine"]),
            reference_point=reference_point,
            scalarization_type=str(payload["scalarization_type"]),
            scalarization_params={
                str(key): float(value)
                for key, value in scalarization_params_payload.items()
            },
            constraint_mode=str(payload["constraint_mode"]),
            warm_start_source=None
            if payload.get("warm_start_source") is None
            else str(payload["warm_start_source"]),
            optimizer_budget=int(payload["optimizer_budget"]),
            rng_seed=int(payload["rng_seed"]),
            rerun_contract=dict(rerun_contract_payload),
        )


@dataclass(frozen=True)
class FrontierLaneRecord:
    schema_version: str
    lane_contract: FrontierLaneContract
    status: str
    command: list[str]
    weights: dict[str, float]
    lane_budget: int
    result_source: str | None
    termination_reason: str | None
    success: bool | None
    provisional_member_ids: list[str]
    certified_member_ids: list[str]
    final_certified: bool
    archive_state: str | None
    archive_member_id: str | None
    archive_member: dict[str, object] | None
    archive_update: dict[str, object] | None
    results_path: str | None
    results: dict[str, object] | None
    error_type: str | None
    error_message: str | None

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("lane_contract", None)
        payload.update(
            {
                "lane_id": self.lane_contract.lane_id,
                "campaign_id": self.lane_contract.campaign_id,
                "engine": self.lane_contract.engine,
                "reference_point": self.lane_contract.reference_point,
                "scalarization_type": self.lane_contract.scalarization_type,
                "scalarization_params": dict(
                    self.lane_contract.scalarization_params
                ),
                "constraint_mode": self.lane_contract.constraint_mode,
                "warm_start_source": self.lane_contract.warm_start_source,
                "optimizer_budget": self.lane_contract.optimizer_budget,
                "rng_seed": self.lane_contract.rng_seed,
                "rerun_contract": dict(self.lane_contract.rerun_contract),
            }
        )
        validate_frontier_lane_record_payload(payload)
        return payload

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> FrontierLaneRecord:
        lane_contract_payload = payload.get("lane_contract")
        if isinstance(lane_contract_payload, Mapping):
            lane_contract = FrontierLaneContract.from_json_dict(
                lane_contract_payload
            )
        else:
            lane_contract = FrontierLaneContract.from_json_dict(payload)
        return cls(
            schema_version=str(
                payload.get("schema_version", FRONTIER_LANE_RECORD_SCHEMA_VERSION)
            ),
            lane_contract=lane_contract,
            status=str(payload["status"]),
            command=[str(item) for item in payload.get("command", [])],
            weights={
                str(key): float(value)
                for key, value in payload.get("weights", {}).items()
            },
            lane_budget=int(payload["lane_budget"]),
            result_source=None
            if payload.get("result_source") is None
            else str(payload["result_source"]),
            termination_reason=None
            if payload.get("termination_reason") is None
            else str(payload["termination_reason"]),
            success=None
            if payload.get("success") is None
            else bool(payload["success"]),
            provisional_member_ids=[
                str(item) for item in payload.get("provisional_member_ids", [])
            ],
            certified_member_ids=[
                str(item) for item in payload.get("certified_member_ids", [])
            ],
            final_certified=bool(payload.get("final_certified", False)),
            archive_state=None
            if payload.get("archive_state") is None
            else str(payload["archive_state"]),
            archive_member_id=None
            if payload.get("archive_member_id") is None
            else str(payload["archive_member_id"]),
            archive_member=None
            if payload.get("archive_member") is None
            else dict(payload["archive_member"]),
            archive_update=None
            if payload.get("archive_update") is None
            else dict(payload["archive_update"]),
            results_path=None
            if payload.get("results_path") is None
            else str(payload["results_path"]),
            results=None
            if payload.get("results") is None
            else dict(payload["results"]),
            error_type=None
            if payload.get("error_type") is None
            else str(payload["error_type"]),
            error_message=None
            if payload.get("error_message") is None
            else str(payload["error_message"]),
        )


@dataclass(frozen=True)
class FrontierCampaignProgress:
    schema_version: str
    campaign_id: str
    frontier_version: str
    frontier_engine: str
    target_payload: dict[str, object] | None
    lane_records: list[FrontierLaneRecord]
    provisional_archive_members: list[FrontierArchiveMember]
    archive_members: list[FrontierArchiveMember]

    def to_json_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "frontier_version": self.frontier_version,
            "frontier_engine": self.frontier_engine,
            "target_payload": self.target_payload,
            "lane_records": [record.to_json_dict() for record in self.lane_records],
            "provisional_archive_members": [
                member.to_json_dict() for member in self.provisional_archive_members
            ],
            "archive_members": [
                member.to_json_dict() for member in self.archive_members
            ],
        }
        validate_frontier_campaign_progress_payload(payload)
        return payload

    @classmethod
    def from_json_dict(
        cls,
        payload: Mapping[str, object],
    ) -> FrontierCampaignProgress:
        lane_records = [
            FrontierLaneRecord.from_json_dict(item)
            for item in payload.get("lane_records", [])
        ]
        archive_members_payload = payload.get("archive_members", [])
        provisional_archive_members_payload = payload.get(
            "provisional_archive_members",
            [],
        )
        archive_members = [
            frontier_archive_member_from_json_dict(item)
            for item in archive_members_payload
        ]
        provisional_archive_members = [
            frontier_archive_member_from_json_dict(item)
            for item in provisional_archive_members_payload
        ]
        if not provisional_archive_members:
            provisional_archive_members = replay_provisional_archive_from_lane_records(
                lane_records
            )
        if not archive_members:
            archive_members = replay_archive_from_lane_records(lane_records)
        target_payload = payload.get("target_payload")
        return cls(
            schema_version=str(
                payload.get(
                    "schema_version",
                    FRONTIER_CAMPAIGN_PROGRESS_SCHEMA_VERSION,
                )
            ),
            campaign_id=str(payload["campaign_id"]),
            frontier_version=str(payload["frontier_version"]),
            frontier_engine=str(payload["frontier_engine"]),
            target_payload=None
            if target_payload is None
            else dict(target_payload),
            lane_records=lane_records,
            provisional_archive_members=provisional_archive_members,
            archive_members=archive_members,
        )


def serialize_goal_mode_payload(
    payload: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if payload is None:
        return None
    serialized = dict(payload)
    results_path = serialized.get("results_path")
    if results_path is not None:
        serialized["results_path"] = str(Path(results_path))
    command = serialized.get("command")
    if isinstance(command, list):
        serialized["command"] = [str(item) for item in command]
    results = serialized.get("results")
    if isinstance(results, Mapping):
        serialized["results"] = dict(results)
    return serialized


def write_frontier_campaign_progress(
    path: Path,
    progress: FrontierCampaignProgress,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(progress.to_json_dict(), f, indent=2)
        os.replace(tmp_path, str(path))
    except BaseException:
        os.unlink(tmp_path)
        raise


def load_frontier_campaign_progress(path: Path) -> FrontierCampaignProgress:
    return FrontierCampaignProgress.from_json_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def replay_archive_from_lane_records(
    lane_records: list[FrontierLaneRecord],
) -> list[FrontierArchiveMember]:
    archive_members: list[FrontierArchiveMember] = []
    for lane_record in lane_records:
        archive_member_payload = lane_record.archive_member
        if archive_member_payload is None:
            continue
        archive_member = frontier_archive_member_from_json_dict(
            archive_member_payload
        )
        archive_members, _ = update_frontier_archive(
            archive_members,
            archive_member,
        )
    return archive_members


def replay_provisional_archive_from_lane_records(
    lane_records: list[FrontierLaneRecord],
) -> list[FrontierArchiveMember]:
    provisional_members: list[FrontierArchiveMember] = []
    for lane_record in lane_records:
        provisional_member = _replay_provisional_member_from_lane_record(
            lane_record
        )
        if provisional_member is not None:
            provisional_members.append(provisional_member)
    return provisional_members


def _replay_provisional_member_from_lane_record(
    lane_record: FrontierLaneRecord,
) -> FrontierArchiveMember | None:
    provisional_member_ids = lane_record.provisional_member_ids
    if not provisional_member_ids:
        return None
    if (
        lane_record.results is not None
        and lane_record.result_source is not None
        and lane_record.results_path is not None
    ):
        return build_archive_member_from_results(
            campaign_id=lane_record.lane_contract.campaign_id,
            lane_id=lane_record.lane_contract.lane_id,
            payload={
                "result_source": lane_record.result_source,
                "results_path": lane_record.results_path,
                "results": lane_record.results,
            },
            rerun_contract=lane_record.lane_contract.rerun_contract,
            archive_state=FRONTIER_ARCHIVE_STATE_PROVISIONAL,
        )
    archive_member_payload = lane_record.archive_member
    if archive_member_payload is None:
        return None
    return replace(
        frontier_archive_member_from_json_dict(archive_member_payload),
        member_id=provisional_member_ids[0],
        archive_state=FRONTIER_ARCHIVE_STATE_PROVISIONAL,
    )


def build_frontier_lane_contract(
    *,
    campaign_id: str,
    lane_id: str,
    engine: str,
    scalarization_type: str,
    scalarization_params: Mapping[str, float],
    constraint_mode: str,
    warm_start_source: str | None,
    optimizer_budget: int,
    rng_seed: int,
    rerun_contract: Mapping[str, object],
    reference_point: Mapping[str, float] | None = None,
) -> FrontierLaneContract:
    return FrontierLaneContract(
        schema_version=FRONTIER_LANE_CONTRACT_SCHEMA_VERSION,
        lane_id=lane_id,
        campaign_id=campaign_id,
        engine=engine,
        reference_point=None
        if reference_point is None
        else {str(key): float(value) for key, value in reference_point.items()},
        scalarization_type=scalarization_type,
        scalarization_params={
            str(key): float(value) for key, value in scalarization_params.items()
        },
        constraint_mode=constraint_mode,
        warm_start_source=warm_start_source,
        optimizer_budget=int(optimizer_budget),
        rng_seed=int(rng_seed),
        rerun_contract=dict(rerun_contract),
    )


def build_frontier_lane_record(
    lane_contract: FrontierLaneContract,
    *,
    command: list[str],
    weights: Mapping[str, float],
    lane_budget: int,
    status: str,
    result_source: str | None = None,
    termination_reason: str | None = None,
    success: bool | None = None,
    provisional_archive_member: FrontierArchiveMember | None = None,
    archive_state: str | None = None,
    archive_member: FrontierArchiveMember | None = None,
    archive_update: Mapping[str, object] | None = None,
    results_path: str | None = None,
    results: Mapping[str, object] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> FrontierLaneRecord:
    provisional_member_ids: list[str] = []
    certified_member_ids: list[str] = []
    final_certified = False
    archive_member_payload = None
    archive_member_id = None
    if provisional_archive_member is not None:
        provisional_member_ids.append(provisional_archive_member.member_id)
    if archive_member is not None:
        archive_member_payload = archive_member.to_json_dict()
        archive_member_id = archive_member.member_id
        if archive_member.archive_state == "certified":
            certified_member_ids.append(archive_member.member_id)
            final_certified = True
    return FrontierLaneRecord(
        schema_version=FRONTIER_LANE_RECORD_SCHEMA_VERSION,
        lane_contract=lane_contract,
        status=status,
        command=[str(item) for item in command],
        weights={str(key): float(value) for key, value in weights.items()},
        lane_budget=int(lane_budget),
        result_source=result_source,
        termination_reason=termination_reason,
        success=success,
        provisional_member_ids=provisional_member_ids,
        certified_member_ids=certified_member_ids,
        final_certified=final_certified,
        archive_state=archive_state,
        archive_member_id=archive_member_id,
        archive_member=archive_member_payload,
        archive_update=None if archive_update is None else dict(archive_update),
        results_path=results_path,
        results=None if results is None else dict(results),
        error_type=error_type,
        error_message=error_message,
    )
