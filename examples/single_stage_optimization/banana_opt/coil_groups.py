"""Self-describing Stage 2 coil-layout manifest.

Replaces partition-by-convention with an explicit JSON manifest written into
Stage 2 ``results.json``. Loaders consult the manifest to partition
``bs.coils`` into role-scoped slices (TF / banana / proxy / VF) without
count-sniffing heuristics or mode flags.

Legacy artifacts lacking the manifest are handled by inferring one from the
pre-existing ``NUM_TF_COILS`` / ``NUM_BANANA_COILS`` / ``NUM_PROXY_COILS`` /
``NUM_VF_COILS`` fields backfilled by
``banana_opt.artifact_contracts.upgrade_legacy_stage2_artifact_results``. The
caller is told whether the manifest was read (``is_legacy_inferred=False``)
or synthesized (``is_legacy_inferred=True``) so it can tag downstream
provenance without silent behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

COIL_GROUPS_RESULTS_KEY = "COIL_GROUPS"

COIL_GROUP_ROLE_TF = "tf"
COIL_GROUP_ROLE_BANANA = "banana"
COIL_GROUP_ROLE_PROXY = "proxy"
COIL_GROUP_ROLE_VF = "vf"

COIL_GROUP_ROLES: tuple[str, ...] = (
    COIL_GROUP_ROLE_TF,
    COIL_GROUP_ROLE_BANANA,
    COIL_GROUP_ROLE_PROXY,
    COIL_GROUP_ROLE_VF,
)


@dataclass(frozen=True)
class CoilGroup:
    role: str
    start: int
    count: int

    @property
    def stop(self) -> int:
        return self.start + self.count


@dataclass(frozen=True)
class CoilGroupsManifest:
    groups: tuple[CoilGroup, ...]

    def total(self) -> int:
        return sum(group.count for group in self.groups)

    def by_role(self, role: str) -> CoilGroup | None:
        for group in self.groups:
            if group.role == role:
                return group
        return None

    def count_for_role(self, role: str) -> int:
        group = self.by_role(role)
        return 0 if group is None else group.count

    def to_json_payload(self) -> list[dict]:
        return [
            {"role": group.role, "start": group.start, "count": group.count}
            for group in self.groups
        ]

    @classmethod
    def from_json_payload(cls, payload) -> "CoilGroupsManifest":
        if not isinstance(payload, (list, tuple)):
            raise TypeError(
                "Coil-groups manifest payload must be a list of entries; "
                f"got {type(payload).__name__}."
            )
        groups: list[CoilGroup] = []
        for entry in payload:
            if not isinstance(entry, Mapping):
                raise TypeError(
                    "Coil-groups manifest entries must be mappings with "
                    "'role', 'start', 'count' keys."
                )
            try:
                role = str(entry["role"])
                start = int(entry["start"])
                count = int(entry["count"])
            except KeyError as missing:
                raise KeyError(
                    f"Coil-groups manifest entry missing required key {missing}."
                ) from None
            if count < 0 or start < 0:
                raise ValueError(
                    f"Coil-groups manifest entry role={role!r} has negative "
                    f"start={start} or count={count}."
                )
            groups.append(CoilGroup(role=role, start=start, count=count))
        return cls(groups=tuple(groups))


def build_contiguous_manifest(
    *,
    num_tf_coils: int,
    num_banana_coils: int,
    num_proxy_coils: int,
    num_vf_coils: int,
) -> CoilGroupsManifest:
    cursor = 0
    groups: list[CoilGroup] = []
    for role, count in (
        (COIL_GROUP_ROLE_TF, int(num_tf_coils)),
        (COIL_GROUP_ROLE_BANANA, int(num_banana_coils)),
        (COIL_GROUP_ROLE_PROXY, int(num_proxy_coils)),
        (COIL_GROUP_ROLE_VF, int(num_vf_coils)),
    ):
        if count < 0:
            raise ValueError(
                f"Coil-group count must be non-negative: role={role!r} count={count}."
            )
        groups.append(CoilGroup(role=role, start=cursor, count=count))
        cursor += count
    return CoilGroupsManifest(groups=tuple(groups))


def validate_manifest_against_coils(
    manifest: CoilGroupsManifest,
    *,
    total_loaded_coils: int,
) -> None:
    if manifest.total() != total_loaded_coils:
        raise ValueError(
            f"Coil-groups manifest expects {manifest.total()} coils but the "
            f"loaded BiotSavart artifact contains {total_loaded_coils}."
        )
    cursor = 0
    for group in manifest.groups:
        if group.start != cursor:
            raise ValueError(
                f"Coil-groups manifest is not contiguous: role={group.role!r} "
                f"expects start={cursor}, got start={group.start}."
            )
        cursor = group.stop


def partition_coils_by_manifest(
    coils: Sequence,
    manifest: CoilGroupsManifest,
) -> dict[str, tuple]:
    validate_manifest_against_coils(manifest, total_loaded_coils=len(coils))
    return {
        group.role: tuple(coils[group.start : group.stop])
        for group in manifest.groups
    }


def read_manifest_from_results(
    stage2_results: Mapping[str, object],
) -> CoilGroupsManifest | None:
    payload = stage2_results.get(COIL_GROUPS_RESULTS_KEY)
    if payload is None or payload == "":
        return None
    return CoilGroupsManifest.from_json_payload(payload)


def infer_manifest_from_legacy_counts(
    stage2_results: Mapping[str, object],
    *,
    total_loaded_coils: int,
    requested_num_tf_coils: int | None = None,
) -> CoilGroupsManifest:
    recorded_num_tf = stage2_results.get("NUM_TF_COILS")
    if recorded_num_tf is not None:
        num_tf = int(recorded_num_tf)
    elif requested_num_tf_coils is not None:
        num_tf = int(requested_num_tf_coils)
    else:
        raise ValueError(
            "Cannot infer legacy coil-groups manifest: NUM_TF_COILS missing "
            "from artifact and no requested_num_tf_coils was provided."
        )
    num_proxy = int(stage2_results.get("NUM_PROXY_COILS", 0) or 0)
    num_vf = int(stage2_results.get("NUM_VF_COILS", 0) or 0)
    recorded_num_banana = stage2_results.get("NUM_BANANA_COILS")
    if recorded_num_banana is None:
        num_banana = total_loaded_coils - num_tf - num_proxy - num_vf
    else:
        num_banana = int(recorded_num_banana)
    if num_banana < 0:
        raise ValueError(
            "Legacy coil-groups inference produced a negative banana-coil "
            f"count: total={total_loaded_coils}, tf={num_tf}, "
            f"proxy={num_proxy}, vf={num_vf}."
        )
    manifest = build_contiguous_manifest(
        num_tf_coils=num_tf,
        num_banana_coils=num_banana,
        num_proxy_coils=num_proxy,
        num_vf_coils=num_vf,
    )
    validate_manifest_against_coils(manifest, total_loaded_coils=total_loaded_coils)
    return manifest


@dataclass(frozen=True)
class ManifestResolution:
    manifest: CoilGroupsManifest
    is_legacy_inferred: bool


def resolve_manifest(
    stage2_results: Mapping[str, object],
    *,
    total_loaded_coils: int,
    requested_num_tf_coils: int | None = None,
) -> ManifestResolution:
    manifest = read_manifest_from_results(stage2_results)
    if manifest is not None:
        validate_manifest_against_coils(
            manifest, total_loaded_coils=total_loaded_coils
        )
        if requested_num_tf_coils is not None:
            recorded_tf_count = manifest.count_for_role(COIL_GROUP_ROLE_TF)
            if recorded_tf_count != int(requested_num_tf_coils):
                raise ValueError(
                    "Coil-groups manifest reports "
                    f"TF count={recorded_tf_count} but caller requested "
                    f"num_tf_coils={int(requested_num_tf_coils)}. "
                    "Single-stage reload refuses to re-slice coils with "
                    "inconsistent TF-count provenance."
                )
        return ManifestResolution(manifest=manifest, is_legacy_inferred=False)
    inferred = infer_manifest_from_legacy_counts(
        stage2_results,
        total_loaded_coils=total_loaded_coils,
        requested_num_tf_coils=requested_num_tf_coils,
    )
    if requested_num_tf_coils is not None:
        recorded_tf_count = inferred.count_for_role(COIL_GROUP_ROLE_TF)
        if recorded_tf_count != int(requested_num_tf_coils):
            raise ValueError(
                "Legacy coil-groups inference produced TF count="
                f"{recorded_tf_count} but caller requested "
                f"num_tf_coils={int(requested_num_tf_coils)}. "
                "Single-stage reload refuses to re-slice coils with "
                "inconsistent TF-count provenance."
            )
    return ManifestResolution(manifest=inferred, is_legacy_inferred=True)


__all__ = [
    "COIL_GROUPS_RESULTS_KEY",
    "COIL_GROUP_ROLES",
    "COIL_GROUP_ROLE_BANANA",
    "COIL_GROUP_ROLE_PROXY",
    "COIL_GROUP_ROLE_TF",
    "COIL_GROUP_ROLE_VF",
    "CoilGroup",
    "CoilGroupsManifest",
    "ManifestResolution",
    "build_contiguous_manifest",
    "infer_manifest_from_legacy_counts",
    "partition_coils_by_manifest",
    "read_manifest_from_results",
    "resolve_manifest",
    "validate_manifest_against_coils",
]
