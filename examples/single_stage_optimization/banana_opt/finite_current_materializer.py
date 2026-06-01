from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from simsopt.field import BiotSavart

from banana_opt.artifact_contracts import (
    STAGE2_BS_SHA256_KEY,
    compute_stage2_bs_sha256,
    upgrade_legacy_stage2_artifact_results,
)
from banana_opt.current_contracts import (
    BoozerCurrentConvention,
    CURRENT_MODE_ZERO_TOL,
    FINITE_CURRENT_MODE_SOURCE_ARTIFACT_METADATA,
    FiniteCurrentMode,
    HBT_PROXY_VF_CURRENT_RATIO,
    HBT_PROXY_VF_CURRENT_TOL_A,
    physical_current_to_boozer_I,
    validate_proxy_vf_current_convention_for_mode,
)
from banana_opt.finite_current_profiles import (
    JHALPERN30_FINITE_CURRENT_MODE,
    get_finite_current_profile,
)
from banana_opt.jhalpern30_compat import (
    build_jhalpern30_proxy_plasma_current_coils,
    resolve_jhalpern30_vf_template_path,
)
from banana_opt.json_compat import load_boozer_finite_i
from banana_opt.stage2_geometry import (
    build_proxy_plasma_current_coils,
    build_vf_coils_for_profile,
)
from banana_opt.stage2_single_stage_handoff import partition_loaded_stage2_coils
from workflow_runner_common import write_json


MATERIALIZED_SEED_SCHEMA_VERSION = 1
MATERIALIZED_IOTA_TRUST_STATUS_UNRUN = "unrun"
MATERIALIZED_IOTA_UNTRUSTED_REASON = "materialized_iota_untrusted"
BOOZER_I_ABS_TOL = 1.0e-12
FINITE_CURRENT_FIELD_SOURCE_MODES: tuple[FiniteCurrentMode, ...] = (
    "wataru_proxy_field",
    "jhalpern30_proxy_field",
)
_MATERIALIZER_OWNED_JHALPERN30_KEYS = frozenset(
    {
        "BOOZER_I",
        "FINITE_CURRENT_MODE",
        "BOOZER_CURRENT_CONVENTION",
        "G0_POLICY",
        "PROXY_PLACEMENT_MODE",
        "PROXY_VF_CURRENT_SCALAR_POLICY",
        "PROXY_PLASMA_CURRENT_A",
        "VF_CURRENT_A",
        "VF_TEMPLATE_PATH",
        "VF_TEMPLATE_SHA256",
        "VF_CURRENT_SIGN_POLICY",
        "VF_CURRENT_MUTABILITY",
        "NUM_TF_COILS",
        "NUM_BANANA_COILS",
        "NUM_PROXY_COILS",
        "NUM_VF_COILS",
        "TOTAL_COILS",
        "COIL_GROUPS",
    }
)
_NULLABLE_JHALPERN30_DONOR_REPLAY_KEYS = frozenset({"BANANA_I_FIXED_S2_KA"})


@dataclass(frozen=True)
class MaterializedFiniteCurrentSeedRequest:
    source_biot_savart: Path
    output_root: Path
    finite_current_mode: FiniteCurrentMode
    proxy_current_A: float
    stage2_results: Path | None = None
    equilibrium_file: Path | None = None
    vf_template_path: Path | None = None
    vf_current_A: float | None = None
    toroidal_flux: float | None = None
    surface_scale_factor: float = 1.0
    nphi: int = 127
    ntheta: int = 32
    num_tf_coils: int = 20
    overwrite: bool = False


@dataclass(frozen=True)
class MaterializedFiniteCurrentSeedResult:
    output_biot_savart: Path
    output_results: Path
    source_biot_savart: Path
    source_results: Path
    source_biot_savart_sha256: str
    output_biot_savart_sha256: str
    source_total_coils: int
    output_total_coils: int
    num_tf_coils: int
    num_banana_coils: int
    num_proxy_coils: int
    num_vf_coils: int
    realized_proxy_current_A: float
    realized_vf_current_A: float
    boozer_I: float
    finite_current_mode: FiniteCurrentMode
    single_stage_replay_eligible: bool

    def to_json_payload(self) -> dict[str, object]:
        profile = get_finite_current_profile(self.finite_current_mode)
        payload: dict[str, object] = {
            "output_biot_savart": str(self.output_biot_savart),
            "output_results": str(self.output_results),
            "source_biot_savart": str(self.source_biot_savart),
            "source_results": str(self.source_results),
            "source_biot_savart_sha256": self.source_biot_savart_sha256,
            "output_biot_savart_sha256": self.output_biot_savart_sha256,
            "source_total_coils": self.source_total_coils,
            "output_total_coils": self.output_total_coils,
            "num_tf_coils": self.num_tf_coils,
            "num_banana_coils": self.num_banana_coils,
            "num_proxy_coils": self.num_proxy_coils,
            "num_vf_coils": self.num_vf_coils,
            "realized_proxy_current_A": self.realized_proxy_current_A,
            "realized_vf_current_A": self.realized_vf_current_A,
            "boozer_I": self.boozer_I,
            "finite_current_mode": self.finite_current_mode,
            "single_stage_replay_eligible": self.single_stage_replay_eligible,
        }
        payload.update(profile.proxy_current_sign_summary_fields())
        return payload


@dataclass(frozen=True)
class _SurfaceMajorRadius:
    value: float

    def major_radius(self) -> float:
        return self.value


def materialize_finite_current_seed(
    request: MaterializedFiniteCurrentSeedRequest,
    *,
    load_fn=load_boozer_finite_i,
) -> MaterializedFiniteCurrentSeedResult:
    """Build proxy/VF field sources into a reloadable Stage-2 seed artifact.

    This materializes prescribed coil fields only. It does not validate or
    certify self-consistent equilibrium iota; that remains a VMEC science gate.
    """

    source_bs_path = _resolved_path(request.source_biot_savart)
    output_root = _resolved_path(request.output_root)
    output_bs_path = output_root / "biot_savart_opt.json"
    output_results_path = output_root / "results.json"
    _validate_output_paths(
        source_bs_path=source_bs_path,
        output_bs_path=output_bs_path,
        output_results_path=output_results_path,
        overwrite=bool(request.overwrite),
    )

    profile = get_finite_current_profile(request.finite_current_mode)
    if profile.mode not in FINITE_CURRENT_FIELD_SOURCE_MODES:
        raise ValueError(
            "Finite-current materialization requires a proxy-field mode; "
            f"got {profile.mode!r}."
        )

    source_results_path = _resolve_source_results_path(
        source_bs_path,
        request.stage2_results,
    )
    _validate_output_does_not_overwrite_source_results(
        output_results_path=output_results_path,
        source_results_path=source_results_path,
    )
    source_results = _load_source_results(
        source_results_path=source_results_path,
        source_bs_path=source_bs_path,
        known_num_tf_coils=int(request.num_tf_coils),
    )
    source_bs = _coerce_biot_savart(load_fn(str(source_bs_path)))
    source_partitions = partition_loaded_stage2_coils(
        source_bs.coils,
        stage2_results=source_results,
        requested_num_tf_coils=int(request.num_tf_coils),
    )

    vf_template_path = _resolve_vf_template_path(
        profile_default=profile.default_vf_template_path,
        requested_template=request.vf_template_path,
        finite_current_mode=profile.mode,
    )
    resolved_vf_current_A = _resolve_vf_current_A(
        finite_current_mode=profile.mode,
        proxy_current_A=float(request.proxy_current_A),
        requested_vf_current_A=request.vf_current_A,
    )
    validate_proxy_vf_current_convention_for_mode(
        profile.mode,
        proxy_plasma_current_A=float(request.proxy_current_A),
        vf_current_A=resolved_vf_current_A,
    )
    _validate_nonzero_proxy_field_current(
        finite_current_mode=profile.mode,
        proxy_current_A=float(request.proxy_current_A),
    )
    proxy_coils = _build_proxy_coils(
        finite_current_mode=profile.mode,
        proxy_current_A=float(request.proxy_current_A),
        source_results=source_results,
        request=request,
    )
    vf_build_result = build_vf_coils_for_profile(
        finite_current_mode=profile.mode,
        proxy_current_A=float(request.proxy_current_A),
        vf_current_A=resolved_vf_current_A,
        vf_template_path=str(vf_template_path),
        load_fn=load_fn,
    )
    vf_template_sha256 = _sha256_file(vf_template_path)
    _validate_presave_contract(
        source_results=source_results,
        profile_mode=profile.mode,
        vf_template_sha256=vf_template_sha256,
        num_tf_coils=len(source_partitions.tf_coils),
        num_banana_coils=len(source_partitions.banana_coils),
        num_proxy_coils=len(proxy_coils),
        num_vf_coils=len(vf_build_result.coils),
    )
    realized_proxy_current_A = _realized_proxy_current_A(proxy_coils)
    realized_vf_current_A = float(resolved_vf_current_A)
    validate_proxy_vf_current_convention_for_mode(
        profile.mode,
        proxy_plasma_current_A=realized_proxy_current_A,
        vf_current_A=realized_vf_current_A,
    )
    boozer_I = physical_current_to_boozer_I(
        realized_proxy_current_A,
        convention=profile.boozer_current_convention,
    )

    output_coils = [
        *source_partitions.tf_coils,
        *source_partitions.banana_coils,
        *proxy_coils,
        *vf_build_result.coils,
    ]
    output_bs = BiotSavart(output_coils)
    output_root.mkdir(parents=True, exist_ok=True)
    output_bs.save(str(output_bs_path))

    output_results = _build_materialized_results(
        source_results=source_results,
        source_bs_path=source_bs_path,
        output_bs_path=output_bs_path,
        profile_mode=profile.mode,
        vf_template_path=vf_template_path,
        vf_template_sha256=vf_template_sha256,
        realized_proxy_current_A=realized_proxy_current_A,
        realized_vf_current_A=realized_vf_current_A,
        boozer_I=boozer_I,
        num_tf_coils=len(source_partitions.tf_coils),
        num_banana_coils=len(source_partitions.banana_coils),
        num_proxy_coils=len(proxy_coils),
        num_vf_coils=len(vf_build_result.coils),
    )
    _validate_derived_boozer_I(
        realized_proxy_current_A=realized_proxy_current_A,
        boozer_I=boozer_I,
        recorded_boozer_I=output_results["BOOZER_I"],
        convention=profile.boozer_current_convention,
    )
    write_json(output_results_path, output_results)

    return MaterializedFiniteCurrentSeedResult(
        output_biot_savart=output_bs_path,
        output_results=output_results_path,
        source_biot_savart=source_bs_path,
        source_results=source_results_path,
        source_biot_savart_sha256=compute_stage2_bs_sha256(source_bs_path),
        output_biot_savart_sha256=str(output_results[STAGE2_BS_SHA256_KEY]),
        source_total_coils=len(source_bs.coils),
        output_total_coils=len(output_coils),
        num_tf_coils=len(source_partitions.tf_coils),
        num_banana_coils=len(source_partitions.banana_coils),
        num_proxy_coils=len(proxy_coils),
        num_vf_coils=len(vf_build_result.coils),
        realized_proxy_current_A=realized_proxy_current_A,
        realized_vf_current_A=realized_vf_current_A,
        boozer_I=boozer_I,
        finite_current_mode=profile.mode,
        single_stage_replay_eligible=True,
    )


def _resolved_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _validate_output_paths(
    *,
    source_bs_path: Path,
    output_bs_path: Path,
    output_results_path: Path,
    overwrite: bool,
) -> None:
    if output_bs_path == source_bs_path:
        raise ValueError("Output BiotSavart path must not overwrite the source seed.")
    for path in (output_bs_path, output_results_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists; pass --overwrite: {path}.")


def _validate_output_does_not_overwrite_source_results(
    *,
    output_results_path: Path,
    source_results_path: Path,
) -> None:
    if output_results_path == source_results_path:
        raise ValueError(
            "Output results.json path must not overwrite the source sidecar."
        )


def _resolve_source_results_path(
    source_bs_path: Path,
    requested_results_path: Path | None,
) -> Path:
    if requested_results_path is not None:
        resolved_results_path = _resolved_path(requested_results_path)
    else:
        resolved_results_path = source_bs_path.with_name("results.json")
    if not resolved_results_path.is_file():
        raise FileNotFoundError(
            "Finite-current materialization requires a Stage-2 results.json "
            f"sidecar; missing {resolved_results_path}."
        )
    return resolved_results_path


def _load_source_results(
    *,
    source_results_path: Path,
    source_bs_path: Path,
    known_num_tf_coils: int,
) -> dict[str, object]:
    payload = _load_json_mapping(source_results_path)
    recorded_source_digest = payload.get(STAGE2_BS_SHA256_KEY)
    if recorded_source_digest in {None, ""}:
        raise ValueError(
            f"Source Stage-2 sidecar is missing {STAGE2_BS_SHA256_KEY}; "
            "finite-current materialization requires checksum-bound donor metadata."
        )
    actual_source_digest = compute_stage2_bs_sha256(source_bs_path)
    if str(recorded_source_digest) != actual_source_digest:
        raise ValueError(
            "Source Stage-2 checksum mismatch: "
            f"{source_results_path} reports {STAGE2_BS_SHA256_KEY}="
            f"{recorded_source_digest!r}, but {source_bs_path} hashes to "
            f"{actual_source_digest!r}."
        )
    return upgrade_legacy_stage2_artifact_results(
        payload,
        known_num_tf_coils=known_num_tf_coils,
    )


def _load_json_mapping(path: Path) -> dict[str, object]:
    payload = Path(path).read_text(encoding="utf-8")
    loaded = json.loads(payload)
    if not isinstance(loaded, Mapping):
        raise ValueError(
            f"Expected JSON object at {path}; got {type(loaded).__name__}."
        )
    return {str(key): value for key, value in loaded.items()}


def _coerce_biot_savart(loaded) -> BiotSavart:
    if isinstance(loaded, BiotSavart):
        return loaded
    nested = getattr(loaded, "biotsavart", None)
    if isinstance(nested, BiotSavart):
        return nested
    coils = getattr(loaded, "coils", None)
    if coils is not None:
        return BiotSavart(list(coils))
    raise TypeError(
        "Materializer expected a BiotSavart, an object with .coils, or an "
        f"object with .biotsavart.coils; got {type(loaded).__name__}."
    )


def _resolve_vf_template_path(
    *,
    profile_default: Path | None,
    requested_template: Path | None,
    finite_current_mode: FiniteCurrentMode,
) -> Path:
    if requested_template is not None:
        resolved = _resolved_path(requested_template)
    elif finite_current_mode == JHALPERN30_FINITE_CURRENT_MODE:
        resolved = Path(resolve_jhalpern30_vf_template_path(None)).resolve()
    elif profile_default is not None:
        resolved = profile_default.resolve()
    else:
        raise ValueError(f"{finite_current_mode!r} requires a VF template path.")
    if not resolved.is_file():
        raise FileNotFoundError(f"VF template not found: {resolved}.")
    return resolved


def _resolve_vf_current_A(
    *,
    finite_current_mode: FiniteCurrentMode,
    proxy_current_A: float,
    requested_vf_current_A: float | None,
) -> float:
    if requested_vf_current_A is not None:
        return float(requested_vf_current_A)
    if finite_current_mode == JHALPERN30_FINITE_CURRENT_MODE:
        return float(proxy_current_A) * HBT_PROXY_VF_CURRENT_RATIO
    return abs(float(proxy_current_A)) * HBT_PROXY_VF_CURRENT_RATIO


def _validate_nonzero_proxy_field_current(
    *,
    finite_current_mode: FiniteCurrentMode,
    proxy_current_A: float,
) -> None:
    if abs(float(proxy_current_A)) <= CURRENT_MODE_ZERO_TOL:
        raise ValueError(
            f"{finite_current_mode!r} materialization requires non-zero "
            "proxy_current_A; use a vacuum seed for zero-current replay."
        )


def _build_proxy_coils(
    *,
    finite_current_mode: FiniteCurrentMode,
    proxy_current_A: float,
    source_results: Mapping[str, object],
    request: MaterializedFiniteCurrentSeedRequest,
):
    if finite_current_mode == JHALPERN30_FINITE_CURRENT_MODE:
        return build_jhalpern30_proxy_plasma_current_coils(
            _SurfaceMajorRadius(_required_float(source_results, "MAJOR_RADIUS")),
            proxy_current_A,
        )
    return build_proxy_plasma_current_coils(
        equilibrium_file=str(_resolve_equilibrium_file(source_results, request)),
        surface_scale_factor=float(request.surface_scale_factor),
        nphi=int(request.nphi),
        ntheta=int(request.ntheta),
        toroidal_flux=_resolve_toroidal_flux(source_results, request),
        plasma_current_A=proxy_current_A,
    )


def _resolve_equilibrium_file(
    source_results: Mapping[str, object],
    request: MaterializedFiniteCurrentSeedRequest,
) -> Path:
    if request.equilibrium_file is not None:
        equilibrium_file = _resolved_path(request.equilibrium_file)
    else:
        raw_equilibrium = source_results.get("PLASMA_SURF_PATH")
        if raw_equilibrium in {None, ""}:
            raise ValueError(
                "Wataru proxy materialization requires --equilibrium-file or "
                "PLASMA_SURF_PATH in the source results.json."
            )
        equilibrium_file = _resolved_path(Path(str(raw_equilibrium)))
    if not equilibrium_file.is_file():
        raise FileNotFoundError(
            f"Proxy equilibrium file not found: {equilibrium_file}."
        )
    return equilibrium_file


def _resolve_toroidal_flux(
    source_results: Mapping[str, object],
    request: MaterializedFiniteCurrentSeedRequest,
) -> float:
    if request.toroidal_flux is not None:
        return float(request.toroidal_flux)
    return _required_float(source_results, "TOROIDAL_FLUX")


def _required_float(source_results: Mapping[str, object], key: str) -> float:
    value = source_results.get(key)
    if value in {None, ""}:
        raise ValueError(f"Source results.json is missing required field {key}.")
    return _json_float(value, key)


def _realized_proxy_current_A(proxy_coils) -> float:
    if len(proxy_coils) != 1:
        raise ValueError(
            "Materialized finite-current profiles must produce exactly one "
            f"proxy current coil; got {len(proxy_coils)}."
        )
    return float(proxy_coils[0].current.get_value())


def _build_materialized_results(
    *,
    source_results: Mapping[str, object],
    source_bs_path: Path,
    output_bs_path: Path,
    profile_mode: FiniteCurrentMode,
    vf_template_path: Path,
    vf_template_sha256: str,
    realized_proxy_current_A: float,
    realized_vf_current_A: float,
    boozer_I: float,
    num_tf_coils: int,
    num_banana_coils: int,
    num_proxy_coils: int,
    num_vf_coils: int,
) -> dict[str, object]:
    profile = get_finite_current_profile(profile_mode)
    coil_groups = profile.build_default_coil_groups_manifest()
    total_coils = profile.default_total_coils
    results = dict(source_results)
    results.update(
        {
            **_materialized_untrusted_physics_fields(),
            "MATERIALIZED_SEED_SCHEMA_VERSION": MATERIALIZED_SEED_SCHEMA_VERSION,
            "MATERIALIZED_CURRENT_SOURCE": "proxy_vf_field_sources",
            "MATERIALIZED_CURRENT_RETARGETING_POLICY": (
                "materialize_new_seed_for_each_current"
            ),
            "MATERIALIZED_SOURCE_BS_PATH": str(source_bs_path),
            "MATERIALIZED_SOURCE_BS_SHA256": compute_stage2_bs_sha256(source_bs_path),
            "MATERIALIZED_IOTA_TRUST_STATUS": MATERIALIZED_IOTA_TRUST_STATUS_UNRUN,
            "MATERIALIZED_IOTA_TRUST_REASON": (
                "No no-collapse, monotonicity, or VMEC curtor cross-check has "
                "been run for this materialized prescribed-field seed."
            ),
            "FINITE_CURRENT_MODE": profile.mode,
            "FINITE_CURRENT_MODE_SOURCE": FINITE_CURRENT_MODE_SOURCE_ARTIFACT_METADATA,
            "EFFECTIVE_CURRENT_MODE": profile.mode,
            "BOOZER_CURRENT_CONVENTION": profile.boozer_current_convention,
            "BOOZER_I": float(boozer_I),
            "G0_POLICY": profile.g0_policy,
            "PROXY_PLACEMENT_MODE": profile.proxy_placement_policy,
            "PROXY_VF_CURRENT_SCALAR_POLICY": profile.proxy_vf_current_scalar_policy,
            **profile.proxy_current_sign_metadata_fields(),
            "PROXY_PLASMA_CURRENT_A": float(realized_proxy_current_A),
            "PLASMA_CURRENT_A": float(realized_proxy_current_A),
            "VF_CURRENT_A": float(realized_vf_current_A),
            "VF_TEMPLATE_PATH": str(vf_template_path),
            "VF_TEMPLATE_SHA256": vf_template_sha256,
            "VF_CURRENT_SIGN_POLICY": profile.vf_current_sign_policy,
            "VF_CURRENT_MUTABILITY": profile.vf_current_mutability,
            "NUM_TF_COILS": num_tf_coils,
            "NUM_BANANA_COILS": num_banana_coils,
            "NUM_PROXY_COILS": num_proxy_coils,
            "NUM_VF_COILS": num_vf_coils,
            "TOTAL_COILS": total_coils,
            "COIL_GROUPS": coil_groups.to_json_payload(),
            "STAGE2_BS_PATH": str(output_bs_path),
            STAGE2_BS_SHA256_KEY: compute_stage2_bs_sha256(output_bs_path),
        }
    )
    return results


def _materialized_untrusted_physics_fields() -> dict[str, object]:
    return {
        "SEED_ROLE": "materialized_finite_current_seed",
        "DIAGNOSTIC_ONLY": True,
        "PRODUCTION_HANDOFF_READY": False,
        "HANDOFF_BLOCKING_GATE": MATERIALIZED_IOTA_UNTRUSTED_REASON,
        "PROMOTION_READY": False,
        "BOOZER_BOOTABLE": False,
        "IOTA_NEAR_TARGET": False,
        "IOTA_FEASIBLE": False,
        "BOOTABILITY_REASON": "missing_artifact_metadata",
        "BOOTABILITY_STAGE": "materialized_seed",
        "BOOTABILITY_TARGET_IOTA": None,
        "BOOTABILITY_SOLVED_IOTA": None,
        "BOOTABILITY_SELF_INTERSECTING": None,
        "BOOTABILITY_SOLVE_SUCCESS": False,
        "BOOTABILITY_ABS_IOTA_ERROR": None,
        "BOOTABILITY_ERROR_TYPE": MATERIALIZED_IOTA_UNTRUSTED_REASON,
        "BOOTABILITY_ERROR_MESSAGE": (
            "Materialized prescribed-field seed has no validated Boozer/iota "
            "solve; run the no-collapse, monotonicity, and VMEC curtor science gate."
        ),
        "BOOZER_SOLVE_SUCCESS": False,
        "BOOZER_SELF_INTERSECTING": None,
        "BOOZER_CONSTRAINED_RESIDUAL_NORM": None,
        "BOOZER_TRUSTED": False,
        "IOTA_OBJECTIVE_ACTIVE": False,
        "BOOZER_TRUST_REASON": MATERIALIZED_IOTA_UNTRUSTED_REASON,
        "BOOZER_TRUST_TOL": None,
        "FINAL_IOTA": None,
        "FINAL_IOTA_AXIS": None,
        "FINAL_IOTA_EDGE": None,
        "FINAL_VOLUME": None,
        "FIELD_ERROR": None,
        "NONQS_RATIO": None,
        "BOOZER_RESIDUAL": None,
    }


def _validate_presave_contract(
    *,
    source_results: Mapping[str, object],
    profile_mode: FiniteCurrentMode,
    vf_template_sha256: str,
    num_tf_coils: int,
    num_banana_coils: int,
    num_proxy_coils: int,
    num_vf_coils: int,
) -> None:
    profile = get_finite_current_profile(profile_mode)
    expected_counts = {
        "NUM_TF_COILS": profile.default_num_tf_coils,
        "NUM_BANANA_COILS": profile.default_num_banana_coils,
        "NUM_PROXY_COILS": profile.default_num_proxy_coils,
        "NUM_VF_COILS": profile.default_num_vf_coils,
    }
    actual_counts = {
        "NUM_TF_COILS": int(num_tf_coils),
        "NUM_BANANA_COILS": int(num_banana_coils),
        "NUM_PROXY_COILS": int(num_proxy_coils),
        "NUM_VF_COILS": int(num_vf_coils),
    }
    mismatches = [
        f"{key}={actual_counts[key]} expected {expected_counts[key]}"
        for key in expected_counts
        if actual_counts[key] != expected_counts[key]
    ]
    if mismatches:
        raise ValueError(
            f"{profile_mode!r} materialization requires profile-canonical coil "
            "counts before writing output: " + "; ".join(mismatches) + "."
        )
    if profile_mode == JHALPERN30_FINITE_CURRENT_MODE:
        _require_jhalpern30_replay_fields(source_results)
        if vf_template_sha256 != profile.vf_template_sha256:
            raise ValueError(
                "jhalpern30 materialization requires the canonical VF template "
                f"hash {profile.vf_template_sha256}; got {vf_template_sha256}."
            )


def _require_jhalpern30_replay_fields(source_results: Mapping[str, object]) -> None:
    profile = get_finite_current_profile(JHALPERN30_FINITE_CURRENT_MODE)
    donor_replay_keys = tuple(
        key
        for key in profile.required_artifact_metadata_keys
        if key not in _MATERIALIZER_OWNED_JHALPERN30_KEYS
    )
    missing_keys = [key for key in donor_replay_keys if key not in source_results]
    if missing_keys:
        raise ValueError(
            "jhalpern30 materialization requires donor replay metadata: "
            + ", ".join(missing_keys)
            + "."
        )
    null_keys = [
        key
        for key in donor_replay_keys
        if key not in _NULLABLE_JHALPERN30_DONOR_REPLAY_KEYS
        and source_results[key] is None
    ]
    if null_keys:
        raise ValueError(
            "jhalpern30 materialization requires non-null donor replay metadata: "
            + ", ".join(null_keys)
            + "."
        )


def _validate_derived_boozer_I(
    *,
    realized_proxy_current_A: float,
    boozer_I: float,
    recorded_boozer_I: object,
    convention: BoozerCurrentConvention,
) -> None:
    recorded = _json_float(recorded_boozer_I, "BOOZER_I")
    expected = physical_current_to_boozer_I(
        realized_proxy_current_A,
        convention=convention,
    )
    if not np.isclose(
        recorded,
        expected,
        rtol=0.0,
        atol=max(BOOZER_I_ABS_TOL, HBT_PROXY_VF_CURRENT_TOL_A * abs(expected)),
    ):
        raise ValueError(
            "Materialized BOOZER_I must match mu0 times the realized proxy "
            f"current: expected {expected:.16g}, got {recorded:.16g}."
        )
    if not np.isclose(float(boozer_I), expected, rtol=0.0, atol=BOOZER_I_ABS_TOL):
        raise ValueError(
            "Internal BOOZER_I derivation drifted from realized proxy current."
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, not bool.")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"{field_name} must be numeric; got {type(value).__name__}.")


__all__ = [
    "BOOZER_I_ABS_TOL",
    "FINITE_CURRENT_FIELD_SOURCE_MODES",
    "MATERIALIZED_IOTA_TRUST_STATUS_UNRUN",
    "MATERIALIZED_IOTA_UNTRUSTED_REASON",
    "MATERIALIZED_SEED_SCHEMA_VERSION",
    "MaterializedFiniteCurrentSeedRequest",
    "MaterializedFiniteCurrentSeedResult",
    "materialize_finite_current_seed",
]
