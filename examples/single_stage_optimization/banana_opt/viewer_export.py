from __future__ import annotations

"""Export Simsopt Boozer/BiotSavart JSON into the clearance-viewer artifact.

The Rust viewer consumes only the viewer-owned JSON contract emitted here. It
does not import Simsopt, execute Python, or interpret SIMSON graphs.
"""

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simsopt._core.optimizable import load
from simsopt.field import BiotSavart

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

if __package__:
    from .artifact_contracts import (
        validate_boozer_surface_json_current_lineage,
        validate_vacuum_boozer_surface_json,
        validate_vacuum_boozer_surface_payload,
    )
    from .coil_groups import (
        COIL_GROUP_ROLE_BANANA,
        COIL_GROUP_ROLE_PROXY,
        COIL_GROUP_ROLE_TF,
        COIL_GROUP_ROLE_VF,
        COIL_GROUPS_RESULTS_KEY,
        CoilGroupsManifest,
        build_contiguous_manifest,
        partition_coils_by_manifest,
        resolve_manifest,
    )
    from .finite_current_profiles import (
        FINITE_CURRENT_PROFILES,
        VACUUM_FINITE_CURRENT_MODE,
        get_finite_current_profile,
    )
    from .hardware_contracts import (
        BANANA_WINDING_CHANNEL_ORIENTATION,
        BANANA_WINDING_MINOR_RADIUS_M,
        BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        TYPE_KK_HARDWARE_CONTRACT_ID,
        TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_MM,
        TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_MM,
    )
    from .json_compat import load_boozer_finite_i
else:
    from banana_opt.artifact_contracts import (
        validate_boozer_surface_json_current_lineage,
        validate_vacuum_boozer_surface_json,
        validate_vacuum_boozer_surface_payload,
    )
    from banana_opt.coil_groups import (
        COIL_GROUP_ROLE_BANANA,
        COIL_GROUP_ROLE_PROXY,
        COIL_GROUP_ROLE_TF,
        COIL_GROUP_ROLE_VF,
        COIL_GROUPS_RESULTS_KEY,
        CoilGroupsManifest,
        build_contiguous_manifest,
        partition_coils_by_manifest,
        resolve_manifest,
    )
    from banana_opt.finite_current_profiles import (
        FINITE_CURRENT_PROFILES,
        VACUUM_FINITE_CURRENT_MODE,
        get_finite_current_profile,
    )
    from banana_opt.hardware_contracts import (
        BANANA_WINDING_CHANNEL_ORIENTATION,
        BANANA_WINDING_MINOR_RADIUS_M,
        BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        TYPE_KK_HARDWARE_CONTRACT_ID,
        TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_MM,
        TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_MM,
    )
    from banana_opt.json_compat import load_boozer_finite_i


ARTIFACT_TYPE = "hbt_clearance_boozer_render_artifact"
SCHEMA_VERSION = 1
COORDINATE_FRAME = "simsopt_xyz_m"
CONTRACT_FAMILIES = (
    "auto",
    "simsopt_surrogate_vacuum",
    "simsopt_surrogate_finite_i",
    "baseline_original_vacuum",
)
CONVERSION_MODES = ("centerline_only", "generated_solid")
COIL_SCOPES = ("banana", "all")
DEFAULT_WINDING_R0_M = BANANA_WINDING_SURFACE_MAJOR_RADIUS_M
DEFAULT_WINDING_A_M = BANANA_WINDING_MINOR_RADIUS_M
DEFAULT_BRACKET_WIDTH_MM = TYPE_KK_OUTER_CHANNEL_WIDTH_BINORMAL_MM
DEFAULT_BRACKET_DEPTH_MM = TYPE_KK_OUTER_CHANNEL_DEPTH_NORMAL_MM
DEFAULT_CHANNEL_WALL_M = 0.003
STAGE2_RESULTS_FILENAME = "results.json"
FINITE_BUILD_ENABLED_KEY = "FINITE_BUILD_ENABLED"
FINITEBUILD_FILAMENTS_PER_BANANA_KEY = "FINITEBUILD_FILAMENTS_PER_BANANA"
FINITEBUILD_NUMFILAMENTS_N_KEY = "FINITEBUILD_NUMFILAMENTS_N"
FINITEBUILD_NUMFILAMENTS_B_KEY = "FINITEBUILD_NUMFILAMENTS_B"


@dataclass(frozen=True)
class ViewerExportConfig:
    input_path: Path
    output_path: Path
    contract_family: str = "auto"
    conversion_mode: str = "centerline_only"
    coil_scope: str = "banana"
    finite_current_mode: str | None = None
    num_tf_coils: int | None = None
    filaments_per_banana: int | None = None
    winding_r0_m: float = DEFAULT_WINDING_R0_M
    winding_a_m: float = DEFAULT_WINDING_A_M
    bracket_width_mm: float = DEFAULT_BRACKET_WIDTH_MM
    bracket_depth_mm: float = DEFAULT_BRACKET_DEPTH_MM
    source_repo: str | None = None
    source_commit: str | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class SourcePartition:
    tf_coils: tuple[object, ...]
    render_coils: tuple[object, ...]
    finite_current_mode: str


@dataclass(frozen=True)
class PartitionSpec:
    manifest: CoilGroupsManifest
    finite_current_mode: str
    filaments_per_banana: int = 1


@dataclass(frozen=True)
class LoadedSource:
    biot_savart: BiotSavart
    source_object_class: str


@dataclass(frozen=True, eq=False)
class PackCenterlineCurve:
    """Duck-typed curve holding a reduced pack centerline (``gamma()`` only)."""

    points: np.ndarray

    def gamma(self) -> np.ndarray:
        return self.points


@dataclass(frozen=True)
class PackNetCurrent:
    """Duck-typed current holding a pack's net current in amperes."""

    value_A: float

    def get_value(self) -> float:
        return self.value_A


@dataclass(frozen=True, eq=False)
class PackCenterlineCoil:
    """One finite-build filament pack reduced to a renderable centerline coil."""

    curve: PackCenterlineCurve
    current: PackNetCurrent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Simsopt Boozer/BiotSavart JSON into a clearance-viewer render artifact."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--contract-family",
        choices=CONTRACT_FAMILIES,
        default="auto",
        help="Input lineage contract to enforce before export.",
    )
    parser.add_argument(
        "--conversion-mode",
        choices=CONVERSION_MODES,
        default="centerline_only",
        help="Emit centerlines only or add generated U-channel display meshes.",
    )
    parser.add_argument(
        "--coil-scope",
        choices=COIL_SCOPES,
        default="banana",
        help="Render only banana coils by default; use all for diagnostics.",
    )
    parser.add_argument(
        "--finite-current-mode",
        default=None,
        help="Finite-current profile used to partition metadata-free inputs.",
    )
    parser.add_argument(
        "--num-tf-coils",
        type=int,
        default=None,
        help=(
            "TF-coil count of a finite-build pack payload; requires "
            "--filaments-per-banana."
        ),
    )
    parser.add_argument(
        "--filaments-per-banana",
        type=int,
        default=None,
        help=(
            "Filaments per banana pack in a finite-build payload; each pack is "
            "reduced to its centerline curve. Requires --num-tf-coils. Without "
            "these flags, pack payloads are partitioned from the Stage-2 "
            "results.json next to the input."
        ),
    )
    parser.add_argument("--winding-r0-m", type=float, default=DEFAULT_WINDING_R0_M)
    parser.add_argument("--winding-a-m", type=float, default=DEFAULT_WINDING_A_M)
    parser.add_argument("--bracket-width-mm", type=float, default=DEFAULT_BRACKET_WIDTH_MM)
    parser.add_argument("--bracket-depth-mm", type=float, default=DEFAULT_BRACKET_DEPTH_MM)
    parser.add_argument("--source-repo", default=None)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ViewerExportConfig:
    return ViewerExportConfig(
        input_path=args.input,
        output_path=args.output,
        contract_family=str(args.contract_family),
        conversion_mode=str(args.conversion_mode),
        coil_scope=str(args.coil_scope),
        finite_current_mode=args.finite_current_mode,
        num_tf_coils=args.num_tf_coils,
        filaments_per_banana=args.filaments_per_banana,
        winding_r0_m=float(args.winding_r0_m),
        winding_a_m=float(args.winding_a_m),
        bracket_width_mm=float(args.bracket_width_mm),
        bracket_depth_mm=float(args.bracket_depth_mm),
        source_repo=args.source_repo,
        source_commit=args.source_commit,
        overwrite=bool(args.overwrite),
    )


def export_viewer_artifact(config: ViewerExportConfig) -> dict[str, object]:
    validate_config(config)
    source_path = config.input_path.expanduser().resolve()
    output_path = config.output_path.expanduser().resolve()
    if output_path.exists() and not config.overwrite:
        raise ValueError(f"Refusing to overwrite existing file {output_path}; pass --overwrite.")

    contract_family, lineage = resolve_contract_family(source_path, config.contract_family)
    loaded = load_source(source_path, contract_family)
    partition = partition_source_coils(
        tuple(loaded.biot_savart.coils),
        contract_family=contract_family,
        coil_scope=config.coil_scope,
        finite_current_mode=config.finite_current_mode,
        num_tf_coils=config.num_tf_coils,
        filaments_per_banana=config.filaments_per_banana,
        source_path=source_path,
    )
    payload = build_artifact_payload(
        config,
        source_path=source_path,
        contract_family=contract_family,
        lineage=lineage,
        loaded=loaded,
        partition=partition,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_config(config: ViewerExportConfig) -> None:
    if config.contract_family not in CONTRACT_FAMILIES:
        raise ValueError(f"Unsupported contract family {config.contract_family!r}.")
    if config.conversion_mode not in CONVERSION_MODES:
        raise ValueError(f"Unsupported conversion mode {config.conversion_mode!r}.")
    if config.coil_scope not in COIL_SCOPES:
        raise ValueError(f"Unsupported coil scope {config.coil_scope!r}.")
    pack_flags_set = (
        config.num_tf_coils is not None or config.filaments_per_banana is not None
    )
    if pack_flags_set and config.finite_current_mode is not None:
        raise ValueError(
            "--finite-current-mode conflicts with --num-tf-coils/"
            "--filaments-per-banana; pass exactly one partition source."
        )
    if pack_flags_set and (
        config.num_tf_coils is None or config.filaments_per_banana is None
    ):
        raise ValueError(
            "--num-tf-coils and --filaments-per-banana must be passed together."
        )
    if config.filaments_per_banana is not None and int(config.filaments_per_banana) < 1:
        raise ValueError("--filaments-per-banana must be a positive integer.")
    if config.num_tf_coils is not None and int(config.num_tf_coils) < 0:
        raise ValueError("--num-tf-coils must be non-negative.")
    for name, value in (
        ("winding_r0_m", config.winding_r0_m),
        ("winding_a_m", config.winding_a_m),
        ("bracket_width_mm", config.bracket_width_mm),
        ("bracket_depth_mm", config.bracket_depth_mm),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be a finite positive number.")


def resolve_contract_family(source_path: Path, requested_family: str) -> tuple[str, dict[str, object]]:
    if requested_family == "auto":
        return auto_contract_family(source_path)
    lineage = lineage_for_family(source_path, requested_family)
    return requested_family, lineage


def auto_contract_family(source_path: Path) -> tuple[str, dict[str, object]]:
    payload = read_json_mapping(source_path)
    baseline_lineage = baseline_vacuum_lineage_payload(payload)
    if "baseline-original" in source_path.parts and baseline_lineage["passed"]:
        return "baseline_original_vacuum", baseline_lineage

    finite_lineage = validate_boozer_surface_json_current_lineage(source_path)
    if finite_lineage.get("passed") and int(finite_lineage.get("finite_i_field_count", 0)) > 0:
        return "simsopt_surrogate_finite_i", finite_lineage

    vacuum_lineage = validate_vacuum_boozer_surface_json(source_path)
    if vacuum_lineage.get("passed"):
        return "simsopt_surrogate_vacuum", vacuum_lineage

    raise ValueError(f"Could not classify Boozer JSON lineage for {source_path}.")


def lineage_for_family(source_path: Path, family: str) -> dict[str, object]:
    if family == "baseline_original_vacuum":
        lineage = baseline_vacuum_lineage_payload(read_json_mapping(source_path))
        if not lineage["passed"]:
            raise ValueError(f"{source_path} is not baseline-original vacuum Boozer JSON.")
        return lineage
    if family == "simsopt_surrogate_vacuum":
        lineage = validate_vacuum_boozer_surface_json(source_path)
        if not lineage.get("passed"):
            raise ValueError(f"{source_path} is not surrogate vacuum Boozer JSON.")
        return lineage
    if family == "simsopt_surrogate_finite_i":
        lineage = validate_boozer_surface_json_current_lineage(source_path)
        if not lineage.get("passed") or int(lineage.get("finite_i_field_count", 0)) < 1:
            raise ValueError(f"{source_path} is not surrogate finite-I Boozer JSON.")
        return lineage
    raise ValueError(f"Unsupported contract family {family!r}.")


def baseline_vacuum_lineage_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return validate_vacuum_boozer_surface_payload(dict(payload))


def load_source(source_path: Path, contract_family: str) -> LoadedSource:
    loaded = (
        load_boozer_finite_i(str(source_path))
        if contract_family == "simsopt_surrogate_finite_i"
        else load(str(source_path))
    )
    return LoadedSource(
        biot_savart=normalize_biot_savart_source(loaded),
        source_object_class=type(loaded).__name__,
    )


def normalize_biot_savart_source(source: object) -> BiotSavart:
    if isinstance(source, BiotSavart):
        return source
    nested_biot_savart = getattr(source, "biotsavart", None)
    if nested_biot_savart is not None:
        return normalize_biot_savart_source(nested_biot_savart)
    coils = getattr(source, "coils", None)
    if coils is not None and not callable(coils):
        return BiotSavart(list(coils))
    raise ValueError(
        "Unsupported source artifact: expected a BiotSavart, an object with "
        ".coils, or an object with .biotsavart.coils."
    )


def partition_source_coils(
    coils: Sequence[object],
    *,
    contract_family: str,
    coil_scope: str,
    finite_current_mode: str | None,
    num_tf_coils: int | None = None,
    filaments_per_banana: int | None = None,
    source_path: Path | None = None,
) -> SourcePartition:
    partition_spec = resolved_partition_spec(
        total_coils=len(coils),
        contract_family=contract_family,
        finite_current_mode=finite_current_mode,
        num_tf_coils=num_tf_coils,
        filaments_per_banana=filaments_per_banana,
        source_path=source_path,
    )
    role_partitions = partition_coils_by_manifest(coils, partition_spec.manifest)
    tf_coils = tuple(role_partitions.get(COIL_GROUP_ROLE_TF, ()))
    banana_coils = tuple(role_partitions.get(COIL_GROUP_ROLE_BANANA, ()))
    if partition_spec.filaments_per_banana > 1:
        banana_coils = reduce_filament_packs_to_centerlines(
            banana_coils, partition_spec.filaments_per_banana
        )
        render_coils = (
            tf_coils
            + banana_coils
            + tuple(role_partitions.get(COIL_GROUP_ROLE_PROXY, ()))
            + tuple(role_partitions.get(COIL_GROUP_ROLE_VF, ()))
            if coil_scope == "all"
            else banana_coils
        )
    else:
        render_coils = tuple(coils) if coil_scope == "all" else banana_coils
    if not render_coils:
        raise ValueError("Selected coil scope contains no coils to render.")
    return SourcePartition(
        tf_coils=tf_coils,
        render_coils=render_coils,
        finite_current_mode=partition_spec.finite_current_mode,
    )


def resolved_partition_spec(
    *,
    total_coils: int,
    contract_family: str,
    finite_current_mode: str | None,
    num_tf_coils: int | None = None,
    filaments_per_banana: int | None = None,
    source_path: Path | None = None,
) -> PartitionSpec:
    if finite_current_mode is not None:
        profile = get_finite_current_profile(finite_current_mode)
        return PartitionSpec(
            manifest=profile.build_default_coil_groups_manifest(),
            finite_current_mode=str(profile.mode),
        )

    if num_tf_coils is not None or filaments_per_banana is not None:
        return pack_partition_spec_from_flags(
            total_coils=total_coils,
            num_tf_coils=num_tf_coils,
            filaments_per_banana=filaments_per_banana,
        )

    if contract_family in {"baseline_original_vacuum", "simsopt_surrogate_vacuum"}:
        vacuum_profile = get_finite_current_profile(VACUUM_FINITE_CURRENT_MODE)
        if vacuum_profile.default_total_coils == total_coils:
            return PartitionSpec(
                manifest=vacuum_profile.build_default_coil_groups_manifest(),
                finite_current_mode=str(vacuum_profile.mode),
            )

    matching = [
        profile
        for profile in FINITE_CURRENT_PROFILES.values()
        if profile.default_total_coils == total_coils
    ]
    if len(matching) == 1:
        profile = matching[0]
        return PartitionSpec(
            manifest=profile.build_default_coil_groups_manifest(),
            finite_current_mode=str(profile.mode),
        )

    equivalent_counts = {
        (
            profile.default_num_tf_coils,
            profile.default_num_banana_coils,
            profile.default_num_proxy_coils,
            profile.default_num_vf_coils,
        )
        for profile in matching
    }
    if len(equivalent_counts) == 1:
        tf_count, banana_count, proxy_count, vf_count = next(iter(equivalent_counts))
        return PartitionSpec(
            manifest=build_contiguous_manifest(
                num_tf_coils=tf_count,
                num_banana_coils=banana_count,
                num_proxy_coils=proxy_count,
                num_vf_coils=vf_count,
            ),
            finite_current_mode=(
                f"metadata_free_{tf_count}tf_{banana_count}banana_"
                f"{proxy_count}proxy_{vf_count}vf"
            ),
        )

    if source_path is not None:
        metadata_spec = pack_partition_spec_from_stage2_results(
            source_path, total_coils=total_coils
        )
        if metadata_spec is not None:
            return metadata_spec

    raise ValueError(
        "Metadata-free input cannot be partitioned uniquely; pass "
        "--finite-current-mode, or --num-tf-coils with --filaments-per-banana "
        "for finite-build pack payloads (a Stage-2 results.json next to the "
        "input is read automatically)."
    )


def pack_partition_spec_from_flags(
    *,
    total_coils: int,
    num_tf_coils: int | None,
    filaments_per_banana: int | None,
) -> PartitionSpec:
    """Explicit TF + banana-filament partition from the CLI pack flags."""
    if num_tf_coils is None or filaments_per_banana is None:
        raise ValueError(
            "--num-tf-coils and --filaments-per-banana must be passed together."
        )
    tf_count = int(num_tf_coils)
    filaments = int(filaments_per_banana)
    if filaments < 1:
        raise ValueError("--filaments-per-banana must be a positive integer.")
    if tf_count < 0 or tf_count >= total_coils:
        raise ValueError(
            f"--num-tf-coils={tf_count} leaves no banana coils in a "
            f"{total_coils}-coil payload."
        )
    filament_coil_count = total_coils - tf_count
    if filament_coil_count % filaments:
        raise ValueError(
            f"{filament_coil_count} banana filament coils are not divisible by "
            f"--filaments-per-banana={filaments}."
        )
    return PartitionSpec(
        manifest=build_contiguous_manifest(
            num_tf_coils=tf_count,
            num_banana_coils=filament_coil_count,
            num_proxy_coils=0,
            num_vf_coils=0,
        ),
        finite_current_mode=(
            f"pack_flags_{tf_count}tf_{filament_coil_count}banana_{filaments}fil"
        ),
        filaments_per_banana=filaments,
    )


def pack_partition_spec_from_stage2_results(
    source_path: Path,
    *,
    total_coils: int,
) -> PartitionSpec | None:
    """Pack partition from the Stage-2 ``results.json`` sibling of the input.

    Returns ``None`` when no sibling exists or it carries no coil-layout keys;
    raises when the recorded layout contradicts the loaded coil payload.
    """
    results_path = source_path.parent / STAGE2_RESULTS_FILENAME
    if not results_path.is_file():
        return None
    stage2_results = read_json_mapping(results_path)
    if (
        stage2_results.get(COIL_GROUPS_RESULTS_KEY) is None
        and stage2_results.get("NUM_TF_COILS") is None
    ):
        return None
    resolution = resolve_manifest(stage2_results, total_loaded_coils=total_coils)
    manifest = resolution.manifest
    filaments = stage2_filaments_per_banana(stage2_results)
    banana_count = manifest.count_for_role(COIL_GROUP_ROLE_BANANA)
    if filaments > 1 and banana_count % filaments:
        raise ValueError(
            f"{results_path} records {banana_count} banana filament coils, not "
            f"divisible by {FINITEBUILD_FILAMENTS_PER_BANANA_KEY}={filaments}."
        )
    return PartitionSpec(
        manifest=manifest,
        finite_current_mode=(
            "stage2_results_metadata_"
            f"{manifest.count_for_role(COIL_GROUP_ROLE_TF)}tf_"
            f"{banana_count}banana_"
            f"{manifest.count_for_role(COIL_GROUP_ROLE_PROXY)}proxy_"
            f"{manifest.count_for_role(COIL_GROUP_ROLE_VF)}vf_"
            f"{filaments}fil"
        ),
        filaments_per_banana=filaments,
    )


def stage2_filaments_per_banana(stage2_results: Mapping[str, object]) -> int:
    """Filaments per banana pack recorded by Stage-2 finite-build metadata (1 = thin)."""
    if not stage2_results.get(FINITE_BUILD_ENABLED_KEY):
        return 1
    recorded = stage2_results.get(FINITEBUILD_FILAMENTS_PER_BANANA_KEY)
    numfilaments_n = stage2_results.get(FINITEBUILD_NUMFILAMENTS_N_KEY)
    numfilaments_b = stage2_results.get(FINITEBUILD_NUMFILAMENTS_B_KEY)
    derived = (
        int(numfilaments_n) * int(numfilaments_b)
        if numfilaments_n is not None and numfilaments_b is not None
        else None
    )
    if recorded is None and derived is None:
        raise ValueError(
            f"{FINITE_BUILD_ENABLED_KEY} Stage-2 results lack "
            f"{FINITEBUILD_FILAMENTS_PER_BANANA_KEY} and "
            f"{FINITEBUILD_NUMFILAMENTS_N_KEY}/{FINITEBUILD_NUMFILAMENTS_B_KEY}; "
            "cannot reduce filament packs."
        )
    if recorded is not None and derived is not None and int(recorded) != derived:
        raise ValueError(
            f"Stage-2 results record {FINITEBUILD_FILAMENTS_PER_BANANA_KEY}="
            f"{int(recorded)} but {FINITEBUILD_NUMFILAMENTS_N_KEY}*"
            f"{FINITEBUILD_NUMFILAMENTS_B_KEY}={derived}; refusing inconsistent "
            "pack metadata."
        )
    filaments = int(recorded) if recorded is not None else int(derived)
    if filaments < 1:
        raise ValueError(
            f"Finite-build filament count must be positive; got {filaments}."
        )
    return filaments


def reduce_filament_packs_to_centerlines(
    filament_coils: Sequence[object],
    filaments_per_banana: int,
) -> tuple[PackCenterlineCoil, ...]:
    """Collapse consecutive filament packs into per-banana centerline coils.

    ``create_multifilament_grid`` lays each pack's filaments on a zero-mean
    shift grid around the centerline, and symmetry expansion transforms packs
    linearly as contiguous wholes, so the pointwise mean of one pack's filament
    gammas is exactly that pack's centerline (to float rounding) and the pack's
    net current is the sum of its filament currents.
    """
    filaments = int(filaments_per_banana)
    if filaments < 1:
        raise ValueError("filaments_per_banana must be a positive integer.")
    if not filament_coils or len(filament_coils) % filaments:
        raise ValueError(
            f"Cannot reduce {len(filament_coils)} filament coils into packs of "
            f"{filaments}."
        )
    centerline_coils: list[PackCenterlineCoil] = []
    for pack_start in range(0, len(filament_coils), filaments):
        pack = filament_coils[pack_start : pack_start + filaments]
        centerline = np.mean(np.stack([coil_points(coil) for coil in pack]), axis=0)
        centerline.setflags(write=False)
        centerline_coils.append(
            PackCenterlineCoil(
                curve=PackCenterlineCurve(points=centerline),
                current=PackNetCurrent(
                    value_A=math.fsum(coil_current_A(coil) for coil in pack)
                ),
            )
        )
    return tuple(centerline_coils)


def build_artifact_payload(
    config: ViewerExportConfig,
    *,
    source_path: Path,
    contract_family: str,
    lineage: Mapping[str, object],
    loaded: LoadedSource,
    partition: SourcePartition,
) -> dict[str, object]:
    coils_payload = rendered_coil_payloads(partition.render_coils)
    generated_meshes = (
        generated_mesh_payloads(
            partition.render_coils,
            winding_r0_m=config.winding_r0_m,
            bracket_width_mm=config.bracket_width_mm,
            bracket_depth_mm=config.bracket_depth_mm,
        )
        if config.conversion_mode == "generated_solid"
        else []
    )
    mesh_triangles = int(sum(len(mesh["indices"]) // 3 for mesh in generated_meshes))
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "display_name": source_path.name,
            "source_sha256": sha256_path(source_path),
            "source_repo": config.source_repo,
            "source_commit": config.source_commit,
            "contract_family": contract_family,
            "source_object_class": loaded.source_object_class,
            "conversion_mode": config.conversion_mode,
            "vacuum_lineage_ok": bool(lineage.get("passed"))
            and contract_family != "simsopt_surrogate_finite_i",
            "finite_current_lineage_ok": contract_family == "simsopt_surrogate_finite_i"
            and bool(lineage.get("passed")),
            "promotion_eligible": contract_family != "simsopt_surrogate_finite_i"
            and bool(lineage.get("passed")),
        },
        "units": "m",
        "coordinate_frame": COORDINATE_FRAME,
        "winding_surface": {"R0_m": config.winding_r0_m, "a_m": config.winding_a_m},
        "hardware_contract": {
            "contract_id": TYPE_KK_HARDWARE_CONTRACT_ID,
            "bracket_width_mm": config.bracket_width_mm,
            "bracket_depth_mm": config.bracket_depth_mm,
            "orientation": BANANA_WINDING_CHANNEL_ORIENTATION,
        },
        "tf_current_kA": tf_current_kA(partition.tf_coils),
        "coils": coils_payload,
        "generated_meshes": generated_meshes,
        "validation": {
            "generated_mesh_triangles": mesh_triangles,
            "source_coil_count": len(partition.render_coils),
            "finite_current_mode": partition.finite_current_mode,
        },
    }


def rendered_coil_payloads(coils: Sequence[object]) -> list[dict[str, object]]:
    return [
        {
            "id": f"banana_{index + 1:03d}",
            "current_A": coil_current_A(coil),
            "points": coil_points(coil).tolist(),
        }
        for index, coil in enumerate(coils)
    ]


def generated_mesh_payloads(
    coils: Sequence[object],
    *,
    winding_r0_m: float,
    bracket_width_mm: float,
    bracket_depth_mm: float,
) -> list[dict[str, object]]:
    return [
        {
            "id": f"generated_coil_{index + 1:03d}",
            **u_channel_mesh_payload(
                coil_points(coil),
                r0=float(winding_r0_m),
                half_width=float(bracket_width_mm) / 2000.0,
                half_depth=float(bracket_depth_mm) / 2000.0,
                wall=DEFAULT_CHANNEL_WALL_M,
            ),
        }
        for index, coil in enumerate(coils)
    ]


def coil_points(coil: object) -> np.ndarray:
    points = np.asarray(coil.curve.gamma(), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Coil curve.gamma() must return an N x 3 point array.")
    if points.shape[0] < 3:
        raise ValueError("Coil centerline must contain at least three points.")
    if not np.all(np.isfinite(points)):
        raise ValueError("Coil centerline contains non-finite coordinates.")
    return points


def coil_current_A(coil: object) -> float:
    value = float(coil.current.get_value())
    if not math.isfinite(value):
        raise ValueError("Coil current must be finite.")
    return value


def tf_current_kA(tf_coils: Sequence[object]) -> float | None:
    if not tf_coils:
        return None
    return coil_current_A(tf_coils[0]) / 1000.0


def u_channel_mesh_payload(
    points: np.ndarray,
    *,
    r0: float,
    half_width: float,
    half_depth: float,
    wall: float,
) -> dict[str, object]:
    frame_values = frames(points, r0)
    profile_points = np.asarray(
        [
            (-half_width, -half_depth),
            (half_width, -half_depth),
            (half_width, half_depth),
            (half_width - wall, half_depth),
            (half_width - wall, -half_depth + wall),
            (-half_width + wall, -half_depth + wall),
            (-half_width + wall, half_depth),
            (-half_width, half_depth),
        ],
        dtype=float,
    )
    positions: list[list[float]] = []
    for point, frame in zip(points, frame_values):
        for tangential_offset, radial_offset in profile_points:
            position = point + tangential_offset * frame["tangential"] + radial_offset * frame["radial"]
            positions.append([float(position[0]), float(position[1]), float(position[2])])

    profile_count = len(profile_points)
    indices: list[int] = []
    for index in range(len(points)):
        next_index = (index + 1) % len(points)
        for profile_index in range(profile_count):
            next_profile = (profile_index + 1) % profile_count
            a = index * profile_count + profile_index
            b = index * profile_count + next_profile
            c = next_index * profile_count + next_profile
            d = next_index * profile_count + profile_index
            indices.extend((a, b, c, a, c, d))
    if signed_volume_x6(positions, indices) < 0.0:
        flip_winding(indices)
    return {"positions": positions, "indices": indices}


def frames(points: np.ndarray, r0: float) -> list[dict[str, np.ndarray]]:
    frame_values: list[dict[str, np.ndarray]] = []
    point_count = len(points)
    for index, point in enumerate(points):
        previous_point = points[(index + point_count - 1) % point_count]
        next_point = points[(index + 1) % point_count]
        tangent = normalized(next_point - previous_point)
        phi = math.atan2(float(point[1]), float(point[0]))
        axis = np.asarray([r0 * math.cos(phi), r0 * math.sin(phi), 0.0], dtype=float)
        radial0 = normalized(point - axis)
        tangential = normalized(np.cross(tangent, radial0))
        radial = normalized(np.cross(tangential, tangent))
        frame_values.append(
            {
                "tangent": tangent,
                "radial": radial,
                "tangential": tangential,
            }
        )
    return frame_values


def normalized(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / (length if length > 1.0e-12 else 1.0e-12)


def signed_volume_x6(positions: Sequence[Sequence[float]], indices: Sequence[int]) -> float:
    total = 0.0
    for offset in range(0, len(indices), 3):
        a = np.asarray(positions[indices[offset]], dtype=float)
        b = np.asarray(positions[indices[offset + 1]], dtype=float)
        c = np.asarray(positions[indices[offset + 2]], dtype=float)
        total += float(np.dot(a, np.cross(b, c)))
    return total


def flip_winding(indices: list[int]) -> None:
    for offset in range(0, len(indices), 3):
        indices[offset + 1], indices[offset + 2] = indices[offset + 2], indices[offset + 1]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_mapping(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = export_viewer_artifact(config_from_args(args))
    print(
        "wrote viewer artifact "
        f"{args.output} ({len(payload['coils'])} coils, "
        f"{sum(len(mesh['indices']) // 3 for mesh in payload['generated_meshes'])} mesh triangles)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
