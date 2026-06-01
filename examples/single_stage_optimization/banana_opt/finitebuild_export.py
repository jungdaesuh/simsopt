from __future__ import annotations

"""Export/evaluation-only finite-build BiotSavart conversion for HBT coils.

The generated finite-build artifact is not a new physics model and is not an
optimization warm-start contract.
"""

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    MGrid,
    apply_symmetries_to_curves,
    apply_symmetries_to_currents,
)
from simsopt.geo import create_multifilament_grid

from banana_opt.coil_groups import (
    COIL_GROUP_ROLE_BANANA,
    COIL_GROUP_ROLE_PROXY,
    COIL_GROUP_ROLE_TF,
    COIL_GROUP_ROLE_VF,
    build_contiguous_manifest,
    partition_coils_by_manifest,
)
from banana_opt.finite_current_profiles import (
    FINITE_CURRENT_PROFILES,
    FiniteCurrentProfile,
    get_finite_current_profile,
)
from banana_opt.json_compat import load_boozer_finite_i
from banana_opt.stage2_single_stage_handoff import (
    Stage2CoilPartitions,
    partition_loaded_stage2_coils,
)


DEFAULT_HBT_BANANA_NFP = 5
DEFAULT_HBT_BANANA_STELLSYM = True
METADATA_SCHEMA_VERSION = 1
FINITEBUILD_SCOPE_BANANA_ONLY = "banana-only"
FINITEBUILD_SCOPE_ALL_FAMILIES = "all-families"
FINITEBUILD_SCOPES = (FINITEBUILD_SCOPE_BANANA_ONLY, FINITEBUILD_SCOPE_ALL_FAMILIES)
MGRID_GROUPING_SINGLE = "single"
MGRID_GROUPING_FAMILY = "family"
MGRID_GROUPINGS = (MGRID_GROUPING_SINGLE, MGRID_GROUPING_FAMILY)


@dataclass(frozen=True)
class FiniteBuildExportConfig:
    """Configuration for converting loaded HBT coils to finite-build artifacts."""

    biot_savart_file: Path
    output: Path | None
    numfilaments_n: int
    numfilaments_b: int
    gapsize_n: float
    gapsize_b: float
    rotation_order: int | None = None
    frame: str = "centroid"
    banana_current_A: float | None = None
    stage2_results: Path | None = None
    finite_current_mode: str | None = None
    num_tf_coils: int = 20
    nfp: int = DEFAULT_HBT_BANANA_NFP
    stellsym: bool = DEFAULT_HBT_BANANA_STELLSYM
    overwrite: bool = False
    finitebuild_scope: str = FINITEBUILD_SCOPE_BANANA_ONLY
    write_mgrid: bool = False
    mgrid_output: Path | None = None
    mgrid_grouping: str = MGRID_GROUPING_SINGLE
    mgrid_nr: int | None = None
    mgrid_nz: int | None = None
    mgrid_nphi: int | None = None
    mgrid_rmin: float | None = None
    mgrid_rmax: float | None = None
    mgrid_zmin: float | None = None
    mgrid_zmax: float | None = None
    mgrid_nfp: int | None = None


@dataclass(frozen=True)
class FiniteBuildCoilPartitions:
    tf_coils: tuple[Coil, ...]
    banana_coils: tuple[Coil, ...]
    proxy_coils: tuple[Coil, ...]
    vf_coils: tuple[Coil, ...]
    finite_current_mode: str
    source: str
    manifest_is_legacy_inferred: bool = False


@dataclass(frozen=True)
class FiniteBuildExportResult:
    output_path: Path
    metadata_path: Path
    mgrid_output_path: Path | None
    mgrid_group_names: tuple[str, ...]
    source_path: Path
    source_counts: dict[str, int]
    output_counts: dict[str, int]
    finitebuild_scope: str
    banana_current_A: float | None
    banana_filament_current_A: float | None
    source_banana_total_current_A: float
    output_banana_total_current_A: float
    current_override: bool
    finite_current_mode: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an HBT BiotSavart artifact into a finite-build export "
            "artifact. The output is for field evaluation and external export, "
            "not single-stage optimization warm starts."
        )
    )
    parser.add_argument("biotsavart_file", help="Path to the source BiotSavart JSON.")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSON path. Defaults to a sibling path replacing "
            "'biot_savart' or 'biotsavart' with a finitebuild name."
        ),
    )
    parser.add_argument("--numfilaments-n", required=True, type=int)
    parser.add_argument("--numfilaments-b", required=True, type=int)
    parser.add_argument("--gapsize-n", required=True, type=float)
    parser.add_argument("--gapsize-b", required=True, type=float)
    parser.add_argument("--rotation-order", default=None, type=int)
    parser.add_argument(
        "--frame",
        default="centroid",
        choices=("centroid", "frenet"),
        help="Frame used by simsopt.geo.create_multifilament_grid.",
    )
    parser.add_argument(
        "--banana-current-A",
        default=None,
        type=float,
        help=(
            "Override the master banana current in amperes. Supported only "
            "for --finitebuild-scope banana-only."
        ),
    )
    parser.add_argument(
        "--finitebuild-scope",
        default=FINITEBUILD_SCOPE_BANANA_ONLY,
        choices=FINITEBUILD_SCOPES,
        help=(
            "Which coil families to convert. 'banana-only' preserves the "
            "existing HBT master-banana symmetry path; 'all-families' converts "
            "each loaded physical TF/banana/proxy/VF coil directly."
        ),
    )
    parser.add_argument(
        "--stage2-results",
        default=None,
        help="Optional Stage 2 results JSON containing COIL_GROUPS metadata.",
    )
    parser.add_argument(
        "--finite-current-mode",
        default=None,
        choices=tuple(sorted(FINITE_CURRENT_PROFILES)),
        help=(
            "Profile used for metadata-free fallback partitioning. Required "
            "for ambiguous 51-coil artifacts without --stage2-results."
        ),
    )
    parser.add_argument(
        "--num-tf-coils",
        default=20,
        type=int,
        help="Expected TF coil count when --stage2-results is supplied.",
    )
    parser.add_argument(
        "--nfp",
        default=DEFAULT_HBT_BANANA_NFP,
        type=int,
        help="Banana winding-surface field periods used to expand the master coil.",
    )
    parser.add_argument(
        "--no-stellsym",
        action="store_false",
        dest="stellsym",
        help="Disable stellarator symmetry when expanding the finite-build banana.",
    )
    parser.set_defaults(stellsym=DEFAULT_HBT_BANANA_STELLSYM)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing finite-build output and metadata files.",
    )
    parser.add_argument(
        "--write-mgrid",
        action="store_true",
        help="Write a VMEC MGrid from the output BiotSavart.",
    )
    parser.add_argument(
        "--mgrid-output",
        default=None,
        help=(
            "Output MGrid NetCDF path. Defaults to a sibling "
            "'<finitebuild-output-stem>_mgrid.nc' when --write-mgrid is used."
        ),
    )
    parser.add_argument(
        "--mgrid-grouping",
        default=MGRID_GROUPING_SINGLE,
        choices=MGRID_GROUPINGS,
        help=(
            "MGrid current grouping. 'single' writes the net field as one "
            "EXTCUR group; 'family' writes one non-empty TF/banana/proxy/VF "
            "group each."
        ),
    )
    parser.add_argument("--mgrid-nr", default=None, type=int)
    parser.add_argument("--mgrid-nz", default=None, type=int)
    parser.add_argument("--mgrid-nphi", default=None, type=int)
    parser.add_argument("--mgrid-rmin", default=None, type=float)
    parser.add_argument("--mgrid-rmax", default=None, type=float)
    parser.add_argument("--mgrid-zmin", default=None, type=float)
    parser.add_argument("--mgrid-zmax", default=None, type=float)
    parser.add_argument("--mgrid-nfp", default=None, type=int)
    return parser


def config_from_args(args: argparse.Namespace) -> FiniteBuildExportConfig:
    return FiniteBuildExportConfig(
        biot_savart_file=Path(args.biotsavart_file),
        output=None if args.output is None else Path(args.output),
        numfilaments_n=int(args.numfilaments_n),
        numfilaments_b=int(args.numfilaments_b),
        gapsize_n=float(args.gapsize_n),
        gapsize_b=float(args.gapsize_b),
        rotation_order=args.rotation_order,
        frame=str(args.frame),
        banana_current_A=args.banana_current_A,
        stage2_results=None
        if args.stage2_results is None
        else Path(args.stage2_results),
        finite_current_mode=args.finite_current_mode,
        num_tf_coils=int(args.num_tf_coils),
        nfp=int(args.nfp),
        stellsym=bool(args.stellsym),
        overwrite=bool(args.overwrite),
        finitebuild_scope=str(args.finitebuild_scope),
        write_mgrid=bool(args.write_mgrid),
        mgrid_output=None if args.mgrid_output is None else Path(args.mgrid_output),
        mgrid_grouping=str(args.mgrid_grouping),
        mgrid_nr=args.mgrid_nr,
        mgrid_nz=args.mgrid_nz,
        mgrid_nphi=args.mgrid_nphi,
        mgrid_rmin=args.mgrid_rmin,
        mgrid_rmax=args.mgrid_rmax,
        mgrid_zmin=args.mgrid_zmin,
        mgrid_zmax=args.mgrid_zmax,
        mgrid_nfp=args.mgrid_nfp,
    )


def normalize_biot_savart_source(source: object) -> BiotSavart:
    """Return a BiotSavart from supported loaded source artifact shapes."""

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


def default_output_path(source_path: Path) -> Path:
    filename = source_path.name
    if "biot_savart" in filename:
        output_name = filename.replace("biot_savart", "biot_savart_finitebuild", 1)
    elif "biotsavart" in filename:
        output_name = filename.replace("biotsavart", "biotsavart_finitebuild", 1)
    else:
        output_name = f"{source_path.stem}_finitebuild{source_path.suffix}"
    return source_path.with_name(output_name)


def metadata_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_metadata.json")


def mgrid_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_mgrid.nc")


def export_finitebuild_biot_savart(
    config: FiniteBuildExportConfig,
) -> FiniteBuildExportResult:
    _validate_config(config)
    source_path = config.biot_savart_file.expanduser().resolve()
    output_path = (
        default_output_path(source_path)
        if config.output is None
        else config.output.expanduser().resolve()
    )
    metadata_path = metadata_output_path(output_path)
    resolved_mgrid_output_path = _resolved_mgrid_output_path(
        output_path=output_path,
        config=config,
    )
    _validate_output_paths(
        source_path,
        output_path=output_path,
        metadata_path=metadata_path,
        mgrid_path=resolved_mgrid_output_path,
        overwrite=config.overwrite,
    )

    source_biot_savart = normalize_biot_savart_source(
        load_boozer_finite_i(str(source_path))
    )
    source_coils = tuple(source_biot_savart.coils)
    partitions = _resolve_partitions(source_coils, config)
    source_counts = _partition_counts(partitions)
    source_banana_total_current_A = _coil_current_sum(partitions.banana_coils)
    source_current_totals_A = _coil_current_totals(partitions)
    source_current_abs_totals_A = _coil_current_abs_totals(partitions)

    if config.finitebuild_scope == FINITEBUILD_SCOPE_BANANA_ONLY:
        (
            output_tf_coils,
            output_banana_coils,
            output_proxy_coils,
            output_vf_coils,
            banana_current_A,
            filament_current_A,
        ) = _build_banana_only_output_coils(partitions, config)
    elif config.finitebuild_scope == FINITEBUILD_SCOPE_ALL_FAMILIES:
        (
            output_tf_coils,
            output_banana_coils,
            output_proxy_coils,
            output_vf_coils,
            banana_current_A,
            filament_current_A,
        ) = _build_all_families_output_coils(partitions, config)
    else:
        raise ValueError(
            f"Unsupported finite-build scope {config.finitebuild_scope!r}."
        )

    output_partitions = FiniteBuildCoilPartitions(
        tf_coils=output_tf_coils,
        banana_coils=output_banana_coils,
        proxy_coils=output_proxy_coils,
        vf_coils=output_vf_coils,
        finite_current_mode=partitions.finite_current_mode,
        source=partitions.source,
        manifest_is_legacy_inferred=partitions.manifest_is_legacy_inferred,
    )
    output_banana_total_current_A = _coil_current_sum(output_banana_coils)
    output_coils = [
        *output_tf_coils,
        *output_banana_coils,
        *output_proxy_coils,
        *output_vf_coils,
    ]
    output_biot_savart = BiotSavart(output_coils)
    output_biot_savart.save(str(output_path))

    mgrid_group_names: tuple[str, ...] = ()
    if resolved_mgrid_output_path is not None:
        mgrid_group_names = _write_mgrid(
            output_biot_savart,
            output_partitions,
            resolved_mgrid_output_path,
            config,
        )

    output_counts = _partition_counts(output_partitions)
    output_current_totals_A = _coil_current_totals(output_partitions)
    output_current_abs_totals_A = _coil_current_abs_totals(output_partitions)
    _write_metadata(
        metadata_path,
        source_path=source_path,
        output_path=output_path,
        mgrid_path=resolved_mgrid_output_path,
        mgrid_group_names=mgrid_group_names,
        config=config,
        partitions=partitions,
        source_counts=source_counts,
        output_counts=output_counts,
        source_current_totals_A=source_current_totals_A,
        source_current_abs_totals_A=source_current_abs_totals_A,
        output_current_totals_A=output_current_totals_A,
        output_current_abs_totals_A=output_current_abs_totals_A,
        banana_current_A=banana_current_A,
        filament_current_A=filament_current_A,
        source_banana_total_current_A=source_banana_total_current_A,
        output_banana_total_current_A=output_banana_total_current_A,
    )

    return FiniteBuildExportResult(
        output_path=output_path,
        metadata_path=metadata_path,
        mgrid_output_path=resolved_mgrid_output_path,
        mgrid_group_names=mgrid_group_names,
        source_path=source_path,
        source_counts=source_counts,
        output_counts=output_counts,
        finitebuild_scope=config.finitebuild_scope,
        banana_current_A=banana_current_A,
        banana_filament_current_A=filament_current_A,
        source_banana_total_current_A=source_banana_total_current_A,
        output_banana_total_current_A=output_banana_total_current_A,
        current_override=config.banana_current_A is not None,
        finite_current_mode=partitions.finite_current_mode,
    )


def _validate_config(config: FiniteBuildExportConfig) -> None:
    if config.finitebuild_scope not in FINITEBUILD_SCOPES:
        raise ValueError(
            f"Unsupported finite-build scope {config.finitebuild_scope!r}."
        )
    if config.mgrid_grouping not in MGRID_GROUPINGS:
        raise ValueError(f"Unsupported MGrid grouping {config.mgrid_grouping!r}.")
    if (
        config.finitebuild_scope == FINITEBUILD_SCOPE_ALL_FAMILIES
        and config.banana_current_A is not None
    ):
        raise ValueError(
            "--banana-current-A is supported only with --finitebuild-scope banana-only."
        )
    if config.banana_current_A is not None and not math.isfinite(
        float(config.banana_current_A)
    ):
        raise ValueError("--banana-current-A must be finite.")
    if int(config.numfilaments_n) <= 0:
        raise ValueError("--numfilaments-n must be positive.")
    if int(config.numfilaments_b) <= 0:
        raise ValueError("--numfilaments-b must be positive.")
    if not math.isfinite(float(config.gapsize_n)):
        raise ValueError("--gapsize-n must be finite.")
    if not math.isfinite(float(config.gapsize_b)):
        raise ValueError("--gapsize-b must be finite.")
    if float(config.gapsize_n) <= 0.0:
        raise ValueError("--gapsize-n must be positive.")
    if float(config.gapsize_b) <= 0.0:
        raise ValueError("--gapsize-b must be positive.")
    if int(config.nfp) <= 0:
        raise ValueError("--nfp must be positive.")
    _validate_mgrid_config(config)


def _validate_mgrid_config(config: FiniteBuildExportConfig) -> None:
    if not config.write_mgrid:
        if _mgrid_options_requested_without_output(config):
            raise ValueError("MGrid options require --write-mgrid.")
        return

    required_values: tuple[tuple[str, object | None], ...] = (
        ("--mgrid-nr", config.mgrid_nr),
        ("--mgrid-nz", config.mgrid_nz),
        ("--mgrid-nphi", config.mgrid_nphi),
        ("--mgrid-rmin", config.mgrid_rmin),
        ("--mgrid-rmax", config.mgrid_rmax),
        ("--mgrid-zmin", config.mgrid_zmin),
        ("--mgrid-zmax", config.mgrid_zmax),
        ("--mgrid-nfp", config.mgrid_nfp),
    )
    missing = tuple(name for name, value in required_values if value is None)
    if missing:
        raise ValueError(
            "--write-mgrid requires explicit grid settings: " + ", ".join(missing) + "."
        )
    if int(config.mgrid_nr) <= 0:
        raise ValueError("--mgrid-nr must be positive.")
    if int(config.mgrid_nz) <= 0:
        raise ValueError("--mgrid-nz must be positive.")
    if int(config.mgrid_nphi) <= 0:
        raise ValueError("--mgrid-nphi must be positive.")
    if int(config.mgrid_nfp) <= 0:
        raise ValueError("--mgrid-nfp must be positive.")
    mgrid_bounds = (
        ("--mgrid-rmin", float(config.mgrid_rmin)),
        ("--mgrid-rmax", float(config.mgrid_rmax)),
        ("--mgrid-zmin", float(config.mgrid_zmin)),
        ("--mgrid-zmax", float(config.mgrid_zmax)),
    )
    nonfinite_bounds = tuple(
        name for name, value in mgrid_bounds if not math.isfinite(value)
    )
    if nonfinite_bounds:
        raise ValueError(
            "MGrid bounds must be finite: " + ", ".join(nonfinite_bounds) + "."
        )
    if float(config.mgrid_rmin) >= float(config.mgrid_rmax):
        raise ValueError("--mgrid-rmin must be smaller than --mgrid-rmax.")
    if float(config.mgrid_zmin) >= float(config.mgrid_zmax):
        raise ValueError("--mgrid-zmin must be smaller than --mgrid-zmax.")


def _mgrid_options_requested_without_output(config: FiniteBuildExportConfig) -> bool:
    return (
        config.mgrid_output is not None
        or config.mgrid_grouping != MGRID_GROUPING_SINGLE
        or config.mgrid_nr is not None
        or config.mgrid_nz is not None
        or config.mgrid_nphi is not None
        or config.mgrid_rmin is not None
        or config.mgrid_rmax is not None
        or config.mgrid_zmin is not None
        or config.mgrid_zmax is not None
        or config.mgrid_nfp is not None
    )


def _validate_output_paths(
    source_path: Path,
    *,
    output_path: Path,
    metadata_path: Path,
    mgrid_path: Path | None,
    overwrite: bool,
) -> None:
    if output_path == source_path:
        raise ValueError(
            "Finite-build output path must not overwrite the source artifact."
        )
    if metadata_path == source_path or metadata_path == output_path:
        raise ValueError(
            "Finite-build metadata path must be distinct from source and output."
        )
    output_paths = (output_path, metadata_path) + (
        () if mgrid_path is None else (mgrid_path,)
    )
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("Finite-build output paths must be distinct.")
    if mgrid_path is not None and mgrid_path == source_path:
        raise ValueError("MGrid output path must not overwrite the source artifact.")
    for path in output_paths:
        if path.exists() and not overwrite:
            raise ValueError(
                f"Refusing to overwrite existing file {path}; pass --overwrite."
            )


def _resolve_partitions(
    coils: Sequence[Coil],
    config: FiniteBuildExportConfig,
) -> FiniteBuildCoilPartitions:
    if config.stage2_results is not None:
        stage2_results = _read_json_mapping(config.stage2_results)
        stage2_partitions = partition_loaded_stage2_coils(
            coils,
            stage2_results=stage2_results,
            requested_num_tf_coils=config.num_tf_coils,
        )
        return _from_stage2_partitions(stage2_partitions)

    profile = _resolve_metadata_free_profile(
        total_loaded_coils=len(coils),
        finite_current_mode=config.finite_current_mode,
    )
    role_partitions = partition_coils_by_manifest(
        coils,
        profile.build_default_coil_groups_manifest(),
    )
    return FiniteBuildCoilPartitions(
        tf_coils=tuple(role_partitions.get(COIL_GROUP_ROLE_TF, ())),
        banana_coils=tuple(role_partitions.get(COIL_GROUP_ROLE_BANANA, ())),
        proxy_coils=tuple(role_partitions.get(COIL_GROUP_ROLE_PROXY, ())),
        vf_coils=tuple(role_partitions.get(COIL_GROUP_ROLE_VF, ())),
        finite_current_mode=str(profile.mode),
        source="finite_current_profile",
        manifest_is_legacy_inferred=False,
    )


def _from_stage2_partitions(
    stage2_partitions: Stage2CoilPartitions,
) -> FiniteBuildCoilPartitions:
    return FiniteBuildCoilPartitions(
        tf_coils=tuple(stage2_partitions.tf_coils),
        banana_coils=tuple(stage2_partitions.banana_coils),
        proxy_coils=tuple(stage2_partitions.proxy_coils),
        vf_coils=tuple(stage2_partitions.vf_coils),
        finite_current_mode=str(stage2_partitions.finite_current_mode),
        source="stage2_results",
        manifest_is_legacy_inferred=bool(
            stage2_partitions.coil_groups_manifest_is_legacy_inferred
        ),
    )


def _resolve_metadata_free_profile(
    *,
    total_loaded_coils: int,
    finite_current_mode: str | None,
) -> FiniteCurrentProfile:
    if finite_current_mode is not None:
        profile = get_finite_current_profile(finite_current_mode)
        if profile.default_total_coils != total_loaded_coils:
            raise ValueError(
                f"Profile {finite_current_mode!r} expects "
                f"{profile.default_total_coils} coils but the source artifact has "
                f"{total_loaded_coils}."
            )
        return profile

    matching_profiles = tuple(
        profile
        for profile in FINITE_CURRENT_PROFILES.values()
        if profile.default_total_coils == total_loaded_coils
    )
    if len(matching_profiles) == 1:
        return matching_profiles[0]
    if not matching_profiles:
        supported_totals = ", ".join(
            str(profile.default_total_coils)
            for profile in sorted(
                FINITE_CURRENT_PROFILES.values(),
                key=lambda profile: str(profile.mode),
            )
        )
        raise ValueError(
            f"Cannot partition metadata-free artifact with {total_loaded_coils} "
            f"coils; supported profile totals are {supported_totals}."
        )
    matching_modes = ", ".join(
        sorted(str(profile.mode) for profile in matching_profiles)
    )
    raise ValueError(
        f"Metadata-free artifact with {total_loaded_coils} coils is ambiguous "
        f"between finite-current modes {matching_modes}; pass --finite-current-mode "
        "or --stage2-results."
    )


def _build_finitebuild_banana_coils(
    source_banana_coils: Sequence[Coil],
    config: FiniteBuildExportConfig,
) -> tuple[tuple[Coil, ...], float, float]:
    if not source_banana_coils:
        raise ValueError("Source artifact has no banana coils to convert.")
    nfp = int(config.nfp)
    stellsym = bool(config.stellsym)
    numfilaments_n = int(config.numfilaments_n)
    numfilaments_b = int(config.numfilaments_b)
    symmetry_copies = nfp * (2 if stellsym else 1)
    if len(source_banana_coils) != symmetry_copies:
        raise ValueError(
            "Finite-build export currently rebuilds one HBT master banana coil; "
            f"--nfp={config.nfp} and stellsym={config.stellsym} imply "
            f"{symmetry_copies} source banana coils, got {len(source_banana_coils)}."
        )

    master_banana_coil = source_banana_coils[0]
    master_current_A = (
        float(master_banana_coil.current.get_value())
        if config.banana_current_A is None
        else float(config.banana_current_A)
    )
    nfilaments = numfilaments_n * numfilaments_b
    filament_current_A = master_current_A / float(nfilaments)
    base_filament_curves = create_multifilament_grid(
        master_banana_coil.curve,
        numfilaments_n,
        numfilaments_b,
        float(config.gapsize_n),
        float(config.gapsize_b),
        rotation_order=config.rotation_order,
        frame=config.frame,
    )
    base_filament_currents = tuple(
        Current(filament_current_A) for _ in base_filament_curves
    )
    for current in base_filament_currents:
        current.fix_all()
    finitebuild_curves = apply_symmetries_to_curves(
        base_filament_curves,
        nfp,
        stellsym,
    )
    finitebuild_currents = apply_symmetries_to_currents(
        base_filament_currents,
        nfp,
        stellsym,
    )
    return (
        tuple(
            Coil(curve, current)
            for curve, current in zip(finitebuild_curves, finitebuild_currents)
        ),
        master_current_A,
        filament_current_A,
    )


def _build_banana_only_output_coils(
    partitions: FiniteBuildCoilPartitions,
    config: FiniteBuildExportConfig,
) -> tuple[
    tuple[Coil, ...],
    tuple[Coil, ...],
    tuple[Coil, ...],
    tuple[Coil, ...],
    float,
    float,
]:
    finitebuild_banana_coils, banana_current_A, filament_current_A = (
        _build_finitebuild_banana_coils(partitions.banana_coils, config)
    )
    return (
        partitions.tf_coils,
        finitebuild_banana_coils,
        partitions.proxy_coils,
        partitions.vf_coils,
        banana_current_A,
        filament_current_A,
    )


def _build_all_families_output_coils(
    partitions: FiniteBuildCoilPartitions,
    config: FiniteBuildExportConfig,
) -> tuple[
    tuple[Coil, ...],
    tuple[Coil, ...],
    tuple[Coil, ...],
    tuple[Coil, ...],
    None,
    None,
]:
    if not partitions.banana_coils:
        raise ValueError("Source artifact has no banana coils to convert.")
    return (
        _build_finitebuild_loaded_coils(partitions.tf_coils, config),
        _build_finitebuild_loaded_coils(partitions.banana_coils, config),
        _build_finitebuild_loaded_coils(partitions.proxy_coils, config),
        _build_finitebuild_loaded_coils(partitions.vf_coils, config),
        None,
        None,
    )


def _build_finitebuild_loaded_coils(
    source_coils: Sequence[Coil],
    config: FiniteBuildExportConfig,
) -> tuple[Coil, ...]:
    numfilaments_n = int(config.numfilaments_n)
    numfilaments_b = int(config.numfilaments_b)
    nfilaments = numfilaments_n * numfilaments_b
    finitebuild_coils: list[Coil] = []
    for source_coil in source_coils:
        source_current_A = float(source_coil.current.get_value())
        filament_current_A = source_current_A / float(nfilaments)
        filament_curves = create_multifilament_grid(
            source_coil.curve,
            numfilaments_n,
            numfilaments_b,
            float(config.gapsize_n),
            float(config.gapsize_b),
            rotation_order=config.rotation_order,
            frame=config.frame,
        )
        for filament_curve in filament_curves:
            filament_current = Current(filament_current_A)
            filament_current.fix_all()
            finitebuild_coils.append(Coil(filament_curve, filament_current))
    return tuple(finitebuild_coils)


def _read_json_mapping(path: Path) -> Mapping[str, object]:
    with path.expanduser().open("r", encoding="utf-8") as infile:
        payload = json.load(infile)
    if not isinstance(payload, Mapping):
        raise TypeError(
            f"Expected JSON object in {path}; got {type(payload).__name__}."
        )
    return payload


def _partition_counts(partitions: FiniteBuildCoilPartitions) -> dict[str, int]:
    return _coil_counts(
        tf_coils=partitions.tf_coils,
        banana_coils=partitions.banana_coils,
        proxy_coils=partitions.proxy_coils,
        vf_coils=partitions.vf_coils,
    )


def _coil_counts(
    *,
    tf_coils: Sequence[Coil],
    banana_coils: Sequence[Coil],
    proxy_coils: Sequence[Coil],
    vf_coils: Sequence[Coil],
) -> dict[str, int]:
    return {
        "tf": len(tf_coils),
        "banana": len(banana_coils),
        "proxy": len(proxy_coils),
        "vf": len(vf_coils),
        "total": len(tf_coils) + len(banana_coils) + len(proxy_coils) + len(vf_coils),
    }


def _coil_current_sum(coils: Sequence[Coil]) -> float:
    return float(sum(float(coil.current.get_value()) for coil in coils))


def _coil_current_abs_sum(coils: Sequence[Coil]) -> float:
    return float(sum(abs(float(coil.current.get_value())) for coil in coils))


def _coil_current_totals(partitions: FiniteBuildCoilPartitions) -> dict[str, float]:
    values = {
        "tf": _coil_current_sum(partitions.tf_coils),
        "banana": _coil_current_sum(partitions.banana_coils),
        "proxy": _coil_current_sum(partitions.proxy_coils),
        "vf": _coil_current_sum(partitions.vf_coils),
    }
    values["total"] = float(sum(values.values()))
    return values


def _coil_current_abs_totals(partitions: FiniteBuildCoilPartitions) -> dict[str, float]:
    values = {
        "tf": _coil_current_abs_sum(partitions.tf_coils),
        "banana": _coil_current_abs_sum(partitions.banana_coils),
        "proxy": _coil_current_abs_sum(partitions.proxy_coils),
        "vf": _coil_current_abs_sum(partitions.vf_coils),
    }
    values["total"] = float(sum(values.values()))
    return values


def _write_metadata(
    metadata_path: Path,
    *,
    source_path: Path,
    output_path: Path,
    mgrid_path: Path | None,
    mgrid_group_names: Sequence[str],
    config: FiniteBuildExportConfig,
    partitions: FiniteBuildCoilPartitions,
    source_counts: Mapping[str, int],
    output_counts: Mapping[str, int],
    source_current_totals_A: Mapping[str, float],
    source_current_abs_totals_A: Mapping[str, float],
    output_current_totals_A: Mapping[str, float],
    output_current_abs_totals_A: Mapping[str, float],
    banana_current_A: float | None,
    filament_current_A: float | None,
    source_banana_total_current_A: float,
    output_banana_total_current_A: float,
) -> None:
    output_manifest = build_contiguous_manifest(
        num_tf_coils=output_counts["tf"],
        num_banana_coils=output_counts["banana"],
        num_proxy_coils=output_counts["proxy"],
        num_vf_coils=output_counts["vf"],
    )
    metadata = {
        "FINITEBUILD_EXPORT_SCHEMA_VERSION": METADATA_SCHEMA_VERSION,
        "FINITEBUILD_EXPORT_KIND": "biot_savart_finitebuild",
        "EXPORT_SCOPE": "field_evaluation_only_not_optimization_seed",
        "FINITEBUILD_SCOPE": config.finitebuild_scope.replace("-", "_"),
        "SOURCE_PATH": str(source_path),
        "SOURCE_SHA256": _sha256_file(source_path),
        "SOURCE_STAGE2_RESULTS_PATH": None
        if config.stage2_results is None
        else str(config.stage2_results.expanduser().resolve()),
        "OUTPUT_PATH": str(output_path),
        "PARTITION_SOURCE": partitions.source,
        "FINITE_CURRENT_MODE": partitions.finite_current_mode,
        "SOURCE_MANIFEST_IS_LEGACY_INFERRED": partitions.manifest_is_legacy_inferred,
        "SOURCE_COIL_COUNTS": dict(source_counts),
        "OUTPUT_COIL_COUNTS": dict(output_counts),
        "OUTPUT_COIL_GROUPS": output_manifest.to_json_payload(),
        "SOURCE_CURRENT_TOTALS_A": dict(source_current_totals_A),
        "SOURCE_CURRENT_ABS_TOTALS_A": dict(source_current_abs_totals_A),
        "OUTPUT_CURRENT_TOTALS_A": dict(output_current_totals_A),
        "OUTPUT_CURRENT_ABS_TOTALS_A": dict(output_current_abs_totals_A),
        "FINITEBUILD_FILAMENT_SETTINGS": {
            "numfilaments_n": int(config.numfilaments_n),
            "numfilaments_b": int(config.numfilaments_b),
            "gapsize_n": float(config.gapsize_n),
            "gapsize_b": float(config.gapsize_b),
            "rotation_order": config.rotation_order,
            "frame": config.frame,
            "nfp": int(config.nfp),
            "stellsym": bool(config.stellsym),
        },
        "BANANA_CURRENT_A": None
        if banana_current_A is None
        else float(banana_current_A),
        "BANANA_FILAMENT_CURRENT_A": None
        if filament_current_A is None
        else float(filament_current_A),
        "BANANA_SOURCE_TOTAL_CURRENT_A": float(source_banana_total_current_A),
        "BANANA_OUTPUT_TOTAL_CURRENT_A": float(output_banana_total_current_A),
        "BANANA_CURRENT_OVERRIDE": config.banana_current_A is not None,
        "MGRID_EXPORT": _mgrid_metadata(mgrid_path, mgrid_group_names, config),
        "PROVENANCE_NOTE": _provenance_note(config),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _provenance_note(config: FiniteBuildExportConfig) -> str:
    if config.finitebuild_scope == FINITEBUILD_SCOPE_BANANA_ONLY:
        return (
            "Finite-build export preserves existing TF/proxy/VF field sources "
            "and does not synthesize proxy or VF coils. It is not a new physics "
            "model."
        )
    return (
        "Finite-build export converts existing TF/banana/proxy/VF field sources "
        "to finite-build filaments. It does not synthesize missing field sources "
        "and is not a new physics model."
    )


def _resolved_mgrid_output_path(
    *,
    output_path: Path,
    config: FiniteBuildExportConfig,
) -> Path | None:
    if not config.write_mgrid:
        return None
    if config.mgrid_output is None:
        return mgrid_output_path(output_path)
    return config.mgrid_output.expanduser().resolve()


def _write_mgrid(
    output_biot_savart: BiotSavart,
    output_partitions: FiniteBuildCoilPartitions,
    mgrid_path: Path,
    config: FiniteBuildExportConfig,
) -> tuple[str, ...]:
    if config.mgrid_grouping == MGRID_GROUPING_SINGLE:
        return _write_single_group_mgrid(output_biot_savart, mgrid_path, config)
    if config.mgrid_grouping == MGRID_GROUPING_FAMILY:
        return _write_family_group_mgrid(output_partitions, mgrid_path, config)
    raise ValueError(f"Unsupported MGrid grouping {config.mgrid_grouping!r}.")


def _write_single_group_mgrid(
    output_biot_savart: BiotSavart,
    mgrid_path: Path,
    config: FiniteBuildExportConfig,
) -> tuple[str, ...]:
    output_biot_savart.to_mgrid(
        str(mgrid_path),
        nr=int(config.mgrid_nr),
        nz=int(config.mgrid_nz),
        nphi=int(config.mgrid_nphi),
        rmin=float(config.mgrid_rmin),
        rmax=float(config.mgrid_rmax),
        zmin=float(config.mgrid_zmin),
        zmax=float(config.mgrid_zmax),
        nfp=int(config.mgrid_nfp),
    )
    return ("simsopt_coils",)


def _write_family_group_mgrid(
    output_partitions: FiniteBuildCoilPartitions,
    mgrid_path: Path,
    config: FiniteBuildExportConfig,
) -> tuple[str, ...]:
    points_cyl = _mgrid_points_cyl(config)
    mgrid = MGrid(
        nr=int(config.mgrid_nr),
        nz=int(config.mgrid_nz),
        nphi=int(config.mgrid_nphi),
        nfp=int(config.mgrid_nfp),
        rmin=float(config.mgrid_rmin),
        rmax=float(config.mgrid_rmax),
        zmin=float(config.mgrid_zmin),
        zmax=float(config.mgrid_zmax),
    )
    group_names: list[str] = []
    for group_name, coils in _family_mgrid_groups(output_partitions):
        if not coils:
            continue
        br, bp, bz = _evaluate_mgrid_group_field(coils, points_cyl, config)
        mgrid.add_field_cylindrical(br, bp, bz, name=group_name)
        group_names.append(group_name)
    if not group_names:
        raise ValueError("Cannot write family MGrid: no non-empty coil families.")
    mgrid.write(str(mgrid_path))
    return tuple(group_names)


def _mgrid_points_cyl(config: FiniteBuildExportConfig) -> np.ndarray:
    rs = np.linspace(
        float(config.mgrid_rmin),
        float(config.mgrid_rmax),
        int(config.mgrid_nr),
        endpoint=True,
    )
    phis = np.linspace(
        0.0,
        2.0 * np.pi / int(config.mgrid_nfp),
        int(config.mgrid_nphi),
        endpoint=False,
    )
    zs = np.linspace(
        float(config.mgrid_zmin),
        float(config.mgrid_zmax),
        int(config.mgrid_nz),
        endpoint=True,
    )
    phi_grid, z_grid, r_grid = np.meshgrid(phis, zs, rs, indexing="ij")
    points_cyl = np.zeros((r_grid.size, 3))
    points_cyl[:, 0] = r_grid.flatten()
    points_cyl[:, 1] = phi_grid.flatten()
    points_cyl[:, 2] = z_grid.flatten()
    return points_cyl


def _family_mgrid_groups(
    output_partitions: FiniteBuildCoilPartitions,
) -> tuple[tuple[str, tuple[Coil, ...]], ...]:
    return (
        ("tf", output_partitions.tf_coils),
        ("banana", output_partitions.banana_coils),
        ("proxy", output_partitions.proxy_coils),
        ("vf", output_partitions.vf_coils),
    )


def _evaluate_mgrid_group_field(
    coils: Sequence[Coil],
    points_cyl: np.ndarray,
    config: FiniteBuildExportConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    biot_savart = BiotSavart(list(coils))
    biot_savart.set_points_cyl(points_cyl)
    br, bp, bz = biot_savart.B_cyl().T
    grid_shape = (int(config.mgrid_nphi), int(config.mgrid_nz), int(config.mgrid_nr))
    return br.reshape(grid_shape), bp.reshape(grid_shape), bz.reshape(grid_shape)


def _mgrid_metadata(
    mgrid_path: Path | None,
    mgrid_group_names: Sequence[str],
    config: FiniteBuildExportConfig,
) -> dict[str, object] | None:
    if mgrid_path is None:
        return None
    return {
        "path": str(mgrid_path),
        "grouping": config.mgrid_grouping,
        "groups": list(mgrid_group_names),
        "grid": {
            "nr": int(config.mgrid_nr),
            "nz": int(config.mgrid_nz),
            "nphi": int(config.mgrid_nphi),
            "rmin": float(config.mgrid_rmin),
            "rmax": float(config.mgrid_rmax),
            "zmin": float(config.mgrid_zmin),
            "zmax": float(config.mgrid_zmax),
            "nfp": int(config.mgrid_nfp),
        },
        "vmec_recommended_settings": {
            "LFREEB": True,
            "MGRID_FILE": str(mgrid_path),
            "NZETA": int(config.mgrid_nphi),
            "EXTCUR": [1.0 for _ in mgrid_group_names],
        },
        "nphi_nzeta_policy": (
            "Simsopt VMEC docs allow MGrid nphi to be an integer multiple of "
            "VMEC NZETA; this exporter recommends equality for the simple "
            "local example-backed path."
        ),
        "plasma_current_note": (
            "MGrid encodes external coil fields only. Set VMEC plasma current "
            "through NCURR=1, CURTOR, AC, or Simsopt current_profile inputs."
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_lines(result: FiniteBuildExportResult) -> tuple[str, ...]:
    lines = [
        f"source: {result.source_path}",
        f"output: {result.output_path}",
        f"metadata: {result.metadata_path}",
        f"finitebuild_scope: {result.finitebuild_scope}",
        f"finite_current_mode: {result.finite_current_mode}",
        f"source_counts: {result.source_counts}",
        f"output_counts: {result.output_counts}",
    ]
    if result.banana_current_A is not None:
        lines.append(
            "banana_current_A: "
            f"{result.banana_current_A:.12g} "
            f"(filament {result.banana_filament_current_A:.12g}, "
            f"override={result.current_override})"
        )
    lines.append(
        "banana_total_current_A: "
        f"source {result.source_banana_total_current_A:.12g}, "
        f"output {result.output_banana_total_current_A:.12g}"
    )
    if result.mgrid_output_path is not None:
        lines.append(f"mgrid: {result.mgrid_output_path}")
        extcur_values = [1.0 for _ in result.mgrid_group_names]
        lines.append(
            "vmec_mgrid_settings: LFREEB=T "
            f"MGRID_FILE={result.mgrid_output_path} "
            f"NZETA=<mgrid nphi> EXTCUR={extcur_values}"
        )
    return tuple(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_finitebuild_biot_savart(config_from_args(args))
    for line in _summary_lines(result):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
