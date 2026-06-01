from __future__ import annotations

"""Export/evaluation-only finite-build BiotSavart conversion for HBT banana coils.

The generated finite-build artifact is not a new physics model and is not an
optimization warm-start contract.
"""

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
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


@dataclass(frozen=True)
class FiniteBuildExportConfig:
    """Configuration for converting one loaded HBT banana coil bundle to finite build."""

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
    source_path: Path
    source_counts: dict[str, int]
    output_counts: dict[str, int]
    banana_current_A: float
    banana_filament_current_A: float
    source_banana_total_current_A: float
    output_banana_total_current_A: float
    current_override: bool
    finite_current_mode: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a single-filament HBT banana BiotSavart artifact into a "
            "finite-build export artifact. The output is for field evaluation "
            "and external export, not single-stage optimization warm starts."
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
        help="Override the master banana current in amperes.",
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
    _validate_output_paths(
        source_path,
        output_path=output_path,
        metadata_path=metadata_path,
        overwrite=config.overwrite,
    )

    source_biot_savart = normalize_biot_savart_source(
        load_boozer_finite_i(str(source_path))
    )
    source_coils = tuple(source_biot_savart.coils)
    partitions = _resolve_partitions(source_coils, config)
    source_counts = _partition_counts(partitions)
    source_banana_total_current_A = _coil_current_sum(partitions.banana_coils)

    finitebuild_banana_coils, banana_current_A, filament_current_A = (
        _build_finitebuild_banana_coils(partitions.banana_coils, config)
    )
    output_banana_total_current_A = _coil_current_sum(finitebuild_banana_coils)
    output_coils = [
        *partitions.tf_coils,
        *finitebuild_banana_coils,
        *partitions.proxy_coils,
        *partitions.vf_coils,
    ]
    output_biot_savart = BiotSavart(output_coils)
    output_biot_savart.save(str(output_path))

    output_counts = _coil_counts(
        tf_coils=partitions.tf_coils,
        banana_coils=finitebuild_banana_coils,
        proxy_coils=partitions.proxy_coils,
        vf_coils=partitions.vf_coils,
    )
    _write_metadata(
        metadata_path,
        source_path=source_path,
        output_path=output_path,
        config=config,
        partitions=partitions,
        source_counts=source_counts,
        output_counts=output_counts,
        banana_current_A=banana_current_A,
        filament_current_A=filament_current_A,
        source_banana_total_current_A=source_banana_total_current_A,
        output_banana_total_current_A=output_banana_total_current_A,
    )

    return FiniteBuildExportResult(
        output_path=output_path,
        metadata_path=metadata_path,
        source_path=source_path,
        source_counts=source_counts,
        output_counts=output_counts,
        banana_current_A=banana_current_A,
        banana_filament_current_A=filament_current_A,
        source_banana_total_current_A=source_banana_total_current_A,
        output_banana_total_current_A=output_banana_total_current_A,
        current_override=config.banana_current_A is not None,
        finite_current_mode=partitions.finite_current_mode,
    )


def _validate_config(config: FiniteBuildExportConfig) -> None:
    if int(config.numfilaments_n) <= 0:
        raise ValueError("--numfilaments-n must be positive.")
    if int(config.numfilaments_b) <= 0:
        raise ValueError("--numfilaments-b must be positive.")
    if float(config.gapsize_n) <= 0.0:
        raise ValueError("--gapsize-n must be positive.")
    if float(config.gapsize_b) <= 0.0:
        raise ValueError("--gapsize-b must be positive.")
    if int(config.nfp) <= 0:
        raise ValueError("--nfp must be positive.")


def _validate_output_paths(
    source_path: Path,
    *,
    output_path: Path,
    metadata_path: Path,
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
    for path in (output_path, metadata_path):
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


def _write_metadata(
    metadata_path: Path,
    *,
    source_path: Path,
    output_path: Path,
    config: FiniteBuildExportConfig,
    partitions: FiniteBuildCoilPartitions,
    source_counts: Mapping[str, int],
    output_counts: Mapping[str, int],
    banana_current_A: float,
    filament_current_A: float,
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
        "BANANA_CURRENT_A": float(banana_current_A),
        "BANANA_FILAMENT_CURRENT_A": float(filament_current_A),
        "BANANA_SOURCE_TOTAL_CURRENT_A": float(source_banana_total_current_A),
        "BANANA_OUTPUT_TOTAL_CURRENT_A": float(output_banana_total_current_A),
        "BANANA_CURRENT_OVERRIDE": config.banana_current_A is not None,
        "PROVENANCE_NOTE": (
            "Finite-build export preserves existing TF/proxy/VF field sources "
            "and does not synthesize proxy or VF coils. It is not a new physics "
            "model."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_lines(result: FiniteBuildExportResult) -> tuple[str, ...]:
    return (
        f"source: {result.source_path}",
        f"output: {result.output_path}",
        f"metadata: {result.metadata_path}",
        f"finite_current_mode: {result.finite_current_mode}",
        f"source_counts: {result.source_counts}",
        f"output_counts: {result.output_counts}",
        (
            "banana_current_A: "
            f"{result.banana_current_A:.12g} "
            f"(filament {result.banana_filament_current_A:.12g}, "
            f"override={result.current_override})"
        ),
        (
            "banana_total_current_A: "
            f"source {result.source_banana_total_current_A:.12g}, "
            f"output {result.output_banana_total_current_A:.12g}"
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_finitebuild_biot_savart(config_from_args(args))
    for line in _summary_lines(result):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
