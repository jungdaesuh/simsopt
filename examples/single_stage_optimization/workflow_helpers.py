from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def format_compact_float(value: float) -> str:
    return f"{value:g}"


@dataclass(frozen=True)
class Stage2SeedSpec:
    plasma_surf_filename: str
    major_radius: float
    toroidal_flux: float
    length_weight: float
    cc_weight: float
    cc_threshold: float
    curvature_weight: float
    curvature_threshold: float
    banana_surf_radius: float
    tf_current_A: float
    order: int
    banana_init_current_A: float = 1.0e4
    banana_current_max_A: float = 1.6e4


@dataclass(frozen=True)
class SingleStageWeightCase:
    name: str
    res_weight: float
    iotas_weight: float
    cc_weight: float
    curvature_weight: float
    length_weight: float
    cs_weight: float
    surf_dist_weight: float


def format_local_stage2_seed_dir(spec: Stage2SeedSpec) -> str:
    return (
        f"R0={format_compact_float(spec.major_radius)}"
        f"-s={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CCT={format_compact_float(spec.cc_threshold)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-CT={format_compact_float(spec.curvature_threshold)}"
        f"-SR={spec.banana_surf_radius:0.3f}"
        f"-INITC={format_compact_float(spec.banana_init_current_A)}"
        f"-MAXC={format_compact_float(spec.banana_current_max_A)}"
        f"-TFC={format_compact_float(spec.tf_current_A)}"
        f"-Order={spec.order}"
    )


def format_database_stage2_seed_dir(spec: Stage2SeedSpec) -> str:
    return (
        f"MR={format_compact_float(spec.major_radius)}"
        f"-TF={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-SR={format_compact_float(spec.banana_surf_radius)}"
        f"-INITC={format_compact_float(spec.banana_init_current_A)}"
        f"-TFC={format_compact_float(spec.tf_current_A)}"
        f"-Order={spec.order}"
    )


def format_database_stage2_seed_dir_without_init_current(spec: Stage2SeedSpec) -> str:
    return (
        f"MR={format_compact_float(spec.major_radius)}"
        f"-TF={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-SR={format_compact_float(spec.banana_surf_radius)}"
        f"-TFC={format_compact_float(spec.tf_current_A)}"
        f"-Order={spec.order}"
    )


def format_legacy_database_stage2_seed_dir(spec: Stage2SeedSpec) -> str:
    return (
        f"MR={format_compact_float(spec.major_radius)}"
        f"-TF={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-SR={format_compact_float(spec.banana_surf_radius)}"
        f"-Order={spec.order}"
    )


def format_local_stage2_seed_dir_without_tf(spec: Stage2SeedSpec) -> str:
    return (
        f"R0={format_compact_float(spec.major_radius)}"
        f"-s={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CCT={format_compact_float(spec.cc_threshold)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-CT={format_compact_float(spec.curvature_threshold)}"
        f"-SR={spec.banana_surf_radius:0.3f}"
        f"-Order={spec.order}"
    )


def format_local_stage2_seed_dir_without_init_current(spec: Stage2SeedSpec) -> str:
    return (
        f"R0={format_compact_float(spec.major_radius)}"
        f"-s={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CCT={format_compact_float(spec.cc_threshold)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-CT={format_compact_float(spec.curvature_threshold)}"
        f"-SR={spec.banana_surf_radius:0.3f}"
        f"-TFC={format_compact_float(spec.tf_current_A)}"
        f"-Order={spec.order}"
    )


def format_legacy_local_stage2_seed_dir(spec: Stage2SeedSpec) -> str:
    return (
        f"R0={format_compact_float(spec.major_radius)}"
        f"-s={format_compact_float(spec.toroidal_flux)}"
        f"-LW={format_compact_float(spec.length_weight)}"
        f"-CCW={format_compact_float(spec.cc_weight)}"
        f"-CW={format_compact_float(spec.curvature_weight)}"
        f"-SR={spec.banana_surf_radius:0.3f}"
        f"-Order={spec.order}"
    )


def format_stage2_constraint_suffix(
    constraint_method: str,
    alm_max_outer_iters: int,
    alm_penalty_init: float,
    alm_penalty_scale: float,
) -> str:
    if constraint_method == "alm":
        return (
            f"-CM=alm-ALMOuter={alm_max_outer_iters}"
            f"-ALMMu={format_compact_float(alm_penalty_init)}"
            f"-ALMScale={format_compact_float(alm_penalty_scale)}"
        )
    return "-CM=penalty"


def format_stage2_basin_suffix(
    basin_hops: int,
    basin_stepsize: float,
    basin_temperature: float = 1.0,
    basin_niter_success: int = 0,
    basin_seed: int | None = None,
) -> str:
    if basin_hops <= 0:
        return ""
    seed_value = "none" if basin_seed is None else str(basin_seed)
    suffix = (
        f"-BH={basin_hops}"
        f"-BS={format_compact_float(basin_stepsize)}"
        f"-BSeed={seed_value}"
    )
    if basin_temperature != 1.0:
        suffix += f"-BT={format_compact_float(basin_temperature)}"
    if basin_niter_success > 0:
        suffix += f"-BNS={basin_niter_success}"
    return suffix


def format_local_stage2_run_dir(
    spec: Stage2SeedSpec,
    *,
    constraint_method: str,
    alm_max_outer_iters: int,
    alm_penalty_init: float,
    alm_penalty_scale: float,
    basin_hops: int,
    basin_stepsize: float,
    basin_temperature: float = 1.0,
    basin_niter_success: int = 0,
    basin_seed: int | None = None,
) -> str:
    return (
        format_local_stage2_seed_dir(spec)
        + format_stage2_constraint_suffix(
            constraint_method,
            alm_max_outer_iters,
            alm_penalty_init,
            alm_penalty_scale,
        )
        + format_stage2_basin_suffix(
            basin_hops,
            basin_stepsize,
            basin_temperature,
            basin_niter_success,
            basin_seed,
        )
    )


def local_stage2_bs_path(
    output_root: str | Path,
    spec: Stage2SeedSpec,
    *,
    constraint_method: str,
    alm_max_outer_iters: int,
    alm_penalty_init: float,
    alm_penalty_scale: float,
    basin_hops: int,
    basin_stepsize: float,
    basin_temperature: float = 1.0,
    basin_niter_success: int = 0,
    basin_seed: int | None = None,
) -> Path:
    return (
        Path(output_root)
        / f"outputs-{spec.plasma_surf_filename}"
        / format_local_stage2_run_dir(
            spec,
            constraint_method=constraint_method,
            alm_max_outer_iters=alm_max_outer_iters,
            alm_penalty_init=alm_penalty_init,
            alm_penalty_scale=alm_penalty_scale,
            basin_hops=basin_hops,
            basin_stepsize=basin_stepsize,
            basin_temperature=basin_temperature,
            basin_niter_success=basin_niter_success,
            basin_seed=basin_seed,
        )
        / "biot_savart_opt.json"
    )


def build_weight_cases(
    base_weights: Mapping[str, float],
    scan_weights: Sequence[str],
    multipliers: Sequence[float],
) -> list[SingleStageWeightCase]:
    baseline = SingleStageWeightCase(
        name="baseline",
        res_weight=float(base_weights["res_weight"]),
        iotas_weight=float(base_weights["iotas_weight"]),
        cc_weight=float(base_weights["cc_weight"]),
        curvature_weight=float(base_weights["curvature_weight"]),
        length_weight=float(base_weights["length_weight"]),
        cs_weight=float(base_weights["cs_weight"]),
        surf_dist_weight=float(base_weights["surf_dist_weight"]),
    )
    cases = [baseline]
    seen = {
        (
            baseline.res_weight,
            baseline.iotas_weight,
            baseline.cc_weight,
            baseline.curvature_weight,
            baseline.length_weight,
            baseline.cs_weight,
            baseline.surf_dist_weight,
        )
    }
    for weight_name in scan_weights:
        for multiplier in multipliers:
            scaled_multiplier = float(multiplier)
            if scaled_multiplier == 1.0:
                continue
            fields = {
                "res_weight": baseline.res_weight,
                "iotas_weight": baseline.iotas_weight,
                "cc_weight": baseline.cc_weight,
                "curvature_weight": baseline.curvature_weight,
                "length_weight": baseline.length_weight,
                "cs_weight": baseline.cs_weight,
                "surf_dist_weight": baseline.surf_dist_weight,
            }
            fields[weight_name] = fields[weight_name] * scaled_multiplier
            signature = (
                fields["res_weight"],
                fields["iotas_weight"],
                fields["cc_weight"],
                fields["curvature_weight"],
                fields["length_weight"],
                fields["cs_weight"],
                fields["surf_dist_weight"],
            )
            if signature in seen:
                continue
            seen.add(signature)
            multiplier_label = str(scaled_multiplier).replace("-", "m").replace(".", "p")
            cases.append(
                SingleStageWeightCase(
                    name=f"{weight_name}-x{multiplier_label}",
                    **fields,
                )
            )
    return cases


def augment_metrics(record: Mapping[str, object]) -> dict[str, float]:
    augmented = {
        key: float(value)
        for key, value in record.items()
        if isinstance(value, (int, float))
    }
    final_iota = record.get("FINAL_IOTA")
    target_iota = record.get("TARGET_IOTA")
    if isinstance(final_iota, (int, float)) and isinstance(target_iota, (int, float)):
        augmented["IOTA_ERROR_ABS"] = abs(float(final_iota) - float(target_iota))
    return augmented


def dominates(
    lhs: Mapping[str, object],
    rhs: Mapping[str, object],
    metrics: Sequence[str],
) -> bool:
    lhs_metrics = augment_metrics(lhs)
    rhs_metrics = augment_metrics(rhs)
    if any(metric not in lhs_metrics or metric not in rhs_metrics for metric in metrics):
        return False
    return all(lhs_metrics[metric] <= rhs_metrics[metric] for metric in metrics) and any(
        lhs_metrics[metric] < rhs_metrics[metric] for metric in metrics
    )


def select_non_dominated_records(
    records: Iterable[Mapping[str, object]],
    metrics: Sequence[str],
) -> list[Mapping[str, object]]:
    materialized = list(records)
    return [
        record
        for index, record in enumerate(materialized)
        if not any(
            other_index != index and dominates(other, record, metrics)
            for other_index, other in enumerate(materialized)
        )
    ]
