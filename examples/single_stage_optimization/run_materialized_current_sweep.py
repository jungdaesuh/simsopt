from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from import_provenance import configure_local_simsopt_imports

configure_local_simsopt_imports(__file__)

from banana_opt.current_contracts import (  # noqa: E402
    CURRENT_MODE_ZERO_TOL,
    FiniteCurrentMode,
    HBT_PROXY_VF_CURRENT_RATIO,
    validate_proxy_vf_current_convention_for_mode,
)
from banana_opt.finite_current_materializer import (  # noqa: E402
    FINITE_CURRENT_FIELD_SOURCE_MODES,
    MATERIALIZED_IOTA_TRUST_STATUS_UNRUN,
    MaterializedFiniteCurrentSeedRequest,
    MaterializedFiniteCurrentSeedResult,
    materialize_finite_current_seed,
)
from banana_opt.finite_current_profiles import (  # noqa: E402
    format_proxy_current_sign_convention_help,
    get_finite_current_profile,
)
from banana_opt.json_compat import load_boozer_finite_i  # noqa: E402
from banana_opt.stage2_single_stage_handoff import (  # noqa: E402
    partition_loaded_stage2_coils,
    validate_loaded_seed_current_source_contract,
)
from workflow_runner_common import (  # noqa: E402
    load_stage2_artifact_results,
    write_csv_rows,
    write_json,
)


DEFAULT_SAMPLE_POINTS = np.asarray(
    [[0.25, 0.10, -0.15], [0.35, -0.05, 0.20], [0.55, 0.15, 0.05]],
    dtype=float,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a current sweep from one donor seed and write a "
            "science-gate summary.\n"
            "This wrapper proves field-source materialization and records iota\n"
            "as untrusted until an explicit no-collapse and VMEC curtor "
            "cross-check is supplied."
        ),
        epilog=format_proxy_current_sign_convention_help(
            FINITE_CURRENT_FIELD_SOURCE_MODES,
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-biot-savart",
        required=True,
        type=Path,
        help="Checksum-bound donor BiotSavart seed to rematerialize for each current.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory where per-current seeds and sweep summaries are written.",
    )
    parser.add_argument(
        "--finite-current-mode",
        required=True,
        choices=FINITE_CURRENT_FIELD_SOURCE_MODES,
        help="Finite-current proxy/VF field-source profile to materialize.",
    )
    parser.add_argument(
        "--current-grid-A",
        required=True,
        help=(
            "Comma-separated nonzero proxy-current grid in amperes. Sign "
            "semantics are selected by --finite-current-mode; see the mode "
            "table below. Use a vacuum seed for zero-current replay."
        ),
    )
    parser.add_argument(
        "--stage2-results",
        type=Path,
        default=None,
        help="Optional donor results.json path; defaults to source sibling.",
    )
    parser.add_argument(
        "--equilibrium-file",
        type=Path,
        default=None,
        help="Optional VMEC/wout file for Wataru proxy placement.",
    )
    parser.add_argument(
        "--vf-template-path",
        type=Path,
        default=None,
        help="Optional VF template override for the selected profile.",
    )
    parser.add_argument(
        "--toroidal-flux",
        type=float,
        default=None,
        help="Optional normalized toroidal flux override for Wataru proxy placement.",
    )
    parser.add_argument(
        "--surface-scale-factor",
        type=float,
        default=1.0,
        help="Surface-family scale factor for Wataru proxy placement.",
    )
    parser.add_argument(
        "--nphi",
        type=int,
        default=127,
        help="Toroidal quadrature count passed to the Wataru proxy builder.",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=32,
        help="Poloidal quadrature count passed to the Wataru proxy builder.",
    )
    parser.add_argument(
        "--num-tf-coils",
        type=int,
        default=20,
        help="Expected TF-coil count used to partition the donor seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing per-current artifacts and sweep summaries.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    proxy_currents_A = _parse_current_grid(args.current_grid_A)
    _validate_sweep_current_grid(
        finite_current_mode=args.finite_current_mode,
        proxy_currents_A=proxy_currents_A,
    )
    _validate_sweep_output_paths(
        output_root,
        proxy_currents_A=proxy_currents_A,
        overwrite=bool(args.overwrite),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    sign_summary_fields = get_finite_current_profile(
        args.finite_current_mode,
    ).proxy_current_sign_summary_fields()
    rows = [
        _materialize_current_row(
            args=args,
            output_root=output_root,
            index=index,
            proxy_current_A=proxy_current_A,
        )
        for index, proxy_current_A in enumerate(proxy_currents_A)
    ]
    summary = {
        "source_biot_savart": str(args.source_biot_savart),
        "finite_current_mode": args.finite_current_mode,
        **sign_summary_fields,
        "materialized_iota_trust_status": MATERIALIZED_IOTA_TRUST_STATUS_UNRUN,
        "materialized_iota_trusted": False,
        "materialized_iota_trust_reason": (
            "This sweep materializes prescribed proxy/VF coil fields only. "
            "Trust iota only after an explicit no-collapse gate and VMEC curtor "
            "cross-check over the same current window."
        ),
        "rows": rows,
    }
    write_json(output_root / "materialized_current_sweep_summary.json", summary)
    write_csv_rows(
        output_root / "materialized_current_sweep_summary.csv",
        rows,
        fieldnames=(
            "index",
            "proxy_current_A",
            "proxy_current_mode",
            "proxy_current_sign_convention",
            "proxy_current_sign_frame",
            "proxy_current_signedness",
            "proxy_current_scalar_policy",
            "proxy_current_replay_reference",
            "proxy_current_operator_warning",
            "output_biot_savart",
            "output_results",
            "output_total_coils",
            "realized_proxy_current_A",
            "realized_vf_current_A",
            "boozer_I",
            "field_values_finite",
            "field_norm_min",
            "field_norm_max",
            "boozer_solve_status",
            "iota",
            "field_error",
            "topology_status",
            "vmec_cross_check_status",
            "materialized_iota_trust_status",
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_current_grid(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("--current-grid-A must contain at least one current.")
    return values


def _validate_sweep_current_grid(
    *,
    finite_current_mode: FiniteCurrentMode,
    proxy_currents_A: tuple[float, ...],
) -> None:
    for index, proxy_current_A in enumerate(proxy_currents_A):
        if abs(float(proxy_current_A)) <= CURRENT_MODE_ZERO_TOL:
            raise ValueError(
                f"current-grid entry {index} is zero; proxy-field sweeps require "
                "nonzero currents. Use a vacuum seed for zero-current replay."
            )
        validate_proxy_vf_current_convention_for_mode(
            finite_current_mode,
            proxy_plasma_current_A=float(proxy_current_A),
            vf_current_A=_default_sweep_vf_current_A(
                finite_current_mode=finite_current_mode,
                proxy_current_A=float(proxy_current_A),
            ),
        )


def _default_sweep_vf_current_A(
    *,
    finite_current_mode: FiniteCurrentMode,
    proxy_current_A: float,
) -> float:
    if finite_current_mode == "jhalpern30_proxy_field":
        return float(proxy_current_A) * HBT_PROXY_VF_CURRENT_RATIO
    return abs(float(proxy_current_A)) * HBT_PROXY_VF_CURRENT_RATIO


def _materialize_current_row(
    *,
    args: argparse.Namespace,
    output_root: Path,
    index: int,
    proxy_current_A: float,
) -> dict[str, object]:
    current_root = output_root / _current_output_name(index, proxy_current_A)
    result = materialize_finite_current_seed(
        MaterializedFiniteCurrentSeedRequest(
            source_biot_savart=args.source_biot_savart,
            output_root=current_root,
            finite_current_mode=args.finite_current_mode,
            proxy_current_A=proxy_current_A,
            stage2_results=args.stage2_results,
            equilibrium_file=args.equilibrium_file,
            vf_template_path=args.vf_template_path,
            toroidal_flux=args.toroidal_flux,
            surface_scale_factor=args.surface_scale_factor,
            nphi=args.nphi,
            ntheta=args.ntheta,
            num_tf_coils=args.num_tf_coils,
            overwrite=args.overwrite,
        )
    )
    _validate_materialized_replay_contract(
        result,
        finite_current_mode=args.finite_current_mode,
    )
    sign_summary_fields = get_finite_current_profile(
        args.finite_current_mode,
    ).proxy_current_sign_summary_fields()
    field_norms = _sample_field_norms(result.output_biot_savart)
    field_values_finite = bool(np.all(np.isfinite(field_norms)))
    return {
        "index": index,
        "proxy_current_A": proxy_current_A,
        **sign_summary_fields,
        "output_biot_savart": str(result.output_biot_savart),
        "output_results": str(result.output_results),
        "output_total_coils": result.output_total_coils,
        "realized_proxy_current_A": result.realized_proxy_current_A,
        "realized_vf_current_A": result.realized_vf_current_A,
        "boozer_I": result.boozer_I,
        "field_values_finite": field_values_finite,
        "field_norm_min": float(np.min(field_norms)) if field_values_finite else None,
        "field_norm_max": float(np.max(field_norms)) if field_values_finite else None,
        "boozer_solve_status": "not_run",
        "iota": None,
        "field_error": None,
        "topology_status": "not_run",
        "vmec_cross_check_status": "not_run",
        "materialized_iota_trust_status": MATERIALIZED_IOTA_TRUST_STATUS_UNRUN,
    }


def _current_output_name(index: int, proxy_current_A: float) -> str:
    token = f"{proxy_current_A:+.6g}".replace("+", "p").replace("-", "m")
    return f"current_{index:03d}_{token}A"


def _validate_sweep_output_paths(
    output_root: Path,
    *,
    proxy_currents_A: tuple[float, ...],
    overwrite: bool,
) -> None:
    output_paths = [
        output_root / "materialized_current_sweep_summary.json",
        output_root / "materialized_current_sweep_summary.csv",
    ]
    for index, proxy_current_A in enumerate(proxy_currents_A):
        current_root = output_root / _current_output_name(index, proxy_current_A)
        output_paths.extend(
            [
                current_root / "biot_savart_opt.json",
                current_root / "results.json",
            ]
        )
    for output_path in output_paths:
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Sweep output already exists; pass --overwrite: {output_path}."
            )


def _validate_materialized_replay_contract(
    result: MaterializedFiniteCurrentSeedResult,
    *,
    finite_current_mode: FiniteCurrentMode,
) -> None:
    _, results = load_stage2_artifact_results(result.output_biot_savart)
    bs = load_boozer_finite_i(str(result.output_biot_savart))
    partitions = partition_loaded_stage2_coils(
        bs.coils,
        stage2_results=results,
        requested_num_tf_coils=result.num_tf_coils,
    )
    validate_loaded_seed_current_source_contract(
        finite_current_mode=finite_current_mode,
        effective_current_mode=finite_current_mode,
        plasma_current_A=result.realized_proxy_current_A,
        plasma_current_input_source="physical_A",
        stage2_results=results,
        coil_partitions=partitions,
    )


def _sample_field_norms(biot_savart_path: Path) -> np.ndarray:
    bs = load_boozer_finite_i(str(biot_savart_path))
    bs.set_points(DEFAULT_SAMPLE_POINTS)
    return np.linalg.norm(bs.B(), axis=1)


if __name__ == "__main__":
    raise SystemExit(main())
