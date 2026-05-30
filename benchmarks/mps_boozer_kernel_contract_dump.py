"""Dump the flattened Boozer MPS custom-kernel contract artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.single_stage_smoke_fixture import (  # noqa: E402
    DEFAULT_OPTIMIZER_BACKEND,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
    build_real_single_stage_init_fixture,
)
from benchmarks.validation_ladder_common import (  # noqa: E402
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    require_requested_platform_runtime,
    require_x64_runtime,
)

PAYLOAD_SCHEMA = "simsopt.mps_boozer_kernel_contract_dump.v1"
DEFAULT_OUTPUT_DIR = Path(".artifacts/mps_custom_kernel_contract")
DEFAULT_CASE_LABEL = "single_stage_smoke_mpol2_ntor2"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the SIMSOPT Boozer MPS custom-kernel flattened contract "
            "artifact for the reduced single-stage smoke fixture."
        )
    )
    parser.add_argument("--platform", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--case-label", default=DEFAULT_CASE_LABEL)
    parser.add_argument("--mpol", type=int, default=DEFAULT_SMOKE_MPOL)
    parser.add_argument("--ntor", type=int, default=DEFAULT_SMOKE_NTOR)
    parser.add_argument("--nphi", type=int, default=DEFAULT_SMOKE_NPHI)
    parser.add_argument("--ntheta", type=int, default=DEFAULT_SMOKE_NTHETA)
    parser.add_argument("--optimizer-backend", default=DEFAULT_OPTIMIZER_BACKEND)
    parser.add_argument("--boozer-optimizer-backend", default=None)
    return parser


def _contract_helpers():
    from simsopt.jax_core.mps_boozer_kernel_contract import (
        build_mps_boozer_direct_kernel_contract,
        evaluate_mps_boozer_direct_cpu_oracle,
        evaluate_mps_boozer_fused_solve_cpu_oracle,
        mps_boozer_kernel_contract_artifact,
    )

    return (
        build_mps_boozer_direct_kernel_contract,
        evaluate_mps_boozer_direct_cpu_oracle,
        evaluate_mps_boozer_fused_solve_cpu_oracle,
        mps_boozer_kernel_contract_artifact,
    )


def solved_state_from_boozer_result(result: dict[str, Any]) -> SimpleNamespace:
    if not result.get("success", False):
        raise RuntimeError("Cannot dump MPS Boozer contract from a failed solve.")
    return SimpleNamespace(
        iota=result["iota"],
        G=result["G"],
        sdofs=result["sdofs"],
        weight_inv_modB=result["weight_inv_modB"],
    )


def build_mps_boozer_contract_payload(
    boozer_residual: object,
    solved_state: object,
    *,
    case_label: str,
    fixture_metadata: dict[str, Any],
    coil_dofs: object | None = None,
) -> dict[str, Any]:
    """Build a JSON-ready contract payload from an existing solved state."""
    (
        build_contract,
        evaluate_oracle,
        evaluate_fused_solve_oracle,
        contract_artifact,
    ) = _contract_helpers()
    contract = build_contract(
        boozer_residual,
        solved_state=solved_state,
        coil_dofs=coil_dofs,
    )
    oracle_result = evaluate_oracle(boozer_residual, contract)
    fused_oracle_result = None
    if coil_dofs is None:
        fused_oracle_result = evaluate_fused_solve_oracle(boozer_residual, contract)
    return {
        "schema": PAYLOAD_SCHEMA,
        "case_label": str(case_label),
        "fixture_metadata": fixture_metadata,
        "contract_artifact": contract_artifact(
            contract,
            oracle_result=oracle_result,
            fused_oracle_result=fused_oracle_result,
        ),
    }


def build_single_stage_smoke_contract_payload(
    args: argparse.Namespace,
) -> dict[str, Any]:
    from simsopt.geo.surfaceobjectives_jax import BoozerResidualJAX

    fixture = build_real_single_stage_init_fixture(
        backend="jax",
        nphi=args.nphi,
        ntheta=args.ntheta,
        mpol=args.mpol,
        ntor=args.ntor,
        optimizer_backend=args.optimizer_backend,
        boozer_optimizer_backend=args.boozer_optimizer_backend,
    )
    boozer_surface = fixture["boozer_surface"]
    result = boozer_surface.res
    if result is None:
        raise RuntimeError("Cannot dump MPS Boozer contract before Boozer solve.")
    solved_state = solved_state_from_boozer_result(result)
    residual = BoozerResidualJAX(boozer_surface, fixture["bs"])
    fixture_metadata = {
        "fixture": "single_stage_smoke",
        "backend": "jax",
        "optimizer_backend": args.optimizer_backend,
        "boozer_optimizer_backend": fixture["boozer_optimizer_backend"],
        "equilibrium_path": fixture["equilibrium_path"],
        "stage2_bs_path": fixture["stage2_bs_path"],
        "surface_shape": fixture["surface_shape"],
    }
    return build_mps_boozer_contract_payload(
        residual,
        solved_state,
        case_label=args.case_label,
        fixture_metadata=fixture_metadata,
    )


def write_mps_boozer_contract_payload(
    payload: dict[str, Any],
    *,
    output_dir: str | Path,
    case_label: str,
) -> Path:
    destination = Path(output_dir) / f"{case_label}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    args = _build_parser().parse_args(raw_argv)
    requested_platform = args.platform
    apply_requested_platform(requested_platform)
    bootstrap_local_simsopt()

    import jax
    import jaxlib

    jax.config.update("jax_enable_x64", True)
    require_x64_runtime(jax, context="MPS Boozer contract dump")
    require_requested_platform_runtime(
        jax,
        requested_platform=requested_platform,
        context="MPS Boozer contract dump",
    )

    payload = build_single_stage_smoke_contract_payload(args)
    payload["provenance"] = build_provenance(
        jax,
        jaxlib,
        title="MPS Boozer custom-kernel contract dump",
        extra={
            "platform_request": args.platform,
            "case_label": args.case_label,
        },
    )
    output_path = write_mps_boozer_contract_payload(
        payload,
        output_dir=args.output_dir,
        case_label=args.case_label,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
