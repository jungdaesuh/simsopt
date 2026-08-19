"""One timed fused flat-675 solve, run as a production-tree child process.

The F3 campaign's L1 lane is this file invoked once per leg.  It is the
counterpart of the instrument's ``genuine_675_dynamic_lane.py``: the harness
owns sequencing, environment, and gating; this child owns exactly one solve and
the record of it.

Three properties the charter rests on are enforced here rather than assumed:

* **Import origin.**  Every ``simsopt_jax`` module must resolve inside the
  production tree this file belongs to.  A child that silently imported the
  instrument tree would time a different program than the row claims.
* **Timer construction.**  ``perf_counter`` starts at the top of ``main`` and
  the primary timer stops the moment the solve completes and its endpoint is
  materialized — the same construction the instrument's lane driver uses.
* **Endpoint quality, after timing.**  The endpoint's own ``(iota, G)`` is
  closed by the flat-675 y-solve *after* the primary timer has stopped, so the
  oracle anchor the harness needs costs the timed region nothing.  Its cost is
  reported separately rather than hidden.

The child never adjudicates: it reports counters, walls, and the endpoint, and
writes them for the harness to gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Final

PRODUCTION_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

LANE_JSON_SCHEMA: Final[str] = "flat675-fused-lane.v1"
# Mirrors the harness's contract module; duplicated as a literal so the child
# stays importable with nothing but the production tree on the path.
LANE_NAME: Final[str] = "fused_gpu"


def _require_production_import_origin() -> dict[str, str]:
    """Fail closed unless the flat-675 program came from this tree."""
    import simsopt_jax
    import simsopt_jax_adapters

    origins: dict[str, str] = {}
    for module in (simsopt_jax, simsopt_jax_adapters):
        origin = Path(str(module.__file__)).resolve()
        origins[module.__name__] = str(origin)
        try:
            origin.relative_to(PRODUCTION_ROOT)
        except ValueError as error:
            raise RuntimeError(
                f"{module.__name__} resolved to {origin}, outside the production "
                f"tree {PRODUCTION_ROOT}; the L1 lane would time a foreign program."
            ) from error
    return origins


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--maxiter", required=True, type=int)
    parser.add_argument(
        "--role",
        default="timed",
        choices=("timed", "primer", "probe"),
        help="primer children are discarded; probe children are untimed",
    )
    parser.add_argument(
        "--expected-charter-sha256",
        required=True,
        help="the F3 charter sha the harness is executing under",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    process_start = perf_counter()
    args = _parse_args(argv)
    origins = _require_production_import_origin()

    import jax
    import numpy as np
    from simsopt_jax.examples.single_stage_flat675 import (
        FLAT675_LBFGS_HISTORY,
        FLAT675_LBFGS_MAXLS,
        prepare_single_stage_flat675,
        solve_single_stage_flat675,
    )
    from simsopt_jax.runtime.host_boundary import host_transfer_audit
    from simsopt_jax.solve.driver import Driver
    from simsopt_jax_adapters.geo.flat675 import (
        FLAT675_COIL_SLICE,
        FLAT675_SURFACE_SLICE,
        FLAT675_VESSEL_SLICE,
        bind_flat675_programs,
        build_flat675_boozer_system,
        flat675_candidate_geometry,
        load_flat675_bundle,
        solve_flat675_y_qr,
    )

    bundle = load_flat675_bundle(args.input_manifest.parent)
    programs = bind_flat675_programs(
        material=bundle.material,
        objective_policy=bundle.objective_policy,
        boozer_policy=bundle.boozer_policy,
    )
    start = bundle.start_candidate.outer_vector()
    prepared = prepare_single_stage_flat675(
        objective_fn=programs.objective_fn,
        diagnostics_fn=programs.diagnostics_fn,
        initial_parameters=jax.device_put(start),
        objective_scale=jax.device_put(np.asarray(1.0, dtype=np.float64)),
    )

    # The fused lane's transfer discipline is part of the claim: a solve that
    # reached the host per step would not be the program under test.
    with host_transfer_audit() as audit, jax.transfer_guard("disallow"):
        result = solve_single_stage_flat675(
            prepared,
            driver=Driver.SIMSOPT_LBFGSB,
            max_steps=int(args.maxiter),
            rtol=0.0,
            atol=0.001,
        )
    endpoint = np.asarray(jax.device_get(result.x), dtype=np.float64)
    primary_wall = perf_counter() - process_start
    transfer_ledger = {entry.phase: entry.calls for entry in audit.summary()}

    # --- after the primary timer: the oracle anchor this endpoint needs ---
    endpoint_start = perf_counter()
    endpoint_device = jax.device_put(endpoint)
    geometry = flat675_candidate_geometry(
        bundle.material.boozer,
        endpoint_device[FLAT675_COIL_SLICE],
        endpoint_device[FLAT675_SURFACE_SLICE],
    )
    boozer_system = build_flat675_boozer_system(geometry, bundle.boozer_policy)
    inner_state = solve_flat675_y_qr(
        boozer_system.design_matrix,
        boozer_system.right_hand_side,
    )
    inner_solution = np.asarray(jax.device_get(inner_state.solution), dtype=np.float64)
    endpoint_seconds = perf_counter() - endpoint_start

    payload = {
        "schema": LANE_JSON_SCHEMA,
        "lane": LANE_NAME,
        "role": str(args.role),
        "f3_charter_sha256": str(args.expected_charter_sha256),
        "import_origins": origins,
        "production_root": str(PRODUCTION_ROOT),
        "policy": {
            "method": "L-BFGS-B",
            "maxiter": int(args.maxiter),
            "maxfun": int(args.maxiter) * 20,
            "gtol": 0.001,
            "ftol": 0.0,
            "maxcor": int(FLAT675_LBFGS_HISTORY),
            "maxls": int(FLAT675_LBFGS_MAXLS),
        },
        # The primary timer spans this process from the top of main to the
        # moment the solve's endpoint is on the host; the endpoint y-solve
        # below is charged separately, never to the claim.
        "process_wall_seconds": primary_wall,
        "endpoint_inner_state_seconds": endpoint_seconds,
        "host_transfer_ledger": transfer_ledger,
        "result": {
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "objective_value": float(result.fun),
            "success": bool(result.success),
            "endpoint_coordinates": endpoint.tolist(),
            "endpoint_candidate": {
                "coil_coordinates": endpoint[FLAT675_COIL_SLICE].tolist(),
                "vessel_coordinates": endpoint[FLAT675_VESSEL_SLICE].tolist(),
                "surface_coordinates": endpoint[FLAT675_SURFACE_SLICE].tolist(),
            },
            "endpoint_inner_state": [
                float(inner_solution[0]),
                float(inner_solution[1]),
            ],
            "endpoint_inner_state_numerics_finite": bool(inner_state.numerics_finite),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"{primary_wall:.6f} {int(result.nfev)} {int(result.nit)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
