"""Flat-675 promotion robustness solves, run as a production-tree child.

F4 work package C3 (charter ``docs/jax_flat675_promotion_plan.md``,
requirement 4).  All sealed flat-675 evidence is one start candidate; this
child produces the multi-start evidence that bounds how far that generalizes.

It owns the solves and nothing else.  The harness
(``flat675_promotion_robustness.py``) owns sequencing, the native oracle, the
gates and the record — the same split the F3 campaign uses, and for the same
reason: the oracle resolves its imports from the pinned instrument tree while
the lane under test must resolve only from this one, so the two cannot share a
process.

**No timing claim is made here or anywhere in C3.**  The seconds this file
records are incidental and non-verdict; they exist so a reader can tell a
finished run from a hung one.  That is also why every bundle-problem start is
solved in ONE process: the perturbed starts are shape-identical to the
archived one, so a single compiled program serves all of them, and there is no
warm/cold distinction to protect.

Two modes:

``bundle``
    The archived frozen bundle's start (the control) plus seeded
    perturbations of it.  A perturbation is drawn as a Gaussian direction,
    normalized, and scaled so that ``||delta||_2 / ||block||_2`` is EXACTLY
    the nominal relative amplitude — the achieved amplitude is recorded
    beside the nominal one so the two can never silently differ.

``constructor``
    The start the shipped example builds from repository geometry.  It is
    loaded from the example file itself rather than rebuilt here: a copy of
    the geometry would be a twin, and the point is to exercise the start
    users actually get.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter
from typing import Final

PRODUCTION_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

CHILD_JSON_SCHEMA: Final[str] = "flat675-promotion-robustness-child.v1"

EXAMPLE_PATH: Final[Path] = (
    PRODUCTION_ROOT / "examples" / "jax" / "3_Advanced" / "single_stage_flat675.py"
)

# The perturbation grid the charter names: three relative amplitudes across
# three blocks of the outer vector.
PERTURBATION_AMPLITUDES: Final[tuple[float, ...]] = (1.0e-3, 1.0e-2, 1.0e-1)
PERTURBATION_BLOCKS: Final[tuple[str, ...]] = ("surface", "coil", "full")


def _require_production_import_origin() -> dict[str, str]:
    """Fail closed unless the flat-675 program came from this tree."""
    import simsopt_jax
    import simsopt_jax_adapters

    origins: dict[str, str] = {}
    for module in (simsopt_jax, simsopt_jax_adapters):
        origin = Path(str(module.__file__)).resolve()
        if PRODUCTION_ROOT not in origin.parents:
            raise RuntimeError(
                f"{module.__name__} resolved to {origin}, outside {PRODUCTION_ROOT}."
            )
        origins[module.__name__] = str(origin)
    return origins


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bundle", "constructor"), required=True)
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--maxiter", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "bundle" and args.input_manifest is None:
        parser.error("--input-manifest is required in bundle mode")
    return args


def _example_module():
    """Load the shipped example so its own start is what gets solved.

    The tier directories are not packages, so a file-location import is the
    only route to the script.  Rebuilding its geometry here instead would put
    a second copy of the configuration in the tree, which is precisely the
    twin this program has paid for before.
    """
    specification = importlib.util.spec_from_file_location(
        "flat675_single_stage_example", EXAMPLE_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load the shipped example at {EXAMPLE_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _perturbed_starts(start, seed: int) -> list[dict[str, object]]:
    """The control plus one seeded perturbation per (amplitude, block).

    Every draw is reproducible from ``seed`` alone and independent of the
    order the runs happen to execute in, so a single re-run reproduces any
    one row without reproducing the whole grid.
    """
    import numpy as np
    from simsopt_jax_adapters.geo.flat675 import (
        FLAT675_COIL_SLICE,
        FLAT675_SURFACE_SLICE,
    )

    baseline = np.asarray(start, dtype=np.float64)
    slices = {
        "surface": FLAT675_SURFACE_SLICE,
        "coil": FLAT675_COIL_SLICE,
        "full": slice(0, baseline.shape[0]),
    }

    runs: list[dict[str, object]] = [
        {
            "run_id": "control",
            "block": None,
            "nominal_amplitude": 0.0,
            "achieved_amplitude": 0.0,
            "seed": None,
            "start": baseline.copy(),
        }
    ]
    for amplitude_index, amplitude in enumerate(PERTURBATION_AMPLITUDES):
        for block_index, block in enumerate(PERTURBATION_BLOCKS):
            draw_seed = [int(seed), amplitude_index, block_index]
            generator = np.random.default_rng(draw_seed)
            target = slices[block]
            values = baseline[target]
            direction = generator.standard_normal(values.shape[0])
            direction_norm = float(np.linalg.norm(direction))
            block_norm = float(np.linalg.norm(values))
            delta = direction / direction_norm * (amplitude * block_norm)
            perturbed = baseline.copy()
            perturbed[target] = values + delta
            runs.append(
                {
                    "run_id": f"{block}-{amplitude:g}",
                    "block": block,
                    "nominal_amplitude": float(amplitude),
                    # Recorded, not assumed: the normalization makes these
                    # equal by construction, and a drift would say the draw
                    # changed.
                    "achieved_amplitude": float(
                        np.linalg.norm(perturbed - baseline) / block_norm
                    ),
                    "seed": draw_seed,
                    "start": perturbed,
                }
            )
    return runs


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

    if args.mode == "bundle":
        problem = load_flat675_bundle(args.input_manifest.parent)
        configuration = "certified-frozen-bundle"
    else:
        problem = _example_module()._repository_problem("native_default")
        configuration = "repository-geometry"

    programs = bind_flat675_programs(
        material=problem.material,
        objective_policy=problem.objective_policy,
        boozer_policy=problem.boozer_policy,
    )
    baseline_start = np.asarray(
        problem.start_candidate.outer_vector(), dtype=np.float64
    )

    if args.mode == "bundle":
        planned = _perturbed_starts(baseline_start, int(args.seed))
    else:
        planned = [
            {
                "run_id": "constructor",
                "block": None,
                "nominal_amplitude": 0.0,
                "achieved_amplitude": 0.0,
                "seed": None,
                "start": baseline_start.copy(),
            }
        ]

    objective_of = jax.jit(programs.objective_fn)
    gradient_of = jax.jit(jax.grad(programs.objective_fn))

    def inner_state(candidate) -> list[float]:
        """The endpoint's own ``(iota, G)``, which the oracle takes as anchor."""
        device_candidate = jax.device_put(candidate)
        geometry = flat675_candidate_geometry(
            problem.material.boozer,
            device_candidate[FLAT675_COIL_SLICE],
            device_candidate[FLAT675_SURFACE_SLICE],
        )
        system = build_flat675_boozer_system(geometry, problem.boozer_policy)
        solution = solve_flat675_y_qr(system.design_matrix, system.right_hand_side)
        return [
            float(value)
            for value in np.asarray(jax.device_get(solution.solution), dtype=np.float64)
        ]

    records: list[dict[str, object]] = []
    for plan in planned:
        start_vector = np.asarray(plan["start"], dtype=np.float64)
        run_start = perf_counter()
        prepared = prepare_single_stage_flat675(
            objective_fn=programs.objective_fn,
            diagnostics_fn=programs.diagnostics_fn,
            initial_parameters=jax.device_put(start_vector),
            objective_scale=jax.device_put(np.asarray(1.0, dtype=np.float64)),
        )
        with host_transfer_audit() as audit, jax.transfer_guard("disallow"):
            result = solve_single_stage_flat675(
                prepared,
                driver=Driver.SIMSOPT_LBFGSB,
                max_steps=int(args.maxiter),
                rtol=0.0,
                atol=0.001,
            )
        endpoint = np.asarray(jax.device_get(result.x), dtype=np.float64)
        ledger = {entry.phase: entry.calls for entry in audit.summary()}

        start_device = jax.device_put(start_vector)
        endpoint_device = jax.device_put(endpoint)
        start_objective = float(jax.device_get(objective_of(start_device)))
        endpoint_objective = float(jax.device_get(objective_of(endpoint_device)))
        start_gradient = np.asarray(
            jax.device_get(gradient_of(start_device)), dtype=np.float64
        )
        endpoint_gradient = np.asarray(
            jax.device_get(gradient_of(endpoint_device)), dtype=np.float64
        )

        records.append(
            {
                "run_id": plan["run_id"],
                "block": plan["block"],
                "nominal_amplitude": plan["nominal_amplitude"],
                "achieved_amplitude": plan["achieved_amplitude"],
                "draw_seed": plan["seed"],
                "start_objective": start_objective,
                "endpoint_objective": endpoint_objective,
                "solver_endpoint_objective": float(result.fun),
                "start_gradient_inf_norm": float(np.max(np.abs(start_gradient))),
                "endpoint_gradient_inf_norm": float(np.max(np.abs(endpoint_gradient))),
                "iterations_run": int(result.nit),
                "objective_evaluations": int(result.nfev),
                "endpoint_finite": bool(np.all(np.isfinite(endpoint))),
                "objective_finite": bool(
                    np.isfinite(start_objective) and np.isfinite(endpoint_objective)
                ),
                "host_transfer_ledger": ledger,
                "endpoint_inner_state": inner_state(endpoint),
                "candidate": {
                    "coil_coordinates": [
                        float(v) for v in endpoint[FLAT675_COIL_SLICE]
                    ],
                    "vessel_coordinates": [
                        float(v) for v in endpoint[FLAT675_VESSEL_SLICE]
                    ],
                    "surface_coordinates": [
                        float(v) for v in endpoint[FLAT675_SURFACE_SLICE]
                    ],
                },
                # Incidental and non-verdict: C3 makes no timing claim.
                "incidental_run_seconds": perf_counter() - run_start,
            }
        )

    payload = {
        "schema": CHILD_JSON_SCHEMA,
        "mode": str(args.mode),
        "configuration": configuration,
        "import_origins": origins,
        "production_root": str(PRODUCTION_ROOT),
        "jax_platform": str(jax.devices()[0].platform),
        "jax_device": str(jax.devices()[0]),
        "seed": int(args.seed),
        "policy": {
            "method": "L-BFGS-B",
            "maxiter": int(args.maxiter),
            "maxcor": int(FLAT675_LBFGS_HISTORY),
            "maxls": int(FLAT675_LBFGS_MAXLS),
        },
        "runs": records,
        "incidental_process_seconds": perf_counter() - process_start,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": args.mode, "runs": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
