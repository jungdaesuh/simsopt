"""Automatic resonance targeting CLI for the Greene residue objective.

Loads a coil ``BiotSavart`` artifact and a candidate-targets JSON, discovers and
ranks the magnetic islands by width, runs the real residue-objective validation
kernels on the worst islands, and writes consumer-ready ``targets``/``seeds``
JSON that ``single_stage_banana_example.py`` (or ``run_residue_probe.py``) can
feed to ``--residue-objective-weight`` without bypassing any validation gate.

Example
-------
    python build_residue_seeds.py \
        --biot-savart-json run_dir/biot_savart_opt.json \
        --targets-json residue_targets.json \
        --axis-r 1.0 \
        --validation-artifact-id auto-2026-06-17 \
        --targets-out residue_targets.resolved.json \
        --seeds-out residue_seeds.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from import_provenance import configure_local_simsopt_imports

EXAMPLE_ROOT, SIMSOPT_ROOT, SRC_ROOT = configure_local_simsopt_imports(__file__)

from banana_opt.json_compat import load_boozer_finite_i as load  # noqa: E402
from banana_opt.topology.fieldline_map import (  # noqa: E402
    FieldlineIntegratorOptions,
)
from banana_opt.topology.periodic_orbit import (  # noqa: E402
    PeriodicOrbitSolverOptions,
)
from banana_opt.topology.poincare_chart import PoincareChart  # noqa: E402
from banana_opt.topology.residue_diagnostics import (  # noqa: E402
    uniform_branch_phase_angles,
)
from banana_opt.topology.residue_objective import (  # noqa: E402
    DEFAULT_RESIDUE_OBJECTIVE_SAMPLES_PER_FULL_TORUS,
    load_residue_objective_targets,
)
from banana_opt.topology.residue_seed_builder import (  # noqa: E402
    DEFAULT_RESIDUE_DIFFERENCE_TOLERANCE,
    DEFAULT_REAL_FIELD_WINDING_TOLERANCE,
    DEFAULT_TAYLOR_PROBE_STEPS,
    generate_residue_seed_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Discover, rank, and validate magnetic islands, then emit "
            "consumer-ready Greene residue targets + seeds JSON."
        )
    )
    parser.add_argument("--biot-savart-json", required=True)
    parser.add_argument("--targets-json", required=True)
    parser.add_argument("--validation-artifact-id", required=True)
    parser.add_argument("--targets-out", required=True)
    parser.add_argument("--seeds-out", required=True)

    parser.add_argument("--axis-r", type=float, required=True)
    parser.add_argument("--axis-z", type=float, default=0.0)
    parser.add_argument("--poloidal-orientation", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--radial-label-scale", type=float, default=1.0)

    parser.add_argument("--rtol", type=float, default=1.0e-9)
    parser.add_argument("--atol", type=float, default=1.0e-11)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument(
        "--samples-per-full-torus",
        type=int,
        default=DEFAULT_RESIDUE_OBJECTIVE_SAMPLES_PER_FULL_TORUS,
    )
    parser.add_argument("--min-bphi-over-b", type=float, default=1.0e-7)
    parser.add_argument("--max-fieldline-rhs-evals", type=int, default=None)

    parser.add_argument("--newton-residual-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--winding-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--det-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--max-newton-iterations", type=int, default=20)
    parser.add_argument("--max-newton-step-norm", type=float, default=0.02)

    parser.add_argument("--branch-phase-count", type=int, default=8)
    parser.add_argument("--min-island-width", type=float, default=0.0)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--taylor-probe-steps",
        type=float,
        nargs="+",
        default=list(DEFAULT_TAYLOR_PROBE_STEPS),
    )
    parser.add_argument(
        "--residue-difference-tolerance",
        type=float,
        default=DEFAULT_RESIDUE_DIFFERENCE_TOLERANCE,
    )
    parser.add_argument(
        "--real-field-winding-tolerance",
        type=float,
        default=DEFAULT_REAL_FIELD_WINDING_TOLERANCE,
    )
    parser.add_argument("--r-satisfied", type=float, default=0.0)
    parser.add_argument("--local-difference-step", type=float, default=1.0e-6)
    parser.add_argument("--direction-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    biot_savart = load(args.biot_savart_json)
    targets = load_residue_objective_targets(args.targets_json)
    chart = PoincareChart(
        axis_r=args.axis_r,
        axis_z=args.axis_z,
        poloidal_orientation=args.poloidal_orientation,
        radial_label_scale=args.radial_label_scale,
    )
    integrator_options = FieldlineIntegratorOptions(
        rtol=args.rtol,
        atol=args.atol,
        max_step=args.max_step,
        samples_per_full_torus=args.samples_per_full_torus,
        min_bphi_over_b=args.min_bphi_over_b,
        max_rhs_evaluations=args.max_fieldline_rhs_evals,
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=args.newton_residual_tolerance,
        winding_tolerance=args.winding_tolerance,
        det_tolerance=args.det_tolerance,
        max_iterations=args.max_newton_iterations,
        max_step_norm=args.max_newton_step_norm,
    )
    phase_angles = uniform_branch_phase_angles(args.branch_phase_count)

    manifests = generate_residue_seed_files(
        biot_savart,
        targets=targets,
        chart=chart,
        validation_artifact_id=args.validation_artifact_id,
        targets_path=args.targets_out,
        seeds_path=args.seeds_out,
        integrator_options=integrator_options,
        solver_options=solver_options,
        phase_angles=phase_angles,
        min_island_width=args.min_island_width,
        max_candidates=args.max_candidates,
        taylor_probe_steps=tuple(args.taylor_probe_steps),
        residue_difference_tolerance=args.residue_difference_tolerance,
        winding_tolerance=args.real_field_winding_tolerance,
        r_satisfied=args.r_satisfied,
        local_difference_step=args.local_difference_step,
        direction_seed=args.direction_seed,
    )

    print(
        f"Wrote {len(manifests.validated)} validated residue seed(s) to "
        f"{args.seeds_out} and targets to {args.targets_out}"
    )
    for seed in manifests.validated:
        print(
            f"  validated target={seed.target.manifest_key()} branch={seed.branch} "
            f"residue={seed.direct_residue:.6g} taylor_order={seed.taylor_min_order:.4g}"
        )
    for skipped in manifests.skipped:
        print(
            f"  skipped target={skipped.target_id} branch={skipped.branch}: "
            f"{skipped.reason}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
