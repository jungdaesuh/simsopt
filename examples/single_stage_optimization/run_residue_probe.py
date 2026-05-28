from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from import_provenance import configure_local_simsopt_imports

EXAMPLE_ROOT, SIMSOPT_ROOT, SRC_ROOT = configure_local_simsopt_imports(__file__)

from banana_opt.json_compat import load_boozer_finite_i as load  # noqa: E402
from banana_opt.topology.fieldline_map import FieldlineIntegratorOptions  # noqa: E402
from banana_opt.topology.iota_profile import (  # noqa: E402
    DEFAULT_AXIS_SEARCH_HALF_WIDTH,
    DEFAULT_IOTA_PROFILE_TORUS_TURNS,
    freeze_probe_inputs,
    linear_radial_labels,
    probe_freezing_payload,
)
from banana_opt.topology.periodic_orbit import PeriodicOrbitSolverOptions  # noqa: E402
from banana_opt.topology.poincare_chart import PoincareChart  # noqa: E402
from banana_opt.topology.rational_target import RationalTarget  # noqa: E402
from banana_opt.topology.residue_diagnostics import (  # noqa: E402
    DEFAULT_BRANCH_PHASE_COUNT,
    run_residue_probe,
    uniform_branch_phase_angles,
)
from banana_opt.topology.residue_objective import (  # noqa: E402
    BiotSavartGreeneResidueObjective,
    DEFAULT_RESIDUE_OBJECTIVE_WEIGHT,
    DEFAULT_RESIDUE_OBJECTIVE_SAMPLES_PER_FULL_TORUS,
    DEFAULT_RESIDUE_SATISFIED_THRESHOLD,
    ResidueBranchSeed,
    disabled_residue_objective_payload,
    load_residue_objective_seeds,
    load_residue_objective_targets,
    residue_objective_target_manifest_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Greene residue branch diagnostics and the optional "
            "direct-BiotSavart residue objective on one or more single-stage "
            "output directories."
        )
    )
    parser.add_argument(
        "output_dirs",
        nargs="+",
        help="Output directories containing biot_savart_opt.json or biot_savart_init.json.",
    )
    parser.add_argument(
        "--targets-json",
        required=True,
        help="JSON file with a top-level 'targets' list of RationalTarget payloads.",
    )
    parser.add_argument(
        "--report-path",
        default="greene_residue_probe_report.json",
        help="Path to the JSON report to write.",
    )
    parser.add_argument(
        "--field-label",
        choices=("auto", "opt", "init"),
        default="auto",
        help="Which BiotSavart artifact to load from each output directory.",
    )
    parser.add_argument("--axis-r", type=float, required=True)
    parser.add_argument("--axis-z", type=float, default=0.0)
    parser.add_argument("--poloidal-orientation", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--radial-label-scale", type=float, default=1.0)
    parser.add_argument(
        "--freeze-targets-from-iota",
        action="store_true",
        help=(
            "Trace the realized iota(r) profile of each field and place each "
            "target's radial_label/window on its iota=p/q crossing, reporting "
            "targets whose iota is out of domain instead of probing them."
        ),
    )
    parser.add_argument(
        "--locate-axis",
        action="store_true",
        help=(
            "Re-center the chart on the magnetic-axis fixed point near --axis-r "
            "(instead of trusting --axis-r). Only used with "
            "--freeze-targets-from-iota."
        ),
    )
    parser.add_argument(
        "--nfp",
        type=int,
        default=None,
        help=(
            "Field period count for magnetic-axis location. Defaults to the "
            "common nfp of the requested targets."
        ),
    )
    parser.add_argument("--iota-profile-lower", type=float, default=0.005)
    parser.add_argument("--iota-profile-upper", type=float, default=0.08)
    parser.add_argument("--iota-profile-samples", type=int, default=24)
    parser.add_argument(
        "--iota-profile-turns",
        type=int,
        default=DEFAULT_IOTA_PROFILE_TORUS_TURNS,
    )
    parser.add_argument(
        "--axis-search-half-width",
        type=float,
        default=DEFAULT_AXIS_SEARCH_HALF_WIDTH,
    )
    parser.add_argument("--rtol", type=float, default=1.0e-9)
    parser.add_argument("--atol", type=float, default=1.0e-11)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument(
        "--samples-per-full-torus",
        type=int,
        default=DEFAULT_RESIDUE_OBJECTIVE_SAMPLES_PER_FULL_TORUS,
    )
    parser.add_argument("--min-bphi-over-b", type=float, default=1.0e-8)
    parser.add_argument("--newton-residual-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--winding-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--det-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--max-newton-iterations", type=int, default=12)
    parser.add_argument("--max-newton-step-norm", type=float, default=0.05)
    parser.add_argument(
        "--branch-phase-count",
        type=int,
        default=DEFAULT_BRANCH_PHASE_COUNT,
        help=(
            "Number of deterministic poloidal phases to scan when discovering "
            "target-winding Greene branches."
        ),
    )
    parser.add_argument(
        "--residue-objective-weight",
        type=float,
        default=DEFAULT_RESIDUE_OBJECTIVE_WEIGHT,
        help=(
            "Opt-in Greene residue objective weight. The default 0 disables "
            "objective/gradient evaluation."
        ),
    )
    parser.add_argument(
        "--residue-objective-scale",
        type=float,
        default=1.0,
        help="Residue scale R_scale for 0.5 * (R_G / R_scale)^2.",
    )
    parser.add_argument(
        "--residue-objective-seeds-json",
        help=(
            "JSON file with target_manifest_id, validation_status='passed', "
            "validation_artifact_id, branch_seeds, and "
            "optimizer_taylor_validations. Required when "
            "--residue-objective-weight is nonzero."
        ),
    )
    parser.add_argument(
        "--residue-objective-r-satisfied",
        type=float,
        default=DEFAULT_RESIDUE_SATISFIED_THRESHOLD,
        help="Near-success |R_G| threshold whose gradient is frozen.",
    )
    parser.add_argument(
        "--residue-objective-local-difference-step",
        type=float,
        default=1.0e-6,
        help="Local finite-difference step used inside the residue VJP kernel.",
    )
    return parser.parse_args()


def resolve_biot_savart_artifact(
    output_dir: str | Path,
    *,
    field_label: str,
) -> tuple[Path, str]:
    output_path = Path(output_dir).resolve()
    candidates = (
        ("opt", output_path / "biot_savart_opt.json"),
        ("init", output_path / "biot_savart_init.json"),
    )
    if field_label == "auto":
        for label, path in candidates:
            if path.exists():
                return path, label
    else:
        for label, path in candidates:
            if label == field_label:
                if path.exists():
                    return path, label
                raise FileNotFoundError(f"Missing BiotSavart artifact: {path}")
    raise FileNotFoundError(
        f"Could not find biot_savart_opt.json or biot_savart_init.json in {output_path}"
    )


@dataclass(frozen=True)
class IotaFreezingConfig:
    """Settings for placing targets on the realized iota profile (doc §3.2)."""

    locate_axis: bool
    radial_labels: tuple[float, ...]
    toroidal_turns: int
    axis_search_half_width: float
    nfp: int


def evaluate_output_dir(
    output_dir: str | Path,
    *,
    targets: Sequence[RationalTarget],
    chart: PoincareChart,
    field_label: str,
    integrator_options: FieldlineIntegratorOptions,
    solver_options: PeriodicOrbitSolverOptions,
    residue_objective_weight: float,
    residue_objective_scale: float,
    residue_objective_validation_id: str,
    residue_objective_branch_seeds: Sequence[ResidueBranchSeed],
    residue_objective_r_satisfied: float,
    residue_objective_local_difference_step: float,
    branch_phase_angles: Sequence[float],
    freezing_config: IotaFreezingConfig | None,
) -> dict[str, object]:
    artifact_path, resolved_field_label = resolve_biot_savart_artifact(
        output_dir,
        field_label=field_label,
    )
    biot_savart = load(artifact_path)
    if freezing_config is None:
        probe_chart = chart
        probe_targets: Sequence[RationalTarget] = targets
        target_freezing_payload = None
    else:
        freezing = freeze_probe_inputs(
            biot_savart,
            requested_targets=targets,
            axis_r_guess=chart.axis_r,
            axis_z_guess=chart.axis_z,
            nfp=freezing_config.nfp,
            locate_axis=freezing_config.locate_axis,
            radial_labels=freezing_config.radial_labels,
            toroidal_turns=freezing_config.toroidal_turns,
            poloidal_orientation=chart.poloidal_orientation,
            radial_label_scale=chart.radial_label_scale,
            axis_search_half_width=freezing_config.axis_search_half_width,
            integrator_options=integrator_options,
        )
        probe_chart = freezing.chart
        probe_targets = freezing.in_domain_targets
        target_freezing_payload = probe_freezing_payload(freezing)
    probe = run_residue_probe(
        biot_savart,
        targets=probe_targets,
        chart=probe_chart,
        integrator_options=integrator_options,
        solver_options=solver_options,
        phase_angles=branch_phase_angles,
    )
    target_manifest_id = residue_objective_target_manifest_id(targets)
    objective_payload = disabled_residue_objective_payload(
        target_manifest_id=target_manifest_id,
        objective_weight=residue_objective_weight,
        residue_scale=residue_objective_scale,
    )
    if float(residue_objective_weight) > 0.0:
        objective = BiotSavartGreeneResidueObjective(
            biot_savart,
            targets=targets,
            chart=chart,
            branch_seeds=residue_objective_branch_seeds,
            objective_weight=residue_objective_weight,
            residue_scale=residue_objective_scale,
            target_manifest_id=target_manifest_id,
            validation_id=residue_objective_validation_id,
            integrator_options=integrator_options,
            solver_options=solver_options,
            r_satisfied=residue_objective_r_satisfied,
            local_difference_step=residue_objective_local_difference_step,
        )
        objective_gradient = np.asarray(objective.dJ(), dtype=float)
        objective_payload = {
            **objective.to_json_dict(),
            "enabled": True,
            "gradient_norm": float(np.linalg.norm(objective_gradient)),
        }
    return {
        "output_dir": str(Path(output_dir).resolve()),
        "field_label": resolved_field_label,
        "biot_savart_path": str(artifact_path),
        "target_freezing": target_freezing_payload,
        "probe": probe,
        "residue_objective": objective_payload,
    }


def _resolve_profile_nfp(
    nfp_argument: int | None,
    targets: Sequence[RationalTarget],
) -> int:
    if nfp_argument is not None:
        nfp = int(nfp_argument)
        if nfp <= 0:
            raise ValueError("--nfp must be positive")
        return nfp
    target_nfps = {int(target.nfp) for target in targets}
    if len(target_nfps) != 1:
        raise ValueError(
            "magnetic-axis location needs a single nfp; requested targets "
            f"declare {sorted(target_nfps)}. Pass --nfp explicitly."
        )
    return target_nfps.pop()


def main() -> None:
    args = parse_args()
    targets = load_residue_objective_targets(args.targets_json)
    target_manifest_id = residue_objective_target_manifest_id(targets)
    residue_objective_weight = float(args.residue_objective_weight)
    if residue_objective_weight < 0.0:
        raise ValueError("--residue-objective-weight must be >= 0")
    residue_objective_validation_id = ""
    residue_objective_branch_seeds: tuple[ResidueBranchSeed, ...] = ()
    if residue_objective_weight > 0.0:
        if args.residue_objective_seeds_json is None:
            raise ValueError(
                "--residue-objective-seeds-json is required when "
                "--residue-objective-weight is nonzero"
            )
        residue_objective_validation_id, residue_objective_branch_seeds = (
            load_residue_objective_seeds(
                args.residue_objective_seeds_json,
                target_manifest_id=target_manifest_id,
            )
        )
    freeze_targets_from_iota = bool(args.freeze_targets_from_iota)
    if args.locate_axis and not freeze_targets_from_iota:
        raise ValueError("--locate-axis requires --freeze-targets-from-iota")
    if freeze_targets_from_iota and residue_objective_weight > 0.0:
        raise ValueError(
            "--freeze-targets-from-iota relocates target labels/windows, which "
            "invalidates residue-objective seeds keyed to the requested targets; "
            "run target freezing and the objective in separate steps"
        )
    freezing_config = None
    if freeze_targets_from_iota:
        freezing_config = IotaFreezingConfig(
            locate_axis=bool(args.locate_axis),
            radial_labels=linear_radial_labels(
                lower=args.iota_profile_lower,
                upper=args.iota_profile_upper,
                count=args.iota_profile_samples,
            ),
            toroidal_turns=int(args.iota_profile_turns),
            axis_search_half_width=float(args.axis_search_half_width),
            nfp=_resolve_profile_nfp(args.nfp, targets),
        )
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
    )
    solver_options = PeriodicOrbitSolverOptions(
        residual_tolerance=args.newton_residual_tolerance,
        winding_tolerance=args.winding_tolerance,
        det_tolerance=args.det_tolerance,
        max_iterations=args.max_newton_iterations,
        max_step_norm=args.max_newton_step_norm,
    )
    branch_phase_angles = uniform_branch_phase_angles(args.branch_phase_count)
    cases = [
        evaluate_output_dir(
            output_dir,
            targets=targets,
            chart=chart,
            field_label=args.field_label,
            integrator_options=integrator_options,
            solver_options=solver_options,
            residue_objective_weight=residue_objective_weight,
            residue_objective_scale=args.residue_objective_scale,
            residue_objective_validation_id=residue_objective_validation_id,
            residue_objective_branch_seeds=residue_objective_branch_seeds,
            residue_objective_r_satisfied=args.residue_objective_r_satisfied,
            residue_objective_local_difference_step=(
                args.residue_objective_local_difference_step
            ),
            branch_phase_angles=branch_phase_angles,
            freezing_config=freezing_config,
        )
        for output_dir in args.output_dirs
    ]
    report = {
        "schema_version": "greene_residue_probe_runner_v1",
        "targets_json": str(Path(args.targets_json).resolve()),
        "cases": cases,
    }
    report_path = Path(args.report_path).resolve()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote Greene residue probe report to {report_path}")


if __name__ == "__main__":
    main()
