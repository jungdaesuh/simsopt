from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from import_provenance import configure_local_simsopt_imports

EXAMPLE_ROOT, SIMSOPT_ROOT, SRC_ROOT = configure_local_simsopt_imports(__file__)

from banana_opt.json_compat import load_boozer_finite_i as load  # noqa: E402
from banana_opt.topology.fieldline_map import FieldlineIntegratorOptions  # noqa: E402
from banana_opt.topology.periodic_orbit import PeriodicOrbitSolverOptions  # noqa: E402
from banana_opt.topology.poincare_chart import PoincareChart  # noqa: E402
from banana_opt.topology.rational_target import RationalTarget  # noqa: E402
from banana_opt.topology.residue_diagnostics import run_residue_probe  # noqa: E402
from banana_opt.topology.residue_objective import (  # noqa: E402
    BiotSavartGreeneResidueObjective,
    DEFAULT_RESIDUE_OBJECTIVE_WEIGHT,
    DEFAULT_RESIDUE_SATISFIED_THRESHOLD,
    ResidueBranchSeed,
    disabled_residue_objective_payload,
    residue_branch_seed_from_payload,
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
    parser.add_argument("--rtol", type=float, default=1.0e-9)
    parser.add_argument("--atol", type=float, default=1.0e-11)
    parser.add_argument("--max-step", type=float, default=0.05)
    parser.add_argument("--samples-per-full-torus", type=int, default=128)
    parser.add_argument("--min-bphi-over-b", type=float, default=1.0e-8)
    parser.add_argument("--newton-residual-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--winding-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--max-newton-iterations", type=int, default=12)
    parser.add_argument("--max-newton-step-norm", type=float, default=0.05)
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
            "JSON file with target_manifest_id, validation_artifact_id, and "
            "branch_seeds. Required when --residue-objective-weight is nonzero."
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


def _target_from_payload(payload: Mapping[str, object]) -> RationalTarget:
    return RationalTarget(
        p=int(payload["p"]),
        q=int(payload["q"]),
        weight=float(payload.get("weight", 1.0)),
        radial_label=payload.get("radial_label"),
        radial_window=payload.get("radial_window"),
        branches=payload.get("branches", ("O", "X")),
        phi0=float(payload.get("phi0", 0.0)),
        nfp=int(payload.get("nfp", 1)),
        fourier_m=payload.get("fourier_m"),
        fourier_n=payload.get("fourier_n"),
    )


def load_targets(path: str | Path) -> tuple[RationalTarget, ...]:
    with Path(path).resolve().open(encoding="utf-8") as infile:
        payload = json.load(infile)
    if not isinstance(payload, Mapping):
        raise ValueError("Greene residue targets JSON must be an object")
    targets_payload = payload["targets"]
    if not isinstance(targets_payload, Sequence):
        raise ValueError("Greene residue targets JSON 'targets' must be a sequence")
    return tuple(_target_from_payload(dict(target)) for target in targets_payload)


def load_residue_objective_seeds(
    path: str | Path,
    *,
    target_manifest_id: str,
) -> tuple[str, tuple[ResidueBranchSeed, ...]]:
    with Path(path).resolve().open(encoding="utf-8") as infile:
        payload = json.load(infile)
    if not isinstance(payload, Mapping):
        raise ValueError("Greene residue objective seeds JSON must be an object")
    seed_target_manifest_id = str(payload["target_manifest_id"])
    if seed_target_manifest_id != target_manifest_id:
        raise ValueError(
            "Greene residue objective seeds JSON target_manifest_id does not "
            "match the requested targets"
        )
    validation_artifact_id = str(payload["validation_artifact_id"])
    if validation_artifact_id == "":
        raise ValueError(
            "Greene residue objective seeds JSON validation_artifact_id must be nonempty"
        )
    seed_payloads = payload["branch_seeds"]
    if not isinstance(seed_payloads, Sequence):
        raise ValueError(
            "Greene residue objective seeds JSON branch_seeds must be a sequence"
        )
    return (
        validation_artifact_id,
        tuple(
            residue_branch_seed_from_payload(
                dict(seed),
                validation_id=validation_artifact_id,
            )
            for seed in seed_payloads
        ),
    )


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
) -> dict[str, object]:
    artifact_path, resolved_field_label = resolve_biot_savart_artifact(
        output_dir,
        field_label=field_label,
    )
    biot_savart = load(artifact_path)
    probe = run_residue_probe(
        biot_savart,
        targets=targets,
        chart=chart,
        integrator_options=integrator_options,
        solver_options=solver_options,
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
        "probe": probe,
        "residue_objective": objective_payload,
    }


def main() -> None:
    args = parse_args()
    targets = load_targets(args.targets_json)
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
        max_iterations=args.max_newton_iterations,
        max_step_norm=args.max_newton_step_norm,
    )
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
