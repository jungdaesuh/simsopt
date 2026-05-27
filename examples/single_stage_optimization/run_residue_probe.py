from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate value-only Greene residue branch diagnostics on one or "
            "more single-stage output directories."
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
) -> dict[str, object]:
    artifact_path, resolved_field_label = resolve_biot_savart_artifact(
        output_dir,
        field_label=field_label,
    )
    return {
        "output_dir": str(Path(output_dir).resolve()),
        "field_label": resolved_field_label,
        "biot_savart_path": str(artifact_path),
        "probe": run_residue_probe(
            load(artifact_path),
            targets=targets,
            chart=chart,
            integrator_options=integrator_options,
            solver_options=solver_options,
        ),
    }


def main() -> None:
    args = parse_args()
    targets = load_targets(args.targets_json)
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
