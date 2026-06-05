#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = SCRIPT_DIR.parent
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from simsopt.mhd import Vmec  # noqa: E402

from banana_opt.vmec_curtor_sign import (  # noqa: E402
    DEFAULT_CURTOR_CONVERGENCE_NS,
    DEFAULT_CURTOR_SIGN_LADDER_A,
    build_curtor_sign_artifact,
    ftol_schedule_for_ns,
    niter_schedule_for_ns,
    radial_schedule_for_final_ns,
    write_curtor_sign_json,
    write_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed-boundary VMEC curtor sign probe and write curtor_sign.json."
        )
    )
    parser.add_argument("--input", required=True, help="VMEC input file.")
    parser.add_argument("--output-dir", required=True, help="Artifact output directory.")
    parser.add_argument(
        "--curtor-points-A",
        default=",".join(str(value) for value in DEFAULT_CURTOR_SIGN_LADDER_A),
        help="Comma-separated curtor sign-probe ladder in amperes.",
    )
    parser.add_argument(
        "--current-magnitude-A",
        type=float,
        default=800.0,
        help="Magnitude used for the NS convergence sign branch.",
    )
    parser.add_argument(
        "--sign-ns",
        type=int,
        default=51,
        help="Final VMEC ns for the sign-probe ladder.",
    )
    parser.add_argument(
        "--convergence-ns",
        default=",".join(str(value) for value in DEFAULT_CURTOR_CONVERGENCE_NS),
        help="Comma-separated final VMEC ns values for edge-lift convergence.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable VMEC logging.")
    return parser.parse_args(argv)


def run_curtor(
    *,
    input_path: Path,
    curtor_A: float,
    final_ns: int,
    verbose: bool,
) -> dict[str, object]:
    ns_schedule = radial_schedule_for_final_ns(final_ns)
    niter_schedule = niter_schedule_for_ns(ns_schedule)
    ftol_schedule = ftol_schedule_for_ns(ns_schedule)
    started_at = time.time()

    vmec = Vmec(str(input_path), verbose=bool(verbose))
    vmec.indata.ns_array[:] = 0
    vmec.indata.niter_array[:] = 0
    vmec.indata.ftol_array[:] = 0.0
    vmec.indata.ns_array[: len(ns_schedule)] = np.asarray(ns_schedule, dtype=int)
    vmec.indata.niter_array[: len(niter_schedule)] = np.asarray(
        niter_schedule,
        dtype=int,
    )
    vmec.indata.ftol_array[: len(ftol_schedule)] = np.asarray(
        ftol_schedule,
        dtype=float,
    )
    vmec.indata.ncurr = 1
    vmec.indata.ac[:] = 0.0
    vmec.indata.ac[0] = 1.0
    vmec.indata.curtor = float(curtor_A)
    vmec.need_to_run_code = True
    vmec.run()

    iotaf = np.asarray(vmec.wout.iotaf, dtype=float)
    return {
        "curtor_A": float(curtor_A),
        "vmec_final_ns": int(final_ns),
        "vmec_ns_array": [int(ns) for ns in ns_schedule],
        "vmec_niter_array": [int(niter) for niter in niter_schedule],
        "vmec_ftol_array": [float(ftol) for ftol in ftol_schedule],
        "ncurr": int(vmec.indata.ncurr),
        "ac0": float(vmec.indata.ac[0]),
        "need_to_run_code": True,
        "iota_axis": float(iotaf[0]),
        "iota_edge": float(iotaf[-1]),
        "iota_mean": float(np.mean(iotaf)),
        "ctor_wout": float(vmec.wout.ctor),
        "volume_p": float(vmec.wout.volume_p),
        "aminor": float(vmec.wout.Aminor_p),
        "rmajor": float(vmec.wout.Rmajor_p),
        "converged": int(vmec.wout.ier_flag) == 0,
        "elapsed_s": round(time.time() - started_at, 1),
    }


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in raw.split(",") if value.strip())


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sign_points = _parse_float_csv(args.curtor_points_A)
    sign_records = [
        run_curtor(
            input_path=input_path,
            curtor_A=curtor_A,
            final_ns=int(args.sign_ns),
            verbose=bool(args.verbose),
        )
        for curtor_A in sign_points
    ]
    sign_jsonl = output_dir / "results_fix3_vmec.jsonl"
    write_jsonl(sign_jsonl, sign_records)

    provisional_artifact = build_curtor_sign_artifact(
        sign_records=sign_records,
        convergence_records=(),
        source_input=str(input_path),
        current_magnitude_A=float(args.current_magnitude_A),
    )
    raising_sign = int(
        provisional_artifact["sign_probe"]["iota_raising_curtor_sign"]
    )
    convergence_points = tuple(
        sorted(
            {
                0.0,
                raising_sign * abs(float(args.current_magnitude_A)),
            }
        )
    )
    convergence_ns_values = _parse_int_csv(args.convergence_ns)
    convergence_records = [
        run_curtor(
            input_path=input_path,
            curtor_A=curtor_A,
            final_ns=final_ns,
            verbose=bool(args.verbose),
        )
        for final_ns in convergence_ns_values
        for curtor_A in convergence_points
    ]
    convergence_jsonl = output_dir / "results_fix3_vmec_convergence.jsonl"
    write_jsonl(convergence_jsonl, convergence_records)

    artifact = build_curtor_sign_artifact(
        sign_records=sign_records,
        convergence_records=convergence_records,
        source_input=str(input_path),
        current_magnitude_A=float(args.current_magnitude_A),
        convergence_ns_values=convergence_ns_values,
    )
    artifact["results_jsonl"] = str(sign_jsonl)
    artifact["convergence_jsonl"] = str(convergence_jsonl)
    write_curtor_sign_json(output_dir / "curtor_sign.json", artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
