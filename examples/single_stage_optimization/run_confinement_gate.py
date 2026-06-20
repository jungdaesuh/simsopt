"""Post-hoc confinement certification for a saved banana-coil candidate (Phase 5 / T5.1).

Composes the existing STRICT topology tier (50 field lines, tmax 7000, + WBA island
verdict, reused verbatim from ``run_topology_fidelity_ladder.evaluate_case``) with the
physics criteria (QA / magnetic well / shear / Mercier) into ONE decisive
``ConfinementVerdict`` and writes ``confinement_verdict.json`` next to the candidate.

This is the in-loop-too-expensive strict gate run once post-opt, matching the established
post-hoc certification pattern (``run_topology_fidelity_ladder.py``). The decisive criterion
is the strict topology + island verdict computed here; the physics values (QA non-QS ratio,
magnetic-well proxy, iota shear, beta, min DMerc) are supplied as inputs because they are
computed by the producing solver / judge and are advisory-by-default in the gate. Vacuum
(beta=0, the campaign default) correctly reports Mercier not-applicable.

    python run_confinement_gate.py --run-dir <dir>
    python run_confinement_gate.py --run-dir <dir> --qa-nonqs 5e-3 --iota-shear 0.05 \
        --require-qa --min-survival 0.9

Converse-KAM (OPT-IN, DEFAULT-OFF). With ``--converse-kam-gate-interval 0`` (the default)
behaviour is byte-identical to before. Setting it > 0 turns on the ADVISORY converse-KAM
cone-crossing non-existence diagnostic (arXiv:2501.06796) on the same persisted Biot-Savart
field; it is recorded under ``confinement_verdict.converse_kam`` but is NOT decisive (it
never flips accept/reject) until the direction-field construction is donor-validated:

    python run_confinement_gate.py --run-dir <dir> --converse-kam-gate-interval 1 \
        --converse-kam-seeds 24 --converse-kam-tf 600 --converse-kam-timeout 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from banana_opt.confinement_gate import (
    ConfinementGateConfig,
    certify_confinement,
)
from banana_opt.topology.converse_kam import ConverseKamConfig


def _optional_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decisive post-hoc confinement gate: strict topology + island verdict "
            "composed with QA / well / shear / Mercier criteria."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory with biot_savart_opt.json + surf_opt.json (or _init).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Equilibrium beta. Default: read BOOZER_I from results.json (0 => vacuum).",
    )
    parser.add_argument("--qa-nonqs", default=None, help="NonQuasiSymmetricRatio value.")
    parser.add_argument("--magnetic-well", default=None, help="Boozer well proxy W.")
    parser.add_argument("--iota-shear", default=None, help="|iota_edge - iota_axis|.")
    parser.add_argument("--mercier-dmerc", default=None, help="min DMerc (finite-beta).")
    parser.add_argument(
        "--min-survival",
        type=float,
        default=0.90,
        help=(
            "Survival accept bar (default 0.90 == 45/50). Intentionally more permissive "
            "than the strict tier's own survival_threshold of 1.0 (50/50), since validated "
            "champions sit at ~47-49/50; the island + classifiability gates still match the "
            "strict tier exactly."
        ),
    )
    parser.add_argument("--min-classifiable", type=float, default=0.90)
    # Islands + classifiability are part of the strict-topology standard and are REQUIRED by
    # default (matching the reused strict tier); relax them only deliberately.
    parser.add_argument("--advisory-islands", action="store_true")
    parser.add_argument("--advisory-classifiability", action="store_true")
    parser.add_argument("--require-qa", action="store_true")
    parser.add_argument("--require-magnetic-well", action="store_true")
    parser.add_argument("--require-shear", action="store_true")
    parser.add_argument("--qa-nonqs-max", type=float, default=1.0e-2)
    parser.add_argument("--iota-shear-min", type=float, default=0.0)
    # --- Converse-KAM (OPT-IN, DEFAULT-OFF) ----------------------------------------------
    # Interval semantics: 0 == OFF (default; byte-identical legacy behaviour). Any value > 0
    # enables the diagnostic for this single post-hoc certification. The "interval" name is
    # kept for parity with the in-loop closed-loop convention (run the converse-KAM gate
    # every Nth candidate); for the one-shot CLI, > 0 simply means "run it".
    parser.add_argument(
        "--converse-kam-gate-interval",
        type=int,
        default=0,
        help=(
            "0 (default) = OFF. > 0 enables the ADVISORY converse-KAM cone-crossing "
            "non-existence diagnostic (arXiv:2501.06796). Recorded under "
            "confinement_verdict.converse_kam; NOT promotion-decisive until donor-validated."
        ),
    )
    parser.add_argument(
        "--converse-kam-seeds",
        type=int,
        default=24,
        help="Radial seeds across the phi=0 midplane for the converse-KAM sweep.",
    )
    parser.add_argument(
        "--converse-kam-tf",
        type=float,
        default=600.0,
        help="Converse-KAM integration window t_f (field-line time; [-t_f/2, +t_f/2]).",
    )
    parser.add_argument(
        "--converse-kam-timeout",
        type=float,
        default=None,
        help=(
            "Converse-KAM timeout (>= t_f). Reaching it without a cone collapse marks the "
            "point UNDECIDED (fail-closed, never 'surface exists'). Default: equals t_f."
        ),
    )
    parser.add_argument(
        "--converse-kam-phi-planes",
        type=int,
        default=8,
        help="Toroidal planes at which to locate the magnetic axis a(phi) for xi.",
    )
    return parser.parse_args()


def converse_kam_config_from_args(args: argparse.Namespace) -> ConverseKamConfig | None:
    """Build a ConverseKamConfig only when the diagnostic is opted in (interval > 0).

    Returns None when ``--converse-kam-gate-interval`` is 0 (the default), which makes
    ``certify_confinement`` skip the diagnostic entirely (legacy path).
    """
    if int(args.converse_kam_gate_interval) <= 0:
        return None
    timeout = (
        float(args.converse_kam_tf)
        if args.converse_kam_timeout is None
        else float(args.converse_kam_timeout)
    )
    return ConverseKamConfig(
        t_f=float(args.converse_kam_tf),
        timeout_t_f=timeout,
        n_seeds=int(args.converse_kam_seeds),
        n_phi_planes=int(args.converse_kam_phi_planes),
    )


def resolve_beta(run_dir: Path, override: float | None) -> float:
    """Beta for the Mercier branch: explicit ``override``, else 0 (vacuum).

    This driver does NOT reconstruct beta from a pressure profile, so it returns 0 (the
    campaign vacuum contract: Mercier reported N/A) unless ``--beta`` is given. When the
    saved ``results.json`` shows a nonzero ``BOOZER_I`` (finite-current lineage), beta is
    still not inferable here, so it emits a loud NOTE that ``--beta`` is required to exercise
    the finite-beta Mercier branch rather than silently certifying it Mercier-N/A.
    """
    if override is not None:
        return float(override)
    results_path = run_dir / "results.json"
    if results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        boozer_I = float(payload.get("BOOZER_I", 0.0) or 0.0)
        if boozer_I != 0.0:
            print(
                f"  NOTE: results.json BOOZER_I={boozer_I:g} is nonzero (finite current), "
                "but this driver does not reconstruct beta from a pressure profile; "
                "defaulting beta=0 (Mercier N/A). Pass --beta to gate the finite-beta branch."
            )
    return 0.0


def config_from_cli_args(args: argparse.Namespace) -> ConfinementGateConfig:
    """Build the externally-owned gate config from parsed CLI flags (SSOT for the CLI)."""
    return ConfinementGateConfig(
        min_survival_fraction=args.min_survival,
        require_islands=not args.advisory_islands,
        min_classifiable_fraction=args.min_classifiable,
        require_classifiability=not args.advisory_classifiability,
        qa_nonqs_max=args.qa_nonqs_max,
        require_qa=args.require_qa,
        require_magnetic_well=args.require_magnetic_well,
        iota_shear_min=args.iota_shear_min,
        require_shear=args.require_shear,
    )


def certify(run_dir: str, args: argparse.Namespace) -> dict:
    run_path = Path(run_dir).resolve()
    return certify_confinement(
        run_path,
        config=config_from_cli_args(args),
        beta=resolve_beta(run_path, args.beta),
        qa_nonqs_ratio=_optional_float(args.qa_nonqs),
        magnetic_well=_optional_float(args.magnetic_well),
        iota_shear=_optional_float(args.iota_shear),
        mercier_dmerc_min=_optional_float(args.mercier_dmerc),
        converse_kam_config=converse_kam_config_from_args(args),
    )


def main() -> None:
    args = parse_args()
    payload = certify(args.run_dir, args)
    verdict = payload["confinement_verdict"]
    print(f"confinement gate: {verdict['decisive_reason']}")
    # The gate's survival bar (default 0.90) is deliberately looser than the strict tier's
    # own survival_threshold (1.0); surface any case where the gate accepts a candidate the
    # underlying strict tier rejected so the divergence is never silent.
    if verdict["accepted"] and not payload["strict_topology_passed"]:
        print(
            "  WARNING: confinement gate ACCEPTED but the strict topology tier's own "
            "verdict was FAIL (survival below the tier's 1.0 threshold, or a tier gate the "
            "config relaxed). Review --min-survival / --advisory-* before promotion."
        )
    converse_kam = verdict.get("converse_kam")
    if converse_kam is not None:
        # ADVISORY ONLY: this fraction does NOT affect the accept/reject above. It is a
        # LOWER bound (undecided seeds are in the denominator, never counted as existent).
        print(
            "  converse-KAM (ADVISORY, NOT decisive until donor-validated): certified "
            f"non-existence {converse_kam['certified_nonexistence_fraction']:.3f} "
            f"({converse_kam['n_certified']}/{converse_kam['n_total']} seeds; "
            f"{converse_kam['n_undecided']} undecided; "
            f"axis={converse_kam['axis_source']}, "
            f"axis_residual_max={converse_kam['axis_residual_max']:g})"
        )
    print(f"  written -> {Path(args.run_dir).resolve() / 'confinement_verdict.json'}")
    raise SystemExit(0 if verdict["accepted"] else 1)


if __name__ == "__main__":
    main()
