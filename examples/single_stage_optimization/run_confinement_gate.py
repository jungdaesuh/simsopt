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
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from banana_opt.confinement_gate import (
    ConfinementGateConfig,
    evaluate_confinement_gate,
    metrics_from_topology_verdict,
)
from banana_opt.topology_fidelity_ladder import DEFAULT_TOPOLOGY_TIER_SPECS

STRICT_TIER = "strict"


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
    return parser.parse_args()


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


def certify(run_dir: str, args: argparse.Namespace) -> dict:
    # Import the strict-tier runner lazily: it transitively pulls in the heavy single-stage
    # module (simsopt/jax), so deferring it keeps this driver module importable cheaply --
    # e.g. to unit-test resolve_beta / parse_args without paying for the full toolchain.
    from run_topology_fidelity_ladder import evaluate_case

    run_path = Path(run_dir).resolve()
    case = evaluate_case(run_path)
    strict_record = case[STRICT_TIER]
    strict_verdict = strict_record["topology_verdict"]
    nfieldlines = DEFAULT_TOPOLOGY_TIER_SPECS[STRICT_TIER].nfieldlines

    metrics = metrics_from_topology_verdict(
        strict_verdict,
        nfieldlines,
        qa_nonqs_ratio=_optional_float(args.qa_nonqs),
        magnetic_well=_optional_float(args.magnetic_well),
        iota_shear=_optional_float(args.iota_shear),
        beta=resolve_beta(run_path, args.beta),
        mercier_dmerc_min=_optional_float(args.mercier_dmerc),
    )
    config = ConfinementGateConfig(
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
    verdict = evaluate_confinement_gate(metrics, config)
    payload = {
        "run_dir": str(run_path),
        "field_label": case.get("field_label"),
        "strict_topology_passed": bool(strict_verdict.get("passed")),
        "confinement_verdict": verdict.to_dict(),
    }
    out_path = run_path / "confinement_verdict.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


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
    print(f"  written -> {Path(args.run_dir).resolve() / 'confinement_verdict.json'}")
    raise SystemExit(0 if verdict["accepted"] else 1)


if __name__ == "__main__":
    main()
