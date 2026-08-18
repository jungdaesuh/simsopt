"""Native-oracle endpoint evaluation for the genuine-675 fair-bar campaign.

Charter: docs/jax_gpu_genuine675_fair_bar_plan.md on pr/jax-port-squashed at
commit 7b6d69041.  Re-evaluates one 675-coordinate candidate through the
archived native evaluator (the same construction the timed native lane uses,
mirrored from benchmarks/genuine_675_dynamic_lane.py) and reports the
objective and gradient infinity norm.  The inner (iota, G) solve mode is
CLOSED_FORM_DIRECT_QR_ANCHOR_INVARIANT, so the anchor supplied here cannot
move the result; the caller passes the evaluated lane's own endpoint anchor
for maximal defensibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simsopt_jax.runtime.fixed_state_genuine_675_input_manifest import (
    validate_frozen_genuine_675_input_bundle,
)
from simsopt_jax.runtime.genuine_675_dynamic import Genuine675CompactAccepted
from simsopt_jax.runtime.single_stage_fullspace_675 import (
    GENUINE_FULLSPACE_675,
    Fullspace675Candidate,
    Fullspace675InnerState,
)
from simsopt_jax_adapters.geo.fixed_state_genuine_675_native import (
    Fullspace675FixedStateNativeEvaluator,
    Fullspace675FixedStateNativeMaterial,
    Fullspace675NativeBoozerMaterial,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-formulation-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.expected_formulation_sha256 != GENUINE_FULLSPACE_675.semantic_sha256:
        raise ValueError("The oracle received a foreign formulation identity.")
    request = json.loads(args.candidate_json.read_text())
    candidate = Fullspace675Candidate(
        coil_coordinates=tuple(float(v) for v in request["coil_coordinates"]),
        vessel_coordinates=tuple(float(v) for v in request["vessel_coordinates"]),
        surface_coordinates=tuple(float(v) for v in request["surface_coordinates"]),
    )
    anchor = Fullspace675InnerState(
        iota=float(request["anchor_iota"]),
        G=float(request["anchor_G"]),
    )
    validated = validate_frozen_genuine_675_input_bundle(
        args.input_manifest,
        source_repo=args.source_root,
    )
    manifest = validated.manifest
    boozer_material = Fullspace675NativeBoozerMaterial.from_runtime_spec(
        validated.runtime_spec,
        native_biot_savart_payload=validated.native_biot_savart_payload,
    )
    native_material = Fullspace675FixedStateNativeMaterial(
        boozer=boozer_material,
        vessel_template=validated.vessel_template,
        tf_current_magnitude_A=abs(float(validated.runtime_spec.seed.tf_current_A)),
    )
    evaluator = Fullspace675FixedStateNativeEvaluator(
        material=native_material,
        objective_policy=manifest.objective_policy,
        boozer_policy=manifest.boozer_construction_policy,
    )
    outcome = evaluator.evaluate_compact(candidate, anchor)
    accepted = isinstance(outcome, Genuine675CompactAccepted)
    if not accepted:
        raise RuntimeError(
            f"The native oracle rejected the candidate: {type(outcome).__name__}."
        )
    payload = {
        "schema": "genuine-675-fair-bar-oracle.v1",
        "formulation_semantic_sha256": GENUINE_FULLSPACE_675.semantic_sha256,
        "input_manifest_semantic_sha256": manifest.semantic_sha256,
        "objective_value": outcome.objective_value,
        "gradient_inf_norm": max(abs(g) for g in outcome.outer_gradient),
        "inner_state": outcome.inner_state.as_payload(),
        "anchor": anchor.as_payload(),
    }
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"objective_value": payload["objective_value"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
