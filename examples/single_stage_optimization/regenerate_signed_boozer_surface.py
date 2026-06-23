#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, Volume

from banana_opt.boozer_finite_current import derive_signed_G_from_field
from banana_opt.json_compat import load_boozer_finite_i, save_boozer_finite_i


def _load_stage_state(path: Path) -> Mapping[str, object]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"stage state must be a JSON object: {path}")
    missing_keys = [key for key in ("iota", "G") if state.get(key) is None]
    if missing_keys:
        raise ValueError(
            f"stage state is missing required keys: {', '.join(missing_keys)}"
        )
    return state


def _validated_signed_G(
    biotsavart: BiotSavart,
    state: Mapping[str, object],
    *,
    num_tf_coils: int,
) -> float:
    expected_G = derive_signed_G_from_field(
        biotsavart,
        tf_coils=biotsavart.coils[:num_tf_coils],
    )
    state_G = float(state["G"])
    if not np.isclose(state_G, expected_G, rtol=0.0, atol=1.0e-12):
        raise ValueError(
            f"state G does not match signed TF-current G: {state_G} != {expected_G}"
        )
    return expected_G


def regenerate_signed_boozer_surface(
    *,
    biot_savart_path: Path,
    surface_path: Path,
    template_boozer_surface_path: Path,
    state_path: Path,
    output_paths: tuple[Path, ...],
    num_tf_coils: int,
) -> dict[str, object]:
    biotsavart = load_boozer_finite_i(str(biot_savart_path))
    if not isinstance(biotsavart, BiotSavart):
        raise TypeError(f"expected BiotSavart at {biot_savart_path}")

    surface = load_boozer_finite_i(str(surface_path))
    template = load_boozer_finite_i(str(template_boozer_surface_path))
    template_I = float(getattr(template, "I", 0.0))
    if template_I != 0.0:
        raise ValueError(
            "regeneration writes standard vacuum BoozerSurface JSON and requires I=0"
        )

    state = _load_stage_state(state_path)
    signed_G = _validated_signed_G(biotsavart, state, num_tf_coils=num_tf_coils)
    iota = float(state["iota"])

    volume = Volume(surface)
    regenerated = BoozerSurface(
        biotsavart,
        surface,
        volume,
        float(template.targetlabel),
        constraint_weight=template.constraint_weight,
        options=dict(template.options),
    )
    result = regenerated.run_code(iota, G=signed_G)
    if not bool(result.get("success")):
        raise RuntimeError(f"Boozer solve failed while regenerating: {result}")

    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_boozer_finite_i(regenerated, output_path)

    return {
        "iota": float(regenerated.res["iota"]),
        "G": float(regenerated.res["G"]),
        "targetlabel": float(regenerated.targetlabel),
        "constraint_weight": float(regenerated.constraint_weight),
        "output_paths": [str(path) for path in output_paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate BoozerSurface JSON from a signed BiotSavart field."
    )
    parser.add_argument("--biot-savart", required=True, type=Path)
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument("--template-boozer-surface", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, action="append")
    parser.add_argument("--num-tf-coils", type=int, default=20)
    args = parser.parse_args()

    summary = regenerate_signed_boozer_surface(
        biot_savart_path=args.biot_savart,
        surface_path=args.surface,
        template_boozer_surface_path=args.template_boozer_surface,
        state_path=args.state,
        output_paths=tuple(args.output),
        num_tf_coils=args.num_tf_coils,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
