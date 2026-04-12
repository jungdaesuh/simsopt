from __future__ import annotations

import argparse
import json
import sys

from import_smoke_cases import (
    block_private_optimizer_imports,
    strip_simsopt_editable_finders,
)


def main(
    *,
    optimizer_backend: str,
    block_private: bool,
    boozer_least_squares_algorithm: str | None,
) -> None:
    strip_simsopt_editable_finders()

    if block_private:
        block_private_optimizer_imports()

    from import_smoke_cases import _REPO_ROOT

    src_dir = str(_REPO_ROOT / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from benchmarks.single_stage_smoke_fixture import (
        build_real_single_stage_init_fixture,
    )
    from simsopt.geo import optimizer_jax

    payload: dict[str, object] = {}
    try:
        fixture = build_real_single_stage_init_fixture(
            backend="jax",
            optimizer_backend=optimizer_backend,
            boozer_least_squares_algorithm=boozer_least_squares_algorithm,
        )
    except Exception as exc:
        payload["exc_type"] = type(exc).__name__
        payload["exc_message"] = str(exc)
    else:
        booz = fixture["boozer_surface"]
        result = booz.res
        payload.update(
            {
                "success": bool(result["success"]),
                "optimizer_method": result.get("optimizer_method"),
                "iota": float(result["iota"]),
                "G": float(result["G"]),
                "fun": float(result["fun"]),
                "iter": int(result["iter"]),
                "surface_dofs": [float(x) for x in booz.surface.get_dofs()],
                "private_pkg_is_none": optimizer_jax._private_pkg is None,
                "private_in_sys_modules": "simsopt.geo.optimizer_jax_private"
                in sys.modules,
            }
        )

    print(json.dumps(payload))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-backend", required=True)
    parser.add_argument("--block-private", action="store_true", default=False)
    parser.add_argument("--boozer-least-squares-algorithm", default=None)
    args = parser.parse_args()

    main(
        optimizer_backend=args.optimizer_backend,
        block_private=args.block_private,
        boozer_least_squares_algorithm=args.boozer_least_squares_algorithm,
    )
