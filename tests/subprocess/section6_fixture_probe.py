"""Subprocess probe for Section 6 public/private optimizer routing."""

from __future__ import annotations

import argparse
import json

from import_smoke_cases import (
    block_private_optimizer_imports,
    prefer_local_simsopt_source_tree,
)
from simsopt_jax.geo.optimizers.single_stage_routing import (
    resolve_single_stage_jax_boozer_optimizer_backend,
)


def _exception_payload(exc: ValueError | ImportError) -> dict[str, str]:
    message = str(exc)
    exc_name = getattr(exc, "name", None)
    if exc_name:
        message = f"{message} ({exc_name})"
    return {
        "exc_type": type(exc).__name__,
        "exc_message": message,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer-backend", required=True)
    parser.add_argument("--block-private", action="store_true")
    parser.add_argument("--boozer-least-squares-algorithm", default="quasi-newton")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    prefer_local_simsopt_source_tree()
    if args.block_private:
        block_private_optimizer_imports()

    try:
        if args.optimizer_backend == "scipy":
            resolve_single_stage_jax_boozer_optimizer_backend(
                "jax",
                args.optimizer_backend,
                None,
            )
        else:
            import jax.numpy as jnp

            from simsopt_jax.geo.optimizers.optimizer import (
                resolve_target_least_squares_optimizer_method,
                target_minimize,
            )

            method = resolve_target_least_squares_optimizer_method(
                limited_memory=False,
                least_squares_algorithm=args.boozer_least_squares_algorithm,
            )
            target_minimize(
                lambda x: jnp.sum(x * x),
                jnp.ones((1,), dtype=jnp.float64),
                method=method,
                options={"maxiter": 1},
            )
    except (ValueError, ImportError) as exc:
        print(json.dumps(_exception_payload(exc), sort_keys=True))
        return 0

    print(json.dumps({"status": "ok"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
