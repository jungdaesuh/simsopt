"""CLI shim for the zero-adjacent ``jnp.where`` division lint."""

from __future__ import annotations

from simsopt._maintenance.jax_where_division_lint import main


if __name__ == "__main__":
    raise SystemExit(main())
