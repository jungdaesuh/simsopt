"""Guard against weight_inv_modB default drift relative to upstream simsopt."""

import inspect

import pytest

from simsopt.geo import boozer_residual_jax, boozersurface, surfaceobjectives


def _weight_inv_modB_default(fn):
    return inspect.signature(fn).parameters["weight_inv_modB"].default


@pytest.mark.parametrize(
    "jax_symbol, upstream_symbol",
    [
        ("boozer_residual_scalar", "boozer_surface_residual"),
        ("boozer_residual_grad", "boozer_surface_residual_dB"),
        ("boozer_residual_hessian", "boozer_surface_residual_dB"),
        ("boozer_residual_vector", "boozer_surface_residual"),
        ("boozer_residual_scalar_and_grad_cpu_ordered", "boozer_surface_residual_dB"),
        ("boozer_residual_coil_vjp", "boozer_surface_dlsqgrad_dcoils_vjp"),
    ],
)
def test_weight_inv_modB_default_matches_upstream(jax_symbol, upstream_symbol):
    jax_fn = getattr(boozer_residual_jax, jax_symbol)
    upstream_fn = getattr(surfaceobjectives, upstream_symbol)
    jax_default = _weight_inv_modB_default(jax_fn)
    upstream_default = _weight_inv_modB_default(upstream_fn)
    assert jax_default == upstream_default, (
        f"{jax_symbol} default {jax_default!r} disagrees with upstream "
        f"{upstream_symbol} default {upstream_default!r}"
    )


def test_boozer_penalty_composed_weight_inv_modB_default_matches_upstream():
    upstream_fn = boozersurface.BoozerSurface.boozer_penalty_constraints_vectorized
    jax_default = _weight_inv_modB_default(boozer_residual_jax.boozer_penalty_composed)
    upstream_default = _weight_inv_modB_default(upstream_fn)
    assert jax_default == upstream_default, (
        f"boozer_penalty_composed default {jax_default!r} disagrees with upstream "
        "BoozerSurface.boozer_penalty_constraints_vectorized default "
        f"{upstream_default!r}"
    )
