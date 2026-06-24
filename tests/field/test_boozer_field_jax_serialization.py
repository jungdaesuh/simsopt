from __future__ import annotations

import jax.numpy as jnp

from simsopt_jax.core.boozer_fixed_state import PiecewisePolynomial1D
from simsopt_jax_adapters.field.boozer_field import (
    BoozerRadialInterpolantFrozenState,
    BoozerRadialInterpolantJAX,
)


def _constant_scalar_profile(value: float) -> PiecewisePolynomial1D:
    return PiecewisePolynomial1D(
        breaks=jnp.array([0.0, 1.0], dtype=jnp.float64),
        coeffs=jnp.array([[value]], dtype=jnp.float64),
    )


def _constant_mode_profile(values: list[float]) -> PiecewisePolynomial1D:
    return PiecewisePolynomial1D(
        breaks=jnp.array([0.0, 1.0], dtype=jnp.float64),
        coeffs=jnp.asarray(values, dtype=jnp.float64)[:, None, None],
    )


def _synthetic_radial_wrapper() -> BoozerRadialInterpolantJAX:
    def zero_modes() -> PiecewisePolynomial1D:
        return _constant_mode_profile([0.0, 0.0])

    state = BoozerRadialInterpolantFrozenState(
        xm=jnp.array([0.0, 1.0], dtype=jnp.float64),
        xn=jnp.array([0.0, 1.0], dtype=jnp.float64),
        psip=_constant_scalar_profile(1.0),
        G=_constant_scalar_profile(2.0),
        I=_constant_scalar_profile(0.2),
        iota=_constant_scalar_profile(0.4),
        dGds=_constant_scalar_profile(0.03),
        dIds=_constant_scalar_profile(0.04),
        diotads=_constant_scalar_profile(0.05),
        bmnc=_constant_mode_profile([1.1, 0.2]),
        dbmncds=_constant_mode_profile([0.01, 0.02]),
        rmnc=_constant_mode_profile([1.4, 0.05]),
        drmncds=_constant_mode_profile([0.02, 0.01]),
        zmns=_constant_mode_profile([0.0, 0.08]),
        dzmnsds=_constant_mode_profile([0.0, 0.01]),
        numns=_constant_mode_profile([0.0, 0.03]),
        dnumnsds=_constant_mode_profile([0.0, 0.004]),
        bmns=zero_modes(),
        dbmnsds=zero_modes(),
        rmns=zero_modes(),
        drmnsds=zero_modes(),
        zmnc=zero_modes(),
        dzmncds=zero_modes(),
        numnc=zero_modes(),
        dnumncds=zero_modes(),
        mn_factor=_constant_mode_profile([1.0, 1.0]),
        d_mn_factor=zero_modes(),
        kmns=_constant_mode_profile([0.0, 0.06]),
        kmnc=zero_modes(),
        stellsym=True,
        no_K=False,
    )
    return BoozerRadialInterpolantJAX.from_frozen_state(state, psi0=1.0, nfp=1).set_points(
        jnp.array([[0.3, 0.2, 0.1]], dtype=jnp.float64)
    )


def test_wrapper_upstream_fallback_stays_serializable_after_rehydration() -> None:
    wrapper = _synthetic_radial_wrapper()
    payload = wrapper.as_dict(serial_objs_dict={})
    assert "upstream" in payload
    assert payload["upstream"] is None

    restored = BoozerRadialInterpolantJAX.from_dict(payload, {}, {})
    restored_payload = restored.as_dict(serial_objs_dict={})
    assert "upstream" in restored_payload
    assert restored_payload["upstream"] is None
