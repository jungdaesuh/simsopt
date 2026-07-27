from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Coil, Current
from simsopt.geo import create_equally_spaced_curves
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX


def test_coil_specs_preserve_fixed_total_current_expression() -> None:
    curves = create_equally_spaced_curves(
        2,
        1,
        stellsym=False,
        R0=1.0,
        R1=0.25,
        order=2,
        numquadpoints=12,
    )
    for curve in curves:
        curve.fix_all()
    first = Current(3.0)
    total = Current(10.0)
    total.fix_all()
    field = BiotSavartJAX(
        [
            Coil(curves[0], first),
            Coil(curves[1], total - first),
        ]
    )

    def currents(owner_dofs: jax.Array) -> jax.Array:
        return jnp.stack(
            tuple(
                coil.current.value[0] for coil in field.coil_specs_from_dofs(owner_dofs)
            )
        )

    owner_dofs = jnp.asarray([4.0], dtype=jnp.float64)
    np.testing.assert_allclose(currents(owner_dofs), np.asarray([4.0, 6.0]))
    np.testing.assert_allclose(
        jax.jacrev(currents)(owner_dofs),
        np.asarray([[1.0], [-1.0]]),
    )
