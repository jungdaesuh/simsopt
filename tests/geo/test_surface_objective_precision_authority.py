"""Certificate-precision authority for public JAX surface objectives."""

from __future__ import annotations

import types
from typing import Callable, Dict, List, Union

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt_jax_adapters.geo import surface_objectives as surface_objectives_jax


ObservedValue = Union[bool, int, str, jax.Array]


def test_non_qs_public_value_and_gradients_share_certificate_precision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = object.__new__(surface_objectives_jax.NonQuasiSymmetricRatioJAX)
    objective.axis = 0
    objective._aux_phi_jax = jnp.asarray([0.0], dtype=jnp.float64)
    objective._aux_theta_jax = jnp.asarray([0.0], dtype=jnp.float64)
    objective.boozer_surface = types.SimpleNamespace(
        mpol=1,
        ntor=1,
        nfp=1,
        stellsym=True,
        scatter_indices=None,
        _surface_geometry_kind="rzfourier",
    )
    objective.biotsavart = types.SimpleNamespace(
        coil_set_spec_from_dofs=lambda _coil_dofs: "coil-spec"
    )
    observed_kwargs: List[Dict[str, ObservedValue]] = []

    def qs_ratio(
        sdofs: jax.Array,
        coil_set_spec: str,
        **kwargs: ObservedValue,
    ) -> jax.Array:
        assert coil_set_spec == "coil-spec"
        observed_kwargs.append(dict(kwargs))
        return jnp.sum(sdofs) * jnp.asarray(0.0, dtype=sdofs.dtype)

    def scalar_gradient(
        scalar_fn: Callable[[jax.Array], jax.Array],
        argument: jax.Array,
    ) -> jax.Array:
        scalar_value = scalar_fn(argument)
        assert scalar_value.shape == ()
        return jnp.ones_like(argument)

    monkeypatch.setattr(surface_objectives_jax, "_qs_ratio_pure", qs_ratio)
    monkeypatch.setattr(
        surface_objectives_jax,
        "_strict_scalar_grad",
        scalar_gradient,
    )
    surface_dofs = jnp.asarray([0.2, -0.1], dtype=jnp.float64)
    coil_dofs = jnp.asarray([0.5], dtype=jnp.float64)

    objective._compute_value(surface_dofs, "coil-spec")
    objective._direct_coil_gradient(coil_dofs, surface_dofs)
    objective._compute_dJ_ds("coil-spec", surface_dofs, decision_size=4)

    assert len(observed_kwargs) == 3
    assert all(kwargs["use_compute_dtype"] is False for kwargs in observed_kwargs)
    assert all(kwargs == observed_kwargs[0] for kwargs in observed_kwargs[1:])


def test_non_qs_kernel_forwards_certificate_precision_to_surface_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_use_compute_dtype: List[bool] = []

    def surface_geometry(
        surface_dofs: jax.Array,
        quadpoints_phi: jax.Array,
        quadpoints_theta: jax.Array,
        mpol: int,
        ntor: int,
        nfp: int,
        stellsym: bool,
        scatter_indices: None,
        *,
        surface_kind: str,
        use_compute_dtype: bool,
    ):
        del (
            surface_dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            surface_kind,
        )
        observed_use_compute_dtype.append(use_compute_dtype)
        gamma = jnp.zeros((2, 2, 3), dtype=jnp.float64)
        xphi = jnp.broadcast_to(
            jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64),
            gamma.shape,
        )
        xtheta = jnp.broadcast_to(
            jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64),
            gamma.shape,
        )
        return gamma, xphi, xtheta

    def field_from_spec(points: jax.Array, coil_set_spec: str) -> jax.Array:
        assert coil_set_spec == "coil-spec"
        magnitudes = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
        return jnp.zeros_like(points).at[:, 0].set(magnitudes)

    monkeypatch.setattr(
        surface_objectives_jax,
        "_surface_geometry_from_dofs",
        surface_geometry,
    )
    monkeypatch.setattr(
        surface_objectives_jax,
        "grouped_biot_savart_B_from_spec",
        field_from_spec,
    )

    value = surface_objectives_jax._qs_ratio_pure(
        jnp.asarray([0.1], dtype=jnp.float64),
        "coil-spec",
        jnp.asarray([0.0, 0.5], dtype=jnp.float64),
        jnp.asarray([0.0, 0.5], dtype=jnp.float64),
        1,
        1,
        1,
        True,
        None,
        "rzfourier",
        0,
        use_compute_dtype=False,
    )

    assert observed_use_compute_dtype == [False]
    assert np.isfinite(float(value))
