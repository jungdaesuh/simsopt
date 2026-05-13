"""JAX port of ``simsopt.geo.surface.SurfaceClassifier`` (item 14 helper).

This module wraps the JAX
:class:`~simsopt.jax_core.regular_grid_interp.RegularGridInterpolant3DSpec`
into a JIT-friendly closure that mirrors the
``LevelsetStoppingCriterion`` predicate used in the fieldline and
guiding-centre tracing paths. The classifier takes a 1-channel signed-distance interpolant in
``(r, phi, z)`` coordinates and exposes a Cartesian-input callable that
returns ``+1`` strictly inside the volume / ``-1`` strictly outside.

The signed-distance grid contract matches the upstream
``simsopt.geo.surface.SurfaceClassifier.__init__`` (see
``opensource/simsopt/src/simsopt/geo/surface.py:925``): the interpolant
is built on a uniform cylindrical mesh with ``phi`` in ``[0, 2*pi]``,
``r`` in some ``[r_min, r_max]``, ``z`` in some ``[z_min, z_max]``, and
``value_size=1``. Out-of-domain queries return ``-1`` via
``out_of_bounds_ok=True`` so the classifier naturally treats points
outside the cuboid as outside the surface — this matches the upstream
fallback established at ``surface.py:973``.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from .regular_grid_interp import (
    RegularGridInterpolant3DSpec,
    evaluate_batch,
)

__all__ = [
    "make_levelset_classifier",
    "signed_distance_to_cartesian_classifier",
]


def signed_distance_to_cartesian_classifier(
    interpolant_spec: RegularGridInterpolant3DSpec,
) -> Callable[[jax.Array], jax.Array]:
    """Return a callable mapping Cartesian points to ``+1`` (inside) / ``-1`` (outside).

    Parameters
    ----------
    interpolant_spec
        A :class:`~simsopt.jax_core.regular_grid_interp.RegularGridInterpolant3DSpec`
        with ``value_size == 1``, built on the cylindrical signed-distance
        grid described in :func:`make_levelset_classifier`.

    Returns
    -------
    classifier
        ``classifier(xyz)`` accepts either a single Cartesian point of
        shape ``[3]`` or a batch of shape ``[N, 3]`` and returns a
        scalar / ``[N]`` float64 array. Values are ``+1.0`` strictly
        inside the volume (signed distance positive), ``-1.0`` outside
        (signed distance negative), and ``0.0`` on the surface (signed
        distance exactly zero, which only occurs on the discretised
        grid). The sign convention matches the upstream
        ``LevelsetStoppingCriterion`` semantics where positive denotes
        "above the level set" (inside).
    """

    if int(interpolant_spec.value_size) != 1:
        raise ValueError(
            "signed-distance classifier requires value_size == 1, "
            f"got value_size = {int(interpolant_spec.value_size)}"
        )

    def classify(xyz: jax.Array) -> jax.Array:
        xyz_arr = jnp.asarray(xyz, dtype=jnp.float64)
        was_single = xyz_arr.ndim == 1
        if was_single:
            xyz_arr = xyz_arr.reshape((1, 3))
        if xyz_arr.ndim != 2 or xyz_arr.shape[-1] != 3:
            raise ValueError(
                f"classifier input must have shape [3] or [N, 3]; got {xyz_arr.shape}"
            )
        r = jnp.linalg.norm(xyz_arr[:, :2], axis=1)
        phi = jnp.mod(
            jnp.arctan2(xyz_arr[:, 1], xyz_arr[:, 0]),
            jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64),
        )
        z = xyz_arr[:, 2]
        rphiz = jnp.stack([r, phi, z], axis=-1)
        dist = evaluate_batch(interpolant_spec, rphiz)
        dist_flat = dist.reshape((-1,))
        in_bounds = (
            (r >= jnp.asarray(interpolant_spec.xmin, dtype=jnp.float64))
            & (r <= jnp.asarray(interpolant_spec.xmax, dtype=jnp.float64))
            & (phi >= jnp.asarray(interpolant_spec.ymin, dtype=jnp.float64))
            & (phi <= jnp.asarray(interpolant_spec.ymax, dtype=jnp.float64))
            & (z >= jnp.asarray(interpolant_spec.zmin, dtype=jnp.float64))
            & (z <= jnp.asarray(interpolant_spec.zmax, dtype=jnp.float64))
        )
        result = jnp.sign(jnp.where(in_bounds, dist_flat, -1.0))
        if was_single:
            return result[0]
        return result

    return classify


def make_levelset_classifier(
    interpolant_spec: RegularGridInterpolant3DSpec,
) -> Callable[[jax.Array], jax.Array]:
    """Return a JAX-traceable surface classifier built on the item-13 interpolant.

    The returned callable replicates the public surface of the
    upstream ``LevelsetStoppingCriterion`` (``sopp.LevelsetStoppingCriterion``)
    in JAX: positive output denotes "inside" the bounded volume, negative
    "outside". The Cartesian fieldline and guiding-centre drivers consume
    this helper through their fixed-shape stopping-criterion loop.

    Parameters
    ----------
    interpolant_spec
        A :class:`~simsopt.jax_core.regular_grid_interp.RegularGridInterpolant3DSpec`
        with ``value_size == 1`` and ``out_of_bounds_ok == True``. The
        underlying scalar field is expected to be the signed distance to
        a toroidal surface evaluated on a cylindrical ``(r, phi, z)``
        grid that fully covers the surface plus a small padding (the
        upstream classifier pads ``[r_min - 0.1, r_max + 0.1]`` etc.).

    Returns
    -------
    classify
        ``classify(xyz)`` is the same callable returned by
        :func:`signed_distance_to_cartesian_classifier`. Provided as a
        public alias to match the naming convention used elsewhere in
        the JAX port (``make_*_classifier`` mirrors the
        ``make_*_spec`` factories in ``simsopt.jax_core.specs``).
    """

    if not bool(interpolant_spec.out_of_bounds_ok):
        raise ValueError(
            "make_levelset_classifier requires out_of_bounds_ok=True so that "
            "queries outside the cuboid map to -1 (outside)."
        )
    return signed_distance_to_cartesian_classifier(interpolant_spec)
