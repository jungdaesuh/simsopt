"""Adapter-owned conversions for legacy surface classifiers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from simsopt_jax.core.regular_grid_interp import (
    UniformInterpolationRule,
    build_regular_grid_interpolant_3d,
)
from simsopt_jax.core.surface_classifier import make_levelset_classifier


_LEVELSET_CLASSIFIER_ATTRS = (
    "dist",
    "_interpolation_degree",
    "_interpolant_xrange",
    "_interpolant_yrange",
    "_interpolant_zrange",
)


def supports_levelset_classifier_conversion(classifier: object) -> bool:
    """Return whether a legacy classifier exposes the grid metadata JAX needs."""
    return all(hasattr(classifier, name) for name in _LEVELSET_CLASSIFIER_ATTRS)


def levelset_classifier_fn_from_surface_classifier(
    classifier: object,
) -> Callable[[object], object]:
    """Build the JAX levelset classifier closure from a legacy classifier."""
    if not supports_levelset_classifier_conversion(classifier):
        raise TypeError(
            "Expected a SurfaceClassifier-compatible object with regular-grid "
            "metadata for JAX levelset conversion."
        )

    def fbatch(rs, phis, zs):
        rphiz = np.stack((rs, phis, zs), axis=1)
        values = -np.ones((rphiz.shape[0], 1))
        classifier.dist.evaluate_batch(rphiz, values)
        return values.reshape(-1)

    spec = build_regular_grid_interpolant_3d(
        rule=UniformInterpolationRule(classifier._interpolation_degree),
        xrange=classifier._interpolant_xrange,
        yrange=classifier._interpolant_yrange,
        zrange=classifier._interpolant_zrange,
        value_size=1,
        f=fbatch,
        out_of_bounds_ok=True,
    )
    return make_levelset_classifier(spec)


__all__ = (
    "levelset_classifier_fn_from_surface_classifier",
    "supports_levelset_classifier_conversion",
)
