"""Correctness gates for the vendored flat-675 single-stage objective.

Every assertion here is anchored on the genuine-675 campaign's archived
native-C++ certificate for the same start candidate, so the port is measured
against a value this repository did not produce.

The gates in order: the objective and its gradient must reproduce the archived
certificate; the two-scalar inner solve must reproduce the archived ``y``
certificate; and the three vessel coordinates must be differentiably connected
to the objective, which the certificate can only witness indirectly because
the archived candidate sits outside the surface-to-vessel hinge.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax_adapters.geo.flat675 import (
    FLAT675_COIL_SLICE,
    FLAT675_OBJECTIVE_TERM_KEYS,
    FLAT675_SURFACE_SLICE,
    FLAT675_VESSEL_SLICE,
    Flat675Bundle,
    build_flat675_boozer_system,
    flat675_candidate_geometry,
    flat675_objective,
    flat675_weighted_terms,
    load_flat675_bundle,
    solve_flat675_y_qr,
)

ARTIFACT_ROOT = Path.home() / "simsopt_mixed_artifacts"
BUNDLE_ROOT = ARTIFACT_ROOT / "genuine675-r3-input-1c23f6c5-20260721-r1"
CERTIFICATE_PATH = (
    ARTIFACT_ROOT
    / "genuine675-fixed-budget-maxiter3-r3-1c23f6c5-20260721T124425Z"
    / "triad"
    / "native_cpp_cpu"
    / "lane.json"
)

# The certificate names its terms in physical language; the shared math names
# them by penalty.  The order is the certified assembly order in both.
CERTIFICATE_TERM_NAMES = (
    "non_qs",
    "boozer_residual",
    "iota_penalty",
    "curve_length",
    "curve_curve_distance",
    "curve_surface_distance",
    "surface_vessel_distance",
    "curve_curvature",
)

# The archived candidate's surface clears the vessel by 0.0529 m, so raising
# the threshold past that distance is what activates the hinge whose gradient
# reaches the vessel block.
ARCHIVED_MINIMUM_SURFACE_VESSEL_DISTANCE_M = 0.05292932805459816
ACTIVATING_SURFACE_VESSEL_THRESHOLD_M = 0.2
# The vessel gradient this port measures at that activated threshold.
ACTIVATED_VESSEL_GRADIENT = (
    0.35290288750786897,
    -0.3310626658438312,
    -0.28375651703846133,
)

pytestmark = pytest.mark.skipif(
    not BUNDLE_ROOT.is_dir() or not CERTIFICATE_PATH.is_file(),
    reason="the frozen genuine-675 input bundle and certificate are host-local",
)


@dataclass(frozen=True)
class ArchivedCertificate:
    """The numbers the archived native lane published at the start candidate."""

    objective_value: float
    term_order: tuple[str, ...]
    weighted_terms: dict[str, float]
    gradient: np.ndarray
    inner_solution: np.ndarray
    inner_singular_values: np.ndarray
    inner_numerical_rank: int


def _load_archived_certificate() -> ArchivedCertificate:
    """Read the archived lane record into a typed handle.

    The lane document is deep untyped JSON; narrowing it once here keeps every
    test below reading like a comparison instead of a traversal.
    """
    lane = json.loads(CERTIFICATE_PATH.read_bytes())
    certificate = lane["result"]["initial_certificate"]
    inner = certificate["y_certificate"]
    return ArchivedCertificate(
        objective_value=float(certificate["objective_value"]),
        term_order=tuple(str(name) for name in certificate["objective_term_order"]),
        weighted_terms={
            str(entry["term"]): float(entry["weighted_value"])
            for entry in certificate["objective_terms"]
        },
        gradient=np.asarray(certificate["gradient"]["full_675"], dtype=np.float64),
        inner_solution=np.asarray(inner["solution"], dtype=np.float64),
        inner_singular_values=np.asarray(inner["singular_values"], dtype=np.float64),
        inner_numerical_rank=int(inner["numerical_rank"]),
    )


@pytest.fixture(scope="module")
def bundle() -> Flat675Bundle:
    return load_flat675_bundle(BUNDLE_ROOT)


@pytest.fixture(scope="module")
def archived() -> ArchivedCertificate:
    return _load_archived_certificate()


@pytest.fixture(scope="module")
def start_coordinates(bundle: Flat675Bundle) -> jnp.ndarray:
    return jnp.asarray(bundle.start_candidate.outer_vector())


@pytest.fixture(scope="module")
def value_and_gradient(
    bundle: Flat675Bundle,
    start_coordinates: jnp.ndarray,
) -> tuple[float, np.ndarray]:
    value, gradient = jax.value_and_grad(
        lambda coordinates: flat675_objective(
            coordinates,
            material=bundle.material,
            objective_policy=bundle.objective_policy,
            boozer_policy=bundle.boozer_policy,
        )
    )(start_coordinates)
    return float(value), np.asarray(gradient, dtype=np.float64)


def test_start_candidate_reproduces_the_archived_objective_value(
    value_and_gradient: tuple[float, np.ndarray],
    archived: ArchivedCertificate,
) -> None:
    """GATE 1: the ported objective is the certified scalar, not a lookalike."""
    value, _gradient = value_and_gradient
    expected = archived.objective_value

    relative_error = abs(value - expected) / abs(expected)
    assert relative_error < 1.0e-12, (
        "the flat-675 objective disagrees with the archived native certificate "
        f"at the start candidate: got {value!r}, expected {expected!r} "
        f"(relative error {relative_error:.3e})"
    )


def test_start_candidate_reproduces_every_archived_weighted_term(
    bundle: Flat675Bundle,
    start_coordinates: jnp.ndarray,
    archived: ArchivedCertificate,
) -> None:
    """A matching total built from mismatched terms would still be wrong."""
    terms = np.asarray(
        flat675_weighted_terms(
            start_coordinates,
            material=bundle.material,
            objective_policy=bundle.objective_policy,
            boozer_policy=bundle.boozer_policy,
        ),
        dtype=np.float64,
    )

    assert archived.term_order == CERTIFICATE_TERM_NAMES, (
        "the archived certificate changed its term order"
    )
    for key, name, value in zip(
        FLAT675_OBJECTIVE_TERM_KEYS,
        CERTIFICATE_TERM_NAMES,
        terms,
        strict=True,
    ):
        expected = archived.weighted_terms[name]
        if expected == 0.0:
            assert value == 0.0, (
                f"the {key} term is inactive in the archived certificate but "
                f"the port evaluated it to {value!r}"
            )
            continue
        relative_error = abs(value - expected) / abs(expected)
        assert relative_error < 1.0e-9, (
            f"the {key} term disagrees with the archived certificate: "
            f"got {value!r}, expected {expected!r} "
            f"(relative error {relative_error:.3e})"
        )


def test_start_candidate_reproduces_the_archived_gradient(
    value_and_gradient: tuple[float, np.ndarray],
    archived: ArchivedCertificate,
) -> None:
    """GATE 1: all 675 coordinates differentiate the way the archive recorded."""
    _value, gradient = value_and_gradient
    expected = archived.gradient

    assert gradient.shape == expected.shape
    scale = float(np.max(np.abs(expected)))
    scaled_error = float(np.max(np.abs(gradient - expected))) / scale
    assert scaled_error < 1.0e-12, (
        "the flat-675 gradient disagrees with the archived native certificate: "
        f"largest component error is {scaled_error:.3e} of the gradient's "
        f"inf-norm {scale!r}"
    )


def test_start_candidate_reproduces_the_archived_inner_solve_certificate(
    bundle: Flat675Bundle,
    start_coordinates: jnp.ndarray,
    archived: ArchivedCertificate,
) -> None:
    """The two closed inner scalars and their conditioning match the archive."""
    geometry = flat675_candidate_geometry(
        bundle.material.boozer,
        start_coordinates[FLAT675_COIL_SLICE],
        start_coordinates[FLAT675_SURFACE_SLICE],
    )
    system = build_flat675_boozer_system(geometry, bundle.boozer_policy)
    inner_solve = solve_flat675_y_qr(system.design_matrix, system.right_hand_side)

    np.testing.assert_allclose(
        np.asarray(inner_solve.solution, dtype=np.float64),
        archived.inner_solution,
        rtol=1.0e-12,
        atol=0.0,
        err_msg="the closed (iota, G) inner state differs from the archive",
    )
    np.testing.assert_allclose(
        np.asarray(inner_solve.singular_values, dtype=np.float64),
        archived.inner_singular_values,
        rtol=1.0e-12,
        atol=0.0,
        err_msg="the Boozer system's conditioning differs from the archive",
    )
    assert int(inner_solve.numerical_rank) == archived.inner_numerical_rank
    assert bool(inner_solve.numerics_finite) is True


def test_vessel_gradient_is_exactly_zero_outside_the_hinge(
    value_and_gradient: tuple[float, np.ndarray],
    bundle: Flat675Bundle,
) -> None:
    """GATE 2a: the archived candidate clears the vessel, so its gradient is 0.

    This is the negative half of the vessel-liveness pair.  A port that leaked
    some other vessel dependence into the objective would move these three
    components away from the archive's exact zeros.
    """
    _value, gradient = value_and_gradient

    assert (
        ARCHIVED_MINIMUM_SURFACE_VESSEL_DISTANCE_M
        > bundle.objective_policy.surface_vessel_threshold_m
    ), (
        "the archived candidate no longer clears the surface-to-vessel "
        "threshold, so the zero below would stop being the expected value"
    )
    np.testing.assert_array_equal(
        gradient[FLAT675_VESSEL_SLICE],
        np.zeros(3),
        err_msg="the flat-675 objective grew a vessel dependence the archived "
        "native certificate does not have",
    )


def test_vessel_gradient_is_live_once_the_hinge_activates(
    bundle: Flat675Bundle,
    start_coordinates: jnp.ndarray,
) -> None:
    """GATE 2b: the three vessel coordinates are differentiably connected.

    Raising only the surface-to-vessel threshold past the archived clearance
    activates the one term the vessel block enters.  If the vessel geometry
    reached the objective as a host constant rather than as traced geometry,
    this gradient would stay at exactly zero and the flat formulation would be
    silently optimizing 672 coordinates instead of 675.
    """
    activated_policy = dataclasses.replace(
        bundle.objective_policy,
        surface_vessel_threshold_m=ACTIVATING_SURFACE_VESSEL_THRESHOLD_M,
    )
    gradient = np.asarray(
        jax.grad(
            lambda coordinates: flat675_objective(
                coordinates,
                material=bundle.material,
                objective_policy=activated_policy,
                boozer_policy=bundle.boozer_policy,
            )
        )(start_coordinates),
        dtype=np.float64,
    )

    vessel_gradient = gradient[FLAT675_VESSEL_SLICE]
    assert np.all(np.isfinite(vessel_gradient))
    np.testing.assert_allclose(
        vessel_gradient,
        ACTIVATED_VESSEL_GRADIENT,
        rtol=1.0e-11,
        atol=0.0,
        err_msg=(
            "the activated vessel gradient moved off its recorded value; a "
            "vessel geometry that stopped being traced reads as exact zeros, "
            "and any other value means the surface-to-vessel term changed"
        ),
    )
