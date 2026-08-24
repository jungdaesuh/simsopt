"""Canonical native-compatible stochastic sample materialization contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from numpy.random import PCG64DXSM, Generator
from simsopt.field import Coil, Current, coils_via_symmetries
from simsopt.geo import (
    CurveCurveDistance,
    CurvePerturbed,
    CurveXYZFourier,
    GaussianSampler,
    PerturbationSample,
    RotatedCurve,
    create_equally_spaced_curves,
)
from simsopt_jax.core.curve_kernels import curve_curve_distance_penalty_pure
from simsopt_jax.examples import materialize_stochastic_coil_perturbations


def _rotation(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    ).T


def test_materialized_samples_match_native_draw_order_and_symmetry() -> None:
    sampler = GaussianSampler(
        np.linspace(0.0, 1.0, 8, endpoint=False),
        sigma=1.0e-3,
        length_scale=0.5,
        n_derivs=1,
    )
    source_indices = (0, 1, 0, 1)
    rotations = np.stack((np.eye(3), np.eye(3), _rotation(np.pi), _rotation(np.pi)))
    bundle = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=source_indices,
        rotations=rotations,
        base_curve_count=2,
        sample_count=3,
        seed=7,
    )

    randomgen = Generator(PCG64DXSM(7))
    expected_gamma = []
    expected_gammadash = []
    for _ in range(3):
        systematic = tuple(
            PerturbationSample(sampler, randomgen=randomgen) for _ in range(2)
        )
        statistical = tuple(
            PerturbationSample(sampler, randomgen=randomgen) for _ in source_indices
        )
        expected_gamma.append(
            np.stack(
                tuple(
                    systematic[source][0] @ rotation + independent[0]
                    for source, rotation, independent in zip(
                        source_indices,
                        rotations,
                        statistical,
                        strict=True,
                    )
                )
            )
        )
        expected_gammadash.append(
            np.stack(
                tuple(
                    systematic[source][1] @ rotation + independent[1]
                    for source, rotation, independent in zip(
                        source_indices,
                        rotations,
                        statistical,
                        strict=True,
                    )
                )
            )
        )

    np.testing.assert_array_equal(bundle.gamma, np.stack(expected_gamma))
    np.testing.assert_array_equal(bundle.gammadash, np.stack(expected_gammadash))
    assert bundle.generator == "numpy.random.Generator(PCG64DXSM)"
    assert bundle.ordering == "sample:systematic-base-curve,statistical-final-coil"
    assert bundle.seed == 7
    assert bundle.dtype == "<f8"
    assert bundle.byte_order == "little"
    assert len(bundle.sha256) == 64
    assert not bundle.gamma.flags.writeable
    assert not bundle.gammadash.flags.writeable


def test_materialized_sample_hash_binds_values_and_metadata() -> None:
    sampler = GaussianSampler(
        np.linspace(0.0, 1.0, 4, endpoint=False),
        sigma=1.0e-3,
        length_scale=0.5,
        n_derivs=1,
    )
    arguments = {
        "source_indices": (0,),
        "rotations": np.eye(3)[None, :, :],
        "base_curve_count": 1,
        "sample_count": 1,
    }

    first = materialize_stochastic_coil_perturbations(sampler, seed=0, **arguments)
    repeated = materialize_stochastic_coil_perturbations(sampler, seed=0, **arguments)

    metadata_only_change = replace(first, sigma=first.sigma * 2.0)
    changed_gamma = np.array(first.gamma, copy=True)
    changed_gamma[0, 0, 0, 0] = np.nextafter(changed_gamma[0, 0, 0, 0], np.inf)
    values_only_change = replace(first, gamma=changed_gamma)

    assert first.sha256 == repeated.sha256
    np.testing.assert_array_equal(metadata_only_change.gamma, first.gamma)
    np.testing.assert_array_equal(metadata_only_change.gammadash, first.gammadash)
    assert metadata_only_change.sigma != first.sigma
    assert metadata_only_change.sha256 != first.sha256

    assert not np.array_equal(values_only_change.gamma, first.gamma)
    np.testing.assert_array_equal(values_only_change.gammadash, first.gammadash)
    assert values_only_change.seed == first.seed
    assert values_only_change.base_curve_count == first.base_curve_count
    assert values_only_change.source_indices == first.source_indices
    np.testing.assert_array_equal(values_only_change.rotations, first.rotations)
    np.testing.assert_array_equal(
        values_only_change.sampler_points, first.sampler_points
    )
    assert values_only_change.sigma == first.sigma
    assert values_only_change.length_scale == first.length_scale
    assert values_only_change.n_derivs == first.n_derivs
    assert values_only_change.sha256 != first.sha256
    with pytest.raises(ValueError, match="read-only"):
        first.gamma[0, 0, 0, 0] = 1.0


def test_materialized_samples_reproduce_native_perturbed_coil_geometry() -> None:
    nfp = 2
    base_curves = create_equally_spaced_curves(
        2,
        nfp,
        stellsym=True,
        R0=1.0,
        R1=0.25,
        order=2,
        numquadpoints=12,
    )
    base_currents = [Current(1.0e5) for _ in base_curves]
    nominal_coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
    source_indices = tuple(
        index % len(base_curves) for index in range(len(nominal_coils))
    )
    rotations_list = []
    for coil, source_index in zip(nominal_coils, source_indices, strict=True):
        if coil.curve is base_curves[source_index]:
            rotations_list.append(np.eye(3, dtype=np.float64))
        else:
            assert isinstance(coil.curve, RotatedCurve)
            rotations_list.append(np.asarray(coil.curve.rotmat, dtype=np.float64))
    rotations = np.stack(rotations_list)
    sampler = GaussianSampler(
        base_curves[0].quadpoints,
        sigma=1.0e-3,
        length_scale=0.5,
        n_derivs=1,
    )
    bundle = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=source_indices,
        rotations=rotations,
        base_curve_count=len(base_curves),
        sample_count=1,
        seed=7,
    )

    randomgen = Generator(PCG64DXSM(7))
    systematic_curves = [
        CurvePerturbed(
            curve,
            PerturbationSample(sampler, randomgen=randomgen),
        )
        for curve in base_curves
    ]
    systematic_coils = coils_via_symmetries(
        systematic_curves,
        base_currents,
        nfp,
        True,
    )
    perturbed_coils = [
        Coil(
            CurvePerturbed(
                coil.curve,
                PerturbationSample(sampler, randomgen=randomgen),
            ),
            coil.current,
        )
        for coil in systematic_coils
    ]

    actual_gamma = np.stack(tuple(coil.curve.gamma() for coil in nominal_coils))
    actual_gammadash = np.stack(tuple(coil.curve.gammadash() for coil in nominal_coils))
    expected_gamma = np.stack(tuple(coil.curve.gamma() for coil in perturbed_coils))
    expected_gammadash = np.stack(
        tuple(coil.curve.gammadash() for coil in perturbed_coils)
    )
    np.testing.assert_allclose(
        actual_gamma + bundle.gamma[0], expected_gamma, rtol=0.0, atol=5.0e-16
    )
    np.testing.assert_allclose(
        actual_gammadash + bundle.gammadash[0],
        expected_gammadash,
        rtol=0.0,
        atol=2.0e-15,
    )


# The remaining functions mirror tests/geo/test_curveperturbed.py behavior-by-behavior
# against this file's data reformulation. Two native behaviors have no equivalent here
# and are deliberately NOT given tests (see the mirror-wave plan, formerly
# docs/jax_native_test_mirror_wave_implementation_plan.md at commit 2221b542a — removed
# from the branch by the 2026-08-24 docs curation — unit 4, and the coverage manifest):
#   - in-place `CurvePerturbed.resample()`: `StochasticPerturbationBundle` is a frozen
#     dataclass (immutable by design); there is no mutation API to mirror.
#   - `tests/geo/test_curveperturbed.py::test_serialization` (GSONEncoder/GSONDecoder
#     round-trip of a `CurvePerturbed` object): the bundle is plain FP64 tensor data with
#     a SHA-256 fingerprint for reproducibility, not a GSONable object graph.
# A third native behavior, the torsion objective (`LpCurveTorsion`, which needs
# gammadashdash and gammadashdashdash), also has no mirror: `StochasticPerturbationBundle`
# only carries `gamma` and `gammadash` by contract (see its `__post_init__`), so no
# second- or third-derivative perturbation data exists to build one from.
# A fourth native behavior also has no mirror: `test_perturbed_objective_distance`
# below Taylor-tests the objective's *derivative* too (`dJ = J.dJ()`, then
# `assert err_new < 0.55 * err` over shrinking finite-difference steps), not just
# its scalar value. `test_perturbed_curve_distance_objective_matches_native_through_bundle`
# only reproduces the scalar `J()` value -- the JAX side exposes no curve-DOF gradient
# graph for this bundle, so there is nothing to Taylor-test against `dJ()` (see
# coverage manifest row CP-4).


def test_materialized_gammadash_matches_finite_difference_of_gamma() -> None:
    """The bundle's gammadash is the derivative of its gamma, through rotation.

    Mirrors ``tests/geo/test_curveperturbed.py::test_perturbed_gammadash`` (same
    5-point finite-difference stencil), but evaluated on the *materialized bundle*
    rather than a raw ``PerturbationSample``, and with one forced deviation from
    native's ``GaussianSampler`` config: native uses ``n_derivs=2``, while
    ``StochasticPerturbationBundle.__post_init__`` requires ``n_derivs >= 1`` and
    the bundle only ever carries two derivative levels (gamma, gammadash), so this
    mirror is constructed with ``n_derivs=1``. This exercises the reformulation's
    own arithmetic: gamma and gammadash are combined from a systematic sample
    (rotated by a non-identity matrix) and an independent statistical sample, so a
    bug that rotated one branch but not the other would break this derivative
    relationship even though the two source samples are each individually
    consistent. Only one coil is materialized here (``source_indices=(0,)``,
    ``base_curve_count=1``), so this test cannot catch cross-coil mispairing of
    gamma and gammadash.
    """
    sigma = 1.0
    length_scale = 0.5
    points = np.linspace(0.0, 1.0, 200, endpoint=False)
    dphi = points[1]
    sampler = GaussianSampler(points, sigma, length_scale, n_derivs=1)
    rotation = _rotation(np.pi / 3)

    bundle = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=(0,),
        rotations=rotation[None, :, :],
        base_curve_count=1,
        sample_count=1,
        seed=1,
    )
    gamma = bundle.gamma[0, 0]
    gammadash = bundle.gammadash[0, 0]

    gammadash_estimate = (
        (-1 / 12) * gamma[4:, :]
        + (2 / 3) * gamma[3:-1, :]
        + (-2 / 3) * gamma[1:-3, :]
        + (1 / 12) * gamma[0:-4, :]
    )
    gammadash_estimate *= 1 / dphi
    error = np.abs(gammadash_estimate - gammadash[2:-2, :])
    # Measured mean error 4.9e-05 (native's own bound for this stencil/config is 3e-4).
    assert np.mean(error) < 3.0e-4


def test_materialized_samples_preserve_periodicity_of_perturbation() -> None:
    """The bundle repeats after one period, across a rotated multi-coil bundle.

    Mirrors ``tests/geo/test_curveperturbed.py::test_perturbed_periodic``, evaluated
    on a bundle materialized for two coils with distinct source curves and a
    non-identity rotation on the second, proving periodicity survives the
    systematic+statistical combination of the reformulation (not just the
    underlying native ``GaussianSampler`` kernel, which native's own test already
    covers). The non-identity rotation on the second coil is a fixed,
    point-independent linear map, so it cannot itself break periodicity
    (``R @ g(phi) == R @ g(phi + 1)`` identically for any rotation ``R``); its
    presence rules out a bug that skips or mis-applies the rotation on one coil,
    not a periodicity defect in the rotation itself.
    """
    sigma = 1.0
    length_scale = 0.5
    half_period_points = 100
    points = np.linspace(0.0, 2.0, 2 * half_period_points, endpoint=False)
    sampler = GaussianSampler(points, sigma, length_scale, n_derivs=1)
    rotations = np.stack((np.eye(3), _rotation(0.7)))

    bundle = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=(0, 1),
        rotations=rotations,
        base_curve_count=2,
        sample_count=1,
        seed=3,
    )

    for coil_index in range(2):
        first_period = bundle.gamma[0, coil_index, :half_period_points, :]
        second_period = bundle.gamma[0, coil_index, half_period_points:, :]
        periodic_error = np.abs(first_period - second_period)
        # Measured mean error 2.1e-6 / 3.9e-6 for the two coils; native's own
        # single-sample bound is 1e-6, so 1e-5 keeps headroom for the extra
        # (independently-drawn, then rotated) statistical branch summed in here.
        assert np.mean(periodic_error) < 1.0e-5


def test_perturbed_curve_distance_objective_matches_native_through_bundle() -> None:
    """A stage-two-style distance objective evaluated on the bundle matches native.

    Mirrors ``tests/geo/test_curveperturbed.py::test_perturbed_objective_distance``
    (distance objective through perturbed curves, covering position and gammadash),
    using this file's own bundle-materialization path as the JAX-side equivalent of
    ``CurveCurveDistance`` for the two curves the native test builds. The native
    oracle nests two ``CurvePerturbed`` wrappers per curve (systematic, then
    statistical) in the exact draw order ``materialize_stochastic_coil_perturbations``
    uses, so both perturbation branches are exercised through a real downstream
    objective, not just through position equality.
    """
    sigma = 1.0
    length_scale = 0.2
    points = np.linspace(0.0, 1.0, 200, endpoint=False)
    sampler = GaussianSampler(points, sigma, length_scale, n_derivs=1)
    minimum_distance = 2.0
    seed = 11

    order = 4
    nquadpoints = 200
    curve1 = CurveXYZFourier(nquadpoints, order)
    dofs = np.zeros((curve1.dof_size,))
    dofs[1] = 1.0
    dofs[2 * order + 3] = 1.0
    dofs[4 * order + 3] = 1.0
    curve1.x = dofs

    curve2 = CurveXYZFourier(nquadpoints, order)
    dofs = np.zeros((curve2.dof_size,))
    dofs[1] = 2.0
    dofs[2 * order + 3] = 2.0
    dofs[4 * order + 3] = 2.0
    curve2.x = dofs

    randomgen = Generator(PCG64DXSM(seed))
    systematic1 = PerturbationSample(sampler, randomgen=randomgen)
    systematic2 = PerturbationSample(sampler, randomgen=randomgen)
    statistical1 = PerturbationSample(sampler, randomgen=randomgen)
    statistical2 = PerturbationSample(sampler, randomgen=randomgen)
    curve1_perturbed = CurvePerturbed(CurvePerturbed(curve1, systematic1), statistical1)
    curve2_perturbed = CurvePerturbed(CurvePerturbed(curve2, systematic2), statistical2)
    native_objective = CurveCurveDistance(
        [curve1_perturbed, curve2_perturbed], minimum_distance
    ).J()

    bundle = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=(0, 1),
        rotations=np.stack((np.eye(3), np.eye(3))),
        base_curve_count=2,
        sample_count=1,
        seed=seed,
    )
    gamma1 = curve1.gamma() + bundle.gamma[0, 0]
    gammadash1 = curve1.gammadash() + bundle.gammadash[0, 0]
    gamma2 = curve2.gamma() + bundle.gamma[0, 1]
    gammadash2 = curve2.gammadash() + bundle.gammadash[0, 1]

    jax_objective = float(
        curve_curve_distance_penalty_pure(
            gamma1, gammadash1, gamma2, gammadash2, minimum_distance
        )
    )
    # Measured relative error 5.7e-16 (pure recombination of the same terms
    # `cc_distance_pure` sums natively; no reassociation across the two paths).
    assert jax_objective == pytest.approx(native_objective, rel=1.0e-12, abs=1.0e-10)
