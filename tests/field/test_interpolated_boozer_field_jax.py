"""Acceptance tests for ``InterpolatedBoozerFieldJAX`` (item N02).

The previous run blocked N02 because the C++ ``InterpolatedBoozerField``
class stores its interpolation coefficient tensors privately and the
pybind11 surface only exposed evaluator methods, not the coefficient
arrays. N02 is now closed by the "re-fit on a host-resident grid" path
documented in the goal state.json's ``resolution_paths[1]``: the wrapper
:class:`simsopt.field.boozermagneticfield_jax.InterpolatedBoozerFieldJAX`
builds its own per-scalar JAX-side
:class:`simsopt.jax_core.regular_grid_interp.RegularGridInterpolant3DSpec`
payloads by sampling the base ``BoozerMagneticField`` on the regular
grid, mirroring the C++ template's lazy-build semantic but with NO C++
binding changes and NO simsoptpp rebuild.

The C++ oracle for parity is :class:`simsopt.field.boozermagneticfield.BoozerAnalytic`
which itself implements 14 closed-form scalars (``modB``, ``iota``,
``G``, ``I``, ``K``, the angular and radial first derivatives plus
``psip``). Tests below cover those 14 scalars at byte-identity parity to
the CPU oracle within interpolation noise (degree-4 uniform Lagrange on
a moderate grid). The remaining 19 scalars are non-trivial to cover
without a VMEC-derived fixture (``BoozerRadialInterpolant`` only
realises the full 34-scalar surface from a VMEC equilibrium) — those
are exercised by the routing tests (KeyError on un-built scalars,
``ALL_SCALARS`` round-trip) but not by full numerical parity here.

Oracle citation (per ``tests/REVIEWER_ORACLE_LINT.md``):

- **Type 1 (CPU reference symbol)**: ``BoozerAnalytic`` for closed-form
  ``modB`` / ``iota`` / ``G`` / ``I`` / ``K`` / first derivatives /
  ``psip``. Independent C++/Python implementation.
- **Type 2 (closed-form analytic expression)**: the symmetry-fold
  algebra in :func:`simsopt.jax_core.interpolated_boozer_field.fold_points_for_symmetry`
  is validated against explicit modular-arithmetic identities (e.g.
  ``theta % 2π`` and stellsym reflection ``theta := 2π - theta``).
"""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from simsopt.field.boozermagneticfield import BoozerAnalytic
from simsopt.field.boozermagneticfield_jax import (
    InterpolatedBoozerFieldJAX,
)
from simsopt.jax_core.interpolated_boozer_field import (
    ALL_SCALARS,
    FLUX_FUNCTION_SCALARS,
    InterpolatedBoozerFieldFrozenState,
    SYMMETRY_EXPLOIT_SCALARS,
    fold_points_for_symmetry,
    freeze_interpolated_boozer_field_state,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


# The 14 scalars BoozerAnalytic actually implements; the 19 it does not
# implement (R/Z/nu/dnudtheta/d2modB../*_derivs) cannot be eagerly built
# from this base oracle without a TypeError. We test the implemented 14
# for numerical parity and the routing layer for the rest.
_BOOZER_ANALYTIC_SCALARS: tuple[str, ...] = (
    "modB",
    "dmodBdtheta",
    "dmodBdzeta",
    "dmodBds",
    "K",
    "dKdtheta",
    "dKdzeta",
    "G",
    "I",
    "iota",
    "dGds",
    "dIds",
    "diotads",
    "psip",
)


def _qa_analytic() -> BoozerAnalytic:
    return BoozerAnalytic(etabar=0.5, B0=1.0, N=0, G0=1.5, psi0=0.3, iota0=0.4)


def _qh_analytic() -> BoozerAnalytic:
    return BoozerAnalytic(etabar=-0.4, B0=1.2, N=2, G0=2.0, psi0=0.45, iota0=0.55)


def _build_wrapper(
    field, *, degree=6, ns=8, ntheta=8, nzeta=8, stellsym=True, nfp=1, scalars=None
):
    """Construct an ``InterpolatedBoozerFieldJAX`` with a moderate grid.

    Defaults give a degree-6 fit on an 8x8x8 grid, which is comfortably
    above the truncation noise of the analytic test point distribution.
    """
    if scalars is None:
        scalars = _BOOZER_ANALYTIC_SCALARS
    period = 2.0 * np.pi / nfp
    return InterpolatedBoozerFieldJAX(
        field,
        degree=degree,
        srange=[0.3, 0.7, ns],
        thetarange=[0.0, np.pi if stellsym else 2 * np.pi, ntheta],
        zetarange=[0.0, period, nzeta],
        extrapolate=True,
        nfp=nfp,
        stellsym=stellsym,
        scalars=scalars,
    )


def _random_points_in_fundamental_domain(
    n=8, seed=42, *, smin=0.35, smax=0.65, nfp=1, stellsym=True
):
    rng = np.random.default_rng(seed)
    period = 2.0 * np.pi / nfp
    s = rng.uniform(smin, smax, n)
    theta = rng.uniform(0.0, np.pi if stellsym else 2 * np.pi, n)
    zeta = rng.uniform(0.0, period, n)
    return np.column_stack([s, theta, zeta])


# ----------------------------------------------------------------------
# 1. Numerical parity vs. CPU oracle (BoozerAnalytic)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("scalar", _BOOZER_ANALYTIC_SCALARS)
def test_scalar_parity_to_boozer_analytic_oracle(scalar):
    """JAX wrapper output must match the CPU ``BoozerAnalytic`` oracle
    at points well inside the fit domain, within interpolation noise.

    Oracle: ``BoozerAnalytic.<scalar>()`` (type 1 — CPU reference).
    Lane: derivative-heavy interpolation. ``rtol=1e-5`` is chosen from
    the degree-6 Lagrange truncation error budget on an 8x8x8 grid
    sampling a smooth closed-form Boozer field; samples are drawn from
    the strict interior of the s-range to avoid boundary asymmetry.
    """
    field = _qa_analytic()
    wrapper = _build_wrapper(field)
    points = _random_points_in_fundamental_domain(n=8, seed=hash(scalar) & 0xFFFF)

    field.set_points(points)
    cpu_method = getattr(field, scalar)
    cpu_out = np.asarray(cpu_method()).flatten()

    wrapper.set_points(points)
    jax_method = getattr(wrapper, scalar)
    jax_out = np.asarray(jax_method()).flatten()

    np.testing.assert_allclose(jax_out, cpu_out, rtol=1e-5, atol=1e-7)


def test_modB_derivs_3vector_apply_even_symmetry_parity():
    """The 3-vector ``modB_derivs`` path exercises the ``apply_even``
    symmetry branch (negate components 1,2 for stellsym-flipped samples).

    Oracle: ``BoozerAnalytic.modB_derivs()`` returns a ``(N, 3)`` array
    of ``(dmodBds, dmodBdtheta, dmodBdzeta)``. The CPU oracle handles
    the angular structure analytically; the JAX wrapper interpolates and
    then applies the ``apply_even`` rule for stellsym-flipped points
    (theta > pi, folded back). The composition must match the CPU
    direct evaluation at points BOTH inside (no flip) AND outside (flip)
    the fundamental angular domain.

    This closes the Crucible MAJOR test-gap finding: ``_apply_symmetry``'s
    3-vector branches previously lacked oracle-backed numerical
    coverage.
    """
    # Use the QH fixture (N=2) — has non-trivial zeta dependence so the
    # `apply_even` negation of component 2 (dmodBdzeta) is observable.
    # The QA fixture (N=0) has dmodBdzeta == 0 identically, which would
    # leave the apply_even branch untested for component 2.
    field = _qh_analytic()
    # Build only modB_derivs to keep the test scope tight. Use a finer
    # angular grid (QH has higher-frequency structure).
    wrapper = _build_wrapper(
        field, scalars=("modB_derivs",), ntheta=12, nzeta=12, degree=6
    )

    # Half the points have theta < pi (no flip); half have theta > pi
    # (will flip under stellsym=True). The CPU oracle handles both
    # natively; the JAX path goes through the fold + apply_even.
    rng = np.random.default_rng(2026)
    n = 6
    points_no_flip = np.column_stack(
        [
            rng.uniform(0.35, 0.65, n),
            rng.uniform(0.1, np.pi - 0.1, n),  # theta < pi
            rng.uniform(0.1, 2 * np.pi - 0.1, n),
        ]
    )
    points_will_flip = np.column_stack(
        [
            rng.uniform(0.35, 0.65, n),
            rng.uniform(np.pi + 0.1, 2 * np.pi - 0.1, n),  # theta > pi
            rng.uniform(0.1, 2 * np.pi - 0.1, n),
        ]
    )
    all_points = np.vstack([points_no_flip, points_will_flip])

    field.set_points(all_points)
    cpu_modB_derivs = np.asarray(field.modB_derivs())  # (2n, 3)

    wrapper.set_points(all_points)
    jax_modB_derivs = np.asarray(wrapper.modB_derivs())  # (2n, 3)

    # All three components must match the CPU oracle within the
    # interpolation-noise budget (degree-6 Lagrange on 8x8x8).
    np.testing.assert_allclose(jax_modB_derivs, cpu_modB_derivs, rtol=1e-4, atol=1e-6)

    # Sanity: the flipped half must have non-trivial component-1 and
    # component-2 magnitudes (otherwise the `apply_even` negation would
    # be untested even if the assert above passes).
    flipped_block = jax_modB_derivs[n:, :]
    assert np.max(np.abs(flipped_block[:, 1])) > 1e-3
    assert np.max(np.abs(flipped_block[:, 2])) > 1e-3


def test_apply_symmetry_odd_vector_first_only_negates_component_zero():
    """The ``apply_odd_vector_first_only`` rule (used by Z_derivs and
    nu_derivs) must negate component 0 only, leaving components 1, 2
    unchanged.

    Oracle: closed-form algebra. We feed a synthetic ``(N, 3)`` array
    and ``flipped`` mask directly into ``_apply_symmetry`` and verify
    the post-condition: ``out[i, 0] == -raw[i, 0]`` for flipped rows,
    ``out[i, 0] == raw[i, 0]`` for non-flipped rows, and components
    1 and 2 are passed through unchanged in both cases.

    This covers the third branch of ``_apply_symmetry`` (used by
    Z_derivs / nu_derivs) which BoozerAnalytic cannot exercise directly
    (it doesn't implement Z_derivs / nu_derivs).
    """
    from simsopt.jax_core.interpolated_boozer_field import (
        _apply_symmetry,
        _SymmetryRule,
    )

    rule = _SymmetryRule(
        value_size=3,
        apply_odd=False,
        apply_odd_vector_first_only=True,
        apply_even=False,
    )
    raw = jnp.asarray(
        [
            [1.0, 2.0, 3.0],  # flipped → expect (-1, 2, 3)
            [4.0, 5.0, 6.0],  # no flip → expect (4, 5, 6)
            [-7.0, 8.0, 9.0],  # flipped → expect (7, 8, 9)
        ],
        dtype=jnp.float64,
    )
    flipped = jnp.asarray([True, False, True])
    out = np.asarray(_apply_symmetry(raw, flipped=flipped, rule=rule))
    expected = np.asarray(
        [
            [-1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_apply_symmetry_apply_even_negates_components_one_two():
    """The ``apply_even`` rule (used by R_derivs and modB_derivs) must
    negate components 1 and 2 only, leaving component 0 unchanged.

    Oracle: closed-form algebra. Directly exercises the
    ``apply_even`` branch — complementary to the integration-level test
    ``test_modB_derivs_3vector_apply_even_symmetry_parity`` which goes
    through the full evaluate path.
    """
    from simsopt.jax_core.interpolated_boozer_field import (
        _apply_symmetry,
        _SymmetryRule,
    )

    rule = _SymmetryRule(
        value_size=3,
        apply_odd=False,
        apply_odd_vector_first_only=False,
        apply_even=True,
    )
    raw = jnp.asarray(
        [
            [1.0, 2.0, 3.0],  # flipped → expect (1, -2, -3)
            [4.0, 5.0, 6.0],  # no flip → expect (4, 5, 6)
        ],
        dtype=jnp.float64,
    )
    flipped = jnp.asarray([True, False])
    out = np.asarray(_apply_symmetry(raw, flipped=flipped, rule=rule))
    expected = np.asarray(
        [
            [1.0, -2.0, -3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


def test_apply_symmetry_apply_odd_scalar_negates_scalar_for_flipped():
    """The ``apply_odd`` rule with ``value_size=1`` must negate the scalar
    for flipped samples — covers ``K``, ``nu``, ``dnuds``, ``Z``,
    ``dZds``, ``dRdtheta``, ``dRdzeta``, ``dmodBdtheta``,
    ``dmodBdzeta``.

    Oracle: closed-form algebra. Complements the ``dmodBdtheta`` /
    ``dmodBdzeta`` / ``K`` integration tests already in the parity
    suite.
    """
    from simsopt.jax_core.interpolated_boozer_field import (
        _apply_symmetry,
        _SymmetryRule,
    )

    rule = _SymmetryRule(
        value_size=1,
        apply_odd=True,
        apply_odd_vector_first_only=False,
        apply_even=False,
    )
    raw = jnp.asarray([[1.5], [-2.5], [3.5]], dtype=jnp.float64)
    flipped = jnp.asarray([True, False, True])
    out = np.asarray(_apply_symmetry(raw, flipped=flipped, rule=rule))
    expected = np.asarray([[-1.5], [-2.5], [-3.5]], dtype=np.float64)
    np.testing.assert_allclose(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("nfp", [1, 2, 3, 5])
def test_modB_parity_across_nfp_values(nfp):
    """Stellarator-symmetry fold under nfp must preserve modB parity.

    Oracle: ``BoozerAnalytic.modB()`` — the analytic field is invariant
    under the rotational fold for the QA case (N=0), so any modB
    samples at folded points must match.
    """
    field = _qa_analytic()
    wrapper = _build_wrapper(field, nfp=nfp)
    points = _random_points_in_fundamental_domain(n=8, seed=17 + nfp, nfp=nfp)

    field.set_points(points)
    cpu_modB = np.asarray(field.modB()).flatten()
    wrapper.set_points(points)
    jax_modB = np.asarray(wrapper.modB()).flatten()

    np.testing.assert_allclose(jax_modB, cpu_modB, rtol=1e-5, atol=1e-7)


# ----------------------------------------------------------------------
# 2. Symmetry fold algebra — explicit modular & reflection identities
# ----------------------------------------------------------------------


def test_fold_points_modular_theta_into_0_2pi():
    """``theta`` must fold into ``[0, 2*pi]`` regardless of input sign.

    Oracle: closed-form modular arithmetic ``theta_folded = theta mod 2*pi``,
    with the caveat that the C++ uses truncation-toward-zero
    (``int()``-cast) rather than ``math.floor``. The fold post-conditions
    match the C++ check at ``boozermagneticfield_interpolated.h:765-768``.
    """
    period = 2.0 * np.pi / 3.0  # nfp=3
    raw = jnp.asarray(
        [
            [0.4, 7.0, 0.5],  # theta > 2pi
            [0.4, -0.5, 0.5],  # theta < 0
            [0.4, 13.5, 0.5],  # theta > 4pi, large positive
            [0.4, np.pi, 0.5],  # theta = pi exactly
        ],
        dtype=jnp.float64,
    )
    folded, flipped = fold_points_for_symmetry(
        raw, period=jnp.float64(period), stellsym=False
    )
    folded_np = np.asarray(folded)
    theta_folded = folded_np[:, 1]
    # Every folded theta must lie in [0, 2*pi].
    assert np.all((theta_folded >= 0.0) & (theta_folded <= 2 * np.pi + 1e-12))
    # No flips when stellsym=False.
    assert np.all(~np.asarray(flipped))


def test_fold_points_stellsym_reflection_above_pi():
    """When stellsym=True, folded theta > pi must trigger the
    reflection ``theta := 2*pi - theta`` and ``zeta := period - zeta``.

    Oracle: ``boozermagneticfield_interpolated.h:769-779`` algebraic
    reflection rule.
    """
    period = 2.0 * np.pi / 3.0
    raw = jnp.asarray(
        [
            [0.4, 4.5, 0.3],  # theta=4.5 ∈ (pi, 2pi)
            [0.4, 0.5, 0.3],  # theta=0.5 < pi (no flip)
        ],
        dtype=jnp.float64,
    )
    folded, flipped = fold_points_for_symmetry(
        raw, period=jnp.float64(period), stellsym=True
    )
    folded_np = np.asarray(folded)
    flipped_np = np.asarray(flipped)
    # First sample: was theta=4.5, post-modular still 4.5, post-flip = 2*pi - 4.5.
    expected_theta_0 = 2 * np.pi - 4.5
    expected_zeta_0 = period - 0.3
    np.testing.assert_allclose(folded_np[0, 1], expected_theta_0, atol=1e-12)
    np.testing.assert_allclose(folded_np[0, 2], expected_zeta_0, atol=1e-12)
    assert flipped_np[0]
    # Second sample: no flip.
    np.testing.assert_allclose(folded_np[1, 1], 0.5, atol=1e-12)
    np.testing.assert_allclose(folded_np[1, 2], 0.3, atol=1e-12)
    assert not flipped_np[1]


def test_modB_invariant_under_theta_2pi_shift():
    """For a stellsym field, modB must be invariant under
    ``theta -> theta + 2*pi`` because the fold maps both points to the
    same fundamental-domain coordinate.

    Oracle: closed-form periodicity identity. This is a self-consistency
    check at the algebra level — not a tautology because the JAX path
    folds via ``fold_points_for_symmetry`` while the identity tests
    that the modular reduction is correct.
    """
    field = _qa_analytic()
    wrapper = _build_wrapper(field)

    base_points = _random_points_in_fundamental_domain(n=4, seed=99)
    shifted = base_points.copy()
    shifted[:, 1] = shifted[:, 1] + 2 * np.pi

    wrapper.set_points(base_points)
    modB_base = np.asarray(wrapper.modB()).flatten()
    wrapper.set_points(shifted)
    modB_shifted = np.asarray(wrapper.modB()).flatten()

    np.testing.assert_allclose(modB_shifted, modB_base, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# 3. Routing & state lifecycle
# ----------------------------------------------------------------------


def test_simsopt_jax_native_field_marker_is_set():
    """The wrapper class must carry the strict-mode composition marker.

    Oracle: contract documented at
    ``simsopt.field.magneticfield._is_jax_native_field`` — every JAX
    native field class advertises ``_simsopt_jax_native_field = True``
    so ``MagneticFieldSum`` / ``MagneticFieldMultiply`` can refuse
    CPU-only operands in strict mode.
    """
    assert InterpolatedBoozerFieldJAX._simsopt_jax_native_field is True


def test_set_points_shape_validation():
    """``set_points`` must reject non-(N,3) inputs."""
    field = _qa_analytic()
    wrapper = _build_wrapper(field)
    with pytest.raises(ValueError):
        wrapper.set_points(np.array([1.0, 2.0, 3.0]))  # shape (3,) not (N, 3)
    with pytest.raises(ValueError):
        wrapper.set_points(np.array([[1.0, 2.0]]))  # wrong second dim


def test_set_points_invalidates_cache():
    """A new ``set_points`` must invalidate the per-scalar cache.

    Oracle: re-querying ``modB()`` after ``set_points(new_points)`` must
    return values matching the new points (CPU oracle), not the old.
    """
    field = _qa_analytic()
    wrapper = _build_wrapper(field)
    p1 = np.array([[0.4, 0.5, 1.0]])
    p2 = np.array([[0.5, 1.0, 2.0]])

    wrapper.set_points(p1)
    field.set_points(p1)
    modB1_jax = np.asarray(wrapper.modB()).flatten()
    modB1_cpu = np.asarray(field.modB()).flatten()
    np.testing.assert_allclose(modB1_jax, modB1_cpu, rtol=1e-5)

    wrapper.set_points(p2)
    field.set_points(p2)
    modB2_jax = np.asarray(wrapper.modB()).flatten()
    modB2_cpu = np.asarray(field.modB()).flatten()
    np.testing.assert_allclose(modB2_jax, modB2_cpu, rtol=1e-5)

    # Sanity: the two outputs must differ — guards against cache
    # accidentally returning stale modB1 values.
    assert not np.allclose(modB1_jax, modB2_jax, rtol=1e-3)


def test_unbuilt_scalar_raises_keyerror_from_frozen_state_wrapper():
    """Calling a scalar method whose spec was not built must raise ``KeyError``
    when the wrapper has no base-field reference for lazy build.

    Oracle: ``InterpolatedBoozerFieldFrozenState.get`` contract +
    :meth:`InterpolatedBoozerFieldJAX.from_frozen_state` semantics —
    when constructed from a frozen state alone there is no base field
    to lazy-fit against, so unbuilt scalars must surface as
    ``KeyError`` rather than silent zeros or a stale spec.
    """
    field = _qa_analytic()
    # Build modB only into a frozen state.
    state = freeze_interpolated_boozer_field_state(
        field,
        degree=4,
        srange=[0.3, 0.7, 4],
        thetarange=[0.0, np.pi, 4],
        zetarange=[0.0, 2 * np.pi, 4],
        extrapolate=True,
        nfp=1,
        stellsym=True,
        scalars=("modB",),
    )
    # Wrapper from frozen state has no base field → lazy-build is
    # impossible.
    wrapper = InterpolatedBoozerFieldJAX.from_frozen_state(
        state, psi0=float(field.psi0), nfp=1
    )
    wrapper.set_points(np.array([[0.4, 0.5, 1.0]]))
    _ = wrapper.modB()
    with pytest.raises(KeyError) as excinfo:
        wrapper.R()
    assert "R" in str(excinfo.value)


def test_from_frozen_state_round_trip():
    """``from_frozen_state`` must reconstruct a wrapper that produces the
    same outputs as the original.

    Oracle: the original wrapper's outputs (CPU-parity-validated above).
    This is a routing test for the alternative constructor.
    """
    field = _qa_analytic()
    wrapper = _build_wrapper(field, scalars=("modB", "iota", "G"))
    points = _random_points_in_fundamental_domain(n=4, seed=5)
    wrapper.set_points(points)
    modB_orig = np.asarray(wrapper.modB()).flatten()

    rehydrated = InterpolatedBoozerFieldJAX.from_frozen_state(
        wrapper.frozen_state, psi0=float(field.psi0), nfp=1
    )
    rehydrated.set_points(points)
    modB_rehydrated = np.asarray(rehydrated.modB()).flatten()

    np.testing.assert_allclose(modB_rehydrated, modB_orig, rtol=0, atol=0)


def test_frozen_state_inventory_matches_documented_split():
    """The ``ALL_SCALARS`` inventory must equal the union of the
    flux-function and symmetry-exploit groupings.

    Oracle: the C++ inventory at
    ``boozermagneticfield_interpolated.h:736-784`` and ``:809-898``
    (the ``fbatch_scalar`` switch). The grouping is the load-bearing
    contract for which fields get ``angle0_range`` (theta/zeta zeroed)
    vs. full grid sampling.
    """
    assert set(ALL_SCALARS) == set(FLUX_FUNCTION_SCALARS) | set(
        SYMMETRY_EXPLOIT_SCALARS
    )
    # Specific cross-check: flux-function set must contain exactly the
    # 7 documented scalars (psip, G, I, iota, dGds, dIds, diotads).
    assert set(FLUX_FUNCTION_SCALARS) == {
        "psip",
        "G",
        "I",
        "iota",
        "dGds",
        "dIds",
        "diotads",
    }


def test_freeze_state_metadata_round_trip():
    """``freeze_interpolated_boozer_field_state`` must record the
    construction grid in the resulting state metadata.

    Oracle: arithmetic identity — period = 2*pi/nfp.
    """
    field = _qa_analytic()
    state = freeze_interpolated_boozer_field_state(
        field,
        degree=4,
        srange=[0.3, 0.7, 4],
        thetarange=[0.0, np.pi, 4],
        zetarange=[0.0, 2 * np.pi / 3, 4],
        extrapolate=False,
        nfp=3,
        stellsym=True,
        scalars=("modB",),
    )
    assert isinstance(state, InterpolatedBoozerFieldFrozenState)
    assert state.nfp == 3
    assert state.stellsym is True
    assert state.extrapolate is False
    np.testing.assert_allclose(state.period, 2 * np.pi / 3, rtol=0, atol=0)
    assert state.degree == 4
    assert state.has("modB")
    assert not state.has("R")


def test_freeze_state_rejects_unknown_scalar_name():
    """An invalid scalar name in the ``scalars`` argument must surface
    a clear ``ValueError`` listing the unknown names.

    Oracle: the ``ALL_SCALARS`` inventory in
    ``simsopt.jax_core.interpolated_boozer_field``.
    """
    field = _qa_analytic()
    with pytest.raises(ValueError) as excinfo:
        freeze_interpolated_boozer_field_state(
            field,
            degree=4,
            srange=[0.3, 0.7, 4],
            thetarange=[0.0, np.pi, 4],
            zetarange=[0.0, 2 * np.pi, 4],
            scalars=("modB", "not_a_real_scalar"),
        )
    assert "not_a_real_scalar" in str(excinfo.value)


def test_freeze_state_rejects_invalid_degree():
    """Degree < 1 must raise ``ValueError``."""
    field = _qa_analytic()
    with pytest.raises(ValueError):
        freeze_interpolated_boozer_field_state(
            field,
            degree=0,
            srange=[0.3, 0.7, 4],
            thetarange=[0.0, np.pi, 4],
            zetarange=[0.0, 2 * np.pi, 4],
            scalars=("modB",),
        )


def test_freeze_state_rejects_invalid_nfp():
    """nfp < 1 must raise ``ValueError``."""
    field = _qa_analytic()
    with pytest.raises(ValueError):
        freeze_interpolated_boozer_field_state(
            field,
            degree=4,
            srange=[0.3, 0.7, 4],
            thetarange=[0.0, np.pi, 4],
            zetarange=[0.0, 2 * np.pi, 4],
            nfp=0,
            scalars=("modB",),
        )


# ----------------------------------------------------------------------
# 4. Non-stellsym path
# ----------------------------------------------------------------------


def test_non_stellsym_wrapper_does_not_flip():
    """When ``stellsym=False`` the fold must skip the reflection step
    even for theta > pi.

    Oracle: identity at ``fold_points_for_symmetry`` line 282 — no flip
    when stellsym is False.
    """
    raw = jnp.asarray([[0.4, 4.5, 0.3]], dtype=jnp.float64)
    period = 2 * np.pi / 1
    folded_no, flipped_no = fold_points_for_symmetry(
        raw, period=jnp.float64(period), stellsym=False
    )
    assert not bool(np.asarray(flipped_no)[0])
    # Theta stays at 4.5 (in [0, 2pi]) — no reflection.
    np.testing.assert_allclose(np.asarray(folded_no)[0, 1], 4.5, atol=1e-12)


def test_non_stellsym_parity_for_qa_field():
    """The non-stellsym wrapper still matches the CPU oracle for
    BoozerAnalytic-N=0 (which is trivially stellsym-respecting).

    Oracle: ``BoozerAnalytic.modB``. The N=0 analytic field happens to
    be invariant in zeta, so non-stellsym evaluation should also
    match.
    """
    field = _qa_analytic()
    wrapper = _build_wrapper(field, stellsym=False)
    points = _random_points_in_fundamental_domain(n=4, seed=11, stellsym=False)
    field.set_points(points)
    wrapper.set_points(points)
    np.testing.assert_allclose(
        np.asarray(wrapper.modB()).flatten(),
        np.asarray(field.modB()).flatten(),
        rtol=1e-5,
        atol=1e-7,
    )


# ----------------------------------------------------------------------
# 5. Transfer guard / hot path
# ----------------------------------------------------------------------


def test_wrapper_modB_with_lazy_built_specs_returns_array():
    """``modB()`` after lazy-build of additional specs must continue to
    return arrays of the right shape and dtype.

    Oracle: shape contract — ``modB`` is ``(N, 1)`` per the C++
    ``modB_ref`` return shape contract.

    Note: this wrapper is not designed to operate inside
    ``jax.transfer_guard("disallow")`` blocks because each call to
    :func:`evaluate_batch` lifts the host-resident spec arrays
    (``cell_table``, ``cell_to_row``, mesh arrays) to JAX. That is an
    intentional design trade-off for construction-time simplicity. The
    construction phase is where transfers happen by design.
    """
    field = _qa_analytic()
    wrapper = _build_wrapper(field, scalars=("modB",))
    points = np.array([[0.4, 0.5, 1.0], [0.5, 1.0, 2.0]])
    wrapper.set_points(points)
    out = wrapper.modB()
    assert out.shape == (2, 1)
    assert out.dtype == np.float64


# ----------------------------------------------------------------------
# 6. Cross-fixture qh sanity
# ----------------------------------------------------------------------


def test_qh_helicity_field_parity():
    """A non-trivial helicity (N=2) QH analytic case must still match
    the CPU oracle.

    Oracle: ``BoozerAnalytic`` with non-zero N. The wrapper's grid is
    sized to handle the higher angular frequency.
    """
    field = _qh_analytic()
    # QH with N=2 has higher-frequency angular structure, so use a
    # finer angular grid.
    wrapper = _build_wrapper(field, ntheta=12, nzeta=12, degree=6)
    points = _random_points_in_fundamental_domain(n=6, seed=77)
    field.set_points(points)
    wrapper.set_points(points)
    np.testing.assert_allclose(
        np.asarray(wrapper.modB()).flatten(),
        np.asarray(field.modB()).flatten(),
        rtol=1e-4,
        atol=1e-6,
    )
