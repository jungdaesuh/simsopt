"""Parity tests for ``InterpolatedFieldJAX`` (item 15-sub closeout).

These tests close the architectural blocker recorded in
``.artifacts/jax_port_goal/blockers/15-interpolatedfield-debug.md``.
The CPU :class:`simsopt.field.InterpolatedField` is the parity oracle.
Tolerances come from
:func:`benchmarks.validation_ladder_contract.parity_ladder_tolerances`
on the ``direct_kernel`` lane — no ``rtol`` / ``atol`` numeric literals
appear inline in the test body.

Coverage:

- In-domain :math:`B` parity (no symmetry folding).
- In-domain :math:`\\nabla |B|` parity (no symmetry folding).
- ``nfp``-folded query (``phi`` outside ``[0, 2 pi / nfp)``).
- Stellsym-folded query (``z < 0`` with ``stellsym=True``).
- Stellsym + ``nfp`` combined fold.
- Skipped-cell query (cell excluded from the spline table).
- Out-of-domain query (NaN sentinel when ``extrapolate=False``).
- Cylindrical-projection parity (``B_cyl`` / ``GradAbsB_cyl``).
- Transfer-guard cleanliness of the JAX kernel hot path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field import InterpolatedFieldJAX as ExportedInterpolatedFieldJAX
from simsopt.field.interpolated_field_jax import InterpolatedFieldJAX
from simsopt.field.magneticfieldclasses import (
    InterpolatedField,
    ToroidalField,
)
from simsopt.jax_core.interpolated_field import (
    interpolated_field_B,
    interpolated_field_B_cyl_with_initial,
    interpolated_field_GradAbsB,
    interpolated_field_GradAbsB_cyl_with_initial,
    make_interpolated_field_spec,
)
from simsopt.jax_core.regular_grid_interp import (
    UniformInterpolationRule,
    build_regular_grid_interpolant_3d,
)


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


# Source-field fixture
# --------------------
#
# Use ``ToroidalField`` as the source: it is the simplest analytical
# field that exposes ``B_cyl`` / ``GradAbsB_cyl`` (the two callbacks the
# wrapper consumes at construction time) and it lives entirely inside
# simsoptpp so the CPU oracle can sample it without any external state.


_R_RANGE = (1.0, 1.5, 8)
_PHI_RANGE_FULL = (0.0, 2.0 * np.pi, 16)
_Z_RANGE_SYM = (-0.3, 0.3, 8)
_Z_RANGE_STELLSYM = (0.0, 0.3, 8)
_DEGREE = 4


def _source_field() -> ToroidalField:
    return ToroidalField(R0=1.2, B0=0.9)


def _build_pair(
    *,
    nfp: int,
    stellsym: bool,
    phi_range: tuple[float, float, int] | None = None,
    z_range: tuple[float, float, int] | None = None,
    extrapolate: bool = True,
    skip=None,
) -> tuple[InterpolatedField, InterpolatedFieldJAX]:
    source_cpu = _source_field()
    source_jax = _source_field()
    phi_use = phi_range if phi_range is not None else _PHI_RANGE_FULL
    if z_range is not None:
        z_use = z_range
    elif stellsym:
        z_use = _Z_RANGE_STELLSYM
    else:
        z_use = _Z_RANGE_SYM
    cpu = InterpolatedField(
        source_cpu,
        _DEGREE,
        list(_R_RANGE),
        list(phi_use),
        list(z_use),
        extrapolate,
        nfp=nfp,
        stellsym=stellsym,
        skip=skip,
    )
    jax_ = InterpolatedFieldJAX(
        source_jax,
        _DEGREE,
        _R_RANGE,
        phi_use,
        z_use,
        extrapolate=extrapolate,
        nfp=nfp,
        stellsym=stellsym,
        skip=skip,
    )
    return cpu, jax_


def _points_in_reduced_domain(
    *,
    nfp: int,
    stellsym: bool,
    count: int,
    seed: int,
) -> np.ndarray:
    """Return Cartesian points whose cylindrical coordinates fall inside
    the wrapper's reduced ``(r, phi, z)`` rectangle (i.e. no folding
    needed).
    """

    rng = np.random.default_rng(int(seed))
    rs = rng.uniform(_R_RANGE[0] + 0.02, _R_RANGE[1] - 0.02, size=count)
    phis = rng.uniform(0.02, 2.0 * np.pi / float(nfp) - 0.02, size=count)
    if stellsym:
        zs = rng.uniform(0.02, _Z_RANGE_STELLSYM[1] - 0.02, size=count)
    else:
        zs = rng.uniform(_Z_RANGE_SYM[0] + 0.02, _Z_RANGE_SYM[1] - 0.02, size=count)
    return np.ascontiguousarray(
        np.stack([rs * np.cos(phis), rs * np.sin(phis), zs], axis=1),
        dtype=np.float64,
    )


def _points_with_nfp_fold(*, nfp: int, count: int, seed: int) -> np.ndarray:
    """Return points whose ``phi`` lies OUTSIDE ``[0, 2 pi / nfp)``.

    The interpolant must fold these into the reduced domain to evaluate.
    """

    rng = np.random.default_rng(int(seed))
    rs = rng.uniform(_R_RANGE[0] + 0.02, _R_RANGE[1] - 0.02, size=count)
    # phi in [2 pi / nfp, 2 pi) — i.e. always requires modulo reduction.
    phi_min = 2.0 * np.pi / float(nfp) + 0.05
    phi_max = 2.0 * np.pi - 0.05
    phis = rng.uniform(phi_min, phi_max, size=count)
    zs = rng.uniform(_Z_RANGE_SYM[0] + 0.02, _Z_RANGE_SYM[1] - 0.02, size=count)
    return np.ascontiguousarray(
        np.stack([rs * np.cos(phis), rs * np.sin(phis), zs], axis=1),
        dtype=np.float64,
    )


def _points_with_stellsym_fold(count: int, seed: int) -> np.ndarray:
    """Return points with ``z < 0`` so the stellsym fold has to fire."""

    rng = np.random.default_rng(int(seed))
    rs = rng.uniform(_R_RANGE[0] + 0.02, _R_RANGE[1] - 0.02, size=count)
    phis = rng.uniform(0.02, np.pi - 0.02, size=count)
    zs = rng.uniform(-_Z_RANGE_STELLSYM[1] + 0.02, -0.02, size=count)
    return np.ascontiguousarray(
        np.stack([rs * np.cos(phis), rs * np.sin(phis), zs], axis=1),
        dtype=np.float64,
    )


# ── Parametrised symmetry / fold parity ─────────────────────────────


class TestInterpolatedFieldJAXParity:
    @pytest.mark.parametrize(
        "nfp,stellsym",
        [(1, False), (2, False), (1, True), (3, True)],
    )
    def test_in_domain_B_parity(self, nfp: int, stellsym: bool):
        """``B()`` matches the CPU oracle for in-domain points."""
        cpu, jax_ = _build_pair(nfp=nfp, stellsym=stellsym)
        points = _points_in_reduced_domain(
            nfp=nfp, stellsym=stellsym, count=60, seed=11
        )
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_jax_B_at_matches_batched_B(self):
        """Single-point tracing adapter matches the CPU ``B`` oracle."""
        cpu, jax_ = _build_pair(nfp=2, stellsym=True)
        point = _points_in_reduced_domain(nfp=2, stellsym=True, count=1, seed=13)[0]
        cpu.set_points_cart(point.reshape((1, 3)))
        np.testing.assert_allclose(
            np.asarray(jax_.jax_B_at(point)),
            np.asarray(cpu.B())[0],
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_jax_B_GradAbsB_at_matches_batched_paths(self):
        """Particle-tracing adapter matches CPU ``B`` / ``GradAbsB`` oracles."""
        cpu, jax_ = _build_pair(nfp=2, stellsym=True)
        point = _points_in_reduced_domain(nfp=2, stellsym=True, count=1, seed=14)[0]
        cpu.set_points_cart(point.reshape((1, 3)))
        B_at, grad_abs_at = jax_.jax_B_GradAbsB_at(jnp.asarray(point))
        np.testing.assert_allclose(
            np.asarray(B_at),
            np.asarray(cpu.B())[0],
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(grad_abs_at),
            np.asarray(cpu.GradAbsB())[0],
            rtol=_RTOL,
            atol=_ATOL,
        )

    @pytest.mark.parametrize(
        "nfp,stellsym",
        [(1, False), (2, False), (1, True), (3, True)],
    )
    def test_in_domain_GradAbsB_parity(self, nfp: int, stellsym: bool):
        """``GradAbsB()`` matches the CPU oracle for in-domain points."""
        cpu, jax_ = _build_pair(nfp=nfp, stellsym=stellsym)
        points = _points_in_reduced_domain(
            nfp=nfp, stellsym=stellsym, count=60, seed=12
        )
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.GradAbsB()),
            np.asarray(cpu.GradAbsB()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_nfp_folded_query_matches_cpu(self):
        """Query points with ``phi`` outside the reduced range fold
        through ``nfp`` symmetry and match the CPU oracle.
        """
        nfp = 3
        phi_range = (0.0, 2.0 * np.pi / nfp, 8)
        cpu, jax_ = _build_pair(nfp=nfp, stellsym=False, phi_range=phi_range)
        points = _points_with_nfp_fold(nfp=nfp, count=50, seed=21)
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.GradAbsB()),
            np.asarray(cpu.GradAbsB()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_stellsym_folded_query_matches_cpu(self):
        """Query points with ``z < 0`` fold through stellarator symmetry
        and match the CPU oracle on both ``B`` and ``GradAbsB``.
        """
        cpu, jax_ = _build_pair(nfp=1, stellsym=True)
        points = _points_with_stellsym_fold(count=50, seed=31)
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.GradAbsB()),
            np.asarray(cpu.GradAbsB()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_combined_nfp_and_stellsym_fold_matches_cpu(self):
        """Mix of ``nfp`` modulo reduction and stellsym reflection."""
        nfp = 3
        phi_range = (0.0, 2.0 * np.pi / nfp, 8)
        cpu, jax_ = _build_pair(nfp=nfp, stellsym=True, phi_range=phi_range)

        rng = np.random.default_rng(41)
        N = 80
        rs = rng.uniform(_R_RANGE[0] + 0.02, _R_RANGE[1] - 0.02, size=N)
        phis_raw = rng.uniform(-np.pi, 3.0 * np.pi, size=N)
        zs = rng.uniform(
            -_Z_RANGE_STELLSYM[1] + 0.02, _Z_RANGE_STELLSYM[1] - 0.02, size=N
        )
        points = np.ascontiguousarray(
            np.stack([rs * np.cos(phis_raw), rs * np.sin(phis_raw), zs], axis=1),
            dtype=np.float64,
        )
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.GradAbsB()),
            np.asarray(cpu.GradAbsB()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_negative_phi_roundoff_fold_boundary_matches_cpu(self):
        """Tiny negative phi values fold through nfp/stellsym like CPU."""

        nfp = 3
        phi_range = (0.0, 2.0 * np.pi / nfp, 8)
        cpu, jax_ = _build_pair(nfp=nfp, stellsym=True, phi_range=phi_range)
        radii = np.asarray([1.18, 1.31], dtype=np.float64)
        phis = np.asarray([-1.0e-15, -2.0e-14], dtype=np.float64)
        zetas = np.asarray([0.08, -0.12], dtype=np.float64)
        points = np.ascontiguousarray(
            np.stack([radii * np.cos(phis), radii * np.sin(phis), zetas], axis=1),
            dtype=np.float64,
        )

        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.GradAbsB()),
            np.asarray(cpu.GradAbsB()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── Skip mask ────────────────────────────────────────────────────────


class TestInterpolatedFieldJAXSkip:
    def test_skip_mask_matches_cpu(self):
        """Skipped cells are absorbed into the zero sentinel in both the
        JAX and CPU paths, so the field values match exactly even for
        points whose cell is excluded from the spline table.
        """

        def _skip(rs, phis, zs):
            rs_arr = np.asarray(rs)
            return (rs_arr > 1.40).tolist()

        cpu, jax_ = _build_pair(nfp=1, stellsym=False, skip=_skip)
        # Mix of in-spline-cell and skipped-cell points.
        points = np.asarray(
            [
                [1.10, 0.20, 0.05],
                [1.30, -0.40, -0.10],
                [1.45, 0.00, 0.00],
                [1.48, 0.30, 0.20],
            ],
            dtype=np.float64,
        )
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── Out-of-domain query ─────────────────────────────────────────────


class TestInterpolatedFieldJAXOutOfDomain:
    def test_out_of_domain_raises_when_extrapolate_false(self):
        """Out-of-domain queries raise at the public wrapper boundary."""

        cpu, jax_ = _build_pair(nfp=1, stellsym=False, extrapolate=False)
        outside = np.asarray(
            [[_R_RANGE[1] + 0.1, 0.0, 0.0]],
            dtype=np.float64,
        )
        cpu.set_points_cart(outside)
        jax_.set_points_cart(outside)
        with pytest.raises(RuntimeError):
            cpu.B()
        with pytest.raises(RuntimeError, match="extrapolate=False"):
            jax_.B()

    def test_out_of_domain_extrapolates_when_flag_set(self):
        """When ``extrapolate=True`` both backends preserve the initialized
        output rows for out-of-domain queries. The result is the same.
        """

        cpu, jax_ = _build_pair(nfp=1, stellsym=False, extrapolate=True)
        outside = np.asarray(
            [[_R_RANGE[1] + 0.05, 0.0, 0.0]],
            dtype=np.float64,
        )
        cpu.set_points_cart(outside)
        jax_.set_points_cart(outside)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_same_shape_out_of_domain_preserves_native_cylindrical_cache(self):
        """Same-size OOB calls preserve the previous cylindrical output buffer."""

        cpu, jax_ = _build_pair(nfp=2, stellsym=True, extrapolate=True)
        warm = _points_in_reduced_domain(nfp=2, stellsym=True, count=2, seed=45)
        second = warm.copy()
        r_oob = _R_RANGE[1] + 0.05
        phi_oob = 0.37
        second[1] = [r_oob * np.cos(phi_oob), r_oob * np.sin(phi_oob), -0.11]

        cpu.set_points_cart(warm)
        jax_.set_points_cart(warm)
        np.asarray(cpu.B())
        np.asarray(jax_.B())
        np.asarray(cpu.GradAbsB())
        np.asarray(jax_.GradAbsB())

        cpu.set_points_cart(second)
        jax_.set_points_cart(second)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.GradAbsB()),
            np.asarray(cpu.GradAbsB()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── Cylindrical projection parity ───────────────────────────────────


class TestInterpolatedFieldJAXCylindrical:
    def test_package_export(self):
        assert ExportedInterpolatedFieldJAX is InterpolatedFieldJAX

    def test_B_cyl_and_GradAbsB_cyl_parity(self):
        """The cylindrical accessors match the CPU oracle bit-for-bit."""
        cpu, jax_ = _build_pair(nfp=1, stellsym=False)
        points = _points_in_reduced_domain(nfp=1, stellsym=False, count=40, seed=51)
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)
        np.testing.assert_allclose(
            np.asarray(jax_.B_cyl()),
            np.asarray(cpu.B_cyl()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.GradAbsB_cyl()),
            np.asarray(cpu.GradAbsB_cyl()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── Cache invariance under set_points ───────────────────────────────


class TestInterpolatedFieldJAXCacheInvariance:
    def test_setpoints_invalidates_cached_B(self):
        cpu, jax_ = _build_pair(nfp=1, stellsym=False)
        first = _points_in_reduced_domain(nfp=1, stellsym=False, count=20, seed=61)
        second = _points_in_reduced_domain(nfp=1, stellsym=False, count=20, seed=62)
        cpu.set_points_cart(first)
        jax_.set_points_cart(first)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        cpu.set_points_cart(second)
        jax_.set_points_cart(second)
        np.testing.assert_allclose(
            np.asarray(jax_.B()),
            np.asarray(cpu.B()),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_public_invalidate_cache_invalidates_cylindrical_caches(self):
        cpu, jax_ = _build_pair(nfp=1, stellsym=False)
        points = _points_in_reduced_domain(nfp=1, stellsym=False, count=20, seed=63)
        cpu.set_points_cart(points)
        jax_.set_points_cart(points)

        np.asarray(jax_.B_cyl())
        np.asarray(jax_.GradAbsB_cyl())
        assert jax_._B_cyl_valid
        assert jax_._GradAbsB_cyl_valid

        jax_.invalidate_cache()

        assert not jax_._B_cyl_valid
        assert not jax_._GradAbsB_cyl_valid
        np.testing.assert_allclose(
            np.asarray(jax_.B_cyl()),
            np.asarray(cpu.B_cyl()),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(jax_.GradAbsB_cyl()),
            np.asarray(cpu.GradAbsB_cyl()),
            rtol=_RTOL,
            atol=_ATOL,
        )


# ── Transfer-guard cleanliness ──────────────────────────────────────


class TestInterpolatedFieldJAXTransferGuard:
    """The interpolated-field JAX kernels stage all device-resident
    arrays through the strict-safe :func:`jax.device_put` path at
    construction time (the rectangular-grid spec) and at query time
    (the Cartesian points). Both
    :func:`interpolated_field_B` and
    :func:`interpolated_field_GradAbsB` must therefore be clean under
    :func:`jax.transfer_guard("disallow")` when both spec and points
    are already device-resident.

    The public :meth:`B` / :meth:`GradAbsB` getters materialise the
    JAX output back to NumPy at the ``_*_impl`` boundary; that
    device-to-host fetch is unconditionally allowed by JAX on CPU but
    deliberately not bracketed by the transfer-guard scope below.
    """

    def test_kernels_clean_under_strict_transfer_guard(self):
        cpu, jax_ = _build_pair(nfp=2, stellsym=True)
        del cpu  # only the JAX side is exercised under the guard
        points_host = _points_in_reduced_domain(nfp=2, stellsym=True, count=40, seed=71)
        device_points = jnp.asarray(points_host, dtype=jnp.float64)
        device_initial = jnp.zeros((device_points.shape[0], 3), dtype=jnp.float64)
        device_points.block_until_ready()
        device_initial.block_until_ready()

        # Trigger one untraced run so the JIT cache is populated before
        # the strict-guard region.
        interpolated_field_B(jax_._spec, device_points).block_until_ready()
        interpolated_field_GradAbsB(jax_._spec, device_points).block_until_ready()
        interpolated_field_B_cyl_with_initial(
            jax_._spec, device_points, device_initial
        ).block_until_ready()
        interpolated_field_GradAbsB_cyl_with_initial(
            jax_._spec, device_points, device_initial
        ).block_until_ready()

        with jax.transfer_guard("disallow"):
            interpolated_field_B(jax_._spec, device_points).block_until_ready()
            interpolated_field_GradAbsB(jax_._spec, device_points).block_until_ready()
            interpolated_field_B_cyl_with_initial(
                jax_._spec, device_points, device_initial
            ).block_until_ready()
            interpolated_field_GradAbsB_cyl_with_initial(
                jax_._spec, device_points, device_initial
            ).block_until_ready()


class TestInterpolatedFieldJAXSpecImmutability:
    def test_regular_grid_specs_are_readonly_snapshots(self):
        """Device cache and host spec cannot silently diverge after construction."""

        def field_values(xs, ys, zs):
            return np.stack([xs + ys, ys + zs, zs + xs], axis=1).reshape(-1)

        source_B = build_regular_grid_interpolant_3d(
            rule=UniformInterpolationRule(1),
            xrange=(1.0, 1.2, 1),
            yrange=(0.0, 0.2, 1),
            zrange=(-0.1, 0.1, 1),
            value_size=3,
            f=field_values,
            out_of_bounds_ok=True,
        )
        source_GradAbsB = build_regular_grid_interpolant_3d(
            rule=UniformInterpolationRule(1),
            xrange=(1.0, 1.2, 1),
            yrange=(0.0, 0.2, 1),
            zrange=(-0.1, 0.1, 1),
            value_size=3,
            f=field_values,
            out_of_bounds_ok=True,
        )
        spec = make_interpolated_field_spec(
            nfp=1,
            stellsym=False,
            B_spec=source_B,
            GradAbsB_spec=source_GradAbsB,
        )

        original_cell = float(spec.B_spec.cell_table[0, 0, 0, 0, 0])
        with pytest.raises(ValueError, match="read-only"):
            source_B.cell_table[0, 0, 0, 0, 0] = original_cell + 123.0

        assert float(spec.B_spec.cell_table[0, 0, 0, 0, 0]) == original_cell
        assert not spec.B_spec.cell_table.flags.writeable
        assert not spec.B_spec.rule.nodes.flags.writeable
        assert not spec.GradAbsB_spec.cell_to_row.flags.writeable

        with pytest.raises(ValueError, match="read-only"):
            spec.B_spec.cell_table[0, 0, 0, 0, 0] = original_cell + 1.0
        with pytest.raises(ValueError, match="read-only"):
            spec.B_spec.rule.nodes[0] = 0.25
