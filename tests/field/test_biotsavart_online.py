"""Transform and numerical contracts for the online mixed ``B`` scout."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt_jax.backend import invalidate_backend_cache
from simsopt_jax.core import field as core_field
import simsopt_jax.core.biotsavart_online as _online
from simsopt_jax.core.specs import (
    CoilGroupSpec,
    GroupedCoilSetSpec,
    make_coil_dof_extraction_spec,
    make_coil_set_dof_extraction_spec,
    make_curve_xyzfourier_spec,
    make_optimizable_dof_map_spec,
)
from simsopt_jax.core.surface_rzfourier import (
    surface_rz_fourier_geometry_from_spec,
    surface_rz_fourier_spec_from_dofs,
)


jax.config.update("jax_enable_x64", True)

_MU0_OVER_4PI = np.float64(1.0e-7)
_flatten_grouped_biot_savart_sources = _online._flatten_grouped_biot_savart_sources
_mixed_biot_savart_B_online_from_flat_sources = (
    _online._mixed_biot_savart_B_online_from_flat_sources
)


def test_online_public_surface_hides_flat_source_representation():
    """Only the grouped operation is part of the online module contract."""
    assert _online.__all__ == ["mixed_grouped_biot_savart_B_online"]
    for removed_name in (
        "flatten_biot_savart_sources",
        "flatten_grouped_biot_savart_sources",
        "mixed_biot_savart_B_online",
    ):
        assert not hasattr(_online, removed_name)


def _fixture(*, point_count: int = 3, source_count: int = 10, seed: int = 1701):
    rng = np.random.default_rng(seed)
    points = rng.normal(scale=0.12, size=(point_count, 3))
    source_positions = rng.normal(scale=0.15, size=(source_count, 3))
    source_positions[:, 0] += 1.1
    source_vectors = rng.normal(scale=8.0e4, size=(source_count, 3))
    return tuple(
        jnp.asarray(value, dtype=jnp.float32)
        for value in (points, source_positions, source_vectors)
    )


def _directions(primals, *, seed: int):
    rng = np.random.default_rng(seed)
    return tuple(
        jnp.asarray(rng.normal(scale=0.03, size=value.shape), dtype=jnp.float32)
        for value in primals
    )


def _dense_float64_reference(points, source_positions, source_vectors):
    points = jnp.asarray(points, dtype=jnp.float64)
    source_positions = jnp.asarray(source_positions, dtype=jnp.float64)
    source_vectors = jnp.asarray(source_vectors, dtype=jnp.float64)
    diff = source_positions[None, :, :] - points[:, None, :]
    radius_squared = jnp.sum(diff * diff, axis=-1)
    cross = jnp.cross(diff, source_vectors[None, :, :])
    contributions = cross / (radius_squared * jnp.sqrt(radius_squared))[..., None]
    return jnp.asarray(_MU0_OVER_4PI) * jnp.sum(contributions, axis=1)


def _candidate(*args):
    return _mixed_biot_savart_B_online_from_flat_sources(
        *args,
        source_tile_size=4,
    )


def _assert_close(actual, expected, *, rtol: float, atol: float) -> None:
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=rtol,
        atol=atol,
    )


def test_online_custom_jvp_primal_matches_primal_entrypoint_bitwise():
    primals = _fixture(source_count=10, seed=1731)
    tangents = _directions(primals, seed=1732)

    direct_primal = _candidate(*primals)
    jvp_primal, _ = jax.jvp(_candidate, primals, tangents)

    direct_array = np.asarray(direct_primal)
    jvp_array = np.asarray(jvp_primal)
    assert jvp_array.dtype == direct_array.dtype
    assert jvp_array.shape == direct_array.shape
    assert jvp_array.tobytes() == direct_array.tobytes()


def test_online_custom_jvp_shares_tile_geometry_in_lowering():
    """The fused JVP must lower one radius evaluation per source tile."""
    primals = _fixture(source_count=8, seed=1738)
    tangents = _directions(primals, seed=1739)

    def value_and_tangent(*args):
        return jax.jvp(_candidate, args[:3], args[3:])

    stablehlo = str(
        jax.jit(value_and_tangent)
        .lower(*primals, *tangents)
        .compiler_ir(dialect="stablehlo")
    )

    assert stablehlo.count("stablehlo.sqrt") == 1


def test_online_custom_jvp_tangent_is_additive():
    primals = _fixture(source_count=10, seed=1733)
    left_directions = _directions(primals, seed=1734)
    right_directions = _directions(primals, seed=1735)
    combined_directions = tuple(
        left + right
        for left, right in zip(left_directions, right_directions, strict=True)
    )

    left = jax.jvp(_candidate, primals, left_directions)[1]
    right = jax.jvp(_candidate, primals, right_directions)[1]
    combined = jax.jvp(_candidate, primals, combined_directions)[1]

    _assert_close(combined, left + right, rtol=1.2e-5, atol=5.0e-9)


def test_online_custom_jvp_tangent_is_homogeneous():
    primals = _fixture(source_count=10, seed=1736)
    directions = _directions(primals, seed=1737)
    scale = jnp.asarray(-0.375, dtype=jnp.float32)
    scaled_directions = tuple(scale * direction for direction in directions)

    tangent = jax.jvp(_candidate, primals, directions)[1]
    scaled_tangent = jax.jvp(_candidate, primals, scaled_directions)[1]

    _assert_close(scaled_tangent, scale * tangent, rtol=2.0e-6, atol=2.0e-12)


@pytest.mark.parametrize(
    ("fixture_seed", "direction_seed"),
    (
        (1701, 1702),
        (1707, 2707),
        (1714, 2714),
        (1720, 2720),
        (1722, 2722),
        (1724, 2724),
    ),
)
def test_online_primal_and_jvp_match_dense_float64_reference(
    fixture_seed,
    direction_seed,
):
    primals = _fixture(seed=fixture_seed)
    tangents = _directions(primals, seed=direction_seed)

    actual_primal, actual_tangent = jax.jvp(_candidate, primals, tangents)
    expected_primal, expected_tangent = jax.jvp(
        _dense_float64_reference,
        primals,
        tangents,
    )

    assert actual_primal.dtype == jnp.float64
    assert actual_tangent.dtype == jnp.float64
    # The same 24-seed audit bounded primal absolute error at 6.8201e-9;
    # 7.5e-9 leaves 10% headroom while retaining a 3e-6 relative bound.
    _assert_close(actual_primal, expected_primal, rtol=3.0e-6, atol=7.5e-9)
    # A deterministic 24-seed audit observed a 2.2455e-9 maximum absolute
    # error. Its 1.343e-4 relative outlier was an 8.58e-6 near-zero component.
    # Keep 11% headroom on that measured FP32 envelope rather than weakening
    # the relative bound around well-scaled components.
    _assert_close(actual_tangent, expected_tangent, rtol=8.0e-6, atol=2.5e-9)


def test_online_vjp_matches_dense_float64_reference():
    primals = _fixture(seed=1703)
    cotangent = jnp.asarray(
        np.random.default_rng(1704).normal(size=(primals[0].shape[0], 3)),
        dtype=jnp.float64,
    )

    actual_value, actual_pullback = jax.vjp(_candidate, *primals)
    expected_value, expected_pullback = jax.vjp(_dense_float64_reference, *primals)
    actual_cotangents = actual_pullback(cotangent)
    expected_cotangents = expected_pullback(cotangent)

    _assert_close(actual_value, expected_value, rtol=3.0e-6, atol=2.0e-10)
    for actual, expected in zip(
        actual_cotangents,
        expected_cotangents,
        strict=True,
    ):
        assert actual.dtype == jnp.float32
        _assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-10)


def test_online_second_jvp_matches_dense_float64_reference():
    primals = _fixture(seed=1705)
    first_directions = _directions(primals, seed=1706)
    second_directions = _directions(primals, seed=1707)

    def second_directional(function):
        def first_directional(*args):
            return jax.jvp(function, args, first_directions)[1]

        return jax.jvp(first_directional, primals, second_directions)[1]

    actual = second_directional(_candidate)
    expected = second_directional(_dense_float64_reference)
    _assert_close(actual, expected, rtol=8.0e-5, atol=3.0e-11)


def test_online_reverse_over_jvp_matches_dense_float64_reference():
    primals = _fixture(seed=1708)
    directions = _directions(primals, seed=1709)
    cotangent = jnp.asarray(
        np.random.default_rng(1710).normal(size=(primals[0].shape[0], 3)),
        dtype=jnp.float64,
    )

    def reverse_over_jvp(function):
        def contracted_directional(*args):
            directional = jax.jvp(function, args, directions)[1]
            return jnp.vdot(cotangent, directional).real

        return jax.grad(contracted_directional, argnums=(0, 1, 2))(*primals)

    actual = reverse_over_jvp(_candidate)
    expected = reverse_over_jvp(_dense_float64_reference)
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        _assert_close(actual_leaf, expected_leaf, rtol=1.2e-4, atol=5.0e-10)


def _make_quadrature_group(*, coil_count: int, quadrature_count: int, phase: float):
    theta = np.linspace(0.0, 2.0 * np.pi, quadrature_count, endpoint=False)
    gammas = []
    gammadashs = []
    for coil_index in range(coil_count):
        angle = theta + phase + 2.0 * np.pi * coil_index / coil_count
        radius = 0.9 + 0.03 * coil_index
        gammas.append(
            np.stack(
                (
                    radius * np.cos(angle),
                    radius * np.sin(angle),
                    0.08 * np.sin(2.0 * angle),
                ),
                axis=-1,
            )
        )
        gammadashs.append(
            np.stack(
                (
                    -radius * np.sin(angle),
                    radius * np.cos(angle),
                    0.16 * np.cos(2.0 * angle),
                ),
                axis=-1,
            )
        )
    currents = np.linspace(8.0e4, 1.1e5, coil_count)
    return tuple(
        jnp.asarray(value, dtype=jnp.float32)
        for value in (np.stack(gammas), np.stack(gammadashs), currents)
    )


def _grouped_float64_reference(points, groups):
    result = jnp.zeros((points.shape[0], 3), dtype=jnp.float64)
    for gammas, gammadashs, currents in groups:
        gamma64 = jnp.asarray(gammas, dtype=jnp.float64)
        gammadash64 = jnp.asarray(gammadashs, dtype=jnp.float64)
        current64 = jnp.asarray(currents, dtype=jnp.float64)
        diff = (
            gamma64[None, :, :, :]
            - jnp.asarray(
                points,
                dtype=jnp.float64,
            )[:, None, None, :]
        )
        radius_squared = jnp.sum(diff * diff, axis=-1)
        contributions = (
            jnp.cross(diff, gammadash64[None, :, :, :])
            / (radius_squared * jnp.sqrt(radius_squared))[..., None]
        )
        coil_integrals = jnp.sum(contributions, axis=2) / gammas.shape[1]
        result = result + jnp.asarray(_MU0_OVER_4PI) * jnp.einsum(
            "c,pcj->pj",
            current64,
            coil_integrals,
        )
    return result


def _grouped_spec(groups):
    group_specs = []
    coil_offset = 0
    for gammas, gammadashs, currents in groups:
        coil_count = int(currents.shape[0])
        group_specs.append(
            CoilGroupSpec(
                gammas=gammas,
                gammadashs=gammadashs,
                currents=currents,
                coil_indices=tuple(range(coil_offset, coil_offset + coil_count)),
            )
        )
        coil_offset += coil_count
    return GroupedCoilSetSpec(groups=tuple(group_specs))


def _grouped_spec_directions(coil_spec, *, seed: int):
    rng = np.random.default_rng(seed)
    return jax.tree.map(
        lambda value: jnp.asarray(
            rng.normal(scale=0.03, size=value.shape),
            dtype=jnp.float32,
        ),
        coil_spec,
    )


def _production_grouped_candidate(points, coil_spec):
    return core_field.grouped_biot_savart_B_from_spec(points, coil_spec)


def _production_grouped_reference(points, coil_spec):
    return _grouped_float64_reference(points, coil_spec.field_inputs())


def _production_dispatch_fixture():
    points = jnp.asarray(
        [[0.20, 0.02, 0.01], [-0.14, 0.11, -0.04], [0.08, -0.16, 0.07]],
        dtype=jnp.float32,
    )
    groups = (
        _make_quadrature_group(coil_count=2, quadrature_count=15, phase=0.1),
        _make_quadrature_group(coil_count=3, quadrature_count=128, phase=0.4),
    )
    grouped_spec = _grouped_spec(groups)
    return points, GroupedCoilSetSpec(
        groups=(
            CoilGroupSpec(
                gammas=grouped_spec.groups[0].gammas,
                gammadashs=grouped_spec.groups[0].gammadashs,
                currents=grouped_spec.groups[0].currents,
                coil_indices=(3, 0),
            ),
            CoilGroupSpec(
                gammas=grouped_spec.groups[1].gammas,
                gammadashs=grouped_spec.groups[1].gammadashs,
                currents=grouped_spec.groups[1].currents,
                coil_indices=(4, 1, 2),
            ),
        )
    )


def _production_reconstruction_fixture():
    curve_definitions = (
        (
            np.asarray(
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.3],
                dtype=np.float64,
            ),
            np.asarray([8.5e4], dtype=np.float64),
            15,
            -1.5,
        ),
        (
            np.asarray(
                [0.15, 0.8, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0, 0.2],
                dtype=np.float64,
            ),
            np.asarray([9.2e4], dtype=np.float64),
            128,
            0.75,
        ),
    )

    def dof_map(template, owner_start):
        size = int(template.size)
        return make_optimizable_dof_map_spec(
            template_full_dofs=template,
            owner_segments=((owner_start, owner_start + size, 0, size),),
            input_mode="full",
            input_start=0,
            input_end=size,
        )

    owner_parts = []
    extraction_specs = []
    owner_offset = 0
    for curve_dofs, current, quadrature_count, scale in curve_definitions:
        curve = make_curve_xyzfourier_spec(
            dofs=curve_dofs,
            quadpoints=np.linspace(
                0.0,
                1.0,
                quadrature_count,
                endpoint=False,
                dtype=np.float64,
            ),
            order=1,
        )
        current_offset = owner_offset + curve_dofs.size
        extraction_specs.append(
            make_coil_dof_extraction_spec(
                curve=curve,
                curve_map=dof_map(curve_dofs, owner_offset),
                current_map=dof_map(current, current_offset),
                scale=scale,
            )
        )
        owner_parts.extend((curve_dofs, current))
        owner_offset = current_offset + current.size

    extraction_spec = make_coil_set_dof_extraction_spec(tuple(extraction_specs))
    owner_dofs = jnp.asarray(np.concatenate(owner_parts), dtype=jnp.float64)
    return extraction_spec, owner_dofs


def _surface_points(*, use_compute_dtype: bool):
    spec = surface_rz_fourier_spec_from_dofs(
        jnp.asarray([1.2, 0.1, 0.08], dtype=jnp.float64),
        quadpoints_phi=jnp.asarray([0.0, 0.25], dtype=jnp.float64),
        quadpoints_theta=jnp.asarray([0.0, 1.0 / 3.0, 2.0 / 3.0], dtype=jnp.float64),
        mpol=1,
        ntor=0,
        nfp=1,
        stellsym=True,
        use_compute_dtype=use_compute_dtype,
    )
    gamma, _gammadash1, _gammadash2 = surface_rz_fourier_geometry_from_spec(spec)
    return gamma.reshape(-1, 3)


def test_production_reconstruction_reaches_online_mixed_field_boundary(monkeypatch):
    extraction_spec, owner_dofs = _production_reconstruction_fixture()
    mode = "jax_gpu_fast" if jax.devices()[0].platform == "gpu" else "jax_cpu_fast"
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", mode)
    monkeypatch.setenv("SIMSOPT_PRECISION", "mixed")
    invalidate_backend_cache()
    try:
        mixed_spec = core_field.coil_set_spec_from_dof_extraction_spec(
            extraction_spec,
            owner_dofs,
            use_compute_dtype=True,
        )
        certificate_spec = core_field.coil_set_spec_from_dof_extraction_spec(
            extraction_spec,
            owner_dofs,
            use_compute_dtype=False,
        )
        surface_points = _surface_points(use_compute_dtype=True)
        certificate_points = _surface_points(use_compute_dtype=False)

        assert surface_points.dtype == jnp.float32
        assert certificate_points.dtype == jnp.float64
        assert all(leaf.dtype == jnp.float64 for leaf in jax.tree.leaves(mixed_spec))
        assert all(
            leaf.dtype == jnp.float64 for leaf in jax.tree.leaves(certificate_spec)
        )
        assert tuple(group.gammas.shape[1] for group in mixed_spec.groups) == (15, 128)
        assert mixed_spec.coil_index_lists() == ((0,), (1,))

        sentinel = jnp.full(
            (surface_points.shape[0], 3),
            19.0,
            dtype=jnp.float64,
        )
        online_calls = []

        def online(kernel_points, coil_arrays):
            online_calls.append((kernel_points, coil_arrays))
            return sentinel

        monkeypatch.setattr(core_field, "_mixed_online_grouped_biot_savart_B", online)

        mixed_result = core_field.grouped_biot_savart_B_from_spec(
            surface_points,
            mixed_spec,
        )
        certificate_result = core_field.grouped_biot_savart_B_from_spec(
            certificate_points,
            certificate_spec,
        )
        certificate_reference = core_field._accumulate_grouped_field(
            certificate_points,
            certificate_spec,
            core_field.biot_savart_B,
        )
    finally:
        monkeypatch.delenv("SIMSOPT_PRECISION", raising=False)
        monkeypatch.delenv("SIMSOPT_BACKEND_MODE", raising=False)
        invalidate_backend_cache()

    assert mixed_result is sentinel
    assert len(online_calls) == 1
    online_points, online_groups = online_calls[0]
    assert online_points.dtype == jnp.float32
    assert all(leaf.dtype == jnp.float32 for group in online_groups for leaf in group)
    assert (
        np.asarray(certificate_result).tobytes()
        == np.asarray(certificate_reference).tobytes()
    )


def test_online_flattening_preserves_q15_and_q128_group_weights():
    groups = (
        _make_quadrature_group(coil_count=2, quadrature_count=15, phase=0.1),
        _make_quadrature_group(coil_count=3, quadrature_count=128, phase=0.4),
    )
    points = jnp.asarray(
        [[0.20, 0.02, 0.01], [-0.14, 0.11, -0.04], [0.08, -0.16, 0.07]],
        dtype=jnp.float32,
    )
    source_positions, source_vectors = _flatten_grouped_biot_savart_sources(groups)

    actual = _online.mixed_grouped_biot_savart_B_online(
        points,
        groups,
        source_tile_size=64,
    )
    expected = _grouped_float64_reference(points, groups)

    assert source_positions.shape == (2 * 15 + 3 * 128, 3)
    assert source_vectors.shape == source_positions.shape
    _assert_close(actual, expected, rtol=4.0e-6, atol=3.0e-10)


def test_production_grouped_dispatch_selects_online_only_for_mixed_float32(
    monkeypatch,
):
    points, coil_spec = _production_dispatch_fixture()
    sentinel = jnp.full((points.shape[0], 3), 7.0, dtype=jnp.float64)
    selected = []

    monkeypatch.setattr(core_field, "is_mixed_precision_enabled", lambda: True)
    monkeypatch.setattr(
        core_field,
        "coil_group_collective_config",
        lambda currents: None,
    )

    def online(kernel_points, coil_arrays):
        selected.append((kernel_points, coil_arrays))
        return sentinel

    monkeypatch.setattr(core_field, "_mixed_online_grouped_biot_savart_B", online)

    result = core_field.grouped_biot_savart_B_from_spec(points, coil_spec)

    assert result is sentinel
    assert len(selected) == 1
    assert selected[0][0] is points
    for selected_group, expected_group in zip(
        selected[0][1],
        coil_spec.field_inputs(),
        strict=True,
    ):
        assert all(
            selected_array is expected_array
            for selected_array, expected_array in zip(
                selected_group,
                expected_group,
                strict=True,
            )
        )


@pytest.mark.parametrize(
    ("mixed_precision", "float32_inputs", "collective_active"),
    (
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ),
)
def test_production_grouped_dispatch_preserves_reference_fallbacks(
    monkeypatch,
    mixed_precision,
    float32_inputs,
    collective_active,
):
    points, coil_spec = _production_dispatch_fixture()
    if not float32_inputs:
        points = jnp.asarray(points, dtype=jnp.float64)
        coil_spec = jax.tree.map(
            lambda value: jnp.asarray(value, dtype=jnp.float64),
            coil_spec,
        )
    sentinel = jnp.full((points.shape[0], 3), 11.0, dtype=jnp.float64)
    fallback_calls = []

    monkeypatch.setattr(
        core_field,
        "is_mixed_precision_enabled",
        lambda: mixed_precision,
    )
    monkeypatch.setattr(
        core_field,
        "coil_group_collective_config",
        lambda currents: object() if collective_active else None,
    )
    monkeypatch.setattr(
        core_field,
        "_mixed_online_grouped_biot_savart_B",
        lambda *args: pytest.fail("online path must not be selected"),
    )

    def fallback(kernel_points, fallback_spec, kernel):
        fallback_calls.append((kernel_points, fallback_spec, kernel))
        return sentinel

    monkeypatch.setattr(core_field, "_accumulate_grouped_field", fallback)

    result = core_field.grouped_biot_savart_B_from_spec(points, coil_spec)

    assert result is sentinel
    assert fallback_calls == [(points, coil_spec, core_field.biot_savart_B)]


def test_production_grouped_dispatch_preserves_point_sharding_fallback(monkeypatch):
    points, coil_spec = _production_dispatch_fixture()
    sentinel = jnp.full((points.shape[0], 3), 13.0, dtype=jnp.float64)
    monkeypatch.setattr(core_field, "is_mixed_precision_enabled", lambda: True)
    monkeypatch.setattr(
        core_field,
        "get_sharding_tuning",
        lambda: SimpleNamespace(
            active=True,
            strategy="points",
            min_points_to_shard=1,
        ),
    )
    monkeypatch.setattr(
        core_field,
        "_mixed_online_grouped_biot_savart_B",
        lambda *args: pytest.fail("online path must not replace point sharding"),
    )
    monkeypatch.setattr(
        core_field,
        "_accumulate_grouped_field",
        lambda kernel_points, fallback_spec, kernel: sentinel,
    )

    result = core_field.grouped_biot_savart_B_from_spec(points, coil_spec)

    assert result is sentinel


def test_production_grouped_dispatch_preserves_fp64_bits(monkeypatch):
    points, coil_spec = _production_dispatch_fixture()
    points = jnp.asarray(points, dtype=jnp.float64)
    coil_spec = jax.tree.map(
        lambda value: jnp.asarray(value, dtype=jnp.float64),
        coil_spec,
    )
    monkeypatch.setattr(core_field, "is_mixed_precision_enabled", lambda: True)

    actual = core_field.grouped_biot_savart_B_from_spec(points, coil_spec)
    expected = core_field._accumulate_grouped_field(
        points,
        coil_spec,
        core_field.biot_savart_B,
    )

    assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()


def test_production_grouped_dispatch_primal_and_jvp_match_fp64(monkeypatch):
    points, coil_spec = _production_dispatch_fixture()
    assert coil_spec.coil_index_lists() == ((3, 0), (4, 1, 2))
    point_direction = jnp.asarray(
        np.random.default_rng(1725).normal(scale=0.03, size=points.shape),
        dtype=jnp.float32,
    )
    coil_direction = _grouped_spec_directions(coil_spec, seed=1726)
    monkeypatch.setattr(core_field, "is_mixed_precision_enabled", lambda: True)
    monkeypatch.setattr(
        core_field,
        "coil_group_collective_config",
        lambda currents: None,
    )

    actual = jax.jvp(
        _production_grouped_candidate,
        (points, coil_spec),
        (point_direction, coil_direction),
    )
    expected = jax.jvp(
        _production_grouped_reference,
        (points, coil_spec),
        (point_direction, coil_direction),
    )

    _assert_close(actual[0], expected[0], rtol=4.0e-6, atol=3.0e-10)
    _assert_close(actual[1], expected[1], rtol=1.2e-5, atol=3.0e-9)


def test_production_grouped_dispatch_custom_jvp_primal_matches_direct_bits(
    monkeypatch,
):
    points, coil_spec = _production_dispatch_fixture()
    directions = (
        jnp.asarray(
            np.random.default_rng(1738).normal(scale=0.03, size=points.shape),
            dtype=jnp.float32,
        ),
        _grouped_spec_directions(coil_spec, seed=1739),
    )
    monkeypatch.setattr(core_field, "is_mixed_precision_enabled", lambda: True)
    monkeypatch.setattr(
        core_field,
        "coil_group_collective_config",
        lambda currents: None,
    )

    direct_primal = _production_grouped_candidate(points, coil_spec)
    jvp_primal, _ = jax.jvp(
        _production_grouped_candidate,
        (points, coil_spec),
        directions,
    )

    assert np.asarray(jvp_primal).tobytes() == np.asarray(direct_primal).tobytes()


def test_production_grouped_dispatch_vjp_matches_fp64(monkeypatch):
    points, coil_spec = _production_dispatch_fixture()
    cotangent = jnp.asarray(
        np.random.default_rng(1727).normal(size=(points.shape[0], 3)),
        dtype=jnp.float64,
    )
    monkeypatch.setattr(core_field, "is_mixed_precision_enabled", lambda: True)
    monkeypatch.setattr(
        core_field,
        "coil_group_collective_config",
        lambda currents: None,
    )

    actual_value, actual_pullback = jax.vjp(
        _production_grouped_candidate,
        points,
        coil_spec,
    )
    expected_value, expected_pullback = jax.vjp(
        _production_grouped_reference,
        points,
        coil_spec,
    )
    actual_cotangents = actual_pullback(cotangent)
    expected_cotangents = expected_pullback(cotangent)

    _assert_close(actual_value, expected_value, rtol=4.0e-6, atol=3.0e-10)
    for actual, expected in zip(
        jax.tree.leaves(actual_cotangents),
        jax.tree.leaves(expected_cotangents),
        strict=True,
    ):
        _assert_close(actual, expected, rtol=3.0e-5, atol=3.0e-9)


def test_production_grouped_dispatch_reverse_over_jvp_matches_fp64(monkeypatch):
    points, coil_spec = _production_dispatch_fixture()
    directions = (
        jnp.asarray(
            np.random.default_rng(1728).normal(scale=0.03, size=points.shape),
            dtype=jnp.float32,
        ),
        _grouped_spec_directions(coil_spec, seed=1729),
    )
    cotangent = jnp.asarray(
        np.random.default_rng(1730).normal(size=(points.shape[0], 3)),
        dtype=jnp.float64,
    )
    monkeypatch.setattr(core_field, "is_mixed_precision_enabled", lambda: True)
    monkeypatch.setattr(
        core_field,
        "coil_group_collective_config",
        lambda currents: None,
    )

    def reverse_over_jvp(function):
        def contracted_directional(candidate_points, candidate_spec):
            directional = jax.jvp(
                function,
                (candidate_points, candidate_spec),
                directions,
            )[1]
            return jnp.vdot(cotangent, directional).real

        return jax.grad(contracted_directional, argnums=(0, 1))(points, coil_spec)

    actual = reverse_over_jvp(_production_grouped_candidate)
    expected = reverse_over_jvp(_production_grouped_reference)

    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual),
        jax.tree.leaves(expected),
        strict=True,
    ):
        _assert_close(actual_leaf, expected_leaf, rtol=2.0e-4, atol=8.0e-9)


def test_production_grouped_dispatch_uses_tuning_tile_and_transfer_guard(
    monkeypatch,
):
    points, coil_spec = _production_dispatch_fixture()
    observed_tile_sizes = []
    original_online = core_field.mixed_grouped_biot_savart_B_online
    monkeypatch.setattr(core_field, "is_mixed_precision_enabled", lambda: True)
    monkeypatch.setattr(
        core_field,
        "coil_group_collective_config",
        lambda currents: None,
    )
    monkeypatch.setattr(
        core_field,
        "get_field_kernel_tuning",
        lambda: SimpleNamespace(mixed_biot_savart_source_tile_size=64),
    )

    def observe_tile_size(
        kernel_points,
        groups,
        *,
        source_tile_size,
    ):
        observed_tile_sizes.append(source_tile_size)
        return original_online(
            kernel_points,
            groups,
            source_tile_size=source_tile_size,
        )

    monkeypatch.setattr(
        core_field,
        "mixed_grouped_biot_savart_B_online",
        observe_tile_size,
    )
    compiled = jax.jit(_production_grouped_candidate).lower(points, coil_spec).compile()

    with jax.transfer_guard("disallow"):
        result = compiled(points, coil_spec)
        jax.block_until_ready(result)

    assert observed_tile_sizes == [64]
    assert result.shape == (points.shape[0], 3)
    assert result.dtype == jnp.float64


def test_grouped_field_sharding_summary_reports_actual_online_dispatch(monkeypatch):
    points, coil_spec = _production_dispatch_fixture()
    sentinel = jnp.full((points.shape[0], 3), 17.0, dtype=jnp.float64)
    summary = {"field_collective": False, "dispatch": "online"}
    observed = []
    monkeypatch.setattr(core_field, "is_mixed_precision_enabled", lambda: True)
    monkeypatch.setattr(
        core_field,
        "get_sharding_tuning",
        lambda: SimpleNamespace(active=False, strategy="none"),
    )
    monkeypatch.setattr(
        core_field,
        "coil_group_collective_config",
        lambda currents: None,
    )
    monkeypatch.setattr(
        core_field,
        "_mixed_online_grouped_biot_savart_B",
        lambda kernel_points, coil_arrays: sentinel,
    )

    def summarize(result, *, config):
        observed.append((result, config))
        return summary

    monkeypatch.setattr(core_field, "collective_field_sharding_summary", summarize)

    actual = core_field.grouped_field_sharding_summary(points, coil_spec)

    assert actual is summary
    assert observed == [(sentinel, None)]


def _naive_sequential_tile_accumulation(points, source_positions, source_vectors):
    diff = source_positions[:, None, :] - points[None, :, :]
    radius_squared = jnp.sum(diff * diff, axis=-1)
    contributions = (
        jnp.cross(diff, source_vectors[:, None, :])
        / (radius_squared * jnp.sqrt(radius_squared))[..., None]
    )

    def body(total, contribution):
        return total + jnp.asarray(contribution, dtype=jnp.float64), None

    total, _ = jax.lax.scan(
        body,
        jnp.zeros_like(points, dtype=jnp.float64),
        contributions,
    )
    return jnp.asarray(_MU0_OVER_4PI) * total


def test_online_compensated_cross_tile_sum_recovers_cancellation_residual():
    points = jnp.zeros((1, 3), dtype=jnp.float32)
    source_positions = jnp.tile(
        jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        (4, 1),
    )
    source_vectors = jnp.asarray(
        [[0.0, 1.0e16, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0e16, 0.0], [0.0, 1.0, 0.0]],
        dtype=jnp.float32,
    )
    expected = np.asarray([[0.0, 0.0, 2.0e-7]], dtype=np.float64)

    compensated = _mixed_biot_savart_B_online_from_flat_sources(
        points,
        source_positions,
        source_vectors,
        source_tile_size=1,
    )
    naive = _naive_sequential_tile_accumulation(
        points,
        source_positions,
        source_vectors,
    )

    np.testing.assert_allclose(np.asarray(compensated), expected, rtol=0.0, atol=1e-22)
    assert np.max(np.abs(np.asarray(compensated) - expected)) < np.max(
        np.abs(np.asarray(naive) - expected)
    )


def _cancellation_jvp_fixture():
    points = jnp.zeros((1, 3), dtype=jnp.float32)
    source_positions = jnp.tile(
        jnp.asarray([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        (4, 1),
    )
    source_vectors = jnp.zeros((4, 3), dtype=jnp.float32)
    source_vector_direction = jnp.asarray(
        [[0.0, 1.0e16, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0e16, 0.0], [0.0, 1.0, 0.0]],
        dtype=jnp.float32,
    )
    primals = (points, source_positions, source_vectors)
    tangents = (
        jnp.zeros_like(points),
        jnp.zeros_like(source_positions),
        source_vector_direction,
    )
    return primals, tangents


def _cancellation_candidate(*args):
    return _mixed_biot_savart_B_online_from_flat_sources(
        *args,
        source_tile_size=1,
    )


def test_online_jvp_preserves_compensated_cancellation_residual():
    primals, tangents = _cancellation_jvp_fixture()

    _, tangent = jax.jvp(_cancellation_candidate, primals, tangents)

    expected = np.asarray([[0.0, 0.0, 2.0e-7]], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(tangent), expected, rtol=0.0, atol=1e-22)


def test_online_cancellation_jvp_remains_reverse_differentiable():
    primals, tangents = _cancellation_jvp_fixture()
    points, source_positions, source_vectors = primals
    point_direction, source_position_direction, source_vector_direction = tangents

    def contracted_directional(variable_source_positions):
        directional = jax.jvp(
            _cancellation_candidate,
            (points, variable_source_positions, source_vectors),
            (point_direction, source_position_direction, source_vector_direction),
        )[1]
        return directional[0, 2]

    reverse_over_jvp = jax.grad(contracted_directional)(source_positions)

    expected = np.asarray(
        [
            [-2.0e9, 0.0, 0.0],
            [-2.0e-7, 0.0, 0.0],
            [2.0e9, 0.0, 0.0],
            [-2.0e-7, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        np.asarray(reverse_over_jvp),
        expected,
        rtol=2e-7,
        atol=1e-13,
    )


def test_online_kernel_is_strict_transfer_guard_clean():
    primals = _fixture(point_count=2, source_count=9, seed=1711)
    compiled = jax.jit(_candidate).lower(*primals).compile()

    with jax.transfer_guard("disallow"):
        result = compiled(*primals)
        jax.block_until_ready(result)

    assert result.shape == (2, 3)
    assert result.dtype == jnp.float64


def test_online_stablehlo_casts_only_reduced_tile_partials_to_float64():
    points, source_positions, source_vectors = _fixture(
        point_count=3,
        source_count=8,
        seed=1712,
    )
    lowered = jax.jit(_candidate).lower(points, source_positions, source_vectors)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))

    assert "tensor<2x4x3xf32>" in stablehlo
    assert "tensor<3x4x3xf64>" not in stablehlo
    assert stablehlo.count("stablehlo.convert") == 2
    assert stablehlo.count("(tensor<3x3xf32>) -> tensor<3x3xf64>") == 2


def test_online_operator_is_mixed_only_and_does_not_replace_fp64_reference():
    primals = tuple(jnp.asarray(value, dtype=jnp.float64) for value in _fixture())
    with pytest.raises(TypeError, match="must have dtype float32"):
        _mixed_biot_savart_B_online_from_flat_sources(
            *primals,
            source_tile_size=4,
        )

    float32_primals = _fixture()
    with pytest.raises(ValueError, match="source_tile_size must be positive"):
        _mixed_biot_savart_B_online_from_flat_sources(
            *float32_primals,
            source_tile_size=0,
        )
