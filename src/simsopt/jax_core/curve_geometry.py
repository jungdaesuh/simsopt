"""Pure curve-geometry helpers that operate on immutable specs."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from ..geo.curve import gamma_curve_on_surface
from ..geo.curvehelical import curve_helical_pure
from ..geo.curveplanarfourier import curveplanarfourier_pure
from ..geo.curverzfourier import curverzfourier_pure
from ..geo.curvexyzfourier import (
    jaxfouriercurve_geometry_pure,
    jaxfouriercurve_pure,
)
from ._math_utils import (
    as_runtime_float64 as _as_runtime_float64,
)
from .framedcurve import (
    rotated_centroid_frame,
    rotated_centroid_frame_dash,
    rotated_frenet_frame,
    rotated_frenet_frame_dash,
    rotation_alpha as jaxrotation_pure,
    rotation_alphadash as jaxrotationdash_pure,
)
from .specs import (
    CurveCWSFourierRZSpec,
    CurveFilamentSpec,
    CurveHelicalSpec,
    CurvePlanarFourierSpec,
    CurvePerturbedSpec,
    CurveRZFourierSpec,
    CurveSpec,
    CurveXYZFourierSpec,
    CurveXYZFourierSymmetriesSpec,
    OptimizableDofMapSpec,
    RotationSpec,
    ZeroRotationSpec,
    curve_spec_kind,
    make_curve_cwsfourier_rz_spec,
)

_SURF_TYPE_RZ_FOURIER = "RZ_Fourier"


def _as_explicit_float64(value, *, reference=None) -> jax.Array:
    if reference is not None:
        return _as_runtime_float64(value, reference=reference)
    if isinstance(value, jax.Array) or hasattr(value, "aval"):
        return jnp.asarray(value, dtype=jnp.float64)
    if isinstance(value, (list, tuple)):
        leaves = jax.tree_util.tree_leaves(value)
        if any(isinstance(leaf, jax.Array) or hasattr(leaf, "aval") for leaf in leaves):
            return jnp.asarray(value, dtype=jnp.float64)
    raise TypeError(
        "curve_geometry pure helpers require JAX/spec-backed arrays; "
        "materialize an immutable spec or explicit device array first."
    )


def _explicit_scalar(value: float, *, reference=None) -> jax.Array:
    return _as_explicit_float64(value, reference=reference)


def _ones_like_float64(array: jax.Array) -> jax.Array:
    return jnp.broadcast_to(_explicit_scalar(1.0, reference=array), array.shape)


def _zeros_like_float64(array: jax.Array) -> jax.Array:
    return jnp.broadcast_to(_explicit_scalar(0.0, reference=array), array.shape)


def _element_count_float64(array: jax.Array) -> jax.Array:
    return _explicit_scalar(float(np.prod(array.shape)), reference=array)


def _slice_1d_static(array: jax.Array, start: int, end: int) -> jax.Array:
    positions = np.arange(int(start), int(end), dtype=np.int64)
    selector = np.zeros((positions.size, array.shape[0]), dtype=np.float64)
    selector[np.arange(positions.size), positions] = 1.0
    return _as_explicit_float64(selector, reference=array) @ array


def _update_1d_static(array: jax.Array, start: int, values: jax.Array) -> jax.Array:
    positions = np.arange(int(start), int(start) + values.shape[0], dtype=np.int64)
    insert = np.zeros((array.shape[0], positions.size), dtype=np.float64)
    insert[positions, np.arange(positions.size)] = 1.0
    keep_mask = np.ones(array.shape[0], dtype=np.float64)
    keep_mask[positions] = 0.0
    return array * _as_explicit_float64(keep_mask, reference=array) + (
        _as_explicit_float64(insert, reference=array) @ values
    )


def curve_spec_from_curve(curve):
    to_spec = getattr(curve, "to_spec", None)
    if callable(to_spec):
        return to_spec()

    surface = getattr(curve, "surf", None)
    if surface is None:
        raise NotImplementedError(
            f"Curve type {type(curve).__name__} does not expose an immutable JAX spec."
        )

    surface_spec_fn = getattr(surface, "surface_spec", None)
    if not callable(surface_spec_fn):
        raise NotImplementedError(
            f"Surface type {type(surface).__name__} does not expose surface_spec()."
        )

    if getattr(curve, "surf_type", None) != _SURF_TYPE_RZ_FOURIER:
        raise NotImplementedError(
            "CWS spec generation requires surf_type='RZ_Fourier', "
            f"got {getattr(curve, 'surf_type', None)!r}."
        )

    return make_curve_cwsfourier_rz_spec(
        dofs=curve.get_dofs(),
        quadpoints=curve.quadpoints,
        surface=surface_spec_fn(),
        order=curve.order,
        G=getattr(curve, "G", 0.0),
        H=getattr(curve, "H", 0.0),
    )


def _curve_gamma_kernel(spec: CurveSpec, dofs=None):
    curve_dofs = (
        spec.dofs if dofs is None else _as_explicit_float64(dofs, reference=spec.dofs)
    )
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "xyz_fourier":
        spec = cast(CurveXYZFourierSpec, spec)
        return lambda quadpoints: jaxfouriercurve_pure(
            curve_dofs,
            quadpoints,
            spec.order,
        )
    if spec_kind == "rz_fourier":
        spec = cast(CurveRZFourierSpec, spec)
        return lambda quadpoints: curverzfourier_pure(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.nfp,
            spec.stellsym,
        )
    if spec_kind == "planar_fourier":
        spec = cast(CurvePlanarFourierSpec, spec)
        return lambda quadpoints: curveplanarfourier_pure(
            curve_dofs,
            quadpoints,
            spec.order,
        )
    if spec_kind == "helical":
        spec = cast(CurveHelicalSpec, spec)
        return lambda quadpoints: curve_helical_pure(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.m,
            spec.ell,
            spec.R0,
            spec.r,
        )
    if spec_kind == "xyz_fourier_symmetries":
        from simsopt.geo.curvexyzfouriersymmetries import (
            jaxXYZFourierSymmetriescurve_pure,
        )

        spec = cast(CurveXYZFourierSymmetriesSpec, spec)
        return lambda quadpoints: jaxXYZFourierSymmetriescurve_pure(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.nfp,
            spec.stellsym,
            spec.ntor,
        )
    if spec_kind == "cws_fourier_rz":
        spec = cast(CurveCWSFourierRZSpec, spec)
        surface_dofs = spec.surface_dofs()
        return lambda quadpoints: gamma_curve_on_surface(
            curve_dofs,
            quadpoints,
            spec.order,
            spec.G,
            spec.H,
            surface_dofs,
            _SURF_TYPE_RZ_FOURIER,
            spec.surface.mpol,
            spec.surface.ntor,
            spec.surface.nfp,
            spec.surface.stellsym,
        )
    raise TypeError(
        "curve_gamma_kernel only supports direct curve specs, "
        f"got {type(spec).__name__}."
    )


def _curve_quadpoints(spec: CurveSpec, *, reference):
    quadpoints = _as_explicit_float64(spec.quadpoints, reference=reference)
    return quadpoints, _ones_like_float64(quadpoints)


def _curve_geometry_terms_from_kernel(gamma_kernel, quadpoints, tangents, *, order):
    gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
    if order == 1:
        return gamma, gammadash

    gammadash_kernel = lambda qp: jax.jvp(gamma_kernel, (qp,), (tangents,))[1]
    _, gammadashdash = jax.jvp(gammadash_kernel, (quadpoints,), (tangents,))
    if order == 2:
        return gamma, gammadash, gammadashdash

    gammadashdash_kernel = lambda qp: jax.jvp(
        gammadash_kernel,
        (qp,),
        (tangents,),
    )[1]
    _, gammadashdashdash = jax.jvp(
        gammadashdash_kernel,
        (quadpoints,),
        (tangents,),
    )
    return gamma, gammadash, gammadashdash, gammadashdashdash


def _direct_curve_geometry_terms(spec: CurveSpec, dofs, *, order):
    if curve_spec_kind(spec) != "xyz_fourier":
        return None
    spec = cast(CurveXYZFourierSpec, spec)
    curve_dofs = spec.dofs if dofs is None else dofs
    geometry = jaxfouriercurve_geometry_pure(
        curve_dofs,
        spec.quadpoints,
        spec.order,
    )
    return geometry[: order + 1]


def _mapped_full_dofs(map_spec: OptimizableDofMapSpec, owner_dofs):
    mapped = _as_explicit_float64(map_spec.template_full_dofs, reference=owner_dofs)
    owner_dofs = _as_explicit_float64(owner_dofs, reference=owner_dofs)
    for owner_start, owner_end, target_start, target_end in map_spec.owner_segments:
        del target_end
        segment = _slice_1d_static(owner_dofs, owner_start, owner_end)
        mapped = _update_1d_static(mapped, target_start, segment)
    return mapped


def _mapped_input_dofs(map_spec: OptimizableDofMapSpec, owner_dofs):
    mapped_full = _mapped_full_dofs(map_spec, owner_dofs)
    if map_spec.input_mode == "full":
        return mapped_full
    return _slice_1d_static(mapped_full, map_spec.input_start, map_spec.input_end)


def optimizable_full_dofs_from_map_spec(
    map_spec: OptimizableDofMapSpec,
    owner_dofs,
):
    return _mapped_full_dofs(map_spec, owner_dofs)


def optimizable_input_dofs_from_map_spec(
    map_spec: OptimizableDofMapSpec,
    owner_dofs,
):
    return _mapped_input_dofs(map_spec, owner_dofs)


def _rotation_alpha_and_dash_from_dofs(
    rotation_spec: RotationSpec,
    rotation_map: OptimizableDofMapSpec,
    owner_dofs,
):
    quadpoints = _as_explicit_float64(rotation_spec.quadpoints, reference=owner_dofs)
    if isinstance(rotation_spec, ZeroRotationSpec):
        zeros = _zeros_like_float64(quadpoints)
        return zeros, zeros

    rotation_dofs = _mapped_input_dofs(rotation_map, owner_dofs)
    rotation_scale = _explicit_scalar(rotation_spec.scale, reference=owner_dofs)
    return (
        rotation_scale
        * jaxrotation_pure(rotation_dofs, quadpoints, rotation_spec.order),
        rotation_scale
        * jaxrotationdash_pure(rotation_dofs, quadpoints, rotation_spec.order),
    )


def _curve_geometry_with_third_derivative_from_dofs(spec: CurveSpec, dofs):
    """Return (gamma, gammadash, gammadashdash, gammadashdashdash) in one pass."""
    if isinstance(spec, CurvePerturbedSpec):
        base_geometry = _curve_geometry_with_third_derivative_from_dofs(
            spec.base_curve,
            _curve_perturbed_base_dofs(spec, dofs),
        )
        return _add_curve_perturbation(spec, *base_geometry)
    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    direct_geometry = _direct_curve_geometry_terms(spec, dofs, order=3)
    if direct_geometry is not None:
        return direct_geometry
    gamma_kernel = _curve_gamma_kernel(spec, dofs)
    return _curve_geometry_terms_from_kernel(
        gamma_kernel,
        quadpoints,
        tangents,
        order=3,
    )


def _curve_perturbed_base_dofs(spec: CurvePerturbedSpec, dofs):
    return _mapped_input_dofs(spec.base_curve_map, dofs)


def _add_curve_perturbation(spec: CurvePerturbedSpec, *geometry_terms):
    sample_terms = (
        spec.sample_gamma,
        spec.sample_gammadash,
        spec.sample_gammadashdash,
        spec.sample_gammadashdashdash,
    )
    return tuple(
        geometry_term + sample_term
        for geometry_term, sample_term in zip(geometry_terms, sample_terms)
    )


def _curve_perturbed_gamma_and_dash_from_dofs(spec: CurvePerturbedSpec, dofs):
    base_geometry = curve_gamma_and_dash_from_dofs(
        spec.base_curve,
        _curve_perturbed_base_dofs(spec, dofs),
    )
    return _add_curve_perturbation(spec, *base_geometry)


def _curve_perturbed_geometry_from_dofs(spec: CurvePerturbedSpec, dofs):
    base_geometry = curve_geometry_from_dofs(
        spec.base_curve,
        _curve_perturbed_base_dofs(spec, dofs),
    )
    return _add_curve_perturbation(spec, *base_geometry)


def _curve_spec_with_quadpoints(spec: CurveSpec, quadpoints):
    quadpoints_jax = _as_explicit_float64(quadpoints, reference=spec.dofs)
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "perturbed":
        spec = cast(CurvePerturbedSpec, spec)
        return replace(
            spec,
            quadpoints=quadpoints_jax,
            base_curve=_curve_spec_with_quadpoints(spec.base_curve, quadpoints_jax),
        )
    if spec_kind == "filament":
        spec = cast(CurveFilamentSpec, spec)
        return replace(
            spec,
            quadpoints=quadpoints_jax,
            base_curve=_curve_spec_with_quadpoints(spec.base_curve, quadpoints_jax),
            rotation=replace(spec.rotation, quadpoints=quadpoints_jax),
        )
    return replace(spec, quadpoints=quadpoints_jax)


def _curve_filament_geometry_from_dofs(spec: CurveFilamentSpec, dofs):
    def gamma_kernel(qp):
        quad_spec = _curve_spec_with_quadpoints(spec, qp)
        base_dofs = _mapped_input_dofs(quad_spec.base_curve_map, dofs)
        alpha, _alphadash = _rotation_alpha_and_dash_from_dofs(
            quad_spec.rotation,
            quad_spec.rotation_map,
            dofs,
        )
        gamma, gammadash = curve_gamma_and_dash_from_dofs(
            quad_spec.base_curve, base_dofs
        )
        if quad_spec.frame_kind == "frenet":
            _gamma, _gammadash, gammadashdash = curve_geometry_from_dofs(
                quad_spec.base_curve,
                base_dofs,
            )
            _tangent, normal, binormal = rotated_frenet_frame(
                gamma,
                gammadash,
                gammadashdash,
                alpha,
            )
        else:
            _tangent, normal, binormal = rotated_centroid_frame(
                gamma,
                gammadash,
                alpha,
            )
        return gamma + quad_spec.dn * normal + quad_spec.db * binormal

    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
    gammadash_kernel = lambda qp: jax.jvp(gamma_kernel, (qp,), (tangents,))[1]
    _, gammadashdash = jax.jvp(gammadash_kernel, (quadpoints,), (tangents,))
    return gamma, gammadash, gammadashdash


def _curve_filament_gamma_and_dash_from_dofs(spec: CurveFilamentSpec, dofs):
    base_dofs = _mapped_input_dofs(spec.base_curve_map, dofs)
    alpha, alphadash = _rotation_alpha_and_dash_from_dofs(
        spec.rotation,
        spec.rotation_map,
        dofs,
    )

    if spec.frame_kind == "frenet":
        gamma, gammadash, gammadashdash, gammadashdashdash = (
            _curve_geometry_with_third_derivative_from_dofs(spec.base_curve, base_dofs)
        )
        _tangent, normal, binormal = rotated_frenet_frame(
            gamma,
            gammadash,
            gammadashdash,
            alpha,
        )
        tangent_dash, normal_dash, binormal_dash = rotated_frenet_frame_dash(
            gamma,
            gammadash,
            gammadashdash,
            gammadashdashdash,
            alpha,
            alphadash,
        )
    else:
        gamma, gammadash, gammadashdash = curve_geometry_from_dofs(
            spec.base_curve, base_dofs
        )
        _tangent, normal, binormal = rotated_centroid_frame(
            gamma,
            gammadash,
            alpha,
        )
        _tangent_dash, normal_dash, binormal_dash = rotated_centroid_frame_dash(
            gamma,
            gammadash,
            gammadashdash,
            alpha,
            alphadash,
        )
    return (
        gamma + spec.dn * normal + spec.db * binormal,
        gammadash + spec.dn * normal_dash + spec.db * binormal_dash,
    )


def curve_spec_with_dofs(spec: CurveSpec, dofs):
    return replace(spec, dofs=_as_runtime_float64(dofs, reference=spec.dofs))


def curve_spec_with_quadpoints(spec: CurveSpec, quadpoints):
    return _curve_spec_with_quadpoints(
        spec,
        _as_explicit_float64(quadpoints, reference=spec.dofs),
    )


def _clamp_unit_interval(value: jax.Array) -> jax.Array:
    return jnp.clip(
        value,
        _explicit_scalar(0.0, reference=value),
        _explicit_scalar(1.0, reference=value),
    )


def _distance_sq(vector: jax.Array) -> jax.Array:
    return jnp.maximum(jnp.dot(vector, vector), _explicit_scalar(0.0, reference=vector))


def _distance(vector: jax.Array) -> jax.Array:
    return jnp.sqrt(_distance_sq(vector))


def segment_segment_distance_pure(
    segment_start: jax.Array,
    segment_end: jax.Array,
    other_start: jax.Array,
    other_end: jax.Array,
) -> jax.Array:
    """Return the minimum distance between two 3D line segments.

    This is the JAX-native Sunday/Lumelsky-style kernel used for curve
    self-intersection checks and penalties. The branching stays entirely in JAX
    control flow so the result is differentiable with respect to the input
    segment endpoints.

    Branch map: first-segment point case, second-segment point case, then
    nondegenerate segment-pair handling. The nondegenerate path splits into
    near-parallel endpoint projections or the standard closest-point clamp.
    """
    segment_start = _as_explicit_float64(segment_start)
    segment_end = _as_explicit_float64(segment_end, reference=segment_start)
    other_start = _as_explicit_float64(other_start, reference=segment_start)
    other_end = _as_explicit_float64(other_end, reference=segment_start)
    zero_len = _explicit_scalar(1e-30, reference=segment_start)
    parallel_eps = _explicit_scalar(1e-10, reference=segment_start)
    zero = _explicit_scalar(0.0, reference=segment_start)
    one = _explicit_scalar(1.0, reference=segment_start)

    u = segment_end - segment_start
    v = other_end - other_start
    w0 = segment_start - other_start

    a = jnp.dot(u, u)
    b = jnp.dot(u, v)
    c = jnp.dot(v, v)
    d = jnp.dot(u, w0)
    e = jnp.dot(v, w0)

    def _both_degenerate(_):
        return _distance(w0)

    def _segment_is_point(_):
        def _other_is_point(__):
            return _both_degenerate(None)

        def _project_to_other(__):
            projected = w0 - _clamp_unit_interval(e / c) * v
            return _distance(projected)

        return jax.lax.cond(c < zero_len, _other_is_point, _project_to_other, None)

    def _other_is_point(_):
        projected = w0 + _clamp_unit_interval(-d / a) * u
        return _distance(projected)

    def _general_case(_):
        denom = a * c - b * b

        def _near_parallel(__):
            best_sq = _distance_sq(w0 - _clamp_unit_interval(e / c) * v)
            best_sq = jnp.minimum(
                best_sq,
                _distance_sq(w0 + u - _clamp_unit_interval((e + b) / c) * v),
            )
            best_sq = jnp.minimum(
                best_sq,
                _distance_sq(w0 + _clamp_unit_interval(-d / a) * u),
            )
            best_sq = jnp.minimum(
                best_sq,
                _distance_sq(w0 + _clamp_unit_interval((b - d) / a) * u - v),
            )

            def _interior_sq(_operand):
                sc_int = (b * e - c * d) / denom
                tc_int = (a * e - b * d) / denom
                interior_valid = (
                    (sc_int >= zero)
                    & (sc_int <= one)
                    & (tc_int >= zero)
                    & (tc_int <= one)
                )
                candidate_sq = _distance_sq(w0 + sc_int * u - tc_int * v)
                inf_sq = _explicit_scalar(jnp.inf, reference=segment_start)
                return jnp.where(interior_valid, candidate_sq, inf_sq)

            interior_sq = jax.lax.cond(
                denom > zero,
                _interior_sq,
                lambda _: _explicit_scalar(jnp.inf, reference=segment_start),
                None,
            )
            return jnp.sqrt(jnp.minimum(best_sq, interior_sq))

        def _non_parallel(__):
            sc = (b * e - c * d) / denom
            tc = (a * e - b * d) / denom

            sc = jnp.where(sc < zero, zero, sc)
            tc = jnp.where(sc == zero, e / c, tc)

            sc = jnp.where(sc > one, one, sc)
            tc = jnp.where(sc == one, (e + b) / c, tc)

            tc = jnp.where(tc < zero, zero, tc)
            sc = jnp.where(tc == zero, _clamp_unit_interval(-d / a), sc)

            tc = jnp.where(tc > one, one, tc)
            sc = jnp.where(tc == one, _clamp_unit_interval((b - d) / a), sc)

            return _distance(w0 + sc * u - tc * v)

        return jax.lax.cond(
            denom < parallel_eps * a * c,
            _near_parallel,
            _non_parallel,
            None,
        )

    return jax.lax.cond(
        a < zero_len,
        _segment_is_point,
        lambda _: jax.lax.cond(c < zero_len, _other_is_point, _general_case, None),
        None,
    )


def _closed_curve_segment_arrays(gamma: jax.Array) -> tuple[jax.Array, jax.Array]:
    gamma = _as_explicit_float64(gamma)
    return gamma, jnp.roll(gamma, shift=-1, axis=0)


def closed_curve_self_intersection_min_distance(
    gamma: jax.Array,
    *,
    neighbor_skip: int = 3,
) -> jax.Array:
    """Return the minimum non-neighbor segment distance of a closed curve."""
    segment_start, segment_end = _closed_curve_segment_arrays(gamma)
    segment_count = segment_start.shape[0]
    inf_distance = _explicit_scalar(jnp.inf, reference=segment_start)
    if int(segment_count) <= (2 * int(neighbor_skip) + 1):
        return inf_distance

    segment_indices = jnp.arange(segment_count, dtype=jnp.int32)

    def _row_minimum(best_distance, row_inputs):
        left_index, left_start, left_end = row_inputs

        def _pair_distance(right_index, right_start, right_end):
            distance = segment_segment_distance_pure(
                left_start,
                left_end,
                right_start,
                right_end,
            )
            delta = jnp.abs(left_index - right_index)
            wrapped_delta = jnp.minimum(delta, segment_count - delta)
            return jnp.where(wrapped_delta > neighbor_skip, distance, inf_distance)

        row_distances = jax.vmap(_pair_distance)(
            segment_indices,
            segment_start,
            segment_end,
        )
        return jnp.minimum(best_distance, jnp.min(row_distances)), None

    minimum_distance, _ = jax.lax.scan(
        _row_minimum,
        inf_distance,
        (segment_indices, segment_start, segment_end),
    )
    return minimum_distance


def closed_curve_self_intersection_tolerance(
    gamma: jax.Array,
    *,
    tolerance_factor: float = 0.1,
) -> jax.Array:
    """Return the segment-length-scaled self-intersection tolerance."""
    segment_start, segment_end = _closed_curve_segment_arrays(gamma)
    segment_lengths = jnp.linalg.norm(segment_end - segment_start, axis=1)
    average_segment_length = jnp.sum(segment_lengths) / _element_count_float64(
        segment_lengths
    )
    return _explicit_scalar(tolerance_factor, reference=gamma) * average_segment_length


def _closed_curve_self_intersection_terms(
    gamma: jax.Array,
    *,
    tolerance_factor: float = 0.1,
    neighbor_skip: int = 3,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    minimum_distance = closed_curve_self_intersection_min_distance(
        gamma,
        neighbor_skip=neighbor_skip,
    )
    tolerance = closed_curve_self_intersection_tolerance(
        gamma,
        tolerance_factor=tolerance_factor,
    )
    deficit = jnp.maximum(
        tolerance - minimum_distance,
        _explicit_scalar(0.0, reference=gamma),
    )
    return minimum_distance, tolerance, deficit


def closed_curve_self_intersection_penalty(
    gamma: jax.Array,
    *,
    tolerance_factor: float = 0.1,
    neighbor_skip: int = 3,
) -> jax.Array:
    """Return a soft quadratic penalty for closed-curve self intersection."""
    _minimum_distance, _tolerance, deficit = _closed_curve_self_intersection_terms(
        gamma,
        tolerance_factor=tolerance_factor,
        neighbor_skip=neighbor_skip,
    )
    return _explicit_scalar(0.5, reference=gamma) * deficit * deficit


def closed_curve_self_intersection_summary(
    gamma: jax.Array,
    *,
    tolerance_factor: float = 0.1,
    neighbor_skip: int = 3,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return (min_distance, tolerance, penalty, intersecting) for a closed curve."""
    minimum_distance, tolerance, deficit = _closed_curve_self_intersection_terms(
        gamma,
        tolerance_factor=tolerance_factor,
        neighbor_skip=neighbor_skip,
    )
    penalty = _explicit_scalar(0.5, reference=gamma) * deficit * deficit
    return minimum_distance, tolerance, penalty, minimum_distance < tolerance


@jax.jit
def pair_linking_number_pure(
    gamma1: jax.Array,
    gammadash1: jax.Array,
    gamma2: jax.Array,
    gammadash2: jax.Array,
    dphi1: jax.Array,
    dphi2: jax.Array,
) -> jax.Array:
    """Gauss linking-number contribution for a single ordered curve pair.

    Mirrors the C++ ``compute_linking_number`` inner pair contribution at
    ``src/simsoptpp/python_distance.cpp:181-211``. Inputs are the dense
    quadrature samples (``gamma``/``gammadash``) and the per-curve
    quadrature step (``dphi``). The returned value is the rounded
    absolute Gauss integral divided by ``4 * pi`` and is a non-negative
    integer JAX scalar.

    The kernel is pure and stateless; the host-side aggregator in
    ``LinkingNumber.J`` iterates curve pairs and sums the integer
    contributions.
    """
    gamma1 = _as_explicit_float64(gamma1)
    gammadash1 = _as_explicit_float64(gammadash1, reference=gamma1)
    gamma2 = _as_explicit_float64(gamma2, reference=gamma1)
    gammadash2 = _as_explicit_float64(gammadash2, reference=gamma1)
    dphi1 = _as_explicit_float64(dphi1, reference=gamma1)
    dphi2 = _as_explicit_float64(dphi2, reference=gamma1)
    difference = gamma1[:, None, :] - gamma2[None, :, :]
    dr = jnp.linalg.norm(difference, axis=-1)
    cross = jnp.cross(gammadash2[None, :, :], difference, axis=-1)
    det = jnp.sum(gammadash1[:, None, :] * cross, axis=-1)
    inv_dr3 = jnp.where(dr > 0, dr ** (-3), _explicit_scalar(0.0, reference=gamma1))
    total = jnp.sum(det * inv_dr3)
    four_pi = _explicit_scalar(4.0 * float(np.pi), reference=gamma1)
    value = jnp.round(jnp.abs(total * dphi1 * dphi2) / four_pi)
    return value.astype(jnp.int32)


def curve_gamma_and_dash_from_spec(spec: CurveSpec):
    return curve_gamma_and_dash_from_dofs(spec, spec.dofs)


def curve_gamma_and_dash_from_dofs(spec: CurveSpec, dofs):
    """Return (gamma, gammadash) from a single kernel build and JVP call."""
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "perturbed":
        spec = cast(CurvePerturbedSpec, spec)
        return _curve_perturbed_gamma_and_dash_from_dofs(spec, dofs)
    if spec_kind == "filament":
        spec = cast(CurveFilamentSpec, spec)
        return _curve_filament_gamma_and_dash_from_dofs(spec, dofs)
    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    direct_geometry = _direct_curve_geometry_terms(spec, dofs, order=1)
    if direct_geometry is not None:
        return direct_geometry
    gamma_kernel = _curve_gamma_kernel(spec, dofs)
    return _curve_geometry_terms_from_kernel(
        gamma_kernel,
        quadpoints,
        tangents,
        order=1,
    )


def curve_geometry_from_spec(spec: CurveSpec):
    return curve_geometry_from_dofs(spec, spec.dofs)


def curve_geometry_from_dofs(spec: CurveSpec, dofs):
    """Return (gamma, gammadash, gammadashdash) from a single kernel build."""
    spec_kind = curve_spec_kind(spec)
    if spec_kind == "perturbed":
        spec = cast(CurvePerturbedSpec, spec)
        return _curve_perturbed_geometry_from_dofs(spec, dofs)
    if spec_kind == "filament":
        spec = cast(CurveFilamentSpec, spec)
        return _curve_filament_geometry_from_dofs(spec, dofs)
    quadpoints, tangents = _curve_quadpoints(spec, reference=dofs)
    direct_geometry = _direct_curve_geometry_terms(spec, dofs, order=2)
    if direct_geometry is not None:
        return direct_geometry
    gamma_kernel = _curve_gamma_kernel(spec, dofs)
    return _curve_geometry_terms_from_kernel(
        gamma_kernel,
        quadpoints,
        tangents,
        order=2,
    )


def _curve_cws_gamma_and_dash_from_parts(
    spec: CurveCWSFourierRZSpec,
    curve_dofs,
    surface_dofs,
):
    quadpoints, tangents = _curve_quadpoints(spec, reference=curve_dofs)

    def gamma_kernel(qp):
        return gamma_curve_on_surface(
            curve_dofs,
            qp,
            spec.order,
            spec.G,
            spec.H,
            surface_dofs,
            _SURF_TYPE_RZ_FOURIER,
            spec.surface.mpol,
            spec.surface.ntor,
            spec.surface.nfp,
            spec.surface.stellsym,
        )

    return jax.jvp(gamma_kernel, (quadpoints,), (tangents,))


def curve_pullback_from_spec(spec: CurveSpec, dg, dgd):
    return curve_pullback_from_dofs(spec, spec.dofs, dg, dgd)


def curve_pullback_from_dofs(spec: CurveSpec, dofs, dg, dgd):
    """Return coefficient and optional surface cotangents for one curve spec."""
    curve_dofs = _as_runtime_float64(dofs, reference=spec.dofs)
    dg_jax = _as_runtime_float64(dg, reference=curve_dofs)
    dgd_jax = _as_runtime_float64(dgd, reference=curve_dofs)

    if curve_spec_kind(spec) == "cws_fourier_rz":
        spec = cast(CurveCWSFourierRZSpec, spec)
        surface_dofs = spec.surface_dofs()

        def outputs(curve_x, surface_x):
            return _curve_cws_gamma_and_dash_from_parts(spec, curve_x, surface_x)

        _, pullback = jax.vjp(outputs, curve_dofs, surface_dofs)
        coeff_cotangent, surface_cotangent = pullback((dg_jax, dgd_jax))
        return coeff_cotangent, surface_cotangent

    def outputs(curve_x):
        return curve_gamma_and_dash_from_dofs(spec, curve_x)

    _, pullback = jax.vjp(outputs, curve_dofs)
    (coeff_cotangent,) = pullback((dg_jax, dgd_jax))
    return coeff_cotangent, None
