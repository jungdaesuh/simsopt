"""JAX-side post-processing of a CPU ``BoozerMagneticField`` onto a
piecewise-polynomial interpolant in :math:`(s, \\theta, \\zeta)`.

Item N02 of the JAX port. Mirrors the C++ template class
``InterpolatedBoozerField`` declared at
``src/simsoptpp/boozermagneticfield_interpolated.h``: each Boozer scalar
field is fit on a regular grid using uniform-Lagrange tensor-product
basis, with the same flux-function / symmetry-exploiting split as the
C++ kernel.

This module deliberately does NOT consume the C++
``InterpolatedBoozerField`` object — the pybind11 surface only exposes
the evaluator methods, not the internal coefficient tensors. Instead we
freshly re-fit on a host-resident grid using
``simsopt.jax_core.regular_grid_interp.build_regular_grid_interpolant_3d``,
sampling the base field's value getters as the user-supplied callback.
The wrapper this module backs (``InterpolatedBoozerFieldJAX``) therefore
never inherits from ``sopp.BoozerMagneticField`` and never touches the
C++ ``InterpolatedBoozerField`` class.

Symmetry/coordinate-fold semantics mirror C++ ``exploit_symmetries_points``
and ``exploit_fluxfunction_points`` (header lines 724-784) plus the
``apply_odd_symmetry`` / ``apply_even_symmetry`` rules (lines 786-807):

- flux-function fields (``psip``, ``G``, ``I``, ``iota``, ``dGds``,
  ``dIds``, ``diotads``): theta/zeta are zeroed before evaluation
  because the field is constant in those angles.
- symmetry-exploit fields: theta is folded modularly into
  :math:`[0, 2\\pi]` and zeta into :math:`[0, 2\\pi/\\mathrm{nfp}]`. If
  ``stellsym`` is True and folded ``theta > pi``, additional reflection
  ``theta := 2*pi - theta``, ``zeta := period - zeta`` is applied and
  the per-sample flip flag is recorded.
- ``apply_odd_symmetry``: negate the scalar result for flipped samples.
  For 3-vector outputs (``Z_derivs``, ``nu_derivs``), only the first
  component is negated, matching C++ lines 786-797.
- ``apply_even_symmetry``: negate components ``[1]`` and ``[2]`` of
  3-vector outputs for flipped samples (C++ lines 799-807). Applies to
  ``R_derivs`` and ``modB_derivs``.

The ``InterpolatedBoozerFieldFrozenState`` dataclass is a plain
immutable container of 33 ``RegularGridInterpolant3DSpec`` instances
plus the small set of meta-fields (nfp, stellsym, ranges, extrapolate)
needed to drive the coordinate fold. It is NOT registered as a JAX
pytree because each contained spec already holds NumPy arrays — the
JIT seam is downstream at ``regular_grid_interp.evaluate_batch``.

References:

- ``src/simsoptpp/boozermagneticfield_interpolated.h``
- ``src/simsopt/jax_core/regular_grid_interp.py``
- ``src/simsopt/jax_core/boozer_radial_interp.py`` (item-32 precedent for
  the frozen-state pattern)
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from .regular_grid_interp import (
    RegularGridInterpolant3DSpec,
    UniformInterpolationRule,
    build_regular_grid_interpolant_3d,
    evaluate_batch,
)


__all__ = [
    "ALL_SCALARS",
    "FLUX_FUNCTION_SCALARS",
    "InterpolatedBoozerFieldFrozenState",
    "SYMMETRY_EXPLOIT_SCALARS",
    "build_flux_function_interpolant",
    "build_symmetry_exploit_interpolant",
    "build_spec_for_scalar",
    "freeze_interpolated_boozer_field_state",
    "evaluate_scalar",
    "fold_points_for_symmetry",
]


# ---------------------------------------------------------------------------
# Field inventory (mirrors C++ header)
# ---------------------------------------------------------------------------


# 7 flux-function scalars use the ``(0, M_PI, 1)`` "angle0" range and the
# C++ ``exploit_fluxfunction_points`` zero-fold (header lines 724-734).
FLUX_FUNCTION_SCALARS: tuple[str, ...] = (
    "psip",
    "G",
    "I",
    "iota",
    "dGds",
    "dIds",
    "diotads",
)


# 27 symmetry-exploit scalars use the user-supplied ``theta_range`` and
# ``zeta_range`` plus the ``exploit_symmetries_points`` coordinate fold
# (header lines 736-784). For each entry we record (value_size,
# scalar_odd, vector_odd_first_component_only, even_components_negate).
# These flags mirror the C++ `apply_odd_symmetry` / `apply_even_symmetry`
# rules at lines 786-807.
@dataclass(frozen=True)
class _SymmetryRule:
    value_size: int
    apply_odd: bool  # negate the (single) scalar for flipped samples
    apply_odd_vector_first_only: bool  # negate component 0 only on 3-vec
    apply_even: bool  # negate components 1,2 of 3-vec for flipped samples


SYMMETRY_EXPLOIT_SCALARS: dict[str, _SymmetryRule] = {
    # modB family
    "modB": _SymmetryRule(1, False, False, False),
    "dmodBdtheta": _SymmetryRule(1, True, False, False),
    "dmodBdzeta": _SymmetryRule(1, True, False, False),
    "dmodBds": _SymmetryRule(1, False, False, False),
    "modB_derivs": _SymmetryRule(3, False, False, True),
    "d2modBdtheta2": _SymmetryRule(1, False, False, False),
    "d2modBdzeta2": _SymmetryRule(1, False, False, False),
    "d2modBdthetadzeta": _SymmetryRule(1, False, False, False),
    # K family
    "K": _SymmetryRule(1, True, False, False),
    "dKdtheta": _SymmetryRule(1, False, False, False),
    "dKdzeta": _SymmetryRule(1, False, False, False),
    "K_derivs": _SymmetryRule(2, False, False, False),
    # nu family
    "nu": _SymmetryRule(1, True, False, False),
    "dnudtheta": _SymmetryRule(1, False, False, False),
    "dnudzeta": _SymmetryRule(1, False, False, False),
    "dnuds": _SymmetryRule(1, True, False, False),
    "nu_derivs": _SymmetryRule(3, False, True, False),
    # R family
    "R": _SymmetryRule(1, False, False, False),
    "dRdtheta": _SymmetryRule(1, True, False, False),
    "dRdzeta": _SymmetryRule(1, True, False, False),
    "dRds": _SymmetryRule(1, False, False, False),
    "R_derivs": _SymmetryRule(3, False, False, True),
    # Z family
    "Z": _SymmetryRule(1, True, False, False),
    "dZdtheta": _SymmetryRule(1, False, False, False),
    "dZdzeta": _SymmetryRule(1, False, False, False),
    "dZds": _SymmetryRule(1, True, False, False),
    "Z_derivs": _SymmetryRule(3, False, True, False),
}


# ---------------------------------------------------------------------------
# Frozen-state pytree
# ---------------------------------------------------------------------------


ALL_SCALARS: tuple[str, ...] = (
    *FLUX_FUNCTION_SCALARS,
    *tuple(SYMMETRY_EXPLOIT_SCALARS),
)


@dataclass(frozen=True)
class InterpolatedBoozerFieldFrozenState:
    """Immutable grid geometry and initially built scalar specs.

    Lazy-built scalars are owned by :class:`InterpolatedBoozerFieldJAX`
    in its ``_lazy_specs`` dict. This state records only the specs
    constructed by :func:`freeze_interpolated_boozer_field_state` so the
    public frozen-state API remains immutable and round-trippable.

    Meta-fields ``nfp``, ``stellsym``, ``period`` and ``extrapolate`` are
    needed at evaluation time to drive the coordinate fold mirroring
    C++ ``exploit_symmetries_points``. ``s_range``, ``theta_range``,
    ``zeta_range`` and ``degree`` are recorded for introspection and to
    let downstream code lazily build additional specs against the same
    grid.

    This dataclass is NOT registered as a JAX pytree: each spec passed
    alongside it already holds raw NumPy arrays that ``evaluate_batch``
    converts to ``jax.Array`` inside its JIT boundary. The wrapper
    class owns Optimizable state and is not differentiated through.
    """

    specs: Mapping[str, RegularGridInterpolant3DSpec]
    nfp: int
    stellsym: bool
    extrapolate: bool
    period: float  # = 2*pi / nfp
    s_range: tuple
    theta_range: tuple
    zeta_range: tuple
    degree: int

    def has(self, scalar_name: str) -> bool:
        """Return ``True`` iff ``scalar_name`` was built into this state."""
        return scalar_name in self.specs

    def get(self, scalar_name: str) -> RegularGridInterpolant3DSpec:
        """Return the built spec for ``scalar_name`` or raise ``KeyError``."""
        spec = self.specs.get(scalar_name)
        if spec is None:
            raise KeyError(
                f"interpolant for scalar {scalar_name!r} has not been built; "
                f"available: {sorted(self.specs)}"
            )
        return spec


# ---------------------------------------------------------------------------
# Coordinate folding (mirrors C++ exploit_symmetries_points)
# ---------------------------------------------------------------------------


_TWO_PI = 2.0 * jnp.pi


def fold_points_for_symmetry(
    points: jax.Array,
    *,
    period: jax.Array,
    stellsym: bool,
) -> tuple[jax.Array, jax.Array]:
    """Fold ``(s, theta, zeta)`` points using the C++ symmetry rule.

    Mirrors ``InterpolatedBoozerField::exploit_symmetries_points`` at
    header lines 736-784. Returns the folded points and a boolean
    ``(N,)`` mask flagging samples that were stellsym-reflected.

    Theta is restricted to ``[0, 2*pi]`` via modular arithmetic, then
    if stellsym and folded theta > pi the reflection
    ``theta := 2*pi - theta``, ``zeta := period - zeta`` is applied and
    the sample is flagged.

    Zeta is restricted to ``[0, period]`` first (period = 2*pi/nfp).

    Args:
        points: ``(N, 3)`` float64 array of (s, theta, zeta).
        period: scalar period ``2*pi/nfp`` for zeta wrapping.
        stellsym: whether to apply the stellsym reflection.

    Returns:
        ``(points_folded, flipped)`` where ``points_folded`` has the
        same shape as ``points`` and ``flipped`` has shape ``(N,)``.
    """
    s = points[:, 0]
    theta = points[:, 1]
    zeta = points[:, 2]

    # theta_mult = int(theta / (2*pi)); theta -= theta_mult * 2*pi
    # Uses truncation-toward-zero like C++ ``int()`` cast.
    theta_mult = jnp.trunc(theta / _TWO_PI)
    theta = theta - theta_mult * _TWO_PI
    # If theta < 0 add 2*pi, if theta > 2*pi subtract 2*pi.
    theta = jnp.where(theta < 0.0, theta + _TWO_PI, theta)
    theta = jnp.where(theta > _TWO_PI, theta - _TWO_PI, theta)

    zeta_mult = jnp.trunc(zeta / period)
    zeta = zeta - zeta_mult * period
    zeta = jnp.where(zeta < 0.0, zeta + period, zeta)
    zeta = jnp.where(zeta > period, zeta - period, zeta)

    if stellsym:
        flipped = theta > jnp.pi
        # Order matters: write zeta first using current (post-fold) value,
        # then write theta — matching C++ lines 769-779.
        zeta = jnp.where(flipped, period - zeta, zeta)
        theta = jnp.where(flipped, _TWO_PI - theta, theta)
    else:
        flipped = jnp.zeros_like(theta, dtype=bool)

    return jnp.stack([s, theta, zeta], axis=1), flipped


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _validate_range(label: str, axis_range) -> tuple[float, float, int]:
    if len(axis_range) != 3:
        raise ValueError(
            f"{label} range must be a 3-tuple (min, max, n_cells); got {axis_range!r}"
        )
    rmin = float(axis_range[0])
    rmax = float(axis_range[1])
    ncells = int(axis_range[2])
    if not rmax > rmin:
        raise ValueError(f"{label} range max ({rmax}) must exceed min ({rmin})")
    if ncells < 1:
        raise ValueError(f"{label} range cell count must be >= 1, got {ncells}")
    return rmin, rmax, ncells


def _make_callback_for_scalar(field, scalar_name: str, value_size: int):
    """Return ``f(xs, ys, zs) -> flat ndarray`` matching the C++ contract.

    The callback sets ``(s, theta, zeta)`` on the base field and reads
    the requested scalar back, flattening into row-major ``(N*value_size,)``
    layout — exactly the contract enforced by
    ``build_regular_grid_interpolant_3d``.

    The flux-function call site at C++ ``fbatch_scalar`` lines 814-818
    zeroes ``theta``/``zeta`` when the requested scalar is
    flux-function; we delegate that zeroing to the caller via the
    ``angle0_range=(0, pi, 1)`` build, which sends DOFs only along the
    s-axis. The Python ``set_points`` happens to receive the literal
    DOF coordinates the build pass produces.
    """
    getter = getattr(field, scalar_name)

    def f(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
        npoints = int(xs.shape[0])
        pts = np.empty((npoints, 3), dtype=np.float64)
        pts[:, 0] = xs
        pts[:, 1] = ys
        pts[:, 2] = zs
        field.set_points(pts)
        raw = np.asarray(getter(), dtype=np.float64)
        if raw.shape != (npoints, value_size):
            raise ValueError(
                f"scalar {scalar_name!r} on {type(field).__name__} returned "
                f"shape {raw.shape}; expected ({npoints}, {value_size})"
            )
        return raw.reshape(npoints * value_size)

    return f


def build_flux_function_interpolant(
    field,
    *,
    scalar_name: str,
    rule,
    s_range,
    extrapolate: bool,
) -> RegularGridInterpolant3DSpec:
    """Fit a single flux-function scalar on the C++ ``angle0_range`` grid.

    C++ reference: ``_psip_impl`` / ``_G_impl`` / ``_I_impl`` /
    ``_iota_impl`` / ``_dGds_impl`` / ``_dIds_impl`` / ``_diotads_impl``
    at header lines 39-170. Each uses the ``angle0_range = {0, M_PI, 1}``
    triplet (header line 902) for both theta and zeta axes, because the
    flux-function value is constant in those angles.

    Args:
        field: base ``BoozerMagneticField`` with the corresponding
            scalar getter method (``field.psip()`` etc.).
        scalar_name: which scalar to fit (e.g. ``"psip"``, ``"G"``).
        rule: ``UniformInterpolationRule(degree)``.
        s_range: ``(smin, smax, ns)`` triplet.
        extrapolate: forwarded to ``build_regular_grid_interpolant_3d``.
    """
    if scalar_name not in FLUX_FUNCTION_SCALARS:
        raise ValueError(
            f"unknown flux-function scalar {scalar_name!r}; "
            f"expected one of {FLUX_FUNCTION_SCALARS}"
        )
    angle0_range = (0.0, float(np.pi), 1)
    return build_regular_grid_interpolant_3d(
        rule=rule,
        xrange=s_range,
        yrange=angle0_range,
        zrange=angle0_range,
        value_size=1,
        f=_make_callback_for_scalar(field, scalar_name, value_size=1),
        out_of_bounds_ok=bool(extrapolate),
    )


def build_symmetry_exploit_interpolant(
    field,
    *,
    scalar_name: str,
    rule,
    s_range,
    theta_range,
    zeta_range,
    extrapolate: bool,
) -> RegularGridInterpolant3DSpec:
    """Fit a single symmetry-exploit scalar on the user-supplied 3D grid.

    C++ reference: ``_modB_impl`` / ``_K_impl`` / ... / ``_d2modBdzeta2_impl``
    at header lines 172-722. Each uses the user-provided
    ``s_range``, ``theta_range``, ``zeta_range`` triplets.

    The C++ class constructs every symmetry-exploit interpolant against
    the same ``(s_range, theta_range, zeta_range)`` triplet — this
    function preserves that contract.
    """
    if scalar_name not in SYMMETRY_EXPLOIT_SCALARS:
        raise ValueError(
            f"unknown symmetry-exploit scalar {scalar_name!r}; "
            f"expected one of {sorted(SYMMETRY_EXPLOIT_SCALARS)}"
        )
    value_size = SYMMETRY_EXPLOIT_SCALARS[scalar_name].value_size
    return build_regular_grid_interpolant_3d(
        rule=rule,
        xrange=s_range,
        yrange=theta_range,
        zrange=zeta_range,
        value_size=value_size,
        f=_make_callback_for_scalar(field, scalar_name, value_size=value_size),
        out_of_bounds_ok=bool(extrapolate),
    )


def build_spec_for_scalar(
    field,
    *,
    scalar_name: str,
    rule,
    s_range,
    theta_range,
    zeta_range,
    extrapolate: bool,
) -> RegularGridInterpolant3DSpec:
    """Build one spec for ``scalar_name`` against the supplied ranges.

    Dispatches to :func:`build_flux_function_interpolant` for the seven
    flux-function scalars (theta/zeta ignored, replaced by
    ``angle0_range``) and to :func:`build_symmetry_exploit_interpolant`
    for the 27 symmetry-exploit scalars. Raises ``ValueError`` for any
    other name so typos surface immediately.
    """
    if scalar_name in FLUX_FUNCTION_SCALARS:
        return build_flux_function_interpolant(
            field,
            scalar_name=scalar_name,
            rule=rule,
            s_range=s_range,
            extrapolate=extrapolate,
        )
    if scalar_name in SYMMETRY_EXPLOIT_SCALARS:
        return build_symmetry_exploit_interpolant(
            field,
            scalar_name=scalar_name,
            rule=rule,
            s_range=s_range,
            theta_range=theta_range,
            zeta_range=zeta_range,
            extrapolate=extrapolate,
        )
    raise ValueError(f"unknown scalar {scalar_name!r}; expected one of {ALL_SCALARS}")


def freeze_interpolated_boozer_field_state(
    field,
    *,
    degree: int,
    srange,
    thetarange,
    zetarange,
    extrapolate: bool = True,
    nfp: int = 1,
    stellsym: bool = True,
    scalars: tuple[str, ...] | None = None,
) -> InterpolatedBoozerFieldFrozenState:
    """Build the immutable grid state and initially requested scalar specs.

    Mirrors the construction sequence the C++
    ``InterpolatedBoozerField`` lazy-executes on its first
    ``modB()`` / ``K()`` / etc. call: each scalar gets its own
    ``RegularGridInterpolant3D`` fit, using ``angle0_range = (0, pi, 1)``
    for the seven flux-function scalars and the user-provided
    ``(s, theta, zeta)`` ranges for the 27 symmetry-exploit scalars.

    Args:
        field: a ``BoozerMagneticField`` that implements every getter in
            ``scalars``. ``BoozerRadialInterpolant`` (from VMEC) is the
            canonical full-coverage choice. ``BoozerAnalytic`` only
            implements 14 of the 34 scalars — pass an explicit
            ``scalars`` tuple naming just those.
        degree: Lagrange interpolation degree (>= 1).
        srange: ``(smin, smax, ns)`` triplet.
        thetarange: ``(thetamin, thetamax, ntheta)``.
        zetarange: ``(zetamin, zetamax, nzeta)``.
        extrapolate: forwarded to the underlying spec
            (``out_of_bounds_ok`` flag). Note that the coordinate fold
            normally keeps the lookup inside the domain after wrapping;
            ``extrapolate=False`` then surfaces NaN for queries that
            still fall outside after folding.
        nfp: rotational period count; folds zeta into ``[0, 2*pi/nfp]``.
        stellsym: when ``True``, additionally reflect samples with
            folded ``theta > pi``.
        scalars: an explicit iterable of scalar names to build at
            construction time. Defaults to all 34 (``ALL_SCALARS``).
            Pass a subset to match a base field that does not implement
            every getter, mirroring the C++ lazy-build behaviour.
    """
    degree_int = int(degree)
    if degree_int < 1:
        raise ValueError(f"degree must be >= 1; got {degree_int}")
    nfp_int = int(nfp)
    if nfp_int < 1:
        raise ValueError(f"nfp must be >= 1; got {nfp_int}")

    s_range = _validate_range("s", srange)
    theta_range = _validate_range("theta", thetarange)
    zeta_range = _validate_range("zeta", zetarange)

    selected = tuple(ALL_SCALARS if scalars is None else scalars)
    unknown = [name for name in selected if name not in ALL_SCALARS]
    if unknown:
        raise ValueError(
            f"unknown scalar(s) in `scalars` argument: {unknown}; "
            f"expected subset of {ALL_SCALARS}"
        )

    rule = UniformInterpolationRule(degree_int)

    specs: dict[str, RegularGridInterpolant3DSpec] = {}
    for name in selected:
        specs[name] = build_spec_for_scalar(
            field,
            scalar_name=name,
            rule=rule,
            s_range=s_range,
            theta_range=theta_range,
            zeta_range=zeta_range,
            extrapolate=extrapolate,
        )

    period = 2.0 * float(np.pi) / float(nfp_int)

    return InterpolatedBoozerFieldFrozenState(
        specs=MappingProxyType(dict(specs)),
        nfp=nfp_int,
        stellsym=bool(stellsym),
        extrapolate=bool(extrapolate),
        period=period,
        s_range=s_range,
        theta_range=theta_range,
        zeta_range=zeta_range,
        degree=degree_int,
    )


# ---------------------------------------------------------------------------
# Evaluation kernels
# ---------------------------------------------------------------------------


def _zeroed_flux_points(points: jax.Array) -> jax.Array:
    """Return ``(s, 0, 0)`` per row — matches C++ ``exploit_fluxfunction_points``.

    The s-column passes through verbatim; the theta/zeta columns are set
    to literal zero (or, for transfer-guard cleanliness, ``theta - theta``
    so we never broadcast a Python literal onto the device when there is
    actual array traffic on the same call).
    """
    s = points[:, 0]
    zeros = jnp.zeros_like(s)
    return jnp.stack([s, zeros, zeros], axis=1)


def _apply_symmetry(
    raw: jax.Array,
    *,
    flipped: jax.Array,
    rule: _SymmetryRule,
) -> jax.Array:
    """Apply C++ ``apply_odd_symmetry`` / ``apply_even_symmetry``.

    Mirrors header lines 786-807 exactly. ``raw`` has shape
    ``(N, value_size)`` (the spec's natural output shape).
    """
    # No symmetry to apply.
    if (
        not rule.apply_odd
        and not rule.apply_odd_vector_first_only
        and not rule.apply_even
    ):
        return raw
    # apply_odd: negate every component for flipped samples. The C++
    # header at lines 786-797 handles only ``value_size == 1`` and
    # ``value_size == 3`` (both arms negate column 0 only — see the
    # ``apply_odd_symmetry`` source). We exhaustively cover those two
    # sizes; any other value_size raises so a new field family cannot
    # silently slip through this branch.
    if rule.apply_odd:
        if rule.value_size == 1:
            sign = jnp.where(flipped, -1.0, 1.0)[:, None]
            return raw * sign
        if rule.value_size == 3:
            sign0 = jnp.where(flipped, -1.0, 1.0)
            col0 = raw[:, 0] * sign0
            return jnp.stack([col0, raw[:, 1], raw[:, 2]], axis=1)
        raise ValueError(
            f"apply_odd symmetry has no C++ oracle for value_size="
            f"{rule.value_size}; supported sizes are 1 and 3."
        )
    # apply_odd_vector_first_only: negate component 0 only on a 3-vec.
    # This branch corresponds to the C++ Z_derivs / nu_derivs path
    # (header lines 558-561, 350-352). value_size != 3 has no oracle.
    if rule.apply_odd_vector_first_only:
        if rule.value_size != 3:
            raise ValueError(
                f"apply_odd_vector_first_only is only defined for value_size=3; "
                f"got value_size={rule.value_size}."
            )
        sign0 = jnp.where(flipped, -1.0, 1.0)
        col0 = raw[:, 0] * sign0
        return jnp.stack([col0, raw[:, 1], raw[:, 2]], axis=1)
    # apply_even: negate components 1, 2 of a 3-vec. C++ ``apply_even_symmetry``
    # at header lines 799-807 checks ``field.shape(1)==3`` and is undefined
    # for other sizes.
    if rule.apply_even:
        if rule.value_size != 3:
            raise ValueError(
                f"apply_even is only defined for value_size=3; "
                f"got value_size={rule.value_size}."
            )
        sign12 = jnp.where(flipped, -1.0, 1.0)
        col1 = raw[:, 1] * sign12
        col2 = raw[:, 2] * sign12
        return jnp.stack([raw[:, 0], col1, col2], axis=1)
    # The early-return at the top already handles the all-False rule;
    # any path reaching here would indicate a rule with multiple flags
    # set, which is not a valid C++ ``apply_*_symmetry`` combination.
    raise ValueError(
        f"_SymmetryRule combination has no C++ oracle: "
        f"apply_odd={rule.apply_odd}, "
        f"apply_odd_vector_first_only={rule.apply_odd_vector_first_only}, "
        f"apply_even={rule.apply_even}."
    )


def evaluate_scalar(
    state: InterpolatedBoozerFieldFrozenState,
    specs: Mapping[str, RegularGridInterpolant3DSpec],
    scalar_name: str,
    points: jax.Array,
) -> jax.Array:
    """Evaluate ``scalar_name`` at ``points`` using ``specs``.

    Dispatches flux-function vs symmetry-exploit based on the C++
    inventory in ``FLUX_FUNCTION_SCALARS`` / ``SYMMETRY_EXPLOIT_SCALARS``.
    Output shape ``(N, value_size)`` matches the C++ ``modB() / R() / etc.``
    return contract (``{npoints, value_size}``).
    """
    spec = specs.get(scalar_name)
    if spec is None:
        raise KeyError(
            f"interpolant for scalar {scalar_name!r} has not been built; "
            f"available: {sorted(specs)}"
        )
    if scalar_name in FLUX_FUNCTION_SCALARS:
        folded = _zeroed_flux_points(points)
        return evaluate_batch(spec, folded)
    if scalar_name in SYMMETRY_EXPLOIT_SCALARS:
        rule = SYMMETRY_EXPLOIT_SCALARS[scalar_name]
        folded, flipped = fold_points_for_symmetry(
            points,
            period=jnp.asarray(state.period, dtype=jnp.float64),
            stellsym=state.stellsym,
        )
        raw = evaluate_batch(spec, folded)
        return _apply_symmetry(raw, flipped=flipped, rule=rule)
    raise ValueError(
        f"unknown scalar {scalar_name!r}; not in flux-function or symmetry-exploit "
        f"inventory"
    )
