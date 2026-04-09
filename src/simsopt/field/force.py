"""Implements the force on a coil in its own magnetic field and the field of other coils."""

from dataclasses import dataclass, replace
from threading import RLock
from weakref import WeakValueDictionary

from scipy import constants
import numpy as np
import jax.numpy as jnp
import jax.scipy as jscp
from jax import grad, vmap
from jax.lax import cond
from .biotsavart import BiotSavart
from .coil import (
    Current,
    CurrentSum,
    RegularizedCoil,
    ScaledCurrent,
    _unwrap_coil_curve_and_current_objects,
)
from .selffield import B_regularized_pure
from ..geo.curve import _curve_jax_eval_from_arg, _optimizable_dof_map_spec
from ..geo.jit import jit
from ..geo.surfacerzfourier import SurfaceRZFourier
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec
from ..jax_core import (
    curve_gamma_and_dash_from_spec,
    curve_geometry_from_spec,
    curve_spec_from_curve,
    curve_spec_with_dofs,
    make_coil_symmetry_spec,
)
from ..jax_core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    zeros as _jax_zeros,
)
from ..jax_core.curve_geometry import optimizable_input_dofs_from_map_spec
from ..jax_core.surface_rzfourier import surface_rz_fourier_spec_from_dofs

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi

__all__ = [
    "_coil_coil_inductances_pure",
    "_coil_coil_inductances_inv_pure",
    "_induced_currents_pure",
    "NetFluxes",
    "B2Energy",
    "SquaredMeanForce",
    "LpCurveForce",
    "SquaredMeanTorque",
    "LpCurveTorque",
]


def _check_quadpoints_consistency(coils, label="coils"):
    """Check that all coils in a list have the same number of quadrature points.

    Args:
        coils: list of coils to check.
        label: descriptive label for the coil group (used in error message).

    Raises:
        ValueError: if not all coils have the same number of quadrature points.
    """
    nquadpoints = [len(c.curve.quadpoints) for c in coils]
    if len(set(nquadpoints)) > 1:
        raise ValueError(
            f"All coils in {label} must have the same number of quadrature points, "
            f"but got {nquadpoints}."
        )


def _check_downsample(coils, downsample, label="coils"):
    """Check that downsample evenly divides the number of quadrature points.

    Args:
        coils: list of coils to check (must be non-empty).
        downsample: downsampling factor.
        label: descriptive label for the coil group (used in error message).

    Raises:
        ValueError: if downsample does not evenly divide the number of quadrature points.
    """
    if downsample < 1:
        raise ValueError(f"downsample must be >= 1, but got {downsample}.")
    nquadpoints = len(coils[0].curve.quadpoints)
    if nquadpoints % downsample != 0:
        raise ValueError(
            f"downsample ({downsample}) must evenly divide the number of quadrature points "
            f"({nquadpoints}) in {label}, but {nquadpoints} % {downsample} = {nquadpoints % downsample}."
        )


def _B_at_point_from_coil_set_pure(
    pt, gammas, gammadashs, currents, exclude_index, eps
):
    r"""
    Compute the magnetic field at a single point due to a set of coils via the Biot-Savart law,
    optionally excluding one coil (e.g. to avoid self-contribution).

    This is a pure JAX implementation of the Biot-Savart integral used by force and torque
    objectives in this module. We do not use the :class:`BiotSavart` class here because
    constructing one :class:`BiotSavart` per objective (e.g. per coil in :class:`LpCurveForce`)
    leads to a large number of optimizable dependencies and weak references. That makes
    operations like ``Jf.x = dofs`` scale poorly with the number of coils (tens of millions of
    function calls and tens of seconds for ~64 coils). See `GitHub issue #487
    <https://github.com/hiddenSymmetries/simsopt/issues/487>`_.

    .. math::
        B = \frac{\mu_0}{4\pi} \frac{1}{n_{pts}} \sum_{j \neq \mathrm{exclude}} I_j \int \frac{d\vec{\ell}_j \times (\vec{r} - \vec{r}_j)}{|\vec{r} - \vec{r}_j|^3}

    Args:
        pt: Array of shape (3,); evaluation point.
        gammas: Array of shape (m, n, 3); positions for m coils with n quadrature points.
        gammadashs: Array of shape (m, n, 3); tangent vectors.
        currents: Array of shape (m,); coil currents.
        exclude_index: Index of coil to exclude from the sum (use -1 to include all).
        eps: Small constant added to distances to avoid division by zero.

    Returns:
        Array of shape (3,); magnetic field contribution (without mu_0/(4*pi)).
    """
    n = gammas.shape[0]
    npts = gammas.shape[1]
    if n == 0:
        return jnp.zeros(3)

    def from_j(j):
        return cond(
            (exclude_index >= 0) & (j == exclude_index),
            lambda _: jnp.zeros(3),
            lambda _: jnp.asarray(
                jnp.sum(
                    jnp.cross(gammadashs[j], pt - gammas[j])
                    / (jnp.linalg.norm(pt - gammas[j] + eps, axis=1) ** 3)[:, None],
                    axis=0,
                )
                * currents[j]
            ),
            operand=None,
        )

    B = jnp.sum(vmap(from_j)(jnp.arange(n)), axis=0)
    return B / npts * 1e-7


def _mutual_B_field_at_point_pure(
    i,
    pt,
    gammas_targets,
    gammadashs_targets,
    currents_targets,
    gammas_sources_coarse,
    gammadashs_sources_coarse,
    currents_sources_coarse,
    gammas_sources_fine,
    gammadashs_sources_fine,
    currents_sources_fine,
    eps,
):
    r"""
    Compute the mutual magnetic field at a point on target coil i from all target coils
    (excluding coil i) and all source coils (coarse and fine) in Tesla.

    Used by :func:`squared_mean_force_pure`, :func:`lp_force_pure`, :func:`lp_torque_pure`,
    and :func:`squared_mean_torque`. See :func:`_B_at_point_from_coil_set_pure`
    for why Biot-Savart is reimplemented here instead of using :class:`BiotSavart`.

    Args:
        i: Index of target coil.
        pt: Array of shape (3,); evaluation point.
        gammas_targets: Array of shape (m, n, 3); positions for m target coils with n quadrature points.
        gammadashs_targets: Array of shape (m, n, 3); tangent vectors for m target coils with n quadrature points.
        currents_targets: Array of shape (m,); currents for m target coils.
        gammas_sources_coarse: Array of shape (m', n', 3); positions for m' coarse source coils.
        gammadashs_sources_coarse: Array of shape (m', n', 3); tangent vectors for coarse source coils.
        currents_sources_coarse: Array of shape (m',); currents for coarse source coils.
        gammas_sources_fine: Array of shape (m'', n'', 3); positions for m'' fine source coils (may be empty).
        gammadashs_sources_fine: Tangent vectors for fine source coils.
        currents_sources_fine: Currents for fine source coils.
        eps: Small constant added to distances to avoid division by zero.

    Returns:
        Array of shape (3,); mutual magnetic field at point pt in Tesla.
    """
    B_targets = _B_at_point_from_coil_set_pure(
        pt,
        gammas_targets,
        gammadashs_targets,
        currents_targets,
        exclude_index=i,
        eps=eps,
    )
    B_sources_coarse = _B_at_point_from_coil_set_pure(
        pt,
        gammas_sources_coarse,
        gammadashs_sources_coarse,
        currents_sources_coarse,
        exclude_index=-1,
        eps=eps,
    )
    B_sources_fine = _B_at_point_from_coil_set_pure(
        pt,
        gammas_sources_fine,
        gammadashs_sources_fine,
        currents_sources_fine,
        exclude_index=-1,
        eps=eps,
    )
    return B_targets + B_sources_coarse + B_sources_fine


def _lorentz_force_density_pure(tangents, current, magnetic_field):
    """Compute Lorentz force density I * (t x B)."""
    return current * jnp.cross(tangents, magnetic_field)


def _prepare_target_source_inputs_pure(
    gammas_targets,
    gammadashs_targets,
    gammas_sources,
    gammadashs_sources,
    currents_targets,
    currents_sources,
    downsample,
):
    r"""
    Downsample and convert shared target/source inputs used by force/torque objectives.

    Args:
        gammas_targets: Array of shape (m, n, 3); positions for m target coils with n quadrature points.
        gammadashs_targets: Array of shape (m, n, 3); tangent vectors for m target coils with n quadrature points.
        gammas_sources: Array of shape (m', n, 3); positions for m' source coils with n quadrature points.
        gammadashs_sources: Array of shape (m', n, 3); tangent vectors for m' source coils with n quadrature points.
        currents_targets: Array of shape (m,); currents for m target coils.
        currents_sources: Array of shape (m',); currents for m' source coils.
        downsample: Factor by which to downsample the quadrature points.

    Returns:
        Tuple of arrays: (gammas_targets, gammadashs_targets, gammas_sources, gammadashs_sources, currents_targets, currents_sources).
    """
    return (
        _as_jax_float64(gammas_targets)[:, ::downsample, :],
        _as_jax_float64(gammadashs_targets)[:, ::downsample, :],
        _as_jax_float64(gammas_sources)[:, ::downsample, :],
        _as_jax_float64(gammadashs_sources)[:, ::downsample, :],
        _as_jax_float64(currents_targets),
        _as_jax_float64(currents_sources),
    )


def _prepare_regularized_target_source_inputs_pure(
    gammas_targets,
    gammadashs_targets,
    gammadashdashs_targets,
    quadpoints,
    gammas_sources,
    gammadashs_sources,
    currents_targets,
    currents_sources,
    regularizations,
    downsample,
):
    """
    Downsample/convert inputs for regularized Lp force/torque objectives. Just a wrapper around
    _prepare_target_source_inputs_pure that also prepares additional inputs for regularized coils.

    Args:
        gammas_targets: Array of shape (m, n, 3); positions for m target coils with n quadrature points.
        gammadashs_targets: Array of shape (m, n, 3); tangent vectors for m target coils with n quadrature points.
        gammadashdashs_targets: Array of shape (m, n, 3); second derivatives of tangent vectors for m target coils with n quadrature points.
        quadpoints: Array of shape (m, n); quadrature points for m target coils with n quadrature points.
        gammas_sources: Array of shape (m', n, 3); positions for m' source coils with n quadrature points.
        gammadashs_sources: Array of shape (m', n, 3); tangent vectors for m' source coils with n quadrature points.
        currents_targets: Array of shape (m,); currents for m target coils.
        currents_sources: Array of shape (m',); currents for m' source coils.
        regularizations: Array of shape (m,); regularizations for m target coils.
        downsample: Factor by which to downsample the quadrature points.

    Returns:
        Tuple of arrays: (gammas_targets, gammadashs_targets, gammadashdashs_targets, quadpoints, gammas_sources, gammadashs_sources, currents_targets, currents_sources, regularizations).
    """
    (
        gammas_targets,
        gammadashs_targets,
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
    ) = _prepare_target_source_inputs_pure(
        gammas_targets,
        gammadashs_targets,
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
        downsample,
    )
    return (
        gammas_targets,
        gammadashs_targets,
        _as_jax_float64(gammadashdashs_targets)[:, ::downsample, :],
        _as_jax_float64(quadpoints[0])[::downsample],
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
        _as_jax_float64(regularizations),
    )


def _empty_coil_state_arrays(*, include_gammadashdash=False):
    empty_curve = _jax_zeros((0, 1, 3))
    empty_current = _jax_zeros((0,))
    if include_gammadashdash:
        return empty_curve, empty_curve, empty_curve, empty_current
    return empty_curve, empty_curve, empty_current


def _empty_source_fine_arrays():
    return _empty_coil_state_arrays(include_gammadashdash=False)


def _optimizable_input_dof_map_spec(owner, opt, *, input_mode):
    """Build an owner->optimizable map for either local or full DOF inputs."""
    from ..jax_core import make_optimizable_dof_map_spec

    template_full_dofs = _as_jax_float64(opt.full_x)
    owner_segments = tuple(
        (
            int(owner._full_dof_indices[dep_opt][0]),
            int(owner._full_dof_indices[dep_opt][1]),
            int(sub_start),
            int(sub_end),
        )
        for dep_opt, (sub_start, sub_end) in opt._full_dof_indices.items()
    )
    if input_mode == "full":
        input_start = 0
        input_end = int(template_full_dofs.shape[0])
    else:
        input_start, input_end = opt._full_dof_indices[opt]
    return make_optimizable_dof_map_spec(
        template_full_dofs=template_full_dofs,
        owner_segments=owner_segments,
        input_mode=input_mode,
        input_start=int(input_start),
        input_end=int(input_end),
    )


def _optimizable_local_full_dof_map_spec(owner, opt):
    """Build an owner->optimizable map that returns local full DOFs."""
    return _optimizable_input_dof_map_spec(owner, opt, input_mode="local")


def _optimizable_full_dof_map_spec(owner, opt):
    """Build an owner->optimizable map that returns full graph DOFs."""
    return _optimizable_input_dof_map_spec(owner, opt, input_mode="full")


def _curve_spec_dof_map_spec(owner, curve, *, curve_spec_template=None):
    """Match the DOF payload shape expected by ``curve.to_spec()``."""
    if curve_spec_template is None:
        try:
            curve_spec_template = curve_spec_from_curve(curve)
        except NotImplementedError:
            return None
    if curve_spec_template.dofs.shape[0] == curve.full_x.shape[0]:
        return _optimizable_full_dof_map_spec(owner, curve)
    return _optimizable_local_full_dof_map_spec(owner, curve)


@dataclass(frozen=True)
class _SurfaceSpecBinding:
    spec_template: object
    dof_map: object


@dataclass(frozen=True)
class _CurveSpecBinding:
    spec_template: object
    base_curve_binding: object = None
    surface_binding: object = None


@dataclass(frozen=True)
class _CurrentStateBinding:
    kind: str
    dof_map: object = None
    child: object = None
    left: object = None
    right: object = None
    scale: float = 1.0


def _build_surface_spec_binding(owner, surface):
    if surface.dof_size == 0:
        return None
    if not isinstance(surface, SurfaceRZFourier):
        raise NotImplementedError(
            "Shared selffield state only supports immutable RZ Fourier surface specs, "
            f"got {type(surface).__name__}."
        )
    return _SurfaceSpecBinding(
        spec_template=surface.surface_spec(),
        dof_map=_optimizable_local_full_dof_map_spec(owner, surface),
    )


def _build_curve_spec_binding(owner, curve):
    base_curve = getattr(curve, "curve", None)
    surface = getattr(curve, "surf", None)
    return _CurveSpecBinding(
        spec_template=curve_spec_from_curve(curve),
        base_curve_binding=(
            None if base_curve is None else _build_curve_spec_binding(owner, base_curve)
        ),
        surface_binding=(
            None if surface is None else _build_surface_spec_binding(owner, surface)
        ),
    )


def _surface_spec_from_binding(binding, owner_dofs):
    surface_dofs = optimizable_input_dofs_from_map_spec(binding.dof_map, owner_dofs)
    spec = binding.spec_template
    return surface_rz_fourier_spec_from_dofs(
        surface_dofs,
        quadpoints_phi=spec.quadpoints_phi,
        quadpoints_theta=spec.quadpoints_theta,
        mpol=spec.mpol,
        ntor=spec.ntor,
        nfp=spec.nfp,
        stellsym=spec.stellsym,
    )


def _curve_spec_from_binding(binding, owner_dofs):
    updates = {}
    if binding.base_curve_binding is not None:
        updates["base_curve"] = _curve_spec_from_binding(
            binding.base_curve_binding, owner_dofs
        )
    if binding.surface_binding is not None:
        updates["surface"] = _surface_spec_from_binding(
            binding.surface_binding, owner_dofs
        )
    if not updates:
        return binding.spec_template
    return replace(binding.spec_template, **updates)


def _build_current_state_binding(owner, current):
    if isinstance(current, ScaledCurrent):
        return _CurrentStateBinding(
            kind="scaled",
            child=_build_current_state_binding(owner, current.current_to_scale),
            scale=float(current.scale),
        )
    if isinstance(current, CurrentSum):
        return _CurrentStateBinding(
            kind="sum",
            left=_build_current_state_binding(owner, current.current_a),
            right=_build_current_state_binding(owner, current.current_b),
        )
    if isinstance(current, Current):
        return _CurrentStateBinding(
            kind="scalar",
            dof_map=_optimizable_local_full_dof_map_spec(owner, current),
        )
    raise NotImplementedError(
        "Shared selffield state only supports scalar Current graphs composed from "
        f"Current, ScaledCurrent, and CurrentSum; got {type(current).__name__}."
    )


@dataclass(frozen=True)
class _CoilStateEntry:
    coil: object
    curve: object
    current: object
    curve_spec_binding: object
    curve_spec_map: object
    curve_jax_map: object
    current_binding: object
    symmetry: object


def _build_coil_state_entry(coil):
    curve, rotmat, current, scale = _unwrap_coil_curve_and_current_objects(
        coil.curve,
        coil.current,
    )
    try:
        curve_spec_binding = _build_curve_spec_binding(coil, curve)
    except NotImplementedError:
        curve_spec_binding = None
    curve_spec_template = (
        None if curve_spec_binding is None else curve_spec_binding.spec_template
    )
    return _CoilStateEntry(
        coil=coil,
        curve=curve,
        current=current,
        curve_spec_binding=curve_spec_binding,
        curve_spec_map=_curve_spec_dof_map_spec(
            coil,
            curve,
            curve_spec_template=curve_spec_template,
        ),
        curve_jax_map=_optimizable_dof_map_spec(coil, curve),
        current_binding=_build_current_state_binding(coil, current),
        symmetry=make_coil_symmetry_spec(rotmat=rotmat, scale=scale),
    )


def _apply_coil_state_symmetry(
    gamma,
    gammadash,
    current,
    symmetry,
    *,
    gammadashdash=None,
):
    if symmetry.has_rotation:
        rotmat = _as_jax_float64(symmetry.rotmat)
        gamma = gamma @ rotmat
        gammadash = gammadash @ rotmat
        if gammadashdash is not None:
            gammadashdash = gammadashdash @ rotmat
    current = current * _as_jax_float64(symmetry.scale)
    if gammadashdash is None:
        return gamma, gammadash, current
    return gamma, gammadash, gammadashdash, current


def _curve_state_from_spec(entry, owner_dofs, *, include_gammadashdash):
    if entry.curve_spec_binding is None or entry.curve_spec_map is None:
        raise NotImplementedError
    curve_spec = curve_spec_with_dofs(
        _curve_spec_from_binding(entry.curve_spec_binding, owner_dofs),
        optimizable_input_dofs_from_map_spec(entry.curve_spec_map, owner_dofs),
    )
    if include_gammadashdash:
        return curve_geometry_from_spec(curve_spec)
    return curve_gamma_and_dash_from_spec(curve_spec)


def _curve_state_from_jax_methods(entry, owner_dofs, *, include_gammadashdash):
    curve_dofs = optimizable_input_dofs_from_map_spec(entry.curve_jax_map, owner_dofs)
    gamma = _curve_jax_eval_from_arg(entry.curve, "gamma_jax", curve_dofs)
    gammadash = _curve_jax_eval_from_arg(entry.curve, "gammadash_jax", curve_dofs)
    if include_gammadashdash:
        if not hasattr(entry.curve, "gammadashdash_jax"):
            raise NotImplementedError
        gammadashdash = _curve_jax_eval_from_arg(
            entry.curve,
            "gammadashdash_jax",
            curve_dofs,
        )
        return gamma, gammadash, gammadashdash
    return gamma, gammadash


def _current_value_from_binding(binding, owner_dofs):
    if binding.kind == "scalar":
        current_dofs = optimizable_input_dofs_from_map_spec(binding.dof_map, owner_dofs)
        if current_dofs.shape[0] != 1:
            raise RuntimeError(
                "Shared selffield state requires scalar current leaf DOFs, "
                f"got {int(current_dofs.shape[0])}."
            )
        return current_dofs[0]
    if binding.kind == "scaled":
        return _as_jax_float64(binding.scale) * _current_value_from_binding(
            binding.child,
            owner_dofs,
        )
    if binding.kind == "sum":
        return _current_value_from_binding(
            binding.left,
            owner_dofs,
        ) + _current_value_from_binding(binding.right, owner_dofs)
    raise TypeError(f"Unsupported current binding kind: {binding.kind!r}")


def _current_value_from_entry(entry, owner_dofs):
    return _current_value_from_binding(entry.current_binding, owner_dofs)


def _curve_state_from_entry(entry, owner_dofs, *, include_gammadashdash):
    try:
        return _curve_state_from_spec(
            entry,
            owner_dofs,
            include_gammadashdash=include_gammadashdash,
        )
    except NotImplementedError:
        pass

    try:
        return _curve_state_from_jax_methods(
            entry,
            owner_dofs,
            include_gammadashdash=include_gammadashdash,
        )
    except (AttributeError, NotImplementedError) as exc:
        raise RuntimeError(
            "Shared selffield state requires immutable curve specs or JAX curve "
            f"methods; unsupported curve type {type(entry.curve).__name__}."
        ) from exc


def _build_shared_coil_state(entry, *, include_gammadashdash=False):
    """Build one coil's packed state via immutable specs, then JAX hooks."""
    owner_dofs = _as_jax_float64(entry.coil.full_x)
    state = _curve_state_from_entry(
        entry,
        owner_dofs,
        include_gammadashdash=include_gammadashdash,
    )
    current = _current_value_from_entry(entry, owner_dofs)
    if include_gammadashdash:
        gamma, gammadash, gammadashdash = state
        return _apply_coil_state_symmetry(
            gamma,
            gammadash,
            current,
            entry.symmetry,
            gammadashdash=gammadashdash,
        )
    gamma, gammadash = state
    return _apply_coil_state_symmetry(
        gamma,
        gammadash,
        current,
        entry.symmetry,
    )


def _stack_coil_states(states, *, include_gammadashdash):
    if include_gammadashdash:
        gammas, gammadashs, gammadashdashs, currents = zip(*states)
        return (
            jnp.stack(gammas),
            jnp.stack(gammadashs),
            jnp.stack(gammadashdashs),
            jnp.stack(currents),
        )
    gammas, gammadashs, currents = zip(*states)
    return jnp.stack(gammas), jnp.stack(gammadashs), jnp.stack(currents)


class _SharedCoilState:
    """Shared packed per-coil state derived from the live Optimizable graph."""

    def __init__(self, coil):
        self.coil = coil
        self._entry = _build_coil_state_entry(coil)
        self._lock = RLock()
        self._dirty = True
        self._gamma = None
        self._gammadash = None
        self._gammadashdash = None
        self._current = None

    def clear(self):
        with self._lock:
            self._dirty = True
            self._gamma = None
            self._gammadash = None
            self._gammadashdash = None
            self._current = None

    def mark_dirty(self, coil):
        with self._lock:
            if coil is not self.coil:
                return False
            self._dirty = True
            return True

    def has_dirty_entries(self):
        with self._lock:
            return self._dirty

    def state(self, *, include_gammadashdash=False):
        with self._lock:
            keep_second_derivatives = (
                include_gammadashdash or self._gammadashdash is not None
            )
            if self._gamma is None or self._dirty:
                self._rebuild_locked(include_gammadashdash=keep_second_derivatives)
            elif include_gammadashdash and self._gammadashdash is None:
                self._rebuild_locked(include_gammadashdash=True)
            return self._state_tuple_locked(include_gammadashdash=include_gammadashdash)

    def _rebuild_locked(self, *, include_gammadashdash):
        state = _build_shared_coil_state(
            self._entry,
            include_gammadashdash=include_gammadashdash,
        )
        if include_gammadashdash:
            self._gamma, self._gammadash, self._gammadashdash, self._current = state
        else:
            self._gamma, self._gammadash, self._current = state
            self._gammadashdash = None
        self._dirty = False

    def _state_tuple_locked(self, *, include_gammadashdash):
        if include_gammadashdash:
            return self._gamma, self._gammadash, self._gammadashdash, self._current
        return self._gamma, self._gammadash, self._current


_SHARED_COIL_STATE_SERVICES = WeakValueDictionary()
_SHARED_COIL_STATE_SERVICES_LOCK = RLock()


def _shared_coil_state_service(coil):
    key = id(coil)
    with _SHARED_COIL_STATE_SERVICES_LOCK:
        service = _SHARED_COIL_STATE_SERVICES.get(key)
        if service is None or service.coil is not coil:
            service = _SharedCoilState(coil)
            _SHARED_COIL_STATE_SERVICES[key] = service
        return service


class _CoilStateGroupCache:
    """Objective-local view over shared per-coil packed state services."""

    def __init__(self, coils, *, include_gammadashdash=False):
        self.coils = tuple(coils)
        self.include_gammadashdash = include_gammadashdash
        self._services = tuple(_shared_coil_state_service(coil) for coil in self.coils)

    def clear(self):
        for service in self._services:
            service.clear()

    def mark_dirty(self, coil):
        return any(service.mark_dirty(coil) for service in self._services)

    def has_dirty_entries(self):
        return any(service.has_dirty_entries() for service in self._services)

    def arrays(self):
        if len(self._services) == 0:
            return _empty_coil_state_arrays(
                include_gammadashdash=self.include_gammadashdash
            )
        states = tuple(
            service.state(include_gammadashdash=self.include_gammadashdash)
            for service in self._services
        )
        return _stack_coil_states(
            states,
            include_gammadashdash=self.include_gammadashdash,
        )


def _invalidate_objective_state(owner, parent, *group_caches):
    """Invalidate packed args and mark only the affected coil-group entries dirty."""
    owner._cached_jax_args = None
    if parent is None:
        if any(group_cache.has_dirty_entries() for group_cache in group_caches):
            return
        for group_cache in group_caches:
            group_cache.clear()
        return
    matched_parent = False
    for group_cache in group_caches:
        matched_parent = group_cache.mark_dirty(parent) or matched_parent
    if not matched_parent:
        for group_cache in group_caches:
            group_cache.clear()


def _assemble_curve_current_derivative(
    coils,
    *,
    dgamma=None,
    dgammadash=None,
    dgammadashdash=None,
    dcurrent=None,
):
    """Map JAX derivative blocks back to Optimizable derivatives."""
    deriv = 0
    if dgamma is not None:
        deriv += sum(
            c.curve.dgamma_by_dcoeff_vjp(dgamma[i]) for i, c in enumerate(coils)
        )
    if dgammadash is not None:
        deriv += sum(
            c.curve.dgammadash_by_dcoeff_vjp(dgammadash[i]) for i, c in enumerate(coils)
        )
    if dgammadashdash is not None:
        deriv += sum(
            c.curve.dgammadashdash_by_dcoeff_vjp(dgammadashdash[i])
            for i, c in enumerate(coils)
        )
    if dcurrent is not None:
        deriv += sum(
            c.current.vjp(jnp.asarray([dcurrent[i]])) for i, c in enumerate(coils)
        )
    return deriv


def _cached_objective_args(owner, build_args):
    """Reuse packed JAX inputs until Optimizable invalidates them."""
    cached_args = owner._cached_jax_args
    if owner.new_x or cached_args is None:
        cached_args = tuple(build_args())
        owner._cached_jax_args = cached_args
        owner.new_x = False
    return cached_args


def _squared_mean_force_eval(
    gammas_targets,
    gammas_coarse,
    gammadashs_targets,
    gammadashs_coarse,
    currents_targets,
    currents_coarse,
    gammas_fine,
    gammadashs_fine,
    currents_fine,
    downsample,
):
    return squared_mean_force_pure(
        gammas_targets,
        gammas_coarse,
        gammadashs_targets,
        gammadashs_coarse,
        currents_targets,
        currents_coarse,
        downsample,
        gammas_sources_fine=gammas_fine,
        gammadashs_sources_fine=gammadashs_fine,
        currents_sources_fine=currents_fine,
    )


def _lp_force_eval(
    gammas_targets,
    gammas_coarse,
    gammadashs_targets,
    gammadashs_coarse,
    gammadashdashs_targets,
    currents_targets,
    currents_coarse,
    gammas_fine,
    gammadashs_fine,
    currents_fine,
    quadpoints,
    regularizations,
    p,
    threshold,
    downsample,
):
    return lp_force_pure(
        gammas_targets,
        gammas_coarse,
        gammadashs_targets,
        gammadashs_coarse,
        gammadashdashs_targets,
        quadpoints,
        currents_targets,
        currents_coarse,
        regularizations,
        p,
        threshold,
        downsample,
        gammas_sources_fine=gammas_fine,
        gammadashs_sources_fine=gammadashs_fine,
        currents_sources_fine=currents_fine,
    )


def _lp_torque_eval(
    gammas_targets,
    gammas_coarse,
    gammadashs_targets,
    gammadashs_coarse,
    gammadashdashs_targets,
    currents_targets,
    currents_coarse,
    gammas_fine,
    gammadashs_fine,
    currents_fine,
    quadpoints,
    regularizations,
    p,
    threshold,
    downsample,
):
    return lp_torque_pure(
        gammas_targets,
        gammas_coarse,
        gammadashs_targets,
        gammadashs_coarse,
        gammadashdashs_targets,
        quadpoints,
        currents_targets,
        currents_coarse,
        regularizations,
        p,
        threshold,
        downsample,
        gammas_sources_fine=gammas_fine,
        gammadashs_sources_fine=gammadashs_fine,
        currents_sources_fine=currents_fine,
    )


def _squared_mean_torque_eval(
    gammas_targets,
    gammas_coarse,
    gammadashs_targets,
    gammadashs_coarse,
    currents_targets,
    currents_coarse,
    gammas_fine,
    gammadashs_fine,
    currents_fine,
    downsample,
):
    return squared_mean_torque(
        gammas_targets,
        gammas_coarse,
        gammadashs_targets,
        gammadashs_coarse,
        currents_targets,
        currents_coarse,
        downsample,
        gammas_sources_fine=gammas_fine,
        gammadashs_sources_fine=gammadashs_fine,
        currents_sources_fine=currents_fine,
    )


def _b2energy_eval(gammas, gammadashs, currents, downsample, regularizations):
    return b2energy_pure(gammas, gammadashs, currents, downsample, regularizations)


def _net_ext_flux_eval(gammadash, A_ext, downsample):
    return net_ext_fluxes_pure(gammadash, A_ext, downsample)


_B2ENERGY_JAX = jit(_b2energy_eval, static_argnums=(3,))
_B2ENERGY_GRAD = jit(grad(_b2energy_eval, argnums=(0, 1, 2)), static_argnums=(3,))
_NET_EXT_FLUX_JAX = jit(_net_ext_flux_eval, static_argnums=(2,))
_NET_EXT_FLUX_GRAD = jit(grad(_net_ext_flux_eval, argnums=(0, 1)), static_argnums=(2,))
_SQUARED_MEAN_FORCE_JAX = jit(_squared_mean_force_eval, static_argnums=(9,))
_SQUARED_MEAN_FORCE_GRAD = jit(
    grad(_squared_mean_force_eval, argnums=tuple(range(9))),
    static_argnums=(9,),
)
_LP_FORCE_JAX = jit(_lp_force_eval, static_argnums=(14,))
_LP_FORCE_GRAD = jit(
    grad(_lp_force_eval, argnums=tuple(range(10))),
    static_argnums=(14,),
)
_LP_TORQUE_JAX = jit(_lp_torque_eval, static_argnums=(14,))
_LP_TORQUE_GRAD = jit(
    grad(_lp_torque_eval, argnums=tuple(range(10))),
    static_argnums=(14,),
)
_SQUARED_MEAN_TORQUE_JAX = jit(_squared_mean_torque_eval, static_argnums=(9,))
_SQUARED_MEAN_TORQUE_GRAD = jit(
    grad(_squared_mean_torque_eval, argnums=tuple(range(9))),
    static_argnums=(9,),
)


def _coil_coil_inductances_pure(
    gammas, gammadashs, downsample, regularizations, eps=1e-10
):
    r"""
    Compute the full inductance matrix for a set of coils, including both mutual and
    self-inductances. All coils are assumed to have the same number of quadrature points,
    denoted n. The units of the inductance matrix are H, where H = henries.

    The mutual inductance between two coils is computed as:

    .. math::

        M = \frac{\mu_0}{4\pi} \iint \frac{d\vec{r}_A \cdot d\vec{r}_B}{|\vec{r}_A - \vec{r}_B|}

    and self-inductance of a regularized coil is computed as:

    .. math::

        L = \frac{\mu_0}{4\pi} \int_0^{2\pi} d\phi \int_0^{2\pi} d\tilde{\phi}
            \frac{\vec{r}_c' \cdot \tilde{\vec{r}}_c'}{\sqrt{|\vec{r}_c - \tilde{\vec{r}}_c|^2 + \delta a b}}

    where $\delta a b$ is a regularization parameter depending on the cross-section. The units
    of the inductance matrices are henries.

    Args:
        gammas (array, shape (m,n,3)):
            Array of coil positions for all m coils.
        gammadashs (array, shape (m,n,3)):
            Array of coil tangent vectors for all m coils.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
        regularizations (array, shape (m,)):
            Array of regularizations coming from finite cross-section for all m coils. The choices
            for each coil are regularization_circ and regularization_rect, although each coil can
            have different size and shape cross-sections in this list of regularization terms.
        eps (float): Small constant to avoid division by zero for mutual inductance between coil_i and itself.
    Returns:
        array (shape (m,m)): Full inductance matrix Lij.
    """
    gammas = _as_jax_float64(gammas)[:, ::downsample, :]
    gammadashs = _as_jax_float64(gammadashs)[:, ::downsample, :]
    N = gammas.shape[0]

    # Compute Lij, i != j
    r_ij = gammas[None, :, None, :, :] - gammas[:, None, :, None, :] + eps
    rij_norm = jnp.linalg.norm(r_ij, axis=-1)
    gammadash_prod = jnp.sum(
        gammadashs[None, :, None, :, :] * gammadashs[:, None, :, None, :], axis=-1
    )

    # Double sum over each of the closed curves for off-diagonal elements
    Lij = (
        jnp.sum(jnp.sum(gammadash_prod / rij_norm, axis=-1), axis=-1)
        / jnp.shape(gammas)[1] ** 2
    )

    # Compute diagonal elements for each coil
    diag_values = (
        jnp.sum(
            jnp.sum(
                gammadash_prod
                / jnp.sqrt(rij_norm**2 + regularizations[None, :, None, None]),
                axis=-1,
            ),
            axis=-1,
        )
        / jnp.shape(gammas)[1] ** 2
    )

    # Now use a mask to replace the wrong diagonal with the correct numbers in diag_values
    diag_mask = jnp.eye(N, dtype=bool)
    Lij = jnp.where(diag_mask, diag_values, Lij)
    return 1e-7 * Lij


def _coil_coil_inductances_inv_pure(gammas, gammadashs, downsample, regularizations):
    """
    Pure function for computing the inverse of the coil inductance matrix L. This matrix
    is symmetric positive definite by definition.

    Performs a Cholesky decomposition of the coil inductance matrix L and then solves for the
    inverse. Note that inverse of a (nonsingular)lower triangular matrix C is upper triangular
    and vice versa.

    .. math::

        L = (C C^T)

    where :math:`C` is a lower triangular matrix from the Cholesky decomposition of :math:`L`.
    Then, we solve two triangular systems of equations:

    .. math::

        C^{-1}C = I

        L^{-1} = (C C^T)^{-1} = (C^T)^{-1} C^{-1}

    so that we can solve for :math:`L^{-1}` by multiplying both sides by :math:`C^T` and
    solving it as an upper triangular system,

    .. math::

        C^TL^{-1} = C^{-1}

    The units of the inverse of the coil inductance matrix are 1/H, where H = henries.

    Args:
        gammas (array, shape (m,n,3)):
            Array of coil positions for all m coils.
        gammadashs (array, shape (m,n,3)):
            Array of coil tangent vectors for all m coils.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
        regularizations (array, shape (m,)):
            Array of regularizations coming from finite cross-section for all m coils. The choices
            for each coil are regularization_circ and regularization_rect, although each coil can
            have different size and shape cross-sections in this list of regularization terms.

    Returns:
        array (shape (m,m)): Array of inverse of the coil inductance matrix.
    """
    # Lij is symmetric positive definite so has a cholesky decomposition
    C = jnp.linalg.cholesky(
        _coil_coil_inductances_pure(gammas, gammadashs, downsample, regularizations)
    )
    inv_C = jscp.linalg.solve_triangular(C, jnp.eye(C.shape[0]), lower=True)
    inv_L = jscp.linalg.solve_triangular(C.T, inv_C, lower=False)
    return inv_L


def _induced_currents_pure(
    gammas_targets,
    gammadashs_targets,
    gammas_sources,
    gammadashs_sources,
    currents_sources,
    downsample,
    regularizations,
):
    r"""
    Pure function for computing the induced currents in a set of m passive coils with n quadrature points
    due to a set of m' source coils with n' quadrature points (and themselves).

    .. math::
        I = -L^{-1} \Psi

    where :math:`L` is the coil inductance matrix, :math:`\Psi` is the net flux through
    the passive coils due to the source coils,
    and :math:`I` is the induced currents in the passive coils.
    The units of the induced currents are Amperes.

    Args:
        gammas_targets (array, shape (m,n,3)):
            Array of passive coil positions for all m coils.
        gammadashs_targets (array, shape (m,n,3)):
            Array of passive coil tangent vectors for all m coils.
        gammas_sources (array, shape (m',n',3)):
            Array of source coil positions for all m' coils.
        gammadashs_sources (array, shape (m',n',3)):
            Array of source coil tangent vectors for all m' coils.
        currents_sources (array, shape (m',)):
            Array of source coil currents.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
        regularizations (array, shape (m,)):
            Array of regularizations coming from finite cross-section for all m coils. The choices
            for each coil are regularization_circ and regularization_rect, although each coil can
            have different size and shape cross-sections in this list of regularization terms.

    Returns:
        array (shape (m,)): Array of induced currents.
    """
    return -_coil_coil_inductances_inv_pure(
        gammas_targets, gammadashs_targets, downsample, regularizations
    ) @ _net_fluxes_pure(
        gammas_targets,
        gammadashs_targets,
        gammas_sources,
        gammadashs_sources,
        currents_sources,
        downsample,
    )


def b2energy_pure(gammas, gammadashs, currents, downsample, regularizations):
    r"""
    Pure function for evaluating the total vacuum magnetic field energy from a set of m coils
    with n quadrature points each.
    The function is

     .. math::
        J = \frac{1}{2}\sum_{i,j}I_iL_{ij}I_j

    where :math:`L_{ij}` is the coil inductance matrix (positive definite),
    and :math:`I_i` is the current in the ith coil.
    The units of the objective function are MJ (megajoules).

    Args:
        gammas (array, shape (m,n,3)):
            Array of coil positions for all m coils.
        gammadashs (array, shape (m,n,3)):
            Array of coil tangent vectors for all m coils.
        currents (array, shape (m,)):
            Array of coil current for all m coils.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
        regularizations (array, shape (m,)):
            Array of regularizations coming from finite cross-section for all m coils.

    Returns:
        float: Value of the objective function in MJ (megajoules).
    """
    Ii_Ij = currents[:, None] * currents[None, :]
    Lij = _coil_coil_inductances_pure(gammas, gammadashs, downsample, regularizations)
    U = 0.5 * (jnp.sum(Ii_Ij * Lij))
    return U / 1e6  # Convert from Joules to MJ


class B2Energy(Optimizable):
    r"""
    Optimizable class for minimizing the total vacuum magnetic field energy from a set of m coils.

    The function is

     .. math::
        J = \frac{1}{2}\sum_{i,j}I_i L_{ij} I_j

    where :math:`L_{ij}` is the coil inductance matrix (positive definite),
    and :math:`I_i` is the current in the ith coil.
    The units of the objective function are MJ (megajoules).

    Args:
        target_coils (list of RegularizedCoil, shape (m,)):
            List of coils contributing to the total energy.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
    """

    def __init__(self, target_coils, downsample=1):
        self.target_coils = target_coils
        self.downsample = downsample
        if not isinstance(self.target_coils[0], RegularizedCoil):
            raise ValueError("B2Energy can only be used with RegularizedCoil objects")
        _check_quadpoints_consistency(self.target_coils, "target_coils")
        _check_downsample(self.target_coils, downsample, "target_coils")
        self.regularizations = _as_jax_float64(
            [c.regularization for c in self.target_coils]
        )
        self.J_jax = _B2ENERGY_JAX
        self.dJ_jax = _B2ENERGY_GRAD
        self._target_state_cache = _CoilStateGroupCache(self.target_coils)
        self._cached_jax_args = None

        super().__init__(depends_on=target_coils)

    def recompute_bell(self, parent=None):
        _invalidate_objective_state(self, parent, self._target_state_cache)

    def _J_args(self):
        def build_args():
            gammas, gammadashs, currents = self._target_state_cache.arrays()
            return gammas, gammadashs, currents, self.downsample

        return _cached_objective_args(self, build_args)

    def J(self):
        r"""Evaluate the B^2 energy objective.

        Returns:
            float: The total vacuum magnetic field energy
                :math:`J = \frac{1}{2}\sum_{i,j} I_i L_{ij} I_j` in MJ.
        """
        gammas, gammadashs, currents, downsample = self._J_args()
        return self.J_jax(
            gammas, gammadashs, currents, downsample, self.regularizations
        )

    @derivative_dec
    def dJ(self):
        r"""Compute the derivative of the B^2 energy objective with respect to
        all optimizable degrees of freedom (coil geometry and currents).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
        gammas, gammadashs, currents, downsample = self._J_args()
        dJ_dgammas, dJ_dgammadashs, dJ_dcurrents = self.dJ_jax(
            gammas, gammadashs, currents, downsample, self.regularizations
        )
        return _assemble_curve_current_derivative(
            self.target_coils,
            dgamma=dJ_dgammas,
            dgammadash=dJ_dgammadashs,
            dcurrent=dJ_dcurrents,
        )

    return_fn_map = {"J": J, "dJ": dJ}


def _net_fluxes_pure(
    gammas_targets,
    gammadashs_targets,
    gammas_sources,
    gammadashs_sources,
    currents_sources,
    downsample,
):
    r"""
    This function computes the total magnetic flux passing through a set of coils
    due to the magnetic field generated by another set of coils. The flux is calculated
    using the line integral of the vector potential along the coil paths.

    math::
        \Psi = \sum_i \int_{C_i} A_{ext}\cdot d\ell_i / L_i

    where :math:`A_{ext}` is the vector potential of an external magnetic field,
    evaluated along the quadpoints along the curve,
    :math:`L_i` is the total length of the ith coil, and :math:`\ell_i` is arclength
    along the ith coil.

    Note that the first set of coils is assumed to all have the same number of quadrature
    points for the purposes of jit speed. Same with the second set of coils, although
    the number of points does not have to be the same between the two sets.

    The units of the objective function are Weber.

    Args:
        gammas_targets (array, shape (m,n,3)):
            Position vectors for the coils receiving flux.
        gammadashs_targets (array, shape (m,n,3)):
            Tangent vectors for the coils receiving flux.
        gammas_sources (array, shape (m',n',3)):
            Position vectors for the coils generating flux.
        gammadashs_sources (array, shape (m',n',3)):
            Tangent vectors for the coils generating flux.
        currents_sources (array, shape (m',)):
            Current values for the coils generating flux.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.

    Returns:
        array (shape (m,)):
            Net magnetic flux through each coil in the first set.
    """
    (
        gammas_targets,
        gammadashs_targets,
        gammas_sources,
        gammadashs_sources,
        _,
        currents_sources,
    ) = _prepare_target_source_inputs_pure(
        gammas_targets,
        gammadashs_targets,
        gammas_sources,
        gammadashs_sources,
        jnp.zeros(len(gammas_targets)),
        currents_sources,
        downsample,
    )
    rij_norm = jnp.linalg.norm(
        gammas_targets[:, :, None, None, :] - gammas_sources[None, None, :, :, :],
        axis=-1,
    )
    # sum over the currents, and sum over the biot savart integral
    A_ext = (
        jnp.sum(
            currents_sources[None, None, :, None]
            * jnp.sum(
                gammadashs_sources[None, None, :, :, :] / rij_norm[:, :, :, :, None],
                axis=-2,
            ),
            axis=-2,
        )
        / jnp.shape(gammadashs_sources)[1]
    )
    # Now sum over all the coil loops
    return (
        1e-7
        * jnp.sum(jnp.sum(A_ext * gammadashs_targets, axis=-1), axis=-1)
        / jnp.shape(gammadashs_targets)[1]
    )


def net_ext_fluxes_pure(gammadash, A_ext, downsample):
    r"""
    Calculate the net magnetic flux through a coil with n quadrature points
    due to an external vector potential evaluated at those points.

    math::
        \Psi = \int A_{ext}\cdot d\ell / L

    where :math:`A_{ext}` is the vector potential of an external magnetic field,
    evaluated along the quadpoints along the curve,
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.

    This function computes the total magnetic flux passing through a coil due to
    an external magnetic field represented by its vector potential. The flux is
    calculated using the line integral of the vector potential along the coil path.

    The units of the objective function are Weber.

    Args:
        gammadash (array, shape (n,3)):
            Tangent vectors along the coil.
        A_ext (array, shape (n,3)):
            External vector potential evaluated at coil points.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.

    Returns:
        float: Net magnetic flux through the coil due to the external field.
    """
    # Downsample if desired
    gammadash = gammadash[::downsample, :]
    A_ext = A_ext[::downsample, :]
    # Dot the vectors (sum over last axis), then sum over the quadpoints
    return (
        jnp.sum(jnp.sum(A_ext * gammadash, axis=-1), axis=-1) / jnp.shape(gammadash)[0]
    )


class NetFluxes(Optimizable):
    r"""
    Optimizable class for minimizing the total net flux from m coils
    through a single coil with n quadrature points.

    The function is

     .. math::
        \Psi = \int A_{ext}\cdot d\ell / L

    where :math:`A_{ext}` is the vector potential of an external magnetic field,
    evaluated along the quadpoints along the curve,
    L is the total length of the coil, and :math:`\ell` is arclength along the coil.

    The units of the objective function are Weber.

    Args:
        target_coil (Coil): Coil whose net flux is being computed.
        source_coils (list of Coil, shape (m,)):
            List of coils to use for computing the net flux.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
    """

    def __init__(self, target_coil, source_coils, downsample=1):
        if not isinstance(source_coils, list):
            source_coils = [source_coils]
        self.target_coil = target_coil
        self.source_coils = [c for c in source_coils if c not in [target_coil]]
        if len(self.source_coils) == 0:
            raise ValueError(
                "source_coils must contain at least one coil not in target_coil."
            )
        self.downsample = downsample
        _check_downsample([self.target_coil], downsample, "target_coil")
        _check_quadpoints_consistency(self.source_coils, "source_coils")
        _check_downsample(self.source_coils, downsample, "source_coils")
        self.biotsavart = BiotSavart(self.source_coils)
        self.J_jax = _NET_EXT_FLUX_JAX
        self.dJ_jax = _NET_EXT_FLUX_GRAD

        super().__init__(depends_on=[target_coil] + source_coils)

    def J(self):
        r"""Evaluate the net flux objective.

        Computes :math:`\Psi = \int A_{ext} \cdot d\ell / L` using the BiotSavart
        vector potential from the source coils evaluated at the target coil quadrature points.

        Returns:
            float: Net magnetic flux through the target coil in Weber.
        """
        gamma = self.target_coil.curve.gamma()
        self.biotsavart.set_points(np.array(gamma[:: self.downsample, :]))
        return self.J_jax(
            self.target_coil.curve.gammadash()[:: self.downsample],
            self.biotsavart.A(),
            1,
        )

    @derivative_dec
    def dJ(self):
        r"""Compute the derivative of the net flux objective with respect to
        all optimizable degrees of freedom (target coil geometry and source coil
        geometry/currents).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
        gamma = self.target_coil.curve.gamma()
        self.biotsavart.set_points(gamma)
        dJ_dgammadash, dJ_dA = self.dJ_jax(
            self.target_coil.curve.gammadash(),
            self.biotsavart.A(),
            self.downsample,
        )
        dA_dX = self.biotsavart.dA_by_dX()
        dJ_dX = np.einsum("ij,ikj->ik", dJ_dA, dA_dX)
        A_vjp = self.biotsavart.A_vjp(dJ_dA)

        dJ = (
            self.target_coil.curve.dgamma_by_dcoeff_vjp(dJ_dX)
            + self.target_coil.curve.dgammadash_by_dcoeff_vjp(dJ_dgammadash)
            + A_vjp
        )
        return dJ

    return_fn_map = {"J": J, "dJ": dJ}


def squared_mean_force_pure(
    gammas_targets,
    gammas_sources,
    gammadashs_targets,
    gammadashs_sources,
    currents_targets,
    currents_sources,
    downsample,
    eps=1e-10,
    gammas_sources_fine=None,
    gammadashs_sources_fine=None,
    currents_sources_fine=None,
):
    r"""
    Compute the squared mean force on a set of m coils with n quadrature points,
    due to themselves and another set of source coils.

    The objective function is

    .. math:
        J = \sum_i \left(\frac{\int \frac{d\vec{F}_i}{d\ell_i} d\ell_i}{L_i}\right)^2

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length,
    in units of MN/m. The units of the squared mean force are therefore (MN/m)^2.
    :math:`L_i` is the total coil length,
    and :math:`\ell_i` is arclength along the ith coil. The units of the objective function are (MN/m)^2, where MN = meganewtons.

    Source coils may be split into coarse and fine groups (with potentially different quadrature
    counts). The fine sources are downsampled to match the coarse resolution when used.
    All coils within each group are assumed to have the same number of quadrature points.

    Args:
        gammas_targets (array, shape (m,n,3)):
            Position vectors for the coils receiving force.
        gammas_sources (array, shape (m',n',3)):
            Position vectors for the coarse-resolution source coils generating force.
        gammadashs_targets (array, shape (m,n,3)):
            Tangent vectors for the coils receiving force.
        gammadashs_sources (array, shape (m',n',3)):
            Tangent vectors for the coarse-resolution source coils.
        currents_targets (array, shape (m,)):
            Currents for the coils receiving force.
        currents_sources (array, shape (m',)):
            Currents for the coarse-resolution source coils.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
        eps (float): Small constant to avoid division by zero for force between coil_i and itself.
        gammas_sources_fine (array, shape (m'',n'',3), optional):
            Position vectors for fine-resolution source coils. Default: None (no fine sources).
        gammadashs_sources_fine (array, shape (m'',n'',3), optional):
            Tangent vectors for fine-resolution source coils. Default: None.
        currents_sources_fine (array, shape (m'',), optional):
            Currents for fine-resolution source coils. Default: None.
    Returns:
        float: The squared mean force.
    """
    (
        gammas_targets,
        gammadashs_targets,
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
    ) = _prepare_target_source_inputs_pure(
        gammas_targets,
        gammadashs_targets,
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
        downsample,
    )
    # Prepare fine sources if provided (list or array)
    if gammas_sources_fine is None:
        _has_fine = False
    elif isinstance(gammas_sources_fine, (list, tuple)):
        _has_fine = len(gammas_sources_fine) > 0
    else:
        _has_fine = gammas_sources_fine.shape[0] > 0
    if _has_fine:
        if isinstance(gammas_sources_fine, (list, tuple)):
            gammas_sources_fine = jnp.stack(gammas_sources_fine)[:, ::downsample, :]
            gammadashs_sources_fine = jnp.stack(gammadashs_sources_fine)[
                :, ::downsample, :
            ]
        else:
            gammas_sources_fine = gammas_sources_fine[:, ::downsample, :]
            gammadashs_sources_fine = gammadashs_sources_fine[:, ::downsample, :]
        currents_sources_fine = _as_jax_float64(currents_sources_fine)
    else:
        gammas_sources_fine = gammadashs_sources_fine = currents_sources_fine = None

    n1 = gammas_targets.shape[0]
    npts1 = gammas_targets.shape[1]

    # Precompute tangents and norms
    gammadash_norms = jnp.linalg.norm(gammadashs_targets, axis=-1)[:, :, None]
    tangents = gammadashs_targets / gammadash_norms

    # Use empty arrays for fine when not provided
    if gammas_sources_fine is None:
        gammas_sources_fine, gammadashs_sources_fine, currents_sources_fine = (
            _empty_source_fine_arrays()
        )

    def mean_force_group1(i, gamma_i, tangent_i, gammadash_norm_i, current_i):
        def B_at_pt(pt):
            return _mutual_B_field_at_point_pure(
                i,
                pt,
                gammas_targets,
                gammadashs_targets,
                currents_targets,
                gammas_sources,
                gammadashs_sources,
                currents_sources,
                gammas_sources_fine,
                gammadashs_sources_fine,
                currents_sources_fine,
                eps,
            )

        B_mutual = vmap(B_at_pt)(gamma_i)
        force_density = _lorentz_force_density_pure(tangent_i, current_i, B_mutual)
        return jnp.sum(force_density * gammadash_norm_i, axis=0) / npts1

    mean_forces = vmap(mean_force_group1, in_axes=(0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas_targets, tangents, gammadash_norms, currents_targets
    )
    # already multiplied by (mu_0/(4*pi)) in _mutual_B_field_at_point_pure,
    # which gives a factor of (mu_0/(4*pi))^2 = 1e-14
    # Then convert from (N/m)^2 to (MN/m)^2 by dividing by (1e6)^2 = 1e12
    mean_forces_squared = jnp.sum(jnp.linalg.norm(mean_forces, axis=-1) ** 2)
    return mean_forces_squared * 1e-12


class SquaredMeanForce(Optimizable):
    r"""
    Optimizable class to minimize the (net (integrated) Lorentz force per unit length)^2 on a set of m coils
    from themselves and another set of m' coils.

    The objective function is

    .. math:
        J = \sum_i \left(\frac{\int \frac{d\vec{F}_i}{d\ell_i} d\ell_i}{L_i}\right)^2

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length,
    in units of MN/m. The units of the squared mean force are therefore (MN/m)^2.
    :math:`L_i` is the total coil length,
    and :math:`\ell_i` is arclength along the ith coil. The units of the objective function are (MN/m)^2, where MN = meganewtons.

    This class assumes there are two (or three) distinct lists of coils,
    which may have different finite-build parameters and/or different numbers of quadrature points.
    In order to avoid buildup of optimizable
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. This is also useful for optimizing passive coils,
    which require a modified Jacobian calculation. Within each list of coils,
    all coils must have the same number of quadrature points. The source_coils_coarse and source_coils_fine lists
    allows one to optimize e.g. the force on target_coils from a set of dipole coils
    (with barely any quadrature points) and a set of TF coils (with many quadrature points).

    Args:
        target_coils (list of Coil or RegularizedCoil, shape (m,)):
            List of coils to use for computing SquaredMeanForce.
        source_coils_coarse (list of Coil or RegularizedCoil, shape (m',)):
            Coarse-resolution source coils that provide forces on the target_coils.
            Forces are not computed on the source_coils.
        source_coils_fine (list of Coil or RegularizedCoil, optional):
            Fine-resolution source coils, used in addition to coarse. Default: []. This functionality
            is provided for when there are two sets of source coils with very different numbers of
            quadrature points. This occurs e.g. when optimizing TF coils and dipole coils.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
    """

    def __init__(
        self,
        target_coils,
        source_coils_coarse,
        source_coils_fine=None,
        downsample: int = 1,
    ):
        if not isinstance(target_coils, list):
            target_coils = [target_coils]
        if not isinstance(source_coils_coarse, list):
            source_coils_coarse = [source_coils_coarse]
        if source_coils_fine is None:
            source_coils_fine = []
        elif not isinstance(source_coils_fine, list):
            source_coils_fine = [source_coils_fine]
        self.target_coils = target_coils
        self.source_coils_coarse = [
            c for c in source_coils_coarse if c not in target_coils
        ]
        self.source_coils_fine = [c for c in source_coils_fine if c not in target_coils]
        if len(self.source_coils_coarse) == 0 and len(self.source_coils_fine) == 0:
            raise ValueError(
                "source_coils_coarse and source_coils_fine must together contain at least one coil not in target_coils."
            )
        self.source_coils_fine = [
            c for c in self.source_coils_fine if c not in self.source_coils_coarse
        ]
        self.source_coils = self.source_coils_coarse + self.source_coils_fine

        # Check that the coils in each list of coils (target_coils, source_coils_coarse, source_coils_fine)
        # all have the same number of quadrature points and that the downsample factor is a valid
        # multiple of the number of quadrature points.
        _check_quadpoints_consistency(self.target_coils, "target_coils")
        if len(self.source_coils_coarse) > 0:
            _check_quadpoints_consistency(
                self.source_coils_coarse, "source_coils_coarse"
            )
        if len(self.source_coils_fine) > 0:
            _check_quadpoints_consistency(self.source_coils_fine, "source_coils_fine")
        self.downsample = downsample
        _check_downsample(self.target_coils, downsample, "target_coils")
        if len(self.source_coils_coarse) > 0:
            _check_downsample(
                self.source_coils_coarse, downsample, "source_coils_coarse"
            )
        if len(self.source_coils_fine) > 0:
            _check_downsample(self.source_coils_fine, downsample, "source_coils_fine")

        self.J_jax = _SQUARED_MEAN_FORCE_JAX
        self.dJ_jax = _SQUARED_MEAN_FORCE_GRAD
        self._target_state_cache = _CoilStateGroupCache(self.target_coils)
        self._source_coarse_state_cache = _CoilStateGroupCache(self.source_coils_coarse)
        self._source_fine_state_cache = _CoilStateGroupCache(self.source_coils_fine)
        self._cached_jax_args = None

        super().__init__(depends_on=(target_coils + self.source_coils))

    def recompute_bell(self, parent=None):
        _invalidate_objective_state(
            self,
            parent,
            self._target_state_cache,
            self._source_coarse_state_cache,
            self._source_fine_state_cache,
        )

    def _J_args(self):
        """Build arguments for evaluation of J and dJ."""

        def build_args():
            gammas_targets, gammadashs_targets, currents_targets = (
                self._target_state_cache.arrays()
            )
            gammas_coarse, gammadashs_coarse, currents_coarse = (
                self._source_coarse_state_cache.arrays()
            )
            gammas_fine, gammadashs_fine, currents_fine = (
                self._source_fine_state_cache.arrays()
            )
            return (
                gammas_targets,
                gammas_coarse,
                gammadashs_targets,
                gammadashs_coarse,
                currents_targets,
                currents_coarse,
                gammas_fine,
                gammadashs_fine,
                currents_fine,
                self.downsample,
            )

        return _cached_objective_args(self, build_args)

    def J(self):
        r"""Evaluate the squared mean force objective."""
        return self.J_jax(*self._J_args())

    @derivative_dec
    def dJ(self):
        r"""Compute the derivative of the squared mean force objective with respect to
        all optimizable degrees of freedom (coil geometry and currents for both
        target_coils and source_coils_coarse and source_coils_fine if passed).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
        (
            dJ_dgamma_targets,
            dJ_dgamma_coarse,
            dJ_dgammadash_targets,
            dJ_dgammadash_coarse,
            dJ_dcurrent_targets,
            dJ_dcurrent_coarse,
            dJ_dgamma_fine,
            dJ_dgammadash_fine,
            dJ_dcurrent_fine,
        ) = self.dJ_jax(*self._J_args())
        dJ = _assemble_curve_current_derivative(
            self.target_coils,
            dgamma=dJ_dgamma_targets,
            dgammadash=dJ_dgammadash_targets,
            dcurrent=dJ_dcurrent_targets,
        )
        if len(self.source_coils_coarse) > 0:
            dJ += _assemble_curve_current_derivative(
                self.source_coils_coarse,
                dgamma=dJ_dgamma_coarse,
                dgammadash=dJ_dgammadash_coarse,
                dcurrent=dJ_dcurrent_coarse,
            )
        if len(self.source_coils_fine) > 0:
            dJ += _assemble_curve_current_derivative(
                self.source_coils_fine,
                dgamma=dJ_dgamma_fine,
                dgammadash=dJ_dgammadash_fine,
                dcurrent=dJ_dcurrent_fine,
            )
        return dJ

    return_fn_map = {"J": J, "dJ": dJ}


def lp_force_pure(
    gammas_targets,
    gammas_sources,
    gammadashs_targets,
    gammadashs_sources,
    gammadashdashs_targets,
    quadpoints,
    currents_targets,
    currents_sources,
    regularizations,
    p,
    threshold,
    downsample,
    eps=1e-10,
    gammas_sources_fine=None,
    gammadashs_sources_fine=None,
    currents_sources_fine=None,
):
    r"""
    Computes the Lp force objective by summing over a set of m coils,
    where each coil receives force from all coils (including itself,
    the other m - 1 target coils and the source coils).
    Source coils may be split into coarse and fine groups (with potentially different quadrature
    counts). The fine sources are downsampled to match the coarse resolution when used.
    All coils within each group are assumed to have the same number of quadrature points.

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{F}/d\ell_i| - F_0 , 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length,
    in units of MN/m, where MN = meganewtons.
    The units of the objective function are therefore (MN/m)^p.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`F_0 ` is a threshold force at the ith coil.

    Args:
        gammas_targets (array, shape (m,n,3)):
            Position vectors for the coils receiving force.
        gammas_sources (array, shape (m',n',3)):
            Position vectors for the coarse-resolution source coils generating force.
        gammadashs_targets (array, shape (m,n,3)):
            Tangent vectors for the coils receiving force.
        gammadashs_sources (array, shape (m',n',3)):
            Tangent vectors for the coarse-resolution source coils.
        gammadashdashs_targets (array, shape (m,n,3)):
            Second derivative of tangent vectors for the coils receiving force.
        quadpoints (array, shape (m,n)):
            Quadrature points for target coils. Since target coils are required to have
            matching quadrature, the first entry is used.
        currents_targets (array, shape (m,)):
            Currents for the coils receiving force.
        currents_sources (array, shape (m',)):
            Currents for the coarse-resolution source coils.
        regularizations (array, shape (m,)):
            Array of regularizations coming from finite cross-section for all coils. The choices
            for each coil are regularization_circ and regularization_rect, although each coil can
            have different size and shape cross-sections in this list of regularization terms.
        p (float):
            Exponent for the Lp force objective.
        threshold (float):
            Threshold force per unit length in units of MN/m (meganewtons per meter).
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
        eps (float): Small constant to avoid division by zero for force between coil_i and itself.
        gammas_sources_fine (array, shape (m'',n'',3), optional):
            Position vectors for fine-resolution source coils. Default: None (no fine sources).
        gammadashs_sources_fine (array, shape (m'',n'',3), optional):
            Tangent vectors for fine-resolution source coils. Default: None.
        currents_sources_fine (array, shape (m'',), optional):
            Currents for fine-resolution source coils. Default: None.
    Returns:
        float: The Lp force objective.
    """
    (
        gammas_targets,
        gammadashs_targets,
        gammadashdashs_targets,
        quadpoints,
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
        regularizations,
    ) = _prepare_regularized_target_source_inputs_pure(
        gammas_targets,
        gammadashs_targets,
        gammadashdashs_targets,
        quadpoints,
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
        regularizations,
        downsample,
    )
    if (
        gammas_sources_fine is None
        or gammadashs_sources_fine is None
        or currents_sources_fine is None
    ):
        gammas_sources_fine, gammadashs_sources_fine, currents_sources_fine = (
            _empty_source_fine_arrays()
        )
    elif hasattr(gammas_sources_fine, "shape") and gammas_sources_fine.shape[0] > 0:
        gammas_sources_fine = gammas_sources_fine[:, ::downsample, :]
        gammadashs_sources_fine = gammadashs_sources_fine[:, ::downsample, :]
        currents_sources_fine = _as_jax_float64(currents_sources_fine)

    n1 = gammas_targets.shape[0]
    npts1 = gammas_targets.shape[1]

    # Precompute tangents and norms
    gammadash_norms = jnp.linalg.norm(gammadashs_targets, axis=-1)[:, :, None]
    tangents = gammadashs_targets / gammadash_norms

    # Precompute B_self for each coil
    B_self = vmap(B_regularized_pure, in_axes=(0, 0, 0, None, 0, 0))(
        gammas_targets,
        gammadashs_targets,
        gammadashdashs_targets,
        quadpoints,
        currents_targets,
        regularizations,
    )

    def per_coil_obj_group1(i, gamma_i, tangent_i, B_self_i, current_i):
        B_mutual = vmap(
            lambda pt: _mutual_B_field_at_point_pure(
                i,
                pt,
                gammas_targets,
                gammadashs_targets,
                currents_targets,
                gammas_sources,
                gammadashs_sources,
                currents_sources,
                gammas_sources_fine,
                gammadashs_sources_fine,
                currents_sources_fine,
                eps,
            )
        )(gamma_i)
        F = _lorentz_force_density_pure(tangent_i, current_i, B_mutual + B_self_i)
        # Force per unit length is in N/m, convert to MN/m
        return jnp.linalg.norm(F, axis=-1) / 1e6

    obj1 = vmap(per_coil_obj_group1, in_axes=(0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas_targets, tangents, B_self, currents_targets
    )

    # obj1 is now in MN/m, threshold is in MN/m
    return (
        jnp.sum(
            jnp.sum(jnp.maximum(obj1 - threshold, 0) ** p * gammadash_norms[:, :, 0])
        )
        / npts1
    ) * (1.0 / p)


class LpCurveForce(Optimizable):
    r"""
    Optimizable class to minimize the total Lp-Lorentz force density (force per unit length) integrated.
    Force density on a coil is computed on each coil in a set of m target coils, using the self-force from
    the coil itself, the force from the other m - 1 target coils and the force from a set of m' source coils.
    If source_coils_coarse and target_coils have coils in common, they are removed during initialization of this class,
    to avoid double counting forces. A typical use case has the target_coils as the unique base_coils
    in a stellarator optimization, and source_coils_coarse are all the coils after applying symmetries.
    Typical initialization is LpCurveForce(base_coils, coils).

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{F}/d\ell_i| - F_0 , 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{F}_i}{d\ell_i}` is the Lorentz force per unit length,
    in units of MN/m, where MN = meganewtons. The units of the objective function are therefore (MN/m)^p.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`F_0 ` is a threshold force at the ith coil.

    This class assumes there are two (or three) distinct lists of coils,
    which may have different finite-build parameters and/or different numbers of quadrature points.
    In order to avoid buildup of optimizable
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. Within each list of coils,
    all coils must have the same number of quadrature points. The source_coils_coarse and source_coils_fine lists
    allows one to optimize e.g. the torque on target_coils from a set of dipole coils
    (with barely any quadrature points) and a set of TF coils (with many quadrature points).

    Args:
        target_coils (list of RegularizedCoil, shape (m,)):
            List of coils on which the LpCurveForce is computed.
        source_coils_coarse (list of Coil or RegularizedCoil, shape (m',)):
            Coarse-resolution source coils that provide forces on the target_coils.
            Forces are not computed on the source_coils.
        source_coils_fine (list of Coil or RegularizedCoil, optional):
            Fine-resolution source coils, used in addition to coarse. Default: []. This functionality
            is provided for when there are two sets of source coils with very different numbers of
            quadrature points. This occurs e.g. when optimizing TF coils and dipole coils.
        p (float): Power of the objective function.
        threshold (float): Threshold force per unit length in units of MN/m (meganewtons per meter).
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
    """

    def __init__(
        self,
        target_coils,
        source_coils_coarse,
        source_coils_fine=None,
        p: float = 2.0,
        threshold: float = 0.0,
        downsample: int = 1,
    ):
        if not isinstance(target_coils, list):
            target_coils = [target_coils]
        if not isinstance(source_coils_coarse, list):
            source_coils_coarse = [source_coils_coarse]
        if source_coils_fine is None:
            source_coils_fine = []
        elif not isinstance(source_coils_fine, list):
            source_coils_fine = [source_coils_fine]
        if not isinstance(target_coils[0], RegularizedCoil):
            raise ValueError(
                "LpCurveForce can only be used with RegularizedCoil objects"
            )
        self.regularizations = _as_jax_float64([c.regularization for c in target_coils])
        self.target_coils = target_coils
        self.source_coils_coarse = [
            c for c in source_coils_coarse if c not in target_coils
        ]
        self.source_coils_fine = [c for c in source_coils_fine if c not in target_coils]
        if len(self.source_coils_coarse) == 0 and len(self.source_coils_fine) == 0:
            raise ValueError(
                "source_coils_coarse and source_coils_fine must together contain at least one coil not in target_coils."
            )
        self.source_coils_fine = [
            c for c in self.source_coils_fine if c not in self.source_coils_coarse
        ]
        self.source_coils = self.source_coils_coarse + self.source_coils_fine

        # Check that the coils in each list of coils (target_coils, source_coils_coarse, source_coils_fine)
        # all have the same number of quadrature points and that the downsample factor is a valid
        # multiple of the number of quadrature points.
        _check_quadpoints_consistency(self.target_coils, "target_coils")
        if len(self.source_coils_coarse) > 0:
            _check_quadpoints_consistency(
                self.source_coils_coarse, "source_coils_coarse"
            )
        if len(self.source_coils_fine) > 0:
            _check_quadpoints_consistency(self.source_coils_fine, "source_coils_fine")
        self.quadpoints = _as_jax_float64([c.curve.quadpoints for c in target_coils])
        self.p = p
        self.threshold = threshold
        self.downsample = downsample
        _check_downsample(self.target_coils, downsample, "target_coils")
        if len(self.source_coils_coarse) > 0:
            _check_downsample(
                self.source_coils_coarse, downsample, "source_coils_coarse"
            )
        if len(self.source_coils_fine) > 0:
            _check_downsample(self.source_coils_fine, downsample, "source_coils_fine")

        self.J_jax = _LP_FORCE_JAX
        self.dJ_jax = _LP_FORCE_GRAD
        self._target_state_cache = _CoilStateGroupCache(
            self.target_coils,
            include_gammadashdash=True,
        )
        self._source_coarse_state_cache = _CoilStateGroupCache(self.source_coils_coarse)
        self._source_fine_state_cache = _CoilStateGroupCache(self.source_coils_fine)
        self._cached_jax_args = None

        super().__init__(depends_on=(target_coils + self.source_coils))

    def recompute_bell(self, parent=None):
        _invalidate_objective_state(
            self,
            parent,
            self._target_state_cache,
            self._source_coarse_state_cache,
            self._source_fine_state_cache,
        )

    def _J_args(self):
        """Build arguments for evaluation of J and dJ."""

        def build_args():
            (
                gammas_targets,
                gammadashs_targets,
                gammadashdashs_targets,
                currents_targets,
            ) = self._target_state_cache.arrays()
            gammas_coarse, gammadashs_coarse, currents_coarse = (
                self._source_coarse_state_cache.arrays()
            )
            gammas_fine, gammadashs_fine, currents_fine = (
                self._source_fine_state_cache.arrays()
            )
            return (
                gammas_targets,
                gammas_coarse,
                gammadashs_targets,
                gammadashs_coarse,
                gammadashdashs_targets,
                currents_targets,
                currents_coarse,
                gammas_fine,
                gammadashs_fine,
                currents_fine,
                self.quadpoints,
                self.regularizations,
                _as_jax_float64(self.p),
                _as_jax_float64(self.threshold),
                self.downsample,
            )

        return _cached_objective_args(self, build_args)

    def J(self):
        r"""Evaluate the Lp curve force objective."""
        return self.J_jax(*self._J_args())

    @derivative_dec
    def dJ(self):
        r"""Compute the derivative of the Lp curve force objective with respect to
        all optimizable degrees of freedom (coil geometry and currents for both
        target_coils and source_coils_coarse and source_coils_fine if passed).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
        (
            dJ_dgamma_targets,
            dJ_dgamma_coarse,
            dJ_dgammadash_targets,
            dJ_dgammadash_coarse,
            dJ_dgammadashdash_targets,
            dJ_dcurrent_targets,
            dJ_dcurrent_coarse,
            dJ_dgamma_fine,
            dJ_dgammadash_fine,
            dJ_dcurrent_fine,
        ) = self.dJ_jax(*self._J_args())
        dJ = _assemble_curve_current_derivative(
            self.target_coils,
            dgamma=dJ_dgamma_targets,
            dgammadash=dJ_dgammadash_targets,
            dgammadashdash=dJ_dgammadashdash_targets,
            dcurrent=dJ_dcurrent_targets,
        )
        if len(self.source_coils_coarse) > 0:
            dJ += _assemble_curve_current_derivative(
                self.source_coils_coarse,
                dgamma=dJ_dgamma_coarse,
                dgammadash=dJ_dgammadash_coarse,
                dcurrent=dJ_dcurrent_coarse,
            )
        if len(self.source_coils_fine) > 0:
            dJ += _assemble_curve_current_derivative(
                self.source_coils_fine,
                dgamma=dJ_dgamma_fine,
                dgammadash=dJ_dgammadash_fine,
                dcurrent=dJ_dcurrent_fine,
            )
        return dJ

    return_fn_map = {"J": J, "dJ": dJ}


def lp_torque_pure(
    gammas_targets,
    gammas_sources,
    gammadashs_targets,
    gammadashs_sources,
    gammadashdashs_targets,
    quadpoints,
    currents_targets,
    currents_sources,
    regularizations,
    p,
    threshold,
    downsample,
    eps=1e-10,
    gammas_sources_fine=None,
    gammadashs_sources_fine=None,
    currents_sources_fine=None,
):
    r"""
    Pure function for computing the Lp torque on a set of m coils with n quadrature points
    from themselves and another set of source coils.

    Source coils may be split into coarse and fine groups (with potentially different quadrature
    counts). The fine sources are downsampled to match the coarse resolution when used.
    All coils within each group are assumed to have the same number of quadrature points.

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{T}/d\ell_i| - T_0 , 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,
    in units of MN, where MN = meganewtons.
    The units of the objective function are therefore (MN)^p.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`T_0 ` is a threshold torque per unit length at the ith coil.

    Args:
        gammas_targets (array, shape (m,n,3)): Array of target coil positions.
        gammas_sources (array, shape (m',n',3)): Array of coarse-resolution source coil positions.
        gammadashs_targets (array, shape (m,n,3)): Array of target coil tangent vectors.
        gammadashs_sources (array, shape (m',n',3)): Array of coarse-resolution source coil tangent vectors.
        gammadashdashs_targets (array, shape (m,n,3)): Array of second derivatives of target coil positions.
        quadpoints (array, shape (m,n)):
            Quadrature points for target coils. Since target coils are required to have
            matching quadrature, the first entry is used.
        currents_targets (array, shape (m,)): Array of target coil currents.
        currents_sources (array, shape (m',)): Array of coarse-resolution source coil currents.
        regularizations (array, shape (m,)):
            Array of regularizations coming from finite cross-section for all m coils. The choices
            for each coil are regularization_circ and regularization_rect, although each coil can
            have different size and shape cross-sections in this list of regularization terms.
        p (float): Power of the objective function.
        threshold (float): Threshold torque per unit length in units of MN (meganewtons).
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
        eps (float): Small constant to avoid division by zero for torque between coil_i and itself.
        gammas_sources_fine (array, shape (m'',n'',3), optional):
            Position vectors for fine-resolution source coils. Default: None (no fine sources).
        gammadashs_sources_fine (array, shape (m'',n'',3), optional):
            Tangent vectors for fine-resolution source coils. Default: None.
        currents_sources_fine (array, shape (m'',), optional):
            Currents for fine-resolution source coils. Default: None.
    Returns:
        float: Value of the objective function.
    """
    from simsopt.geo.curve import centroid_pure

    (
        gammas_targets,
        gammadashs_targets,
        gammadashdashs_targets,
        quadpoints,
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
        regularizations,
    ) = _prepare_regularized_target_source_inputs_pure(
        gammas_targets,
        gammadashs_targets,
        gammadashdashs_targets,
        quadpoints,
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
        regularizations,
        downsample,
    )
    if (
        gammas_sources_fine is None
        or gammadashs_sources_fine is None
        or currents_sources_fine is None
    ):
        gammas_sources_fine, gammadashs_sources_fine, currents_sources_fine = (
            _empty_source_fine_arrays()
        )
    elif hasattr(gammas_sources_fine, "shape") and gammas_sources_fine.shape[0] > 0:
        gammas_sources_fine = gammas_sources_fine[:, ::downsample, :]
        gammadashs_sources_fine = gammadashs_sources_fine[:, ::downsample, :]
        currents_sources_fine = _as_jax_float64(currents_sources_fine)

    centers = vmap(centroid_pure, in_axes=(0, 0))(gammas_targets, gammadashs_targets)

    # Precompute B_self for each coil
    B_self = vmap(B_regularized_pure, in_axes=(0, 0, 0, None, 0, 0))(
        gammas_targets,
        gammadashs_targets,
        gammadashdashs_targets,
        quadpoints,
        currents_targets,
        regularizations,
    )
    gammadash_norms = jnp.linalg.norm(gammadashs_targets, axis=-1)[:, :, None]
    tangents = gammadashs_targets / gammadash_norms

    n1 = gammas_targets.shape[0]
    npts1 = gammas_targets.shape[1]

    def per_coil_obj_group1(i, gamma_i, center_i, tangent_i, B_self_i, current_i):
        def torque_at_point(idx):
            B_mutual = _mutual_B_field_at_point_pure(
                i,
                gamma_i[idx],
                gammas_targets,
                gammadashs_targets,
                currents_targets,
                gammas_sources,
                gammadashs_sources,
                currents_sources,
                gammas_sources_fine,
                gammadashs_sources_fine,
                currents_sources_fine,
                eps,
            )
            F = current_i * jnp.cross(tangent_i[idx], B_mutual + B_self_i[idx])
            tau = jnp.cross(gamma_i[idx] - center_i, F)
            # Torque per unit length is in N, convert to MN
            torque_per_unit_length_N = jnp.linalg.norm(tau)
            return torque_per_unit_length_N / 1e6  # Convert to MN

        return vmap(torque_at_point)(jnp.arange(npts1))

    obj1 = vmap(per_coil_obj_group1, in_axes=(0, 0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas_targets, centers, tangents, B_self, currents_targets
    )

    # obj1 is now in MN, threshold is in MN
    return (
        jnp.sum(
            jnp.sum(jnp.maximum(obj1 - threshold, 0) ** p * gammadash_norms[:, :, 0])
        )
        / npts1
        * (1.0 / p)
    )


class LpCurveTorque(Optimizable):
    r"""
    Optimizable class to minimize the total Lp-Lorentz torque density (torque per unit length) integrated.
    Torque density on a coil is computed on each coil in a set of m target coils, using the self-force from
    the coil itself, the force from the other m - 1 target coils and the force from a set of m' source coils.
    If source_coils and target_coils have coils in common, they are removed during initialization of this class,
    to avoid double counting forces. A typical use case has the target_coils as the unique base_coils
    in a stellarator optimization, and source_coils are all the coils after applying symmetries.
    Typical initialization is LpCurveTorque(base_coils, coils).

    The objective function is

    .. math::
        J = \frac{1}{p}\sum_i\frac{1}{L_i}\left(\int \text{max}(|d\vec{T}/d\ell_i| - T_0 , 0)^p d\ell_i\right)

    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,
    in units of MN, where MN = meganewtons.
    The units of the objective function are therefore (MN)^p.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length,
    and :math:`T_0 ` is a threshold torque per unit length at the ith coil.

    This class assumes there are two (or three) distinct lists of coils,
    which may have different finite-build parameters and/or different numbers of quadrature points.
    In order to avoid buildup of optimizable
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. Within each list of coils,
    all coils must have the same number of quadrature points. The source_coils_coarse and source_coils_fine lists
    allows one to optimize e.g. the torque on target_coils from a set of dipole coils
    (with barely any quadrature points) and a set of TF coils (with many quadrature points).

    Args:
        target_coils (list of RegularizedCoil, shape (m,)): List of coils to use for computing LpCurveTorque.
        source_coils_coarse (list of Coil or RegularizedCoil, shape (m',)):
            Coarse-resolution source coils that provide torques on the target_coils.
            Torques are not computed on the source_coils.
        source_coils_fine (list of Coil or RegularizedCoil, optional):
            Fine-resolution source coils, used in addition to coarse. Default: []. This functionality
            is provided for when there are two sets of source coils with very different numbers of
            quadrature points. This occurs e.g. when optimizing TF coils and dipole coils.
        p (float): Power of the objective function.
        threshold (float): Threshold torque per unit length in units of MN (meganewtons).
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
    """

    def __init__(
        self,
        target_coils,
        source_coils_coarse,
        source_coils_fine=None,
        p: float = 2.0,
        threshold: float = 0.0,
        downsample: int = 1,
    ):
        if not isinstance(target_coils, list):
            target_coils = [target_coils]
        if not isinstance(source_coils_coarse, list):
            source_coils_coarse = [source_coils_coarse]
        if source_coils_fine is None:
            source_coils_fine = []
        elif not isinstance(source_coils_fine, list):
            source_coils_fine = [source_coils_fine]
        if not isinstance(target_coils[0], RegularizedCoil):
            raise ValueError(
                "LpCurveTorque can only be used with RegularizedCoil objects"
            )
        self.regularizations = _as_jax_float64([c.regularization for c in target_coils])
        self.target_coils = target_coils
        self.source_coils_coarse = [
            c for c in source_coils_coarse if c not in target_coils
        ]
        self.source_coils_fine = [c for c in source_coils_fine if c not in target_coils]
        if len(self.source_coils_coarse) == 0 and len(self.source_coils_fine) == 0:
            raise ValueError(
                "source_coils_coarse and source_coils_fine must together contain at least one coil not in target_coils."
            )
        self.source_coils_fine = [
            c for c in self.source_coils_fine if c not in self.source_coils_coarse
        ]
        self.source_coils = self.source_coils_coarse + self.source_coils_fine

        # Check that the coils in each list of coils (target_coils, source_coils_coarse, source_coils_fine)
        # all have the same number of quadrature points and that the downsample factor is a valid
        # multiple of the number of quadrature points.
        _check_quadpoints_consistency(self.target_coils, "target_coils")
        if len(self.source_coils_coarse) > 0:
            _check_quadpoints_consistency(
                self.source_coils_coarse, "source_coils_coarse"
            )
        if len(self.source_coils_fine) > 0:
            _check_quadpoints_consistency(self.source_coils_fine, "source_coils_fine")
        self.quadpoints = _as_jax_float64([c.curve.quadpoints for c in target_coils])
        self.p = p
        self.threshold = threshold
        self.downsample = downsample
        _check_downsample(self.target_coils, downsample, "target_coils")
        if len(self.source_coils_coarse) > 0:
            _check_downsample(
                self.source_coils_coarse, downsample, "source_coils_coarse"
            )
        if len(self.source_coils_fine) > 0:
            _check_downsample(self.source_coils_fine, downsample, "source_coils_fine")

        self.J_jax = _LP_TORQUE_JAX
        self.dJ_jax = _LP_TORQUE_GRAD
        self._target_state_cache = _CoilStateGroupCache(
            self.target_coils,
            include_gammadashdash=True,
        )
        self._source_coarse_state_cache = _CoilStateGroupCache(self.source_coils_coarse)
        self._source_fine_state_cache = _CoilStateGroupCache(self.source_coils_fine)
        self._cached_jax_args = None

        super().__init__(depends_on=(target_coils + self.source_coils))

    def recompute_bell(self, parent=None):
        _invalidate_objective_state(
            self,
            parent,
            self._target_state_cache,
            self._source_coarse_state_cache,
            self._source_fine_state_cache,
        )

    def _J_args(self):
        """Build arguments for evaluation of J and dJ."""

        def build_args():
            (
                gammas_targets,
                gammadashs_targets,
                gammadashdashs_targets,
                currents_targets,
            ) = self._target_state_cache.arrays()
            gammas_coarse, gammadashs_coarse, currents_coarse = (
                self._source_coarse_state_cache.arrays()
            )
            gammas_fine, gammadashs_fine, currents_fine = (
                self._source_fine_state_cache.arrays()
            )
            return (
                gammas_targets,
                gammas_coarse,
                gammadashs_targets,
                gammadashs_coarse,
                gammadashdashs_targets,
                currents_targets,
                currents_coarse,
                gammas_fine,
                gammadashs_fine,
                currents_fine,
                self.quadpoints,
                self.regularizations,
                _as_jax_float64(self.p),
                _as_jax_float64(self.threshold),
                self.downsample,
            )

        return _cached_objective_args(self, build_args)

    def J(self):
        r"""Evaluate the Lp curve torque objective."""
        return self.J_jax(*self._J_args())

    @derivative_dec
    def dJ(self):
        r"""Compute the derivative of the Lp curve torque objective with respect to
        all optimizable degrees of freedom (coil geometry and currents for both
        target_coils and source_coils_coarse and source_coils_fine if passed).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
        (
            dJ_dgamma_targets,
            dJ_dgamma_coarse,
            dJ_dgammadash_targets,
            dJ_dgammadash_coarse,
            dJ_dgammadashdash_targets,
            dJ_dcurrent_targets,
            dJ_dcurrent_coarse,
            dJ_dgamma_fine,
            dJ_dgammadash_fine,
            dJ_dcurrent_fine,
        ) = self.dJ_jax(*self._J_args())
        dJ = _assemble_curve_current_derivative(
            self.target_coils,
            dgamma=dJ_dgamma_targets,
            dgammadash=dJ_dgammadash_targets,
            dgammadashdash=dJ_dgammadashdash_targets,
            dcurrent=dJ_dcurrent_targets,
        )
        if len(self.source_coils_coarse) > 0:
            dJ += _assemble_curve_current_derivative(
                self.source_coils_coarse,
                dgamma=dJ_dgamma_coarse,
                dgammadash=dJ_dgammadash_coarse,
                dcurrent=dJ_dcurrent_coarse,
            )
        if len(self.source_coils_fine) > 0:
            dJ += _assemble_curve_current_derivative(
                self.source_coils_fine,
                dgamma=dJ_dgamma_fine,
                dgammadash=dJ_dgammadash_fine,
                dcurrent=dJ_dcurrent_fine,
            )
        return dJ

    return_fn_map = {"J": J, "dJ": dJ}


def squared_mean_torque(
    gammas_targets,
    gammas_sources,
    gammadashs_targets,
    gammadashs_sources,
    currents_targets,
    currents_sources,
    downsample,
    eps=1e-10,
    gammas_sources_fine=None,
    gammadashs_sources_fine=None,
    currents_sources_fine=None,
):
    r"""
    Compute the squared mean torque on a set of m coils with n quadrature points
    due to themselves and another set of source coils.

    Source coils may be split into coarse and fine groups (with potentially different quadrature
    counts). The fine sources are downsampled to match the coarse resolution when used.
    All coils within each group are assumed to have the same number of quadrature points.

    The objective function is

    .. math:
        J = \sum_i(\frac{\int \frac{d\vec{T}_i}{d\ell_i} d\ell_i}{L_i})^2

    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,
    in units of MN. The units of the squared mean torque are therefore (MN)^2.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length.

    Args:
        gammas_targets (array, shape (m,n,3)): Array of target coil positions.
        gammas_sources (array, shape (m',n',3)): Array of coarse-resolution source coil positions.
        gammadashs_targets (array, shape (m,n,3)): Array of target coil tangent vectors.
        gammadashs_sources (array, shape (m',n',3)): Array of coarse-resolution source coil tangent vectors.
        currents_targets (array, shape (m,)): Array of target coil currents.
        currents_sources (array, shape (m',)): Array of coarse-resolution source coil currents.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.
        eps (float): Small constant to avoid division by zero for torque between coil_i and itself.
        gammas_sources_fine (array, shape (m'',n'',3), optional):
            Position vectors for fine-resolution source coils. Default: None (no fine sources).
        gammadashs_sources_fine (array, shape (m'',n'',3), optional):
            Tangent vectors for fine-resolution source coils. Default: None.
        currents_sources_fine (array, shape (m'',), optional):
            Currents for fine-resolution source coils. Default: None.
    Returns:
        float: Value of the objective function.
    """
    from simsopt.geo.curve import centroid_pure

    (
        gammas_targets,
        gammadashs_targets,
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
    ) = _prepare_target_source_inputs_pure(
        gammas_targets,
        gammadashs_targets,
        gammas_sources,
        gammadashs_sources,
        currents_targets,
        currents_sources,
        downsample,
    )
    if (
        gammas_sources_fine is None
        or gammadashs_sources_fine is None
        or currents_sources_fine is None
    ):
        gammas_sources_fine, gammadashs_sources_fine, currents_sources_fine = (
            _empty_source_fine_arrays()
        )
    elif (
        isinstance(gammas_sources_fine, (list, tuple)) and len(gammas_sources_fine) > 0
    ):
        gammas_sources_fine = jnp.stack(gammas_sources_fine)[:, ::downsample, :]
        gammadashs_sources_fine = jnp.stack(gammadashs_sources_fine)[:, ::downsample, :]
        currents_sources_fine = _as_jax_float64(currents_sources_fine)
    elif hasattr(gammas_sources_fine, "shape") and gammas_sources_fine.shape[0] > 0:
        gammas_sources_fine = gammas_sources_fine[:, ::downsample, :]
        gammadashs_sources_fine = gammadashs_sources_fine[:, ::downsample, :]
        currents_sources_fine = _as_jax_float64(currents_sources_fine)

    n1 = gammas_targets.shape[0]
    npts1 = gammas_targets.shape[1]

    centers = vmap(centroid_pure, in_axes=(0, 0))(gammas_targets, gammadashs_targets)

    def mean_torque_group1(i, gamma_i, gammadash_i, center_i, current_i):
        arclength = jnp.linalg.norm(gammadash_i, axis=-1)
        tangent = gammadash_i / arclength[:, None]
        B_mutual = vmap(
            lambda pt: _mutual_B_field_at_point_pure(
                i,
                pt,
                gammas_targets,
                gammadashs_targets,
                currents_targets,
                gammas_sources,
                gammadashs_sources,
                currents_sources,
                gammas_sources_fine,
                gammadashs_sources_fine,
                currents_sources_fine,
                eps,
            )
        )(gamma_i)
        F = _lorentz_force_density_pure(tangent, current_i, B_mutual)
        torques = jnp.cross(gamma_i - center_i[None, :], F) * arclength[:, None]
        return jnp.sum(torques, axis=0) / npts1

    mean_torques = vmap(mean_torque_group1, in_axes=(0, 0, 0, 0, 0))(
        jnp.arange(n1), gammas_targets, gammadashs_targets, centers, currents_targets
    )
    # already multiplied by (mu_0/(4*pi)) in _mutual_B_field_at_point_pure,
    # which gives a factor of (mu_0/(4*pi))^2 = 1e-14
    # Then convert from (N)^2 to (MN)^2 by dividing by (1e6)^2 = 1e12
    mean_torques_squared = jnp.sum(jnp.linalg.norm(mean_torques, axis=-1) ** 2)
    return mean_torques_squared * 1e-12


class SquaredMeanTorque(Optimizable):
    r"""
    Optimizable class to minimize the (net (integrated) Lorentz torque per unit length)^2 summed
    over a set of m coils due to themselves and another set of m' coils.

    The objective function is

    .. math:
        J = \sum_i(\frac{\int \frac{d\vec{T}_i}{d\ell_i} d\ell_i}{L_i})^2

    where :math:`\frac{d\vec{T}_i}{d\ell_i}` is the Lorentz torque per unit length,
    in units of MN. The units of the squared mean torque are therefore (MN)^2.
    :math:`d\ell_i` is the arclength along the ith coil,
    :math:`L_i` is the total coil length.

    The units of the objective function are (MN)^2, where MN = meganewtons.

    This class assumes there are two (or three) distinct lists of coils,
    which may have different finite-build parameters and/or different numbers of quadrature points.
    In order to avoid buildup of optimizable
    dependencies, it directly computes the BiotSavart law terms, instead of relying on the existing
    C++ code that computes BiotSavart related terms. Within each list of coils,
    all coils must have the same number of quadrature points. The source_coils_coarse and source_coils_fine lists
    allows one to optimize e.g. the torque on target_coils from a set of dipole coils
    (with barely any quadrature points) and a set of TF coils (with many quadrature points).

    Args:
        target_coils (list of Coil or RegularizedCoil, shape (m,)): List of coils to use for computing SquaredMeanTorque.
        source_coils_coarse (list of Coil or RegularizedCoil, shape (m',)):
            Coarse-resolution source coils that provide torques on the target_coils.
            Torques are not computed on the source_coils.
        source_coils_fine (list of Coil or RegularizedCoil, optional):
            Fine-resolution source coils, used in addition to coarse. Default: []. This functionality
            is provided for when there are two sets of source coils with very different numbers of
            quadrature points. This occurs e.g. when optimizing TF coils and dipole coils.
        downsample (int):
            Factor by which to downsample the quadrature points
            by skipping through the array by a factor of ``downsample``,
            e.g. curve.gamma()[::downsample, :].
            Setting this parameter to a value larger than 1 will speed up the calculation,
            which may be useful if the set of coils is large, though it may introduce
            inaccuracy if ``downsample`` is set too large, or not a multiple of the
            total number of quadrature points (since this will produce a nonuniform set of points).
            This parameter is used to speed up expensive calculations during optimization,
            while retaining higher accuracy for the other objectives.

    Returns:
        float: Value of the objective function.
    """

    def __init__(
        self,
        target_coils,
        source_coils_coarse,
        source_coils_fine=None,
        downsample: int = 1,
    ):
        if not isinstance(target_coils, list):
            target_coils = [target_coils]
        if not isinstance(source_coils_coarse, list):
            source_coils_coarse = [source_coils_coarse]
        if source_coils_fine is None:
            source_coils_fine = []
        elif not isinstance(source_coils_fine, list):
            source_coils_fine = [source_coils_fine]
        self.target_coils = target_coils
        self.source_coils_coarse = [
            c for c in source_coils_coarse if c not in target_coils
        ]
        self.source_coils_fine = [c for c in source_coils_fine if c not in target_coils]
        if len(self.source_coils_coarse) == 0 and len(self.source_coils_fine) == 0:
            raise ValueError(
                "source_coils_coarse and source_coils_fine must together contain at least one coil not in target_coils."
            )
        self.source_coils_fine = [
            c for c in self.source_coils_fine if c not in self.source_coils_coarse
        ]
        self.source_coils = self.source_coils_coarse + self.source_coils_fine

        # Check that the coils in each list of coils (target_coils, source_coils_coarse, source_coils_fine)
        # all have the same number of quadrature points and that the downsample factor is a valid
        # multiple of the number of quadrature points.
        _check_quadpoints_consistency(self.target_coils, "target_coils")
        if len(self.source_coils_coarse) > 0:
            _check_quadpoints_consistency(
                self.source_coils_coarse, "source_coils_coarse"
            )
        if len(self.source_coils_fine) > 0:
            _check_quadpoints_consistency(self.source_coils_fine, "source_coils_fine")
        self.downsample = downsample
        _check_downsample(self.target_coils, downsample, "target_coils")
        if len(self.source_coils_coarse) > 0:
            _check_downsample(
                self.source_coils_coarse, downsample, "source_coils_coarse"
            )
        if len(self.source_coils_fine) > 0:
            _check_downsample(self.source_coils_fine, downsample, "source_coils_fine")

        self.J_jax = _SQUARED_MEAN_TORQUE_JAX
        self.dJ_jax = _SQUARED_MEAN_TORQUE_GRAD
        self._target_state_cache = _CoilStateGroupCache(self.target_coils)
        self._source_coarse_state_cache = _CoilStateGroupCache(self.source_coils_coarse)
        self._source_fine_state_cache = _CoilStateGroupCache(self.source_coils_fine)
        self._cached_jax_args = None

        super().__init__(depends_on=(target_coils + self.source_coils))

    def recompute_bell(self, parent=None):
        _invalidate_objective_state(
            self,
            parent,
            self._target_state_cache,
            self._source_coarse_state_cache,
            self._source_fine_state_cache,
        )

    def _J_args(self):
        """Build arguments for evaluation of J and dJ."""

        def build_args():
            gammas_targets, gammadashs_targets, currents_targets = (
                self._target_state_cache.arrays()
            )
            gammas_coarse, gammadashs_coarse, currents_coarse = (
                self._source_coarse_state_cache.arrays()
            )
            gammas_fine, gammadashs_fine, currents_fine = (
                self._source_fine_state_cache.arrays()
            )
            return (
                gammas_targets,
                gammas_coarse,
                gammadashs_targets,
                gammadashs_coarse,
                currents_targets,
                currents_coarse,
                gammas_fine,
                gammadashs_fine,
                currents_fine,
                self.downsample,
            )

        return _cached_objective_args(self, build_args)

    def J(self):
        r"""Evaluate the squared mean torque objective."""
        return self.J_jax(*self._J_args())

    @derivative_dec
    def dJ(self):
        r"""Compute the derivative of the squared mean torque objective with respect to
        all optimizable degrees of freedom (coil geometry and currents for both
        target_coils and source_coils_coarse and source_coils_fine if passed).

        Returns:
            Derivative: The gradient of J with respect to all DOFs.
        """
        (
            dJ_dgamma_targets,
            dJ_dgamma_coarse,
            dJ_dgammadash_targets,
            dJ_dgammadash_coarse,
            dJ_dcurrent_targets,
            dJ_dcurrent_coarse,
            dJ_dgamma_fine,
            dJ_dgammadash_fine,
            dJ_dcurrent_fine,
        ) = self.dJ_jax(*self._J_args())
        dJ = _assemble_curve_current_derivative(
            self.target_coils,
            dgamma=dJ_dgamma_targets,
            dgammadash=dJ_dgammadash_targets,
            dcurrent=dJ_dcurrent_targets,
        )
        if len(self.source_coils_coarse) > 0:
            dJ += _assemble_curve_current_derivative(
                self.source_coils_coarse,
                dgamma=dJ_dgamma_coarse,
                dgammadash=dJ_dgammadash_coarse,
                dcurrent=dJ_dcurrent_coarse,
            )
        if len(self.source_coils_fine) > 0:
            dJ += _assemble_curve_current_derivative(
                self.source_coils_fine,
                dgamma=dJ_dgamma_fine,
                dgammadash=dJ_dgammadash_fine,
                dcurrent=dJ_dcurrent_fine,
            )
        return dJ

    return_fn_map = {"J": J, "dJ": dJ}
