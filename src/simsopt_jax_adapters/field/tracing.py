"""JAX tracing wrappers for fieldlines and particle trajectories."""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import repeat
from math import sqrt

import jax
import jax.numpy as jnp
import numpy as np
import simsoptpp as sopp

from simsopt._core.types import RealArray
from simsopt._core.util import parallel_loop_bounds
from simsopt._core.tracing_metadata import (
    levelset_classifier_from_interpolant,
    levelset_classifier_for,
)
from simsopt.field.magneticfield import MagneticField
from simsopt.field.tracing import gc_to_fullorbit_initial_guesses
from simsopt_jax.core._math_utils import as_jax_float64 as _as_jax_float64
from simsopt_jax.runtime.host_boundary import host_array as _jax_trace_host_array
from simsopt_jax.core.tracing import (
    FieldlineTracingSpec,
    FullorbitTracingSpec,
    GuidingCenterTracingSpec,
    IterStoppingCriterion as JaxIterStoppingCriterion,
    LevelsetStoppingCriterion as JaxLevelsetStoppingCriterion,
    MaxRStoppingCriterion as JaxMaxRStoppingCriterion,
    MaxToroidalFluxStoppingCriterion as JaxMaxToroidalFluxStoppingCriterion,
    MaxZStoppingCriterion as JaxMaxZStoppingCriterion,
    MinRStoppingCriterion as JaxMinRStoppingCriterion,
    MinToroidalFluxStoppingCriterion as JaxMinToroidalFluxStoppingCriterion,
    MinZStoppingCriterion as JaxMinZStoppingCriterion,
    ToroidalTransitStoppingCriterion as JaxToroidalTransitStoppingCriterion,
    _boozer_particle_dtmaxs,
    _cartesian_particle_dtmaxs,
    _cartesian_radii,
    _fieldline_dtmaxs,
    _magnetic_moments,
    _quarter_turn_dtmaxs,
    trace_fieldlines_batched,
    trace_fullorbits_batched,
    trace_guiding_centers_batched,
    trace_guiding_centers_boozer_batched,
)
from simsopt_jax_adapters.field.boozer_field import (
    BoozerRadialInterpolantJAX,
    InterpolatedBoozerFieldJAX,
)
from simsopt_jax_adapters.geo.surface_classifier import (
    levelset_classifier_fn_from_surface_classifier,
    supports_levelset_classifier_conversion,
)
from simsopt.util.constants import (
    ALPHA_PARTICLE_CHARGE,
    ALPHA_PARTICLE_MASS,
    FUSION_ALPHA_PARTICLE_ENERGY,
)


logger = logging.getLogger(__name__)

__all__ = [
    "compute_fieldlines",
    "trace_particles",
    "trace_particles_boozer",
]


def _normalize_parallel_speeds(
    parallel_speeds: RealArray, nparticles: int
) -> np.ndarray:
    speed_par = np.asarray(parallel_speeds).reshape(-1)
    if speed_par.shape != (nparticles,):
        raise ValueError(
            f"Expected {nparticles} parallel speeds, got shape {speed_par.shape}"
        )
    return speed_par


def _allgather_flat(comm, values):
    if comm is None:
        return values
    return [item for rank_values in comm.allgather(values) for item in rank_values]


def _event_hits_prefix(phi_hits, phi_hits_count, *, context: str) -> np.ndarray:
    hits = np.asarray(phi_hits, dtype=np.float64)
    count = int(phi_hits_count)
    max_phi_hits = hits.shape[0]
    if count > max_phi_hits:
        raise RuntimeError(
            f"{context} recorded {count} event rows but max_phi_hits={max_phi_hits}; "
            "increase the JAX tracing event buffer before using these results."
        )
    return hits[:count]


def _batched_jax_trace_payloads(
    result,
    *,
    forget_exact_path: bool,
    event_context: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    trajectories = _jax_trace_host_array(result.trajectory, dtype=np.float64)
    masks = _jax_trace_host_array(result.mask, dtype=bool)
    phi_hits = _jax_trace_host_array(result.phi_hits, dtype=np.float64)
    phi_hit_counts = _jax_trace_host_array(result.phi_hits_count, dtype=np.int64)
    phi_hit_counts = phi_hit_counts.reshape(-1)

    res_tys: list[np.ndarray] = []
    res_phi_hits: list[np.ndarray] = []
    for traj, mask, hits, hit_count in zip(
        trajectories, masks, phi_hits, phi_hit_counts, strict=True
    ):
        live = traj[mask]
        if forget_exact_path and live.shape[0] >= 2:
            res_tys.append(np.stack([live[0], live[-1]], axis=0))
        else:
            res_tys.append(live)
        res_phi_hits.append(_event_hits_prefix(hits, hit_count, context=event_context))
    return res_tys, res_phi_hits


def _jax_trace_lost(status: int, t_final: float, tmax: float) -> bool:
    return status < 0 or t_final < float(tmax) - 1e-15


def _batched_trace_status_arrays(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _jax_trace_host_array(result.status, dtype=np.int32).reshape(-1),
        _jax_trace_host_array(result.t_final, dtype=np.float64).reshape(-1),
        _jax_trace_host_array(result.steps_taken, dtype=np.int32).reshape(-1),
    )


def _log_batched_jax_trace_statuses(
    result,
    *,
    first: int,
    total: int,
    tmax: float,
    label: str,
    live_trajectories: list[np.ndarray] | None = None,
) -> int:
    statuses, t_finals, steps_taken = _batched_trace_status_arrays(result)
    loss_ctr = 0
    live_items = (
        repeat(None, statuses.shape[0])
        if live_trajectories is None
        else live_trajectories
    )
    status_items = zip(live_items, statuses, t_finals, steps_taken, strict=True)
    for offset, (live, status, t_final, step_count) in enumerate(status_items):
        i = first + offset
        status_int = int(status)
        t_final_float = float(t_final)
        if status_int > 0:
            logger.warning(
                f"{i + 1:3d}/{total}, {label} status={status_int} "
                f"t_final={t_final_float}, steps_taken={int(step_count)}"
            )
        elif status_int < 0:
            logger.debug(
                f"{i + 1:3d}/{total}, {label} stopped by criterion "
                f"index {-1 - status_int}, t_final={t_final_float}"
            )
        elif live is not None and live.shape[0] > 0:
            dtavg = live[-1, 0] / max(live.shape[0], 1)
            logger.debug(
                f"{i + 1:3d}/{total}, t_final={live[-1, 0]}, "
                f"average timestep {dtavg:.10f}s (JAX backend)"
            )
        if _jax_trace_lost(status_int, t_final_float, tmax):
            loss_ctr += 1
    return loss_ctr


def _require_jax_field_B(field: MagneticField) -> Callable[[object], object]:
    field_fn = getattr(field, "jax_B_at", None)
    if not callable(field_fn):
        raise TypeError(
            "JAX tracing requires a JAX-native MagneticField wrapper exposing "
            "`jax_B_at(point)`. Use a *JAX field class such as ToroidalFieldJAX, "
            "or call simsopt.field.tracing for the CPU path."
        )
    return field_fn


def _resolve_jax_field_B(
    field: MagneticField,
) -> tuple[Callable[..., object], object | None]:
    state_fn = getattr(field, "jax_B_at_state", None)
    state_getter = getattr(field, "jax_tracing_state", None)
    if callable(state_fn) and callable(state_getter):
        return state_fn, state_getter()
    return _require_jax_field_B(field), None


def _require_jax_field_B_dB(
    field: MagneticField,
) -> Callable[[object], tuple[object, object]]:
    field_fn = getattr(field, "jax_B_dB_at", None)
    if callable(field_fn):
        return field_fn
    grad_abs_fn = getattr(field, "jax_B_GradAbsB_at", None)
    if callable(grad_abs_fn):

        def _field_fn(point):
            B_raw, grad_abs_raw = grad_abs_fn(point)
            B = jnp.asarray(B_raw, dtype=jnp.float64).reshape((3,))
            grad_abs_B = jnp.asarray(grad_abs_raw, dtype=jnp.float64).reshape((3,))
            abs_B = jnp.linalg.norm(B)
            dB_by_dX = grad_abs_B[:, None] * B[None, :] / abs_B
            return B, dB_by_dX

        return _field_fn
    raise TypeError(
        "JAX particle tracing requires a JAX-native MagneticField wrapper "
        "exposing `jax_B_dB_at(point)` or `jax_B_GradAbsB_at(point)`. Use a "
        "*JAX field class such as ToroidalFieldJAX or InterpolatedFieldJAX, "
        "or call simsopt.field.tracing for the CPU path."
    )


def _resolve_jax_field_B_dB(
    field: MagneticField,
) -> tuple[Callable[..., tuple[object, object]], object | None]:
    state_fn = getattr(field, "jax_B_dB_at_state", None)
    state_getter = getattr(field, "jax_tracing_state", None)
    if callable(state_fn) and callable(state_getter):
        return state_fn, state_getter()
    state_fn = getattr(field, "jax_B_GradAbsB_at_state", None)
    if callable(state_fn) and callable(state_getter):

        def _state_field_fn(state, point):
            B_raw, grad_abs_raw = state_fn(state, point)
            B = jnp.asarray(B_raw, dtype=jnp.float64).reshape((3,))
            grad_abs_B = jnp.asarray(grad_abs_raw, dtype=jnp.float64).reshape((3,))
            abs_B = jnp.linalg.norm(B)
            dB_by_dX = grad_abs_B[:, None] * B[None, :] / abs_B
            return B, dB_by_dX

        return _state_field_fn, state_getter()
    return _require_jax_field_B_dB(field), None


def trace_particles_boozer(
    field: BoozerRadialInterpolantJAX | InterpolatedBoozerFieldJAX,
    stz_inits: RealArray,
    parallel_speeds: RealArray,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    comm=None,
    zetas=[],
    stopping_criteria=[],
    mode="gc_vac",
    forget_exact_path=False,
):
    """Trace Boozer-coordinate particles with the JAX tracing backend."""
    nparticles = stz_inits.shape[0]
    speed_par = _normalize_parallel_speeds(parallel_speeds, nparticles)
    m = mass
    speed_total = sqrt(2 * Ekin / m)
    mode = mode.lower()
    assert mode in ["gc", "gc_vac", "gc_nok"]
    return _trace_particles_boozer_jax(
        field,
        stz_inits,
        speed_par,
        speed_total,
        tmax=tmax,
        mass=m,
        charge=charge,
        tol=tol,
        comm=comm,
        zetas=zetas,
        stopping_criteria=stopping_criteria,
        mode=mode,
        forget_exact_path=forget_exact_path,
    )


def _trace_particles_boozer_jax(
    field,
    stz_inits,
    speed_par,
    speed_total,
    *,
    tmax,
    mass,
    charge,
    tol,
    comm,
    zetas,
    stopping_criteria,
    mode,
    forget_exact_path,
):
    """JAX backend for Boozer-coordinate guiding-center tracing."""
    if not isinstance(field, (BoozerRadialInterpolantJAX, InterpolatedBoozerFieldJAX)):
        raise NotImplementedError(
            "trace_particles_boozer JAX backend requires a "
            "BoozerRadialInterpolantJAX or InterpolatedBoozerFieldJAX field "
            f"instance; got {type(field).__name__}. Wrap the upstream "
            "BoozerRadialInterpolant via BoozerRadialInterpolantJAX(upstream), "
            "wrap the upstream InterpolatedBoozerField via "
            "InterpolatedBoozerFieldJAX(...), or call simsopt.field.tracing "
            "for the CPU path."
        )
    rhs_mode_map = {
        "gc_vac": "vacuum",
        "gc_nok": "no_k",
        "gc": "full",
    }
    if mode not in rhs_mode_map:
        raise NotImplementedError(
            "trace_particles_boozer JAX backend supports mode in "
            f"{set(rhs_mode_map)}; got mode={mode!r}. Full-orbit "
            "Boozer tracing is not implemented."
        )
    rhs_mode = rhs_mode_map[mode]

    jax_stopping_criteria = _translate_stopping_criteria_to_jax(stopping_criteria)
    if len(zetas) > 0:
        zetas_arr = _as_jax_float64(np.asarray(list(zetas), dtype=np.float64))
    else:
        zetas_arr = None

    nparticles = stz_inits.shape[0]
    max_steps = 4000
    max_phi_hits = 4096
    res_tys = []
    res_zeta_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    local_stz = np.asarray(stz_inits[first:last], dtype=np.float64)
    local_speed_par = np.asarray(speed_par[first:last], dtype=np.float64)
    if local_stz.shape[0] > 0:
        field.set_points(local_stz)
        abs_B_initial = _jax_trace_host_array(field.modB(), dtype=np.float64).reshape(
            -1
        )
        G0 = _jax_trace_host_array(field.G(), dtype=np.float64).reshape(-1)
        mus = _magnetic_moments(float(speed_total), local_speed_par, abs_B_initial)
        dtmaxs = _boozer_particle_dtmaxs(G0, abs_B_initial, float(speed_total))
        spec = GuidingCenterTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=max_steps,
            max_phi_hits=max_phi_hits,
        )
        result = trace_guiding_centers_boozer_batched(
            spec,
            _as_jax_float64(np.column_stack([local_stz, local_speed_par])),
            _as_jax_float64(dtmaxs),
            _as_jax_float64(mus),
            field,
            m=float(mass),
            q=float(charge),
            mode=rhs_mode,
            zetas=zetas_arr,
            stopping_criteria=jax_stopping_criteria,
        )
        local_res_tys, local_res_zeta_hits = _batched_jax_trace_payloads(
            result,
            forget_exact_path=forget_exact_path,
            event_context="JAX Boozer guiding-centre tracing",
        )
        res_tys.extend(local_res_tys)
        res_zeta_hits.extend(local_res_zeta_hits)
        loss_ctr += _log_batched_jax_trace_statuses(
            result,
            first=first,
            total=nparticles,
            tmax=tmax,
            label="JAX Boozer guiding-centre",
        )
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    res_tys = _allgather_flat(comm, res_tys)
    res_zeta_hits = _allgather_flat(comm, res_zeta_hits)
    logger.debug(
        f"Particles lost {loss_ctr}/{nparticles}="
        f"{(100 * loss_ctr) // max(nparticles, 1):d}% (JAX Boozer backend)"
    )
    return res_tys, res_zeta_hits


def trace_particles(
    field: MagneticField,
    xyz_inits: RealArray,
    parallel_speeds: RealArray,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    comm=None,
    phis=[],
    stopping_criteria=[],
    mode="gc_vac",
    forget_exact_path=False,
    phase_angle=0,
):
    """Trace particles with the JAX tracing backend."""
    nparticles = xyz_inits.shape[0]
    speed_par = _normalize_parallel_speeds(parallel_speeds, nparticles)
    mode = mode.lower()
    assert mode in ["gc", "gc_vac", "full"]
    m = mass
    speed_total = sqrt(2 * Ekin / m)
    if mode == "full":
        return _trace_particles_jax_fullorbit_vacuum(
            field,
            xyz_inits,
            speed_par,
            speed_total,
            tmax=tmax,
            mass=m,
            charge=charge,
            tol=tol,
            comm=comm,
            phis=phis,
            stopping_criteria=stopping_criteria,
            forget_exact_path=forget_exact_path,
            phase_angle=phase_angle,
        )
    return _trace_particles_jax_guiding_center_vacuum(
        field,
        xyz_inits,
        speed_par,
        speed_total,
        tmax=tmax,
        mass=m,
        charge=charge,
        tol=tol,
        comm=comm,
        phis=phis,
        stopping_criteria=stopping_criteria,
        mode=mode,
        forget_exact_path=forget_exact_path,
    )


def _trace_particles_jax_guiding_center_vacuum(
    field,
    xyz_inits,
    speed_par,
    speed_total,
    *,
    tmax,
    mass,
    charge,
    tol,
    comm,
    phis,
    stopping_criteria,
    mode,
    forget_exact_path,
):
    """JAX backend for Cartesian vacuum guiding-center tracing."""
    if mode != "gc_vac":
        raise NotImplementedError(
            "trace_particles JAX guiding-centre helper currently only "
            f"supports mode='gc_vac' (got mode={mode!r}). The non-vacuum "
            "guiding-centre mode ('gc') is a deferred follow-up. The "
            "'full' mode is routed separately to "
            "_trace_particles_jax_fullorbit_vacuum from the public "
            "trace_particles wrapper."
        )
    field_fn, field_state = _resolve_jax_field_B_dB(field)

    jax_stopping_criteria = _translate_stopping_criteria_to_jax(stopping_criteria)
    if len(phis) > 0:
        phis_arr = _as_jax_float64(np.asarray(list(phis), dtype=np.float64))
    else:
        phis_arr = None

    nparticles = xyz_inits.shape[0]
    max_steps = 4000
    max_phi_hits = 4096
    res_tys = []
    res_phi_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    local_xyz = np.asarray(xyz_inits[first:last], dtype=np.float64)
    local_speed_par = np.asarray(speed_par[first:last], dtype=np.float64)
    if local_xyz.shape[0] > 0:
        local_xyz_device = _as_jax_float64(local_xyz)
        if field_state is None:
            B_initial, _ = jax.vmap(field_fn)(local_xyz_device)
        else:
            B_initial, _ = jax.vmap(lambda point: field_fn(field_state, point))(
                local_xyz_device
            )
        abs_B_initial = np.linalg.norm(
            _jax_trace_host_array(B_initial, dtype=np.float64), axis=1
        )
        mus = _magnetic_moments(float(speed_total), local_speed_par, abs_B_initial)
        dtmaxs = _cartesian_particle_dtmaxs(local_xyz, float(speed_total))
        spec = GuidingCenterTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=max_steps,
            max_phi_hits=max_phi_hits,
        )
        y0s = np.column_stack([local_xyz, local_speed_par])
        result = trace_guiding_centers_batched(
            spec,
            _as_jax_float64(y0s),
            _as_jax_float64(dtmaxs),
            _as_jax_float64(mus),
            field_fn,
            m=float(mass),
            q=float(charge),
            phis=phis_arr,
            stopping_criteria=jax_stopping_criteria,
            magnetic_field_state=field_state,
        )
        local_res_tys, local_res_phi_hits = _batched_jax_trace_payloads(
            result,
            forget_exact_path=forget_exact_path,
            event_context="JAX guiding-centre tracing",
        )
        res_tys.extend(local_res_tys)
        res_phi_hits.extend(local_res_phi_hits)
        loss_ctr += _log_batched_jax_trace_statuses(
            result,
            first=first,
            total=nparticles,
            tmax=tmax,
            label="JAX guiding-centre",
        )
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    res_tys = _allgather_flat(comm, res_tys)
    res_phi_hits = _allgather_flat(comm, res_phi_hits)
    logger.debug(
        f"Particles lost {loss_ctr}/{nparticles}="
        f"{(100 * loss_ctr) // max(nparticles, 1):d}% (JAX backend)"
    )
    return res_tys, res_phi_hits


def _trace_particles_jax_fullorbit_vacuum(
    field,
    xyz_inits,
    speed_par,
    speed_total,
    *,
    tmax,
    mass,
    charge,
    tol,
    comm,
    phis,
    stopping_criteria,
    forget_exact_path,
    phase_angle,
):
    """JAX backend for Cartesian full-orbit tracing."""
    field_fn, field_state = _resolve_jax_field_B(field)
    jax_stopping_criteria = _translate_stopping_criteria_to_jax(stopping_criteria)
    if len(phis) > 0:
        phis_arr = _as_jax_float64(np.asarray(list(phis), dtype=np.float64))
    else:
        phis_arr = None

    max_steps = 20000
    max_phi_hits = 4096
    nparticles = xyz_inits.shape[0]
    first, last = parallel_loop_bounds(comm, nparticles)
    local_xyz = np.asarray(xyz_inits[first:last], dtype=np.float64)
    local_speed_par = np.asarray(speed_par[first:last], dtype=np.float64)
    if local_xyz.shape[0] > 0:
        xyz_inits_full, v_inits, _ = gc_to_fullorbit_initial_guesses(
            field,
            local_xyz,
            local_speed_par,
            float(speed_total),
            float(mass),
            float(charge),
            eta=float(phase_angle),
        )
    else:
        xyz_inits_full = np.zeros((0, 3), dtype=np.float64)
        v_inits = np.zeros((0, 3), dtype=np.float64)
    res_tys = []
    res_phi_hits = []
    loss_ctr = 0
    if xyz_inits_full.shape[0] > 0:
        speeds = np.linalg.norm(v_inits, axis=1)
        dtmaxs = _quarter_turn_dtmaxs(_cartesian_radii(xyz_inits_full), speeds)
        spec = FullorbitTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=max_steps,
            max_phi_hits=max_phi_hits,
        )
        result = trace_fullorbits_batched(
            spec,
            _as_jax_float64(np.column_stack([xyz_inits_full, v_inits])),
            _as_jax_float64(dtmaxs),
            field_fn,
            m=float(mass),
            q=float(charge),
            phis=phis_arr,
            stopping_criteria=jax_stopping_criteria,
            magnetic_field_state=field_state,
        )
        local_res_tys, local_res_phi_hits = _batched_jax_trace_payloads(
            result,
            forget_exact_path=forget_exact_path,
            event_context="JAX full-orbit tracing",
        )
        res_tys.extend(local_res_tys)
        res_phi_hits.extend(local_res_phi_hits)
        loss_ctr += _log_batched_jax_trace_statuses(
            result,
            first=first,
            total=nparticles,
            tmax=tmax,
            label="JAX full-orbit",
        )
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    res_tys = _allgather_flat(comm, res_tys)
    res_phi_hits = _allgather_flat(comm, res_phi_hits)
    logger.debug(
        f"Particles lost {loss_ctr}/{nparticles}="
        f"{(100 * loss_ctr) // max(nparticles, 1):d}% (JAX full-orbit backend)"
    )
    return res_tys, res_phi_hits


def compute_fieldlines(
    field,
    R0,
    Z0,
    tmax=200,
    tol=1e-7,
    phis=[],
    stopping_criteria=[],
    comm=None,
):
    """Compute fieldlines with the JAX tracing backend."""
    assert len(R0) == len(Z0)
    return _compute_fieldlines_jax(
        field,
        R0,
        Z0,
        tmax=tmax,
        tol=tol,
        phis=phis,
        stopping_criteria=stopping_criteria,
        comm=comm,
    )


def _translate_stopping_criteria_to_jax(stopping_criteria: list) -> tuple:
    """Translate CPU stopping criteria into JAX-side dataclasses."""

    def _attr(obj, name):
        if not hasattr(obj, name):
            raise NotImplementedError(
                f"JAX tracing path cannot read attribute '{name}' on "
                f"{type(obj).__name__}; use the simsopt.field.tracing "
                "Python wrapper class rather than the bound C++ "
                "``simsoptpp.*`` class directly."
            )
        return getattr(obj, name)

    translated = []
    for crit in stopping_criteria:
        if isinstance(crit, sopp.LevelsetStoppingCriterion):
            classifier_obj = levelset_classifier_for(crit) or getattr(
                crit, "classifier", None
            )
            if (
                classifier_obj is not None
                and not supports_levelset_classifier_conversion(classifier_obj)
            ):
                classifier_obj = levelset_classifier_from_interpolant(classifier_obj)
            if classifier_obj is None:
                raise NotImplementedError(
                    "JAX tracing path cannot translate a raw "
                    "sopp.LevelsetStoppingCriterion: build the criterion "
                    "via simsopt.field.tracing.LevelsetStoppingCriterion("
                    "SurfaceClassifier(surface)) so the JAX-side interpolant "
                    "spec can be rebuilt from the classifier grid metadata."
                )
            if not supports_levelset_classifier_conversion(classifier_obj):
                raise NotImplementedError(
                    "JAX tracing path requires a SurfaceClassifier-derived "
                    "interpolant as the LevelsetStoppingCriterion argument; "
                    f"received {type(classifier_obj).__name__}."
                )
            jax_classifier_fn = levelset_classifier_fn_from_surface_classifier(
                classifier_obj
            )
            translated.append(
                JaxLevelsetStoppingCriterion(classifier_fn=jax_classifier_fn)
            )
            continue
        if isinstance(crit, sopp.MinRStoppingCriterion):
            translated.append(
                JaxMinRStoppingCriterion(crit_r=float(_attr(crit, "crit_r")))
            )
            continue
        if isinstance(crit, sopp.MaxRStoppingCriterion):
            translated.append(
                JaxMaxRStoppingCriterion(crit_r=float(_attr(crit, "crit_r")))
            )
            continue
        if isinstance(crit, sopp.MinZStoppingCriterion):
            translated.append(
                JaxMinZStoppingCriterion(crit_z=float(_attr(crit, "crit_z")))
            )
            continue
        if isinstance(crit, sopp.MaxZStoppingCriterion):
            translated.append(
                JaxMaxZStoppingCriterion(crit_z=float(_attr(crit, "crit_z")))
            )
            continue
        if isinstance(crit, sopp.ToroidalTransitStoppingCriterion):
            if bool(_attr(crit, "flux")):
                raise NotImplementedError(
                    "JAX tracing path supports "
                    "ToroidalTransitStoppingCriterion only with flux=False; "
                    "flux=True is a flux-coordinate transit criterion and is "
                    "not implemented by the Cartesian JAX tracing route."
                )
            translated.append(
                JaxToroidalTransitStoppingCriterion(
                    max_transits=float(_attr(crit, "max_transits"))
                )
            )
            continue
        if isinstance(crit, sopp.IterationStoppingCriterion):
            translated.append(
                JaxIterStoppingCriterion(max_iter=int(_attr(crit, "max_iter")))
            )
            continue
        if isinstance(crit, sopp.MinToroidalFluxStoppingCriterion):
            min_s = getattr(crit, "min_s", None)
            if min_s is None:
                raise NotImplementedError(
                    "JAX tracing path cannot translate a raw "
                    "sopp.MinToroidalFluxStoppingCriterion: build the criterion "
                    "via simsopt.field.tracing.MinToroidalFluxStoppingCriterion."
                )
            translated.append(JaxMinToroidalFluxStoppingCriterion(min_s=float(min_s)))
            continue
        if isinstance(crit, sopp.MaxToroidalFluxStoppingCriterion):
            max_s = getattr(crit, "max_s", None)
            if max_s is None:
                raise NotImplementedError(
                    "JAX tracing path cannot translate a raw "
                    "sopp.MaxToroidalFluxStoppingCriterion: build the criterion "
                    "via simsopt.field.tracing.MaxToroidalFluxStoppingCriterion."
                )
            translated.append(JaxMaxToroidalFluxStoppingCriterion(max_s=float(max_s)))
            continue
        raise NotImplementedError(
            "JAX tracing path cannot translate stopping criterion of type "
            f"{type(crit).__name__}; supported classes are LevelsetStoppingCriterion, "
            "MinR/MaxR/MinZ/MaxZ/ToroidalTransit/Iteration/Min/MaxToroidalFlux."
        )
    return tuple(translated)


def _compute_fieldlines_jax(field, R0, Z0, tmax, tol, phis, stopping_criteria, comm):
    """JAX backend for fieldline tracing."""
    field_fn, field_state = _resolve_jax_field_B(field)

    jax_stopping_criteria = _translate_stopping_criteria_to_jax(stopping_criteria)
    if len(phis) > 0:
        phis_arr = _as_jax_float64(np.asarray(list(phis), dtype=np.float64))
    else:
        phis_arr = None

    max_steps = 4000
    max_phi_hits = 4096
    nlines = len(R0)
    res_tys = []
    res_phi_hits = []
    R0_arr = np.asarray(R0, dtype=np.float64)
    Z0_arr = np.asarray(Z0, dtype=np.float64)
    first, last = parallel_loop_bounds(comm, nlines)
    local_y0 = np.stack(
        [
            R0_arr[first:last],
            np.zeros(last - first, dtype=np.float64),
            Z0_arr[first:last],
        ],
        axis=1,
    )
    if local_y0.shape[0] > 0:
        local_y0_device = _as_jax_float64(local_y0)
        if field_state is None:
            B_initial = jax.vmap(field_fn)(local_y0_device)
        else:
            B_initial = jax.vmap(lambda point: field_fn(field_state, point))(
                local_y0_device
            )
        abs_B_initial = np.linalg.norm(
            _jax_trace_host_array(B_initial, dtype=np.float64), axis=1
        )
        dtmaxs = _fieldline_dtmaxs(local_y0, abs_B_initial)
        spec = FieldlineTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=max_steps,
            max_phi_hits=max_phi_hits,
        )
        result = trace_fieldlines_batched(
            spec,
            local_y0_device,
            _as_jax_float64(dtmaxs),
            field_fn,
            phis=phis_arr,
            stopping_criteria=jax_stopping_criteria,
            magnetic_field_state=field_state,
        )
        local_res_tys, local_res_phi_hits = _batched_jax_trace_payloads(
            result,
            forget_exact_path=False,
            event_context="JAX fieldline tracing",
        )
        res_tys.extend(local_res_tys)
        res_phi_hits.extend(local_res_phi_hits)
        _log_batched_jax_trace_statuses(
            result,
            first=first,
            total=nlines,
            tmax=tmax,
            label="JAX fieldline",
            live_trajectories=local_res_tys,
        )
    res_tys = _allgather_flat(comm, res_tys)
    res_phi_hits = _allgather_flat(comm, res_phi_hits)
    return res_tys, res_phi_hits
