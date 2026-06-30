"""JAX fixed-step tracer for the default Poincare render contract.

This script is intentionally separate from ``poincare_surfaces.py``.  The
existing script remains the adaptive C++/SIMSOPT reference path; this one is an
opt-in throughput path for the default render mode only:

* extended-surface seed radii,
* native Biot-Savart field (no interpolation),
* box guardrails plus the toroidal-angle ODE singularity guard
  (no Boozer-surface exit guard),
* four Poincare planes per field period.

The integrator uses toroidal angle as the independent variable and a fixed-step
RK4 scan over one or more field periods.  That makes the main loop device-side
and suitable for GPU benchmarking, while host-side JSON loading and plotting
stay shared with the existing Poincare tooling.
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt.design_only_fields import (  # noqa: E402
    assert_topology_field_allowed,
    load_design_only_results_metadata,
)
from banana_opt.json_compat import load_boozer_finite_i as load  # noqa: E402
from poincare_surfaces import (  # noqa: E402
    assert_no_stale_unselected_metrics,
    build_poincare_mode_artifact,
    build_poincare_render_modes,
    design_only_override_enabled,
    plot_poincare_data,
)
from simsopt.geo import BoozerSurface, curves_to_vtk  # noqa: E402
from topology_scorer import (  # noqa: E402
    STOP_LABELS_DIAGNOSTIC,
    build_stopping_criteria,
    extended_surface_trace_domain,
    topology_iteration_limit,
    trace_metrics as _trace_metrics,
)


MU0_OVER_4PI = 1e-7
DEFAULT_CPU_TMAX_REFERENCE = 7000.0
DEFAULT_TOL_REFERENCE = 1e-7
DEFAULT_NFIELDLINES = 50
# Render-fidelity defaults (raised from 64/64): low-iota surfaces (iota~0.09)
# need ~200 toroidal returns to fill each poloidal Poincare loop with punctures,
# plus ~256 RK4 steps/period for trajectory convergence. At 64/64 clean-nested
# surfaces render as sparse smears that read as "broken" though the field is
# nested; 256/200 was validated to converge (s256==s512) and match the C++
# compute_fieldlines reference. Pass smaller CLI/env values for fast GPU-
# throughput benchmarking, where render fidelity is not the goal.
DEFAULT_JAX_FIELD_PERIODS = 200
DEFAULT_JAX_STEPS_PER_FIELD_PERIOD = 256
DEFAULT_MIN_BPHI_OVER_B = 1e-10
SEED_INSET_FRACTION = 0.05
DEFAULT_EXTEND_DISTANCE = 0.05
PHI_ODE_SINGULARITY_STOP_REASON = len(STOP_LABELS_DIAGNOSTIC)
JAX_DEFAULT_STOP_LABELS = [
    *STOP_LABELS_DIAGNOSTIC,
    "phi_ode_singularity",
]


@dataclasses.dataclass(frozen=True)
class DiscreteBiotSavartCoils:
    """Rectangular coil quadrature arrays consumed by the JAX field kernel."""

    gamma: np.ndarray
    gammadash: np.ndarray
    quadrature_mask: np.ndarray
    currents: np.ndarray

    @property
    def ncoils(self):
        return int(self.gamma.shape[0])

    @property
    def nquadpoints(self):
        return int(self.gamma.shape[1])

    @property
    def quadrature_counts(self):
        return np.sum(self.quadrature_mask, axis=1).astype(int)


def _env_int(name, default):
    value = os.environ.get(name)
    return int(default if value is None else value)


def _env_float(name, default):
    value = os.environ.get(name)
    return float(default if value is None else value)


def resolve_output_dir(env=None):
    environ = os.environ if env is None else env
    if environ.get("POINCARE_OUT_DIR"):
        return Path(environ["POINCARE_OUT_DIR"]).resolve()
    outputs_root = EXAMPLE_ROOT / "SINGLE_STAGE" / "outputs"
    raw_candidates = sorted(
        glob.glob(str(outputs_root / "mpol=*")),
        key=os.path.getmtime,
        reverse=True,
    )
    candidates = [
        candidate
        for candidate in raw_candidates
        if (
            Path(candidate, "biot_savart_opt.json").exists()
            and Path(candidate, "surf_opt.json").exists()
        )
        or (
            Path(candidate, "biot_savart_init.json").exists()
            and Path(candidate, "surf_init.json").exists()
        )
    ]
    if not candidates:
        raise FileNotFoundError(
            "No single-stage output with a complete field/surface pair found in "
            f"{outputs_root}. Set POINCARE_OUT_DIR."
        )
    out_dir = Path(candidates[0]).resolve()
    print(f"Auto-selected: {out_dir.name}")
    return out_dir


def validate_jax_trace_settings(n_field_periods, steps_per_field_period):
    n_field_periods = int(n_field_periods)
    steps_per_field_period = int(steps_per_field_period)
    if n_field_periods <= 0:
        raise ValueError("field periods must be positive")
    if steps_per_field_period <= 0:
        raise ValueError("steps per field period must be positive")
    if steps_per_field_period % 4 != 0:
        raise ValueError(
            "steps per field period must be divisible by 4 so the four default "
            "Poincare planes are hit exactly"
        )
    return steps_per_field_period // 4


def configure_jax_platform(platform):
    if platform:
        jax.config.update("jax_platform_name", str(platform))


def load_field_and_surface(out_dir):
    opt_bs_path = out_dir / "biot_savart_opt.json"
    init_bs_path = out_dir / "biot_savart_init.json"
    opt_surf_path = Path(
        os.environ.get("POINCARE_OPT_SURF_PATH", str(out_dir / "surf_opt.json"))
    )
    init_surf_path = out_dir / "surf_init.json"
    results_meta = load_design_only_results_metadata(str(out_dir))
    allow_design_only_field = design_only_override_enabled()

    has_opt = opt_bs_path.exists() and opt_surf_path.exists()
    if has_opt:
        bs = load(str(opt_bs_path))
        assert_topology_field_allowed(
            bs,
            results_meta,
            allow_design_only_field=allow_design_only_field,
            consumer="poincare_surfaces_jax_default",
        )
        surf = load(str(opt_surf_path))
        field_label = "opt"
        print("Loaded OPTIMIZED field + surface")
    else:
        bs = load(str(init_bs_path))
        assert_topology_field_allowed(
            bs,
            results_meta,
            allow_design_only_field=allow_design_only_field,
            consumer="poincare_surfaces_jax_default",
        )
        surf = load(str(init_surf_path))
        field_label = "init"
        if opt_bs_path.exists() != opt_surf_path.exists():
            print(
                "WARNING: mismatched opt files "
                f"(bs={opt_bs_path.exists()}, surf={opt_surf_path.exists()}). "
                "Using init for both."
            )
        else:
            print("Loaded INITIAL field + surface (no opt found)")

    if isinstance(surf, BoozerSurface):
        surf = surf.surface
    return bs, surf, field_label, results_meta, allow_design_only_field


def export_reference_geometry(bs, surf, out_dir, field_label):
    curves_to_vtk(
        [coil.curve for coil in bs.coils],
        str(out_dir / f"curves_{field_label}_poincare_jax_default"),
        close=True,
    )
    surf.to_vtk(str(out_dir / f"surf_{field_label}_poincare_jax_default"))


def extract_discrete_biot_savart_coils(bs):
    coils = list(bs.coils)
    if not coils:
        raise ValueError("BiotSavart field contains no coils")

    gamma_arrays = []
    gammadash_arrays = []
    currents = []
    for index, coil in enumerate(coils):
        gamma = np.asarray(coil.curve.gamma(), dtype=float)
        gammadash = np.asarray(coil.curve.gammadash(), dtype=float)
        if gamma.ndim != 2 or gamma.shape[1] != 3:
            raise ValueError(f"coil {index} gamma has invalid shape {gamma.shape}")
        if gammadash.shape != gamma.shape:
            raise ValueError(
                f"coil {index} gammadash shape {gammadash.shape} does not match "
                f"gamma shape {gamma.shape}"
            )
        gamma_arrays.append(gamma)
        gammadash_arrays.append(gammadash)
        currents.append(float(np.asarray(coil.current.get_value())))

    max_nquad = max(gamma.shape[0] for gamma in gamma_arrays)
    gamma_padded = np.zeros((len(coils), max_nquad, 3), dtype=float)
    gammadash_padded = np.zeros((len(coils), max_nquad, 3), dtype=float)
    quadrature_mask = np.zeros((len(coils), max_nquad), dtype=bool)
    for index, (gamma, gammadash) in enumerate(zip(gamma_arrays, gammadash_arrays)):
        nquad = gamma.shape[0]
        gamma_padded[index, :nquad, :] = gamma
        gammadash_padded[index, :nquad, :] = gammadash
        quadrature_mask[index, :nquad] = True

    return DiscreteBiotSavartCoils(
        gamma=gamma_padded,
        gammadash=gammadash_padded,
        quadrature_mask=quadrature_mask,
        currents=np.asarray(currents, dtype=float),
    )


def _biot_savart_points_jax(points, gamma, gammadash, quadrature_mask, currents):
    displacement = gamma[None, :, :, :] - points[:, None, None, :]
    r2 = jnp.sum(displacement * displacement, axis=-1)
    inv_r3 = jnp.where(r2 > 0.0, jnp.power(r2, -1.5), 0.0)
    integrand = jnp.cross(displacement, gammadash[None, :, :, :], axis=-1)
    weights = quadrature_mask[None, :, :, None]
    counts = jnp.sum(quadrature_mask, axis=1)
    coil_integrals = (
        jnp.sum(integrand * inv_r3[..., None] * weights, axis=2)
        / counts[None, :, None]
    )
    return MU0_OVER_4PI * jnp.sum(currents[None, :, None] * coil_integrals, axis=1)


def _cartesian_from_cylindrical(states, phi):
    r = states[:, 0]
    z = states[:, 1]
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    return jnp.stack((r * cos_phi, r * sin_phi, z), axis=1)


def _rhs_phi(
    states,
    phi,
    gamma,
    gammadash,
    quadrature_mask,
    currents,
    min_bphi_over_b,
):
    points = _cartesian_from_cylindrical(states, phi)
    field = _biot_savart_points_jax(
        points,
        gamma,
        gammadash,
        quadrature_mask,
        currents,
    )
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    b_r = field[:, 0] * cos_phi + field[:, 1] * sin_phi
    b_phi = -field[:, 0] * sin_phi + field[:, 1] * cos_phi
    b_z = field[:, 2]
    b_norm = jnp.linalg.norm(field, axis=1)
    valid = (
        jnp.isfinite(b_norm)
        & jnp.isfinite(b_phi)
        & (b_norm > 0.0)
        & ((jnp.abs(b_phi) / b_norm) > min_bphi_over_b)
    )
    safe_b_phi = jnp.where(jnp.abs(b_phi) > 0.0, b_phi, 1.0)
    deriv = jnp.stack(
        (
            states[:, 0] * b_r / safe_b_phi,
            states[:, 0] * b_z / safe_b_phi,
        ),
        axis=1,
    )
    return jnp.where(valid[:, None], deriv, 0.0), valid


def _rk4_step(
    states,
    phi,
    dphi,
    gamma,
    gammadash,
    quadrature_mask,
    currents,
    min_bphi_over_b,
):
    k1, valid1 = _rhs_phi(
        states,
        phi,
        gamma,
        gammadash,
        quadrature_mask,
        currents,
        min_bphi_over_b,
    )
    k2, valid2 = _rhs_phi(
        states + 0.5 * dphi * k1,
        phi + 0.5 * dphi,
        gamma,
        gammadash,
        quadrature_mask,
        currents,
        min_bphi_over_b,
    )
    k3, valid3 = _rhs_phi(
        states + 0.5 * dphi * k2,
        phi + 0.5 * dphi,
        gamma,
        gammadash,
        quadrature_mask,
        currents,
        min_bphi_over_b,
    )
    k4, valid4 = _rhs_phi(
        states + dphi * k3,
        phi + dphi,
        gamma,
        gammadash,
        quadrature_mask,
        currents,
        min_bphi_over_b,
    )
    next_states = states + (dphi / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    valid = valid1 & valid2 & valid3 & valid4 & jnp.all(jnp.isfinite(next_states), axis=1)
    return next_states, valid


def _box_guard_status(states, valid_field, rmin, rmax, zmax):
    r = states[:, 0]
    z = states[:, 1]
    finite_state = jnp.all(jnp.isfinite(states), axis=1)
    in_bounds = (
        finite_state
        & valid_field
        & (z <= zmax)
        & (z >= -zmax)
        & (r >= rmin)
        & (r <= rmax)
    )
    reason = jnp.full(
        r.shape,
        PHI_ODE_SINGULARITY_STOP_REASON,
        dtype=jnp.int32,
    )
    reason = jnp.where(z > zmax, 0, reason)
    reason = jnp.where(z < -zmax, 1, reason)
    reason = jnp.where(r < rmin, 2, reason)
    reason = jnp.where(r > rmax, 3, reason)
    return in_bounds, reason


@partial(
    jax.jit,
    static_argnames=("n_field_periods", "steps_per_quarter", "nfp"),
)
def _trace_default_mode_jax_kernel(
    seed_radii,
    z0,
    gamma,
    gammadash,
    quadrature_mask,
    currents,
    *,
    n_field_periods,
    steps_per_quarter,
    nfp,
    rmin,
    rmax,
    zmax,
    min_bphi_over_b,
):
    field_period = (2.0 * jnp.pi) / float(nfp)
    segment_dphi = field_period / 4.0
    dphi = segment_dphi / float(steps_per_quarter)
    nsegments = int(n_field_periods) * 4
    nlines = seed_radii.shape[0]

    states0 = jnp.stack((seed_radii, z0), axis=1)
    alive0 = jnp.ones((nlines,), dtype=bool)
    stop_time0 = jnp.full((nlines,), jnp.nan)
    stop_xyz0 = jnp.zeros((nlines, 3), dtype=states0.dtype)
    stop_reason0 = jnp.full((nlines,), -1, dtype=jnp.int32)
    phi0 = jnp.asarray(0.0, dtype=states0.dtype)

    def integrate_segment(carry, segment_index):
        states, alive, stop_time, stop_xyz, stop_reason, phi = carry
        hit_xyz = _cartesian_from_cylindrical(states, phi)
        hit_alive = alive
        hit_time = phi

        def integrate_step(step_carry, _):
            states_s, alive_s, stop_time_s, stop_xyz_s, stop_reason_s, phi_s = (
                step_carry
            )
            next_states, valid_field = _rk4_step(
                states_s,
                phi_s,
                dphi,
                gamma,
                gammadash,
                quadrature_mask,
                currents,
                min_bphi_over_b,
            )
            in_bounds, reason = _box_guard_status(
                next_states,
                valid_field,
                rmin,
                rmax,
                zmax,
            )
            next_alive_raw = alive_s & in_bounds
            newly_stopped = alive_s & ~next_alive_raw
            next_phi = phi_s + dphi
            next_xyz = _cartesian_from_cylindrical(next_states, next_phi)
            stop_time_next = jnp.where(newly_stopped, next_phi, stop_time_s)
            stop_xyz_next = jnp.where(newly_stopped[:, None], next_xyz, stop_xyz_s)
            stop_reason_next = jnp.where(newly_stopped, reason, stop_reason_s)
            return (
                next_states,
                next_alive_raw,
                stop_time_next,
                stop_xyz_next,
                stop_reason_next,
                next_phi,
            ), None

        next_carry, _ = jax.lax.scan(
            integrate_step,
            carry,
            xs=jnp.arange(steps_per_quarter),
        )
        return next_carry, (
            hit_xyz,
            hit_alive,
            hit_time,
            jnp.asarray(segment_index % 4, dtype=jnp.int32),
        )

    final_carry, hits = jax.lax.scan(
        integrate_segment,
        (states0, alive0, stop_time0, stop_xyz0, stop_reason0, phi0),
        xs=jnp.arange(nsegments),
    )
    final_states, final_alive, stop_time, stop_xyz, stop_reason, final_phi = final_carry
    hit_xyz, hit_alive, hit_time, hit_phi_index = hits
    return {
        "hit_xyz": hit_xyz,
        "hit_alive": hit_alive,
        "hit_time": hit_time,
        "hit_phi_index": hit_phi_index,
        "stop_time": stop_time,
        "stop_xyz": stop_xyz,
        "stop_reason": stop_reason,
        "final_states": final_states,
        "final_alive": final_alive,
        "final_phi": final_phi,
    }


def _rhs_arclength(states, gamma, gammadash, quadrature_mask, currents):
    """Arc-length field-line ODE ``dx/ds = B/|B|`` in cylindrical ``(R, phi, Z)``.

    Non-singular everywhere a field exists (``|B| > 0``, ``R > 0``) -- unlike the
    toroidal-angle RHS, there is no division by ``B_phi``, so trajectories near
    the low-``B_phi`` separatrix edge stay accurate.  ``states`` is ``(nlines, 3)``;
    returns ``(deriv (nlines, 3), valid (nlines,))`` with ``deriv`` zeroed where
    the field is invalid so a dead line freezes.
    """
    r = states[:, 0]
    phi = states[:, 1]
    z = states[:, 2]
    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)
    points = jnp.stack((r * cos_phi, r * sin_phi, z), axis=1)
    field = _biot_savart_points_jax(points, gamma, gammadash, quadrature_mask, currents)
    b_r = field[:, 0] * cos_phi + field[:, 1] * sin_phi
    b_phi = -field[:, 0] * sin_phi + field[:, 1] * cos_phi
    b_z = field[:, 2]
    b_norm = jnp.linalg.norm(field, axis=1)
    valid = jnp.isfinite(b_norm) & (b_norm > 0.0) & (r > 0.0)
    safe_norm = jnp.where(b_norm > 0.0, b_norm, 1.0)
    safe_r = jnp.where(r > 0.0, r, 1.0)
    deriv = jnp.stack(
        (b_r / safe_norm, b_phi / (safe_r * safe_norm), b_z / safe_norm),
        axis=1,
    )
    return jnp.where(valid[:, None], deriv, 0.0), valid


def _rk4_step_arclength(states, ds, gamma, gammadash, quadrature_mask, currents):
    """One fixed-step RK4 step of length ``ds`` (arc length) of ``_rhs_arclength``."""
    k1, v1 = _rhs_arclength(states, gamma, gammadash, quadrature_mask, currents)
    k2, v2 = _rhs_arclength(states + 0.5 * ds * k1, gamma, gammadash, quadrature_mask, currents)
    k3, v3 = _rhs_arclength(states + 0.5 * ds * k2, gamma, gammadash, quadrature_mask, currents)
    k4, v4 = _rhs_arclength(states + ds * k3, gamma, gammadash, quadrature_mask, currents)
    next_states = states + (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    valid = v1 & v2 & v3 & v4 & jnp.all(jnp.isfinite(next_states), axis=1)
    return next_states, valid


@partial(
    jax.jit,
    static_argnames=("n_field_periods", "max_inner_steps", "nfp"),
)
def _trace_arclength_mode_jax_kernel(
    seed_radii,
    z0,
    gamma,
    gammadash,
    quadrature_mask,
    currents,
    *,
    n_field_periods,
    max_inner_steps,
    nfp,
    rmin,
    rmax,
    zmax,
    ds,
):
    """Fixed-step arc-length tracer producing the SAME record contract as the
    toroidal-angle kernel.  Each outer segment emits the puncture on its
    Poincare plane (``phi == k*segment_dphi``), then integrates in arc length
    until ``phi`` reaches the next plane, interpolating the crossing.  The
    per-line ``phi`` is pinned to ``k*segment_dphi`` at every crossing, so the
    emitted ``hit_time`` is the same scalar-per-segment the toroidal kernel
    produced and ``jax_trace_records_to_simsopt_format`` is unchanged.
    """
    field_period = (2.0 * jnp.pi) / float(nfp)
    segment_dphi = field_period / 4.0
    nsegments = int(n_field_periods) * 4
    nlines = seed_radii.shape[0]

    phi_col = jnp.zeros((nlines,), dtype=seed_radii.dtype)
    states0 = jnp.stack((seed_radii, phi_col, z0), axis=1)
    alive0 = jnp.ones((nlines,), dtype=bool)
    stop_time0 = jnp.full((nlines,), jnp.nan)
    stop_xyz0 = jnp.zeros((nlines, 3), dtype=states0.dtype)
    stop_reason0 = jnp.full((nlines,), -1, dtype=jnp.int32)

    def integrate_segment(carry, segment_index):
        states, alive, stop_time, stop_xyz, stop_reason = carry
        rz = jnp.stack((states[:, 0], states[:, 2]), axis=1)
        seg_phi = jnp.asarray(segment_index, dtype=states.dtype) * segment_dphi
        hit_xyz = _cartesian_from_cylindrical(rz, seg_phi)
        hit_alive = alive
        hit_time = seg_phi
        target = seg_phi + segment_dphi

        def integrate_step(step_carry, _):
            s_states, s_alive, s_stime, s_sxyz, s_sreason = step_carry
            s_phi = s_states[:, 1]
            active = s_alive & (s_phi < target)
            next_states, valid_field = _rk4_step_arclength(
                s_states, ds, gamma, gammadash, quadrature_mask, currents
            )
            rz_next = jnp.stack((next_states[:, 0], next_states[:, 2]), axis=1)
            in_bounds, reason = _box_guard_status(rz_next, valid_field, rmin, rmax, zmax)
            newly_stopped = active & (~in_bounds)
            crossed = active & in_bounds & (next_states[:, 1] >= target)
            denom = next_states[:, 1] - s_phi
            frac = jnp.where(jnp.abs(denom) > 0.0, (target - s_phi) / denom, 0.0)
            cross_states = jnp.stack(
                (
                    s_states[:, 0] + frac * (next_states[:, 0] - s_states[:, 0]),
                    jnp.full_like(s_phi, target),
                    s_states[:, 2] + frac * (next_states[:, 2] - s_states[:, 2]),
                ),
                axis=1,
            )
            advanced = jnp.where(crossed[:, None], cross_states, next_states)
            out_states = jnp.where((active & in_bounds)[:, None], advanced, s_states)
            out_alive = s_alive & (~newly_stopped)
            stop_pt = _cartesian_from_cylindrical(rz_next, next_states[:, 1])
            s_stime2 = jnp.where(newly_stopped, next_states[:, 1], s_stime)
            s_sxyz2 = jnp.where(newly_stopped[:, None], stop_pt, s_sxyz)
            s_sreason2 = jnp.where(newly_stopped, reason, s_sreason)
            return (out_states, out_alive, s_stime2, s_sxyz2, s_sreason2), None

        next_carry, _ = jax.lax.scan(
            integrate_step,
            (states, alive, stop_time, stop_xyz, stop_reason),
            xs=jnp.arange(max_inner_steps),
        )
        return next_carry, (
            hit_xyz,
            hit_alive,
            hit_time,
            jnp.asarray(segment_index % 4, dtype=jnp.int32),
        )

    final_carry, hits = jax.lax.scan(
        integrate_segment,
        (states0, alive0, stop_time0, stop_xyz0, stop_reason0),
        xs=jnp.arange(nsegments),
    )
    final_states, final_alive, stop_time, stop_xyz, stop_reason = final_carry
    hit_xyz, hit_alive, hit_time, hit_phi_index = hits
    final_rz = jnp.stack((final_states[:, 0], final_states[:, 2]), axis=1)
    final_phi = jnp.asarray(nsegments, dtype=final_states.dtype) * segment_dphi
    return {
        "hit_xyz": hit_xyz,
        "hit_alive": hit_alive,
        "hit_time": hit_time,
        "hit_phi_index": hit_phi_index,
        "stop_time": stop_time,
        "stop_xyz": stop_xyz,
        "stop_reason": stop_reason,
        "final_states": final_rz,
        "final_alive": final_alive,
        "final_phi": final_phi,
    }


def jax_trace_records_to_simsopt_format(
    trace_result,
    *,
    nfp,
):
    hit_xyz = np.asarray(trace_result["hit_xyz"], dtype=float)
    hit_alive = np.asarray(trace_result["hit_alive"], dtype=bool)
    hit_time = np.asarray(trace_result["hit_time"], dtype=float)
    hit_phi_index = np.asarray(trace_result["hit_phi_index"], dtype=int)
    stop_time = np.asarray(trace_result["stop_time"], dtype=float)
    stop_xyz = np.asarray(trace_result["stop_xyz"], dtype=float)
    stop_reason = np.asarray(trace_result["stop_reason"], dtype=int)
    final_states = np.asarray(trace_result["final_states"], dtype=float)
    final_phi = float(np.asarray(trace_result["final_phi"]))

    final_xyz = np.asarray(
        _cartesian_from_cylindrical(jnp.asarray(final_states), jnp.asarray(final_phi)),
        dtype=float,
    )
    fieldlines_tys = []
    fieldlines_phi_hits = []
    for line_index in range(hit_xyz.shape[1]):
        rows = []
        for segment_index in range(hit_xyz.shape[0]):
            if not hit_alive[segment_index, line_index]:
                continue
            xyz = hit_xyz[segment_index, line_index]
            rows.append(
                [
                    float(hit_time[segment_index]),
                    float(hit_phi_index[segment_index]),
                    float(xyz[0]),
                    float(xyz[1]),
                    float(xyz[2]),
                ]
            )

        if stop_reason[line_index] >= 0:
            xyz = stop_xyz[line_index]
            rows.append(
                [
                    float(stop_time[line_index]),
                    float(-stop_reason[line_index] - 1),
                    float(xyz[0]),
                    float(xyz[1]),
                    float(xyz[2]),
                ]
            )
            final_row = [
                float(stop_time[line_index]),
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
            ]
        else:
            xyz = final_xyz[line_index]
            final_row = [
                float(final_phi),
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
            ]

        fieldlines_tys.append(np.asarray([final_row], dtype=float))
        fieldlines_phi_hits.append(np.asarray(rows, dtype=float).reshape((-1, 5)))

    phis = [(i / 4.0) * (2.0 * np.pi / int(nfp)) for i in range(4)]
    return fieldlines_tys, fieldlines_phi_hits, phis


def trace_default_mode_jax(
    coil_data,
    render_mode,
    *,
    n_field_periods,
    steps_per_field_period,
    nfp,
    min_bphi_over_b,
    integrator="phi",
):
    steps_per_quarter = validate_jax_trace_settings(
        n_field_periods,
        steps_per_field_period,
    )
    trace_domain = render_mode["trace_domain"]
    seed_radii = jnp.asarray(render_mode["radii"], dtype=jnp.float64)
    z0 = jnp.asarray(render_mode["z0"], dtype=jnp.float64)
    gamma = jnp.asarray(coil_data.gamma, dtype=jnp.float64)
    gammadash = jnp.asarray(coil_data.gammadash, dtype=jnp.float64)
    quadrature_mask = jnp.asarray(coil_data.quadrature_mask, dtype=jnp.float64)
    currents = jnp.asarray(coil_data.currents, dtype=jnp.float64)

    start = time.perf_counter()
    if integrator == "arclength":
        segment_dphi = (2.0 * np.pi / float(nfp)) / 4.0
        r0 = float(np.mean(np.asarray(render_mode["radii"])))
        ds = float(segment_dphi * r0 / float(steps_per_quarter))
        max_inner_steps = int(4 * steps_per_quarter)
        trace_result = _trace_arclength_mode_jax_kernel(
            seed_radii,
            z0,
            gamma,
            gammadash,
            quadrature_mask,
            currents,
            n_field_periods=int(n_field_periods),
            max_inner_steps=max_inner_steps,
            nfp=int(nfp),
            rmin=float(trace_domain.stopping_rmin),
            rmax=float(trace_domain.stopping_rmax),
            zmax=float(trace_domain.stopping_zmax),
            ds=ds,
        )
    else:
        trace_result = _trace_default_mode_jax_kernel(
            seed_radii,
            z0,
            gamma,
            gammadash,
            quadrature_mask,
            currents,
            n_field_periods=int(n_field_periods),
            steps_per_quarter=int(steps_per_quarter),
            nfp=int(nfp),
            rmin=float(trace_domain.stopping_rmin),
            rmax=float(trace_domain.stopping_rmax),
            zmax=float(trace_domain.stopping_zmax),
            min_bphi_over_b=float(min_bphi_over_b),
        )
    trace_result = jax.tree.map(lambda value: np.asarray(value), trace_result)
    elapsed_s = time.perf_counter() - start
    fieldlines_tys, fieldlines_phi_hits, phis = jax_trace_records_to_simsopt_format(
        trace_result,
        nfp=nfp,
    )
    return fieldlines_tys, fieldlines_phi_hits, phis, elapsed_s


def build_default_render_modes(surf, nfieldlines, tmax_reference):
    iteration_limit = topology_iteration_limit(float(tmax_reference))
    guarded_stopping_criteria, guarded_stop_labels = build_stopping_criteria(
        surf,
        include_surface_exit=True,
        max_iterations=iteration_limit,
    )
    default_stopping_domain = extended_surface_trace_domain(
        surf,
        DEFAULT_EXTEND_DISTANCE,
    )
    default_stopping_criteria, default_stop_labels = build_stopping_criteria(
        surf,
        include_surface_exit=False,
        max_iterations=iteration_limit,
        trace_domain=default_stopping_domain,
    )
    render_modes, _ = build_poincare_render_modes(
        surf,
        int(nfieldlines),
        seed_inset_fraction=SEED_INSET_FRACTION,
        default_extend_distance=DEFAULT_EXTEND_DISTANCE,
        guarded_stopping_criteria=guarded_stopping_criteria,
        guarded_stop_labels=guarded_stop_labels,
        default_stopping_criteria=default_stopping_criteria,
        default_stop_labels=default_stop_labels,
    )
    default_render_mode = next(
        render_mode for render_mode in render_modes if render_mode["mode"] == "default"
    )
    default_render_mode = {
        **default_render_mode,
        "stop_labels": JAX_DEFAULT_STOP_LABELS,
        "jax_additional_stop_labels": [
            JAX_DEFAULT_STOP_LABELS[PHI_ODE_SINGULARITY_STOP_REASON]
        ],
    }
    return [
        default_render_mode if render_mode["mode"] == "default" else render_mode
        for render_mode in render_modes
    ], default_render_mode


def build_default_render_mode(surf, nfieldlines, tmax_reference):
    _, default_render_mode = build_default_render_modes(
        surf,
        nfieldlines,
        tmax_reference,
    )
    return default_render_mode


def build_field_model_metadata(
    *,
    coil_data,
    render_mode,
    n_field_periods,
    steps_per_field_period,
    min_bphi_over_b,
    elapsed_s,
    cpu_tmax_reference,
    phi_horizon,
):
    return {
        "policy": "native",
        "selected_mode": "jax_discrete_biot_savart_fixed_phi_rk4",
        "reason": "opt-in JAX default-mode Poincare throughput tracer",
        "poincare_field_key": render_mode["field_key"],
        "poincare_trace_semantics": render_mode["trace_semantics"],
        "time_parameterization": "toroidal_angle",
        "cpu_tmax_reference": float(cpu_tmax_reference),
        "phi_horizon": float(phi_horizon),
        "n_field_periods": int(n_field_periods),
        "steps_per_field_period": int(steps_per_field_period),
        "rk_order": 4,
        "min_bphi_over_b": float(min_bphi_over_b),
        "coil_count": coil_data.ncoils,
        "coil_quadrature_points_max": coil_data.nquadpoints,
        "coil_quadrature_points_min": int(np.min(coil_data.quadrature_counts)),
        "coil_quadrature_points_by_coil": [
            int(count) for count in coil_data.quadrature_counts
        ],
        "coil_quadrature_rule": (
            "masked per-coil mean over curve.gamma()/curve.gammadash() samples"
        ),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "trace_wall_seconds": float(elapsed_s),
    }


def run_jax_default_poincare(args):
    configure_jax_platform(args.jax_platform)
    out_dir = resolve_output_dir()
    bs, surf, field_label, results_meta, allow_design_only_field = load_field_and_surface(
        out_dir
    )
    if not args.no_vtk:
        export_reference_geometry(bs, surf, out_dir, field_label)

    all_render_modes, render_mode = build_default_render_modes(
        surf,
        args.nfieldlines,
        args.cpu_tmax_reference,
    )
    assert_no_stale_unselected_metrics(
        out_dir,
        field_label,
        all_render_modes,
        [render_mode],
    )
    coil_data = extract_discrete_biot_savart_coils(bs)
    print(
        "Tracing default Poincare mode with JAX "
        f"({args.field_periods} field periods, "
        f"{args.steps_per_field_period} RK4 steps/field-period, "
        f"{coil_data.ncoils} coils x {coil_data.nquadpoints} quadpoints)..."
    )
    fieldlines_tys, fieldlines_phi_hits, phis, elapsed_s = trace_default_mode_jax(
        coil_data,
        render_mode,
        n_field_periods=args.field_periods,
        steps_per_field_period=args.steps_per_field_period,
        nfp=surf.nfp,
        min_bphi_over_b=args.min_bphi_over_b,
        integrator=args.integrator,
    )
    metrics = _trace_metrics(
        fieldlines_tys,
        fieldlines_phi_hits,
        phis,
        render_mode["stop_labels"],
        "default",
        iota=(results_meta or {}).get("FINAL_IOTA"),
        surface_resolution=(len(surf.quadpoints_phi), len(surf.quadpoints_theta)),
    )
    plot_filename = out_dir / f"PoincarePlot_{field_label}_default.png"
    metrics["plot_filename"] = plot_filename.name
    metrics["seed_contract"] = render_mode["seed_contract"]
    metrics["trace_semantics"] = render_mode["trace_semantics"]
    metrics["jax_trace_wall_seconds"] = float(elapsed_s)
    metrics["field_periods"] = int(args.field_periods)
    metrics["steps_per_field_period"] = int(args.steps_per_field_period)
    plot_poincare_data(
        fieldlines_phi_hits,
        phis,
        str(plot_filename),
        dpi=args.dpi,
        surf=surf,
        mark_lost=False,
    )
    print(f"Saved: {plot_filename.name}")

    phi_horizon = args.field_periods * (2.0 * np.pi / float(surf.nfp))
    field_model = build_field_model_metadata(
        coil_data=coil_data,
        render_mode=render_mode,
        n_field_periods=args.field_periods,
        steps_per_field_period=args.steps_per_field_period,
        min_bphi_over_b=args.min_bphi_over_b,
        elapsed_s=elapsed_s,
        cpu_tmax_reference=args.cpu_tmax_reference,
        phi_horizon=phi_horizon,
    )
    metrics_sidecar_path = out_dir / f"PoincareMetrics_{field_label}_default.json"
    mode_artifact = build_poincare_mode_artifact(
        field_label=field_label,
        render_mode=render_mode,
        nfieldlines=args.nfieldlines,
        tmax=phi_horizon,
        tol=args.tol_reference,
        phis=phis,
        field_model=field_model,
        metrics=metrics,
        design_only_override=allow_design_only_field,
    )
    mode_artifact["cpu_default_reference"] = {
        "script": "poincare_surfaces.py",
        "mode": "default",
        "tmax": float(args.cpu_tmax_reference),
        "tol": float(args.tol_reference),
    }
    mode_artifact["jax_default_runner"] = {
        "script": Path(__file__).name,
        "fixed_step_integrator": "RK4",
        "time_parameterization": "toroidal_angle",
    }
    with open(metrics_sidecar_path, "w", encoding="utf-8") as output_file:
        json.dump(mode_artifact, output_file, indent=2)
    print(
        f"Saved: {metrics_sidecar_path.name} "
        f"(phi hit counts={metrics['per_phi_hit_counts']}; "
        f"status={metrics['validation_status']}; "
        f"survival={metrics['survived_lines']}/{metrics['nfieldlines']}; "
        f"wall={elapsed_s:.3f}s; backend={field_model['jax_backend']})"
    )
    return mode_artifact


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the default Poincare plot contract with a JAX RK4 tracer."
    )
    parser.add_argument(
        "--nfieldlines",
        type=int,
        default=_env_int("POINCARE_JAX_NFIELDLINES", DEFAULT_NFIELDLINES),
    )
    parser.add_argument(
        "--field-periods",
        type=int,
        default=_env_int("POINCARE_JAX_FIELD_PERIODS", DEFAULT_JAX_FIELD_PERIODS),
        help="Number of toroidal field periods to trace.",
    )
    parser.add_argument(
        "--steps-per-field-period",
        type=int,
        default=_env_int(
            "POINCARE_JAX_STEPS_PER_FIELD_PERIOD",
            DEFAULT_JAX_STEPS_PER_FIELD_PERIOD,
        ),
        help="RK4 steps per field period; must be divisible by 4.",
    )
    parser.add_argument(
        "--integrator",
        choices=["phi", "arclength"],
        default=os.environ.get("POINCARE_JAX_INTEGRATOR", "phi"),
        help=(
            "Field-line integrator: 'phi' (fast fixed-step toroidal-angle RK4, "
            "the GPU-throughput default) or 'arclength' (non-singular fixed-step "
            "arc-length RK4 matching the C++ compute_fieldlines trajectory near "
            "the low-B_phi separatrix edge)."
        ),
    )
    parser.add_argument(
        "--min-bphi-over-b",
        type=float,
        default=_env_float("POINCARE_JAX_MIN_BPHI_OVER_B", DEFAULT_MIN_BPHI_OVER_B),
    )
    parser.add_argument(
        "--jax-platform",
        default=os.environ.get("POINCARE_JAX_PLATFORM", ""),
        help="Optional JAX platform override, e.g. gpu on Perlmutter.",
    )
    parser.add_argument(
        "--cpu-tmax-reference",
        type=float,
        default=_env_float("POINCARE_TMAX", DEFAULT_CPU_TMAX_REFERENCE),
        help="Reference tmax recorded for parity with poincare_surfaces.py metadata.",
    )
    parser.add_argument(
        "--tol-reference",
        type=float,
        default=_env_float("POINCARE_TOL", DEFAULT_TOL_REFERENCE),
        help="Reference adaptive tolerance recorded for parity metadata.",
    )
    parser.add_argument("--dpi", type=int, default=_env_int("POINCARE_JAX_DPI", 600))
    parser.add_argument("--no-vtk", action="store_true")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_jax_trace_settings(args.field_periods, args.steps_per_field_period)
    return run_jax_default_poincare(args)


if __name__ == "__main__":
    main()
