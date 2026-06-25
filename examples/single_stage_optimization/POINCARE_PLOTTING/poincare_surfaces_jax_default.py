"""JAX fixed-step tracer for the default Poincare plot contract.

This script is intentionally separate from ``poincare_surfaces.py``.  The
existing script remains the adaptive SIMSOPT reference path; this one is an
opt-in throughput path for the default render only:

* extended-surface seed radii,
* native Biot-Savart field, not an interpolated field,
* box guardrails plus a toroidal-angle ODE singularity guard,
* four Poincare planes per field period.

The integrator uses toroidal angle as the independent variable and a fixed-step
RK4 scan.  Host-side artifact loading and plotting remain ordinary Python so
the expensive field-line loop can run on a JAX backend without changing the
existing default CPU plotting script.
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
from simsopt._core.optimizable import load
from simsopt.geo import BoozerSurface, curves_to_vtk

jax.config.update("jax_enable_x64", True)


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from poincare_surfaces import plot_poincare_data  # noqa: E402


MU0_OVER_4PI = 1e-7
DEFAULT_CPU_TMAX_REFERENCE = 7000.0
DEFAULT_TOL_REFERENCE = 1e-7
DEFAULT_NFIELDLINES = 50
DEFAULT_JAX_FIELD_PERIODS = 64
DEFAULT_JAX_STEPS_PER_FIELD_PERIOD = 64
DEFAULT_MIN_BPHI_OVER_B = 1e-10
DEFAULT_EXTEND_DISTANCE = 0.05
STOP_LABELS = (
    "max_z_guardrail",
    "min_z_guardrail",
    "min_r_guardrail",
    "max_r_guardrail",
    "phi_ode_singularity",
)
PHI_ODE_SINGULARITY_STOP_REASON = STOP_LABELS.index("phi_ode_singularity")


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


@dataclasses.dataclass(frozen=True)
class DefaultTraceDomain:
    """Extended-surface radial/vertical bounds for the default plot guardrails."""

    seed_rmin: float
    seed_rmax: float
    stopping_rmin: float
    stopping_rmax: float
    stopping_zmax: float

    def as_metadata(self):
        return dataclasses.asdict(self)


def _env_int(name, default):
    value = os.environ.get(name)
    return int(default if value is None else value)


def _env_float(name, default):
    value = os.environ.get(name)
    return float(default if value is None else value)


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off")


def _plain_surface(surface):
    if isinstance(surface, BoozerSurface):
        return surface.surface
    return surface


def resolve_output_dir(env=None):
    environ = os.environ if env is None else env
    if environ.get("POINCARE_OUT_DIR"):
        return Path(environ["POINCARE_OUT_DIR"]).resolve()
    outputs_root = EXAMPLE_ROOT / "SINGLE_STAGE" / "outputs"
    candidates = sorted(
        glob.glob(str(outputs_root / "mpol=*")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No single-stage output found in {outputs_root}. Set POINCARE_OUT_DIR."
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


def load_field_and_surface(out_dir, *, use_opt=False):
    opt_bs_path = out_dir / "biot_savart_opt.json"
    opt_surf_path = Path(
        os.environ.get("POINCARE_OPT_SURF_PATH", str(out_dir / "surf_opt.json"))
    )
    init_bs_path = out_dir / "biot_savart_init.json"
    init_surf_path = out_dir / "surf_init.json"

    if use_opt:
        missing_opt_paths = [
            str(path)
            for path in (opt_bs_path, opt_surf_path)
            if not path.exists()
        ]
        if missing_opt_paths:
            raise FileNotFoundError(
                "--use-opt requested, but optimized Poincare input is missing: "
                + ", ".join(missing_opt_paths)
            )
        bs = load(str(opt_bs_path))
        surf = _plain_surface(load(str(opt_surf_path)))
        field_label = "opt"
        surface_path = opt_surf_path
        print("Loaded OPTIMIZED field + surface")
    else:
        bs = load(str(init_bs_path))
        surf = _plain_surface(load(str(init_surf_path)))
        field_label = "init"
        surface_path = init_surf_path
        print("Loaded INITIAL field + surface")
    return bs, surf, field_label, surface_path


def export_reference_geometry(bs, surf, out_dir, field_label):
    curves_to_vtk(
        [coil.curve for coil in bs.coils],
        str(out_dir / f"curves_{field_label}_poincare_jax_default"),
        close=True,
    )
    surf.to_vtk(str(out_dir / f"surf_{field_label}_poincare_jax_default"))


def build_default_trace_domain(surface_path, extend_distance=DEFAULT_EXTEND_DISTANCE):
    surf_extended = _plain_surface(load(str(surface_path)))
    surf_extended.extend_via_normal(float(extend_distance))
    gamma = np.asarray(surf_extended.gamma(), dtype=float)
    radii = np.sqrt(gamma[:, :, 0] ** 2 + gamma[:, :, 1] ** 2)
    z = gamma[:, :, 2]
    seed_rmin = float(np.min(radii))
    seed_rmax = float(np.max(radii))
    zmax = float(np.max(z))
    return DefaultTraceDomain(
        seed_rmin=seed_rmin,
        seed_rmax=seed_rmax,
        stopping_rmin=seed_rmin * 0.95,
        stopping_rmax=seed_rmax * 1.05,
        stopping_zmax=zmax * 1.05,
    )


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


def jax_trace_records_to_simsopt_format(trace_result, *, nfp):
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
    seed_radii,
    trace_domain,
    *,
    n_field_periods,
    steps_per_field_period,
    nfp,
    min_bphi_over_b,
):
    steps_per_quarter = validate_jax_trace_settings(
        n_field_periods,
        steps_per_field_period,
    )
    seed_radii = jnp.asarray(seed_radii, dtype=jnp.float64)
    z0 = jnp.zeros(seed_radii.shape, dtype=jnp.float64)
    gamma = jnp.asarray(coil_data.gamma, dtype=jnp.float64)
    gammadash = jnp.asarray(coil_data.gammadash, dtype=jnp.float64)
    quadrature_mask = jnp.asarray(coil_data.quadrature_mask, dtype=jnp.float64)
    currents = jnp.asarray(coil_data.currents, dtype=jnp.float64)

    start = time.perf_counter()
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
    trace_result = jax.tree_util.tree_map(lambda value: np.asarray(value), trace_result)
    elapsed_s = time.perf_counter() - start
    fieldlines_tys, fieldlines_phi_hits, phis = jax_trace_records_to_simsopt_format(
        trace_result,
        nfp=nfp,
    )
    return fieldlines_tys, fieldlines_phi_hits, phis, trace_result, elapsed_s


def trace_metrics(fieldlines_phi_hits, trace_result, phis, nfieldlines):
    per_phi_hit_counts = [
        int(
            sum(
                np.count_nonzero(line_hits[:, 1] == phi_index)
                for line_hits in fieldlines_phi_hits
            )
        )
        for phi_index in range(len(phis))
    ]
    stop_reason = np.asarray(trace_result["stop_reason"], dtype=int)
    final_alive = np.asarray(trace_result["final_alive"], dtype=bool)
    stop_reason_counts = {
        label: int(np.count_nonzero(stop_reason == index))
        for index, label in enumerate(STOP_LABELS)
    }
    return {
        "nfieldlines": int(nfieldlines),
        "survived_lines": int(np.count_nonzero(final_alive)),
        "lost_lines": int(int(nfieldlines) - np.count_nonzero(final_alive)),
        "per_phi_hit_counts": per_phi_hit_counts,
        "stop_reason_counts": stop_reason_counts,
        "validation_status": "diagnostic_only",
    }


def build_field_model_metadata(
    *,
    coil_data,
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
    bs, surf, field_label, surface_path = load_field_and_surface(
        out_dir,
        use_opt=args.use_opt,
    )
    if not args.no_vtk:
        export_reference_geometry(bs, surf, out_dir, field_label)

    trace_domain = build_default_trace_domain(surface_path, args.extend_distance)
    seed_radii = np.linspace(
        trace_domain.seed_rmin,
        trace_domain.seed_rmax,
        int(args.nfieldlines),
    )
    coil_data = extract_discrete_biot_savart_coils(bs)
    print(
        "Tracing default Poincare mode with JAX "
        f"({args.field_periods} field periods, "
        f"{args.steps_per_field_period} RK4 steps/field-period, "
        f"{coil_data.ncoils} coils x {coil_data.nquadpoints} quadpoints)..."
    )
    fieldlines_tys, fieldlines_phi_hits, phis, trace_result, elapsed_s = (
        trace_default_mode_jax(
            coil_data,
            seed_radii,
            trace_domain,
            n_field_periods=args.field_periods,
            steps_per_field_period=args.steps_per_field_period,
            nfp=surf.nfp,
            min_bphi_over_b=args.min_bphi_over_b,
        )
    )
    metrics = trace_metrics(
        fieldlines_phi_hits,
        trace_result,
        phis,
        args.nfieldlines,
    )
    metrics["jax_trace_wall_seconds"] = float(elapsed_s)
    metrics["field_periods"] = int(args.field_periods)
    metrics["steps_per_field_period"] = int(args.steps_per_field_period)

    plot_filename = out_dir / f"PoincarePlot_{field_label}_default_jax.png"
    metrics["plot_filename"] = plot_filename.name
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
        n_field_periods=args.field_periods,
        steps_per_field_period=args.steps_per_field_period,
        min_bphi_over_b=args.min_bphi_over_b,
        elapsed_s=elapsed_s,
        cpu_tmax_reference=args.cpu_tmax_reference,
        phi_horizon=phi_horizon,
    )
    artifact = {
        "schema_version": "poincare_jax_default_v1",
        "field_label": field_label,
        "use_optimized_input": bool(args.use_opt),
        "render_mode": "default",
        "nfieldlines": int(args.nfieldlines),
        "tmax": float(phi_horizon),
        "tol": float(args.tol_reference),
        "phis": [float(phi) for phi in phis],
        "seed_contract": {
            "mode": "extended_surface_radial_sweep",
            "nplanes": 1,
            "nfieldlines": int(args.nfieldlines),
            "extend_distance": float(args.extend_distance),
            "phi": 0.0,
            "Z": 0.0,
            "radial_sampling_source": "global_extended_surface_bounds",
            "r_min_seed": float(trace_domain.seed_rmin),
            "r_max_seed": float(trace_domain.seed_rmax),
        },
        "trace_domain": trace_domain.as_metadata(),
        "stop_labels": list(STOP_LABELS),
        "field_model": field_model,
        "trace_semantics": "baseline_wander",
        "plot_filename": metrics["plot_filename"],
        "metrics": metrics,
        "validation_status": metrics["validation_status"],
        "cpu_default_reference": {
            "script": "poincare_surfaces.py",
            "output": "PoincarePlot.png",
            "tmax": float(args.cpu_tmax_reference),
            "tol": float(args.tol_reference),
        },
        "jax_default_runner": {
            "script": Path(__file__).name,
            "fixed_step_integrator": "RK4",
            "time_parameterization": "toroidal_angle",
        },
    }
    metrics_sidecar_path = out_dir / f"PoincareMetrics_{field_label}_default_jax.json"
    with open(metrics_sidecar_path, "w", encoding="utf-8") as output_file:
        json.dump(artifact, output_file, indent=2)
    print(
        f"Saved: {metrics_sidecar_path.name} "
        f"(phi hit counts={metrics['per_phi_hit_counts']}; "
        f"status={metrics['validation_status']}; "
        f"survival={metrics['survived_lines']}/{metrics['nfieldlines']}; "
        f"wall={elapsed_s:.3f}s; backend={field_model['jax_backend']})"
    )
    return artifact


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
        "--min-bphi-over-b",
        type=float,
        default=_env_float("POINCARE_JAX_MIN_BPHI_OVER_B", DEFAULT_MIN_BPHI_OVER_B),
    )
    parser.add_argument(
        "--extend-distance",
        type=float,
        default=_env_float("POINCARE_EXTEND_DISTANCE", DEFAULT_EXTEND_DISTANCE),
    )
    parser.add_argument(
        "--jax-platform",
        default=os.environ.get("POINCARE_JAX_PLATFORM", ""),
        help="Optional JAX platform override, e.g. gpu on Perlmutter.",
    )
    parser.add_argument(
        "--use-opt",
        action="store_true",
        default=_env_bool("POINCARE_JAX_USE_OPT", False),
        help=(
            "Use biot_savart_opt.json and surf_opt.json. The default matches "
            "poincare_surfaces.py and uses the init pair."
        ),
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
