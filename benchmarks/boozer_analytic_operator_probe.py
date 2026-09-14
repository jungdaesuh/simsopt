"""Fixed-state native/JAX analytic Boozer operator benchmark.

This probe isolates operator assembly.  NCSX geometry, magnetic-field values,
field derivatives, and the surface coefficient basis are prepared once by the
native C++ path and then passed byte-for-byte to both operators.  It measures
the exact residual/Jacobian and the least-squares value/gradient/Hessian; it
does not run a Boozer solve or an end-to-end optimization.  The native exact
scalar oracle is checked separately and excluded from residual/Jacobian timing.

The default exact case has 253 ``SurfaceXYZTensorFourier`` coefficients
(``mpol=ntor=6``) on a 13x13 grid.  The default LS case has 661 coefficients
(``mpol=ntor=10``) on a configurable 255x64 grid.  ``--small`` supplies a
bounded smoke configuration for local validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _requested_platform(argv: list[str]) -> str | None:
    """Read the platform flag before importing JAX."""

    for index, argument in enumerate(argv):
        if argument == "--platform" and index + 1 < len(argv):
            return argv[index + 1]
        if argument.startswith("--platform="):
            return argument.partition("=")[2]
    return None


REQUESTED_PLATFORM = _requested_platform(sys.argv[1:])
if REQUESTED_PLATFORM in ("cpu", "cuda"):
    os.environ["JAX_PLATFORMS"] = REQUESTED_PLATFORM

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import simsoptpp as sopp
from numpy.typing import NDArray
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt_jax.geo.boozer_analytic_hessian import (
    boozer_residual_analytic_value_grad_hessian,
)
from simsopt_jax.geo.boozer_residual import boozer_residual_vector_and_jacobian

jax.config.update("jax_enable_x64", True)


def _validate_requested_platform() -> None:
    """Reject a run whose explicit platform flag does not match JAX."""

    if REQUESTED_PLATFORM in ("cpu", "cuda"):
        actual_platform = jax.default_backend()
        expected_backend = "gpu" if REQUESTED_PLATFORM == "cuda" else "cpu"
        if actual_platform != expected_backend:
            raise RuntimeError(
                f"requested --platform {REQUESTED_PLATFORM!r}, "
                f"but JAX selected {actual_platform!r}"
            )
        if REQUESTED_PLATFORM == "cuda":
            cuda_devices = [
                device
                for device in jax.devices()
                if device.platform == "gpu"
                and "cuda" in str(device.client.platform_version).lower()
            ]
            if not cuda_devices:
                raise RuntimeError(
                    "requested --platform 'cuda', but no CUDA-backed JAX GPU "
                    "device was selected"
                )


FloatArray = NDArray[np.float64]
CaseName = Literal["exact", "ls"]
WeightName = Literal["unweighted", "inv_modB"]
JaxOperator = Callable[..., tuple[jax.Array, ...]]


@dataclass(frozen=True, slots=True)
class CaseConfiguration:
    """Immutable resolution and quadrature settings for one operator case."""

    name: CaseName
    mpol: int
    ntor: int
    nphi: int
    ntheta: int


@dataclass(frozen=True, slots=True)
class FixedState:
    """Native-prepared arrays shared by both operator implementations."""

    configuration: CaseConfiguration
    nfp: int
    G: float
    iota: float
    B: FloatArray
    dB_dX: FloatArray
    d2B_dXdX: FloatArray
    xphi: FloatArray
    xtheta: FloatArray
    dx_ds: FloatArray
    dxphi_ds: FloatArray
    dxtheta_ds: FloatArray
    surface_dofs: FloatArray
    preparation_s: float
    coil_count: int
    min_modB: float

    @property
    def npoints(self) -> int:
        return self.configuration.nphi * self.configuration.ntheta

    @property
    def nresidual(self) -> int:
        return 3 * self.npoints

    @property
    def nsurface(self) -> int:
        return int(self.dx_ds.shape[-1])

    @property
    def nvariables(self) -> int:
        return self.nsurface + 2


@dataclass(frozen=True, slots=True)
class NativeExactResult:
    residual: FloatArray
    jacobian: FloatArray


@dataclass(frozen=True, slots=True)
class NativeLsResult:
    value_raw: float
    gradient_raw: FloatArray
    hessian_raw: FloatArray


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark configuration without constructing scientific state."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("exact", "ls", "both"),
        default="both",
        help="Operator lane to measure (default: both).",
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform selection; use cpu for the bounded local checks.",
    )
    parser.add_argument(
        "--weights",
        choices=("unweighted", "inv_modB", "both"),
        default="both",
        help="Weighting lane(s): raw residual or 1/|B| (default: both).",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Use a small exact 5x5 and LS 9x8 smoke configuration.",
    )
    parser.add_argument("--exact-mpol", type=_positive_int, default=None)
    parser.add_argument("--exact-ntor", type=_positive_int, default=None)
    parser.add_argument("--exact-nphi", type=_positive_int, default=None)
    parser.add_argument("--exact-ntheta", type=_positive_int, default=None)
    parser.add_argument("--ls-mpol", type=_positive_int, default=None)
    parser.add_argument("--ls-ntor", type=_positive_int, default=None)
    parser.add_argument("--ls-nphi", type=_positive_int, default=None)
    parser.add_argument("--ls-ntheta", type=_positive_int, default=None)
    parser.add_argument(
        "--quadrature-tile-size",
        type=_positive_int,
        default=64,
        help="LS analytic tile size for the JAX Hessian (default: 64).",
    )
    parser.add_argument(
        "--repeat",
        type=_positive_int,
        default=3,
        help="Number of warmed operator timings (default: 3).",
    )
    parser.add_argument(
        "--warmup",
        type=_positive_int,
        default=1,
        help="Warmed calls before recording repeats (default: 1).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the structured JSON result.",
    )
    return parser.parse_args(argv)


def _case_configuration(args: argparse.Namespace, name: CaseName) -> CaseConfiguration:
    """Resolve one case while keeping full-scale defaults explicit."""

    if args.small:
        defaults = {
            "exact": (1, 1, 5, 5),
            "ls": (2, 2, 9, 8),
        }
    else:
        defaults = {
            "exact": (6, 6, 13, 13),
            "ls": (10, 10, 255, 64),
        }
    mpol, ntor, nphi, ntheta = defaults[name]
    overrides = {
        "mpol": getattr(args, f"{name}_mpol"),
        "ntor": getattr(args, f"{name}_ntor"),
        "nphi": getattr(args, f"{name}_nphi"),
        "ntheta": getattr(args, f"{name}_ntheta"),
    }
    return CaseConfiguration(
        name=name,
        mpol=int(overrides["mpol"] if overrides["mpol"] is not None else mpol),
        ntor=int(overrides["ntor"] if overrides["ntor"] is not None else ntor),
        nphi=int(overrides["nphi"] if overrides["nphi"] is not None else nphi),
        ntheta=int(overrides["ntheta"] if overrides["ntheta"] is not None else ntheta),
    )


def _selected_cases(mode: str) -> tuple[CaseName, ...]:
    if mode == "exact":
        return ("exact",)
    if mode == "ls":
        return ("ls",)
    return ("exact", "ls")


def _selected_weights(weights: str) -> tuple[WeightName, ...]:
    if weights == "unweighted":
        return ("unweighted",)
    if weights == "inv_modB":
        return ("inv_modB",)
    return ("unweighted", "inv_modB")


def _as_f64(values: object) -> FloatArray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _array_sha256(values: FloatArray) -> str:
    packed = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(packed.tobytes(order="C")).hexdigest()


def _array_inventory(state: FixedState) -> dict[str, object]:
    arrays = {
        "B": state.B,
        "dB_dX": state.dB_dX,
        "d2B_dXdX": state.d2B_dXdX,
        "xphi": state.xphi,
        "xtheta": state.xtheta,
        "dx_ds": state.dx_ds,
        "dxphi_ds": state.dxphi_ds,
        "dxtheta_ds": state.dxtheta_ds,
        "surface_dofs": state.surface_dofs,
    }
    return {
        name: {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "order": "C" if values.flags.c_contiguous else "non-C",
            "sha256": _array_sha256(values),
        }
        for name, values in arrays.items()
    }


def _build_surface(
    configuration: CaseConfiguration,
) -> tuple[SurfaceXYZTensorFourier, BiotSavart, int, int, float]:
    """Construct the existing NCSX surface fixture and native coil field."""

    _base_curves, base_currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    surface = SurfaceXYZTensorFourier(
        nfp=int(nfp),
        stellsym=True,
        mpol=configuration.mpol,
        ntor=configuration.ntor,
        quadpoints_phi=np.linspace(
            0.0,
            1.0 / int(nfp),
            configuration.nphi,
            endpoint=False,
        ),
        quadpoints_theta=np.linspace(
            0.0,
            1.0,
            configuration.ntheta,
            endpoint=False,
        ),
    )
    surface.fit_to_curve(magnetic_axis, 0.10, flip_theta=True)
    current_sum = int(nfp) * sum(abs(current.get_value()) for current in base_currents)
    G = float(2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi)))
    return surface, native_field, int(nfp), len(native_field.coils), G


def prepare_fixed_state(configuration: CaseConfiguration) -> FixedState:
    """Prepare positive native Biot-Savart field and all shared basis arrays."""

    started = time.perf_counter()
    surface, native_field, nfp, coil_count, G = _build_surface(configuration)
    gamma = _as_f64(surface.gamma())
    xphi = _as_f64(surface.gammadash1())
    xtheta = _as_f64(surface.gammadash2())
    dx_ds = _as_f64(surface.dgamma_by_dcoeff())
    dxphi_ds = _as_f64(surface.dgammadash1_by_dcoeff())
    dxtheta_ds = _as_f64(surface.dgammadash2_by_dcoeff())
    points = np.ascontiguousarray(gamma.reshape((-1, 3)), dtype=np.float64)

    field = BiotSavart(native_field.coils)
    field.set_points(points)
    field.compute(2)
    B = _as_f64(field.B().reshape((configuration.nphi, configuration.ntheta, 3)))
    dB_dX = _as_f64(
        field.dB_by_dX().reshape((configuration.nphi, configuration.ntheta, 3, 3))
    )
    d2B_dXdX = _as_f64(
        field.d2B_by_dXdX().reshape((configuration.nphi, configuration.ntheta, 3, 3, 3))
    )
    modB = np.linalg.norm(B, axis=-1)
    min_modB = float(np.min(modB))
    if not np.all(np.isfinite(B)) or not np.all(modB > 0.0):
        raise ValueError(
            "native NCSX fixture produced a non-positive or non-finite |B|"
        )

    return FixedState(
        configuration=configuration,
        nfp=nfp,
        G=G,
        iota=-0.406,
        B=B,
        dB_dX=dB_dX,
        d2B_dXdX=d2B_dXdX,
        xphi=xphi,
        xtheta=xtheta,
        dx_ds=dx_ds,
        dxphi_ds=dxphi_ds,
        dxtheta_ds=dxtheta_ds,
        surface_dofs=_as_f64(surface.get_dofs()),
        preparation_s=float(time.perf_counter() - started),
        coil_count=coil_count,
        min_modB=min_modB,
    )


def _native_weight(weight: WeightName) -> bool:
    return weight == "inv_modB"


def _native_exact(state: FixedState, weight: WeightName) -> NativeExactResult:
    """Assemble only the exact raw residual/Jacobian with the C++ oracle."""

    B = state.B
    xphi = state.xphi
    xtheta = state.xtheta
    iota = state.iota
    G = state.G
    tang = xphi + iota * xtheta
    B2 = np.sum(B * B, axis=-1)
    residual_unweighted = G * B - B2[..., None] * tang
    dB_dc = _as_f64(np.einsum("...km,...ka->...ma", state.dB_dX, state.dx_ds))
    dresidual_dc = _as_f64(
        sopp.boozer_dresidual_dc(
            G,
            dB_dc,
            B,
            tang,
            B2,
            state.dxphi_ds,
            iota,
            state.dxtheta_ds,
        )
    )
    dresidual_diota = -B2[..., None] * xtheta
    dresidual_dG = B
    if _native_weight(weight):
        modB = np.sqrt(B2)
        inverse_modB = 1.0 / modB
        dB2_dc = 2.0 * np.einsum("...m,...ma->...a", B, dB_dc)
        dweight_dc = -0.5 * dB2_dc / np.power(B2[..., None], 1.5)
        residual = inverse_modB[..., None] * residual_unweighted
        dresidual_dc = (
            inverse_modB[..., None, None] * dresidual_dc
            + residual_unweighted[..., :, None] * dweight_dc[..., None, :]
        )
        dresidual_diota = inverse_modB[..., None] * dresidual_diota
        dresidual_dG = inverse_modB[..., None] * dresidual_dG
    else:
        residual = residual_unweighted

    residual_flat = np.ascontiguousarray(residual.reshape((-1,)), dtype=np.float64)
    jacobian = np.ascontiguousarray(
        np.concatenate(
            (
                dresidual_dc.reshape((state.nresidual, state.nsurface)),
                dresidual_diota.reshape((state.nresidual, 1)),
                dresidual_dG.reshape((state.nresidual, 1)),
            ),
            axis=1,
        ),
        dtype=np.float64,
    )
    return NativeExactResult(residual_flat, jacobian)


def _native_exact_scalar(state: FixedState, weight: WeightName) -> float:
    """Evaluate the scalar oracle outside the residual/Jacobian timing scope."""

    return float(
        sopp.boozer_residual(
            state.G,
            state.iota,
            state.xphi,
            state.xtheta,
            state.B,
            _native_weight(weight),
        )
    )


def _native_ls(state: FixedState, weight: WeightName) -> NativeLsResult:
    """Evaluate the native raw LS scalar, gradient, and full Hessian oracle."""

    value_raw, gradient_raw, hessian_raw = sopp.boozer_residual_ds2(
        state.G,
        state.iota,
        state.B,
        state.dB_dX,
        state.d2B_dXdX,
        state.xphi,
        state.xtheta,
        state.dx_ds,
        state.dxphi_ds,
        state.dxtheta_ds,
        _native_weight(weight),
    )
    return NativeLsResult(
        value_raw=float(value_raw),
        gradient_raw=_as_f64(gradient_raw),
        hessian_raw=_as_f64(hessian_raw),
    )


def _jax_operands(state: FixedState) -> tuple[jax.Array, ...]:
    """Convert the already-pinned host arrays once, outside timed calls."""

    return tuple(
        jnp.asarray(values, dtype=jnp.float64)
        for values in (
            state.B,
            state.dB_dX,
            state.d2B_dXdX,
            state.xphi,
            state.xtheta,
            state.dx_ds,
            state.dxphi_ds,
            state.dxtheta_ds,
        )
    )


def _make_exact_jax_operator(state: FixedState, weight: WeightName) -> JaxOperator:
    """Bind fixed scalar state while retaining the public analytic API call."""

    weighted = _native_weight(weight)
    G = jnp.asarray(state.G, dtype=jnp.float64)
    iota = jnp.asarray(state.iota, dtype=jnp.float64)

    def operator(
        B: jax.Array,
        dB_dX: jax.Array,
        xphi: jax.Array,
        xtheta: jax.Array,
        dx_ds: jax.Array,
        dxphi_ds: jax.Array,
        dxtheta_ds: jax.Array,
    ) -> tuple[jax.Array, ...]:
        return boozer_residual_vector_and_jacobian(
            G=G,
            iota=iota,
            B=B,
            dB_dX=dB_dX,
            xphi=xphi,
            xtheta=xtheta,
            dx_ds=dx_ds,
            dxphi_ds=dxphi_ds,
            dxtheta_ds=dxtheta_ds,
            optimize_G=True,
            weight_inv_modB=weighted,
        )

    return operator


def _make_ls_jax_operator(
    state: FixedState,
    weight: WeightName,
    quadrature_tile_size: int,
) -> JaxOperator:
    """Bind fixed scalars and call the analytic value/gradient/Hessian API."""

    weighted = _native_weight(weight)
    G = jnp.asarray(state.G, dtype=jnp.float64)
    iota = jnp.asarray(state.iota, dtype=jnp.float64)

    def operator(
        B: jax.Array,
        dB_dX: jax.Array,
        d2B_dXdX: jax.Array,
        xphi: jax.Array,
        xtheta: jax.Array,
        dx_ds: jax.Array,
        dxphi_ds: jax.Array,
        dxtheta_ds: jax.Array,
    ) -> tuple[jax.Array, ...]:
        value, gradient, hessian = boozer_residual_analytic_value_grad_hessian(
            G=G,
            iota=iota,
            B=B,
            dB_dX=dB_dX,
            d2B_dXdX=d2B_dXdX,
            xphi=xphi,
            xtheta=xtheta,
            dx_ds=dx_ds,
            dxphi_ds=dxphi_ds,
            dxtheta_ds=dxtheta_ds,
            optimize_G=True,
            weight_inv_modB=weighted,
            quadrature_tile_size=quadrature_tile_size,
        )
        return value, gradient, hessian

    return operator


def _synchronize(output: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    return tuple(jax.block_until_ready(value) for value in output)


def _stats(samples: list[float]) -> dict[str, object]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(values.size),
        "min_s": float(np.min(values)),
        "median_s": float(np.median(values)),
        "mean_s": float(np.mean(values)),
        "max_s": float(np.max(values)),
    }


def _measure_jax(
    operator: JaxOperator,
    operands: tuple[jax.Array, ...],
    *,
    repeat: int,
    warmup: int,
) -> tuple[dict[str, object], tuple[jax.Array, ...]]:
    """Measure explicit compilation, synchronized first execution, and warm calls."""

    jitted = jax.jit(operator)
    compile_started = time.perf_counter()
    compiled = jitted.lower(*operands).compile()
    compile_s = float(time.perf_counter() - compile_started)

    first_started = time.perf_counter()
    first_output = _synchronize(compiled(*operands))
    first_execute_s = float(time.perf_counter() - first_started)
    for _ in range(warmup):
        _synchronize(compiled(*operands))
    samples: list[float] = []
    final_output = first_output
    for _ in range(repeat):
        started = time.perf_counter()
        final_output = _synchronize(compiled(*operands))
        samples.append(float(time.perf_counter() - started))
    return (
        {
            "cold_compile_s": compile_s,
            "cold_execute_s": first_execute_s,
            "cold_total_s": compile_s + first_execute_s,
            "warm": _stats(samples),
            "synchronized": True,
            "synchronization": "jax.block_until_ready per output leaf",
        },
        final_output,
    )


def _measure_native(
    operation: Callable[[], object],
    *,
    repeat: int,
    warmup: int,
) -> tuple[dict[str, object], object]:
    """Measure native execution with a zero compile boundary."""

    cold_started = time.perf_counter()
    cold_output = operation()
    cold_execute_s = float(time.perf_counter() - cold_started)
    for _ in range(warmup):
        operation()
    samples: list[float] = []
    final_output = cold_output
    for _ in range(repeat):
        started = time.perf_counter()
        final_output = operation()
        samples.append(float(time.perf_counter() - started))
    return (
        {
            "cold_compile_s": 0.0,
            "cold_execute_s": cold_execute_s,
            "cold_total_s": cold_execute_s,
            "warm": _stats(samples),
            "synchronized": True,
            "synchronization": "native synchronous C++ call",
        },
        final_output,
    )


def _host_array(value: jax.Array) -> FloatArray:
    return _as_f64(jax.device_get(value))


def _error_metrics(actual: object, reference: object) -> dict[str, object]:
    actual_array = np.asarray(actual, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    delta = actual_array - reference_array
    delta_l2 = float(np.linalg.norm(delta.reshape(-1)))
    reference_l2 = float(np.linalg.norm(reference_array.reshape(-1)))
    if delta.ndim == 2:
        operator_norm = float(np.linalg.norm(delta, ord=2))
        reference_operator_norm = float(np.linalg.norm(reference_array, ord=2))
    else:
        operator_norm = delta_l2
        reference_operator_norm = reference_l2
    scale = max(reference_l2, np.finfo(np.float64).tiny)
    operator_scale = max(reference_operator_norm, np.finfo(np.float64).tiny)
    return {
        "shape": list(actual_array.shape),
        "dtype": str(actual_array.dtype),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "l2": delta_l2,
        "reference_l2": reference_l2,
        "relative_l2": delta_l2 / scale,
        "operator_norm": operator_norm,
        "reference_operator_norm": reference_operator_norm,
        "relative_operator_norm": operator_norm / operator_scale,
        "all_finite": bool(np.all(np.isfinite(actual_array))),
    }


def _raw_scalar_from_residual(residual: FloatArray) -> float:
    return float(0.5 * np.dot(residual, residual))


def _run_exact_case(
    state: FixedState,
    weight: WeightName,
    *,
    repeat: int,
    warmup: int,
) -> dict[str, object]:
    native_timing, native_output = _measure_native(
        lambda: _native_exact(state, weight), repeat=repeat, warmup=warmup
    )
    native_result = native_output
    if not isinstance(native_result, NativeExactResult):
        raise TypeError("native exact timing returned an unexpected result")
    operands = _jax_operands(state)
    exact_operands = (
        operands[0],
        operands[1],
        operands[3],
        operands[4],
        operands[5],
        operands[6],
        operands[7],
    )
    jax_timing, jax_output = _measure_jax(
        _make_exact_jax_operator(state, weight),
        exact_operands,
        repeat=repeat,
        warmup=warmup,
    )
    jax_residual = _host_array(jax_output[0])
    jax_jacobian = _host_array(jax_output[1])
    native_scalar = _native_exact_scalar(state, weight)
    jax_scalar = _raw_scalar_from_residual(jax_residual)
    return {
        "operator": "exact residual and Jacobian",
        "weight": weight,
        "native_oracle": "simsoptpp.boozer_dresidual_dc (timed); simsoptpp.boozer_residual (excluded scalar check)",
        "jax_api": "simsopt_jax.geo.boozer_residual.boozer_residual_vector_and_jacobian",
        "timing_native": native_timing,
        "timing_jax": jax_timing,
        "operator_norm_errors_vs_native": {
            "residual": _error_metrics(jax_residual, native_result.residual),
            "jacobian": _error_metrics(jax_jacobian, native_result.jacobian),
            "raw_scalar": _error_metrics(jax_scalar, native_scalar),
        },
        "outputs": {
            "native_residual_sha256": _array_sha256(native_result.residual),
            "jax_residual_sha256": _array_sha256(jax_residual),
            "native_jacobian_sha256": _array_sha256(native_result.jacobian),
            "jax_jacobian_sha256": _array_sha256(jax_jacobian),
            "native_scalar_raw": native_scalar,
            "jax_scalar_raw_from_residual": jax_scalar,
            "normalization": "raw 0.5 sum(residual**2); native scalar has no 3N normalization",
            "scalar_oracle_timing": "excluded from timing_native",
        },
    }


def _run_ls_case(
    state: FixedState,
    weight: WeightName,
    *,
    quadrature_tile_size: int,
    repeat: int,
    warmup: int,
) -> dict[str, object]:
    native_timing, native_output = _measure_native(
        lambda: _native_ls(state, weight), repeat=repeat, warmup=warmup
    )
    native_result = native_output
    if not isinstance(native_result, NativeLsResult):
        raise TypeError("native LS timing returned an unexpected result")
    operands = _jax_operands(state)
    jax_timing, jax_output = _measure_jax(
        _make_ls_jax_operator(state, weight, quadrature_tile_size),
        operands,
        repeat=repeat,
        warmup=warmup,
    )
    jax_value = float(np.asarray(jax.device_get(jax_output[0]), dtype=np.float64))
    jax_gradient = _host_array(jax_output[1])
    jax_hessian = _host_array(jax_output[2])
    normalization = float(state.nresidual)
    native_value = native_result.value_raw / normalization
    native_gradient = native_result.gradient_raw / normalization
    native_hessian = native_result.hessian_raw / normalization
    return {
        "operator": "least-squares value, gradient, and full Hessian",
        "weight": weight,
        "native_oracle": "simsoptpp.boozer_residual_ds2",
        "jax_api": "simsopt_jax.geo.boozer_analytic_hessian.boozer_residual_analytic_value_grad_hessian",
        "quadrature_tile_size": quadrature_tile_size,
        "timing_native": native_timing,
        "timing_jax": jax_timing,
        "operator_norm_errors_vs_native": {
            "value_normalized": _error_metrics(jax_value, native_value),
            "gradient_normalized": _error_metrics(jax_gradient, native_gradient),
            "hessian_normalized": _error_metrics(jax_hessian, native_hessian),
        },
        "outputs": {
            "native_value_raw": native_result.value_raw,
            "native_value_normalized": native_value,
            "jax_value_normalized": jax_value,
            "native_gradient_sha256": _array_sha256(native_gradient),
            "jax_gradient_sha256": _array_sha256(jax_gradient),
            "native_hessian_sha256": _array_sha256(native_hessian),
            "jax_hessian_sha256": _array_sha256(jax_hessian),
            "normalization": f"divide native raw scalar/derivatives by 3N={state.nresidual}",
        },
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return platform.processor()


def _git_output(arguments: list[str]) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _provenance(
    args: argparse.Namespace, source_paths: tuple[Path, ...]
) -> dict[str, object]:
    benchmark_path = Path(__file__).resolve()
    git_sha = _git_output(["rev-parse", "HEAD"]).decode().strip()
    git_status = _git_output(["status", "--porcelain", "--", str(benchmark_path)])
    git_diff = _git_output(["diff", "--binary", "HEAD", "--", str(benchmark_path)])
    diff_payload = git_diff if git_diff else benchmark_path.read_bytes()
    source_hashes = {str(path): _sha256_file(path) for path in source_paths}
    source_payload = "\n".join(
        f"{path} {source_hashes[path]}" for path in sorted(source_hashes)
    ).encode()
    original_argv = (
        list(sys.orig_argv)
        if hasattr(sys, "orig_argv")
        else [
            sys.executable,
            *sys.argv,
        ]
    )
    extension_path = Path(sopp.__file__).resolve()
    devices = [
        {
            "platform": device.platform,
            "device_kind": str(device.device_kind),
            "id": int(device.id),
            "repr": str(device),
            "client_platform": str(device.client.platform),
            "client_platform_version": str(device.client.platform_version),
        }
        for device in jax.devices()
    ]
    return {
        "git_sha": git_sha,
        "benchmark_sha256": _sha256_file(benchmark_path),
        "benchmark_diff_hash": hashlib.sha256(diff_payload).hexdigest(),
        "diffhash": hashlib.sha256(diff_payload + b"\n" + source_payload).hexdigest(),
        "git_status_for_benchmark": git_status.decode().strip(),
        "sourcepaths": source_hashes,
        "native_extension_path": str(extension_path),
        "nativeextensionhash": _sha256_file(extension_path),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "jax_backend": jax.default_backend(),
        "devices": devices,
        "cpu_model": _cpu_model(),
        "requested_platform": args.platform,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONPATH",
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "exactcommand": shlex.join(original_argv),
        "claim_ceiling": "component operator gate only; E2E solver and optimization remain unproven",
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """Run selected fixed-state lanes and return a JSON-serializable report."""

    _validate_requested_platform()
    configurations = {
        name: _case_configuration(args, name) for name in _selected_cases(args.mode)
    }
    states = {
        name: prepare_fixed_state(configuration)
        for name, configuration in configurations.items()
    }
    source_paths = (
        Path(__file__).resolve(),
        (SRC_ROOT / "simsopt_jax" / "geo" / "boozer_residual.py").resolve(),
        (SRC_ROOT / "simsopt_jax" / "geo" / "boozer_analytic_hessian.py").resolve(),
        (SRC_ROOT / "simsopt" / "field" / "biotsavart.py").resolve(),
        (SRC_ROOT / "simsopt" / "geo" / "surfacexyztensorfourier.py").resolve(),
        (SRC_ROOT / "simsoptpp" / "boozerresidual_py.cpp").resolve(),
        (SRC_ROOT / "simsopt" / "configs" / "zoo.py").resolve(),
    )
    report: dict[str, object] = {
        "measurement": "native C++ vs JAX analytic Boozer fixed-state operator probe",
        "provenance": _provenance(args, source_paths),
        "configuration": {
            name: {
                "mpol": state.configuration.mpol,
                "ntor": state.configuration.ntor,
                "nphi": state.configuration.nphi,
                "ntheta": state.configuration.ntheta,
                "surface_dof_count": state.nsurface,
                "residual_count": state.nresidual,
                "coil_count": state.coil_count,
                "nfp": state.nfp,
                "G": state.G,
                "iota": state.iota,
                "min_modB": state.min_modB,
                "field_preparation_s": state.preparation_s,
                "dtype": "float64",
                "array_inventory": _array_inventory(state),
            }
            for name, state in states.items()
        },
        "weights": list(_selected_weights(args.weights)),
        "timing_scope": {
            "operator_assembly_only": True,
            "field_preparation_excluded": True,
            "native_timing": "simsoptpp dresidual_dc and host-side residual/J assembly only; scalar oracle excluded",
            "jax_timing": "compiled JAX operator execution only",
        },
        "cases": {},
        "e2e_proven": False,
        "claim_ceiling": "operator assembly parity/performance only; E2E solver and optimization are unproven",
    }
    case_reports: dict[str, object] = {}
    for name, state in states.items():
        weight_reports: dict[str, object] = {}
        for weight in _selected_weights(args.weights):
            if name == "exact":
                weight_reports[weight] = _run_exact_case(
                    state,
                    weight,
                    repeat=args.repeat,
                    warmup=args.warmup,
                )
            else:
                weight_reports[weight] = _run_ls_case(
                    state,
                    weight,
                    quadrature_tile_size=args.quadrature_tile_size,
                    repeat=args.repeat,
                    warmup=args.warmup,
                )
        case_reports[name] = weight_reports
    report["cases"] = case_reports
    return report


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_benchmark(args)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
