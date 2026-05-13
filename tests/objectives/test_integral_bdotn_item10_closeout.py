"""Item 10 closeout parity test for the chained Biot-Savart + integral_BdotN.

Audit summary (`.artifacts/jax_port_goal/plans/10.md`):

- `BiotSavartJAX.B()` and `integral_BdotN(...)` already match the C++
  oracle in isolation. C++ Biot-Savart parity is locked in at
  `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppParity`
  (ncoils=33 NCSX fixture). C++ `integral_BdotN` parity is locked in at
  `tests/objectives/test_integral_bdotn_jax.py::TestIntegralBdotNCppParity`
  (nphi=ntheta=15 isolated B array).
- The audit identified ONE missing fixture: no parity test chains
  `BiotSavartJAX.B()` -> `integral_BdotN(...)` against
  `BiotSavart.B()` -> `sopp.integral_BdotN(...)` at the production-scale
  floor (`ncoils >= 4`, `nphi >= 16`, `ntheta >= 8`) for all three
  definition variants under a strict transfer-guard discipline.

This file closes that gap. Tolerances are imported from
`benchmarks.validation_ladder_contract.parity_ladder_tolerances("direct_kernel")`
(`rtol=1e-10`, `atol=1e-12`). No `atol=` / `rtol=` numeric literals are
inlined in this file.

The test runs under `jax.transfer_guard("disallow")` AND is also runnable
under the process-wide `SIMSOPT_JAX_TRANSFER_GUARD=disallow` discipline
mandated by section 4c of `jax_port_goal_prompt_2026-05-12.md`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.integral_bdotn_jax import integral_BdotN

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]

# Production-scale floor (Biot-Savart / flux family): nphi >= 16, ntheta >= 8,
# ncoils >= 4. We use the minimum allowed floor so the test runs inside the
# validation budget while exercising the chained kernel path.
_NCOILS_BASE = 4
_NPHI = 16
_NTHETA = 8
_FOURIER_ORDER = 3
_R0 = 1.0
_R1_COIL = 0.5
_R0_SURF = 1.0
_R1_SURF = 0.2

_SQUARED_FLUX_DEFINITIONS = (
    "quadratic flux",
    "normalized",
    "local",
)
_STELLSYM_MODES = (False, True)


def _build_coil_surface_case(stellsym: bool):
    """Build a production-scale coil + surface fixture.

    ncoils_base = 4 base curves before symmetry expansion. With
    ``stellsym=True``, ``coils_via_symmetries`` doubles the coil list to
    8 entries (the second 4 are stellarator-reflected). With
    ``stellsym=False``, the result has 4 entries (nfp=1, no z-reflection).

    The surface is a slightly-perturbed circular cross-section
    ``SurfaceRZFourier`` (mpol=1, ntor=1) with ``nphi=16``, ``ntheta=8``
    parametric quadpoints. The surface is well-separated from the coil
    geometry so the integrand is well-conditioned.
    """
    nfp = 1
    base_curves = create_equally_spaced_curves(
        _NCOILS_BASE,
        nfp,
        stellsym=stellsym,
        R0=_R0,
        R1=_R1_COIL,
        order=_FOURIER_ORDER,
    )
    base_currents = [Current(1.0e5) for _ in range(_NCOILS_BASE)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0, _NPHI, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, _NTHETA, endpoint=False),
    )
    surface.set_rc(0, 0, _R0_SURF)
    surface.set_rc(1, 0, _R1_SURF)
    surface.set_zs(1, 0, _R1_SURF)
    surface.fix_all()
    return coils, surface


def _build_nontrivial_target(
    B_cpu_grid: np.ndarray, normal_host: np.ndarray
) -> np.ndarray:
    """Return a nontrivial ``target`` array of shape ``(nphi, ntheta)``.

    A nonzero target exercises the residual subtraction inside
    ``integral_BdotN`` for all three definitions. We construct it from
    the CPU oracle's own ``B . n_hat`` shifted by a 30% scalar so the
    residual is finite, sign-stable, and identical between the CPU and
    JAX chains (both sides see the SAME target array — the parity claim
    is about evaluation, not about the target itself).
    """
    norm_n = np.sqrt(np.sum(normal_host * normal_host, axis=-1))
    safe_norm = np.where(norm_n > 0.0, norm_n, 1.0)
    unit_n = normal_host / safe_norm[..., None]
    BdotN_cpu = np.sum(B_cpu_grid * unit_n, axis=-1)
    return 0.3 * BdotN_cpu


@pytest.mark.parametrize("stellsym", _STELLSYM_MODES, ids=("no_stellsym", "stellsym"))
@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
def test_chained_biotsavartjax_integral_bdotn_matches_cpp_at_production_scale(
    stellsym: bool,
    definition: str,
):
    """Chained `BiotSavartJAX.B()` -> `integral_BdotN` vs C++ oracle.

    For each (stellsym, definition) combination this test:

    1. Builds a 4-base-coil setup (8 after stellsym expansion) and a
       ``SurfaceRZFourier`` with ``nphi=16``, ``ntheta=8`` (production
       floor).
    2. Evaluates ``B_cpu = BiotSavart(coils).set_points(...).B()`` and
       ``B_jax = BiotSavartJAX(coils).set_points(...).B()`` and checks
       direct-kernel parity between them.
    3. Builds a nontrivial ``target`` array from ``B_cpu . n_hat``
       scaled by 0.3.
    4. Computes the C++ reducer chain
       ``sopp.integral_BdotN(B_cpu, target, normal, definition)`` and
       the chained JAX value
       ``integral_BdotN(jnp.asarray(B_jax_grid), target_jax, normal_jax,
       definition)``.
    5. Asserts the JAX chain matches the C++ chain at the
       ``direct-kernel`` lane tolerance.

    All JAX operations inside the parity boundary run under
    ``jax.transfer_guard("disallow")``.
    """
    import simsoptpp as sopp

    coils, surface = _build_coil_surface_case(stellsym)
    points = np.ascontiguousarray(
        surface.gamma().reshape((-1, 3)),
        dtype=np.float64,
    )
    normal_host = np.ascontiguousarray(surface.normal(), dtype=np.float64)

    # CPU C++ oracle chain
    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(points)
    B_cpu_flat = np.asarray(bs_cpu.B(), dtype=np.float64)
    B_cpu_grid = np.ascontiguousarray(B_cpu_flat.reshape((_NPHI, _NTHETA, 3)))

    # Build nontrivial target on host; both chains will consume the same array.
    target_host = np.ascontiguousarray(
        _build_nontrivial_target(B_cpu_grid, normal_host),
        dtype=np.float64,
    )

    # C++ reducer applied to C++ B
    J_cpp = float(sopp.integral_BdotN(B_cpu_grid, target_host, normal_host, definition))

    # JAX chain: stage device arrays outside the jit boundary, then enter
    # transfer_guard("disallow") around the compiled lane.
    points_device = jax.device_put(jnp.asarray(points, dtype=jnp.float64))
    target_device = jax.device_put(jnp.asarray(target_host, dtype=jnp.float64))
    normal_device = jax.device_put(jnp.asarray(normal_host, dtype=jnp.float64))

    bs_jax = BiotSavartJAX(coils)
    bs_jax.set_points(points_device)

    with jax.transfer_guard("disallow"):
        B_jax_flat = bs_jax.B()
        B_jax_grid = jnp.reshape(B_jax_flat, (_NPHI, _NTHETA, 3))
        J_jax_device = integral_BdotN(
            B_jax_grid,
            target_device,
            normal_device,
            definition,
        )
        J_jax_device.block_until_ready()

    B_jax_grid_host = np.asarray(B_jax_grid, dtype=np.float64)
    J_jax = float(np.asarray(J_jax_device, dtype=np.float64))

    # (a) Forward B parity at the direct-kernel lane tolerance.
    np.testing.assert_allclose(
        B_jax_grid_host,
        B_cpu_grid,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=(
            "BiotSavartJAX.B() does not match BiotSavart.B() at production-"
            f"scale floor (stellsym={stellsym}, definition={definition!r})."
        ),
    )

    # (b) Anchor: C++ reducer on the JAX B equals C++ reducer on the C++ B
    # at the same tolerance — guards against a silent reshape / contiguity
    # corruption in the JAX -> NumPy boundary.
    J_cpp_on_jax_B = float(
        sopp.integral_BdotN(
            np.ascontiguousarray(B_jax_grid_host),
            target_host,
            normal_host,
            definition,
        )
    )
    np.testing.assert_allclose(
        J_cpp_on_jax_B,
        J_cpp,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=(
            "C++ integral_BdotN(B_jax) != C++ integral_BdotN(B_cpu) at "
            "direct-kernel parity floor — JAX B array roundtrip is corrupt "
            f"(stellsym={stellsym}, definition={definition!r})."
        ),
    )

    # (c) Full chained parity: JAX reducer on JAX B equals C++ reducer on C++ B.
    np.testing.assert_allclose(
        J_jax,
        J_cpp,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=(
            "Chained BiotSavartJAX -> integral_BdotN does not match "
            "BiotSavart -> sopp.integral_BdotN at production-scale floor "
            f"(stellsym={stellsym}, definition={definition!r})."
        ),
    )
