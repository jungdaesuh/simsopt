"""Direct-kernel closeout parity test for ``BiotSavartJAX.A()``.

Audit summary (`jax_native_remaining_impl_plan_2026-04-24.md`, line ~615):

- ``BiotSavartJAX.A()`` already matches the CPU ``BiotSavart.A()`` oracle
  inside the legacy ``tests/integration/test_stage2_jax.py::test_A_parity``
  fixture, but with inline numeric tolerances (``rtol=1e-10``,
  ``atol=1e-15``) that do not source the parity-ladder SSOT.
- The audit identified ONE missing fixture: no ``direct_kernel`` lane row
  that anchors the vector-potential public API
  (``BiotSavartJAX.A()`` vs ``BiotSavart.A()``) to
  ``benchmarks.validation_ladder_contract.parity_ladder_tolerances(
  "direct_kernel")`` at the production-scale floor (``ncoils_base >= 4``,
  ``nfp = 2``, ``npoints >= 50``) for both stellsym modes under a strict
  transfer-guard discipline.

This file closes that gap. Tolerances are imported from
``benchmarks.validation_ladder_contract.parity_ladder_tolerances(
"direct_kernel")`` (``rtol=1e-10``, ``atol=1e-12``). No ``atol=`` /
``rtol=`` numeric literals are inlined.

The transfer-guard variant runs inside ``jax.transfer_guard("disallow")``
AND is also runnable under the process-wide
``SIMSOPT_JAX_TRANSFER_GUARD=disallow`` discipline mandated by section 4c
of ``jax_port_goal_prompt_2026-05-12.md``.
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

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]

# Production-scale floor (Biot-Savart vector-potential family):
# ``ncoils_base >= 4``, ``nfp = 2``, ``npoints >= 50``. With
# ``nphi = 16``, ``ntheta = 8`` the surface yields 128 evaluation points,
# comfortably above the 50-point floor while staying inside the validation
# budget. ``nfp = 2`` plus stellsym expansion produces 16 coils; without
# stellsym it produces 8 coils — both within the requested 8-16 range.
_NCOILS_BASE = 4
_NFP = 2
_NPHI = 16
_NTHETA = 8
_FOURIER_ORDER = 3
_R0 = 1.0
_R1_COIL = 0.5
_R0_SURF = 1.0
_R1_SURF = 0.2

_STELLSYM_MODES = (False, True)


def _build_coil_surface_case(stellsym: bool):
    """Build a production-scale coil + surface fixture for A() parity.

    ``ncoils_base = 4`` base curves before symmetry expansion. With
    ``nfp = 2``:

    - ``stellsym = False``: ``coils_via_symmetries`` yields 8 coils
      (4 base × 2 field periods).
    - ``stellsym = True``: ``coils_via_symmetries`` yields 16 coils
      (4 base × 2 field periods × 2 stellarator reflections).

    The surface is a slightly-perturbed circular cross-section
    ``SurfaceRZFourier`` (``mpol = 1``, ``ntor = 1``) with ``nphi = 16``,
    ``ntheta = 8`` parametric quadpoints (128 points). The surface is
    well-separated from the coil geometry so the integrand is
    well-conditioned and the vector potential is finite.
    """
    base_curves = create_equally_spaced_curves(
        _NCOILS_BASE,
        _NFP,
        stellsym=stellsym,
        R0=_R0,
        R1=_R1_COIL,
        order=_FOURIER_ORDER,
    )
    base_currents = [Current(1.0e5) for _ in range(_NCOILS_BASE)]
    coils = coils_via_symmetries(base_curves, base_currents, _NFP, stellsym)

    surface = SurfaceRZFourier(
        nfp=_NFP,
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


@pytest.mark.parametrize("stellsym", _STELLSYM_MODES, ids=("no_stellsym", "stellsym"))
def test_biotsavartjax_A_matches_cpu_at_production_scale(stellsym: bool):
    """``BiotSavartJAX.A()`` matches ``BiotSavart.A()`` bit-for-bit.

    For each stellsym mode this test:

    1. Builds a 4-base-coil setup (``nfp = 2``; 8 coils for
       ``stellsym = False``, 16 coils for ``stellsym = True``) and a
       ``SurfaceRZFourier`` with ``nphi = 16``, ``ntheta = 8`` (128
       evaluation points — above the 50-point production floor).
    2. Evaluates
       ``A_cpu = BiotSavart(coils).set_points(...).A()`` and
       ``A_jax = BiotSavartJAX(coils).set_points(...).A()`` on identical
       host-side points.
    3. Asserts the JAX result matches the CPU oracle at the
       ``direct_kernel`` lane tolerance (``rtol=1e-10``, ``atol=1e-12``).
    """
    coils, surface = _build_coil_surface_case(stellsym)
    points = np.ascontiguousarray(
        surface.gamma().reshape((-1, 3)),
        dtype=np.float64,
    )

    # CPU oracle.
    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(points)
    A_cpu = np.asarray(bs_cpu.A(), dtype=np.float64)

    # JAX backend on the same host-side points.
    bs_jax = BiotSavartJAX(coils)
    bs_jax.set_points(points)
    A_jax = np.asarray(bs_jax.A(), dtype=np.float64)

    assert A_jax.shape == A_cpu.shape, (
        f"Shape mismatch: JAX {A_jax.shape} vs CPU {A_cpu.shape} (stellsym={stellsym})."
    )
    assert A_cpu.shape[0] >= 50, (
        f"Production-scale floor violated: npoints={A_cpu.shape[0]} < 50 "
        f"(stellsym={stellsym})."
    )

    np.testing.assert_allclose(
        A_jax,
        A_cpu,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=(
            "BiotSavartJAX.A() does not match BiotSavart.A() at the "
            f"direct_kernel parity floor (stellsym={stellsym})."
        ),
    )


def test_biotsavartjax_A_matches_cpu_under_transfer_guard():
    """Transfer-guard variant: same parity claim, no implicit host transfer.

    Stages the evaluation points onto the device outside the guarded
    block, then enters ``jax.transfer_guard("disallow")`` around the JAX
    ``A()`` evaluation. The guarded block must not implicitly transfer
    arrays between host and device. The CPU oracle is computed on the
    host (outside the guard) and the comparison is made on host-side
    NumPy arrays after the JAX result has been materialized via
    ``block_until_ready()``.

    Tolerances are imported from the ``direct_kernel`` lane.
    """
    stellsym = True  # Exercise the larger 16-coil expansion under the guard.
    coils, surface = _build_coil_surface_case(stellsym)
    points_host = np.ascontiguousarray(
        surface.gamma().reshape((-1, 3)),
        dtype=np.float64,
    )

    # Host-side CPU oracle (outside the guard).
    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(points_host)
    A_cpu = np.asarray(bs_cpu.A(), dtype=np.float64)

    # Stage the points onto the device before entering the guard.
    points_device = jax.device_put(jnp.asarray(points_host, dtype=jnp.float64))

    bs_jax = BiotSavartJAX(coils)
    bs_jax.set_points(points_device)

    with jax.transfer_guard("disallow"):
        A_jax_device = bs_jax.A()
        A_jax_device.block_until_ready()

    A_jax = np.asarray(A_jax_device, dtype=np.float64)

    assert A_jax.shape == A_cpu.shape, (
        f"Shape mismatch under transfer_guard: JAX {A_jax.shape} vs CPU {A_cpu.shape}."
    )

    np.testing.assert_allclose(
        A_jax,
        A_cpu,
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=(
            "BiotSavartJAX.A() under jax.transfer_guard('disallow') does "
            "not match BiotSavart.A() at the direct_kernel parity floor."
        ),
    )
