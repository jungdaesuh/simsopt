"""Tests for the CPU-ordered Biot-Savart twin (Phase 3 of the bit-identity plan).

Mirrors ``src/simsoptpp/biot_savart_impl.h`` operator-for-operator:

* ``diff = point - gamma`` (C++ sign);
* ``norm_diff_3_inv = r_inv * r_inv * r_inv`` (NOT ``r_inv * inv(r²)``);
* ``cross = dgamma × diff`` (C++ operand order);
* sequential ``lax.fori_loop`` over quadrature points (no XLA pairwise
  tree);
* ``B = sum_c (currents[c] · B_per_coil[c])`` accumulated sequentially.

The Phase 3 acceptance gate matches Phase 2's: byte identity OR documented
arithmetic-order reason. Today FMA-fusion remains the residual; these tests
pin the cpu_ordered output within the documented ULP ceiling and assert
no-regression vs the production matmul kernel. The routing test additionally
asserts the cpu_ordered branch meets the same absolute ULP ceiling, so the
routing path cannot silently drift while only beating production.
"""

from __future__ import annotations

import numpy as np
import pytest


pytestmark = [pytest.mark.parity_census, pytest.mark.boozer]


# ULP ceilings for cpu_ordered vs C++ oracle on the NCSX Boozer fixture.
# Documented arithmetic-order residual is FMA-fusion only; these bounds are
# the SSOT across every test in this module.
_BIOT_SAVART_B_ULP_CEILING = 5e-14
_BIOT_SAVART_DB_ULP_CEILING = 1e-13


@pytest.fixture(scope="module")
def fixture_for_bs_parity():
    """Build coils and observation points using the NCSX helper fixture."""
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    helpers_dir = repo_root / "tests" / "geo"
    if str(helpers_dir) not in sys.path:
        sys.path.insert(0, str(helpers_dir))

    from surface_test_helpers import get_boozer_surface

    bs_cpu, booz = get_boozer_surface(
        label="Volume",
        boozer_type="ls",
        optimize_G=True,
        converge=False,
        weight_inv_modB=False,
    )
    points = np.asarray(booz.surface.gamma().reshape(-1, 3), dtype=np.float64)
    return {"bs_cpu": bs_cpu, "points": points}


def _grouped_inputs_for_jax(bs_cpu):
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
    from simsopt.geo.boozersurface_jax import _hostify_tree
    from simsopt.jax_core.field import grouped_field_inputs_from_spec

    bs_jax = BiotSavartJAX(bs_cpu.coils)
    spec = _hostify_tree(bs_jax.coil_set_spec())
    return spec, grouped_field_inputs_from_spec(spec)


def _accumulate_cpu_ordered(points, groups):
    import jax

    from simsopt.jax_core.biotsavart_cpu_ordered import (
        biot_savart_B_and_dB_cpu_ordered,
    )

    B_total = None
    dB_total = None
    for gammas, gammadashs, currents in groups:
        B_g, dB_g = biot_savart_B_and_dB_cpu_ordered(
            points, gammas, gammadashs, currents
        )
        B_g_host = np.asarray(jax.device_get(B_g), dtype=np.float64)
        dB_g_host = np.asarray(jax.device_get(dB_g), dtype=np.float64)
        B_total = B_g_host if B_total is None else B_total + B_g_host
        dB_total = dB_g_host if dB_total is None else dB_total + dB_g_host
    return B_total, dB_total


def test_bs_cpu_ordered_B_within_ulp_of_cpp(fixture_for_bs_parity):
    bs_cpu = fixture_for_bs_parity["bs_cpu"]
    points = fixture_for_bs_parity["points"]
    bs_cpu.set_points(points)
    bs_cpu.compute(0)
    B_cpp = np.asarray(bs_cpu.B(), dtype=np.float64)

    _, groups = _grouped_inputs_for_jax(bs_cpu)
    import jax
    from simsopt.jax_core.biotsavart_cpu_ordered import (
        biot_savart_B_cpu_ordered,
    )

    B_cpu_ordered = None
    for gammas, gammadashs, currents in groups:
        B_g = biot_savart_B_cpu_ordered(points, gammas, gammadashs, currents)
        B_g_host = np.asarray(jax.device_get(B_g), dtype=np.float64)
        B_cpu_ordered = B_g_host if B_cpu_ordered is None else B_cpu_ordered + B_g_host
    cpu_ordered_drift = np.max(np.abs(B_cpu_ordered - B_cpp))
    # FMA-fusion bracket — production drift is ~1e-15; cpu_ordered must not
    # exceed this and must remain within the documented ULP ceiling.
    assert cpu_ordered_drift < _BIOT_SAVART_B_ULP_CEILING
    assert np.all(np.isfinite(B_cpu_ordered))


def test_bs_cpu_ordered_dB_within_ulp_of_cpp(fixture_for_bs_parity):
    bs_cpu = fixture_for_bs_parity["bs_cpu"]
    points = fixture_for_bs_parity["points"]
    bs_cpu.set_points(points)
    bs_cpu.compute(1)
    dB_cpp = np.asarray(bs_cpu.dB_by_dX(), dtype=np.float64)

    _, groups = _grouped_inputs_for_jax(bs_cpu)
    _, dB_cpu_ordered = _accumulate_cpu_ordered(points, groups)
    cpu_ordered_drift = np.max(np.abs(dB_cpu_ordered - dB_cpp))
    assert cpu_ordered_drift < _BIOT_SAVART_DB_ULP_CEILING
    assert np.all(np.isfinite(dB_cpu_ordered))


def test_bs_cpu_ordered_does_not_regress_vs_production(fixture_for_bs_parity):
    bs_cpu = fixture_for_bs_parity["bs_cpu"]
    points = fixture_for_bs_parity["points"]
    bs_cpu.set_points(points)
    bs_cpu.compute(1)
    B_cpp = np.asarray(bs_cpu.B(), dtype=np.float64)
    dB_cpp = np.asarray(bs_cpu.dB_by_dX(), dtype=np.float64)

    spec, groups = _grouped_inputs_for_jax(bs_cpu)
    import jax
    from simsopt.jax_core.field import (
        grouped_biot_savart_B_and_dB_from_spec,
    )

    B_prod, dB_prod = grouped_biot_savart_B_and_dB_from_spec(points, spec)
    B_prod = np.asarray(jax.device_get(B_prod), dtype=np.float64)
    dB_prod = np.asarray(jax.device_get(dB_prod), dtype=np.float64)

    B_cpu, dB_cpu = _accumulate_cpu_ordered(points, groups)

    prod_B_drift = np.max(np.abs(B_prod - B_cpp))
    cpu_B_drift = np.max(np.abs(B_cpu - B_cpp))
    prod_dB_drift = np.max(np.abs(dB_prod - dB_cpp))
    cpu_dB_drift = np.max(np.abs(dB_cpu - dB_cpp))

    # cpu_ordered should match or beat production; allow a tiny ULP slack.
    assert cpu_B_drift <= prod_B_drift * 1.001 + 1e-18
    assert cpu_dB_drift <= prod_dB_drift * 1.001 + 1e-18


def test_field_terms_parity_policy_routes_through_cpu_ordered_and_meets_ulp_ceiling(
    fixture_for_bs_parity,
):
    """Confirm ``_field_terms_for_local_label`` honours ``parity_policy``.

    Asserts BOTH:
    * no-regression: ``cpu_ordered`` drift does not exceed ``production`` drift
      (catches cpu_ordered regressing relative to production);
    * absolute ULP ceiling: ``cpu_ordered`` drift stays within the documented
      ULP bound vs the C++ oracle (catches the case where both production
      and cpu_ordered drift together away from C++).
    """
    import jax

    from simsopt.geo.boozersurface_jax import (
        _field_terms_for_local_label,
        _hostify_tree,
    )
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

    bs_cpu = fixture_for_bs_parity["bs_cpu"]
    points = fixture_for_bs_parity["points"]
    bs_jax = BiotSavartJAX(bs_cpu.coils)
    spec = _hostify_tree(bs_jax.coil_set_spec())

    bs_cpu.set_points(points)
    bs_cpu.compute(1)
    B_cpp = np.asarray(bs_cpu.B(), dtype=np.float64).reshape(-1, 3)
    dB_cpp = np.asarray(bs_cpu.dB_by_dX(), dtype=np.float64).reshape(-1, 3, 3)

    # ``_field_shape_from_geometry`` returns ``(nphi, ntheta)``; reshape
    # helpers append the trailing 3-vector dim.
    surf = fixture_for_bs_parity["bs_cpu"]
    nphi = surf.coils[0].curve.gamma().shape[0]  # placeholder; not used
    del nphi
    field_shape = (points.shape[0],)
    terms_prod = _field_terms_for_local_label(
        points,
        field_shape,
        label_points=None,
        label_field_shape=None,
        coil_set_spec=spec,
        parity_policy="production",
    )
    terms_cpu = _field_terms_for_local_label(
        points,
        field_shape,
        label_points=None,
        label_field_shape=None,
        coil_set_spec=spec,
        parity_policy="cpu_ordered",
    )
    B_prod = np.asarray(jax.device_get(terms_prod.B), dtype=np.float64)
    B_cpu = np.asarray(jax.device_get(terms_cpu.B), dtype=np.float64)
    dB_prod = np.asarray(jax.device_get(terms_prod.dB_dX), dtype=np.float64)
    dB_cpu = np.asarray(jax.device_get(terms_cpu.dB_dX), dtype=np.float64)

    # No-regression: cpu_ordered must match or beat production drift vs C++.
    assert np.max(np.abs(B_cpu - B_cpp)) <= np.max(np.abs(B_prod - B_cpp)) + 1e-18
    assert np.max(np.abs(dB_cpu - dB_cpp)) <= np.max(np.abs(dB_prod - dB_cpp)) + 1e-18

    # Absolute ULP ceiling: cpu_ordered must meet the same hard parity bound
    # the sibling cpu_ordered tests enforce. This catches the failure mode
    # where production AND cpu_ordered both drift together away from C++.
    assert np.max(np.abs(B_cpu - B_cpp)) < _BIOT_SAVART_B_ULP_CEILING
    assert np.max(np.abs(dB_cpu - dB_cpp)) < _BIOT_SAVART_DB_ULP_CEILING
