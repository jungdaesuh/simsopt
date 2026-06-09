import os

import numpy as np

import jax

from simsopt_jax.core.vmec_frozen import (
    vmec_spline_deriv_eval,
    vmec_spline_eval,
)
from simsopt_jax_adapters.mhd.vmec_diagnostics import vmec_freeze_splines
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_splines

from . import TEST_DIR


def _stack_cpu_spline_values(splines, s):
    return np.column_stack([spline(s) for spline in splines])


def test_vmec_freeze_splines_metadata_and_pytree_contract():
    """Oracle: upstream ``vmec_splines`` Struct metadata and pytree flattening."""
    vmec = Vmec(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
    splines = vmec_splines(vmec)
    frozen = vmec_freeze_splines(splines)

    assert frozen.stellsym == splines.stellsym
    assert frozen.mnmax == splines.mnmax
    assert frozen.mnmax_nyq == splines.mnmax_nyq
    assert frozen.nfp == splines.nfp
    np.testing.assert_allclose(np.asarray(frozen.xm), splines.xm)
    np.testing.assert_allclose(np.asarray(frozen.xn), splines.xn)
    np.testing.assert_allclose(np.asarray(frozen.xm_nyq), splines.xm_nyq)
    np.testing.assert_allclose(np.asarray(frozen.xn_nyq), splines.xn_nyq)
    np.testing.assert_allclose(np.asarray(frozen.phiedge), splines.phiedge)
    np.testing.assert_allclose(np.asarray(frozen.Aminor_p), splines.Aminor_p)

    leaves = jax.tree_util.tree_leaves(frozen)
    assert leaves
    assert all(hasattr(leaf, "dtype") for leaf in leaves)
    assert all(leaf.dtype != np.dtype("O") for leaf in leaves)


def test_vmec_frozen_spline_eval_matches_cpu_splines():
    """Oracle: CPU ``InterpolatedUnivariateSpline`` objects frozen by ``vmec_splines``."""
    vmec = Vmec(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
    splines = vmec_splines(vmec)
    frozen = vmec_freeze_splines(splines)
    s = np.array([0.25, 0.5, 0.75])

    scalar_fields = ("pressure", "d_pressure_d_s", "iota", "d_iota_d_s")
    for field_name in scalar_fields:
        actual = vmec_spline_eval(getattr(frozen, field_name), s)
        expected = getattr(splines, field_name)(s)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12)

    mode_fields = (
        "rmnc",
        "zmns",
        "lmns",
        "d_rmnc_d_s",
        "d_zmns_d_s",
        "d_lmns_d_s",
        "gmnc",
        "bmnc",
        "d_bmnc_d_s",
        "bsupumnc",
        "bsupvmnc",
        "d_bsupumnc_d_s",
        "d_bsupvmnc_d_s",
        "bsubsmns",
        "bsubumnc",
        "bsubvmnc",
    )
    for field_name in mode_fields:
        actual = vmec_spline_eval(getattr(frozen, field_name), s)
        expected = _stack_cpu_spline_values(getattr(splines, field_name), s)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12)


def test_vmec_frozen_spline_derivative_matches_cpu_derivative_splines():
    """Oracle: CPU derivative splines sourced from the same FITPACK state."""
    vmec = Vmec(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
    splines = vmec_splines(vmec)
    frozen = vmec_freeze_splines(splines)
    s = np.array([0.25, 0.5, 0.75])

    scalar_pairs = (
        ("pressure", "d_pressure_d_s"),
        ("iota", "d_iota_d_s"),
    )
    for value_name, derivative_name in scalar_pairs:
        actual = vmec_spline_deriv_eval(getattr(frozen, value_name), s)
        expected = getattr(splines, derivative_name)(s)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12)

    mode_pairs = (
        ("rmnc", "d_rmnc_d_s"),
        ("zmns", "d_zmns_d_s"),
        ("lmns", "d_lmns_d_s"),
        ("bmnc", "d_bmnc_d_s"),
        ("bsupumnc", "d_bsupumnc_d_s"),
        ("bsupvmnc", "d_bsupvmnc_d_s"),
    )
    for value_name, derivative_name in mode_pairs:
        actual = vmec_spline_deriv_eval(getattr(frozen, value_name), s)
        expected = _stack_cpu_spline_values(getattr(splines, derivative_name), s)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12)


def test_vmec_freeze_splines_zeroes_stellsym_asymmetry_tables():
    """Oracle: stellsym VMEC geometry never consumes asymmetric mode values."""
    vmec = Vmec(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
    frozen = vmec_freeze_splines(vmec)
    s = np.array([0.25, 0.5, 0.75])

    assert frozen.stellsym
    for field_name, mode_count in (
        ("rmns", frozen.mnmax),
        ("zmnc", frozen.mnmax),
        ("lmnc", frozen.mnmax),
        ("d_rmns_d_s", frozen.mnmax),
        ("d_zmnc_d_s", frozen.mnmax),
        ("d_lmnc_d_s", frozen.mnmax),
        ("gmns", frozen.mnmax_nyq),
        ("bmns", frozen.mnmax_nyq),
        ("d_bmns_d_s", frozen.mnmax_nyq),
        ("bsupumns", frozen.mnmax_nyq),
        ("bsupvmns", frozen.mnmax_nyq),
        ("d_bsupumns_d_s", frozen.mnmax_nyq),
        ("d_bsupvmns_d_s", frozen.mnmax_nyq),
        ("bsubsmnc", frozen.mnmax_nyq),
        ("bsubumns", frozen.mnmax_nyq),
        ("bsubvmns", frozen.mnmax_nyq),
    ):
        actual = vmec_spline_eval(getattr(frozen, field_name), s)
        np.testing.assert_allclose(
            np.asarray(actual),
            np.zeros((s.size, mode_count)),
            rtol=0.0,
            atol=0.0,
        )


def test_vmec_freeze_splines_non_stellsym_asymmetry_tables_match_cpu():
    """Oracle: non-stellsym CPU ``vmec_splines`` asymmetric mode tables."""
    vmec = Vmec(os.path.join(TEST_DIR, "wout_10x10.nc"))
    splines = vmec_splines(vmec)
    frozen = vmec_freeze_splines(splines)
    s = np.array([0.25, 0.5, 0.75])

    assert not frozen.stellsym
    for field_name in (
        "rmns",
        "zmnc",
        "lmnc",
        "d_rmns_d_s",
        "d_zmnc_d_s",
        "d_lmnc_d_s",
        "gmns",
        "bmns",
        "d_bmns_d_s",
        "bsupumns",
        "bsupvmns",
        "d_bsupumns_d_s",
        "d_bsupvmns_d_s",
        "bsubsmnc",
        "bsubumns",
        "bsubvmns",
    ):
        actual = vmec_spline_eval(getattr(frozen, field_name), s)
        expected = _stack_cpu_spline_values(getattr(splines, field_name), s)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12)
