"""Frozen VMEC spline state for JAX geometry diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp

from simsopt.jax_core._math_utils import as_jax_float64
from simsopt.jax_core._spline_utils import bspline_deriv_1d, bspline_eval_1d

__all__ = [
    "VmecFrozenSplineState",
    "VmecSplineData",
    "vmec_freeze_splines",
    "vmec_spline_deriv_eval",
    "vmec_spline_eval",
]


@dataclass(frozen=True)
class VmecSplineData:
    """FITPACK spline coefficients for one scalar profile or a mode table."""

    knots: jax.Array
    coeffs: jax.Array
    degree: int


jax.tree_util.register_dataclass(
    VmecSplineData,
    data_fields=["knots", "coeffs"],
    meta_fields=["degree"],
)


@dataclass(frozen=True)
class VmecFrozenSplineState:
    """Immutable spline payload produced by :func:`vmec_splines`."""

    xm: jax.Array
    xn: jax.Array
    xm_nyq: jax.Array
    xn_nyq: jax.Array
    Aminor_p: jax.Array
    phiedge: jax.Array
    pressure: VmecSplineData
    d_pressure_d_s: VmecSplineData
    iota: VmecSplineData
    d_iota_d_s: VmecSplineData
    rmnc: VmecSplineData
    zmns: VmecSplineData
    lmns: VmecSplineData
    d_rmnc_d_s: VmecSplineData
    d_zmns_d_s: VmecSplineData
    d_lmns_d_s: VmecSplineData
    rmns: VmecSplineData
    zmnc: VmecSplineData
    lmnc: VmecSplineData
    d_rmns_d_s: VmecSplineData
    d_zmnc_d_s: VmecSplineData
    d_lmnc_d_s: VmecSplineData
    gmnc: VmecSplineData
    bmnc: VmecSplineData
    d_bmnc_d_s: VmecSplineData
    bsupumnc: VmecSplineData
    bsupvmnc: VmecSplineData
    d_bsupumnc_d_s: VmecSplineData
    d_bsupvmnc_d_s: VmecSplineData
    bsubsmns: VmecSplineData
    bsubumnc: VmecSplineData
    bsubvmnc: VmecSplineData
    gmns: VmecSplineData
    bmns: VmecSplineData
    d_bmns_d_s: VmecSplineData
    bsupumns: VmecSplineData
    bsupvmns: VmecSplineData
    d_bsupumns_d_s: VmecSplineData
    d_bsupvmns_d_s: VmecSplineData
    bsubsmnc: VmecSplineData
    bsubumns: VmecSplineData
    bsubvmns: VmecSplineData
    stellsym: bool
    mnmax: int
    mnmax_nyq: int
    nfp: int


jax.tree_util.register_dataclass(
    VmecFrozenSplineState,
    data_fields=[
        "xm",
        "xn",
        "xm_nyq",
        "xn_nyq",
        "Aminor_p",
        "phiedge",
        "pressure",
        "d_pressure_d_s",
        "iota",
        "d_iota_d_s",
        "rmnc",
        "zmns",
        "lmns",
        "d_rmnc_d_s",
        "d_zmns_d_s",
        "d_lmns_d_s",
        "rmns",
        "zmnc",
        "lmnc",
        "d_rmns_d_s",
        "d_zmnc_d_s",
        "d_lmnc_d_s",
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
    ],
    meta_fields=["stellsym", "mnmax", "mnmax_nyq", "nfp"],
)


_MODE_SPLINE_FIELDS = (
    "rmnc",
    "zmns",
    "lmns",
    "d_rmnc_d_s",
    "d_zmns_d_s",
    "d_lmns_d_s",
)

_ASYM_MODE_REFERENCES = {
    "rmns": "rmnc",
    "zmnc": "zmns",
    "lmnc": "lmns",
    "d_rmns_d_s": "d_rmnc_d_s",
    "d_zmnc_d_s": "d_zmns_d_s",
    "d_lmnc_d_s": "d_lmns_d_s",
}

_NYQ_SPLINE_FIELDS = (
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

_ASYM_NYQ_REFERENCES = {
    "gmns": "gmnc",
    "bmns": "bmnc",
    "d_bmns_d_s": "d_bmnc_d_s",
    "bsupumns": "bsupumnc",
    "bsupvmns": "bsupvmnc",
    "d_bsupumns_d_s": "d_bsupumnc_d_s",
    "d_bsupvmns_d_s": "d_bsupvmnc_d_s",
    "bsubsmnc": "bsubsmns",
    "bsubumns": "bsubumnc",
    "bsubvmns": "bsubvmnc",
}


def _is_vmec_object(value) -> bool:
    return any(
        cls.__module__ == "simsopt.mhd.vmec" and cls.__name__ == "Vmec"
        for cls in type(value).__mro__
    )


def _spline_data(spline) -> VmecSplineData:
    knots, coeffs, degree = spline._eval_args
    return VmecSplineData(
        knots=as_jax_float64(np.asarray(knots, dtype=np.float64)),
        coeffs=as_jax_float64(np.asarray(coeffs, dtype=np.float64)),
        degree=int(degree),
    )


def _stack_spline_table(splines) -> VmecSplineData:
    entries = tuple(_spline_data(spline) for spline in splines)
    degree = entries[0].degree
    return VmecSplineData(
        knots=as_jax_float64(np.stack([np.asarray(entry.knots) for entry in entries])),
        coeffs=as_jax_float64(
            np.stack([np.asarray(entry.coeffs) for entry in entries])
        ),
        degree=degree,
    )


def _zero_spline_table_like(splines) -> VmecSplineData:
    table = _stack_spline_table(splines)
    return VmecSplineData(
        knots=table.knots,
        coeffs=jnp.zeros_like(table.coeffs),
        degree=table.degree,
    )


def _freeze_asym_table(vs, field_name: str, reference_name: str) -> VmecSplineData:
    if vs.stellsym:
        return _zero_spline_table_like(getattr(vs, reference_name))
    return _stack_spline_table(getattr(vs, field_name))


def vmec_freeze_splines(vmec_or_splines) -> VmecFrozenSplineState:
    """Freeze VMEC radial splines into a JAX pytree.

    ``vmec_or_splines`` may be a live :class:`~simsopt.mhd.vmec.Vmec`
    instance or the host ``Struct`` returned by :func:`vmec_splines`.
    """
    if _is_vmec_object(vmec_or_splines):
        from .vmec_diagnostics import vmec_splines

        vs = vmec_splines(vmec_or_splines)
    else:
        vs = vmec_or_splines
    mode_tables = {
        field_name: _stack_spline_table(getattr(vs, field_name))
        for field_name in _MODE_SPLINE_FIELDS
    }
    asym_mode_tables = {
        field_name: _freeze_asym_table(vs, field_name, reference_name)
        for field_name, reference_name in _ASYM_MODE_REFERENCES.items()
    }
    nyq_tables = {
        field_name: _stack_spline_table(getattr(vs, field_name))
        for field_name in _NYQ_SPLINE_FIELDS
    }
    asym_nyq_tables = {
        field_name: _freeze_asym_table(vs, field_name, reference_name)
        for field_name, reference_name in _ASYM_NYQ_REFERENCES.items()
    }
    return VmecFrozenSplineState(
        xm=as_jax_float64(vs.xm),
        xn=as_jax_float64(vs.xn),
        xm_nyq=as_jax_float64(vs.xm_nyq),
        xn_nyq=as_jax_float64(vs.xn_nyq),
        Aminor_p=as_jax_float64(vs.Aminor_p),
        phiedge=as_jax_float64(vs.phiedge),
        pressure=_spline_data(vs.pressure),
        d_pressure_d_s=_spline_data(vs.d_pressure_d_s),
        iota=_spline_data(vs.iota),
        d_iota_d_s=_spline_data(vs.d_iota_d_s),
        rmnc=mode_tables["rmnc"],
        zmns=mode_tables["zmns"],
        lmns=mode_tables["lmns"],
        d_rmnc_d_s=mode_tables["d_rmnc_d_s"],
        d_zmns_d_s=mode_tables["d_zmns_d_s"],
        d_lmns_d_s=mode_tables["d_lmns_d_s"],
        rmns=asym_mode_tables["rmns"],
        zmnc=asym_mode_tables["zmnc"],
        lmnc=asym_mode_tables["lmnc"],
        d_rmns_d_s=asym_mode_tables["d_rmns_d_s"],
        d_zmnc_d_s=asym_mode_tables["d_zmnc_d_s"],
        d_lmnc_d_s=asym_mode_tables["d_lmnc_d_s"],
        gmnc=nyq_tables["gmnc"],
        bmnc=nyq_tables["bmnc"],
        d_bmnc_d_s=nyq_tables["d_bmnc_d_s"],
        bsupumnc=nyq_tables["bsupumnc"],
        bsupvmnc=nyq_tables["bsupvmnc"],
        d_bsupumnc_d_s=nyq_tables["d_bsupumnc_d_s"],
        d_bsupvmnc_d_s=nyq_tables["d_bsupvmnc_d_s"],
        bsubsmns=nyq_tables["bsubsmns"],
        bsubumnc=nyq_tables["bsubumnc"],
        bsubvmnc=nyq_tables["bsubvmnc"],
        gmns=asym_nyq_tables["gmns"],
        bmns=asym_nyq_tables["bmns"],
        d_bmns_d_s=asym_nyq_tables["d_bmns_d_s"],
        bsupumns=asym_nyq_tables["bsupumns"],
        bsupvmns=asym_nyq_tables["bsupvmns"],
        d_bsupumns_d_s=asym_nyq_tables["d_bsupumns_d_s"],
        d_bsupvmns_d_s=asym_nyq_tables["d_bsupvmns_d_s"],
        bsubsmnc=asym_nyq_tables["bsubsmnc"],
        bsubumns=asym_nyq_tables["bsubumns"],
        bsubvmns=asym_nyq_tables["bsubvmns"],
        stellsym=bool(vs.stellsym),
        mnmax=int(vs.mnmax),
        mnmax_nyq=int(vs.mnmax_nyq),
        nfp=int(vs.nfp),
    )


def vmec_spline_eval(spline_data: VmecSplineData, s) -> jax.Array:
    """Evaluate frozen scalar or mode-table spline data at ``s``."""
    s_jax = as_jax_float64(s)
    if spline_data.coeffs.ndim == 1:
        return bspline_eval_1d(
            spline_data.knots, spline_data.coeffs, spline_data.degree, s_jax
        )
    values_by_mode = jax.vmap(
        lambda knots, coeffs: bspline_eval_1d(
            knots,
            coeffs,
            spline_data.degree,
            s_jax,
        )
    )(spline_data.knots, spline_data.coeffs)
    return jnp.moveaxis(values_by_mode, 0, -1)


def vmec_spline_deriv_eval(spline_data: VmecSplineData, s) -> jax.Array:
    """Evaluate d/ds for frozen scalar or mode-table spline data."""
    s_jax = as_jax_float64(s)
    if spline_data.coeffs.ndim == 1:
        return bspline_deriv_1d(
            spline_data.knots, spline_data.coeffs, spline_data.degree, s_jax
        )
    values_by_mode = jax.vmap(
        lambda knots, coeffs: bspline_deriv_1d(
            knots,
            coeffs,
            spline_data.degree,
            s_jax,
        )
    )(spline_data.knots, spline_data.coeffs)
    return jnp.moveaxis(values_by_mode, 0, -1)
