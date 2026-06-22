"""Contract tests for ``banana_opt.reshape_run_dir`` (post-hoc reshape run-dir hygiene).

Self-contained: no external run artifacts. Each assertion is an externally checkable
consequence of the contract, not a mirror of the implementation:

* ``is_derived_output`` classifies a seed run's DERIVED outputs (results.json,
  confinement_verdict.json, PoincareMetrics, *.viewer.json, *_poincare.vts/vtu, ...)
  as never-copy, and true inputs as copy-eligible -- the recurrence-prevention core.
* ``populate_inputs`` copies only whitelisted inputs and never plants a derived output.
* ``_max_kappa_highres`` carries the CWS secular terms ``G``/``H`` (which
  ``get_dofs()`` does not round-trip): two coils with identical dofs but different
  ``G`` give different curvature, and the helper matches a native dense evaluation.
* ``write_reshape_provenance`` emits an honest record -- the measured reshape metrics,
  a TF-first coil-group manifest, shared inputs -- and never lifts a seed solver
  verdict to the top level.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _EXAMPLE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from banana_opt import reshape_run_dir as R  # noqa: E402
from simsopt.geo import CurveCWSFourierCPP, SurfaceRZFourier  # noqa: E402


def _cws_curve(g: int) -> CurveCWSFourierCPP:
    """A non-degenerate CWS curve on a small winding torus, with secular term G=g.

    The Fourier dofs trace a closed (phi, theta) ellipse (phi=0.3cos, theta=0.3sin)
    so the curve is a genuine 3-D loop with finite curvature even at G=0 (default
    dofs collapse to a point => NaN); G then adds toroidal winding on top.
    """
    surf = SurfaceRZFourier(nfp=2, stellsym=True, mpol=1, ntor=1)
    surf.set_rc(0, 0, 1.0)
    surf.set_rc(1, 0, 0.3)
    surf.set_zs(1, 0, 0.3)
    curve = CurveCWSFourierCPP(np.linspace(0.0, 1.0, 64, endpoint=False), 3, surf, G=g, H=0)
    dofs = np.zeros(len(np.asarray(curve.get_dofs(), dtype=float)))
    dofs[1] = 0.3   # phic[1]:   phi(s)   = 0.3 cos(2*pi*s)
    dofs[11] = 0.3  # thetas[1]: theta(s) = 0.3 sin(2*pi*s)
    curve.set_dofs(dofs)
    return curve


def test_is_derived_output() -> None:
    for n in ("results.json", "confinement_verdict.json", "PoincareMetrics_opt_default.json",
              "PoincarePlot_opt_default.png", "curves_opt_poincare.vtu", "surf_opt_poincare.vts",
              "surf_x.viewer.json", "TransitNormalization_x.json", "results_best.partial.json"):
        assert R.is_derived_output(n), f"{n} must be classified derived"
    for n in ("surf_opt.json", "surf_opt_boozer_surface.json", "biot_savart_init.json"):
        assert not R.is_derived_output(n), f"{n} must be classified an input"


def test_populate_inputs_skips_derived() -> None:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        out = Path(td) / "out"
        src.mkdir()
        (src / "surf_opt.json").write_text("{}")
        (src / "results.json").write_text('{"CONFINEMENT_GATE_ACCEPTED": false}')
        (src / "confinement_verdict.json").write_text("{}")
        (src / "PoincareMetrics_opt_default.json").write_text("{}")
        copied = R.populate_inputs(src, out)
        assert "surf_opt.json" in copied
        assert not (out / "results.json").exists()
        assert not (out / "confinement_verdict.json").exists()
        assert not (out / "PoincareMetrics_opt_default.json").exists()


def test_max_kappa_highres_preserves_GH() -> None:
    c0 = _cws_curve(0)
    c1 = _cws_curve(1)
    k0 = R._max_kappa_highres(c0)
    k1 = R._max_kappa_highres(c1)
    assert np.isfinite(k0) and np.isfinite(k1), (k0, k1)
    # G changes the toroidal route -> curvature; if the rebuild dropped G both would
    # rebuild as G=0 and read identical (the latent fail-open bug this pins).
    assert abs(k1 - k0) > 1e-3, f"G must change kappa; got G=0 {k0} vs G=1 {k1}"
    # the helper must equal a native dense evaluation of the SAME (G=1) curve
    dense = CurveCWSFourierCPP(np.linspace(0.0, 1.0, R.KAPPA_RESOLUTION, endpoint=False),
                               c1.order, c1.surf, G=1, H=0)
    dense.set_dofs(c1.get_dofs())
    assert abs(k1 - float(dense.kappa().max())) < 1e-6, (k1, float(dense.kappa().max()))


def test_write_reshape_provenance_is_honest() -> None:
    m = R.ReshapeMetrics(
        n_banana=10, n_tf=20, max_kappa=37.2, max_coil_length_m=1.7,
        max_poloidal_extent_rad=1.16, keepout_clearance_m=0.0157,
        banana_current_abs_max_a=15992.0, default_survived_lines=38, default_nfieldlines=50,
        volume=0.05, aspect_ratio=19.3, winding_r0_m=0.976,
    )
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base_results.json"
        # a seed results.json: shared inputs + a baseline REJECT verdict that must NOT be carried
        base.write_text(json.dumps({
            "BOOZER_I": 0.0, "NUM_BANANA_COILS": 10, "NUM_TF_COILS": 20,
            "COIL_WINDING_SURFACE_MAJOR_RADIUS_M": 0.976,
            "CONFINEMENT_GATE_ACCEPTED": False, "MAX_CURVATURE": 56.0,
        }))
        out = Path(td) / "run"
        out.mkdir()
        R.write_reshape_provenance(out, base, m, "seed/abc",
                                   reconverged_surface_name="surf_x_reconverged_boozer_surface.json",
                                   curvature_cap_inv_m=43.31)
        prov = json.loads((out / R.PROVENANCE_BASENAME).read_text())

        assert prov["is_full_solver_results"] is False
        # the seed REJECT verdict is NOT lifted to the top level
        assert "CONFINEMENT_GATE_ACCEPTED" not in prov
        assert "MAX_CURVATURE" not in prov
        rm = prov["RESHAPE_MEASURED"]
        assert rm["MAX_CURVATURE"] == 37.2 and rm["FINITEBUILD_CURVATURE_OK"] is True
        assert rm["DEFAULT_TOPOLOGY_SURVIVED_LINES"] == 38
        assert rm["HARDWARE_KEEPOUT_OK"] is True
        roles = {g["role"]: (g["start"], g["count"]) for g in prov[R.COIL_GROUPS_RESULTS_KEY]}
        assert roles["tf"] == (0, 20) and roles["banana"] == (20, 10)
        assert prov["SHARED_INPUTS"]["BOOZER_I"] == 0.0
        # a cap-violating curvature must read OK=False
        m_over = R.ReshapeMetrics(**{**m.__dict__, "max_kappa": 76.0})
        R.write_reshape_provenance(out, base, m_over, "seed/abc",
                                   reconverged_surface_name=None, curvature_cap_inv_m=43.31)
        prov2 = json.loads((out / R.PROVENANCE_BASENAME).read_text())
        assert prov2["RESHAPE_MEASURED"]["FINITEBUILD_CURVATURE_OK"] is False
        assert prov2["decisive_artifacts"]["reconverged_boozer_surface"] is None
