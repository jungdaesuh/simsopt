"""Contract tests for the Boozer-surface artifact SSOT save guard.

``save_boozer_finite_i`` is the sanctioned save for ``*_boozer_surface.json``
artifacts (the counterpart to ``load_boozer_finite_i``). The clearance viewer and
the topology/Poincare consumers render the COILS embedded in the saved SIMSON
graph, so a *surface-only* artifact (a bare ``Surface`` with no ``BiotSavart``)
is a silent corruption that only surfaces downstream as the viewer
``NoBiotSavart`` error. These tests pin the guard: it refuses -- before writing --
a save that would produce such a graph, and lets a real ``BoozerSurface`` through
as a renderable graph.

Each assertion is an externally checkable consequence of the contract (the saved
JSON graph / the raise), not a mirror of the implementation.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _EXAMPLE_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simsopt.field import BiotSavart  # noqa: E402
from simsopt.field.coil import Coil, Current  # noqa: E402
from simsopt.geo import (  # noqa: E402
    CurveXYZFourier,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    Volume,
)
from simsopt.geo.boozersurface import BoozerSurface  # noqa: E402

from banana_opt.boozer_finite_current import BoozerSurfaceFiniteI  # noqa: E402
from banana_opt.json_compat import save_boozer_finite_i  # noqa: E402


def _winding_surface() -> SurfaceRZFourier:
    surf = SurfaceRZFourier(nfp=2, stellsym=True, mpol=1, ntor=1)
    surf.set_rc(0, 0, 1.0)
    surf.set_rc(1, 0, 0.3)
    surf.set_zs(1, 0, 0.3)
    return surf


def _planar_coil() -> Coil:
    curve = CurveXYZFourier(64, 1)
    x = np.zeros_like(curve.x)
    x[0] = 1.2                      # xc(0)
    x[2] = 0.4                      # xc(1)
    x[1 + 2 * (2 * 1 + 1)] = 0.4    # ys(1) -> a closed planar loop
    curve.x = x
    return Coil(curve, Current(50000.0))


def _tensor_surface() -> SurfaceXYZTensorFourier:
    # BoozerSurface requires a SurfaceXYZTensorFourier / SurfaceXYZFourier.
    mpol = ntor = 1
    nfp = 2
    return SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False),
    )


def _boozer_surface(coils) -> BoozerSurface:
    # The target label is a stored scalar (no solve here), so a dummy positive
    # value is enough for a save-contract test.
    surf = _tensor_surface()
    return BoozerSurface(BiotSavart(list(coils)), surf, Volume(surf), 1.0)


def _boozer_surface_finite_i(coils) -> BoozerSurfaceFiniteI:
    surf = _tensor_surface()
    return BoozerSurfaceFiniteI(BiotSavart(list(coils)), surf, Volume(surf), 1.0, I=1000.0)


def _graph_class_counts(path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for obj in payload.get("simsopt_objs", {}).values():
        if isinstance(obj, dict) and obj.get("@class"):
            counts[obj["@class"]] = counts.get(obj["@class"], 0) + 1
    return counts


def test_save_boozer_finite_i_rejects_surface_only(tmp_path):
    """A bare Surface saved under a boozer name is the recurring viewer failure;
    the guard refuses it BEFORE writing, so no surface-only file is left behind."""
    out = tmp_path / "surf_bad_boozer_surface.json"
    with pytest.raises(ValueError, match="SurfaceRZFourier"):
        save_boozer_finite_i(_winding_surface(), out)
    assert not out.exists()


def test_save_boozer_finite_i_rejects_fieldless_boozer_surface(tmp_path):
    """A BoozerSurface whose field carries no coils is non-renderable (nothing for
    the viewer to draw); the guard rejects it on the coil count, not the type."""
    out = tmp_path / "surf_empty_boozer_surface.json"
    with pytest.raises(ValueError, match=r"BoozerSurface carrying 0"):
        save_boozer_finite_i(_boozer_surface([]), out)
    assert not out.exists()


def test_save_boozer_finite_i_writes_renderable_graph(tmp_path):
    """A real BoozerSurface is written as a graph that embeds the BiotSavart +
    coils the viewer / topology consumers require -- the fix, checked on the
    SERIALIZED artifact, not on the in-memory object."""
    out = tmp_path / "surf_ok_boozer_surface.json"
    save_boozer_finite_i(_boozer_surface([_planar_coil()]), out)
    counts = _graph_class_counts(out)
    assert counts.get("BiotSavart", 0) >= 1
    assert counts.get("Coil", 0) >= 1
    assert any(name.startswith("BoozerSurface") for name in counts)


def test_save_boozer_finite_i_writes_finite_i_graph(tmp_path):
    """The finite-I subclass (the module's namesake) is also accepted and
    serialized with its BoozerSurfaceFiniteI node plus the embedded BiotSavart
    and coils -- pins that the guard's isinstance gate covers the subclass."""
    out = tmp_path / "surf_finitei_boozer_surface.json"
    save_boozer_finite_i(_boozer_surface_finite_i([_planar_coil()]), out)
    counts = _graph_class_counts(out)
    assert counts.get("BoozerSurfaceFiniteI", 0) >= 1
    assert counts.get("BiotSavart", 0) >= 1
    assert counts.get("Coil", 0) >= 1
