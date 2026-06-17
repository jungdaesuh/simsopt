"""Coverage for ``--stage2-plasma-surface-path`` (``load_stage2_plasma_surface``).

Warm-start coherence lever: load a saved plasma surface directly as the Stage-2
SquaredFlux field-fit target (overriding the wout-re-derived working surface).
These assert observable behaviour of the loader -- a round-tripped Surface loads,
its field-period count is validated against the device, and bad inputs raise. The
BoozerSurface unwrap and the in-``main`` field-fit override are exercised
end-to-end by the live warm-start smoke (a converged seed reaching ~3e-4 field
error against its own saved boozer surface).
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from STAGE_2.banana_coil_solver import load_stage2_plasma_surface  # noqa: E402
from simsopt import save  # noqa: E402
from simsopt.field import Current  # noqa: E402
from simsopt.geo import SurfaceRZFourier  # noqa: E402


def _surf(nfp: int = 5) -> SurfaceRZFourier:
    surface = SurfaceRZFourier(nfp=nfp, stellsym=True, mpol=1, ntor=0)
    surface.set_rc(0, 0, 0.95)
    surface.set_rc(1, 0, 0.10)
    surface.set_zs(1, 0, 0.10)
    return surface


def test_loads_bare_surface_and_validates_nfp(tmp_path):
    """A saved Surface loads, passes the nfp gate, and is usable as a field target."""
    path = tmp_path / "surf.json"
    save(_surf(nfp=5), str(path))

    out = load_stage2_plasma_surface(str(path), expected_nfp=5)

    assert int(out.nfp) == 5
    assert hasattr(out, "gamma")
    assert out.volume() == pytest.approx(_surf(nfp=5).volume())


def test_nfp_mismatch_raises(tmp_path):
    """A surface whose nfp differs from the device is rejected (no silent accept)."""
    path = tmp_path / "surf.json"
    save(_surf(nfp=5), str(path))

    with pytest.raises(ValueError, match="nfp 5 does not match"):
        load_stage2_plasma_surface(str(path), expected_nfp=3)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ValueError, match="file not found"):
        load_stage2_plasma_surface(str(tmp_path / "nope.json"), expected_nfp=5)


def test_non_surface_artifact_raises(tmp_path):
    """A saved non-Surface simsopt object is rejected loudly, never used as a target."""
    path = tmp_path / "notsurf.json"
    save(Current(1.0), str(path))

    with pytest.raises(ValueError, match="did not load a Surface"):
        load_stage2_plasma_surface(str(path), expected_nfp=5)
