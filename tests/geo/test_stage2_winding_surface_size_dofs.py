"""Coverage for the opt-in bounded winding-surface SIZE dofs (T1.2).

``configure_winding_surface_shape_dofs`` gained two independent levers,
``free_r0`` (the major-radius translation ``rc(0,0)``) and ``free_minor`` (the
minor-radius pair ``rc(1,0)`` / ``zs(1,0)``). These are real continuation knobs
on the on-spec mpol=1/ntor=0 winding surface, where the shape-mode range frees
nothing net. Default OFF must reproduce the historical fixed-CWS behavior.

The assertions read OBSERVABLE simsopt-Optimizable state (which dofs are free
and their (lower, upper) bounds), not the implementation.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from STAGE_2.banana_coil_solver import (  # noqa: E402
    validate_winding_surface_shape_cli_args,
)
from banana_opt.stage2_geometry import (  # noqa: E402
    WINDING_SURFACE_FREE_MINOR_BOUNDS_M,
    WINDING_SURFACE_FREE_R0_BOUNDS_M,
    configure_winding_surface_shape_dofs,
)
from simsopt.geo import SurfaceRZFourier  # noqa: E402


def _on_spec_cws() -> SurfaceRZFourier:
    """A representative on-spec coil-winding surface: nfp=5 mpol=1/ntor=0."""
    surface = SurfaceRZFourier(nfp=5, stellsym=True, mpol=1, ntor=0)
    surface.set_rc(0, 0, 0.903)
    surface.set_rc(1, 0, 0.142)
    surface.set_zs(1, 0, 0.142)
    return surface


def _bounds_of(surf: SurfaceRZFourier, name: str) -> tuple[float, float]:
    """Read the (lower, upper) bound recorded for ``name`` (free or fixed)."""
    names = list(surf.local_full_dof_names)
    index = names.index(name)
    lower = float(surf.local_full_lower_bounds[index])
    upper = float(surf.local_full_upper_bounds[index])
    return lower, upper


def test_default_off_frees_no_size_dofs():
    """(a) Default OFF excludes rc(0,0)/rc(1,0)/zs(1,0) -- legacy behavior.

    Default OFF must also leave the bound arrays byte-identical to a pristine
    surface that was only ``fix_all()``-ed: it must not call
    set_lower_bound/set_upper_bound on any DOF.
    """
    surf = _on_spec_cws()
    ref = _on_spec_cws()
    ref.fix_all()

    free_names = configure_winding_surface_shape_dofs(
        surf,
        free_mpol=0,
        free_ntor=0,
        free_r0=False,
        free_minor=False,
    )

    assert free_names == ()
    assert "rc(0,0)" not in free_names
    assert "rc(1,0)" not in free_names
    assert "zs(1,0)" not in free_names
    assert surf.is_fixed("rc(0,0)")
    assert surf.is_fixed("rc(1,0)")
    assert surf.is_fixed("zs(1,0)")
    assert np.array_equal(
        surf.local_full_lower_bounds, ref.local_full_lower_bounds
    )
    assert np.array_equal(
        surf.local_full_upper_bounds, ref.local_full_upper_bounds
    )


def test_free_r0_unfixes_and_bounds_major_radius():
    """(b) free_r0 unfixes rc(0,0) with the vessel-clearance corridor bound."""
    surf = _on_spec_cws()

    free_names = configure_winding_surface_shape_dofs(
        surf,
        free_r0=True,
    )

    assert "rc(0,0)" in free_names
    assert not surf.is_fixed("rc(0,0)")
    assert _bounds_of(surf, "rc(0,0)") == WINDING_SURFACE_FREE_R0_BOUNDS_M
    assert _bounds_of(surf, "rc(0,0)") == (0.908, 0.993)
    # free_minor was not requested: the minor pair stays pinned.
    assert "rc(1,0)" not in free_names
    assert "zs(1,0)" not in free_names
    assert surf.is_fixed("rc(1,0)")
    assert surf.is_fixed("zs(1,0)")


def test_free_minor_unfixes_and_bounds_minor_radius():
    """(c) free_minor unfixes rc(1,0)/zs(1,0) with the minor-a bound."""
    surf = _on_spec_cws()

    free_names = configure_winding_surface_shape_dofs(
        surf,
        free_minor=True,
    )

    assert "rc(1,0)" in free_names
    assert "zs(1,0)" in free_names
    assert not surf.is_fixed("rc(1,0)")
    assert not surf.is_fixed("zs(1,0)")
    assert _bounds_of(surf, "rc(1,0)") == WINDING_SURFACE_FREE_MINOR_BOUNDS_M
    assert _bounds_of(surf, "zs(1,0)") == WINDING_SURFACE_FREE_MINOR_BOUNDS_M
    assert _bounds_of(surf, "rc(1,0)") == (0.142, 0.20)
    # free_r0 was not requested: the major radius stays pinned.
    assert "rc(0,0)" not in free_names
    assert surf.is_fixed("rc(0,0)")


def test_free_r0_survives_shape_mode_path():
    """(d) free_r0 with free_mpol>0: rc(0,0) stays free+bounded, not re-fixed."""
    surf = _on_spec_cws()

    free_names = configure_winding_surface_shape_dofs(
        surf,
        free_mpol=1,
        free_ntor=0,
        free_r0=True,
    )

    assert "rc(0,0)" in free_names
    assert not surf.is_fixed("rc(0,0)")
    assert _bounds_of(surf, "rc(0,0)") == WINDING_SURFACE_FREE_R0_BOUNDS_M
    # free_minor not requested: even though the mpol=1 window touches the minor
    # modes, the re-fix pins them back because they were not requested free.
    assert "rc(1,0)" not in free_names
    assert "zs(1,0)" not in free_names
    assert surf.is_fixed("rc(1,0)")
    assert surf.is_fixed("zs(1,0)")


def test_free_dof_perturbation_moves_surface_geometry():
    """A freed size dof actually drives the winding-surface geometry."""
    surf = _on_spec_cws()
    before = surf.gamma().copy()

    configure_winding_surface_shape_dofs(surf, free_r0=True)

    dofs = surf.x.copy()
    free_index = list(surf.local_dof_names).index("rc(0,0)")
    dofs[free_index] += 1.0e-3
    surf.x = dofs

    assert np.max(np.abs(surf.gamma() - before)) > 0.0


def _loaded_seed_args(**overrides) -> SimpleNamespace:
    """Minimal args object mirroring how the validator reads its attrs.

    The validator reads loose attributes via getattr; a loaded seed is signalled
    by a truthy ``stage2_bs_path``.
    """
    base = dict(
        stage2_bs_path="/seed/banana_bs.json",
        winding_surface_free_mpol=0,
        winding_surface_free_ntor=0,
        winding_surface_free_r0=False,
        winding_surface_free_minor=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_loaded_seed_with_free_r0_raises():
    """free_r0 on a loaded seed is rejected (fresh-only guard, FIX 1)."""
    args = _loaded_seed_args(winding_surface_free_r0=True)
    with pytest.raises(ValueError, match="requires fresh Stage 2 initialization"):
        validate_winding_surface_shape_cli_args(args)


def test_loaded_seed_with_free_minor_raises():
    """free_minor on a loaded seed is rejected (fresh-only guard, FIX 1)."""
    args = _loaded_seed_args(winding_surface_free_minor=True)
    with pytest.raises(ValueError, match="requires fresh Stage 2 initialization"):
        validate_winding_surface_shape_cli_args(args)


def test_loaded_seed_without_size_frees_passes():
    """A loaded seed with no winding-surface-free flags is accepted."""
    validate_winding_surface_shape_cli_args(_loaded_seed_args())
