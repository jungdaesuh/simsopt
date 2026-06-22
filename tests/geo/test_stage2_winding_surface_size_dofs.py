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
    build_lbfgsb_bounds,
    free_loaded_winding_surface_size_dofs,
    validate_winding_surface_shape_cli_args,
)
from alm_utils import _build_box_bounds  # noqa: E402
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
    assert _bounds_of(surf, "rc(0,0)") == (0.903, 0.993)
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
    assert _bounds_of(surf, "rc(1,0)") == (0.130, 0.20)
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


def test_build_lbfgsb_bounds_exposes_free_r0_corridor():
    """B1.1(1): build_lbfgsb_bounds carries the freed rc(0,0) corridor bound.

    This is the bound the ALM path threads to the inner optimizer; it must be the
    vessel-clearance corridor, not +/-inf.
    """
    surf = _on_spec_cws()
    configure_winding_surface_shape_dofs(surf, free_r0=True)

    names = list(surf.local_dof_names)
    bounds = build_lbfgsb_bounds(surf)

    assert bounds[names.index("rc(0,0)")] == (0.903, 0.993)


def test_box_bounds_clamp_free_r0_to_corridor_lower_bound():
    """B1.1(2): the trust box intersected with the corridor clamps rc(0,0).

    The 0.015 trust box around the on-spec seed rc(0,0)=0.903 reaches down to
    0.888, but the 0.903 corridor floor clamps the lower bound; the trust radius
    caps the upper bound below the 0.993 corridor ceiling.
    """
    surf = _on_spec_cws()
    configure_winding_surface_shape_dofs(surf, free_r0=True)

    names = list(surf.local_dof_names)
    index = names.index("rc(0,0)")
    box_bounds = _build_box_bounds(
        surf.x,
        0.015,
        base_bounds=build_lbfgsb_bounds(surf),
    )
    lower, upper = box_bounds[index]

    # Corridor floor 0.903 clamps the lower bound (the trust box reaches 0.888);
    # the 0.015 trust box around 0.903 caps the upper bound at ~0.918, inside the
    # 0.993 corridor ceiling.
    assert lower == pytest.approx(0.903)
    assert upper == pytest.approx(0.918)


def test_box_bounds_default_off_matches_trust_only():
    """B1.1(3): with nothing freed, base_bounds (all +/-inf) is a no-op.

    The intersection with the trust box reproduces the trust-box-only behavior
    byte-identically -- the default ALM path is unchanged.
    """
    surf = _on_spec_cws()
    configure_winding_surface_shape_dofs(surf, free_r0=False)

    with_base = _build_box_bounds(
        surf.x,
        0.015,
        base_bounds=build_lbfgsb_bounds(surf),
    )
    trust_only = _build_box_bounds(surf.x, 0.015, base_bounds=None)

    assert with_base == trust_only


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


def test_loaded_seed_with_free_r0_is_allowed():
    """T1.5: free_r0 on a loaded seed is ALLOWED.

    Re-centering an already-converged seed is the intended use of the major-
    radius lever, so the validator no longer rejects it (the wiring frees
    rc(0,0) directly on the loaded master winding surface).
    """
    validate_winding_surface_shape_cli_args(
        _loaded_seed_args(winding_surface_free_r0=True)
    )


def test_loaded_seed_with_free_minor_is_allowed():
    """T1.5: free_minor on a loaded seed is ALLOWED (re-sizing a converged seed)."""
    validate_winding_surface_shape_cli_args(
        _loaded_seed_args(winding_surface_free_minor=True)
    )


def test_loaded_seed_with_free_mpol_still_raises():
    """The shape-mode frees stay fresh-only: a loaded seed's recorded (m, n)
    modes are what its coils converged against, so reopening them still raises."""
    args = _loaded_seed_args(winding_surface_free_mpol=1)
    with pytest.raises(ValueError, match="fresh Stage 2 initialization"):
        validate_winding_surface_shape_cli_args(args)


def test_loaded_seed_with_free_ntor_still_raises():
    """The ntor shape-mode free is likewise fresh-only on a loaded seed."""
    args = _loaded_seed_args(winding_surface_free_ntor=1)
    with pytest.raises(ValueError, match="fresh Stage 2 initialization"):
        validate_winding_surface_shape_cli_args(args)


def test_loaded_seed_without_size_frees_passes():
    """A loaded seed with no winding-surface-free flags is accepted."""
    validate_winding_surface_shape_cli_args(_loaded_seed_args())


def _fake_loaded_banana_curve():
    """Stand-in for the loaded master banana curve: only ``.surf`` is read by the
    loaded-seed wiring (``free_loaded_winding_surface_size_dofs``)."""
    return SimpleNamespace(surf=_on_spec_cws())


def test_free_loaded_winding_dofs_off_is_byte_identical_noop():
    """T1.5 wiring, default OFF: returns () and leaves the loaded surf UNCHANGED.

    Guards the loaded-seed branch directly (not just the validator): the helper
    must not mutate the master winding surface when no size lever is requested.
    Asserted as state-invariance (the real loaded surf arrives with its own
    fixed/free state from the artifact; the helper must preserve it exactly).
    """
    curve = _fake_loaded_banana_curve()
    before_fixed = [
        curve.surf.is_fixed(name) for name in curve.surf.local_full_dof_names
    ]
    before_x = curve.surf.x.copy()
    before_lb = curve.surf.local_full_lower_bounds.copy()
    before_ub = curve.surf.local_full_upper_bounds.copy()

    names = free_loaded_winding_surface_size_dofs(curve, _loaded_seed_args())

    assert names == ()
    assert [
        curve.surf.is_fixed(name) for name in curve.surf.local_full_dof_names
    ] == before_fixed
    assert np.array_equal(curve.surf.x, before_x)
    assert np.array_equal(curve.surf.local_full_lower_bounds, before_lb)
    assert np.array_equal(curve.surf.local_full_upper_bounds, before_ub)


def test_free_loaded_winding_dofs_free_r0_frees_bounded_on_loaded_surf():
    """T1.5 wiring: free_r0 frees+bounds rc(0,0) on the LOADED master surf.

    This is the regression guard for the loaded-seed branch -- a misroute (wrong
    surface object, dropped guard, renamed flag) would leave rc(0,0) pinned and
    fail here, instead of silently no-opping at run time.
    """
    curve = _fake_loaded_banana_curve()

    names = free_loaded_winding_surface_size_dofs(
        curve, _loaded_seed_args(winding_surface_free_r0=True)
    )

    assert "rc(0,0)" in names
    assert not curve.surf.is_fixed("rc(0,0)")
    assert _bounds_of(curve.surf, "rc(0,0)") == WINDING_SURFACE_FREE_R0_BOUNDS_M
    # free_minor not requested: the minor pair stays pinned.
    assert curve.surf.is_fixed("rc(1,0)")
    assert curve.surf.is_fixed("zs(1,0)")


def test_free_loaded_winding_dofs_free_minor_frees_bounded_on_loaded_surf():
    """T1.5 wiring: free_minor frees+bounds rc(1,0)/zs(1,0) on the loaded surf."""
    curve = _fake_loaded_banana_curve()

    names = free_loaded_winding_surface_size_dofs(
        curve, _loaded_seed_args(winding_surface_free_minor=True)
    )

    assert "rc(1,0)" in names
    assert "zs(1,0)" in names
    assert _bounds_of(curve.surf, "rc(1,0)") == WINDING_SURFACE_FREE_MINOR_BOUNDS_M
    assert _bounds_of(curve.surf, "zs(1,0)") == WINDING_SURFACE_FREE_MINOR_BOUNDS_M
    assert curve.surf.is_fixed("rc(0,0)")


def test_alm_seed_clip_keeps_trust_box_feasible_for_freed_r0():
    """B1.1: the ALM seed must be clipped into the corridor before the trust box.

    A freed rc(0,0) seed BELOW the corridor floor (here 0.900 < the 0.903 floor)
    is not clipped by ALM the way L-BFGS-B clips x0, so a small trust radius
    around 0.900 never reaches 0.903: the trust-box / corridor intersection
    inverts and raises. The solver clips the seed into the corridor first -- this
    proves the unclipped seed inverts and the clipped seed yields a feasible box.
    """
    surf = _on_spec_cws()
    surf.set_rc(0, 0, 0.900)  # below the 0.903 corridor floor
    configure_winding_surface_shape_dofs(surf, free_r0=True)
    base_bounds = build_lbfgsb_bounds(surf)
    index = list(surf.local_dof_names).index("rc(0,0)")
    tiny_radius = 0.001  # 0.900 +/- 0.001 never reaches the 0.903 corridor floor

    # Unclipped 0.900 seed: trust box (0.899, 0.901) misses the (0.903, 0.993)
    # corridor -> empty intersection -> raises.
    with pytest.raises(ValueError):
        _build_box_bounds(surf.x, tiny_radius, base_bounds=base_bounds)

    # The solver's seed clip lifts rc(0,0) to the 0.903 floor; the box is feasible.
    lower = np.array([lo for lo, _ in base_bounds])
    upper = np.array([hi for _, hi in base_bounds])
    clipped = np.clip(surf.x, lower, upper)
    assert clipped[index] == pytest.approx(0.903)
    box_lower, box_upper = _build_box_bounds(
        clipped, tiny_radius, base_bounds=base_bounds
    )[index]
    assert box_lower <= box_upper
