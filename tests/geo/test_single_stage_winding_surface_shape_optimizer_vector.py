"""Additive coverage for single-stage winding-surface SHAPE optimization.

This file complements ``test_single_stage_winding_surface_shape.py`` (which
verifies the re-embed wiring) with the explicit optimizer-vector and
Defect-1/Defect-2 guarantees from RECOMMENDATION-wataru Tier-1 item 5:

* Defect-1 guard: freeing winding-surface shape dofs on the *rebuilt* banana
  family grows the assembled objective vector ``len(JF.x)`` by exactly the freed
  mode count AND perturbing a freed surface dof moves ``banana_curve.gamma()``.
  (Freeing modes on the stale clearance-reference surface would be a silent
  no-op on the optimized coils -- this catches that.)

* Defect-2 documentation: the embedded warm-seed surface (an mpol=1/ntor=0
  circle) carries *no* shape modes, so shape modes can only be optimized by
  rebuilding the coil on a surface that ALREADY has the target (m,n)
  resolution. Trying to free (m,n) shape modes that do not exist on the warm
  surface raises, which is precisely why ``build_hbt_reference_surfaces`` raises
  the resolution BEFORE the coil is constructed.

* Step-3 contract: requesting shaping that resolves to zero free shape dofs
  (the degenerate ``free_mpol=1 free_ntor=0`` window) warns loudly instead of
  silently no-opping.

These tests construct objects directly (no heavy optimization / Poincare run).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
SINGLE_STAGE_ROOT = EXAMPLES_ROOT / "SINGLE_STAGE"
for _path in (str(EXAMPLES_ROOT), str(SINGLE_STAGE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import single_stage_banana_example as module  # noqa: E402
from banana_opt.stage2_geometry import (  # noqa: E402
    configure_winding_surface_shape_dofs,
)
from banana_opt.stage2_single_stage_handoff import Stage2CoilPartitions  # noqa: E402
from simsopt.field import BiotSavart  # noqa: E402
from simsopt.field.coil import Current, coils_via_symmetries  # noqa: E402
from simsopt.geo import (  # noqa: E402
    CurveCWSFourierCPP,
    CurveLength,
    SurfaceRZFourier,
)


def _warm_seed_circle() -> SurfaceRZFourier:
    """The warm-seed embedded winding surface: a plain mpol=1/ntor=0 circle."""
    surface = SurfaceRZFourier(nfp=2, stellsym=True, mpol=1, ntor=0)
    surface.set_rc(0, 0, 0.976)
    surface.set_rc(1, 0, 0.142)
    surface.set_zs(1, 0, 0.142)
    return surface


def _banana_curve(surface: SurfaceRZFourier) -> CurveCWSFourierCPP:
    curve = CurveCWSFourierCPP(
        np.linspace(0.0, 1.0, 32, endpoint=False),
        order=2,
        surf=surface,
        G=3,
        H=5,
    )
    curve.set("phic(1)", 0.30)
    curve.set("thetas(1)", 0.40)
    return curve


def _loaded_seed(surface: SurfaceRZFourier):
    """A warm-seed BiotSavart + partitions whose CWS family rides ``surface``."""
    curve = _banana_curve(surface)
    current = Current(1.1e4)
    current.fix_all()
    banana_coils = tuple(
        coils_via_symmetries(
            [curve], [current], surface.nfp, surface.stellsym
        )
    )
    biot_savart = BiotSavart(list(banana_coils))
    biot_savart.set_points(np.zeros((1, 3)))
    partitions = Stage2CoilPartitions(
        tf_coils=(),
        banana_coils=banana_coils,
        proxy_coils=(),
        vf_coils=(),
        num_tf_coils=0,
        num_banana_coils=len(banana_coils),
        num_proxy_coils=0,
        num_vf_coils=0,
        finite_current_mode="wataru_proxy_field",
    )
    return biot_savart, partitions, curve


def test_reembed_grows_objective_vector_and_moves_gamma():
    """Defect-1 guard: freed surface modes enter len(JF.x) and move gamma."""
    warm_surface = _warm_seed_circle()
    biot_savart, partitions, _ = _loaded_seed(warm_surface)

    _, _, surf_coils = module.build_hbt_reference_surfaces(
        2,
        0.142,
        module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        winding_surface_free_mpol=1,
        winding_surface_free_ntor=1,
    )

    resolved_bs, resolved_partitions, free_names = (
        module.reembed_loaded_banana_cws_family_on_surface(
            biot_savart,
            partitions,
            surf_coils,
            winding_surface_free_mpol=1,
            winding_surface_free_ntor=1,
        )
    )
    resolved_curve = resolved_partitions.banana_coils[0].curve

    # free(1,1) on a stellsym mpol>=1/ntor>=1 surface frees the low-(m,n) rc/zs
    # modes other than the pinned size dofs rc(0,0)/rc(1,0)/zs(1,0).
    assert len(free_names) == 6

    # Assemble an objective over the rebuilt curve. Its optimizer vector must
    # include the freed winding-surface shape dofs via depends_on=[surf]. The
    # apples-to-apples baseline is the SAME raised winding surface with every
    # shape dof fixed -- isolating the free-vs-fixed contribution at constant
    # resolution. (Comparing against the unraised warm circle would conflate the
    # resolution bump with the freeing.)
    baseline_surface = _warm_seed_circle()
    baseline_bs, baseline_partitions, _ = _loaded_seed(baseline_surface)
    _, _, baseline_surf = module.build_hbt_reference_surfaces(
        2,
        0.142,
        module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        winding_surface_free_mpol=1,
        winding_surface_free_ntor=1,
    )
    # Re-embed on the same raised resolution but fix every shape dof.
    _, baseline_resolved, _ = (
        module.reembed_loaded_banana_cws_family_on_surface(
            baseline_bs,
            baseline_partitions,
            baseline_surf,
            winding_surface_free_mpol=1,
            winding_surface_free_ntor=1,
        )
    )
    baseline_surf.fix_all()
    baseline_curve = baseline_resolved.banana_coils[0].curve

    base_len = len(CurveLength(baseline_curve).x)
    freed_len = len(CurveLength(resolved_curve).x)
    assert freed_len - base_len == len(free_names)

    # Perturbing a freed surface dof must move the rebuilt curve geometry.
    before = resolved_curve.gamma().copy()
    surface_dofs = surf_coils.get_dofs().copy()
    free_index = list(surf_coils.local_full_dof_names).index(free_names[0])
    surface_dofs[free_index] += 1.0e-3
    surf_coils.local_full_x = surface_dofs
    after = resolved_curve.gamma()
    assert np.max(np.abs(after - before)) > 0.0

    # bs identity changed (rebuilt) and coil count preserved.
    assert resolved_bs is not biot_savart
    assert len(resolved_bs.coils) == len(biot_savart.coils)


def test_warm_seed_circle_carries_no_shape_modes_requires_prebuilt_surface():
    """Defect-2 documentation: shape modes must exist on the surface at build.

    The warm-seed embedded surface is an mpol=1/ntor=0 circle whose only dofs
    are the pinned size modes; it has no shape modes to free. Therefore shape
    optimization is only possible by building the coil on a surface that
    ALREADY carries the target resolution (build raised first, then construct),
    which is exactly what ``build_hbt_reference_surfaces`` does.
    """
    warm_surface = _warm_seed_circle()
    assert list(warm_surface.local_full_dof_names) == [
        "rc(0,0)",
        "rc(1,0)",
        "zs(1,0)",
    ]

    # No shape modes exist on the warm circle: freeing (1,1) must raise.
    with pytest.raises(ValueError):
        configure_winding_surface_shape_dofs(
            warm_surface, free_mpol=1, free_ntor=1
        )

    # The raised reference surface DOES carry those modes, so the rebuild path
    # can free them. This is the resolution that must be in place BEFORE the
    # CurveCWSFourierCPP forward model is constructed.
    _, _, surf_coils = module.build_hbt_reference_surfaces(
        2,
        0.142,
        module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        winding_surface_free_mpol=1,
        winding_surface_free_ntor=1,
    )
    assert int(surf_coils.mpol) >= 1
    assert int(surf_coils.ntor) >= 1
    free_names = configure_winding_surface_shape_dofs(
        surf_coils, free_mpol=1, free_ntor=1
    )
    assert free_names


def test_zero_free_shape_dofs_request_warns_not_silent(capsys):
    """Step-3 contract: shaping requested but unrealizable warns loudly.

    free_mpol=1, free_ntor=0 has only the pinned size modes in window, so it
    resolves to zero free shape dofs. The re-embed must warn rather than
    silently no-op the shape request.
    """
    warm_surface = _warm_seed_circle()
    biot_savart, partitions, _ = _loaded_seed(warm_surface)

    _, _, surf_coils = module.build_hbt_reference_surfaces(
        2,
        0.142,
        module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        winding_surface_free_mpol=1,
        winding_surface_free_ntor=0,
    )

    resolved_bs, resolved_partitions, free_names = (
        module.reembed_loaded_banana_cws_family_on_surface(
            biot_savart,
            partitions,
            surf_coils,
            winding_surface_free_mpol=1,
            winding_surface_free_ntor=0,
        )
    )

    assert free_names == ()
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "zero free shape dofs" in captured.out
    # Re-embed still happened (the family rides the live surface) even though no
    # shape dof entered the optimizer.
    assert resolved_bs is not biot_savart
    assert resolved_partitions.banana_coils[0].curve.surf is surf_coils
