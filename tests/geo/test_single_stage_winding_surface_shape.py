import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
SINGLE_STAGE_ROOT = EXAMPLES_ROOT / "SINGLE_STAGE"
for path in (str(EXAMPLES_ROOT), str(SINGLE_STAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import single_stage_banana_example as module  # noqa: E402
from banana_opt.stage2_single_stage_handoff import Stage2CoilPartitions  # noqa: E402
from simsopt.field import BiotSavart  # noqa: E402
from simsopt.field.coil import Current, coils_via_symmetries  # noqa: E402
from simsopt.geo import CurveCWSFourierCPP, SurfaceRZFourier  # noqa: E402


def _seed_surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier(nfp=2, stellsym=True, mpol=1, ntor=0)
    surface.set_rc(0, 0, 0.976)
    surface.set_rc(1, 0, 0.21)
    surface.set_zs(1, 0, 0.21)
    return surface


def _banana_curve(surface: SurfaceRZFourier) -> CurveCWSFourierCPP:
    curve = CurveCWSFourierCPP(
        np.linspace(0.0, 1.0, 32, endpoint=False),
        order=2,
        surf=surface,
        G=3,
        H=5,
    )
    curve.local_full_x = curve.get_dofs() + np.array(
        [0.12, -0.03, 0.04, 0.02, -0.01, 0.37, 0.05, -0.02, 0.03, -0.04]
    )
    curve.fix("phic(1)")
    return curve


def _partitions(banana_coils):
    return Stage2CoilPartitions(
        tf_coils=(),
        banana_coils=tuple(banana_coils),
        proxy_coils=(),
        vf_coils=(),
        num_tf_coils=0,
        num_banana_coils=len(banana_coils),
        num_proxy_coils=0,
        num_vf_coils=0,
        finite_current_mode="wataru_proxy_field",
    )


def _loaded_seed():
    seed_surface = _seed_surface()
    curve = _banana_curve(seed_surface)
    current = Current(1.1e4)
    current.fix_all()
    banana_coils = tuple(
        coils_via_symmetries(
            [curve],
            [current],
            seed_surface.nfp,
            seed_surface.stellsym,
        )
    )
    biot_savart = BiotSavart(list(banana_coils))
    biot_savart.set_points(np.zeros((1, 3)))
    return biot_savart, _partitions(banana_coils), curve


def test_single_stage_reference_surface_resolution_tracks_free_shape_flags():
    _, _, surf_coils = module.build_hbt_reference_surfaces(
        2,
        0.142,
        module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        winding_surface_free_mpol=1,
        winding_surface_free_ntor=1,
    )

    assert int(surf_coils.mpol) == 1
    assert int(surf_coils.ntor) == 1
    free_names = module.configure_winding_surface_shape_dofs(
        surf_coils,
        free_mpol=1,
        free_ntor=1,
    )
    assert free_names
    assert "rc(0,0)" not in free_names
    assert "rc(1,0)" not in free_names
    assert "zs(1,0)" not in free_names


def test_zero_free_shape_flags_preserve_loaded_seed_surface_binding():
    biot_savart, partitions, seed_curve = _loaded_seed()
    _, _, surf_coils = module.build_hbt_reference_surfaces(
        2,
        0.142,
        module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
    )

    resolved_bs, resolved_partitions, free_names = (
        module.reembed_loaded_banana_cws_family_on_surface(
            biot_savart,
            partitions,
            surf_coils,
            winding_surface_free_mpol=0,
            winding_surface_free_ntor=0,
        )
    )

    assert resolved_bs is biot_savart
    assert resolved_partitions is partitions
    assert free_names == ()
    assert resolved_partitions.banana_coils[0].curve.surf is seed_curve.surf


def test_reembed_request_does_not_require_a_free_shape_mode():
    biot_savart, partitions, seed_curve = _loaded_seed()
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

    assert resolved_bs is not biot_savart
    assert free_names == ()
    assert resolved_partitions.banana_coils[0].curve.surf is surf_coils
    assert partitions.banana_coils[0].curve.surf is seed_curve.surf


def test_free_shape_flags_reembed_warm_cws_family_on_live_surface():
    biot_savart, partitions, seed_curve = _loaded_seed()
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

    assert resolved_bs is not biot_savart
    assert len(resolved_bs.coils) == len(biot_savart.coils)
    assert resolved_partitions.banana_coils[0] is not partitions.banana_coils[0]
    assert resolved_curve.surf is surf_coils
    assert partitions.banana_coils[0].curve.surf is seed_curve.surf
    assert free_names
    assert surf_coils in resolved_curve.parents
    assert resolved_curve.full_dof_size == (
        resolved_curve.local_full_dof_size + surf_coils.local_full_dof_size
    )

    before = resolved_curve.gamma().copy()
    surface_dofs = surf_coils.get_dofs().copy()
    free_index = list(surf_coils.local_full_dof_names).index(free_names[0])
    surface_dofs[free_index] += 1.0e-3
    surf_coils.local_full_x = surface_dofs
    after = resolved_curve.gamma()

    assert np.max(np.abs(after - before)) > 0.0
