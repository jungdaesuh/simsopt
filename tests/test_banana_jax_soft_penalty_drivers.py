from __future__ import annotations

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("simsoptpp")

from examples.single_stage_optimization.banana_opt.jax_banana_drivers import (  # noqa: E402
    CurveSelfDistanceJAX,
    PoloidalExtentJAX,
    ProjectedEllipseWidthJAX,
    banana_base_curve,
    build_biotsavart,
    build_boozer_surface_copy,
)
from examples.single_stage_optimization.banana_opt.jax_banana_types import (  # noqa: E402
    DEFAULT_BANANA_DOFS,
    DEFAULT_PROXY_RZ,
    HBT_BANANA_WS,
)
from simsopt.geo import SurfaceXYZTensorFourier  # noqa: E402
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX  # noqa: E402


def _assert_finite_positive_gradient(objective) -> None:
    value = float(objective.J())
    gradient = np.asarray(objective.dJ(), dtype=float)
    assert np.isfinite(value)
    assert value > 0.0
    assert gradient.shape == np.asarray(objective.x).shape
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0


def test_banana_geometry_jax_objectives_have_finite_gradients():
    biotsavart = build_biotsavart(
        tf_current_ka=-80.0,
        tf_fix_current=True,
        banana_current_ka=16.0,
        banana_fix_current=True,
        banana_order=1,
        banana_dofs=dict(DEFAULT_BANANA_DOFS),
        proxy_current_ka=0.0,
        proxy_rz=DEFAULT_PROXY_RZ,
        vf_current_ka=0.0,
        vf_fix_current=True,
    )
    curve = banana_base_curve(BiotSavartJAX(biotsavart.coils))

    objectives = [
        PoloidalExtentJAX(curve, HBT_BANANA_WS.major_radius, theta_target=0.05),
        ProjectedEllipseWidthJAX(
            curve,
            HBT_BANANA_WS.major_radius,
            HBT_BANANA_WS.minor_radius,
        ),
        CurveSelfDistanceJAX(
            curve,
            minimum_distance=2.0,
            neighbor_skip=3,
        ),
    ]
    for objective in objectives:
        _assert_finite_positive_gradient(objective)


def test_build_boozer_surface_copy_regrids_tensor_surface_seed():
    source = SurfaceXYZTensorFourier(
        mpol=2,
        ntor=2,
        nfp=1,
        stellsym=False,
        quadpoints_phi=np.linspace(0.0, 1.0, 7, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 8, endpoint=False),
    )

    copied = build_boozer_surface_copy(
        source,
        mpol=2,
        ntor=2,
        nphi=8,
        ntheta=8,
    )

    assert isinstance(copied, SurfaceXYZTensorFourier)
    assert copied.mpol == 2
    assert copied.ntor == 2
    assert copied.quadpoints_phi.size == 8
    assert copied.quadpoints_theta.size == 8
    assert np.all(np.isfinite(copied.gamma()))
