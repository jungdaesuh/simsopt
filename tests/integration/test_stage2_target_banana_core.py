from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.single_stage_optimization.banana_opt.jax_banana_specs import (
    build_banana_local_objective_spec_from_coils,
)
from examples.single_stage_optimization.banana_opt.jax_banana_types import (
    Stage2Weights,
)
from simsopt_jax.core import make_coil_symmetry_spec, make_curve_xyzfourier_spec
from simsopt_jax.core.banana import (
    banana_geometry_from_dofs,
    banana_stage2_terms,
    banana_stage2_value_and_grad,
    make_banana_system_spec,
)
from simsopt_jax.core.curve_geometry import curve_geometry_from_dofs
from simsopt_jax.core.specs import CurveCWSFourierRZSpec
from simsopt_jax_adapters.objectives import stage2_target


jax.config.update("jax_enable_x64", True)


class _HostCurrent:
    def __init__(self, value):
        self.local_full_x = np.asarray([value], dtype=np.float64)

    def get_value(self):
        return float(self.local_full_x[0])


class _SpecBackedCurrentShape:
    def __init__(self, value):
        self._value = float(value)

    def get_value(self):
        return self._value


class _ScaledCurrent:
    def __init__(self, current, scale):
        self.current_to_scale = current
        self.scale = scale


class _RotatedCurve:
    def __init__(self, curve, rotmat):
        self.curve = curve
        self.rotmat = np.asarray(rotmat, dtype=np.float64)


class _HostCoil:
    def __init__(self, curve, current):
        self.curve = curve
        self.current = current


def _rotated_coil(owner_curve, owner_current, *, scale=1.0):
    return _HostCoil(
        _RotatedCurve(owner_curve, np.eye(3, dtype=np.float64)),
        _ScaledCurrent(owner_current, scale),
    )


def test_stage2_dynamic_curve_data_matches_banana_geometry_from_dofs():
    curve_spec = make_curve_xyzfourier_spec(
        dofs=np.array(
            [0.0, 0.7, 0.1, 0.0, -0.2, 0.9, 0.1, 0.2, -0.1],
            dtype=np.float64,
        ),
        quadpoints=np.linspace(0.0, 1.0, 8, endpoint=False, dtype=np.float64),
        order=1,
    )
    rotmats = jnp.asarray(
        [
            np.eye(3, dtype=np.float64),
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=jnp.float64,
    )
    current_scales = jnp.asarray([2.0, -0.5], dtype=jnp.float64)
    decision_vector = jnp.concatenate(
        (
            jnp.asarray([4.0], dtype=jnp.float64),
            jnp.asarray(curve_spec.dofs, dtype=jnp.float64),
        )
    )
    state = stage2_target.stage2_target_optimizer_state_from_dofs(
        decision_vector,
        curve_dof_count=int(curve_spec.dofs.shape[0]),
    )
    base_gamma, base_gammadash, base_gammadashdash = curve_geometry_from_dofs(
        curve_spec,
        state.curve_dofs,
    )
    target_gammas, target_gammadashs, target_currents = (
        stage2_target._build_dynamic_curve_data(
            base_gamma,
            base_gammadash,
            rotmats,
            current_scales,
            state.current_dof,
        )
    )

    banana_system = make_banana_system_spec(
        curve=curve_spec,
        symmetries=(
            make_coil_symmetry_spec(scale=2.0),
            make_coil_symmetry_spec(
                rotmat=np.asarray(rotmats[1], dtype=np.float64),
                scale=-0.5,
            ),
        ),
        current_dof_count=1,
    )
    banana_geometry = banana_geometry_from_dofs(banana_system, decision_vector)

    np.testing.assert_allclose(banana_geometry.base_gamma, base_gamma)
    np.testing.assert_allclose(banana_geometry.base_gammadash, base_gammadash)
    np.testing.assert_allclose(
        banana_geometry.base_gammadashdash,
        base_gammadashdash,
    )
    np.testing.assert_allclose(banana_geometry.base_current_dof, state.current_dof)
    np.testing.assert_allclose(banana_geometry.gammas, target_gammas)
    np.testing.assert_allclose(banana_geometry.gammadashs, target_gammadashs)
    np.testing.assert_allclose(banana_geometry.currents, target_currents)


def test_stage2_target_bundle_value_and_grad_matches_banana_overlap_terms():
    pytest.importorskip(
        "simsoptpp",
        reason="Stage 2 public-bundle parity requires SIMSOPT host curves.",
    )
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.field.coil import ScaledCurrent
    from simsopt.geo import SurfaceRZFourier
    from simsopt_jax_adapters.geo.curvecwsfourier import CurveCWSFourierCPP

    surface = SurfaceRZFourier(
        nfp=2,
        stellsym=False,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.arange(8) / 8,
        quadpoints_theta=np.arange(8) / 8,
    )
    surface.set_rc(0, 0, 0.9)
    surface.set_rc(1, 0, 0.12)
    surface.set_zs(1, 0, 0.12)
    surface.fix_all()

    coil_surf = SurfaceRZFourier(
        nfp=2,
        stellsym=False,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.arange(8) / 8,
        quadpoints_theta=np.arange(8) / 8,
    )
    coil_surf.set_rc(0, 0, 1.0)
    coil_surf.set_rc(1, 0, 0.2)
    coil_surf.set_zs(1, 0, 0.2)

    banana_curve = CurveCWSFourierCPP(
        np.linspace(0.0, 1.0, 16, endpoint=False),
        order=2,
        surf=coil_surf,
    )
    banana_curve.set("phic(0)", 0.05)
    banana_curve.set("thetac(0)", 0.4)
    banana_curve.set("phic(1)", 0.02)
    banana_curve.set("thetas(1)", 0.08)
    banana_current = ScaledCurrent(Current(1.0), 1.0)
    banana_coils = tuple(
        coils_via_symmetries(
            [banana_curve],
            [banana_current],
            surface.nfp,
            surface.stellsym,
        )
    )

    penalty_config = stage2_target.Stage2PenaltyConfig(
        squared_flux_weight=0.0,
        length_weight=0.7,
        length_target=0.1,
        cc_weight=2.0,
        cc_threshold=2.0,
        curvature_weight=0.03,
        curvature_threshold=0.1,
        curvature_p_norm=4,
    )
    target_bundle = stage2_target.build_stage2_target_objective(
        surface=surface,
        tf_coils=(),
        banana_coils=banana_coils,
        banana_curve=banana_curve,
        penalty_config=penalty_config,
    )

    local_weights = replace(
        Stage2Weights(),
        length=penalty_config.length_weight,
        curvature=penalty_config.curvature_weight,
        ccdist=penalty_config.cc_weight,
        poloidal=0.0,
        width=0.0,
        selfint=0.0,
        currents=0.0,
    )
    banana_spec, decision_vector = build_banana_local_objective_spec_from_coils(
        banana_coils,
        stage="stage2",
        weights=local_weights,
        include_current_penalty=False,
        include_min_length=False,
        include_width=False,
    )
    assert isinstance(banana_spec.system.curve, CurveCWSFourierRZSpec)
    assert len(banana_spec.system.symmetries) == len(banana_coils)
    banana_spec = replace(
        banana_spec,
        length_max_threshold=penalty_config.length_target,
        curvature_threshold=penalty_config.curvature_threshold,
        curvature_p=int(penalty_config.curvature_p_norm),
        coil_coil_distance_minimum=penalty_config.cc_threshold,
    )
    decision_vector_jax = jnp.asarray(decision_vector, dtype=jnp.float64)

    target_value, target_grad = target_bundle.value_and_grad(decision_vector_jax)
    banana_value, banana_grad = banana_stage2_value_and_grad(
        banana_spec,
        decision_vector_jax,
    )
    target_terms = target_bundle.raw_terms(decision_vector_jax)
    banana_terms = banana_stage2_terms(banana_spec, decision_vector_jax)

    assert decision_vector_jax.shape[0] == target_bundle.expected_dof_count
    assert target_terms[2] > 0.0
    np.testing.assert_allclose(
        banana_terms.length_max,
        target_terms[1],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        banana_terms.coil_coil_distance,
        target_terms[2],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        banana_terms.curvature,
        target_terms[3],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(banana_value, target_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(banana_grad, target_grad, rtol=1e-10, atol=1e-10)


def test_stage2_dynamic_curve_data_matches_core_banana_symmetry_helper():
    base_gamma = jnp.asarray(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=jnp.float64,
    )
    base_gammadash = jnp.asarray(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        dtype=jnp.float64,
    )
    rotmats = jnp.asarray(
        [
            np.eye(3, dtype=np.float64),
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=jnp.float64,
    )
    current_scales = jnp.asarray([2.0, -0.5], dtype=jnp.float64)
    current_dof = jnp.asarray(4.0, dtype=jnp.float64)

    actual = stage2_target._build_dynamic_curve_data(
        base_gamma,
        base_gammadash,
        rotmats,
        current_scales,
        current_dof,
    )
    expected = (
        jnp.stack((base_gamma, base_gamma @ rotmats[1]), axis=0),
        jnp.stack((base_gammadash, base_gammadash @ rotmats[1]), axis=0),
        current_dof * current_scales,
    )

    for actual_leaf, expected_leaf in zip(actual, expected):
        np.testing.assert_allclose(actual_leaf, expected_leaf)


def test_stage2_banana_owner_contract_accepts_shared_owner():
    owner_curve = object()
    owner_current = _HostCurrent(2.0)
    banana_coils = [
        _rotated_coil(owner_curve, owner_current, scale=2.0),
        _rotated_coil(owner_curve, owner_current, scale=-2.0),
    ]

    stage2_target._validate_stage2_banana_owner_contract(
        banana_coils,
        owner_curve,
    )


def test_stage2_banana_owner_contract_accepts_spec_backed_current_shape():
    owner_curve = object()
    owner_current = _SpecBackedCurrentShape(2.0)
    banana_coils = [
        _rotated_coil(owner_curve, owner_current, scale=2.0),
        _rotated_coil(owner_curve, owner_current, scale=-2.0),
    ]

    stage2_target._validate_stage2_banana_owner_contract(
        banana_coils,
        owner_curve,
    )


def test_stage2_banana_owner_contract_rejects_non_owner_curve():
    owner_curve = object()
    owner_current = _HostCurrent(2.0)
    banana_coils = [_rotated_coil(owner_curve, owner_current)]

    with pytest.raises(ValueError, match="banana_curve must be the shared owner"):
        stage2_target._validate_stage2_banana_owner_contract(
            banana_coils,
            object(),
        )


def test_stage2_banana_owner_contract_rejects_non_shared_current():
    owner_curve = object()
    banana_coils = [
        _rotated_coil(owner_curve, _HostCurrent(2.0)),
        _rotated_coil(owner_curve, _HostCurrent(2.0)),
    ]

    with pytest.raises(ValueError, match="share one current"):
        stage2_target._validate_stage2_banana_owner_contract(
            banana_coils,
            owner_curve,
        )


def test_stage2_banana_owner_contract_rejects_non_shared_curve():
    owner_curve = object()
    owner_current = _HostCurrent(2.0)
    banana_coils = [
        _rotated_coil(owner_curve, owner_current),
        _rotated_coil(object(), owner_current),
    ]

    with pytest.raises(ValueError, match="share one curve"):
        stage2_target._validate_stage2_banana_owner_contract(
            banana_coils,
            owner_curve,
        )
