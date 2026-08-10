from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax_adapters.geo.single_stage_fullspace as bootstrap_module


class _FakeCurrent:
    def __init__(self, value: float) -> None:
        self.value = value
        self.fixed = False

    def fix_all(self) -> None:
        self.fixed = True

    def get_value(self) -> float:
        return self.value


class _FakeSurface:
    def __init__(
        self,
        *,
        mpol: int,
        ntor: int,
        stellsym: bool,
        nfp: int,
        quadpoints_phi: np.ndarray,
        quadpoints_theta: np.ndarray,
    ) -> None:
        self.constructor_arguments = {
            "mpol": mpol,
            "ntor": ntor,
            "stellsym": stellsym,
            "nfp": nfp,
            "quadpoints_phi": np.array(quadpoints_phi, copy=True),
            "quadpoints_theta": np.array(quadpoints_theta, copy=True),
        }
        self.dofs = np.linspace(-1.0, 1.0, 253, dtype=np.float64)
        self.fitted = False
        self.major_radius_value = -1.0

    def fit_to_curve(
        self,
        magnetic_axis: object,
        distance: float,
        *,
        flip_theta: bool,
    ) -> None:
        del magnetic_axis
        assert distance == 0.1
        assert flip_theta is True
        self.fitted = True

    def get_dofs(self) -> np.ndarray:
        return np.array(self.dofs, copy=True)

    def major_radius(self) -> float:
        return self.major_radius_value


class _FakeVolume:
    def __init__(self, surface: _FakeSurface) -> None:
        self.surface = surface

    def J(self) -> float:
        return 0.125


class _FakeCurveLength:
    def __init__(self, curve: SimpleNamespace) -> None:
        self.curve = curve

    def J(self) -> float:
        return float(self.curve.length)


def _install_fake_construction_boundary(
    monkeypatch: pytest.MonkeyPatch,
    *,
    nfp: int = 3,
    solve_success: bool = True,
    coil_dof_count: int = 461,
    surface_dof_count: int = 253,
    mask_count: int = 254,
) -> SimpleNamespace:
    currents = tuple(_FakeCurrent(value) for value in (7.0, -2.0, 3.0))
    curves = tuple(SimpleNamespace(length=value) for value in (1.0, 2.0, 3.0))
    magnetic_axis = object()
    native_coils = object()
    native_field = SimpleNamespace(coils=native_coils)
    trace = SimpleNamespace(
        currents=currents,
        curves=curves,
        magnetic_axis=magnetic_axis,
        native_coils=native_coils,
        fields=[],
        surfaces=[],
        boozer_instances=[],
        freeze_inputs=[],
        surface_spec_calls=[],
    )

    def fake_get_data(name: str):
        assert name == "ncsx"
        return curves, currents, magnetic_axis, nfp, native_field

    class FakeField:
        def __init__(self, coils: object) -> None:
            assert coils is native_coils
            self.x = np.arange(coil_dof_count, dtype=np.float64)
            self.coil_set_spec_value = object()
            trace.fields.append(self)

        def coil_set_spec(self) -> object:
            return self.coil_set_spec_value

    class FakeBoozerSurface:
        def __init__(
            self,
            field: FakeField,
            surface: _FakeSurface,
            volume_label: _FakeVolume,
            volume_target: float,
            *,
            options: dict[str, object],
        ) -> None:
            self.field = field
            self.surface = surface
            self.volume_label = volume_label
            self.volume_target = volume_target
            self.options = options
            self.run_arguments: tuple[object, ...] | None = None
            self.install_count = 0
            trace.boozer_instances.append(self)

        def run_code_traceable(self, *arguments: object) -> dict[str, object]:
            self.run_arguments = arguments
            residual = np.zeros(254, dtype=np.float64)
            residual[:2] = (3.0, 4.0)
            return {
                "success": np.asarray(solve_success),
                "sdofs": jnp.asarray(
                    np.linspace(-2.0, 2.0, surface_dof_count, dtype=np.float64)
                ),
                "iota": jnp.asarray(-0.5, dtype=jnp.float64),
                "G": jnp.asarray(9.0, dtype=jnp.float64),
                "residual": jnp.asarray(residual),
            }

        def install_traceable_solved_runtime_state(
            self,
            result: dict[str, object],
        ) -> None:
            self.install_count += 1
            self.surface.dofs = np.asarray(result["sdofs"], dtype=np.float64)
            self.surface.major_radius_value = 1.5

    def fake_surface(*args: object, **kwargs: object) -> _FakeSurface:
        assert not args
        surface = _FakeSurface(**kwargs)
        trace.surfaces.append(surface)
        return surface

    def fake_freeze(field: FakeField) -> object:
        trace.freeze_inputs.append(field)
        return trace

    def fake_surface_spec(
        surface_dofs: np.ndarray,
        quadpoints_phi: np.ndarray,
        quadpoints_theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result = (
            np.array(surface_dofs, copy=True),
            np.array(quadpoints_phi, copy=True),
            np.array(quadpoints_theta, copy=True),
        )
        trace.surface_spec_calls.append(result)
        return result

    monkeypatch.setattr(bootstrap_module, "get_data", fake_get_data)
    monkeypatch.setattr(bootstrap_module, "BiotSavartJAX", FakeField)
    monkeypatch.setattr(bootstrap_module, "SurfaceXYZTensorFourier", fake_surface)
    monkeypatch.setattr(bootstrap_module, "Volume", _FakeVolume)
    monkeypatch.setattr(bootstrap_module, "CurveLength", _FakeCurveLength)
    monkeypatch.setattr(bootstrap_module, "BoozerSurfaceJAX", FakeBoozerSurface)
    monkeypatch.setattr(
        bootstrap_module,
        "stellsym_mask_indices_for_grid_host",
        lambda **_kwargs: np.arange(mask_count, dtype=np.int32),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "freeze_coil_dof_extraction_spec",
        fake_freeze,
    )
    monkeypatch.setattr(bootstrap_module, "_surface_spec", fake_surface_spec)
    return trace


def test_bootstrap_builds_exact_frozen_state_and_target_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _install_fake_construction_boundary(monkeypatch)

    bootstrap = bootstrap_module.build_single_stage_fullspace_bootstrap()
    state = bootstrap.problem.layout.unpack(bootstrap.z0)
    config = bootstrap.problem.config
    boozer = trace.boozer_instances[0]

    assert bootstrap.z0.shape == (716,)
    assert bootstrap.z0.dtype == jnp.float64
    assert state.coil_dofs.shape == (461,)
    assert state.surface_dofs.shape == (253,)
    assert bootstrap.problem.exact_mask_indices.shape == (254,)
    assert bootstrap_module.bootstrap_target_payload(bootstrap)["equality_count"] == 255
    assert trace.currents[0].fixed is True
    assert all(not current.fixed for current in trace.currents[1:])
    assert trace.freeze_inputs == trace.fields
    assert boozer.install_count == 1
    assert boozer.volume_target == 0.125
    assert boozer.options == {
        "newton_maxiter": 20,
        "newton_tol": 1.0e-13,
        "verbose": False,
    }
    assert boozer.run_arguments is not None
    assert boozer.run_arguments[0] is trace.fields[0].coil_set_spec_value
    assert np.asarray(boozer.run_arguments[1]).shape == (253,)
    assert float(np.asarray(boozer.run_arguments[2])) == -0.406
    expected_initial_G = 2.0 * np.pi * 36.0 * 2.0e-7
    assert float(np.asarray(boozer.run_arguments[3])) == pytest.approx(
        expected_initial_G,
        rel=1.0e-15,
        abs=0.0,
    )

    assert np.array_equal(np.asarray(state.coil_dofs), np.arange(461))
    assert np.array_equal(
        np.asarray(state.surface_dofs),
        np.linspace(-2.0, 2.0, 253, dtype=np.float64),
    )
    assert float(np.asarray(state.iota)) == -0.5
    assert float(np.asarray(state.G)) == 9.0
    assert float(np.asarray(config.volume_target)) == 0.125
    assert float(np.asarray(config.iota_target)) == -0.5
    assert float(np.asarray(config.major_radius_target)) == 1.5
    assert float(np.asarray(config.length_target)) == 6.0
    assert config.length_coil_indices == (0, 1, 2)
    assert bootstrap.initial_boozer_residual_norm == 5.0
    assert len(trace.surface_spec_calls) == 3
    assert all(call[0].shape == (253,) for call in trace.surface_spec_calls)
    assert trace.surface_spec_calls[0][1].shape == (13,)
    assert trace.surface_spec_calls[0][2].shape == (13,)
    assert trace.surface_spec_calls[1][1].shape == (13,)
    assert trace.surface_spec_calls[2][1].shape == (40,)
    assert trace.surface_spec_calls[2][2].shape == (40,)


def test_public_target_payload_pins_float64_hex_and_little_endian_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_construction_boundary(monkeypatch)
    bootstrap = bootstrap_module.build_single_stage_fullspace_bootstrap()

    assert bootstrap_module.bootstrap_target_payload(bootstrap) == {
        "schema_version": "single-stage-fullspace-bootstrap-targets-v1",
        "targets": [
            {
                "name": "volume_target",
                "value": 0.125,
                "hexadecimal": "0x1.0000000000000p-3",
                "little_endian_sha256": (
                    "1cab600f57951016c0b4bd619177c26235366a7f52e26e839e3aac1219cda82d"
                ),
            },
            {
                "name": "iota_target",
                "value": -0.5,
                "hexadecimal": "-0x1.0000000000000p-1",
                "little_endian_sha256": (
                    "e1c54f41b449d2997ce426b22b0e24103c258a4e35632dcce8da80d964140bd8"
                ),
            },
            {
                "name": "major_radius_target",
                "value": 1.5,
                "hexadecimal": "0x1.8000000000000p+0",
                "little_endian_sha256": (
                    "e163f8cb0f7067a7fc78ca859a77f849aea3214f38fb75b884e4a16be725c905"
                ),
            },
            {
                "name": "length_target",
                "value": 6.0,
                "hexadecimal": "0x1.8000000000000p+2",
                "little_endian_sha256": (
                    "3e6357a56fbae74413051d518261f4b70e5b3758172a70e7f101e996e00a9ee0"
                ),
            },
        ],
        "first_base_current": {
            "name": "first_base_current",
            "value": 7.0,
            "hexadecimal": "0x1.c000000000000p+2",
            "little_endian_sha256": (
                "f52df18731eea8d020801fe2c6b3164648d9d81256a6c37964533a25999961d3"
            ),
        },
        "initial_boozer_residual_norm": 5.0,
        "joint_dof_count": 716,
        "equality_count": 255,
    }


def test_bootstrap_problem_and_initial_state_are_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_construction_boundary(monkeypatch)
    bootstrap = bootstrap_module.build_single_stage_fullspace_bootstrap()
    original_z0 = np.asarray(bootstrap.z0).copy()
    original_mask = np.asarray(bootstrap.problem.exact_mask_indices).copy()

    with pytest.raises(FrozenInstanceError):
        bootstrap.problem.layout = object()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        bootstrap.problem.config.iota_target = jnp.asarray(0.0)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        bootstrap.z0 = jnp.zeros_like(bootstrap.z0)  # type: ignore[misc]

    changed_z0 = bootstrap.z0.at[0].set(-999.0)
    changed_mask = bootstrap.problem.exact_mask_indices.at[0].set(999)
    assert np.array_equal(np.asarray(bootstrap.z0), original_z0)
    assert np.array_equal(
        np.asarray(bootstrap.problem.exact_mask_indices), original_mask
    )
    assert float(np.asarray(changed_z0[0])) == -999.0
    assert int(np.asarray(changed_mask[0])) == 999


def test_bootstrap_rejects_wrong_configuration_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _install_fake_construction_boundary(monkeypatch, nfp=5)

    with pytest.raises(ValueError, match="nfp=3"):
        bootstrap_module.build_single_stage_fullspace_bootstrap()

    assert all(not current.fixed for current in trace.currents)
    assert trace.fields == []


def test_bootstrap_fails_closed_before_installing_unsuccessful_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = _install_fake_construction_boundary(monkeypatch, solve_success=False)

    with pytest.raises(RuntimeError, match="exact Boozer bootstrap failed"):
        bootstrap_module.build_single_stage_fullspace_bootstrap()

    assert trace.boozer_instances[0].install_count == 0
    assert trace.freeze_inputs == []


@pytest.mark.parametrize(
    ("coil_dof_count", "surface_dof_count"),
    ((460, 253), (461, 252)),
)
def test_bootstrap_rejects_noncanonical_dof_counts(
    monkeypatch: pytest.MonkeyPatch,
    coil_dof_count: int,
    surface_dof_count: int,
) -> None:
    _install_fake_construction_boundary(
        monkeypatch,
        coil_dof_count=coil_dof_count,
        surface_dof_count=surface_dof_count,
    )

    with pytest.raises(ValueError, match="DOF counts"):
        bootstrap_module.build_single_stage_fullspace_bootstrap()


def test_bootstrap_rejects_noncanonical_exact_mask_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_construction_boundary(monkeypatch, mask_count=253)

    with pytest.raises(ValueError, match="mask must have 254 components"):
        bootstrap_module.build_single_stage_fullspace_bootstrap()
