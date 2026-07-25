"""Strict-transfer coverage for device-resident traceable quadrature state."""

from __future__ import annotations

import types

import jax
import numpy as np
import pytest

from simsopt_jax_adapters.geo import surface_objectives_traceable as _traceable


def test_traceable_cache_state_accepts_device_quadrature_under_transfer_guard():
    gpu_devices = [device for device in jax.devices() if device.platform == "gpu"]
    if not gpu_devices:
        pytest.skip("CUDA device required for device-to-host transfer guard coverage")
    device = gpu_devices[0]

    def device_array(values):
        return jax.device_put(np.asarray(values, dtype=np.float64), device=device)

    full_phi = [0.0, 1.0 / 3.0, 2.0 / 3.0]
    full_theta = [0.0, 1.0 / 3.0, 2.0 / 3.0]

    class _FakeSurface:
        quadpoints_phi = device_array(full_phi)
        quadpoints_theta = device_array(full_theta)

    class _FakeBooz:
        boozer_type = "ls"
        surface = _FakeSurface()
        res = {
            "success": True,
            "primal_success": True,
            "linearization_kind": "operator-gmres",
        }
        quadpoints_phi = device_array(full_phi)
        quadpoints_theta = device_array(full_theta)
        mpol = 1
        ntor = 1
        nfp = 1
        stellsym = True
        scatter_indices = jax.device_put(
            np.asarray([0, 1], dtype=np.int32),
            device=device,
        )
        _surface_geometry_kind = "surface-geometry-marker"
        label_quadpoints_phi = device_array([0.0, 0.25])
        label_quadpoints_theta = device_array([0.0, 0.5])
        label_mpol = 1
        label_ntor = 1
        label_nfp = 1
        label_stellsym = True
        label_scatter_indices = jax.device_put(
            np.asarray([0, 1], dtype=np.int32),
            device=device,
        )
        _label_surface_geometry_kind = "label-geometry-marker"
        options = {"weight_inv_modB": False, "newton_stab": 0.0}
        constraint_weight = 1.0
        targetlabel = 0.0
        label_type = "iota"
        phi_idx = 0
        need_to_run_code = False
        _traceable_solve_state_token = "solve-token"

        def _resolve_optimizer_method(self):
            return "lm-minpack-ondevice"

        def _collect_optimizer_options(self, *, method):
            del method
            return {}

        def _linear_solve_tolerance(self):
            return 1.0e-10

        def get_solved_runtime_state(self):
            return types.SimpleNamespace(
                sdofs=device_array([1.0, 0.1]),
                iota=device_array(0.23),
                G=device_array(1.7),
                weight_inv_modB=False,
            )

    class _FakeBS:
        x = device_array([0.2, -0.1])
        _coil_dof_state_token = "coil-dof-token"

        def coil_dof_extraction_spec(self):
            return {
                "gamma": device_array([[1.0, 2.0, 3.0]]),
            }

    booz_jax = _FakeBooz()
    with jax.transfer_guard("disallow"):
        state = _traceable._build_traceable_objective_cache_state(
            booz_jax,
            _FakeBS(),
            device_array(0.28),
        )
        cache_key = _traceable._traceable_runtime_cache_key(
            booz_jax,
            state,
        )

    assert state["objective_kwargs"]["quadpoints_phi"].shape == (3,)
    assert cache_key.solve_state_token == "solve-token"
