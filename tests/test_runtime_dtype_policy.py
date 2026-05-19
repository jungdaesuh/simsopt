from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _restore_backend_config(config) -> None:
    from simsopt.backend import set_backend

    set_backend(
        config.mode,
        strict=config.strict,
        debug_nans=config.debug_nans,
        disable_jit=config.disable_jit,
        transfer_guard=config.transfer_guard,
        compilation_cache_dir=config.compilation_cache_dir,
        xla_gpu_preallocate=config.xla_gpu_preallocate,
        xla_gpu_mem_fraction=config.xla_gpu_mem_fraction,
        xla_gpu_allocator=config.xla_gpu_allocator,
        tf_gpu_allocator=config.tf_gpu_allocator,
        configure_runtime=False,
    )


@contextmanager
def _temporary_backend(mode: str):
    from simsopt.backend import get_backend_config, set_backend

    previous = get_backend_config()
    try:
        set_backend(mode, configure_runtime=False)
        yield
    finally:
        _restore_backend_config(previous)


def test_jax_mps_smoke_policy_runtime_dtype():
    from simsopt.backend import get_backend_policy

    with _temporary_backend("jax_mps_smoke"):
        policy = get_backend_policy()

        assert policy.runtime_dtype == "float32"
        assert policy.host_dtype == "float32"


def test_non_mps_modes_keep_float64_policy_dtype():
    from simsopt.backend import get_backend_policy

    for mode in (
        "native_cpu",
        "jax_cpu_fast",
        "jax_cpu_parity",
        "jax_gpu_fast",
        "jax_gpu_parity",
    ):
        policy = get_backend_policy(mode)

        assert policy.runtime_dtype == "float64"
        assert policy.host_dtype == "float64"


def test_require_runtime_dtype_follows_backend_policy():
    from simsopt.backend import set_backend
    from simsopt.jax_core._math_utils import require_runtime_dtype

    with _temporary_backend("jax_mps_smoke"):
        require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float32))
        with pytest.raises(TypeError, match="x must have runtime dtype float32"):
            require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float64))

        set_backend("jax_cpu_parity", configure_runtime=False)
        require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float64))
        with pytest.raises(TypeError, match="x must have runtime dtype float64"):
            require_runtime_dtype("x", jnp.asarray([1.0], dtype=jnp.float32))


def test_as_runtime_value_uses_policy_dtype_for_host_values():
    from simsopt.backend import set_backend
    from simsopt.jax_core._math_utils import as_runtime_value

    with _temporary_backend("jax_mps_smoke"):
        reference32 = jnp.asarray([0.0], dtype=jnp.float32)
        value32 = as_runtime_value(
            np.asarray([1.0, 2.0], dtype=np.float64),
            reference=reference32,
        )
        assert value32.dtype == jnp.float32

        set_backend("jax_cpu_parity", configure_runtime=False)
        reference64 = jnp.asarray([0.0], dtype=jnp.float64)
        value64 = as_runtime_value(
            np.asarray([1.0, 2.0], dtype=np.float32),
            reference=reference64,
        )
        assert value64.dtype == jnp.float64


def test_axis0_entries_preserves_empty_axis0_tuple():
    from simsopt.jax_core._math_utils import axis0_entries

    assert axis0_entries(jnp.zeros((0, 3))) == ()


def test_as_runtime_float64_uses_runtime_policy_dtype_for_host_values():
    from simsopt.backend.dtypes import as_runtime_float64

    with _temporary_backend("jax_mps_smoke"):
        reference32 = jnp.asarray([0.0], dtype=jnp.float32)

        value32 = as_runtime_float64(
            np.asarray([1.0, 2.0], dtype=np.float64),
            reference=reference32,
        )

        assert value32.dtype == jnp.float32


def test_as_runtime_float64_alias_does_not_gate_on_host_reference_dtype():
    from simsopt.backend.dtypes import as_runtime_float64

    with _temporary_backend("jax_mps_smoke"):
        host64 = np.asarray([1.0, 2.0], dtype=np.float64)

        value32 = as_runtime_float64(host64, reference=host64)

        assert value32.dtype == jnp.float32


def test_as_jax_float64_compat_alias_uses_runtime_policy_dtype():
    from simsopt.backend.dtypes import as_jax_float64

    with _temporary_backend("jax_mps_smoke"):
        value = as_jax_float64(np.asarray([1.0, 2.0], dtype=np.float64))

        assert value.dtype == jnp.float32


def test_runtime_device_put_uses_policy_dtype_for_float_hosts():
    from simsopt.backend.dtypes import runtime_device_put

    with _temporary_backend("jax_mps_smoke"):
        value = runtime_device_put(np.asarray([1.0, 2.0], dtype=np.float64))
        indices = runtime_device_put([0, 1], dtype=np.int32)

        assert value.dtype == jnp.float32
        assert indices.dtype == jnp.int32


def test_runtime_device_put_resolves_explicit_float_dtype_through_policy():
    from simsopt.backend.dtypes import runtime_device_put

    with _temporary_backend("jax_mps_smoke"):
        value = runtime_device_put([1.0, 2.0], dtype=np.float64)

        assert value.dtype == jnp.float32


def test_boozer_optimizer_backend_auto_uses_policy_default():
    from simsopt.backend import set_backend
    from simsopt.geo import boozersurface_jax

    with _temporary_backend("native_cpu"):
        native_options = boozersurface_jax._normalize_solver_options(
            {"optimizer_backend": "auto"},
            "ls",
        )

        set_backend("jax_cpu_fast", configure_runtime=False)
        jax_options = boozersurface_jax._normalize_solver_options(
            {"optimizer_backend": "auto"},
            "ls",
        )

        assert native_options["optimizer_backend"] == "scipy"
        assert jax_options["optimizer_backend"] == "ondevice"

        set_backend("jax_mps_smoke", configure_runtime=False)
        mps_options = boozersurface_jax._normalize_solver_options(
            {"optimizer_backend": "auto"},
            "ls",
        )

        assert mps_options["optimizer_backend"] == "scipy"


def test_boozer_ls_mps_smoke_default_avoids_target_x64_gate(monkeypatch):
    from simsopt.geo import boozersurface_jax
    from simsopt.geo import optimizer_jax as optimizer_module

    with _temporary_backend("jax_mps_smoke"):
        monkeypatch.setattr(optimizer_module, "_x64_enabled", lambda: False)

        default_options = boozersurface_jax._normalize_solver_options({}, "ls")
        auto_options = boozersurface_jax._normalize_solver_options(
            {"optimizer_backend": "auto"},
            "ls",
        )

        assert default_options["optimizer_backend"] == "scipy"
        assert auto_options["optimizer_backend"] == "scipy"
        optimizer_module.require_target_backend_x64(
            default_options["optimizer_backend"]
        )
        optimizer_module.require_target_backend_x64(auto_options["optimizer_backend"])

        with pytest.raises(RuntimeError, match="requires jax_enable_x64=True"):
            optimizer_module.require_target_backend_x64("ondevice")


def test_boozer_ls_mps_smoke_default_reaches_reference_method(monkeypatch):
    from simsopt.geo import optimizer_jax as optimizer_module
    from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX, _normalize_solver_options

    with _temporary_backend("jax_mps_smoke"):
        monkeypatch.setattr(optimizer_module, "_x64_enabled", lambda: False)
        options = _normalize_solver_options({}, "ls")
        options["limited_memory"] = False
        options["force_ondevice_limited_memory"] = False

        with pytest.warns(RuntimeWarning, match="legacy adapter seam"):
            method = BoozerSurfaceJAX._resolve_optimizer_method(
                SimpleNamespace(options=options),
                optimize_G=True,
            )

        assert method == "bfgs"


def test_boozer_linearization_residency_uses_policy_default():
    from simsopt.backend import set_backend
    from simsopt.geo import boozersurface_jax

    with _temporary_backend("native_cpu"):
        native_options = boozersurface_jax._normalize_solver_options({}, "ls")

        set_backend("jax_cpu_fast", configure_runtime=False)
        jax_options = boozersurface_jax._normalize_solver_options({}, "ls")

        assert native_options["linearization_residency"] == "host"
        assert jax_options["linearization_residency"] == "device"


def test_boozer_residual_accepts_float32_under_mps_policy():
    from simsopt.geo.boozer_residual_jax import (
        boozer_residual_scalar,
        boozer_residual_vector,
    )

    with _temporary_backend("jax_mps_smoke"):
        B = jnp.ones((2, 3, 3), dtype=jnp.float32)
        xphi = jnp.full((2, 3, 3), 2.0, dtype=jnp.float32)
        xtheta = jnp.full((2, 3, 3), -0.5, dtype=jnp.float32)

        scalar_value = boozer_residual_scalar(
            np.float32(1.25),
            np.float32(-0.2),
            B,
            xphi,
            xtheta,
            weight_inv_modB=True,
        )
        vector_value = boozer_residual_vector(
            np.float32(1.25),
            np.float32(-0.2),
            B,
            xphi,
            xtheta,
            weight_inv_modB=True,
        )

        assert scalar_value.dtype == jnp.float32
        assert vector_value.dtype == jnp.float32
        assert bool(jnp.isfinite(scalar_value))
        assert bool(jnp.all(jnp.isfinite(vector_value)))
