"""Public compatibility contract for typed precision selection."""

from __future__ import annotations

import os

import pytest
from simsopt_jax.backend import runtime
from simsopt_jax.backend.runtime import BackendConfig

_MODE_DEFAULTS = {
    "native_cpu": ("fp64", "float64", "float64", None, "highest"),
    "jax_cpu_fast": ("fp64", "float64", "float64", None, "default"),
    "jax_cpu_parity": ("fp64", "float64", "float64", None, "highest"),
    "jax_cpu_float32_smoke": (
        "fp32_smoke",
        "float32",
        "float32",
        None,
        "default",
    ),
    "jax_gpu_fast": ("fp64", "float64", "float64", None, "default"),
    "jax_gpu_parity": ("fp64", "float64", "float64", None, "highest"),
}


def _policy(mode: str, *, precision=None):
    config = runtime._config_from_mode(
        mode,
        strict=False,
        precision=precision,
    )
    return runtime._policy_from_config(config)


@pytest.fixture(autouse=True)
def _clear_precision_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SIMSOPT_PRECISION", raising=False)
    monkeypatch.delenv("SIMSOPT_MIXED_PRECISION", raising=False)
    runtime.invalidate_backend_cache()
    yield
    runtime.invalidate_backend_cache()


@pytest.mark.parametrize(
    ("mode", "expected"),
    _MODE_DEFAULTS.items(),
)
def test_omitted_precision_preserves_mode_defaults(mode: str, expected: tuple):
    policy = _policy(mode)

    assert policy.precision == "mode_default"
    assert (
        policy.resolved_precision,
        policy.runtime_dtype,
        policy.compute_dtype,
        policy.certificate_dtype,
        policy.matmul_precision,
    ) == expected


@pytest.mark.parametrize(
    "mode", ("jax_cpu_fast", "jax_cpu_parity", "jax_gpu_fast", "jax_gpu_parity")
)
def test_mixed_precision_preserves_fp64_results_and_certificates(mode: str):
    policy = _policy(mode, precision="mixed")

    assert policy.precision == "mixed"
    assert policy.resolved_precision == "mixed"
    assert policy.runtime_dtype == "float64"
    assert policy.host_dtype == "float64"
    assert policy.compute_dtype == "float32"
    assert policy.certificate_dtype == "float64"
    assert policy.matmul_precision == "highest"


def test_explicit_fp64_preserves_the_selected_mode_matmul_contract():
    policy = _policy("jax_gpu_fast", precision="fp64")

    assert policy.precision == "fp64"
    assert policy.resolved_precision == "fp64"
    assert policy.compute_dtype == "float64"
    assert policy.certificate_dtype is None
    assert policy.matmul_precision == "default"


@pytest.mark.parametrize(
    ("mode", "precision", "message"),
    (
        ("native_cpu", "mixed", "native_cpu does not support mixed precision"),
        (
            "jax_cpu_float32_smoke",
            "mixed",
            "only supports precision='mode_default'",
        ),
        (
            "jax_cpu_float32_smoke",
            "fp64",
            "only supports precision='mode_default'",
        ),
    ),
)
def test_unsupported_mode_precision_pairs_fail_loudly(
    mode: str,
    precision: str,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        runtime.set_backend(mode, precision=precision, configure_runtime=False)


def test_public_precision_precedence_and_normalized_environment(monkeypatch):
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")

    inherited = runtime.set_backend("jax_cpu_parity", configure_runtime=False)
    assert inherited.precision == "fp64"
    assert os.environ["SIMSOPT_PRECISION"] == "fp64"

    explicit = runtime.set_backend(
        "jax_cpu_parity",
        precision="mixed",
        configure_runtime=False,
    )
    assert explicit.precision == "mixed"
    assert os.environ["SIMSOPT_PRECISION"] == "mixed"

    cleared = runtime.set_backend(
        "jax_cpu_float32_smoke",
        precision="mode_default",
        configure_runtime=False,
    )
    assert cleared.precision == "mode_default"
    assert os.environ["SIMSOPT_PRECISION"] == "mode_default"
    assert runtime.get_resolved_precision() == "fp32_smoke"


@pytest.mark.parametrize("invalid", ("", "fp32", "true", "MIXED", "auto"))
def test_invalid_explicit_precision_fails_before_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
):
    def _unexpected_runtime_initialization():
        raise AssertionError("invalid precision reached JAX runtime configuration")

    monkeypatch.setattr(
        runtime,
        "apply_jax_runtime_config",
        _unexpected_runtime_initialization,
    )

    with pytest.raises(ValueError, match="Accepted"):
        runtime.set_backend("jax_cpu_parity", precision=invalid)


def test_invalid_environment_precision_fails_before_runtime_configuration(monkeypatch):
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp32")

    with pytest.raises(ValueError, match="SIMSOPT_PRECISION='fp32'.*Accepted"):
        runtime.set_backend("jax_cpu_parity", configure_runtime=False)


def test_obsolete_mixed_precision_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("SIMSOPT_MIXED_PRECISION", "1")

    with pytest.raises(ValueError, match="use SIMSOPT_PRECISION=mixed"):
        runtime.set_backend("jax_cpu_parity", configure_runtime=False)


def test_backend_config_defaulted_precision_is_constructor_compatible():
    config = BackendConfig(
        mode="jax_cpu_parity",
        backend="jax",
        jax_platform="cpu",
    )

    assert config.precision == "mode_default"


def test_use_runtime_threads_the_typed_precision_selection():
    config = runtime.use_runtime(
        "jax_cpu_parity",
        precision="mixed",
        configure_runtime=False,
    )

    assert config.precision == "mixed"
    assert runtime.get_resolved_precision() == "mixed"
