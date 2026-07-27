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
    for name in (
        "SIMSOPT_BACKEND_MODE",
        "SIMSOPT_BACKEND",
        "STAGE2_BACKEND",
        "SIMSOPT_JAX_PLATFORM",
        "SIMSOPT_JAX_BACKEND",
        "SIMSOPT_PRECISION",
        "SIMSOPT_MIXED_PRECISION",
    ):
        monkeypatch.delenv(name, raising=False)
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


@pytest.mark.parametrize(
    ("backend", "platform", "expected_mode"),
    (
        ("cpu", "cpu", "native_cpu"),
        ("jax", "cpu", "jax_cpu_fast"),
        ("jax", "cuda", "jax_gpu_fast"),
    ),
)
def test_legacy_backend_selection_defaults_explicit_jax_to_fast(
    backend: str,
    platform: str,
    expected_mode: str,
) -> None:
    assert runtime._mode_from_legacy_env(backend, platform) == expected_mode


def test_fully_unset_selection_remains_native_cpu() -> None:
    assert runtime.get_backend_config().mode == "native_cpu"


def test_explicit_full_mode_wins_over_legacy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_BACKEND", "jax")
    monkeypatch.setenv("SIMSOPT_JAX_PLATFORM", "cuda")

    assert runtime.get_backend_config().mode == "jax_cpu_parity"


@pytest.mark.parametrize(
    ("device", "intent", "expected_mode"),
    (
        ("cpu", None, "jax_cpu_fast"),
        ("gpu", None, "jax_gpu_fast"),
        ("cpu", "fast", "jax_cpu_fast"),
        ("gpu", "fast", "jax_gpu_fast"),
        ("cpu", "parity", "jax_cpu_parity"),
        ("gpu", "parity", "jax_gpu_parity"),
    ),
)
def test_typed_jax_selector_resolves_device_and_intent(
    device: str,
    intent: str | None,
    expected_mode: str,
) -> None:
    keywords: dict[str, object] = {
        "device": device,
        "configure_runtime": False,
    }
    if intent is not None:
        keywords["intent"] = intent

    config = runtime.set_backend("jax", **keywords)

    assert config.mode == expected_mode


def test_use_runtime_forwards_typed_jax_selector() -> None:
    config = runtime.use_runtime(
        "jax",
        device="gpu",
        intent="parity",
        configure_runtime=False,
    )

    assert config.mode == "jax_gpu_parity"


def test_public_config_reexports_typed_use_runtime() -> None:
    import simsopt_jax.config as simsopt_config

    config = simsopt_config.use_runtime(
        "jax",
        device="cpu",
        configure_runtime=False,
    )

    assert config.mode == "jax_cpu_fast"


@pytest.mark.parametrize(
    ("mode", "keywords", "message"),
    (
        ("jax", {}, "device"),
        ("jax", {"device": "tpu"}, "device"),
        ("jax", {"device": "cpu", "intent": "debug"}, "intent"),
        ("jax_cpu_fast", {"device": "cpu"}, "cannot be combined"),
        ("jax_gpu_parity", {"intent": "parity"}, "cannot be combined"),
    ),
)
def test_typed_jax_selector_rejects_ambiguous_or_invalid_calls(
    mode: str,
    keywords: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        runtime.set_backend(mode, configure_runtime=False, **keywords)
