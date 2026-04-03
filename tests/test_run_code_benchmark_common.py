import pytest

from benchmarks import run_code_benchmark_common as benchmark_common


@pytest.fixture(autouse=True)
def _force_x64(monkeypatch):
    monkeypatch.setattr(benchmark_common, "_x64_enabled", lambda: True)


@pytest.fixture
def _supported_runtime(monkeypatch):
    monkeypatch.setattr(
        benchmark_common,
        "_current_jax_version",
        lambda: benchmark_common.EXPECTED_BENCHMARK_JAX_VERSION,
    )


def test_explicit_scipy_backend_allowed_on_supported_runtime(_supported_runtime):
    assert benchmark_common.resolve_benchmark_backends(["scipy"]) == ("scipy",)


def test_explicit_private_backend_allowed_on_supported_runtime(_supported_runtime):
    assert benchmark_common.resolve_benchmark_backends(["ondevice"]) == ("ondevice",)


def test_default_benchmark_backend_uses_target_lane(_supported_runtime):
    assert benchmark_common.resolve_benchmark_backends() == ("ondevice",)


def test_private_backend_rejected_on_wrong_runtime(monkeypatch):
    monkeypatch.setattr(
        benchmark_common,
        "_current_jax_version",
        lambda: "0.9.1",
    )

    with pytest.raises(RuntimeError, match="configured for JAX"):
        benchmark_common.resolve_benchmark_backends(["ondevice"])
