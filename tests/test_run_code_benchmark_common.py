import pytest

from benchmarks import run_code_benchmark_common as benchmark_common


@pytest.fixture(autouse=True)
def _force_x64(monkeypatch):
    monkeypatch.setattr(benchmark_common, "_x64_enabled", lambda: True)


def test_explicit_scipy_backend_allowed_on_private_optimizer_lane(monkeypatch):
    monkeypatch.setattr(
        benchmark_common,
        "_current_jax_version",
        lambda: benchmark_common.PRIVATE_OPTIMIZER_JAX_VERSION,
    )

    assert benchmark_common.resolve_benchmark_backends(["scipy"]) == ("scipy",)


def test_private_backend_rejected_on_public_lane(monkeypatch):
    monkeypatch.setattr(
        benchmark_common,
        "_current_jax_version",
        lambda: benchmark_common.PUBLIC_EXPECTED_JAX_VERSION,
    )

    with pytest.raises(RuntimeError, match="require the private JAX"):
        benchmark_common.resolve_benchmark_backends(["ondevice"])
