"""Independent Hessian reverse-tile configuration regression tests."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401
from simsopt_jax.backend.runtime import get_field_kernel_tuning


def test_reverse_point_tuning_is_independent_of_forward_override(monkeypatch):
    """Overriding forward point tiles leaves the Hessian reverse tile unchanged."""
    monkeypatch.setenv("SIMSOPT_JAX_POINT_CHUNK_SIZE", "4096")
    tuning = get_field_kernel_tuning("jax_gpu_parity")
    assert tuning.point_chunk_size == 4096
    assert tuning.hessian_vjp_point_chunk_size == 512


def test_reverse_point_tuning_preserves_dense_override(monkeypatch):
    """A zero Hessian reverse tile selects dense evaluation independently."""
    monkeypatch.setenv("SIMSOPT_JAX_HESSIAN_VJP_POINT_CHUNK_SIZE", "0")
    tuning = get_field_kernel_tuning("jax_gpu_parity")
    assert tuning.point_chunk_size == 256
    assert tuning.hessian_vjp_point_chunk_size == 0


def test_dense_audit_zeroes_reverse_tile_despite_environment_override(monkeypatch):
    """The disallow transfer guard disables even an explicitly set reverse tile."""
    monkeypatch.setenv("SIMSOPT_JAX_HESSIAN_VJP_POINT_CHUNK_SIZE", "128")
    monkeypatch.setenv("SIMSOPT_JAX_TRANSFER_GUARD", "disallow")
    tuning = get_field_kernel_tuning("jax_cpu_parity")
    assert tuning.chunk_policy == "stable_default_dense_audit"
    assert tuning.hessian_vjp_point_chunk_size == 0
