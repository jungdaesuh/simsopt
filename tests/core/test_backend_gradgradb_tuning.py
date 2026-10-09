"""Independent Hessian reverse-tile configuration regression tests."""

from unittest_jax_support import JaxTestCase
from unittest import mock
import os
from simsopt_jax.backend.runtime import get_field_kernel_tuning


class TestBackendGradgradbTuning(JaxTestCase):
    def test_reverse_point_tuning_is_independent_of_forward_override(self):
        """Overriding forward point tiles leaves the Hessian reverse tile unchanged."""
        self.patches.enter_context(
            mock.patch.dict(os.environ, {"SIMSOPT_JAX_POINT_CHUNK_SIZE": "4096"})
        )
        tuning = get_field_kernel_tuning("jax_gpu_parity")
        self.assertEqual(
            tuning.point_chunk_size, 4096, "tuning.point_chunk_size == 4096"
        )
        self.assertEqual(
            tuning.hessian_vjp_point_chunk_size,
            512,
            "tuning.hessian_vjp_point_chunk_size == 512",
        )

    def test_reverse_point_tuning_preserves_dense_override(self):
        """A zero Hessian reverse tile selects dense evaluation independently."""
        self.patches.enter_context(
            mock.patch.dict(
                os.environ, {"SIMSOPT_JAX_HESSIAN_VJP_POINT_CHUNK_SIZE": "0"}
            )
        )
        tuning = get_field_kernel_tuning("jax_gpu_parity")
        self.assertEqual(tuning.point_chunk_size, 256, "tuning.point_chunk_size == 256")
        self.assertEqual(
            tuning.hessian_vjp_point_chunk_size,
            0,
            "tuning.hessian_vjp_point_chunk_size == 0",
        )

    def test_dense_audit_zeroes_reverse_tile_despite_environment_override(self):
        """The disallow transfer guard disables even an explicitly set reverse tile."""
        self.patches.enter_context(
            mock.patch.dict(
                os.environ, {"SIMSOPT_JAX_HESSIAN_VJP_POINT_CHUNK_SIZE": "128"}
            )
        )
        self.patches.enter_context(
            mock.patch.dict(os.environ, {"SIMSOPT_JAX_TRANSFER_GUARD": "disallow"})
        )
        tuning = get_field_kernel_tuning("jax_cpu_parity")
        self.assertEqual(
            tuning.chunk_policy,
            "stable_default_dense_audit",
            "tuning.chunk_policy == 'stable_default_dense_audit'",
        )
        self.assertEqual(
            tuning.hessian_vjp_point_chunk_size,
            0,
            "tuning.hessian_vjp_point_chunk_size == 0",
        )
