"""Independent gradgradB VJP pair-budget configuration regression tests."""

from unittest_jax_support import JaxTestCase, optional_jax_imports
from unittest import mock
import os

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    from simsopt_jax.backend.runtime import get_field_kernel_tuning


class TestBackendGradgradbTuning(JaxTestCase):
    def test_hessian_vjp_pair_budget_is_independent_of_forward_override(self):
        """Overriding forward point tiles leaves the gradgradB VJP pair budget unchanged."""
        self.patches.enter_context(
            mock.patch.dict(os.environ, {"SIMSOPT_JAX_POINT_CHUNK_SIZE": "4096"})
        )
        tuning = get_field_kernel_tuning("jax_gpu_parity")
        self.assertEqual(
            tuning.point_chunk_size, 4096, "tuning.point_chunk_size == 4096"
        )
        self.assertEqual(
            tuning.hessian_vjp_pair_budget,
            1_048_576,
            "tuning.hessian_vjp_pair_budget == 1_048_576",
        )

    def test_hessian_vjp_pair_budget_preserves_dense_override(self):
        """A zero gradgradB VJP pair budget selects dense evaluation independently."""
        self.patches.enter_context(
            mock.patch.dict(
                os.environ, {"SIMSOPT_JAX_HESSIAN_VJP_PAIR_BUDGET": "0"}
            )
        )
        tuning = get_field_kernel_tuning("jax_gpu_parity")
        self.assertEqual(tuning.point_chunk_size, 256, "tuning.point_chunk_size == 256")
        self.assertEqual(
            tuning.hessian_vjp_pair_budget,
            0,
            "tuning.hessian_vjp_pair_budget == 0",
        )

    def test_dense_audit_zeroes_pair_budget_despite_environment_override(self):
        """The disallow transfer guard disables even an explicitly set pair budget."""
        self.patches.enter_context(
            mock.patch.dict(
                os.environ, {"SIMSOPT_JAX_HESSIAN_VJP_PAIR_BUDGET": "262144"}
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
            tuning.hessian_vjp_pair_budget,
            0,
            "tuning.hessian_vjp_pair_budget == 0",
        )
