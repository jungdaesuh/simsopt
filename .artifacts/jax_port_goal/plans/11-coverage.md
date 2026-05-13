# Item 11 Coverage Matrix -- Dommaschk + Reiman JAX kernels

Upstream audit SHA:
``1b0cc3a96063197cdbdd01559e04c25456fbe6ff``.

Lane: ``direct_kernel`` for the new JAX cross-oracle and grad-symmetry
checks; ``fd_gradient`` (``central_fd_error_rate``) for the Reiman
Taylor test. ``np.allclose`` defaults are used only against the
historically-printed Dommaschk paper references where the printed
precision is the limiting factor.

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | ``tests/field/test_magneticfields.py::test_Dommaschk`` (lines 642-708) | Hard-coded paper fixtures: four ``(mn, coeffs, point, expected_B_wrapper, expected_grad_wrapper)`` triples plus serialization smoke | ``oracle_only`` | These four printed references become inputs to ``test_dommaschk_paper_fixtures`` in the new JAX test file. The existing CPU test continues to exercise the C++ wrapper. |
| current | ``tests/field/test_magneticfields.py::test_Reiman`` (lines 958-1003) | Single ``k=6, epsilon=0.01`` Reiman setup with a closed-form Cartesian B expression plus a hard-coded ``dB`` per-point reference | ``oracle_only`` | The closed-form B expression is replicated in ``test_reiman_closed_form``. The CPU test continues to exercise the C++ wrapper. |
| current | ``tests/field/test_magneticfields.py::subtest_reiman_dBdX_taylortest`` and ``test_reiman_dBdX_taylortest`` (lines 1005-1031) | Forward-difference Taylor test on Reiman ``dB_by_dX`` at two random points | ``oracle_only`` | Recipe replicated in ``test_reiman_dB_taylor`` against the new JAX kernel, upgraded to central differences so the convergence-rate floor matches ``fd_gradient::central_fd_error_rate``. |
| current | ``tests/jax_core/test_analytic_fields_item11.py::test_dommaschk_paper_fixtures`` | JAX raw-kernel sum + ``ToroidalField(1, 1)`` baseline vs the four printed Dommaschk paper references | ``covered_by_unit_parity`` | 7/7 passed at ``np.allclose`` defaults (matches the historical precision of the printed values). |
| current | ``tests/jax_core/test_analytic_fields_item11.py::test_dommaschk_cpp_cross_oracle`` | JAX vs ``sopp.DommaschkB`` / ``sopp.DommaschkdB`` at randomized well-conditioned ``(mn, coeffs, points)`` | ``covered_by_unit_parity`` | 7/7 passed at ``direct_kernel`` (``rtol=1e-10, atol=1e-12``). |
| current | ``tests/jax_core/test_analytic_fields_item11.py::test_dommaschk_grad_symmetric`` | ``dB[k, p, i, j] == dB[k, p, j, i]`` per (mode, point) for the vacuum field | ``covered_by_unit_parity`` | 7/7 passed at ``direct_kernel``. |
| current | ``tests/jax_core/test_analytic_fields_item11.py::test_reiman_closed_form`` | JAX ``reiman_B`` vs the upstream closed-form Bx/By/Bz expression at 20 randomized points | ``covered_by_unit_parity`` | 7/7 passed at ``direct_kernel``. |
| current | ``tests/jax_core/test_analytic_fields_item11.py::test_reiman_cpp_cross_oracle`` | JAX vs ``sopp.ReimanB`` / ``sopp.ReimandB`` at 20 randomized points | ``covered_by_unit_parity`` | 7/7 passed at ``direct_kernel``. |
| current | ``tests/jax_core/test_analytic_fields_item11.py::test_reiman_dB_taylor`` | Central-FD Taylor test on ``reiman_dB`` for two point indices ``{0, 16}`` | ``covered_by_unit_parity`` | 7/7 passed at ``fd_gradient::central_fd_error_rate``. |
| upstream | ``src/simsoptpp/dommaschk.cpp`` (533 LOC) | C++ reference kernel | ``oracle_only`` | Used as cross-oracle in ``test_dommaschk_cpp_cross_oracle``. |
| upstream | ``src/simsoptpp/reiman.cpp`` (106 LOC) | C++ reference kernel | ``oracle_only`` | Used as cross-oracle in ``test_reiman_cpp_cross_oracle``. |
| out_of_scope | ``src/simsopt/field/magneticfieldclasses.py::Dommaschk`` and ``::Reiman`` Optimizable wrappers | Public adapter Python classes | ``out_of_scope_item_15`` | Item 15 will rewrite these wrappers to consume the new JAX kernels. Item 11 leaves the wrappers unchanged. |

All current-tree rows are mapped. The C++ ``oracle_only`` rows remain
authoritative for the existing CPU regression and are the parity
oracles for the new JAX kernel; the new JAX rows close out the unit
parity for the new ``src/simsopt/jax_core/analytic_fields.py`` module.
