CPU vs JAX Acceptance Criteria
==============================

This document defines when the JAX code path is ready for research use
alongside the existing CPU (simsoptpp) implementation.

The current migration target is the JAX ``0.9.2`` runtime. The trusted public
acceptance gates remain centered on the ``scipy`` backend. Private
``hybrid`` / ``ondevice`` optimizer behavior remains a separate validation
track on the same runtime and is not part of these public acceptance gates.

Parity Gates
------------

Before using the JAX path for production research runs, all of the following
must hold:

**Value parity**

- Stage 2 objective (``SquaredFluxJAX.J()``) matches CPU within
  ``rel_err < 1e-10`` on the same coil/surface configuration.
- Boozer residual (``BoozerResidualJAX.J()``) is finite and small
  (< 1.0) at a converged Boozer surface.
- ``IotasJAX.J()`` and ``NonQuasiSymmetricRatioJAX.J()`` are finite
  and non-negative at converged solutions.
- Label constraints (Volume, Area, ToroidalFlux) match CPU within
  ``rel_err < 1e-12``.

**Gradient parity**

- Stage 2 gradient (``SquaredFluxJAX.dJ()``) matches CPU within
  ``rtol < 1e-9``.
- Single-stage direct gradient term (``∂J/∂coils``) passes
  fixed-surface FD validation with ``rel_err < 1e-3``.

  *Status:* The original ~10x FD discrepancy was caused by the Boozer
  inner solve finding different local minima during FD perturbation on
  small test grids. Fixed-surface FD (perturbing coils without
  re-solving) validates the direct term correctly. Full adjoint-term
  validation requires a well-conditioned representative case.

**Reduction-order stress tiers**

Mirrored reduction-stress tests must use named acceptance tiers instead of
one-off tolerances. The current tiers are:

- ``biotsavart_chunked_dense``: CPU ``rtol=1e-12``, ``atol=1e-14``; GPU
  ``rtol=1e-12``, ``atol=1e-13``.
- ``biotsavart_accumulation_order``: CPU ``rtol=1e-12``, ``atol=1e-14``;
  GPU ``rtol=1e-12``, ``atol=2e-13``.
- ``integral_bdotn_normalized_stress``: CPU/GPU ``rtol=1e-12``,
  ``atol=1e-14``.
- ``boozer_residual_floor_vector``: CPU ``rtol=1e-12``, ``atol=1e-24``;
  GPU ``rtol=1e-10``, ``atol=1e-22``.
- ``boozer_residual_floor_scalar``: CPU ``rtol=1e-12``, ``atol=1e-15``;
  GPU ``rtol=1e-10``, ``atol=1e-14``.

Use those tiers for reduction-heavy parity probes until new parity data shows
that a kernel needs either a tighter contract or stronger arithmetic.

**Solver convergence**

- ``BoozerSurfaceJAX.run_code()`` (both LS and exact paths) converges
  within the same iteration budget as the CPU solver on test cases.
- Short optimization runs (20+ outer iterations) produce finite
  objectives with monotonically decreasing trend.

Performance Gates
-----------------

These gates require a CUDA-capable GPU environment (A100/H100):

- Stage 2 end-to-end speedup >= 1.25x over CPU on the same problem.
- Boozer ``run_code()`` wall time reduced by >= 15%.
- XLA first-compile time < 60s.
- No unexpected recompilation on unchanged array shapes.
- GPU memory stays within the target tier (A100 40GB or H100 80GB).

CPU Non-Regression
------------------

The CPU code path must remain fully functional:

- All existing CPU tests pass without modification.
- No regressions in shared files (``boozersurface.py``,
  ``surfaceobjectives.py``, ``magneticfield.py``).
- The CPU path remains the correctness oracle during validation.

When To Use Which Backend
-------------------------

+---------------------+-----+-----+-----------------------------------------+
| Scenario            | CPU | JAX | Notes                                   |
+=====================+=====+=====+=========================================+
| Production research | Yes |     | Until all acceptance gates pass         |
+---------------------+-----+-----+-----------------------------------------+
| Stage 2 (GPU)       |     | Yes | Value + gradient parity validated        |
+---------------------+-----+-----+-----------------------------------------+
| Single-stage (GPU)  |     |     | Separate ``hybrid`` / ``ondevice`` validation track only |
+---------------------+-----+-----+-----------------------------------------+
| Development/testing  | Yes | Yes | Both paths exercised in CI              |
+---------------------+-----+-----+-----------------------------------------+
| Benchmarking         |     | Yes | Separate compile-time from steady-state |
+---------------------+-----+-----+-----------------------------------------+

Validation Checklist
--------------------

Before switching a research workflow to the JAX backend, verify:

.. code-block:: text

   [ ] All M1–M4 unit tests pass (jax_smoke CI green)
   [ ] Stage 2 parity tests pass (integration/test_stage2_jax.py)
   [ ] Single-stage value sanity tests pass (small/finite/non-negative)
   [ ] Single-stage gradient FD validation passes (fixed-surface direct term)
   [ ] Short optimization run shows progress
   [ ] GPU memory fits within available VRAM
   [ ] Compile time acceptable for the workload
   [ ] No unexpected recompilations observed
