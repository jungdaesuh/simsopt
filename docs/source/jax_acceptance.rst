CPU vs JAX Acceptance Criteria
==============================

This document defines when the JAX code path is ready for research use
alongside the existing CPU (simsoptpp) implementation.

The current migration target is the JAX ``0.9.2`` runtime. The trusted public
acceptance gates remain centered on the ``scipy`` backend. Private
``ondevice`` optimizer behavior remains a separate JAX target-lane validation
track on the same runtime. The removed ``hybrid`` backend is no longer part of
the public contract.

Parity Gates
------------

Before using the JAX path for production research runs, all of the following
must hold:

Precision gates are lane-specific. The source of truth is
``benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES``; do not
apply a same-state ``1e-10`` tolerance to branch-divergent solves,
ill-conditioned exact adjoints, or derivative-heavy paths without the matching
lane evidence.

**Value parity**

- Stage 2 objective (``SquaredFluxJAX.J()``) matches CPU within
  ``rel_err < 1e-10`` on the same coil/surface configuration.
- Same-state direct kernels use the ``direct-kernel`` lane
  (``rtol=1e-10``, ``atol=1e-12``) when a direct C++ oracle is present.
- Branch-stable re-solves use the ``branch-stable-resolve`` lane: core values
  at ``rtol=1e-6``, ``atol=1e-7`` and derived NQS-style values at
  ``rtol=5e-5``, ``atol=1e-7``. Branch-divergent small-grid cases remain
  finite/residual health checks, not parity evidence.
- Label constraints (Volume, Area, ToroidalFlux) match CPU within
  ``rel_err < 1e-12``.

**Gradient parity**

- Stage 2 gradient (``SquaredFluxJAX.dJ()``) matches CPU within
  ``rtol < 1e-9``.
- Existing reduced-real LS wrapper gradients use the ``ls-wrapper-gradient``
  lane (``rtol=1e-10``, ``atol=1e-12``).
- Derivative-heavy direct C++ oracle tests use the ``derivative-heavy`` lane:
  representative first derivatives (``dB/dX``, Biot-Savart VJPs, surface
  coefficient Jacobians, composed Boozer residual Jacobians) at
  ``rtol=1e-8``, ``atol=1e-10``. Second derivatives/Hessians remain TODOs at
  ``rtol=1e-6``, ``atol=1e-8``.
- Full directional FD checks use the ``fd-gradient`` lane
  (``rtol=1e-5``, ``atol=1e-7``) on branch-stable fixtures.
- Exact adjoints are split: ``exact-well-conditioned-adjoint`` permits vector
  parity at ``rtol=1e-6``, ``atol=1e-8`` plus residual ``<=1e-10``;
  ``exact-ill-conditioned-adjoint`` is residual/failure-only and must not
  assert vector parity.

  *Status:* The original ~10x FD discrepancy was caused by the Boozer
  inner solve finding different local minima during FD perturbation on
  small test grids. Fixed-surface FD (perturbing coils without
  re-solving) validates the direct term correctly. Full adjoint-term
  validation uses branch-stable reduced-real fixtures; exact adjoint vector
  parity is asserted only on well-conditioned operator-vs-dense/PLU fixtures.

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
| Single-stage (GPU)  |     | Yes | JAX target lane requires ``optimizer_backend="ondevice"`` and still rides the single-stage validation/proof gates |
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
