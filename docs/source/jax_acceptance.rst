CPU vs JAX Acceptance Criteria
==============================

This document defines when the JAX code path is ready for research use
alongside the existing CPU (simsoptpp) implementation.

The trusted public acceptance gates remain centered on the ``scipy`` backend.
Private ``ondevice`` optimizer behavior remains a separate JAX target-lane
validation track on the same runtime. The removed ``hybrid`` backend is no
longer part of the public contract.

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
  ``rtol=1e-8``, ``atol=1e-10``. Other second-derivative families remain in
  this lane until they have their own direct oracle closures.
- Column-complete Boozer penalty Hessian parity uses the
  ``direct-hessian-oracle`` lane and compares the CPU/C++ Hessian oracle
  against one JAX HVP per decision variable at ``rtol=1e-8``,
  ``atol=1e-10``.
- Full directional FD checks use the ``fd-gradient`` lane
  (``rtol=1e-5``, ``atol=1e-7``) on branch-stable fixtures.
- Exact adjoints are split: ``exact-well-conditioned-adjoint`` permits vector
  parity at ``rtol=1e-6``, ``atol=1e-8`` plus residual ``<=1e-10``. Current
  exact operator-status coverage is mixed-RHS: Iotas satisfies the residual
  success contract, while NQS exercises the residual/failure-only branch.
  True ``exact-ill-conditioned-adjoint`` fixtures, when present, are
  residual/failure-only and must not assert vector parity.

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

Domain-edge behavior
--------------------

The JAX Biot-Savart kernel and the C++ Biot-Savart kernel diverge on
inputs that land at the singular core ``r = ‖x − γ(s)‖ = 0`` (an
evaluation point coincident with a coil quadrature point):

- C++ ``simsoptpp`` returns ``NaN``/``Inf`` from the ``1/r^3`` and
  ``1/r^5`` factors, surfacing the divergence to the caller.
- JAX (``simsopt.jax_core.biotsavart._safe_radius_squared``) clamps
  ``r²`` at ``1e-60`` so the ``1/r^{1.5}`` factor stays inside
  float64 (using the float64 subnormal minimum ``~5e-324`` would
  yield ``1/(5e-324)^{1.5} ~ 9e484``, ~177 orders of magnitude above
  float64 max ``~1.8e308``). The clamp produces a finite-but-huge
  numeric value rather than ``NaN``/``Inf``.

This divergence is **documented and intentional** for the current
target lane: no production research workflow lands on
point-on-coil geometry, the JAX kernel must keep autodiff finite for
trace stability, and matching the C++ behavior would require a
separate validation cycle. Callers that need C++-equivalent
``NaN``/``Inf`` behavior on degenerate inputs should use the C++
backend explicitly.

Optimizer family equivalence
----------------------------

Two JAX least-squares methods are exposed by
``simsopt.geo.optimizer_jax``:

- ``method="lm"`` (``reference_least_squares``) is a host-driven
  Levenberg-Marquardt loop with JAX value/grad and a matrix-free
  GMRES inner solve.
- ``method="lm-ondevice"`` (``target_least_squares``) is the
  trace-safe JAX-on-device version of the same algorithm.

Neither method is a port of MINPACK ``lmder``. Both use:

- A matrix-free GMRES inner solve against the regularized
  Gauss-Newton operator ``J^T J + λI`` (no pivoted-QR
  factorization, no dense Jacobian materialization in the inner
  step).
- Matrix-free MINPACK-style termination bookkeeping. The JAX LM
  surfaces ``info`` codes 1, 2, 3, 5, 6, and 7 for the
  ``ftol``/``xtol``/budget/stringent-tolerance subset that can be
  computed without a pivoted-QR factorization. When callers provide
  ``gtol``, the matrix-free infinity-norm gradient gate uses that
  threshold; otherwise the legacy ``‖∇‖_∞ ≤ tol`` convergence gate is
  preserved. MINPACK ``info`` codes 4 and 8 require the pivoted-QR
  scaled-gradient norm and remain outside this lane.
- A symmetric Marquardt damping update — decrease ``× 0.5`` on
  ``ratio > 0.75`` and increase ``× 2.0`` on ``ratio < 0.25`` or
  rejected steps.

The ``lm-ondevice`` backend is **doubly opt-in**: it requires both
``optimizer_backend="ondevice"`` and ``least_squares_algorithm="lm"``
on ``BoozerSurfaceJAX``. ``"lm"`` (host-driven) and ``"lm-ondevice"``
(trace-safe) are each other's byte-equality oracle for the JAX LM
family; callers needing MINPACK ``lmder`` byte-equality must invoke
``scipy.optimize.least_squares(method="lm")`` directly. The JAX LM
family delivers tolerance equivalence on the target lane, not MINPACK
byte-equality.

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
