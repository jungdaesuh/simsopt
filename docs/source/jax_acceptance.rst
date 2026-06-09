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
- Independent optimizer endpoints are not same-state kernel comparisons. For
  LM validation, well-conditioned direct fixtures may assert raw final-state
  parity at the solver-specific ``1e-10`` gate, but singular or flat fixtures
  must gate on residual, cost, and first-order optimality rather than raw
  parameter-vector equality.
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

Remaining Port Surface Gates
----------------------------

The remaining JAX-port surfaces use the same lane vocabulary as the parity
ladder, but their acceptance is narrower than a full workflow endorsement until
CUDA smoke and committed proof artifacts exist.

**Bootstrap and Redl current**

- ``compute_trapped_fraction_jax`` must match
  ``compute_trapped_fraction`` on the same ``modB`` / ``sqrtg`` arrays for
  both 2-D and 3-D inputs. Extrema and flux-surface averages use the
  direct-kernel lane; the trapped-fraction scalar keeps the measured
  fixed-quadrature-vs-adaptive-``quad`` error budget.
- ``j_dot_B_Redl_jax`` must match the CPU ``j_dot_B_Redl`` fixture for
  ``helicity_n in {0, +1, -1}``. Profile values and derivatives are evaluated
  once at the host boundary and then passed as explicit arrays to the pure JAX
  kernel.
- ``RedlBootstrapJAX`` must keep live ``Profile`` dependencies at the public
  wrapper boundary while its density / temperature / ``Zeff`` DOF Jacobians are
  checked against centered finite differences.

**Profiles**

- ``ProfilePolynomialJAX``, ``ProfileScaledJAX``, and
  ``ProfilePressureJAX`` must match independent closed-form oracles at the
  same DOF state.
- ``ProfileSplineJAX`` must replay the host FITPACK spline coefficients without
  refitting in the JAX value/derivative path. At-knot comparisons use the
  FITPACK replay round-off budget; off-knot comparisons use the documented
  SciPy spline truncation budget.
- Reusable JIT callers pass profile DOFs or frozen spline state explicitly.
  Direct differentiation through mutable ``Optimizable.f(s)`` object state is
  not an acceptance claim.

**Frozen VMEC diagnostics**

- ``vmec_freeze_splines`` is the host boundary for VMEC spline state. The
  JAX kernels consume the frozen pytree and must not read from a live
  ``Vmec.wout`` object inside compiled code.
- ``vmec_compute_geometry_jax`` is accepted only when its public wrapper and
  frozen-state kernel match ``vmec_compute_geometry`` on the same
  ``vmec_splines`` state and preserve the CPU result field contract modulo JAX
  array types.
- ``vmec_fieldlines_jax`` is accepted as a direct-kernel coordinate-line
  diagnostic plus a branch-stable ``theta_vmec`` resolve. It is not an ODE
  integrator parity claim.

**QFM and generic solve wrappers**

- ``QfmSurfaceJAX`` penalty solves use fixed-state objective / gradient
  evidence. The augmented-Lagrangian exact path accepts success from the
  natural equality KKT residual for ``label(dofs) = targetlabel``; raw inner
  BFGS status is diagnostic only. Branch stability is checked with objective,
  label, and KKT invariants, not host-SLSQP DOF identity.
- ``least_squares_serial_solve_jax``, ``serial_solve_jax``,
  ``constrained_serial_solve_jax``, and ``least_squares_mpi_solve_jax`` accept
  explicit traceable problem adapters. They must reject arbitrary mutable host
  ``Optimizable`` graphs instead of wrapping them or falling back to SciPy.

**Permanent-magnet and wireframe workflows**

- PM and wireframe live-loop JAX paths are accepted for numerical workflow
  orchestration and inner value/gradient lanes only. Plot, FAMUS, and VTK
  writers remain host-side output concerns unless a later plan states
  otherwise.

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

The JAX Biot-Savart kernel preserves the C++ Biot-Savart singularity
contract on inputs that land at the core ``r = ‖x − γ(s)‖ = 0`` (an
evaluation point coincident with a coil quadrature point):

- C++ ``simsoptpp`` returns ``NaN``/``Inf`` from the ``1/r^3`` and
  ``1/r^5`` factors, surfacing the divergence to the caller.
- JAX (``simsopt_jax.core.biotsavart._radius_squared``) preserves exact
  ``r² = 0`` so the same singularity is loud instead of being hidden by
  a finite clamp.

This edge is **documented and intentional** for the current target lane:
no production research workflow lands on point-on-coil geometry, but if a
caller constructs one, the JAX lane must not silently convert the
singularity into a finite value.

Optimizer family equivalence
----------------------------

Four JAX least-squares methods are exposed by
``simsopt_jax.geo.optimizers.optimizer``:

- ``method="lm"`` (``reference_least_squares``) is a host-driven
  Levenberg-Marquardt loop with JAX value/grad and a matrix-free
  GMRES inner solve.
- ``method="lm-ondevice"`` (``target_least_squares``) is the
  trace-safe JAX-on-device version of the same algorithm.
- ``method="lm-minpack-ondevice"`` (``target_least_squares``) is an
  opt-in trace-safe JAX-on-device dense-QR Levenberg-Marquardt lane.
  It materializes the dense Jacobian and solves the Marquardt
  augmented least-squares step with column-pivoted QR.
- ``method="optimistix-lm-ondevice"`` (``target_least_squares``) is an
  Optimistix Levenberg-Marquardt lane with Lineax LSMR inner solves. It
  is exposed by ``least_squares_algorithm="optimistix-lm"`` and uses the
  optimizer runtime dependencies declared by the ``JAX`` and ``JAX_GPU``
  extras on Python 3.11+.

Neither ``method="lm"`` nor ``method="lm-ondevice"`` is a port of
MINPACK ``lmder``. Both use:

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
matrix-free family.

The ``lm-minpack-ondevice`` backend is **doubly opt-in**: it requires
both ``optimizer_backend="ondevice"`` and
``least_squares_algorithm="lm-minpack"`` on ``BoozerSurfaceJAX``. It
is MINPACK-style at the solver level because it uses a dense
column-pivoted QR step and emits the QR-scaled ``gtol`` ``info`` codes
4 and 8. Its contract is classified tolerance equivalence against
SciPy/MINPACK, not packed-QR or per-iteration byte identity:
well-conditioned direct fixtures assert raw final-state parity at
``rtol=atol=1e-10``; singular/flat fixtures assert residual, cost, and
optimality instead of raw ``x`` equality; independent Boozer solves use the
``branch-stable-resolve`` lane for endpoint drift while keeping residual and
objective agreement strict. Hardware certification or CPU exact polish remains
an explicit caller workflow, not automatic behavior in the JAX solver modules.
Because the dense Jacobian is required by the solver itself,
``max_dense_linearization_bytes`` is a hard preflight cap for this lane,
not a final-artifact reporting preference.
Callers needing MINPACK ``lmder`` byte-equality must invoke
``scipy.optimize.least_squares(method="lm")`` directly.

The ``optimistix-lm-ondevice`` backend is **doubly opt-in** and remains
experimental: it requires both ``optimizer_backend="ondevice"`` and
``least_squares_algorithm="optimistix-lm"`` on ``BoozerSurfaceJAX``.
It is useful as a library-backed LSMR comparison lane. It uses a single
``tol`` value as Optimistix ``rtol``/``atol`` and Lineax LSMR
``rtol``/``atol``; non-default ``ftol``/``xtol``/``gtol`` and solver
callbacks are rejected rather than silently reinterpreted. Its optional dense
linearization artifacts are post-hoc compatibility artifacts, not the in-loop
dense QR factors used by ``lm-minpack-ondevice``. On the current oversampled
Boozer fixture it matches the in-tree JAX LM objective and residual-norm scale
to tolerance, but it does not match the endpoint state or the residual vector at
the ``branch-stable-resolve`` gate; on the default near-rank-deficient fixture
it does not improve over the matrix-free lane. Treat it as a diagnostic lane
unless a later validation artifact promotes it.

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
