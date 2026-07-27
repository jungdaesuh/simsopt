JAX Backend Migration Guide
===========================

The JAX backend coexists with SIMSOPT's native CPU implementation.  Existing
CPU APIs remain the default and the correctness reference.  Applications can
adopt JAX one boundary at a time instead of converting an entire optimization
workflow at once.

Runnable examples
-----------------

The `JAX-first examples <../../examples/jax/README.md>`_ collection provides
pure and adapter lessons plus isolated CPU and strict-GPU runner commands. Its
machine-readable manifest records native-example inspiration, remaining host
boundaries, correctness owners, and deliberately deferred external workflows.

Adapter APIs
------------

The compatibility adapters preserve SIMSOPT's host-side ``Optimizable``
interfaces while delegating numerical work to JAX:

.. list-table::
   :header-rows: 1
   :widths: 31 31 38

   * - Native CPU API
     - JAX adapter
     - Import module
   * - ``BiotSavart``
     - ``BiotSavartJAX``
     - ``simsopt_jax_adapters.field.biotsavart_backend``
   * - ``SquaredFlux``
     - ``SquaredFluxJAX``
     - ``simsopt_jax_adapters.objectives.flux``
   * - ``BoozerSurface``
     - ``BoozerSurfaceJAX``
     - ``simsopt_jax_adapters.geo.boozer_surface``
   * - ``BoozerResidual``
     - ``BoozerResidualJAX``
     - ``simsopt_jax_adapters.geo.surface_objectives``
   * - ``Iotas``
     - ``IotasJAX``
     - ``simsopt_jax_adapters.geo.surface_objectives``
   * - ``NonQuasiSymmetricRatio``
     - ``NonQuasiSymmetricRatioJAX``
     - ``simsopt_jax_adapters.geo.surface_objectives``

For example::

    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    bs_jax = BiotSavartJAX(coils)
    objective = SquaredFluxJAX(surface, bs_jax)

The adapter layer is the normal migration path for an application that already
uses ``Optimizable`` objects.  The lower-level ``simsopt_jax.core`` package
provides immutable PyTrees and pure numerical functions for code that can stay
inside transformations such as ``jax.jit``, ``jax.grad``, and ``jax.vmap``.

State boundaries
----------------

JAX transformations require explicit, traceable state.  Mutable host objects
therefore cross a boundary before compiled work begins:

* host ``Optimizable`` objects continue to own dependencies and degrees of
  freedom;
* adapters snapshot the required values into immutable arrays or frozen
  PyTrees;
* compiled functions consume that explicit state without reading or mutating
  the live host graph;
* a changed host object requires a new snapshot or adapter evaluation.

The same rule applies to VMEC diagnostics.  Use ``vmec_freeze_splines`` from
``simsopt_jax_adapters.mhd.vmec_diagnostics`` to create the frozen spline state
passed to JAX diagnostic kernels.

Optimizer lanes
---------------

The native ``optimizer_backend="scipy"`` lane remains the CPU reference.  JAX
workflows can select among these control strategies where the objective
supports them:

.. list-table::
   :header-rows: 1
   :widths: 33 67

   * - Backend
     - Control model
   * - ``scipy-jax``
     - SciPy controls L-BFGS-B on the host; JAX evaluates target-lane values
       and gradients.
   * - ``scipy-jax-fullgraph``
     - SciPy retains host control while JAX evaluates the full traceable
       objective graph.
   * - ``ondevice``
     - The supported optimization loop executes through JAX control flow on
       the target device.
   * - ``optax-lbfgs``
     - Optional Optax L-BFGS target lane.
   * - ``optimistix-lbfgs``
     - Optional Optimistix L-BFGS target lane.

Choose an optimizer lane explicitly and verify that the objective supports its
traceability contract.  Host callbacks, Python mutation, and implicit NumPy
conversion cannot occur inside a compiled on-device loop.

JAX device and execution intent
-------------------------------

New code can select JAX placement independently from its execution policy::

    import simsopt_jax.config as simsopt_config

    simsopt_config.set_backend("jax", device="cpu")
    simsopt_config.set_backend("jax", device="gpu", intent="parity")

After JAX is explicitly selected, omitted ``intent`` means ``"fast"``.  An
entirely unset selector still means ``native_cpu``.  Explicit canonical modes
such as ``jax_gpu_fast``, ``jax_gpu_parity``, and
``jax_cpu_float32_smoke`` remain supported and take no ``device`` or ``intent``
keywords.  Requested GPU execution fails if CUDA is unavailable rather than
falling back to CPU.

For the example suite, replace ``--lane cpu-smoke`` with ``--device cpu
--intent parity`` and ``--lane gpu-strict`` with ``--device gpu --intent
parity`` when the historical parity behavior is required.  Omit ``--intent``
for the new fast default.  The legacy lane spellings remain parity aliases for
one compatibility release and emit a deprecation warning.

Fast and parity retain the same FP64 scientific contract, public objectives,
custom SIMSOPT JAX solver family, accepted-state publication, and terminal
scientific gates.  Fast output is never certification evidence.  Use the
dedicated ``examples/jax/run_parity.py`` workflow for hash-bound native/JAX
certification artifacts.

Precision and certificate authority
-----------------------------------

Precision is an independent, typed runtime selection.  Existing applications
that omit it retain their current mode-owned defaults.  New applications can
select FP64 or mixed proposal compute programmatically::

    import simsopt_jax.config as simsopt_config

    simsopt_config.set_backend("jax_gpu_fast", precision="fp64")
    # Or, for supported proposal paths:
    simsopt_config.set_backend("jax_gpu_fast", precision="mixed")

For subprocesses, use ``SIMSOPT_PRECISION=fp64`` or
``SIMSOPT_PRECISION=mixed``.  An explicit ``precision=`` value takes
precedence over the environment.  The compatibility value
``precision="mode_default"`` restores the selected mode's established policy.
The source-only ``SIMSOPT_MIXED_PRECISION`` spelling is rejected, and the
import-time ``SIMSOPT_TRACEABLE_NEWTON_LINEAR_SOLVER`` selector is not part of
the public API.

Mixed precision changes proposal computation, not acceptance authority.  A
mixed candidate must pass live FP64 residual, refinement, condition, and final
accuracy gates.  A failed gate routes to the canonical FP64 fallback or fails
closed; cast-up FP32 values are never treated as an FP64 certificate.

Dense iterative refinement is opt-in through the typed Newton policy.  The
default remains ``"operator_gmres"``.  Select the hybrid dense-IR path only at
an explicit solver boundary::

    from simsopt_jax.geo.optimizers import TraceableNewtonLinearSolver

    linear_solver: TraceableNewtonLinearSolver = "hybrid_final_dense_ir"
    boozer.options["newton_linear_solver"] = linear_solver

The other exact selections are ``"dense_lu"`` and
``"hybrid_final_dense_lu"``.  Dense-IR is not self-selected from problem size
or environment state, so upgrading SIMSOPT does not silently change the
existing operator-GMRES route.

VJP callback convention
-----------------------

JAX Boozer-surface VJP callbacks stored in ``result["vjp"]`` accept
``(lm, booz_surf, iota, G)``.  This differs from the native CPU callback
``(lm, booz_surf)`` because the JAX callback constructs its decision state
from explicit arguments instead of reading mutable solver state.

Tracing and MPI
---------------

JAX-compatible field-line tracing adapters are available in
``simsopt_jax_adapters.field.tracing``.  They do not imply that every native
tracing option or callback is traceable; validate the specific operation used
by an application.

``least_squares_mpi_solve_jax`` in ``simsopt_jax_adapters.solve.mpi`` supports
``TraceableLeastSquaresProblem`` with MPI-distributed finite-difference
Jacobian columns and a SciPy solve on rank zero.  It is a scoped MPI path, not
a claim that every JAX adapter can execute under MPI.

Migration checklist
-------------------

#. Keep a native CPU run as the numerical reference.
#. Select a JAX runtime mode before importing JAX-heavy modules.
#. Select precision explicitly only when departing from the mode default.
#. Replace one native object with its adapter and compare values and
   derivatives in FP64.
#. For mixed compute, verify the live FP64 certificate and fallback path rather
   than comparing proposal values alone.
#. Move immutable numerical state into ``simsopt_jax.core`` only when the
   workflow benefits from a larger compiled region.
#. Measure first-call compilation, steady-state time, device transfers, and
   peak memory separately.
#. Expand the migrated boundary only after the representative optimization
   trajectory remains within the application's tolerances.

Not every mutable SIMSOPT object or third-party callback has a traceable JAX
equivalent.  Falling back to the native CPU path at a documented host boundary
is supported; silently mixing host work into a supposedly compiled region is
not.
