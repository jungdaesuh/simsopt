JAX Backend
===========

SIMSOPT ships an optional JAX backend that mirrors the native CPU
implementation and can execute on CPU or GPU. This page covers installing
and configuring it, then migrating existing code onto its APIs.

JAX Backend Setup
-----------------

SIMSOPT includes an opt-in JAX backend for CPU and NVIDIA GPU execution.  The
native CPU backend remains the default and the reference implementation.  JAX
is most useful when a workload can keep arrays and optimization steps on the
target device long enough to amortize compilation and dispatch overhead.

Install
~~~~~~~

For CPU development, create the repository environment or install the JAX
extra directly::

    conda env create -f envs/jax.yml
    conda activate jax

    # Alternative, from the repository root:
    python -m pip install -e ".[JAX,dev]"

For NVIDIA GPUs, install the CUDA extra::

    python -m pip install -e ".[JAX_GPU,dev]"

The repository pins the supported JAX, CUDA 12, and NVIDIA component versions
in ``pyproject.toml``.  The ``JAX_GPU`` extra uses the CUDA libraries packaged
for the Python environment.  Avoid placing incompatible system CUDA libraries
ahead of those packages through ``LD_LIBRARY_PATH``.

Base SIMSOPT remains installable on Python ``>=3.8``.  The pinned ``JAX`` and
``JAX_GPU`` extras require Python ``>=3.11``; use a Python 3.11 or newer
environment before installing either extra.

Static type checking
~~~~~~~~~~~~~~~~~~~~

The development extra pins Pyright.  From an environment installed with
``.[JAX,dev]`` or ``.[JAX_GPU,dev]``, run the configured blocking check with::

    pyright --warnings

The checked paths are owned by ``[tool.pyright]`` in ``pyproject.toml``.  The
initial gate covers the already-green JAX objectives and solver-contract slice;
it does not claim that the complete historical JAX tree is type-clean.  Expand
that include list as existing diagnostics are repaired.  Do not disable a
diagnostic category or add blanket ignores to make a new path pass.

Verify that JAX can see a GPU before running a GPU workload::

    python - <<'PY'
    import jax

    devices = jax.devices()
    print(devices)
    assert any(device.platform == "gpu" for device in devices)
    PY

Runtime modes
~~~~~~~~~~~~~

``SIMSOPT_BACKEND_MODE`` selects the runtime contract.  The supported modes
are:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Mode
     - Intended use
   * - ``native_cpu``
     - Native CPU reference implementation and default.
   * - ``jax_cpu_parity``
     - FP64 JAX CPU execution with reference-oriented numerical policy.
   * - ``jax_cpu_fast``
     - JAX CPU execution with performance-oriented policy.
   * - ``jax_cpu_float32_smoke``
     - CPU smoke testing of float32-compatible paths.
   * - ``jax_gpu_parity``
     - FP64 GPU execution with reference-oriented numerical policy.
   * - ``jax_gpu_fast``
     - GPU execution with performance-oriented policy.

Set the mode in the environment::

    SIMSOPT_BACKEND_MODE=jax_gpu_parity python your_program.py

Applications can configure it programmatically instead::

    import simsopt_jax.config as simsopt_config

    simsopt_config.set_backend("jax_gpu_parity", strict=True)

For the ordinary typed JAX selector, device and execution intent are
orthogonal::

    simsopt_config.set_backend("jax", device="cpu")
    simsopt_config.set_backend("jax", device="gpu")
    simsopt_config.set_backend("jax", device="cpu", intent="parity")
    simsopt_config.set_backend("jax", device="gpu", intent="parity")

The ``intent`` defaults to ``"fast"`` after JAX is selected.  With no backend
selector at all, ``native_cpu`` remains the default.  Fast and parity both use
FP64 scientific arrays, strict device-fallback rejection, and the same custom
SIMSOPT JAX solver family and scientific-success checks.  Parity is explicit
and is the only intent eligible for certification; an ordinary example-runner
result is diagnostic even when its intent is parity.

The full canonical modes remain available for low-level configuration,
including ``jax_cpu_float32_smoke``.  A canonical mode cannot be combined with
``device`` or ``intent``, and ``set_backend("jax")`` requires an explicit
device.

Select the mode before importing JAX-heavy SIMSOPT modules.  Backend selection,
precision, allocator configuration, and compilation-cache policy are process
settings; changing them after JAX initializes a device is not supported.

Precision policy
~~~~~~~~~~~~~~~~

The runtime keeps the existing mode defaults unless precision is selected
explicitly.  Use the typed ``precision`` argument to ``set_backend``::

    import simsopt_jax.config as simsopt_config

    simsopt_config.set_backend(
        "jax_gpu_fast",
        precision="mixed",
        strict=True,
    )

The accepted selections are ``"mode_default"``, ``"fp64"``, and ``"mixed"``.
``"mode_default"`` preserves the historical dtype policy for the selected
runtime mode.  ``"fp64"`` requests FP64 explicitly.  ``"mixed"`` uses FP32
only for supported proposal computations while retaining FP64 result and
certificate dtypes.  The equivalent process setting is
``SIMSOPT_PRECISION=mode_default|fp64|mixed``; an explicit non-``None``
``precision`` argument takes precedence.

Mixed precision never promotes widened FP32 proposal values into accepted
evidence.  Supported mixed solvers validate candidates with the live FP64
operator, refine against that operator, and use the canonical FP64 fallback
when a proposal, refinement, condition, contraction, or tolerance gate fails.
Accepted public results and certificate-side gradients therefore retain FP64
authority.  The native CPU default and every omitted-precision JAX route remain
unchanged.

Quick smoke benchmark
~~~~~~~~~~~~~~~~~~~~~

The feasibility benchmark reports first-call compilation time and steady-state
kernel timings::

    python benchmarks/jax_feasibility_spike.py --platform cuda

Synchronize device work before timing custom benchmarks.  JAX dispatch is
asynchronous, so a timer must call ``jax.block_until_ready`` on the result.
Compare steady-state timings separately from the first compiled call.

GPU memory
~~~~~~~~~~

GPU modes default ``XLA_PYTHON_CLIENT_PREALLOCATE`` to ``false``.  This avoids
reserving most GPU memory when JAX first starts, although it can increase
allocator overhead.  The following SIMSOPT settings are resolved before JAX
initialization:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Setting
     - Effect
   * - ``SIMSOPT_JAX_GPU_PREALLOCATE``
     - Controls ``XLA_PYTHON_CLIENT_PREALLOCATE``.
   * - ``SIMSOPT_JAX_GPU_MEM_FRACTION``
     - Limits the allocator memory fraction.
   * - ``SIMSOPT_JAX_GPU_ALLOCATOR``
     - Selects the supported ``platform`` or ``vmm`` allocator policy.
   * - ``SIMSOPT_TF_GPU_ALLOCATOR=cuda_malloc_async``
     - Selects CUDA's asynchronous allocator through ``TF_GPU_ALLOCATOR``.

The same settings are available as ``set_backend`` keyword arguments:
``xla_gpu_preallocate``, ``xla_gpu_mem_fraction``, ``xla_gpu_allocator``, and
``tf_gpu_allocator``.  See the `JAX GPU memory allocation documentation
<https://docs.jax.dev/en/latest/gpu_memory_allocation.html>`_ for the underlying
allocator behavior.

JAX executables are compiled for a platform and array shape.  A GPU executable
cannot transparently continue on CPU after an out-of-memory error.  Reduce the
problem or chunk size, or checkpoint and restart the process in a CPU mode.

Compilation cache
~~~~~~~~~~~~~~~~~

Repeated runs can use JAX's persistent compilation cache::

    export JAX_COMPILATION_CACHE_DIR=/path/to/cache

Alternatively, pass ``compilation_cache_dir`` to ``set_backend``.  Use a cache
location that is writable by the process.  First use of a new function, shape,
dtype, device topology, or compiler configuration can still trigger
compilation.

Troubleshooting
~~~~~~~~~~~~~~~

No GPU is listed
  Confirm that ``nvidia-smi`` works on the allocated compute node and that the
  environment contains the repository's ``JAX_GPU`` extra.  Check the JAX CUDA
  driver requirements if the driver and wheel are incompatible.

The first call is slow
  This is normally compilation.  Measure later synchronized calls separately
  and enable ``JAX_EXPLAIN_CACHE_MISSES=1`` when unexpected recompilation is
  suspected.

GPU execution is slower than native CPU
  Small calls can be dominated by compilation, dispatch, or host-device
  transfers.  Consumer GPUs also have limited FP64 throughput.  Profile a
  representative steady-state workload on an FP64-capable accelerator before
  drawing a crossover conclusion.

Out of memory
  Avoid materializing dense Jacobians or Hessians when a matrix-free path is
  available.  Reduce grid sizes or configure the workload's chunk sizes.  An
  allocator setting can change reservation behavior, but it does not reduce
  the arrays required by the computation.

JAX Backend Migration Guide
---------------------------

The JAX backend coexists with SIMSOPT's native CPU implementation.  Existing
CPU APIs remain the default and the correctness reference.  Applications can
adopt JAX one boundary at a time instead of converting an entire optimization
workflow at once.

Runnable examples
~~~~~~~~~~~~~~~~~

The `JAX-first examples <../../examples/jax/README.md>`_ collection provides
pure and adapter lessons plus isolated CPU and strict-GPU runner commands. Its
machine-readable manifest records native-example inspiration, remaining host
boundaries, correctness owners, and deliberately deferred external workflows.

Adapter APIs
~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Dense iterative refinement is opt-in through the typed Newton policy.
``BoozerSurfaceJAX`` least-squares Newton defaults to ``"dense_lu"``, matching
native ``np.linalg.solve``.  The generic ``newton_polish_traceable`` kernel
still defaults to matrix-free ``"operator_gmres"``; pass
``newton_linear_solver="operator_gmres"`` on a Boozer LS solve to keep that
route.  Select the hybrid dense-IR path only at an explicit solver boundary::

    from simsopt_jax.geo.optimizers import TraceableNewtonLinearSolver

    linear_solver: TraceableNewtonLinearSolver = "hybrid_final_dense_ir"
    boozer.options["newton_linear_solver"] = linear_solver

The other exact LS selections are ``"operator_gmres"`` and
``"hybrid_final_dense_lu"``.  Dense-IR is not self-selected from problem size
or environment state.

VJP callback convention
~~~~~~~~~~~~~~~~~~~~~~~

JAX Boozer-surface VJP callbacks stored in ``result["vjp"]`` accept
``(lm, booz_surf, iota, G)``.  This differs from the native CPU callback
``(lm, booz_surf)`` because the JAX callback constructs its decision state
from explicit arguments instead of reading mutable solver state.

Tracing and MPI
~~~~~~~~~~~~~~~

JAX-compatible field-line tracing adapters are available in
``simsopt_jax_adapters.field.tracing``.  They do not imply that every native
tracing option or callback is traceable; validate the specific operation used
by an application.

``least_squares_mpi_solve_jax`` in ``simsopt_jax_adapters.solve.mpi`` supports
``TraceableLeastSquaresProblem`` with MPI-distributed finite-difference
Jacobian columns and a SciPy solve on rank zero.  It is a scoped MPI path, not
a claim that every JAX adapter can execute under MPI.

Migration checklist
~~~~~~~~~~~~~~~~~~~

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
