JAX GPU Environment Setup
=========================

This page documents how to set up and run the JAX-accelerated code path
in simsopt on GPU-equipped nodes.

Supported Hardware
------------------

+----------+-------------------------------------+-------------------------------+
| Tier     | Hardware                            | Purpose                       |
+==========+=====================================+===============================+
| Primary  | NVIDIA A100 / H100 (FP64-capable)   | Performance benchmarks        |
+----------+-------------------------------------+-------------------------------+
| Minimum  | NVIDIA V100 (FP64-capable)          | Lowest acceptable reference   |
+----------+-------------------------------------+-------------------------------+
| Smoke    | Consumer NVIDIA (RTX 3090/4090)     | Functional correctness only   |
+----------+-------------------------------------+-------------------------------+
| Always   | CPU                                 | Reference oracle, CI, dev     |
+----------+-------------------------------------+-------------------------------+

The Boozer and field kernels are FP64-heavy scientific workloads.  Consumer
GPUs have severely reduced FP64 throughput and are not representative of
the target use case.

Dependencies
------------

Base install (CPU development/testing)::

    conda env create -f envs/jax.yml
    conda activate jax

The environment recipe already performs the editable CPU-side
``simsopt[JAX,dev]`` install, so no extra ``pip install`` step is needed for
the public CPU development lane.

GPU install (CUDA 12)::

    # From the repo root:
    pip install -e ".[JAX_GPU,dev]"

    # or for a non-editable runtime:
    pip install "simsopt[JAX_GPU]"

Both the public and private optimizer lanes are Python ``3.11+`` with the
``jax`` / ``jaxlib`` distributions resolved by ``pyproject.toml``. The trusted
reference backend remains ``SIMSOPT_BACKEND_MODE=native_cpu`` with
``optimizer_backend="scipy"``. Plain ``optimizer_backend="scipy"`` is
CPU/reference-only. JAX Stage 2 and single-stage target runs may use
``optimizer_backend="ondevice"``, ``optimizer_backend="scipy-jax"``, or
``optimizer_backend="scipy-jax-fullgraph"``; the two SciPy-control JAX lanes
keep JAX value/gradient evaluation while SciPy owns the host control loop.

Verify the install::

    python -c "import jax; print(jax.devices())"
    # Should show: [CudaDevice(id=0)]

Frozen VMEC State
-----------------

VMEC diagnostics cross a host boundary before entering JAX. Use
``vmec_freeze_splines`` to snapshot the radial spline state from either a live
``Vmec`` object or an existing ``vmec_splines`` structure, then pass the frozen
state to the JAX diagnostic kernels::

    from simsopt_jax_adapters.mhd.vmec_diagnostics import (
        vmec_compute_geometry_jax,
        vmec_freeze_splines,
    )

    frozen = vmec_freeze_splines(vmec)
    geometry = vmec_compute_geometry_jax(frozen, s, theta, phi)

The frozen object is an immutable snapshot of the VMEC spline coefficients and
metadata. If the VMEC object is rerun, mutated, or reloaded, call
``vmec_freeze_splines`` again and use the new frozen state. Compiled JAX
diagnostics must consume the frozen pytree state; they must not read live
``Vmec.wout`` state or refit radial splines inside the compiled region.

This boundary preserves the upstream CPU diagnostic contract: CPU VMEC and
``vmec_splines`` remain the reference oracle, while
``vmec_compute_geometry_jax`` and ``vmec_fieldlines_jax`` replay the frozen
spline state for CPU/GPU parity checks.

Environment Variables
---------------------

The preferred runtime selector is the explicit backend mode:

.. list-table::
   :header-rows: 1

   * - Variable
     - Values
     - Default
     - Purpose
   * - ``SIMSOPT_BACKEND_MODE``
     - ``native_cpu``, ``jax_cpu_parity``, ``jax_gpu_parity``, ``jax_gpu_fast``
     - ``native_cpu``
     - Select the runtime contract directly
   * - ``SIMSOPT_BACKEND_STRICT``
     - truthy / falsy env values
     - falsy
     - Record strict fallback policy for callsites that honor strict-mode guards

The older env pair remains supported for compatibility.

**Code-path backend**:

.. list-table::
   :header-rows: 1

   * - Variable
     - Values
     - Default
     - Purpose
   * - ``SIMSOPT_BACKEND``
     - ``cpu``, ``jax``
     - ``cpu``
     - Select simsoptpp (CPU) or JAX code path
   * - ``STAGE2_BACKEND``
     - ``cpu``, ``jax``
     - ``cpu``
     - Legacy alias for ``SIMSOPT_BACKEND``

**JAX device platform**:

.. list-table::
   :header-rows: 1

   * - Variable
     - Values
     - Default
     - Purpose
   * - ``SIMSOPT_JAX_PLATFORM``
     - ``cpu``, ``cuda``
     - ``cpu``
     - JAX XLA device selection
   * - ``SIMSOPT_JAX_BACKEND``
     - ``cpu``, ``cuda``
     - ``cpu``
     - Legacy alias for ``SIMSOPT_JAX_PLATFORM``

Example: run Stage 2 on the GPU parity lane::

    SIMSOPT_BACKEND_MODE=jax_gpu_parity \
        python banana_coil_solver.py

Legacy-compatible example::

    SIMSOPT_BACKEND=jax SIMSOPT_JAX_PLATFORM=cuda \
        python banana_coil_solver.py

Programmatic access::

    import simsopt_jax.config as simsopt_config
    from simsopt_jax.backend import (
        get_backend,
        get_backend_mode,
        get_backend_policy,
        get_chunk_tuning,
        is_jax_backend,
    )
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

    simsopt_config.set_backend("jax_gpu_parity", strict=True)
    policy = get_backend_policy()
    chunk_tuning = get_chunk_tuning()

    assert get_backend_mode() == "jax_gpu_parity"
    assert policy.chunk_policy == "stable_default"
    assert chunk_tuning.point_chunk_size >= 0
    assert policy.tolerance_tier == "parity"
    if is_jax_backend():
        ...

Call ``set_backend(...)`` before importing JAX-heavy simsopt subpackages so the
requested runtime mode is visible to those imports.

Mode-owned policy
-----------------

The mode contract now owns the baseline numerical-policy labels that should
appear in provenance and validation output.

.. list-table::
   :header-rows: 1

   * - Mode
     - X64
     - Matmul precision
     - Chunk policy
     - Sharding default
     - Tolerance tier
     - Compilation cache policy
   * - ``native_cpu``
     - required by the current import-time scientific runtime contract
     - ``highest``
     - ``host_reference``
     - ``none``
     - ``cpu_reference``
     - ``not_applicable``
   * - ``jax_cpu_parity``
     - required
     - ``highest``
     - ``stable_default``
     - ``none``
     - ``parity``
     - ``optional_persistent``
   * - ``jax_gpu_parity``
     - required
     - ``highest``
     - ``stable_default``
     - ``none``
     - ``parity``
     - ``optional_persistent``
   * - ``jax_gpu_fast``
     - currently still required
     - JAX default
     - ``performance_tuned``
     - ``hybrid``
     - ``fast``
     - ``optional_persistent``

These labels do not mean every kernel already implements the final chunked
parity/fast architecture. They define the runtime contract and provenance
surface while the remaining kernel work is still in progress.

Sharding defaults
-----------------

``jax_gpu_parity`` intentionally stays single-device by default even on
multi-GPU hosts. Sharding changes reduction order and therefore needs a real
GPU parity proof before it can become the parity default. Users can opt into
``SIMSOPT_JAX_SHARDING=hybrid`` for experiments, but those runs do not carry
round-3 parity signoff until the multi-GPU proof artifact is recorded.
``jax_gpu_fast`` defaults to ``hybrid`` because it is a speed lane rather than a
byte-identity lane.

GPU memory policy
-----------------

The runtime owns the common JAX/XLA GPU allocator environment variables and
sets them before importing JAX when a CUDA backend mode is selected. JAX's
allocator variables are consumed before the first JAX GPU operation; SIMSOPT
keeps a stricter pre-import setup rule so backend mode, allocator env, platform
selection, transfer guard, precision, and provenance are resolved from one
process contract instead of depending on whether a previous import touched a
device.

- ``XLA_PYTHON_CLIENT_PREALLOCATE`` defaults to ``false`` for
  ``jax_gpu_*`` modes.
- ``SIMSOPT_JAX_GPU_MEM_FRACTION`` maps to
  ``XLA_PYTHON_CLIENT_MEM_FRACTION`` for the default allocator and to
  ``XLA_CLIENT_MEM_FRACTION`` when
  ``SIMSOPT_JAX_GPU_ALLOCATOR=vmm``.
- ``SIMSOPT_JAX_GPU_ALLOCATOR`` accepts ``platform`` or ``vmm``. Leaving it
  unset keeps JAX's default BFC allocator.
- ``SIMSOPT_TF_GPU_ALLOCATOR=cuda_malloc_async`` maps to
  ``TF_GPU_ALLOCATOR=cuda_malloc_async``.

Programmatic callers may pass the same knobs explicitly to
``simsopt_jax.config.set_backend(...)`` using
``xla_gpu_preallocate``, ``xla_gpu_mem_fraction``, ``xla_gpu_allocator``, and
``tf_gpu_allocator``. Explicit keyword arguments override ``SIMSOPT_*`` env
vars, and ``SIMSOPT_*`` env vars override mode defaults.

CUDA determinism policy
-----------------------

CUDA backend modes require the current OpenXLA execution-determinism flag in
``XLA_FLAGS`` before JAX backend initialization::

    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_exclude_nondeterministic_ops=true"

The older ``--xla_gpu_deterministic_ops`` spelling is not accepted by the
SIMSOPT runtime gate, even if it appears alongside the current flag.

Chunk autotuning
----------------

The runtime now resolves one effective chunk-tuning contract for the active
mode:

- explicit env overrides still win for coil/quadrature/pairwise chunk sizes
- otherwise, JAX CUDA modes try to bucket chunk sizes from available GPU VRAM
- if VRAM cannot be detected, the checked-in mode defaults remain the fallback

The VRAM probe looks at ``SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB`` first and then
falls back to ``nvidia-smi``. Set ``SIMSOPT_JAX_CHUNK_AUTOTUNE=0`` to disable
autotuning and force the checked-in mode defaults.

GPU Node Quick-Start
--------------------

1. **Load CUDA toolkit** (cluster-specific)::

       module load cuda/12.x  # or equivalent

2. **Create environment**::

       conda env create -f envs/jax.yml
       conda activate jax

3. **Upgrade the editable install to the CUDA extra set**::

       cd simsopt-jax
       pip install -e ".[JAX_GPU,dev]"

4. **Verify GPU access**::

       python -c "
       import jax
       print('Devices:', jax.devices())
       x = jax.numpy.ones(1000)
       print('Sum:', float(x.sum()))
       "

5. **Run a benchmark**::

       SIMSOPT_JAX_PLATFORM=cuda python benchmarks/jax_feasibility_spike.py

Repo-local bootstrap
--------------------

When developing in a workspace that also contains sibling ``simsopt*``
repositories or older editable installs, plain ``import simsopt`` /
``import simsoptpp`` can resolve to the wrong checkout or to a stale
compiled extension in ``site-packages``. This is especially easy to miss
when the Python package comes from this repo but ``simsoptpp`` still comes
from some other environment.

Use the repo bootstrap helper in scripts, examples, and ad hoc probes::

    from repo_bootstrap import bootstrap_local_simsopt

    bootstrap_local_simsopt("src")

``bootstrap_local_simsopt("src")`` does two things:

- strips editable import finders that hijack ``simsopt`` imports
- if a matching local ``build/*/simsoptpp*.so`` exists for the active
  interpreter, loads that repo-local extension explicitly before importing
  ``simsopt``

This is the supported way to keep this repo self-contained at runtime
without relying on whichever ``simsoptpp`` binary happens to be first on
the ambient Python path.

To refresh the compiled extension for the active interpreter from the repo
root::

    python -m pip wheel --no-deps . -w .artifacts/wheels

Verify that the active process is really using the local extension::

    python - <<'PY'
    from repo_bootstrap import bootstrap_local_simsopt
    bootstrap_local_simsopt("src")
    import simsoptpp
    print(simsoptpp.__file__)
    PY

Troubleshooting
---------------

**"No GPU/TPU found"**
  Check ``nvidia-smi`` output.  Ensure ``jax[cuda12]==0.10.0`` is installed
  through the repository ``JAX_GPU`` extra, which also pins the CUDA 12.9
  ``nvidia-cuda-nvcc-cu12``, ``nvidia-cuda-nvrtc-cu12``,
  ``nvidia-cuda-runtime-cu12``, and ``nvidia-nvjitlink-cu12`` components for
  this lane.  The CUDA extras are on the ``jax`` package, not ``jaxlib``.  Keep
  ``LD_LIBRARY_PATH`` unset for pip CUDA wheels so the JAX plugin can discover
  the full packaged NVIDIA library set.  When the bootstrap selects the pip
  ``nvidia/cuda_nvcc`` root, it removes stale
  ``--xla_gpu_enable_libnvjitlink`` tokens because this JAX wheel reports that
  forced libnvJitLink is unsupported; the guarded invariant is pip ``ptxas``
  12.9 on ``PATH`` without a host ``nvlink`` shadow.
  Verify CUDA driver version is compatible with the installed ``jaxlib`` (JAX
  docs have a compatibility table).

**Slow first call**
  Expected.  JAX compiles XLA kernels on first invocation.  Subsequent
  calls on the same array shapes are fast (cached).  Use
  ``jax.clear_caches()`` and ``JAX_EXPLAIN_CACHE_MISSES=1`` to debug
  unexpected recompilations.

**Out of memory**
  Large Jacobian/Hessian objects can exhaust GPU memory.  Reduce grid
  resolution (``mpol``, ``ntor``). CUDA backend modes already set
  ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` before JAX GPU initialization unless
  an explicit override changes that policy; SIMSOPT's supported entrypoint
  still resolves that policy before importing JAX.

OOM Recovery
------------

JAX platform selection and compiled executables are process-scoped. A GPU
compiled JIT cannot be transparently retargeted to CPU after an OOM. Use an
explicit checkpoint/restart workflow instead:

1. Save the current SIMSOPT artifact state, including ``biot_savart_opt.json``
   and adjacent run metadata for Stage 2 / single-stage workflows.
2. Restart Python with ``SIMSOPT_BACKEND_MODE=jax_cpu_fast`` or
   ``SIMSOPT_BACKEND_MODE=jax_cpu_parity``.
3. Load the saved artifact through the normal SIMSOPT load path and resume.

For new CPU-resident construction inside an already-running process, use
``simsopt_jax.backend.with_cpu_device_for_construction()``. It wraps
``jax.default_device(jax.devices("cpu")[0])`` for fresh arrays and fresh
compiles only; it does not retarget existing GPU-compiled functions.

**Wrong repo or wrong ``simsoptpp`` binary loaded**
  If ``simsopt`` or ``simsoptpp`` resolves to ``site-packages`` or a sibling
  repo checkout, call ``bootstrap_local_simsopt("src")`` before importing
  simsopt-heavy modules. Rebuild the local extension with
  ``python -m pip wheel --no-deps . -w .artifacts/wheels`` if no matching
  local ``build/*/simsoptpp*.so`` exists for the active interpreter.

**``libomp.dylib`` missing on macOS**
  A repo-local virtualenv can still fail to import ``simsoptpp`` if the
  extension was linked against OpenMP but the runtime loader cannot find
  ``libomp.dylib``. In that case, either provide a working OpenMP runtime
  for that environment or use an interpreter whose local repo build already
  loads successfully.

**FP64 precision on consumer GPUs**
  Consumer GPUs (RTX series) have 1/32 or 1/64 FP64 throughput.
  Performance will be poor; use only for correctness validation.

Version Upgrade Policy
----------------------

JAX version upgrades are explicit maintenance work.  Each upgrade requires:

1. Re-run all parity tests.
2. Re-benchmark steady-state and compile-time overhead.
3. Verify no unexpected recompilation from changed XLA semantics.
