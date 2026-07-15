JAX Backend Setup
=================

SIMSOPT includes an opt-in JAX backend for CPU and NVIDIA GPU execution.  The
native CPU backend remains the default and the reference implementation.  JAX
is most useful when a workload can keep arrays and optimization steps on the
target device long enough to amortize compilation and dispatch overhead.

Install
-------

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

Verify that JAX can see a GPU before running a GPU workload::

    python - <<'PY'
    import jax

    devices = jax.devices()
    print(devices)
    assert any(device.platform == "gpu" for device in devices)
    PY

Runtime modes
-------------

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

Select the mode before importing JAX-heavy SIMSOPT modules.  Backend selection,
precision, allocator configuration, and compilation-cache policy are process
settings; changing them after JAX initializes a device is not supported.

Quick smoke benchmark
---------------------

The feasibility benchmark reports first-call compilation time and steady-state
kernel timings::

    python benchmarks/jax_feasibility_spike.py --platform cuda

Synchronize device work before timing custom benchmarks.  JAX dispatch is
asynchronous, so a timer must call ``jax.block_until_ready`` on the result.
Compare steady-state timings separately from the first compiled call.

GPU memory
----------

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
-----------------

Repeated runs can use JAX's persistent compilation cache::

    export JAX_COMPILATION_CACHE_DIR=/path/to/cache

Alternatively, pass ``compilation_cache_dir`` to ``set_backend``.  Use a cache
location that is writable by the process.  First use of a new function, shape,
dtype, device topology, or compiler configuration can still trigger
compilation.

Troubleshooting
---------------

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
